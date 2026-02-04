"""Normalize chat messages from various sources into a unified format.

Supports:
  - chat-replay-downloader JSON output
  - Twitch native chat format
  - YouTube live chat format
  - Generic JSON with timestamp fields
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .store import ChatMessage


# Known timestamp field names to check
_TS_KEYS = (
    "time_in_seconds",
    "timestamp",
    "time",
    "ts",
    "offset",
    "offset_seconds",
    "seconds",
    "timestamp_ms",
    "time_ms",
    "offset_ms",
    "content_offset_seconds",
    # YouTube live chat replay (yt-dlp)
    "videoOffsetTimeMsec",
    "timestampUsec",
)


class ChatFormat:
    """Detected chat format."""

    CHAT_REPLAY_DOWNLOADER = "chat_replay_downloader"
    TWITCH_VOD = "twitch_vod"
    YOUTUBE_LIVE = "youtube_live"
    GENERIC_JSON = "generic_json"
    UNKNOWN = "unknown"


def detect_chat_format(data: Any) -> str:
    """Detect the format of chat data.

    Args:
        data: Parsed JSON data (list or dict)

    Returns:
        One of ChatFormat constants
    """
    if isinstance(data, list) and len(data) > 0:
        sample = data[0] if isinstance(data[0], dict) else {}

        # chat-replay-downloader format
        if "time_in_seconds" in sample and "message" in sample:
            return ChatFormat.CHAT_REPLAY_DOWNLOADER

        # Twitch VOD chat format
        if "content_offset_seconds" in sample and "commenter" in sample:
            return ChatFormat.TWITCH_VOD

        # YouTube live chat format
        if "replayChatItemAction" in sample:
            return ChatFormat.YOUTUBE_LIVE

        # Generic JSON with timestamp
        for key in _TS_KEYS:
            if key in sample:
                return ChatFormat.GENERIC_JSON

    elif isinstance(data, dict):
        # Check for nested messages
        for key in ("messages", "chat", "items", "comments"):
            if key in data and isinstance(data[key], list):
                return detect_chat_format(data[key])

    return ChatFormat.UNKNOWN


def _parse_hhmmss(val: str) -> Optional[float]:
    """Parse HH:MM:SS or MM:SS format to seconds."""
    parts = val.strip().split(":")
    if len(parts) < 2:
        return None
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        return None
    sec = 0.0
    for p in parts_f:
        sec = sec * 60.0 + p
    return sec


def _parse_timestamp(val: Any, key: str) -> Optional[float]:
    """Parse a timestamp value to seconds.
    
    NOTE: The codebase receives a mix of:
      - relative seconds (e.g., 12.34)
      - relative milliseconds (e.g., 12340) usually with *_ms keys
      - epoch seconds (~1.7e9)
      - epoch milliseconds (~1.7e12)
    The previous heuristic treated any number > 1e7 as "milliseconds",
    which incorrectly downscaled epoch seconds. Use a safer threshold.
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        x = float(val)
        k = key.lower()
        if "usec" in k:
            return x / 1_000_000.0
        if "ms" in k:
            return x / 1000.0
        # Epoch milliseconds are ~1e12+. Treat those as ms.
        if x >= 1e11:
            return x / 1000.0
        # Otherwise assume seconds (covers epoch seconds and relative seconds).
        return x

    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None

        # Try numeric parse
        try:
            return _parse_timestamp(float(val), key)
        except ValueError:
            pass

        # Try HH:MM:SS
        hhmmss = _parse_hhmmss(val)
        if hhmmss is not None:
            return hhmmss

        # Try ISO datetime
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            pass

    return None


def _extract_message_text(msg: Dict[str, Any]) -> str:
    """Extract message text from various formats."""
    # Direct text fields
    for key in ("message", "text", "body", "content"):
        if key in msg and isinstance(msg[key], str):
            return msg[key]

    # Twitch fragments can appear at this level (or nested inside message)
    fragments = msg.get("fragments")
    if isinstance(fragments, list) and fragments:
        parts = [f.get("text", "") for f in fragments if isinstance(f, dict)]
        if parts:
            return "".join(parts)

    # Nested message object (common in Twitch)
    inner = msg.get("message")
    if isinstance(inner, dict):
        # Check for body first
        if "body" in inner and isinstance(inner["body"], str):
            return inner["body"]
        # Check for fragments inside message
        inner_fragments = inner.get("fragments")
        if isinstance(inner_fragments, list) and inner_fragments:
            parts = [f.get("text", "") for f in inner_fragments if isinstance(f, dict)]
            if parts:
                return "".join(parts)
        # Recurse as last resort
        return _extract_message_text(inner)

    return ""


def _extract_author(msg: Dict[str, Any]) -> Tuple[str, str]:
    """Extract (author_name, author_id) from message."""
    author = ""
    author_id = ""

    # Direct author fields
    for key in ("author", "username", "user", "name", "display_name"):
        if key in msg and isinstance(msg[key], str):
            author = msg[key]
            break

    # Author object
    if "author" in msg and isinstance(msg["author"], dict):
        author_obj = msg["author"]
        author = author_obj.get("name", author_obj.get("display_name", ""))
        author_id = str(author_obj.get("id", author_obj.get("channel_id", "")))

    # Commenter object (Twitch)
    if "commenter" in msg and isinstance(msg["commenter"], dict):
        commenter = msg["commenter"]
        author = commenter.get("display_name", commenter.get("name", ""))
        author_id = str(commenter.get("_id", commenter.get("id", "")))

    # Direct author_id field
    if not author_id:
        for key in ("author_id", "user_id", "channel_id"):
            if key in msg:
                author_id = str(msg[key])
                break

    return author, author_id


def _extract_badges(msg: Dict[str, Any]) -> List[str]:
    """Extract badge list from message."""
    badges = []

    # Direct badges array
    if "badges" in msg:
        b = msg["badges"]
        if isinstance(b, list):
            for badge in b:
                if isinstance(badge, str):
                    badges.append(badge)
                elif isinstance(badge, dict):
                    badges.append(badge.get("_id", badge.get("id", str(badge))))

    # Message object with badges
    if "message" in msg and isinstance(msg["message"], dict):
        user_badges = msg["message"].get("user_badges", [])
        for badge in user_badges:
            if isinstance(badge, dict):
                badges.append(badge.get("_id", badge.get("id", "")))

    return badges


def _count_emotes(msg: Dict[str, Any]) -> int:
    """Count emotes in message."""
    count = 0

    # Emotes array
    if "emotes" in msg and isinstance(msg["emotes"], list):
        count = len(msg["emotes"])

    # Message fragments with emoticon
    if "message" in msg and isinstance(msg["message"], dict):
        fragments = msg["message"].get("fragments", [])
        for frag in fragments:
            if isinstance(frag, dict) and frag.get("emoticon"):
                count += 1

    return count


def load_chat_data(path: Path) -> Any:
    """Load chat data from JSON or JSONL.

    yt-dlp live chat exports and some tools emit newline-delimited JSON (JSONL).
    This loader accepts both formats.
    """
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # JSONL: one JSON object per line
        items: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except json.JSONDecodeError:
                continue
        return items


def _extract_youtube_replay_action(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the inner renderer-ish object from a yt-dlp YouTube replay action."""
    rci = msg.get("replayChatItemAction")
    if not isinstance(rci, dict):
        return None
    actions = rci.get("actions")
    if not isinstance(actions, list):
        return None
    for act in actions:
        if not isinstance(act, dict):
            continue
        for key in ("addChatItemAction", "addLiveChatTickerItemAction"):
            inner = act.get(key)
            if isinstance(inner, dict) and "item" in inner:
                return inner["item"]
    return None


def _extract_youtube_text_and_emotes(renderer: Dict[str, Any]) -> tuple[str, int]:
    """Best-effort extraction of text and emoji count from YouTube renderers."""
    # Common payloads:
    # - liveChatTextMessageRenderer
    # - liveChatPaidMessageRenderer
    # - liveChatMembershipItemRenderer
    for key in (
        "liveChatTextMessageRenderer",
        "liveChatPaidMessageRenderer",
        "liveChatMembershipItemRenderer",
    ):
        r = renderer.get(key)
        if not isinstance(r, dict):
            continue
        msg_obj = r.get("message")
        if not isinstance(msg_obj, dict):
            continue
        runs = msg_obj.get("runs")
        if not isinstance(runs, list):
            continue
        parts = []
        emoji_count = 0
        for run in runs:
            if not isinstance(run, dict):
                continue
            if "text" in run:
                parts.append(run["text"])
            if "emoji" in run:
                emoji_count += 1
                eid = run["emoji"].get("shortcuts", [""])[0]
                parts.append(eid if eid else "")
        return "".join(parts), emoji_count
    return "", 0


def _extract_youtube_author(renderer: Dict[str, Any]) -> tuple[str, str]:
    """Best-effort extraction of author name/id from YouTube renderers."""
    for key in (
        "liveChatTextMessageRenderer",
        "liveChatPaidMessageRenderer",
        "liveChatMembershipItemRenderer",
    ):
        r = renderer.get(key)
        if not isinstance(r, dict):
            continue
        author_name = ""
        author_id = ""
        an = r.get("authorName")
        if isinstance(an, dict):
            author_name = an.get("simpleText", "")
        elif isinstance(an, str):
            author_name = an
        ae = r.get("authorExternalChannelId")
        if isinstance(ae, str):
            author_id = ae
        if author_name or author_id:
            return author_name, author_id
    return "", ""


def normalize_chat_messages_with_timebase(
    data: Any,
    *,
    format_hint: Optional[str] = None,
) -> tuple[List[ChatMessage], str]:
    """Normalize chat data into ChatMessage list, returning a timebase label.

    Returns:
        (messages, timebase) where timebase is one of:
          - "relative": timestamps were already relative to the video start
          - "absolute_normalized": timestamps appeared to be epoch-based and were shifted
    """
    # Handle dict wrapper with messages array
    messages_raw: List[Dict[str, Any]] = []
    if isinstance(data, list):
        messages_raw = [m for m in data if isinstance(m, dict)]
    elif isinstance(data, dict):
        for key in ("messages", "chat", "items", "comments"):
            if key in data and isinstance(data[key], list):
                messages_raw = [m for m in data[key] if isinstance(m, dict)]
                break

    if not messages_raw:
        return [], "relative"

    detected_format = format_hint or detect_chat_format(data)

    # First pass: parse timestamps (seconds) and keep raw messages
    parsed: list[tuple[float, Dict[str, Any]]] = []
    for msg in messages_raw:
        # Handle YouTube replay actions specially
        if detected_format == ChatFormat.YOUTUBE_LIVE and "replayChatItemAction" in msg:
            inner = _extract_youtube_replay_action(msg)
            if inner is None:
                continue
            # videoOffsetTimeMsec is in the outer message
            t_sec = None
            for key in ("videoOffsetTimeMsec", "timestampUsec"):
                if key in msg:
                    t_sec = _parse_timestamp(msg[key], key)
                    if t_sec is not None:
                        break
            if t_sec is None:
                continue
            parsed.append((t_sec, {"_yt_renderer": inner, "_raw": msg}))
            continue

        # Generic timestamp extraction
        t_sec: Optional[float] = None
        for key in _TS_KEYS:
            if key in msg:
                t_sec = _parse_timestamp(msg.get(key), key)
                if t_sec is not None:
                    break
        if t_sec is None:
            continue
        parsed.append((t_sec, msg))

    if not parsed:
        return [], "relative"

    # Detect epoch/absolute timestamps and shift them to a 0-based timeline.
    # Epoch seconds are ~1e9+; epoch milliseconds become ~1e9+ after /1000.
    times = [t for (t, _) in parsed]
    min_t = min(times)
    max_t = max(times)
    timebase = "relative"
    if min_t >= 1e8 and max_t >= 1e8:
        timebase = "absolute_normalized"
        parsed = [(t - min_t, m) for (t, m) in parsed]

    # Second pass: build ChatMessage objects
    result: List[ChatMessage] = []
    for t_sec, msg in parsed:
        t_ms = max(0, int(t_sec * 1000))

        # Handle YouTube renderer
        if "_yt_renderer" in msg:
            renderer = msg["_yt_renderer"]
            raw = msg["_raw"]
            text, emote_count = _extract_youtube_text_and_emotes(renderer)
            author, author_id = _extract_youtube_author(renderer)
            result.append(
                ChatMessage(
                    t_ms=t_ms,
                    author=author,
                    author_id=author_id,
                    text=text,
                    badges_json="[]",
                    emote_count=emote_count,
                    raw_json=json.dumps(raw, ensure_ascii=False, default=str),
                )
            )
            continue

        # Standard extraction
        text = _extract_message_text(msg)
        author, author_id = _extract_author(msg)
        badges = _extract_badges(msg)
        emote_count = _count_emotes(msg)

        result.append(
            ChatMessage(
                t_ms=t_ms,
                author=author,
                author_id=author_id,
                text=text,
                badges_json=json.dumps(badges),
                emote_count=emote_count,
                raw_json=json.dumps(msg, ensure_ascii=False, default=str),
            )
        )

    result.sort(key=lambda m: m.t_ms)
    return result, timebase


def normalize_chat_messages(
    data: Any,
    *,
    format_hint: Optional[str] = None,
) -> List[ChatMessage]:
    """Normalize chat data into ChatMessage list.

    Args:
        data: Raw parsed JSON (list or dict)
        format_hint: Optional format hint (one of ChatFormat constants)

    Returns:
        List of normalized ChatMessage objects
    """
    messages, _ = normalize_chat_messages_with_timebase(data, format_hint=format_hint)
    return messages


def load_and_normalize(path: Path) -> Tuple[List[ChatMessage], str]:
    """Load a chat file and normalize its messages.

    Args:
        path: Path to chat JSON file

    Returns:
        Tuple of (messages, detected_format)
    """
    data = load_chat_data(path)
    detected_format = detect_chat_format(data)
    messages = normalize_chat_messages(data, format_hint=detected_format)
    return messages, detected_format
