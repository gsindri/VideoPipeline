from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from .ffmpeg import ffprobe_duration_seconds
from .peaks import moving_average, robust_z
from .project import Project, save_npz, update_project


_TS_KEYS = (
    "time_in_seconds",
    "timestamp",
    "time",
    "ts",
    "t",
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


def _parse_hhmmss(val: str) -> Optional[float]:
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
    if val is None:
        return None
    # NOTE: The codebase receives a mix of:
    #   - relative seconds (e.g., 12.34)
    #   - relative milliseconds (e.g., 12340) usually with *_ms keys
    #   - epoch seconds (~1.7e9)
    #   - epoch milliseconds (~1.7e12)
    # The previous heuristic treated any number > 1e7 as "milliseconds",
    # which incorrectly downscaled epoch seconds. Use a safer threshold.
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
        try:
            return _parse_timestamp(float(val), key)
        except ValueError:
            pass
        hhmmss = _parse_hhmmss(val)
        if hhmmss is not None:
            return hhmmss
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt.timestamp()
    return None


def _extract_timestamps(messages: Iterable[Dict[str, Any]]) -> list[float]:
    times: list[float] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for key in _TS_KEYS:
            if key in msg:
                t = _parse_timestamp(msg.get(key), key)
                if t is not None:
                    times.append(float(t))
                    break
    return times


def _normalize_timestamps(
    timestamps: list[float], duration_s: float
) -> tuple[list[float], str]:
    """Normalize timestamps to a 0-based timeline.
    
    Detects whether timestamps are absolute (epoch seconds) or relative,
    and normalizes absolute timestamps to start from 0.
    
    Returns:
        Tuple of (normalized_timestamps, timebase) where timebase is one of:
        - "relative": timestamps were already relative to video start
        - "absolute_normalized": timestamps were absolute and have been shifted
    """
    if not timestamps:
        return [], "relative"
    
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    
    # Heuristic: if max > duration * 5 AND min > 10000 (likely epoch seconds),
    # treat as absolute timestamps that need normalization
    is_absolute = max_ts > duration_s * 5 and min_ts > 10_000
    
    if is_absolute:
        # Normalize by subtracting the minimum timestamp (stream start)
        normalized = [t - min_ts for t in timestamps]
        return normalized, "absolute_normalized"
    
    return timestamps, "relative"


def _load_messages(path: Path) -> list[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Best-effort JSONL support (yt-dlp live_chat.json, some tools)
        items: list[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                items.append(obj)
        data = items
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    if isinstance(data, dict):
        for key in ("messages", "chat", "items"):
            if key in data and isinstance(data[key], list):
                return [m for m in data[key] if isinstance(m, dict)]
    return []


def compute_chat_analysis(
    proj: Project,
    *,
    chat_path: Path,
    hop_s: float,
    smooth_s: float,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    chat_path = Path(chat_path)
    if not chat_path.exists():
        raise FileNotFoundError(f"Chat JSON not found: {chat_path}")

    messages = _load_messages(chat_path)
    raw_timestamps = _extract_timestamps(messages)

    duration_s = ffprobe_duration_seconds(proj.video_path)
    if hop_s <= 0:
        raise ValueError("hop_s must be > 0")

    # Normalize timestamps (handles absolute epoch timestamps)
    timestamps, timebase = _normalize_timestamps(raw_timestamps, duration_s)

    n = int(duration_s / hop_s) + 1
    counts = np.zeros(n, dtype=np.float64)
    valid_count = 0
    for t in timestamps:
        if t < 0 or t > duration_s:
            continue
        idx = int(t / hop_s)
        counts[idx] += 1.0
        valid_count += 1

    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    smoothed = moving_average(counts, smooth_frames) if len(counts) > 0 else counts
    scores = robust_z(smoothed) if len(smoothed) > 0 else smoothed

    save_npz(
        proj.chat_features_path,
        counts=counts,
        smoothed=smoothed,
        scores=scores,
        hop_seconds=np.array([hop_s], dtype=np.float64),
    )

    payload = {
        "method": "chat_message_rate",
        "config": {
            "hop_seconds": hop_s,
            "smooth_seconds": smooth_s,
            "messages_total": len(messages),
            "messages_in_range": valid_count,
            "timebase": timebase,
        },
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    # Use the actual chat_path passed, not proj.chat_raw_path
    try:
        raw_json_rel = str(chat_path.relative_to(proj.project_dir))
    except ValueError:
        # chat_path is outside project_dir, use absolute path
        raw_json_rel = str(chat_path)

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["chat"] = {
            **payload,
            "features_npz": str(proj.chat_features_path.relative_to(proj.project_dir)),
            "raw_json": raw_json_rel,
        }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload
