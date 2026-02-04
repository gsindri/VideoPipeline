"""Chat spike feature extraction (Twitch-first), with laughter/emote awareness.

What this adds vs baseline:
- Laughter signal tuned for Twitch emote culture
- Optional one-time local LLM pass to learn channel-specific laugh emotes
- Multi-signal composite score (activity + authors + emotes + laughter)

Design goals:
- Keep "where" detection stable: scores_activity remains pure activity
- Improve "what kind of moment": scores_laugh + composite scores help rank peaks
- Never call the LLM per-message. If enabled, LLM is used ONCE to classify top tokens.
"""

from __future__ import annotations

import json
import heapq
import logging
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .store import ChatStore

log = logging.getLogger(__name__)
from ..peaks import moving_average, robust_z
from ..project import Project, save_npz, update_project
from ..utils import utc_iso as _utc_iso


# ---------------------------------------------------------------------------
# Meta keys (stored in chat.sqlite meta table)
# ---------------------------------------------------------------------------

_META_LAUGH_TOKENS = "laugh_tokens_json"
_META_LAUGH_TOKENS_SOURCE = "laugh_tokens_source"
_META_LAUGH_TOKENS_UPDATED_AT = "laugh_tokens_updated_at"
_META_LAUGH_TOKENS_VERSION = "laugh_tokens_version"


# ---------------------------------------------------------------------------
# Twitch-first laughter seeds (case-insensitive; stored as lowercase)
# ---------------------------------------------------------------------------

# A small-but-high-value seed set of common laugh emotes / tokens.
# (You can extend this over time, but the LLM learner is meant to capture custom channel emotes.)
_SEED_LAUGH_TOKENS: Set[str] = {
    # Global / common 3rd-party laugh emotes
    "kekw",
    "omegalul",
    "lul",
    "lulw",
    "lulwut",
    "kekwait",
    "pepelaugh",
    "pepolaugh",
    "pepela",
    "pepega",     # sometimes used in a comedic way; if you dislike it, remove
    "hahahaa",    # sometimes appears as emote-ish token
    "hahaa",      # Twitch global emote haHAA is often represented like this in text dumps
    "haha",
    # Text reactions still worth keeping (low cost, helps non-emote chats too)
    "lol",
    "lmao",
    "lmfao",
    "rofl",
    "xd",
    "xdd",
}

# Common laughter emojis (not Twitch-primary, but cheap + helpful)
_LAUGH_EMOJIS = {
    "😂",
    "🤣",
    "😹",
    "😆",
    "😁",
    "😄",
    "😅",
    "😝",
    "😜",
}

# Lightweight text laughter patterns (Twitch-safe)
# - includes "ㅋㅋㅋ", "jajaja", "www" too
_LAUGH_TEXT_RE = re.compile(
    r"(?ix)"
    r"(\b(?:lol+|lmao+|lmfao+|rofl)\b)"
    r"|(\b[xX][dD]{1,}\b)"
    r"|((?:ha){2,})"
    r"|((?:he){2,})"
    r"|(\b(?:kekw|omegalul|lulw?|kekwait|pepe\s*laugh|pepelaugh|pepolaugh)\b)"
    r"|(\b(?:ja){3,}\b)"          # jajaja
    r"|([ㅋ]{3,})"                # ㅋㅋㅋ
    r"|(\b[w]{3,}\b)"             # www (JP laughter)
)

# Tokenization for emote-like tokens in Twitch logs
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,32}")

# Minimal stop list to keep candidates "emote-like"
# (We mainly filter by casing/shape; this list just trims obvious English glue.)
_STOPWORDS = {
    "the", "and", "you", "your", "for", "with", "this", "that", "have", "not",
    "are", "was", "were", "just", "like", "what", "when", "then", "them", "they",
    "from", "about", "there", "here", "out", "get", "got", "im", "ive", "dont",
    "cant", "wont", "its", "its", "it", "to", "of", "in", "on", "at", "is", "a",
}

_URLISH = {"http", "https", "www", "com", "net", "org"}


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def _looks_emote_like(tok: str) -> bool:
    """Heuristic: keep tokens that look like Twitch emote codes / short reactions."""
    if not tok:
        return False
    tl = tok.lower()
    if tl in _URLISH:
        return False
    if tl in _STOPWORDS:
        return False
    if tl in _SEED_LAUGH_TOKENS:
        return True

    # CamelCase / has uppercase -> typical emote names (PogChamp, monkaS, etc)
    if any(c.isupper() for c in tok):
        return True
    # Digits/underscore also common in emotes (4Head, NODDERS_123)
    if any(c.isdigit() for c in tok) or "_" in tok:
        return True
    # Very short tokens can be reactions or emotes (gg, wp, xd)
    if len(tok) <= 3:
        return True

    # All-lowercase longer tokens are usually normal words, so drop them.
    return False


def _fmt_set_sample(items: Set[str], max_items: int = 25) -> str:
    """Compact preview for DEBUG logs without dumping huge sets."""
    if not items:
        return "{}"
    if len(items) <= max_items:
        return "{" + ", ".join(sorted(items)) + "}"
    sample = ", ".join(heapq.nsmallest(max_items, items))
    return f"({len(items)} items) {{{sample}, ...}}"


def load_laugh_tokens_from_store(store: ChatStore) -> Set[str]:
    """Load cached laugh tokens from chat DB meta (lowercased)."""
    raw = store.get_meta(_META_LAUGH_TOKENS, "[]")
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return {str(x).strip().lower() for x in items if str(x).strip()}
    except Exception:
        pass
    return set()


def save_laugh_tokens_to_store(store: ChatStore, tokens: Set[str], *, source: str, llm_learned: Optional[Set[str]] = None) -> None:
    """Persist laugh tokens to chat DB meta."""
    store.set_meta(_META_LAUGH_TOKENS, json.dumps(sorted(tokens), ensure_ascii=False))
    store.set_meta(_META_LAUGH_TOKENS_SOURCE, source)
    store.set_meta(_META_LAUGH_TOKENS_UPDATED_AT, _utc_iso())
    store.set_meta(_META_LAUGH_TOKENS_VERSION, "v1")
    # Also store which tokens came from LLM vs seeds
    if llm_learned:
        store.set_meta("laugh_tokens_llm_learned", json.dumps(sorted(llm_learned), ensure_ascii=False))


def _extract_llm_json(text: str) -> Dict[str, Any]:
    """Try hard to extract a JSON object from an LLM response."""
    if not text:
        return {}
    s = text.strip()

    # Strip common code fences
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

    # Find first JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    blob = s[start : end + 1]
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _collect_candidate_tokens(
    store: ChatStore,
    *,
    top_k: int = 200,
    min_freq: int = 10,
    max_messages: Optional[int] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> List[Tuple[str, int, str]]:
    """Collect top emote-like tokens with a short example text."""
    counts: Counter[str] = Counter()
    example: Dict[str, str] = {}

    total = store.get_message_count()
    for i, msg in enumerate(store.iter_messages()):
        if max_messages is not None and i >= max_messages:
            break
        if not msg.text:
            continue

        toks = _tokenize(msg.text)
        for tok in toks:
            if not _looks_emote_like(tok):
                continue
            tl = tok.lower()
            counts[tl] += 1

            if tl not in example:
                t = msg.text.strip()
                if len(t) > 140:
                    t = t[:137] + "..."
                example[tl] = t

        if on_progress and (i % 20000 == 0):
            # only used for the token-collection sub-phase
            frac = min(0.25, (i / max(1, total)) * 0.25)
            on_progress(frac)

    items: List[Tuple[str, int, str]] = []
    for tok, freq in counts.most_common(top_k):
        if freq < min_freq:
            continue
        items.append((tok, int(freq), example.get(tok, "")))
    return items


def learn_laugh_tokens_with_llm(
    store: ChatStore,
    *,
    llm_complete: Callable[[str], str],
    top_k: int = 200,
    min_freq: int = 10,
    batch_size: int = 60,
    max_messages: Optional[int] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Set[str]:
    """Use a local LLM once to classify which emote-like tokens indicate laughter.

    Returns a lowercase token set including seed tokens.
    """
    candidates = _collect_candidate_tokens(
        store,
        top_k=top_k,
        min_freq=min_freq,
        max_messages=max_messages,
        on_progress=on_progress,
    )

    # Always include seeds
    laugh: Set[str] = set(_SEED_LAUGH_TOKENS)

    # Add cheap pattern-based expansions (helps even without LLM)
    for tok, _, _ in candidates:
        if re.search(r"(laugh|lul|kek)", tok, re.IGNORECASE):
            laugh.add(tok.lower())

    # Tokens to classify (exclude ones already known)
    to_classify = [(t, f, ex) for (t, f, ex) in candidates if t.lower() not in laugh]

    if not to_classify:
        return laugh

    def chunks(lst: List[Tuple[str, int, str]], n: int) -> List[List[Tuple[str, int, str]]]:
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    batches = chunks(to_classify, batch_size)

    for bi, batch in enumerate(batches):
        if on_progress:
            # Progress for LLM phase: 0.25 -> 0.45
            on_progress(0.25 + 0.20 * (bi / max(1, len(batches))))

        token_lines = []
        for t, freq, ex in batch:
            # Keep prompt compact but informative
            if ex:
                token_lines.append(f"- {t} (freq={freq}) example: {ex}")
            else:
                token_lines.append(f"- {t} (freq={freq})")

        prompt = (
            "You are classifying Twitch chat tokens (emote codes / short reactions).\n"
            "Task: identify which tokens strongly indicate LAUGHTER / being amused.\n"
            "Rules:\n"
            "- Include laugh emotes/reactions (e.g., KEKW, OMEGALUL, LULW, PepeLaugh variants).\n"
            "- Exclude hype (Pog/PogChamp), shock (monkaS), sadness (BibleThump), anger (Madge), greetings, etc.\n"
            "- If uncertain, EXCLUDE.\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            '{"laugh_tokens": ["token1", "token2", ...]}\n\n'
            "Tokens:\n"
            + "\n".join(token_lines)
        )

        try:
            resp = llm_complete(prompt)
        except Exception:
            # If LLM fails mid-way, return what we have (seeds + pattern expansions)
            return laugh

        obj = _extract_llm_json(resp)
        items = obj.get("laugh_tokens", [])
        if isinstance(items, list):
            for x in items:
                s = str(x).strip()
                if s:
                    laugh.add(s.lower())

    if on_progress:
        on_progress(0.45)

    return laugh


def _is_laughter_message(text: str, laugh_tokens: Set[str]) -> bool:
    """Message-level laughter detection."""
    if not text:
        return False

    # Emoji check
    if any(ch in _LAUGH_EMOJIS for ch in text):
        return True

    # Lightweight text patterns (covers e.g. "ㅋㅋㅋ", "jajaja", "KEKW")
    if _LAUGH_TEXT_RE.search(text):
        return True

    # Token membership (this is where channel emotes show up)
    if laugh_tokens:
        for tok in _tokenize(text):
            if tok.lower() in laugh_tokens:
                return True

    return False


# Default weights for composite score (conservative)
_DEFAULT_W = {
    "activity": 1.0,
    "authors": 0.35,
    "emotes": 0.20,
    "laugh": 0.30,  # slightly higher since Twitch laughter is valuable for clips
}


def compute_chat_features(
    store: ChatStore,
    *,
    duration_s: float,
    hop_s: float = 0.5,
    smooth_s: float = 3.0,
    on_progress: Optional[Callable[[float], None]] = None,
    laugh_tokens: Optional[Set[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute chat features from a ChatStore.

    Returns (backwards compatible keys):
      - counts, msg_rate, unique_authors, emote_rate, smoothed, scores

    Adds:
      - emote_counts
      - laugh_counts, laugh_rate
      - scores_activity, scores_authors, scores_emote, scores_laugh
      - messages_total, messages_in_range
      - composite_weights
    """
    if hop_s <= 0:
        raise ValueError("hop_s must be > 0")

    w = dict(_DEFAULT_W)
    if weights:
        for k in w.keys():
            if k in weights and weights[k] is not None:
                w[k] = float(weights[k])

    laugh_tokens = {t.lower() for t in (laugh_tokens or set())} | set(_SEED_LAUGH_TOKENS)

    n_bins = int(duration_s / hop_s) + 1
    hop_ms = int(hop_s * 1000)

    counts = np.zeros(n_bins, dtype=np.float64)
    emote_counts = np.zeros(n_bins, dtype=np.float64)
    laugh_counts = np.zeros(n_bins, dtype=np.float64)

    # Lazily created sets to avoid allocating N empty sets
    author_sets: List[Optional[set]] = [None] * n_bins

    if on_progress:
        on_progress(0.05)
    total_messages = store.get_message_count()
    if on_progress:
        on_progress(0.10)

    in_range = 0
    for i, msg in enumerate(store.iter_messages()):
        idx = int(msg.t_ms / hop_ms)
        if 0 <= idx < n_bins:
            counts[idx] += 1.0
            emote_counts[idx] += float(msg.emote_count or 0)

            key = msg.author_id or msg.author
            if key:
                s = author_sets[idx]
                if s is None:
                    s = set()
                    author_sets[idx] = s
                s.add(key)

            if msg.text and _is_laughter_message(msg.text, laugh_tokens):
                laugh_counts[idx] += 1.0

            in_range += 1

        if on_progress and i % 10000 == 0:
            on_progress(0.10 + 0.55 * (i / max(1, total_messages)))

    unique_authors = np.array(
        [len(s) if s is not None else 0 for s in author_sets], dtype=np.float64
    )

    msg_rate = counts / hop_s
    emote_rate = emote_counts / hop_s
    laugh_rate = laugh_counts / hop_s

    if on_progress:
        on_progress(0.70)

    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    smoothed_counts = moving_average(counts, smooth_frames) if len(counts) > 0 else counts
    smoothed_authors = moving_average(unique_authors, smooth_frames) if len(unique_authors) > 0 else unique_authors
    smoothed_emotes = moving_average(emote_counts, smooth_frames) if len(emote_counts) > 0 else emote_counts
    smoothed_laugh = moving_average(laugh_counts, smooth_frames) if len(laugh_counts) > 0 else laugh_counts

    scores_activity = robust_z(smoothed_counts) if len(smoothed_counts) > 0 else smoothed_counts
    scores_authors = robust_z(smoothed_authors) if len(smoothed_authors) > 0 else smoothed_authors
    scores_emote = robust_z(smoothed_emotes) if len(smoothed_emotes) > 0 else smoothed_emotes
    scores_laugh = robust_z(smoothed_laugh) if len(smoothed_laugh) > 0 else smoothed_laugh

    composite_raw = (
        w["activity"] * scores_activity
        + w["authors"] * scores_authors
        + w["emotes"] * scores_emote
        + w["laugh"] * scores_laugh
    )
    scores = robust_z(composite_raw) if len(composite_raw) > 0 else composite_raw

    if on_progress:
        on_progress(1.0)

    return {
        # Backwards compatible keys:
        "counts": counts,
        "msg_rate": msg_rate,
        "unique_authors": unique_authors,
        "emote_rate": emote_rate,
        "smoothed": smoothed_counts,
        "scores": scores,  # composite by default now

        # New helpful keys:
        "emote_counts": emote_counts,
        "laugh_counts": laugh_counts,
        "laugh_rate": laugh_rate,
        "scores_activity": scores_activity,
        "scores_authors": scores_authors,
        "scores_emote": scores_emote,
        "scores_laugh": scores_laugh,
        "messages_total": int(total_messages),
        "messages_in_range": int(in_range),
        "composite_weights": w,
    }


def compute_and_save_chat_features(
    proj: Project,
    *,
    hop_s: float = 0.5,
    smooth_s: float = 3.0,
    on_progress: Optional[Callable[[float], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    weights: Optional[Dict[str, float]] = None,
    # Laughter lexicon options:
    auto_learn_laugh_tokens: bool = True,
    llm_complete: Optional[Callable[[str], str]] = None,
    laugh_top_k: int = 200,
    laugh_min_freq: int = 10,
    laugh_batch_size: int = 60,
    laugh_max_messages: Optional[int] = None,
    force_relearn_laugh: bool = False,
    # Channel info for global emote DB
    channel_id: Optional[str] = None,
    channel_name: Optional[str] = None,
    platform: str = "twitch",
) -> Dict[str, Any]:
    """Compute chat features and save to project.

    If auto_learn_laugh_tokens=True and llm_complete is provided:
      - learns laughter tokens once (cached in chat sqlite meta)
      - then computes features using that lexicon
    
    Learned emotes are also saved to a global database so they can be
    reused across videos from the same channel.
    
    Args:
        on_status: Optional callback for status messages (e.g., "Learning laugh emotes...")
        force_relearn_laugh: If True, re-learn laugh tokens even if cached
        channel_id: Channel identifier for global emote persistence (e.g., "shroud")
        channel_name: Human-readable channel name
        platform: Platform name ("twitch", "youtube", etc.)
    """
    from ..ffmpeg import ffprobe_duration_seconds
    from ..project import get_project_data
    from .emote_db import GlobalEmoteDB

    proj_data = get_project_data(proj)
    duration_s = float(proj_data.get("video", {}).get("duration_seconds", 0))
    if duration_s <= 0:
        duration_s = ffprobe_duration_seconds(proj.audio_source)

    chat_db_path = proj.analysis_dir / "chat.sqlite"
    if not chat_db_path.exists():
        raise FileNotFoundError(f"Chat database not found: {chat_db_path}")

    # Initialize global emote database for cross-project persistence
    global_emote_db = GlobalEmoteDB()
    global_emotes_loaded = set()
    
    store = ChatStore(chat_db_path)
    llm_used = False
    llm_identified: Set[str] = set()
    newly_learned: Set[str] = set()  # Emotes that are truly NEW (not in global DB before)
    
    try:
        # 1) Load cached laugh tokens from project (if any)
        laugh_tokens = load_laugh_tokens_from_store(store)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[EMOTE] Initial laugh_tokens from store: %s", _fmt_set_sample(laugh_tokens))
        
        # Check if we should skip LLM learning (already learned with LLM previously)
        existing_source = store.get_meta(_META_LAUGH_TOKENS_SOURCE, "")
        already_llm_learned = existing_source.startswith("llm")
        log.debug("[EMOTE] existing_source=%r already_llm_learned=%s", existing_source, already_llm_learned)
        
        # 1b) If we have a channel identifier, load from global emote DB FIRST
        log.debug("[EMOTE] channel key passed in: %r", channel_id)
        if channel_id:
            global_emotes_loaded = global_emote_db.get_channel_emotes(channel_id)
            if log.isEnabledFor(logging.DEBUG):
                log.debug("[EMOTE] Global emotes loaded for %r: %s", channel_id, _fmt_set_sample(global_emotes_loaded))
            if global_emotes_loaded:
                # Merge global emotes with local
                laugh_tokens = laugh_tokens | global_emotes_loaded
                if on_status:
                    on_status(f"Loaded {len(global_emotes_loaded)} known emotes for channel '{channel_id}'")
                # If global emotes exist, treat as already learned (skip LLM)
                if not already_llm_learned:
                    # Save global emotes to local store so we don't re-learn
                    llm_learned_from_global = global_emotes_loaded - _SEED_LAUGH_TOKENS
                    save_laugh_tokens_to_store(store, laugh_tokens, source="llm_global", llm_learned=llm_learned_from_global)
                    already_llm_learned = True
        
        if laugh_tokens and not force_relearn_laugh:
            if on_status:
                source_info = " (AI-learned)" if already_llm_learned else " (seeds only)"
                if global_emotes_loaded:
                    source_info = f" ({len(global_emotes_loaded)} from channel history)"
                on_status(f"Using cached laugh lexicon ({len(laugh_tokens)} tokens){source_info}")

        # 2) Optionally learn via local LLM (once, or force)
        should_learn = auto_learn_laugh_tokens and llm_complete is not None
        # Need to learn if: no tokens OR forcing relearn
        # Note: If global emotes were loaded above, already_llm_learned is now True
        need_learn = (not laugh_tokens) or force_relearn_laugh
        log.debug(
            "[EMOTE] should_learn=%s need_learn=%s (laugh_tokens=%s force_relearn=%s)",
            should_learn,
            need_learn,
            bool(laugh_tokens),
            force_relearn_laugh,
        )
        
        if should_learn and need_learn:
            if on_status:
                on_status("Learning channel-specific laugh emotes via LLM...")
            if on_progress:
                on_progress(0.02)

            learned = learn_laugh_tokens_with_llm(
                store,
                llm_complete=llm_complete,
                top_k=laugh_top_k,
                min_freq=laugh_min_freq,
                batch_size=laugh_batch_size,
                max_messages=laugh_max_messages,
                on_progress=on_progress,
            )
            # Track which tokens came from LLM vs seeds
            llm_identified = learned - _SEED_LAUGH_TOKENS
            laugh_tokens = set(learned)
            save_laugh_tokens_to_store(store, laugh_tokens, source="llm", llm_learned=llm_identified)
            llm_used = True
            
            log.info(f"[EMOTE PERSIST] channel_id={channel_id}, platform={platform}, llm_identified={llm_identified}")
            
            # Save to global emote DB for cross-project persistence
            # merge_channel_emotes returns (total, new_count) - track truly new ones
            if channel_id and llm_identified:
                total, new_count = global_emote_db.merge_channel_emotes(
                    channel_id,
                    llm_identified,
                    source="llm",
                    channel_name=channel_name,
                    platform=platform,
                )
                # Figure out which emotes are truly new (not in global DB before)
                newly_learned = llm_identified - global_emotes_loaded
                
                if on_status:
                    seed_count = len(laugh_tokens & _SEED_LAUGH_TOKENS)
                    llm_count = len(llm_identified)
                    if new_count > 0:
                        new_list = sorted(newly_learned)[:5]
                        new_display = ', '.join(new_list)
                        if len(newly_learned) > 5:
                            new_display += f" +{len(newly_learned) - 5} more"
                        on_status(f"Learned {llm_count} emotes ({new_count} NEW: {new_display})")
                    else:
                        on_status(f"Using {len(laugh_tokens)} emotes (all already known for '{channel_id}')")
            else:
                if on_status:
                    seed_count = len(laugh_tokens & _SEED_LAUGH_TOKENS)
                    llm_count = len(llm_identified)
                    on_status(f"Learned {len(laugh_tokens)} laugh tokens ({llm_count} from AI, {seed_count} seeds)")
                    
        elif (not laugh_tokens) and auto_learn_laugh_tokens and llm_complete is None:
            # No LLM available - use seed tokens and save them
            laugh_tokens = set(_SEED_LAUGH_TOKENS)
            save_laugh_tokens_to_store(store, laugh_tokens, source="seed", llm_learned=None)
            if on_status:
                on_status(f"Using seed laugh tokens ({len(laugh_tokens)} tokens, no LLM configured)")
        elif not laugh_tokens:
            # Fallback: at least use seeds even if auto_learn is off
            laugh_tokens = set(_SEED_LAUGH_TOKENS)

        # 3) Compute features
        if on_status:
            on_status("Computing chat features...")
        features = compute_chat_features(
            store,
            duration_s=duration_s,
            hop_s=hop_s,
            smooth_s=smooth_s,
            on_progress=on_progress,
            laugh_tokens=laugh_tokens,
            weights=weights,
        )
        message_count = int(features.get("messages_total", store.get_message_count()))
        messages_in_range = int(features.get("messages_in_range", message_count))

        laugh_source = store.get_meta(_META_LAUGH_TOKENS_SOURCE, "seed_only")
        if llm_used:
            laugh_source = "llm"
        laugh_version = store.get_meta(_META_LAUGH_TOKENS_VERSION, "")
        
        # Get LLM-learned token count for pipeline status display
        llm_learned_json = store.get_meta("laugh_tokens_llm_learned", "[]")
        try:
            llm_learned_list = json.loads(llm_learned_json) if llm_learned_json else []
        except Exception:
            llm_learned_list = []
        llm_learned_count = len(llm_learned_list)
        laugh_tokens_count = len(laugh_tokens)
    finally:
        store.close()

    # Save NPZ (adds new arrays but keeps existing keys)
    save_npz(
        proj.chat_features_path,
        counts=features["counts"],
        msg_rate=features["msg_rate"],
        unique_authors=features["unique_authors"],
        emote_counts=features["emote_counts"],
        emote_rate=features["emote_rate"],
        laugh_counts=features["laugh_counts"],
        laugh_rate=features["laugh_rate"],
        smoothed=features["smoothed"],
        scores=features["scores"],
        scores_activity=features["scores_activity"],
        scores_authors=features["scores_authors"],
        scores_emote=features["scores_emote"],
        scores_laugh=features["scores_laugh"],
        hop_seconds=np.array([hop_s], dtype=np.float64),
    )

    payload = {
        "method": "chat_multisignal_laugh_v1",
        # Top-level fields for pipeline status UI
        "laugh_source": laugh_source,
        "laugh_tokens_count": laugh_tokens_count,
        "llm_learned_count": llm_learned_count,
        "newly_learned_count": len(newly_learned),  # Truly new emotes (not in global DB before)
        "newly_learned_tokens": sorted(newly_learned)[:20],  # Show max 20 new ones
        "loaded_from_global": len(global_emotes_loaded),  # How many came from channel history
        "config": {
            "hop_seconds": hop_s,
            "smooth_seconds": smooth_s,
            "messages_total": int(message_count),
            "messages_in_range": int(messages_in_range),
            "weights": features.get("composite_weights", {}),
            "laugh_lexicon": {
                "source": laugh_source,
                "version": laugh_version,
                "top_k": laugh_top_k,
                "min_freq": laugh_min_freq,
                "batch_size": laugh_batch_size,
                "max_messages": laugh_max_messages,
            },
        },
        "generated_at": _utc_iso(),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["chat"] = {
            **payload,
            "features_npz": str(proj.chat_features_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    return payload
