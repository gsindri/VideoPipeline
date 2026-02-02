from __future__ import annotations

import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from .analysis_audio import compute_audio_analysis
from .analysis_chat import compute_chat_analysis
from .analysis_motion import compute_motion_analysis
from .analysis_scenes import compute_scene_analysis
from .peaks import moving_average, pick_top_peaks, robust_z
from .project import Project, get_project_data, load_npz, save_npz, update_project


def _coerce_llm_response_to_text(resp: Any) -> str:
    """Coerce various LLM response formats to a string.
    
    Handles:
    - Raw string (passthrough)
    - bytes/bytearray (decode UTF-8)
    - OpenAI-style response payload ({"choices": [{"message": {"content": ...}}]})
    - Already-parsed JSON dict/list (re-serialize to JSON string)
    - Anything else (str() coercion)
    
    This makes the highlight LLM code resilient to different backends/wrappers
    that may return different response shapes.
    """
    if resp is None:
        return ""
    
    if isinstance(resp, str):
        return resp
    
    if isinstance(resp, (bytes, bytearray)):
        return resp.decode("utf-8", errors="replace")
    
    if isinstance(resp, dict):
        # OpenAI-style response payload
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg.get("content") or "")
                if "text" in c0:
                    return str(c0.get("text") or "")
        
        # Check for cached text wrapper from llm_client
        if "__text__" in resp:
            return str(resp["__text__"])
        
        # Already-parsed JSON object - re-serialize
        return json.dumps(resp, ensure_ascii=False)
    
    if isinstance(resp, list):
        # Already-parsed JSON array - re-serialize
        return json.dumps(resp, ensure_ascii=False)
    
    return str(resp)


def _as_1d_f64(x: np.ndarray) -> np.ndarray:
    """Convert to 1D float64 array (no copy if possible)."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    # Replace NaN/Inf with 0 so they don't poison interpolation / fusion
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def resample_series(
    values: np.ndarray,
    *,
    src_hop_s: float,
    target_hop_s: float,
    target_len: int,
    src_times: Optional[np.ndarray] = None,
    target_times: Optional[np.ndarray] = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Resample a 1D series onto a new hop / length using linear interpolation.

    Improvements vs plain np.interp:
    - Supports explicit src/target time arrays (better alignment when windowed).
    - Fills out-of-range with 0 by default (prevents extending last value past EOF).
    - Cleans NaN/Inf.
    """
    if src_hop_s <= 0 or target_hop_s <= 0:
        raise ValueError("hop size must be > 0")
    if target_len <= 0:
        return np.array([], dtype=np.float64)

    v = _as_1d_f64(values)

    # Target times
    if target_times is not None:
        tt = _as_1d_f64(target_times)
        if len(tt) != target_len:
            tt = np.arange(target_len, dtype=np.float64) * float(target_hop_s)
    else:
        tt = np.arange(target_len, dtype=np.float64) * float(target_hop_s)

    if len(v) == 0:
        return np.full(target_len, float(fill_value), dtype=np.float64)

    # Source times
    use_src_times = False
    if src_times is not None:
        st = _as_1d_f64(src_times)
        if len(st) == len(v) and len(st) > 0:
            use_src_times = True
    if use_src_times:
        st = _as_1d_f64(src_times)  # type: ignore[arg-type]
    else:
        st = np.arange(len(v), dtype=np.float64) * float(src_hop_s)

    # Filter non-finite times (rare, but can happen with corrupted caches)
    mask = np.isfinite(st)
    if not np.all(mask):
        st = st[mask]
        v = v[mask]

    if len(v) == 0:
        return np.full(target_len, float(fill_value), dtype=np.float64)

    # Ensure strictly increasing times for interpolation
    if len(st) >= 2 and np.any(np.diff(st) < 0):
        order = np.argsort(st)
        st = st[order]
        v = v[order]

    # Deduplicate equal times (np.interp requires xp to be increasing)
    if len(st) >= 2:
        uniq, inv = np.unique(st, return_inverse=True)
        if len(uniq) != len(st):
            # Average values at duplicate times
            v_dedup = np.zeros(len(uniq), dtype=np.float64)
            counts = np.zeros(len(uniq), dtype=np.float64)
            np.add.at(v_dedup, inv, v)
            np.add.at(counts, inv, 1.0)
            v = v_dedup / np.maximum(counts, 1.0)
            st = uniq

    y = np.interp(tt, st, v, left=float(fill_value), right=float(fill_value)).astype(np.float64)
    return y


def _pos_transform(z: np.ndarray, *, clip_z: float, mode: str) -> np.ndarray:
    """Transform a z-scored signal into positive-only evidence with optional soft-clipping."""
    x = _as_1d_f64(z)
    x = np.clip(x, 0.0, None)

    if clip_z <= 0 or mode in ("none", "off", "false"):
        return x

    if mode == "tanh":
        # Smoothly compress large values but keep units roughly in 'z'
        return float(clip_z) * np.tanh(x / float(clip_z))

    # Default: hard clip
    return np.clip(x, 0.0, float(clip_z))


def _linear_gate(x: np.ndarray, *, threshold: float, scale: float) -> np.ndarray:
    """0..1 gate where x<=threshold -> 0, x>=threshold+scale -> 1."""
    xx = _as_1d_f64(x)
    denom = float(scale) if float(scale) > 1e-9 else 1e-9
    return np.clip((xx - float(threshold)) / denom, 0.0, 1.0)


def snap_time_to_cuts(time_s: float, cuts: Iterable[float], window_s: float) -> float:
    if window_s <= 0:
        return time_s
    best = None
    best_dist = None
    for cut in cuts:
        dist = abs(cut - time_s)
        if dist <= window_s and (best_dist is None or dist < best_dist):
            best = float(cut)
            best_dist = dist
    return best if best is not None else time_s


def _valley_index(scores: np.ndarray, start_idx: int, end_idx: int) -> int:
    if end_idx < start_idx:
        return start_idx
    seg = scores[start_idx : end_idx + 1]
    if len(seg) == 0:
        return start_idx
    return start_idx + int(np.argmin(seg))


def _check_overlap(
    new_start: float,
    new_end: float,
    existing_clips: List[Dict[str, Any]],
    *,
    max_overlap_ratio: float = 0.0,
    max_overlap_seconds: float = 0.0,
    denom: str = "shorter",
) -> bool:
    """Check if a new clip overlaps too much with existing clips.

    This supports BOTH:
    - ratio-based limits (fractional overlap)
    - absolute overlap limits (seconds)

    The absolute limit is important for long clips: a "small" ratio can still be a
    big chunk of duplicated footage.

    Args:
        new_start: Start time of new clip.
        new_end: End time of new clip.
        existing_clips: Existing clip dicts with start_s/end_s.
        max_overlap_ratio: Maximum allowed overlap ratio.
        max_overlap_seconds: Maximum allowed overlap in seconds (0 disables).
        denom: Which duration to normalize overlap by for ratio:
               - "shorter" (default, previous behavior)
               - "new"
               - "existing"
               - "longer"

    Returns:
        True if overlap is acceptable.
    """
    new_duration = float(new_end - new_start)
    if new_duration <= 0:
        return False

    denom = (denom or "shorter").lower().strip()

    for clip in existing_clips:
        clip_start = float(clip.get("start_s", 0.0))
        clip_end = float(clip.get("end_s", 0.0))
        clip_duration = float(clip_end - clip_start)
        if clip_duration <= 0:
            continue

        overlap_start = max(float(new_start), clip_start)
        overlap_end = min(float(new_end), clip_end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap <= 0:
            continue

        # Absolute seconds rule (most intuitive)
        if float(max_overlap_seconds) > 0 and overlap > float(max_overlap_seconds):
            return False

        # Ratio rule
        # If ratio is disabled (<= 0), we still allow overlap if the caller
        # explicitly set an absolute overlap allowance.
        if float(max_overlap_ratio) <= 0:
            # If we get here, overlap > 0 but no ratio limit is set.
            # Only fail if no absolute allowance was set either.
            if float(max_overlap_seconds) <= 0:
                return False
            continue

        # Compute denominator for ratio
        if denom == "new":
            ratio_denom = new_duration
        elif denom == "existing":
            ratio_denom = clip_duration
        elif denom == "longer":
            ratio_denom = max(new_duration, clip_duration)
        else:  # "shorter" (default)
            ratio_denom = min(new_duration, clip_duration)

        if ratio_denom <= 0:
            ratio_denom = 1.0

        overlap_ratio = overlap / ratio_denom
        if overlap_ratio > float(max_overlap_ratio):
            return False

    return True


@dataclass(frozen=True)
class ClipConfig:
    min_seconds: float
    max_seconds: float
    min_pre_seconds: float
    max_pre_seconds: float
    min_post_seconds: float
    max_post_seconds: float


def shape_clip_bounds_with_boundaries(
    *,
    peak_idx: int,
    hop_s: float,
    duration_s: float,
    clip_cfg: ClipConfig,
    boundary_graph: Any,  # BoundaryGraph from analysis_boundaries
) -> Optional[Dict[str, float]]:
    """Shape clip bounds using the BoundaryGraph for natural edit points.
    
    This produces clips that land on sentence ends, silence boundaries, or scene cuts
    instead of arbitrary signal valleys.
    
    Args:
        peak_idx: Index of the peak in the hop-aligned timeline
        hop_s: Hop size in seconds
        duration_s: Video duration in seconds
        clip_cfg: Clip duration constraints
        boundary_graph: BoundaryGraph instance with start/end boundaries
        
    Returns:
        Dict with start_s and end_s, or None if no valid bounds found
    """
    from .analysis_boundaries import find_start_boundary_candidates, find_end_boundary_candidates
    
    peak_time = peak_idx * hop_s
    
    # Use an "ideal" pre/post time that provides good context, not just the minimum
    # This ensures we search for boundaries that give proper setup before the peak
    ideal_pre_seconds = min(clip_cfg.max_pre_seconds, max(clip_cfg.min_pre_seconds, 6.0))
    ideal_post_seconds = min(clip_cfg.max_post_seconds, max(clip_cfg.min_post_seconds, 10.0))
    
    # Get ranked candidates for start boundaries
    # Search centered on ideal_pre, with flexibility to go earlier (more context) or later (tighter)
    start_candidates = find_start_boundary_candidates(
        boundary_graph,
        target_s=peak_time - ideal_pre_seconds,
        max_before_s=clip_cfg.max_pre_seconds - ideal_pre_seconds,  # Can go further back
        max_after_s=ideal_pre_seconds - clip_cfg.min_pre_seconds,   # Can come closer to peak
        prefer_before=True,  # Prefer more context over less
    )
    
    # If no start candidates, try expanding the search window significantly
    if not start_candidates:
        start_candidates = find_start_boundary_candidates(
            boundary_graph,
            target_s=peak_time - ideal_pre_seconds,
            max_before_s=clip_cfg.max_pre_seconds * 2,  # Double the window
            max_after_s=ideal_pre_seconds,
            prefer_before=True,
        )
    
    # Get ranked candidates for end boundaries
    # Search centered on ideal_post, with flexibility
    end_candidates = find_end_boundary_candidates(
        boundary_graph,
        target_s=peak_time + ideal_post_seconds,
        max_before_s=ideal_post_seconds - clip_cfg.min_post_seconds,  # Can end sooner
        max_after_s=clip_cfg.max_post_seconds - ideal_post_seconds,   # Can extend further
        prefer_after=True,  # Prefer completing the moment
    )
    
    # If no end candidates, try expanding the search window significantly
    if not end_candidates:
        end_candidates = find_end_boundary_candidates(
            boundary_graph,
            target_s=peak_time + clip_cfg.min_post_seconds,
            max_before_s=clip_cfg.min_post_seconds * 2,
            max_after_s=clip_cfg.max_post_seconds * 2,  # Double the window
            prefer_after=True,
        )
    
    if not start_candidates or not end_candidates:
        return None
    
    # Try combinations of start/end boundaries to find one that fits duration constraints
    # Prioritize higher-scored boundaries (they come first in the lists)
    for start_bp in start_candidates[:10]:  # Top 10 start candidates
        for end_bp in end_candidates[:10]:  # Top 10 end candidates
            start_s = start_bp.time_s
            end_s = end_bp.time_s
            clip_duration = end_s - start_s
            
            if clip_duration >= clip_cfg.min_seconds and clip_duration <= clip_cfg.max_seconds:
                # Found a valid combination
                start_s = max(0.0, min(duration_s, start_s))
                end_s = max(0.0, min(duration_s, end_s))
                
                if end_s > start_s:
                    return {
                        "start_s": float(start_s),
                        "end_s": float(end_s),
                        "used_boundary_graph": True,
                    }
    
    # Relaxed pass: try any valid duration from expanded candidate set
    # Sort by proximity to ideal timing (start before peak, end after peak)
    all_starts = start_candidates[:20]
    all_ends = end_candidates[:20]
    
    for start_bp in all_starts:
        for end_bp in all_ends:
            start_s = start_bp.time_s
            end_s = end_bp.time_s
            clip_duration = end_s - start_s
            
            # More relaxed duration check (allow 75% to 150% of target range)
            min_dur = clip_cfg.min_seconds * 0.75
            max_dur = clip_cfg.max_seconds * 1.5
            
            if clip_duration >= min_dur and clip_duration <= max_dur:
                start_s = max(0.0, min(duration_s, start_s))
                end_s = max(0.0, min(duration_s, end_s))
                
                if end_s > start_s:
                    return {
                        "start_s": float(start_s),
                        "end_s": float(end_s),
                        "used_boundary_graph": True,
                    }
    
    return None


def shape_clip_bounds(
    *,
    peak_idx: int,
    scores: np.ndarray,
    hop_s: float,
    duration_s: float,
    clip_cfg: ClipConfig,
    scene_cuts: Iterable[float],
    snap_window_s: float,
    boundary_graph: Any = None,  # Optional BoundaryGraph for better shaping
) -> Dict[str, float]:
    """Shape clip bounds around a peak, using boundary graph if available.
    
    Priority:
    1. Use BoundaryGraph for natural edit points (sentence ends, silence, etc.)
    2. Fall back to valley detection in the signal + scene cut snapping
    """
    # Try boundary graph first (produces "feels edited" clips)
    if boundary_graph is not None:
        bg_result = shape_clip_bounds_with_boundaries(
            peak_idx=peak_idx,
            hop_s=hop_s,
            duration_s=duration_s,
            clip_cfg=clip_cfg,
            boundary_graph=boundary_graph,
        )
        if bg_result is not None:
            return bg_result
    
    # Fallback: valley detection method
    n = len(scores)
    peak_time = peak_idx * hop_s

    min_pre_frames = int(round(clip_cfg.min_pre_seconds / hop_s))
    max_pre_frames = int(round(clip_cfg.max_pre_seconds / hop_s))
    min_post_frames = int(round(clip_cfg.min_post_seconds / hop_s))
    max_post_frames = int(round(clip_cfg.max_post_seconds / hop_s))

    pre_start = max(0, peak_idx - max_pre_frames)
    pre_end = max(0, peak_idx - min_pre_frames)
    start_idx = _valley_index(scores, pre_start, pre_end) if pre_end > pre_start else max(0, peak_idx - min_pre_frames)

    post_start = min(n - 1, peak_idx + min_post_frames)
    post_end = min(n - 1, peak_idx + max_post_frames)
    end_idx = _valley_index(scores, post_start, post_end) if post_end > post_start else min(n - 1, peak_idx + min_post_frames)

    start_s = start_idx * hop_s
    end_s = end_idx * hop_s

    start_s = max(0.0, min(duration_s, start_s))
    end_s = max(0.0, min(duration_s, end_s))

    if end_s <= start_s:
        start_s = max(0.0, peak_time - clip_cfg.min_pre_seconds)
        end_s = min(duration_s, peak_time + clip_cfg.min_post_seconds)

    if end_s - start_s < clip_cfg.min_seconds:
        extra = clip_cfg.min_seconds - (end_s - start_s)
        start_s = max(0.0, start_s - extra / 2.0)
        end_s = min(duration_s, end_s + extra / 2.0)
        if end_s - start_s < clip_cfg.min_seconds:
            if start_s <= 0.0:
                end_s = min(duration_s, start_s + clip_cfg.min_seconds)
            elif end_s >= duration_s:
                start_s = max(0.0, end_s - clip_cfg.min_seconds)

    if end_s - start_s > clip_cfg.max_seconds:
        half = clip_cfg.max_seconds / 2.0
        start_s = max(0.0, peak_time - half)
        end_s = min(duration_s, peak_time + half)
        if end_s - start_s > clip_cfg.max_seconds:
            if start_s <= 0.0:
                end_s = min(duration_s, start_s + clip_cfg.max_seconds)
            elif end_s >= duration_s:
                start_s = max(0.0, end_s - clip_cfg.max_seconds)

    snapped_start = snap_time_to_cuts(start_s, scene_cuts, snap_window_s)
    snapped_end = snap_time_to_cuts(end_s, scene_cuts, snap_window_s)
    if snapped_end > snapped_start:
        start_s, end_s = snapped_start, snapped_end

    return {
        "start_s": float(max(0.0, min(duration_s, start_s))),
        "end_s": float(max(0.0, min(duration_s, end_s))),
    }


# ==============================================================================
# LLM Semantic Scoring for Highlight Candidates
# ==============================================================================

import json
import logging
import re as _re

_highlight_logger = logging.getLogger(__name__)

LLM_HIGHLIGHT_SYSTEM_PROMPT = """You are an expert video editor analyzing highlight candidates for a gaming/streaming video.

Your task is to evaluate candidate moments based on their transcript content and assign semantic scores.

Score each candidate from 0.0 to 1.0 based on:
- Excitement/energy in the speech (reactions, exclamations)
- Entertainment value (funny, surprising, impressive moments)
- Quotability (memorable phrases viewers might share)
- Context completeness (does it tell a mini-story or capture a full moment?)

Output ONLY valid JSON. No explanations outside the JSON."""


# Content type descriptions for LLM prompt customization
CONTENT_TYPE_GUIDANCE = {
    "gaming": {
        "description": "gaming/streaming content",
        "high_value": [
            "Genuine hype moments - real excitement, not just loud noises",
            "Clutch plays with authentic reactions",
            "Funny fails or unexpected moments",
            "Rage moments that are entertaining (not just angry)",
            "Victory celebrations with quotable phrases",
            "Genuine surprise or shock at game events",
        ],
        "low_value": [
            "Generic gameplay commentary ('okay let me just...')",
            "Reading donations/chat without interesting reaction",
            "Routine play-by-play that isn't exciting",
            "Incomplete sentences or cut-off thoughts",
            "Background noise without clear content",
        ],
    },
    "reaction": {
        "description": "reaction/commentary content",
        "high_value": [
            "Strong emotional reactions (laughing, shocked, moved)",
            "Insightful or funny commentary",
            "Memorable quotes or hot takes",
            "Genuine surprise at content being watched",
            "Relatable reactions viewers would share",
        ],
        "low_value": [
            "Silent watching without commentary",
            "Vague reactions ('that's interesting')",
            "Just describing what's on screen",
            "Mid-thought interruptions",
        ],
    },
    "podcast": {
        "description": "podcast/interview/discussion content",
        "high_value": [
            "Controversial or surprising statements",
            "Funny stories or anecdotes",
            "Emotional or vulnerable moments",
            "Sharp comebacks or witty exchanges",
            "Quotable insights or hot takes",
            "Heated debates with good points",
        ],
        "low_value": [
            "Administrative talk ('so anyway...')",
            "Long-winded setup without payoff",
            "Unclear inside jokes",
            "Topic transitions without content",
        ],
    },
    "irl": {
        "description": "IRL/vlog content",
        "high_value": [
            "Unexpected events or encounters",
            "Funny interactions with people",
            "Genuine emotional moments",
            "Fails or embarrassing situations",
            "Wholesome or heartwarming moments",
        ],
        "low_value": [
            "Walking without events",
            "Mundane logistics talk",
            "Unclear audio or crowd noise",
            "Incomplete interactions",
        ],
    },
    "music": {
        "description": "music/performance content",
        "high_value": [
            "Impressive musical moments or riffs",
            "Crowd reactions during performance",
            "Emotional song moments",
            "Funny banter between songs",
            "Technical skill showcase",
        ],
        "low_value": [
            "Tuning or setup",
            "Incomplete song fragments",
            "Technical difficulties",
        ],
    },
    "educational": {
        "description": "educational/tutorial content",
        "high_value": [
            "Key insights or 'aha' moments",
            "Surprising facts or revelations",
            "Clear explanations of complex topics",
            "Demonstrations with visible results",
            "Memorable analogies or examples",
        ],
        "low_value": [
            "Setup or context without payoff",
            "Reading from notes/slides",
            "Incomplete explanations",
        ],
    },
}


def _build_llm_highlight_prompt(
    candidates: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    chat_context: Optional[str] = None,
    content_type: str = "gaming",
) -> str:
    """Build prompt for LLM to semantically score highlight candidates.
    
    Args:
        candidates: List of candidate dicts with timing and score info
        transcript_segments: Transcript segments with start/end/text
        chat_context: Optional chat activity summary
        content_type: Type of content (gaming, reaction, podcast, irl, music, educational)
    """
    import bisect
    
    # Get content-specific guidance
    guidance = CONTENT_TYPE_GUIDANCE.get(
        content_type.lower(), 
        CONTENT_TYPE_GUIDANCE["gaming"]  # fallback
    )
    content_desc = guidance["description"]
    high_value_items = "\n".join(f"  - {item}" for item in guidance["high_value"])
    low_value_items = "\n".join(f"  - {item}" for item in guidance["low_value"])
    
    # Pre-sort segments by start time for faster lookup
    sorted_segments = sorted(transcript_segments, key=lambda s: s.get("start", 0.0))
    seg_starts = [s.get("start", 0.0) for s in sorted_segments]
    
    # Get transcript text for each candidate's time window
    candidate_data = []
    for cand in candidates:
        start_s = cand.get("start_s", 0.0)
        end_s = cand.get("end_s", 0.0)
        
        # Use binary search to find relevant segments (much faster for large transcripts)
        start_idx = bisect.bisect_left(seg_starts, start_s)
        start_idx = max(0, start_idx - 5)
        
        relevant_text = []
        for i in range(start_idx, len(sorted_segments)):
            seg = sorted_segments[i]
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            
            if seg_start > end_s:
                break
            
            if seg_end >= start_s and seg_start <= end_s:
                relevant_text.append(seg.get("text", "").strip())
        
        candidate_data.append({
            "rank": cand.get("rank", 0),
            "start_s": round(start_s, 1),
            "end_s": round(end_s, 1),
            "duration_s": round(end_s - start_s, 1),
            "signal_score": round(cand.get("score", 0.0), 2),
            "transcript": " ".join(relevant_text)[:500],
            "breakdown": {
                k: round(v, 2) if isinstance(v, (int, float)) else v 
                for k, v in cand.get("breakdown", {}).items()
            },
        })
    
    prompt_data = {
        "task": f"Score {content_desc} highlight candidates",
        "candidates": candidate_data,
    }
    
    if chat_context:
        prompt_data["chat_context"] = chat_context[:300]
    
    prompt = f"""You are scoring highlight candidates from {content_desc}.

IMPORTANT: The signal_score already measures audio energy, chat activity, and motion.
High signal_score = loud/active moment. Your job is to evaluate CONTENT QUALITY.

Only give LOW semantic scores if the transcript is genuinely boring DESPITE the activity.
If a moment has high signal_score AND good content, give it a HIGH semantic score.
Trust the signals - they detected this moment for a reason.

Input:
{json.dumps(prompt_data, indent=2)}

For {content_desc}, give HIGH semantic scores (0.7-1.0) for:
{high_value_items}

Give LOW semantic scores (0.0-0.4) for:
{low_value_items}

Medium scores (0.4-0.7) for decent content that's not exceptional.

Output JSON object with "scores" array:
{{
  "scores": [
    {{"rank": 1, "semantic_score": 0.8, "reason": "brief explanation", "best_quote": "short quotable phrase"}},
    {{"rank": 2, "semantic_score": 0.6, "reason": "brief explanation", "best_quote": "short quotable phrase"}}
  ]
}}

Respond with ONLY the JSON object:"""
    
    return prompt


def _parse_llm_highlight_scores(
    response_text: str,
    num_candidates: int,
) -> Dict[int, Dict[str, Any]]:
    """Parse LLM response into semantic scores by rank.

    The codebase originally expected a JSON **array** response, but when using
    OpenAI-compatible servers with `response_format={"type":"json_object"}`, many
    models will (correctly) return a top-level JSON **object** instead.

    This parser accepts:
    - JSON array: [{rank, semantic_score, reason, best_quote}, ...]
    - JSON object mapping rank -> {...} (e.g. {"1": {...}, "2": {...}})
    - Wrapper objects with one of: scores/results/items/candidates/data -> list
    
    Returns: Dict mapping rank -> {"semantic_score": float, "reason": str, "best_quote": str}
    """
    result: Dict[int, Dict[str, Any]] = {}

    try:
        text = response_text.strip()

        # Handle markdown code blocks
        if "```" in text:
            match = _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, _re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Try to extract a JSON object/array if the model included extra text
        if not (text.startswith("{") or text.startswith("[")):
            match = _re.search(r"(\{.*\}|\[.*\])", text, _re.DOTALL)
            if match:
                text = match.group(1).strip()

        parsed = json.loads(text)

        items: List[Dict[str, Any]] = []

        if isinstance(parsed, list):
            items = [it for it in parsed if isinstance(it, dict)]

        elif isinstance(parsed, dict):
            # Common wrappers - check for list values
            for key in ("scores", "results", "items", "candidates", "data", "semantic_scores", "highlights"):
                val = parsed.get(key)
                if isinstance(val, list):
                    items = [it for it in val if isinstance(it, dict)]
                    break

            # Dict-of-dicts: {"1": {...}, "2": {...}} or {"candidate_1": {...}}
            # Check if we have dict values (allows for metadata keys like "model", "version" etc.)
            if not items:
                dict_values = [(k, v) for k, v in parsed.items() if isinstance(v, dict)]
                if dict_values:
                    for k, v in dict_values:
                        item = dict(v)
                        if "rank" not in item:
                            # Try to extract rank from key like "1", "candidate_1", etc.
                            try:
                                digits = "".join(c for c in str(k) if c.isdigit())
                                if digits:
                                    item["rank"] = int(digits)
                            except Exception:
                                pass
                        items.append(item)

        for item in items:
            if "rank" not in item:
                continue

            try:
                rank = int(item["rank"])
            except Exception:
                continue

            # Ignore nonsense/out-of-range ranks if we know the candidate count
            if num_candidates and not (1 <= rank <= num_candidates):
                continue

            try:
                semantic = float(item.get("semantic_score", 0.5))
            except Exception:
                semantic = 0.5
            semantic = max(0.0, min(1.0, semantic))

            result[rank] = {
                "semantic_score": semantic,
                "reason": str(item.get("reason", "")),
                "best_quote": str(item.get("best_quote", "")),
            }

    except Exception as e:
        _highlight_logger.warning(f"Failed to parse LLM highlight scores: {e}")

    return result


def _build_llm_filter_prompt(
    candidates: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    content_type: str = "gaming",
) -> str:
    """Build prompt for LLM to filter highlight candidates by content quality.
    
    This is a more aggressive filter than semantic scoring - it asks the LLM
    to identify which candidates have actually interesting/entertaining content
    vs those that just have high signal activity but boring content.
    """
    import bisect
    
    # Pre-sort segments by start time for faster lookup
    sorted_segments = sorted(transcript_segments, key=lambda s: s.get("start", 0.0))
    seg_starts = [s.get("start", 0.0) for s in sorted_segments]
    
    # Get transcript text for each candidate's time window
    candidate_data = []
    for cand in candidates:
        start_s = cand.get("start_s", 0.0)
        end_s = cand.get("end_s", 0.0)
        
        # Use binary search to find relevant segments
        start_idx = bisect.bisect_left(seg_starts, start_s)
        start_idx = max(0, start_idx - 5)
        
        relevant_text = []
        for i in range(start_idx, len(sorted_segments)):
            seg = sorted_segments[i]
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            
            if seg_start > end_s:
                break
            
            if seg_end >= start_s and seg_start <= end_s:
                relevant_text.append(seg.get("text", "").strip())
        
        candidate_data.append({
            "rank": cand.get("rank", 0),
            "start_s": round(start_s, 1),
            "end_s": round(end_s, 1),
            "duration_s": round(end_s - start_s, 1),
            "transcript": " ".join(relevant_text)[:600],  # Slightly more context for filtering
        })
    
    prompt = f"""You are a content quality filter for {content_type} stream highlights.

Your job is to REJECT clips that are boring or low quality, even if they had high audio/activity signals.
Be STRICT - only KEEP clips that would genuinely entertain viewers.

Candidates to evaluate:
{json.dumps(candidate_data, indent=2)}

For each candidate, decide:
- "keep": genuinely entertaining, funny, exciting, or memorable moment
- "reject": boring, mundane commentary, incomplete thought, nothing interesting happens

REJECT if the transcript shows:
- Just reading chat/donations without humor
- Generic gameplay commentary ("okay let's go here")
- Incomplete sentences or unclear context
- Nothing quotable or shareable

KEEP if the transcript shows:
- Genuine reactions (surprise, excitement, frustration)
- Funny moments or jokes
- Impressive plays with good commentary
- Quotable/memorable phrases
- Strong emotional moments

Output JSON object with "results" array:
{{
  "results": [
    {{"rank": 1, "decision": "keep", "quality_score": 8, "reason": "brief reason"}},
    {{"rank": 2, "decision": "reject", "quality_score": 3, "reason": "brief reason"}}
  ]
}}

quality_score is 1-10 (10 = amazing clip, 1 = terrible).
Be strict! It's better to have fewer great clips than many mediocre ones.

Respond with ONLY the JSON object:"""
    
    return prompt


def _parse_llm_filter_response(
    response_text: str,
) -> Dict[int, Dict[str, Any]]:
    """Parse LLM filter response.
    
    Returns: Dict mapping rank -> {"decision": "keep"/"reject", "quality_score": int, "reason": str}
    """
    result = {}
    
    try:
        text = response_text.strip()
        
        # Handle markdown code blocks
        if "```" in text:
            match = _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, _re.DOTALL)
            if match:
                text = match.group(1).strip()
        
        # Try to extract a JSON object/array if the model included extra text
        if not (text.startswith("{") or text.startswith("[")):
            match = _re.search(r"(\{.*\}|\[.*\])", text, _re.DOTALL)
            if match:
                text = match.group(1).strip()

        parsed = json.loads(text)

        items: List[Dict[str, Any]] = []

        if isinstance(parsed, list):
            items = [it for it in parsed if isinstance(it, dict)]

        elif isinstance(parsed, dict):
            # Common wrappers - check for list values
            for key in ("scores", "results", "items", "candidates", "data", "filter_results", "filtered"):
                val = parsed.get(key)
                if isinstance(val, list):
                    items = [it for it in val if isinstance(it, dict)]
                    break

            # Dict-of-dicts: {"1": {...}, "2": {...}} or {"candidate_1": {...}}
            # Check if we have dict values (allows for metadata keys)
            if not items:
                dict_values = [(k, v) for k, v in parsed.items() if isinstance(v, dict)]
                if dict_values:
                    for k, v in dict_values:
                        item = dict(v)
                        if "rank" not in item:
                            # Try to extract rank from key like "1", "candidate_1", etc.
                            try:
                                digits = "".join(c for c in str(k) if c.isdigit())
                                if digits:
                                    item["rank"] = int(digits)
                            except Exception:
                                pass
                        items.append(item)

        for item in items:
            if "rank" not in item:
                continue

            try:
                rank = int(item["rank"])
            except Exception:
                continue

            decision = str(item.get("decision", "keep")).strip().lower()
            if decision not in ("keep", "reject"):
                decision = "keep"

            try:
                quality = int(item.get("quality_score", 5))
            except Exception:
                quality = 5
            quality = max(1, min(10, quality))

            result[rank] = {
                "decision": decision,
                "quality_score": quality,
                "reason": str(item.get("reason", "")),
            }

    except Exception as e:
        _highlight_logger.warning(f"Failed to parse LLM filter response: {e}")
        _highlight_logger.debug(f"Raw response was: {response_text[:500] if response_text else '(empty)'}...")

    if not result:
        _highlight_logger.warning(f"[llm_filter_parse] Parsed 0 items. Response preview: {response_text[:300] if response_text else '(empty)'}...")
    else:
        _highlight_logger.info(f"[llm_filter_parse] Parsed {len(result)} filter results")

    return result


def compute_llm_filter(
    candidates: List[Dict[str, Any]],
    proj: "Project",
    llm_complete: Callable[[str], str],
    *,
    min_quality_score: int = 5,
    max_keep: Optional[int] = None,
    content_type: str = "gaming",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Use LLM to filter out low-quality candidates.
    
    This is more aggressive than semantic scoring - it actually removes candidates
    that the LLM determines have boring/uninteresting content.
    
    Args:
        candidates: List of candidate dicts
        proj: Project instance for accessing transcript
        llm_complete: Function that takes prompt string and returns response string
        min_quality_score: Minimum score (1-10) to keep a candidate
        max_keep: Maximum candidates to keep (None = no limit, just use threshold)
        content_type: Type of content for prompt context (gaming, irl, etc.)
        
    Returns:
        Tuple of (filtered_candidates, filter_stats)
    """
    from .analysis_transcript import load_transcript
    
    # Load transcript
    transcript = load_transcript(proj)
    if transcript is None:
        _highlight_logger.info("[llm_filter] No transcript available, skipping filter")
        return candidates, {"skipped": True, "reason": "no_transcript"}
    
    if not candidates:
        return candidates, {"skipped": True, "reason": "no_candidates"}
    
    # Get transcript segments
    transcript_segments = [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in transcript.segments
    ]
    
    # Build and send prompt
    prompt = _build_llm_filter_prompt(candidates, transcript_segments, content_type)
    
    try:
        _highlight_logger.info(f"[llm_filter] Sending {len(candidates)} candidates for quality filtering...")
        
        import time as _t
        start = _t.time()
        response = llm_complete(prompt)
        elapsed = _t.time() - start
        
        # Coerce response to text (handles string, dict, list, OpenAI-style payloads)
        response_text = _coerce_llm_response_to_text(response)
        
        _highlight_logger.info(f"[llm_filter] LLM responded in {elapsed:.1f}s ({len(response_text)} chars)")
        
        filter_results = _parse_llm_filter_response(response_text)
        
        if not filter_results:
            _highlight_logger.warning("[llm_filter] No valid filter results, keeping all candidates")
            return candidates, {"skipped": True, "reason": "parse_failed"}
        
        # Apply filtering
        kept = []
        rejected = []
        
        for cand in candidates:
            rank = cand.get("rank", 0)
            result = filter_results.get(rank, {"decision": "keep", "quality_score": 5, "reason": ""})
            
            # Add filter metadata to candidate
            cand["llm_filter"] = result
            cand["quality_score"] = result["quality_score"]
            
            if result["decision"] == "keep" and result["quality_score"] >= min_quality_score:
                kept.append(cand)
            else:
                rejected.append(cand)
        
        # Sort kept by quality score (highest first)
        kept.sort(key=lambda c: c.get("quality_score", 0), reverse=True)
        
        # Apply max_keep limit if specified
        if max_keep is not None and len(kept) > max_keep:
            extra_rejected = kept[max_keep:]
            kept = kept[:max_keep]
            rejected.extend(extra_rejected)
        
        # Re-rank kept candidates
        for i, cand in enumerate(kept, start=1):
            cand["rank_before_filter"] = cand["rank"]
            cand["rank"] = i
        
        stats = {
            "total_input": len(candidates),
            "kept": len(kept),
            "rejected": len(rejected),
            "min_quality_score": min_quality_score,
            "max_keep": max_keep,
            "elapsed_s": round(elapsed, 2),
            "rejected_ranks": [c.get("rank_before_filter", c.get("rank")) for c in rejected],
        }
        
        _highlight_logger.info(
            f"[llm_filter] Kept {len(kept)}/{len(candidates)} candidates "
            f"(rejected {len(rejected)} below quality threshold)"
        )
        
        return kept, stats
        
    except Exception as e:
        _highlight_logger.warning(f"[llm_filter] Error: {e}, keeping all candidates")
        return candidates, {"skipped": True, "reason": str(e)}


def compute_llm_semantic_scores(
    candidates: List[Dict[str, Any]],
    proj: "Project",
    llm_complete: Callable[[str], str],
    *,
    max_candidates: int = 20,
    content_type: str = "gaming",
) -> Dict[int, Dict[str, Any]]:
    """Use LLM to compute semantic scores for highlight candidates.
    
    Args:
        candidates: List of candidate dicts with start_s, end_s, score, breakdown
        proj: Project instance for accessing transcript
        llm_complete: Function that takes prompt string and returns response string
        max_candidates: Maximum candidates to send to LLM (for token limits)
        content_type: Type of content for prompt customization (gaming, reaction, podcast, irl, music, educational)
        
    Returns:
        Dict mapping candidate rank -> {"semantic_score": float, "reason": str, "best_quote": str}
    """
    from .analysis_transcript import load_transcript
    
    # Load transcript
    transcript = load_transcript(proj)
    if transcript is None:
        _highlight_logger.info("No transcript available for LLM semantic scoring")
        return {}
    
    # Get transcript segments as dicts
    transcript_segments = [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in transcript.segments
    ]
    
    # Limit candidates for LLM
    candidates_to_score = candidates[:max_candidates]
    
    if not candidates_to_score:
        return {}
    
    # Build prompt with content type
    _highlight_logger.info(f"Using content_type='{content_type}' for semantic scoring prompt")
    prompt = _build_llm_highlight_prompt(candidates_to_score, transcript_segments, content_type=content_type)
    
    try:
        prompt_len = len(prompt)
        _highlight_logger.info(f"Requesting LLM semantic scores for {len(candidates_to_score)} candidates (prompt: {prompt_len} chars)")
        
        import time as _t
        start = _t.time()
        response = llm_complete(prompt)
        elapsed = _t.time() - start
        
        # Coerce response to text (handles string, dict, list, OpenAI-style payloads)
        response_text = _coerce_llm_response_to_text(response)
        
        _highlight_logger.info(f"LLM responded in {elapsed:.1f}s ({len(response_text)} chars)")
        _highlight_logger.debug(f"LLM raw response type: {type(response)}, text preview: {response_text[:500]}...")
        
        scores = _parse_llm_highlight_scores(response_text, len(candidates_to_score))
        
        if not scores:
            _highlight_logger.warning(f"LLM response could not be parsed into scores. Response preview: {response_text[:1000]}")
        else:
            _highlight_logger.info(f"LLM returned semantic scores for {len(scores)} candidates")
        
        return scores
    except Exception as e:
        _highlight_logger.warning(f"LLM semantic scoring failed: {e}", exc_info=True)
        return {}


def compute_highlights_analysis(
    proj: Project,
    *,
    audio_cfg: Dict[str, Any],
    motion_cfg: Dict[str, Any],
    scenes_cfg: Dict[str, Any],
    highlights_cfg: Dict[str, Any],
    audio_events_cfg: Optional[Dict[str, Any]] = None,
    include_chat: bool = True,
    include_audio_events: bool = True,
    llm_complete: Optional[Callable[[str], str]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute highlight analysis with optional LLM semantic scoring.
    
    Args:
        proj: Project instance
        audio_cfg: Audio analysis config
        motion_cfg: Motion analysis config  
        scenes_cfg: Scenes analysis config
        highlights_cfg: Highlights config with weights, clip settings, etc.
        audio_events_cfg: Audio events (laughter, applause) config
        include_chat: Whether to include chat signal
        include_audio_events: Whether to include audio events signal
        llm_complete: Optional LLM completion function for semantic scoring
        on_progress: Optional progress callback
        
    Returns:
        Dict with candidates, weights, config used, etc.
    """
    start_time = _time.time()
    proj_data = get_project_data(proj)

    hop_s = float(audio_cfg.get("hop_seconds", 0.5))
    if not proj.audio_features_path.exists():
        if on_progress:
            on_progress(0.02)  # Starting audio analysis
        compute_audio_analysis(
            proj,
            sample_rate=int(audio_cfg.get("sample_rate", 16000)),
            hop_s=hop_s,
            smooth_s=float(audio_cfg.get("smooth_seconds", 3.0)),
            top=int(audio_cfg.get("top", 12)),
            min_gap_s=float(audio_cfg.get("min_gap_seconds", 20.0)),
            pre_s=float(audio_cfg.get("pre_seconds", 8.0)),
            post_s=float(audio_cfg.get("post_seconds", 22.0)),
            skip_start_s=float(audio_cfg.get("skip_start_seconds", 10.0)),
            on_progress=None,
        )

    if not proj.motion_features_path.exists():
        if on_progress:
            on_progress(0.10)  # Starting motion analysis
        compute_motion_analysis(
            proj,
            sample_fps=float(motion_cfg.get("sample_fps", 3.0)),
            scale_width=int(motion_cfg.get("scale_width", 160)),
            smooth_s=float(motion_cfg.get("smooth_seconds", 2.5)),
            on_progress=None,
        )

    scenes_enabled = bool(scenes_cfg.get("enabled", True))
    if scenes_enabled and not proj.scenes_path.exists():
        if on_progress:
            on_progress(0.20)  # Starting scene detection
        compute_scene_analysis(
            proj,
            threshold_z=float(scenes_cfg.get("threshold_z", 3.5)),
            min_scene_len_seconds=float(scenes_cfg.get("min_scene_len_seconds", 1.2)),
            snap_window_seconds=float(scenes_cfg.get("snap_window_seconds", 1.0)),
            on_progress=None,
        )

    # Chat analysis: prefer SQLite store, fall back to raw JSON
    if include_chat and not proj.chat_features_path.exists():
        if on_progress:
            on_progress(0.25)  # Starting chat analysis
        chat_db_path = proj.analysis_dir / "chat.sqlite"
        if chat_db_path.exists():
            # Use new SQLite-based chat features
            # Note: LLM learning is auto-enabled but requires llm_complete to be passed
            # When called from here (headless), we skip LLM learning and use seed tokens only
            from .chat.features import compute_and_save_chat_features
            compute_and_save_chat_features(
                proj,
                hop_s=hop_s,
                smooth_s=float(highlights_cfg.get("chat_smooth_seconds", 3.0)),
                on_progress=None,
                llm_complete=None,  # Headless mode: seed tokens only
            )
        elif proj.chat_raw_path.exists():
            # Fall back to legacy JSON-based analysis
            compute_chat_analysis(
                proj,
                chat_path=proj.chat_raw_path,
                hop_s=hop_s,
                smooth_s=float(highlights_cfg.get("chat_smooth_seconds", 3.0)),
                on_progress=None,
            )

    if on_progress:
        on_progress(0.35)  # Loading audio features

    audio_data = load_npz(proj.audio_features_path)
    audio_scores = audio_data.get("scores")
    if audio_scores is None:
        raise ValueError("audio_features.npz missing scores")
    audio_scores = audio_scores.astype(np.float64)

    if on_progress:
        on_progress(0.38)  # Loading motion features

    motion_data = load_npz(proj.motion_features_path)
    motion_scores = motion_data.get("scores")
    if motion_scores is None:
        raise ValueError("motion_features.npz missing scores")
    motion_scores = motion_scores.astype(np.float64)
    motion_fps_arr = motion_data.get("fps")
    motion_fps = float(motion_fps_arr[0]) if motion_fps_arr is not None and len(motion_fps_arr) > 0 else 1.0
    motion_hop_s = 1.0 / motion_fps
    motion_resampled = resample_series(
        motion_scores,
        src_hop_s=motion_hop_s,
        target_hop_s=hop_s,
        target_len=len(audio_scores),
        fill_value=0.0,
    )

    chat_scores = np.zeros_like(audio_scores)
    chat_used = False
    if include_chat and proj.chat_features_path.exists():
        chat_data = load_npz(proj.chat_features_path)
        chat_raw = chat_data.get("scores")
        chat_hop_arr = chat_data.get("hop_seconds")
        if chat_raw is not None and chat_hop_arr is not None and len(chat_hop_arr) > 0:
            chat_scores = resample_series(
                chat_raw.astype(np.float64),
                src_hop_s=float(chat_hop_arr[0]),
                target_hop_s=hop_s,
                target_len=len(audio_scores),
                fill_value=0.0,
            )
            chat_used = True

    # Load audio events (7.3) if available, auto-compute if missing and requested
    audio_events_scores = np.zeros_like(audio_scores)
    audio_events_used = False
    if include_audio_events:
        # Fix #12: Auto-compute audio events if missing
        if not proj.audio_events_features_path.exists() and audio_events_cfg is not None:
            # Check video duration - skip auto-compute for long videos (>1 hour)
            # Audio events ML inference can take 30+ minutes for 4-hour videos
            try:
                from .analysis_audio_events import compute_audio_events_analysis, AudioEventsConfig
                if on_progress:
                    on_progress(0.40)  # Starting audio events ML analysis
                
                # Create a sub-progress callback to scale audio events progress into (0.40-0.55)
                def audio_events_progress(p: float) -> None:
                    if on_progress:
                        scaled = 0.40 + p * 0.15
                        on_progress(min(0.55, scaled))
                
                compute_audio_events_analysis(
                    proj,
                    cfg=AudioEventsConfig.from_dict(audio_events_cfg),
                    on_progress=audio_events_progress,
                )
            except Exception:
                pass  # Continue without audio events if computation fails
        
        if proj.audio_events_features_path.exists():
            try:
                events_data = load_npz(proj.audio_events_features_path)
                events_raw = events_data.get("event_combo_z")
                events_hop_arr = events_data.get("hop_seconds")
                if events_raw is not None and events_hop_arr is not None and len(events_hop_arr) > 0:
                    audio_events_scores = resample_series(
                        events_raw.astype(np.float64),
                        src_hop_s=float(events_hop_arr[0]),
                        target_hop_s=hop_s,
                        target_len=len(audio_scores),
                        fill_value=0.0,
                    )
                    audio_events_used = True
            except Exception:
                # If loading fails, continue without audio events
                pass

    # Load speech features (speech rate, pitch variation, etc.)
    speech_scores = np.zeros_like(audio_scores)
    speech_used = False
    if proj.speech_features_path.exists():
        try:
            speech_data = load_npz(proj.speech_features_path)
            # Prefer speech_score if available, else compute from word_count + speech_rate
            speech_raw = speech_data.get("speech_score")
            if speech_raw is None:
                # Fallback: use word_count z-score as speech activity indicator
                word_count = speech_data.get("word_count")
                if word_count is not None:
                    speech_raw = robust_z(word_count.astype(np.float64))
            
            speech_hop_arr = speech_data.get("hop_seconds")
            if speech_raw is not None and speech_hop_arr is not None and len(speech_hop_arr) > 0:
                speech_scores = resample_series(
                    speech_raw.astype(np.float64),
                    src_hop_s=float(speech_hop_arr[0]),
                    target_hop_s=hop_s,
                    target_len=len(audio_scores),
                    fill_value=0.0,
                )
                speech_used = True
        except Exception:
            pass  # Continue without speech features

    # Load reaction audio features (streamer reaction loudness)
    reaction_scores = np.zeros_like(audio_scores)
    reaction_used = False
    if proj.reaction_audio_features_path.exists():
        try:
            reaction_data = load_npz(proj.reaction_audio_features_path)
            reaction_raw = reaction_data.get("scores")
            reaction_hop_arr = reaction_data.get("hop_seconds")
            if reaction_raw is not None and reaction_hop_arr is not None and len(reaction_hop_arr) > 0:
                reaction_scores = resample_series(
                    reaction_raw.astype(np.float64),
                    src_hop_s=float(reaction_hop_arr[0]),
                    target_hop_s=hop_s,
                    target_len=len(audio_scores),
                    fill_value=0.0,
                )
                reaction_used = True
        except Exception:
            pass  # Continue without reaction audio

    weights = highlights_cfg.get("weights", {})
    w_audio = float(weights.get("audio", 0.45))
    w_motion = float(weights.get("motion", 0.15))

    w_chat_cfg = float(weights.get("chat", 0.20))
    w_chat = w_chat_cfg if chat_used else 0.0

    w_audio_events_cfg = float(weights.get("audio_events", 0.20))
    w_audio_events = w_audio_events_cfg if audio_events_used else 0.0

    w_speech_cfg = float(weights.get("speech", 0.15))
    w_speech = w_speech_cfg if speech_used else 0.0

    w_reaction_cfg = float(weights.get("reaction", 0.15))
    w_reaction = w_reaction_cfg if reaction_used else 0.0

    # Normalize weights (pure scaling; ratios are preserved)
    w_total = w_audio + w_motion + w_chat + w_audio_events + w_speech + w_reaction
    if w_total > 0 and abs(w_total - 1.0) > 0.001:
        w_audio = w_audio / w_total
        w_motion = w_motion / w_total
        w_chat = w_chat / w_total
        w_audio_events = w_audio_events / w_total
        w_speech = w_speech / w_total
        w_reaction = w_reaction / w_total

    use_relu = bool(highlights_cfg.get("relu_zscores", True))

    # Per-signal positive transform (prevents one signal from dominating due to outliers)
    signal_clip_z = float(highlights_cfg.get("signal_clip_z", 6.0))
    signal_softclip = str(highlights_cfg.get("signal_softclip", "clip")).lower().strip()

    # Motion gating: motion matters more when *something* else is happening
    use_motion_gating = bool(highlights_cfg.get("motion_gating", True))
    motion_gate_threshold_z = float(highlights_cfg.get("motion_gate_threshold_z", 0.0))
    motion_gate_scale_z = float(highlights_cfg.get("motion_gate_scale_z", 2.0))

    # Audio-events gating: suppress quiet classifier noise
    use_audio_events_gating = bool(highlights_cfg.get("audio_events_gating", True))
    audio_events_gate_threshold_z = float(highlights_cfg.get("audio_events_gate_threshold_z", 0.0))
    audio_events_gate_scale_z = float(highlights_cfg.get("audio_events_gate_scale_z", 2.0))

    # Initialize gate arrays for debugging output
    motion_gate = np.ones_like(audio_scores)
    audio_events_gate = np.ones_like(audio_scores)

    if use_relu:
        # Apply positive transform with optional soft clipping
        audio_scores_pos = _pos_transform(audio_scores, clip_z=signal_clip_z, mode=signal_softclip)
        motion_resampled_pos = _pos_transform(motion_resampled, clip_z=signal_clip_z, mode=signal_softclip)
        chat_scores_pos = _pos_transform(chat_scores, clip_z=signal_clip_z, mode=signal_softclip)
        audio_events_scores_pos = _pos_transform(audio_events_scores, clip_z=signal_clip_z, mode=signal_softclip)
        speech_scores_pos = _pos_transform(speech_scores, clip_z=signal_clip_z, mode=signal_softclip)
        reaction_scores_pos = _pos_transform(reaction_scores, clip_z=signal_clip_z, mode=signal_softclip)

        # Motion gating: motion only contributes when audio or chat activity is present
        if use_motion_gating and w_motion > 0:
            motion_gate = _linear_gate(
                np.maximum(audio_scores_pos, chat_scores_pos),
                threshold=motion_gate_threshold_z,
                scale=motion_gate_scale_z,
            )
            term_motion = w_motion * motion_resampled_pos * motion_gate
        else:
            term_motion = w_motion * motion_resampled_pos

        # Audio events gating: suppress when audio is quiet
        if use_audio_events_gating and w_audio_events > 0:
            audio_events_gate = _linear_gate(
                audio_scores_pos,
                threshold=audio_events_gate_threshold_z,
                scale=audio_events_gate_scale_z,
            )
            term_audio_events = w_audio_events * audio_events_scores_pos * audio_events_gate
        else:
            term_audio_events = w_audio_events * audio_events_scores_pos

        term_audio = w_audio * audio_scores_pos
        term_chat = w_chat * chat_scores_pos
        term_speech = w_speech * speech_scores_pos
        term_reaction = w_reaction * reaction_scores_pos

        combined_raw = (
            term_audio +
            term_motion +
            term_chat +
            term_audio_events +
            term_speech +
            term_reaction
        )
    else:
        # Non-ReLU path (still apply gating for consistency)
        if use_motion_gating and w_motion > 0:
            motion_gate = _linear_gate(
                np.maximum(audio_scores, chat_scores),
                threshold=motion_gate_threshold_z,
                scale=motion_gate_scale_z,
            )
            term_motion = w_motion * motion_resampled * motion_gate
        else:
            term_motion = w_motion * motion_resampled

        if use_audio_events_gating and w_audio_events > 0:
            audio_events_gate = _linear_gate(
                audio_scores,
                threshold=audio_events_gate_threshold_z,
                scale=audio_events_gate_scale_z,
            )
            term_audio_events = w_audio_events * audio_events_scores * audio_events_gate
        else:
            term_audio_events = w_audio_events * audio_events_scores

        term_audio = w_audio * audio_scores
        term_chat = w_chat * chat_scores
        term_speech = w_speech * speech_scores
        term_reaction = w_reaction * reaction_scores

        combined_raw = (
            term_audio +
            term_motion +
            term_chat +
            term_audio_events +
            term_speech +
            term_reaction
        )

    # Smooth + robust z-score the combined signal
    smooth_s = float(highlights_cfg.get("smooth_seconds", max(1.0, 2.0 * hop_s)))
    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    combined_smoothed = moving_average(combined_raw, smooth_frames) if len(combined_raw) > 0 else combined_raw
    combined_scores = robust_z(combined_smoothed) if len(combined_smoothed) > 0 else combined_smoothed

    # Peak selection masks (avoid persisting -inf into features unless explicitly wanted)
    scores_for_peaks = combined_scores.copy()
    skip_start_frames = int(round(float(highlights_cfg.get("skip_start_seconds", 10.0)) / hop_s))
    if skip_start_frames > 0 and skip_start_frames < len(scores_for_peaks):
        scores_for_peaks[:skip_start_frames] = -np.inf

    if on_progress:
        on_progress(0.60)  # Done with signal fusion

    min_gap_frames = max(1, int(round(float(highlights_cfg.get("min_gap_seconds", 15.0)) / hop_s)))
    min_peak_score = float(highlights_cfg.get("min_peak_score", 0.0))
    peak_idxs = pick_top_peaks(
        scores_for_peaks,
        top_k=int(highlights_cfg.get("top", 20)),
        min_gap_frames=min_gap_frames,
        min_score=min_peak_score,
    )

    if on_progress:
        on_progress(0.65)  # Done picking peaks

    scene_cuts = []
    snap_window_s = float(scenes_cfg.get("snap_window_seconds", 0.0))
    if scenes_enabled and proj.scenes_path.exists():
        scene_payload = json_load(proj.scenes_path)
        scene_cuts = scene_payload.get("cuts_seconds", []) or scene_payload.get("cuts", [])

    clip_cfg_dict = highlights_cfg.get("clip", {})
    clip_cfg = ClipConfig(
        min_seconds=float(clip_cfg_dict.get("min_seconds", 12.0)),
        max_seconds=float(clip_cfg_dict.get("max_seconds", 60.0)),
        min_pre_seconds=float(clip_cfg_dict.get("min_pre_seconds", 2.0)),
        max_pre_seconds=float(clip_cfg_dict.get("max_pre_seconds", 12.0)),
        min_post_seconds=float(clip_cfg_dict.get("min_post_seconds", 4.0)),
        max_post_seconds=float(clip_cfg_dict.get("max_post_seconds", 28.0)),
    )

    # Fix #9: Load boundary graph for better clip shaping
    boundary_graph = None
    try:
        from .analysis_boundaries import load_boundary_graph
        boundary_graph = load_boundary_graph(proj)
    except Exception:
        pass  # Boundary graph not available, will use valley fallback

    duration_s = float(proj_data.get("video", {}).get("duration_seconds", hop_s * len(audio_scores)))
    
    # Overlap filtering config
    max_overlap_ratio = float(highlights_cfg.get("max_overlap_ratio", 0.0))
    max_overlap_seconds = float(highlights_cfg.get("max_overlap_seconds", 0.0))
    overlap_denom = str(highlights_cfg.get("overlap_denominator", "shorter"))
    
    # Build raw candidates first (may have overlaps)
    raw_candidates: List[Dict[str, Any]] = []

    for rank, idx in enumerate(peak_idxs, start=1):
        bounds = shape_clip_bounds(
            peak_idx=idx,
            scores=combined_smoothed,
            hop_s=hop_s,
            duration_s=duration_s,
            clip_cfg=clip_cfg,
            scene_cuts=scene_cuts,
            snap_window_s=snap_window_s,
            boundary_graph=boundary_graph,
        )
        start_s = bounds["start_s"]
        end_s = bounds["end_s"]
        if end_s - start_s < max(1.0, clip_cfg.min_seconds * 0.5):
            continue
        
        # Compute weighted breakdown contributions (what actually goes into score)
        weighted_audio = float(term_audio[idx])
        weighted_motion = float(term_motion[idx])
        weighted_chat = float(term_chat[idx]) if chat_used else 0.0
        weighted_events = float(term_audio_events[idx]) if audio_events_used else 0.0
        weighted_speech = float(term_speech[idx]) if speech_used else 0.0
        weighted_reaction = float(term_reaction[idx]) if reaction_used else 0.0
        
        raw_candidates.append(
            {
                "rank": rank,
                "peak_time_s": float(idx * hop_s),
                "start_s": float(start_s),
                "end_s": float(end_s),
                "score": float(combined_scores[idx]),
                "used_boundary_graph": bounds.get("used_boundary_graph", False),
                "breakdown": {
                    "audio": weighted_audio,
                    "motion": weighted_motion,
                    "chat": weighted_chat,
                    "audio_events": weighted_events,
                    "speech": weighted_speech,
                    "reaction": weighted_reaction,
                },
                # Also keep raw signals for debugging
                "raw_signals": {
                    "audio": float(audio_scores[idx]),
                    "motion": float(motion_resampled[idx]),
                    "chat": float(chat_scores[idx]) if chat_used else 0.0,
                    "audio_events": float(audio_events_scores[idx]) if audio_events_used else 0.0,
                    "speech": float(speech_scores[idx]) if speech_used else 0.0,
                    "reaction": float(reaction_scores[idx]) if reaction_used else 0.0,
                },
            }
        )

    # Non-overlap filtering: sort by score desc, keep only if overlap is acceptable
    raw_candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    candidates: List[Dict[str, Any]] = []
    
    for cand in raw_candidates:
        if _check_overlap(
            cand["start_s"],
            cand["end_s"],
            candidates,
            max_overlap_ratio=max_overlap_ratio,
            max_overlap_seconds=max_overlap_seconds,
            denom=overlap_denom,
        ):
            # Re-assign rank based on filtered position
            cand["rank"] = len(candidates) + 1
            candidates.append(cand)

    if on_progress:
        on_progress(0.75)  # Done shaping candidates

    # LLM Semantic Scoring (optional enhancement)
    llm_semantic_used = False
    llm_semantic_scores: Dict[int, Dict[str, Any]] = {}
    
    if llm_complete is not None and candidates:
        if on_progress:
            on_progress(0.78)  # Starting LLM scoring
        
        try:
            _highlight_logger.info(f"[highlights] Starting LLM semantic scoring for {len(candidates)} candidates...")
            content_type = str(highlights_cfg.get("content_type", "gaming"))
            llm_semantic_scores = compute_llm_semantic_scores(
                candidates=candidates,
                proj=proj,
                llm_complete=llm_complete,
                max_candidates=int(highlights_cfg.get("llm_max_candidates", 15)),
                content_type=content_type,
            )
            
            if on_progress:
                on_progress(0.95)  # LLM scoring done
            
            if llm_semantic_scores:
                llm_semantic_used = True
                # Blend semantic score with signal score
                semantic_weight = float(highlights_cfg.get("llm_semantic_weight", 0.3))
                
                for cand in candidates:
                    rank = cand.get("rank", 0)
                    if rank in llm_semantic_scores:
                        llm_data = llm_semantic_scores[rank]
                        semantic_score = llm_data.get("semantic_score", 0.5)
                        signal_score = cand.get("score", 0.0)
                        
                        # Blend: final = (1 - w) * signal + w * semantic (normalized to signal scale)
                        # Semantic is 0-1, signal is z-score (~-2 to +4 typically)
                        # Convert semantic to z-score scale: (semantic - 0.5) * 4 gives ~-2 to +2
                        semantic_z = (semantic_score - 0.5) * 4.0
                        blended_score = (1.0 - semantic_weight) * signal_score + semantic_weight * semantic_z
                        
                        cand["score_signal"] = signal_score
                        cand["score_semantic"] = semantic_score
                        cand["score"] = blended_score
                        cand["llm_reason"] = llm_data.get("reason", "")
                        cand["llm_quote"] = llm_data.get("best_quote", "")
                
                # Re-sort by blended score and re-rank
                candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
                for i, cand in enumerate(candidates, start=1):
                    cand["rank"] = i
                    
                _highlight_logger.info(f"LLM semantic scoring applied to {len(llm_semantic_scores)} candidates")
        except Exception as e:
            _highlight_logger.warning(f"LLM semantic scoring failed, continuing without: {e}")

    save_npz(
        proj.highlights_features_path,
        combined_raw=combined_raw,
        combined_smoothed=combined_smoothed,
        combined_scores=combined_scores,
        scores_for_peaks=scores_for_peaks,
        audio_scores=audio_scores,
        motion_scores=motion_resampled,
        chat_scores=chat_scores,
        audio_events_scores=audio_events_scores,
        speech_scores=speech_scores,
        reaction_scores=reaction_scores,
        term_audio=term_audio,
        term_motion=term_motion,
        term_chat=term_chat,
        term_audio_events=term_audio_events,
        term_speech=term_speech,
        term_reaction=term_reaction,
        motion_gate=motion_gate,
        audio_events_gate=audio_events_gate,
        hop_seconds=np.array([hop_s], dtype=np.float64),
    )

    payload: Dict[str, Any] = {
        "method": "multi_signal_highlights_v6",
        "config": {
            "audio": audio_cfg,
            "motion": motion_cfg,
            "scenes": scenes_cfg,
            "highlights": highlights_cfg,
            "audio_events": audio_events_cfg,
        },
        "weights": {
            "audio": w_audio,
            "motion": w_motion,
            "chat": w_chat,
            "audio_events": w_audio_events,
            "speech": w_speech,
            "reaction": w_reaction,
        },
        "signals_used": {
            "chat": chat_used,
            "audio_events": audio_events_used,
            "speech": speech_used,
            "reaction": reaction_used,
            "boundary_graph": boundary_graph is not None,
            "llm_semantic": llm_semantic_used,
        },
        "signal_processing": {
            "relu_zscores": use_relu,
            "signal_clip_z": signal_clip_z,
            "signal_softclip": signal_softclip,
            "motion_gating": use_motion_gating,
            "motion_gate_threshold_z": motion_gate_threshold_z,
            "motion_gate_scale_z": motion_gate_scale_z,
            "audio_events_gating": use_audio_events_gating,
            "audio_events_gate_threshold_z": audio_events_gate_threshold_z,
            "audio_events_gate_scale_z": audio_events_gate_scale_z,
        },
        "filtering": {
            "max_overlap_ratio": max_overlap_ratio,
            "max_overlap_seconds": max_overlap_seconds,
            "overlap_denominator": overlap_denom,
            "min_peak_score": min_peak_score,
            "candidates_before_filter": len(raw_candidates),
            "candidates_after_filter": len(candidates),
        },
        "candidates": candidates,
        "elapsed_seconds": _time.time() - start_time,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["highlights"] = {
            **payload,
            "features_npz": str(proj.highlights_features_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload


def json_load(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Progress reporting helper
# =============================================================================

def _report_progress(
    on_progress: Optional[Callable[..., None]],
    frac: float,
    msg: Optional[str] = None
) -> None:
    """Report progress with optional message, handling both old and new signatures."""
    if not on_progress:
        return
    try:
        # Try new signature with message
        on_progress(frac, msg)
    except TypeError:
        # Fall back to old signature (fraction only)
        on_progress(frac)


# =============================================================================
# Split functions for DAG runner
# =============================================================================

def compute_highlights_scores(
    proj: Project,
    *,
    audio_cfg: Dict[str, Any],
    motion_cfg: Dict[str, Any],
    highlights_cfg: Dict[str, Any],
    audio_events_cfg: Optional[Dict[str, Any]] = None,
    include_chat: bool = True,
    include_audio_events: bool = True,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute signal fusion and peak detection (SCORING phase).
    
    This function:
    1. Loads all available signal features (audio, motion, chat, events, speech)
    2. Fuses them with weights
    3. Computes z-scores and picks peaks
    4. Saves highlights_features.npz with scores and peak indices
    
    This is separate from candidate shaping so that shaping can be re-run
    when boundary information improves.
    
    Args:
        proj: Project instance
        audio_cfg: Audio analysis config
        motion_cfg: Motion analysis config
        highlights_cfg: Highlights config with weights, etc.
        audio_events_cfg: Audio events config
        include_chat: Whether to include chat signal
        include_audio_events: Whether to include audio events signal
        on_progress: Progress callback
        
    Returns:
        Dict with peak_indices, weights, hop_s, signals_used
    """
    proj_data = get_project_data(proj)
    hop_s = float(audio_cfg.get("hop_seconds", 0.5))
    
    _report_progress(on_progress, 0.05, "Loading audio features")
    
    # Load audio features (required)
    if not proj.audio_features_path.exists():
        raise ValueError("audio_features.npz not found - run audio_features task first")
    
    audio_data = load_npz(proj.audio_features_path)
    audio_scores = audio_data.get("scores")
    if audio_scores is None:
        raise ValueError("audio_features.npz missing scores")
    audio_scores = audio_scores.astype(np.float64)
    
    _report_progress(on_progress, 0.10, "Loading motion features")
    
    # Load motion features (optional)
    motion_resampled = np.zeros_like(audio_scores)
    motion_used = False
    if proj.motion_features_path.exists():
        motion_data = load_npz(proj.motion_features_path)
        motion_scores = motion_data.get("scores")
        if motion_scores is not None:
            motion_fps_arr = motion_data.get("fps")
            motion_fps = float(motion_fps_arr[0]) if motion_fps_arr is not None and len(motion_fps_arr) > 0 else 1.0
            motion_hop_s = 1.0 / motion_fps
            motion_resampled = resample_series(
                motion_scores.astype(np.float64),
                src_hop_s=motion_hop_s,
                target_hop_s=hop_s,
                target_len=len(audio_scores),
                fill_value=0.0,
            )
            motion_used = True
    
    _report_progress(on_progress, 0.20, "Loading chat features")
    
    # Load chat features (optional)
    chat_scores = np.zeros_like(audio_scores)
    chat_used = False
    if include_chat and proj.chat_features_path.exists():
        chat_data = load_npz(proj.chat_features_path)
        chat_raw = chat_data.get("scores")
        chat_hop_arr = chat_data.get("hop_seconds")
        if chat_raw is not None and chat_hop_arr is not None and len(chat_hop_arr) > 0:
            chat_scores = resample_series(
                chat_raw.astype(np.float64),
                src_hop_s=float(chat_hop_arr[0]),
                target_hop_s=hop_s,
                target_len=len(audio_scores),
                fill_value=0.0,
            )
            chat_used = True
    
    _report_progress(on_progress, 0.30, "Loading audio events features")
    
    # Load audio events (optional)
    audio_events_scores = np.zeros_like(audio_scores)
    audio_events_used = False
    if include_audio_events and proj.audio_events_features_path.exists():
        try:
            events_data = load_npz(proj.audio_events_features_path)
            events_raw = events_data.get("event_combo_z")
            events_hop_arr = events_data.get("hop_seconds")
            if events_raw is not None and events_hop_arr is not None and len(events_hop_arr) > 0:
                audio_events_scores = resample_series(
                    events_raw.astype(np.float64),
                    src_hop_s=float(events_hop_arr[0]),
                    target_hop_s=hop_s,
                    target_len=len(audio_scores),
                    fill_value=0.0,
                )
                audio_events_used = True
        except Exception:
            pass
    
    _report_progress(on_progress, 0.40, "Loading speech features")
    
    # Load speech features (optional)
    speech_scores = np.zeros_like(audio_scores)
    speech_used = False
    if proj.speech_features_path.exists():
        try:
            speech_data = load_npz(proj.speech_features_path)
            speech_raw = speech_data.get("speech_score")
            if speech_raw is None:
                word_count = speech_data.get("word_count")
                if word_count is not None:
                    speech_raw = robust_z(word_count.astype(np.float64))
            
            speech_hop_arr = speech_data.get("hop_seconds")
            if speech_raw is not None and speech_hop_arr is not None and len(speech_hop_arr) > 0:
                speech_scores = resample_series(
                    speech_raw.astype(np.float64),
                    src_hop_s=float(speech_hop_arr[0]),
                    target_hop_s=hop_s,
                    target_len=len(audio_scores),
                    fill_value=0.0,
                )
                speech_used = True
        except Exception:
            pass
    
    _report_progress(on_progress, 0.50, "Loading reaction audio features")
    
    # Load reaction audio (optional)
    reaction_scores = np.zeros_like(audio_scores)
    reaction_used = False
    if proj.reaction_audio_features_path.exists():
        try:
            reaction_data = load_npz(proj.reaction_audio_features_path)
            reaction_raw = reaction_data.get("scores")
            reaction_hop_arr = reaction_data.get("hop_seconds")
            if reaction_raw is not None and reaction_hop_arr is not None and len(reaction_hop_arr) > 0:
                reaction_scores = resample_series(
                    reaction_raw.astype(np.float64),
                    src_hop_s=float(reaction_hop_arr[0]),
                    target_hop_s=hop_s,
                    target_len=len(audio_scores),
                    fill_value=0.0,
                )
                reaction_used = True
        except Exception:
            pass
    
    _report_progress(on_progress, 0.55, "Loading VAD speech fraction")
    
    # Load VAD speech fraction for gating (Step 4)
    speech_fraction = None
    vad_used = False
    vad_path = proj.analysis_dir / "audio_vad.npz"
    if vad_path.exists():
        try:
            vad_data = load_npz(vad_path)
            vad_raw = vad_data.get("speech_fraction")
            if vad_raw is None:
                vad_raw = vad_data.get("scores") or vad_data.get("values")
            vad_hop_arr = vad_data.get("hop_seconds")
            if vad_raw is not None:
                vad_hop_s = float(vad_hop_arr[0]) if vad_hop_arr is not None and len(vad_hop_arr) > 0 else hop_s
                speech_fraction = resample_series(
                    vad_raw.astype(np.float64),
                    src_hop_s=vad_hop_s,
                    target_hop_s=hop_s,
                    target_len=len(audio_scores),
                    fill_value=0.0,
                )
                # Clamp to [0, 1]
                speech_fraction = np.clip(speech_fraction, 0.0, 1.0)
                vad_used = True
        except Exception:
            pass
    
    # Load diarization-derived signals: turn_rate and overlap (Step 4)
    turn_rate_scores = np.zeros_like(audio_scores)
    overlap_scores = np.zeros_like(audio_scores)
    turn_rate_used = False
    overlap_used = False
    diar_path = proj.analysis_dir / "diarization.json"
    if diar_path.exists():
        try:
            import json as _json
            diar = _json.loads(diar_path.read_text(encoding="utf-8"))
            
            # Turn rate timeline
            tr = diar.get("turn_rate") or {}
            if tr and "times" in tr and "turns_per_minute" in tr:
                tr_times = np.asarray(tr["times"], dtype=np.float64)
                tr_values = np.asarray(tr["turns_per_minute"], dtype=np.float64)
                if len(tr_times) > 0 and len(tr_values) > 0:
                    tr_hop_s = float(np.median(np.diff(tr_times))) if len(tr_times) > 1 else 1.0
                    turn_rate_scores = resample_series(
                        tr_values,
                        src_hop_s=tr_hop_s,
                        target_hop_s=hop_s,
                        target_len=len(audio_scores),
                        fill_value=0.0,
                    )
                    turn_rate_scores = robust_z(turn_rate_scores)
                    turn_rate_used = True
            
            # Overlap fraction timeline
            ov = diar.get("overlap_fraction") or {}
            if ov and "times" in ov and "fraction" in ov:
                ov_times = np.asarray(ov["times"], dtype=np.float64)
                ov_values = np.asarray(ov["fraction"], dtype=np.float64)
                if len(ov_times) > 0 and len(ov_values) > 0:
                    ov_hop_s = float(np.median(np.diff(ov_times))) if len(ov_times) > 1 else 1.0
                    overlap_scores = resample_series(
                        ov_values,
                        src_hop_s=ov_hop_s,
                        target_hop_s=hop_s,
                        target_len=len(audio_scores),
                        fill_value=0.0,
                    )
                    overlap_scores = robust_z(overlap_scores)
                    overlap_used = True
        except Exception:
            pass
    
    if on_progress:
        on_progress(0.60)
    
    # Get weights (including new Step 4 signals)
    weights = highlights_cfg.get("weights", {})
    w_audio = float(weights.get("audio", 0.45))
    w_motion = float(weights.get("motion", 0.15)) if motion_used else 0.0
    w_chat = float(weights.get("chat", 0.20)) if chat_used else 0.0
    w_audio_events = float(weights.get("audio_events", 0.20)) if audio_events_used else 0.0
    w_speech = float(weights.get("speech", 0.15)) if speech_used else 0.0
    w_reaction = float(weights.get("reaction", 0.15)) if reaction_used else 0.0
    # Diarization-derived weights (small defaults, only if signal present)
    w_turn_rate = float(weights.get("turn_rate", 0.05)) if turn_rate_used else 0.0
    w_overlap = float(weights.get("overlap", 0.03)) if overlap_used else 0.0
    
    # Normalize weights
    w_total = w_audio + w_motion + w_chat + w_audio_events + w_speech + w_reaction + w_turn_rate + w_overlap
    if w_total > 0 and abs(w_total - 1.0) > 0.001:
        w_audio /= w_total
        w_motion /= w_total
        w_chat /= w_total
        w_audio_events /= w_total
        w_speech /= w_total
        w_reaction /= w_total
        w_turn_rate /= w_total
        w_overlap /= w_total
    
    # Signal processing config
    use_relu = bool(highlights_cfg.get("relu_zscores", True))
    signal_clip_z = float(highlights_cfg.get("signal_clip_z", 6.0))
    signal_softclip = str(highlights_cfg.get("signal_softclip", "clip")).lower().strip()
    use_motion_gating = bool(highlights_cfg.get("motion_gating", True))
    motion_gate_threshold_z = float(highlights_cfg.get("motion_gate_threshold_z", 0.0))
    motion_gate_scale_z = float(highlights_cfg.get("motion_gate_scale_z", 2.0))
    use_audio_events_gating = bool(highlights_cfg.get("audio_events_gating", True))
    audio_events_gate_threshold_z = float(highlights_cfg.get("audio_events_gate_threshold_z", 0.0))
    audio_events_gate_scale_z = float(highlights_cfg.get("audio_events_gate_scale_z", 2.0))
    
    motion_gate = np.ones_like(audio_scores)
    audio_events_gate = np.ones_like(audio_scores)
    
    _report_progress(on_progress, 0.60, "Fusing signals with weights")
    
    # Compute weighted terms
    if use_relu:
        audio_scores_pos = _pos_transform(audio_scores, clip_z=signal_clip_z, mode=signal_softclip)
        motion_resampled_pos = _pos_transform(motion_resampled, clip_z=signal_clip_z, mode=signal_softclip)
        chat_scores_pos = _pos_transform(chat_scores, clip_z=signal_clip_z, mode=signal_softclip)
        audio_events_scores_pos = _pos_transform(audio_events_scores, clip_z=signal_clip_z, mode=signal_softclip)
        speech_scores_pos = _pos_transform(speech_scores, clip_z=signal_clip_z, mode=signal_softclip)
        reaction_scores_pos = _pos_transform(reaction_scores, clip_z=signal_clip_z, mode=signal_softclip)
        
        if use_motion_gating and w_motion > 0:
            motion_gate = _linear_gate(
                np.maximum(audio_scores_pos, chat_scores_pos),
                threshold=motion_gate_threshold_z,
                scale=motion_gate_scale_z,
            )
            term_motion = w_motion * motion_resampled_pos * motion_gate
        else:
            term_motion = w_motion * motion_resampled_pos
        
        if use_audio_events_gating and w_audio_events > 0:
            audio_events_gate = _linear_gate(
                audio_scores_pos,
                threshold=audio_events_gate_threshold_z,
                scale=audio_events_gate_scale_z,
            )
            term_audio_events = w_audio_events * audio_events_scores_pos * audio_events_gate
        else:
            term_audio_events = w_audio_events * audio_events_scores_pos
        
        term_audio = w_audio * audio_scores_pos
        term_chat = w_chat * chat_scores_pos
        term_speech = w_speech * speech_scores_pos
        term_reaction = w_reaction * reaction_scores_pos
        term_turn_rate = w_turn_rate * _pos_transform(turn_rate_scores, clip_z=signal_clip_z, mode=signal_softclip)
        term_overlap = w_overlap * _pos_transform(overlap_scores, clip_z=signal_clip_z, mode=signal_softclip)
    else:
        if use_motion_gating and w_motion > 0:
            motion_gate = _linear_gate(
                np.maximum(audio_scores, chat_scores),
                threshold=motion_gate_threshold_z,
                scale=motion_gate_scale_z,
            )
            term_motion = w_motion * motion_resampled * motion_gate
        else:
            term_motion = w_motion * motion_resampled
        
        if use_audio_events_gating and w_audio_events > 0:
            audio_events_gate = _linear_gate(
                audio_scores,
                threshold=audio_events_gate_threshold_z,
                scale=audio_events_gate_scale_z,
            )
            term_audio_events = w_audio_events * audio_events_scores * audio_events_gate
        else:
            term_audio_events = w_audio_events * audio_events_scores
        
        term_audio = w_audio * audio_scores
        term_chat = w_chat * chat_scores
        term_speech = w_speech * speech_scores
        term_reaction = w_reaction * reaction_scores
        term_turn_rate = w_turn_rate * turn_rate_scores
        term_overlap = w_overlap * overlap_scores
    
    combined_raw = (
        term_audio + term_motion + term_chat + 
        term_audio_events + term_speech + term_reaction +
        term_turn_rate + term_overlap
    )
    
    _report_progress(on_progress, 0.70, "Applying speech gating")
    
    # Speech gating (Step 4): strongly prefer moments with speech
    # This prevents picking "loud gameplay" moments where nobody is talking
    speech_gated = combined_raw.copy()
    require_speech = bool(highlights_cfg.get("require_speech", True))
    min_peak_speech_fraction = float(highlights_cfg.get("min_peak_speech_fraction", 0.15))
    speech_gate_min_fraction = float(highlights_cfg.get("speech_gate_min_fraction", 0.05))
    speech_gate_floor = float(highlights_cfg.get("speech_gate_floor", 0.08))
    speech_gate_power = float(highlights_cfg.get("speech_gate_power", 1.0))
    
    if speech_fraction is not None:
        # Compute gate: scales score by speech presence
        gate = np.clip((speech_fraction - speech_gate_min_fraction) / max(1e-9, 1.0 - speech_gate_min_fraction), 0.0, 1.0)
        gate = np.power(gate, speech_gate_power)
        # Apply gate with floor (keep some score even when no speech)
        speech_gated = combined_raw * (speech_gate_floor + (1.0 - speech_gate_floor) * gate)
    
    combined_raw_gated = speech_gated
    
    # Smooth and z-score (use speech-gated signal)
    smooth_s = float(highlights_cfg.get("smooth_seconds", max(1.0, 2.0 * hop_s)))
    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    combined_smoothed = moving_average(combined_raw_gated, smooth_frames) if len(combined_raw_gated) > 0 else combined_raw_gated
    combined_scores = robust_z(combined_smoothed) if len(combined_smoothed) > 0 else combined_smoothed
    
    # Apply hard speech requirement for peak selection (if enabled)
    scores_for_peaks = combined_scores.copy()
    if require_speech and speech_fraction is not None:
        # Mark non-speech frames as invalid for peak picking
        scores_for_peaks = np.where(speech_fraction >= min_peak_speech_fraction, scores_for_peaks, -np.inf)
    
    skip_start_frames = int(round(float(highlights_cfg.get("skip_start_seconds", 10.0)) / hop_s))
    if skip_start_frames > 0 and skip_start_frames < len(scores_for_peaks):
        scores_for_peaks[:skip_start_frames] = -np.inf
    
    min_gap_frames = max(1, int(round(float(highlights_cfg.get("min_gap_seconds", 15.0)) / hop_s)))
    min_peak_score = float(highlights_cfg.get("min_peak_score", 0.0))
    peak_idxs = pick_top_peaks(
        scores_for_peaks,
        top_k=int(highlights_cfg.get("top", 20)),
        min_gap_frames=min_gap_frames,
        min_score=min_peak_score,
    )
    
    _report_progress(on_progress, 0.85, "Saving highlights_features.npz")
    
    # Save features NPZ with peak indices
    save_npz(
        proj.highlights_features_path,
        combined_raw=combined_raw,
        combined_smoothed=combined_smoothed,
        combined_scores=combined_scores,
        scores_for_peaks=scores_for_peaks,
        peak_indices=np.array(peak_idxs, dtype=np.int64),
        audio_scores=audio_scores,
        motion_scores=motion_resampled,
        chat_scores=chat_scores,
        audio_events_scores=audio_events_scores,
        speech_scores=speech_scores,
        reaction_scores=reaction_scores,
        turn_rate_scores=turn_rate_scores,
        overlap_scores=overlap_scores,
        term_audio=term_audio,
        term_motion=term_motion,
        term_chat=term_chat,
        term_audio_events=term_audio_events,
        term_speech=term_speech,
        term_reaction=term_reaction,
        term_turn_rate=term_turn_rate,
        term_overlap=term_overlap,
        motion_gate=motion_gate,
        audio_events_gate=audio_events_gate,
        # Step 4: speech gating data
        speech_fraction=speech_fraction if speech_fraction is not None else np.ones_like(audio_scores),
        combined_raw_ungated=combined_raw,
        hop_seconds=np.array([hop_s], dtype=np.float64),
    )
    
    _report_progress(on_progress, 1.0, "Done")
    
    return {
        "peak_count": len(peak_idxs),
        "hop_s": hop_s,
        "weights": {
            "audio": w_audio,
            "motion": w_motion,
            "chat": w_chat,
            "audio_events": w_audio_events,
            "speech": w_speech,
            "reaction": w_reaction,
            "turn_rate": w_turn_rate,
            "overlap": w_overlap,
        },
        "signals_used": {
            "motion": motion_used,
            "chat": chat_used,
            "audio_events": audio_events_used,
            "speech": speech_used,
            "reaction": reaction_used,
            "vad": vad_used,
            "turn_rate": turn_rate_used,
            "overlap": overlap_used,
        },
        "speech_gating": {
            "enabled": speech_fraction is not None,
            "require_speech": require_speech,
            "min_peak_speech_fraction": min_peak_speech_fraction,
            "speech_gate_floor": speech_gate_floor,
        },
    }


def compute_highlights_candidates(
    proj: Project,
    *,
    highlights_cfg: Dict[str, Any],
    scenes_cfg: Optional[Dict[str, Any]] = None,
    llm_complete: Optional[Callable[[str], str]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Shape peaks into clip candidates (SHAPING phase).
    
    This function:
    1. Loads peaks from highlights_features.npz
    2. Uses boundary graph (if available) for optimal clip edges
    3. Applies overlap filtering
    4. Optionally runs LLM semantic scoring
    5. Saves highlights.json
    
    This can be re-run quickly when boundary information improves.
    
    Args:
        proj: Project instance
        highlights_cfg: Highlights config with clip settings
        scenes_cfg: Scenes config (for snap_window)
        llm_complete: Optional LLM function for semantic scoring
        on_progress: Progress callback
        
    Returns:
        Dict with candidates
    """
    import json
    from datetime import datetime, timezone
    
    if scenes_cfg is None:
        scenes_cfg = {}
    
    proj_data = get_project_data(proj)
    
    _report_progress(on_progress, 0.05, "Loading highlights_features.npz")
    
    # Load scores and peaks
    if not proj.highlights_features_path.exists():
        raise ValueError("highlights_features.npz not found - run highlights_scores task first")
    
    features = load_npz(proj.highlights_features_path)
    combined_smoothed = features.get("combined_smoothed")
    combined_scores = features.get("combined_scores")
    peak_idxs = features.get("peak_indices")
    hop_arr = features.get("hop_seconds")
    
    if combined_smoothed is None or hop_arr is None:
        raise ValueError("highlights_features.npz missing required arrays")
    
    hop_s = float(hop_arr[0]) if len(hop_arr) > 0 else 0.5
    
    # Handle legacy format that may be missing peak_indices
    if peak_idxs is None:
        _highlight_logger.warning(
            "[highlights_candidates] Legacy highlights_features.npz detected (missing peak_indices). "
            "Computing peaks from combined_scores. Re-run highlights_scores for best results."
        )
        # Compute peaks on-the-fly from combined_scores
        if combined_scores is None:
            combined_scores = combined_smoothed  # fallback
        
        # Simple peak finding: local maxima above threshold
        top_n = int(highlights_cfg.get("top", 20))
        min_gap_s = float(highlights_cfg.get("min_gap_seconds", 15.0))
        skip_start_s = float(highlights_cfg.get("skip_start_seconds", 10.0))
        min_gap_idx = max(1, int(min_gap_s / hop_s))
        skip_start_idx = int(skip_start_s / hop_s)
        
        # Zero out the skip-start region
        scores_for_peaks = combined_scores.copy()
        if skip_start_idx > 0:
            scores_for_peaks[:skip_start_idx] = -np.inf
        
        peak_idxs = pick_top_peaks(
            scores_for_peaks,
            top_k=top_n,
            min_gap_frames=min_gap_idx,
            min_score=0.0,
        )
    else:
        peak_idxs = peak_idxs.astype(np.int64).tolist()
    
    # Load term arrays for breakdown
    term_audio = features.get("term_audio", np.zeros_like(combined_smoothed))
    term_motion = features.get("term_motion", np.zeros_like(combined_smoothed))
    term_chat = features.get("term_chat", np.zeros_like(combined_smoothed))
    term_audio_events = features.get("term_audio_events", np.zeros_like(combined_smoothed))
    term_speech = features.get("term_speech", np.zeros_like(combined_smoothed))
    term_reaction = features.get("term_reaction", np.zeros_like(combined_smoothed))
    
    # Load raw signals for debugging
    audio_scores = features.get("audio_scores", np.zeros_like(combined_smoothed))
    motion_scores = features.get("motion_scores", np.zeros_like(combined_smoothed))
    chat_scores = features.get("chat_scores", np.zeros_like(combined_smoothed))
    audio_events_scores = features.get("audio_events_scores", np.zeros_like(combined_smoothed))
    speech_scores = features.get("speech_scores", np.zeros_like(combined_smoothed))
    reaction_scores = features.get("reaction_scores", np.zeros_like(combined_smoothed))
    
    _report_progress(on_progress, 0.15, "Loading scene cuts")
    
    # Load scene cuts
    scene_cuts = []
    scenes_enabled = bool(scenes_cfg.get("enabled", True))
    snap_window_s = float(scenes_cfg.get("snap_window_seconds", 0.0))
    if scenes_enabled and proj.scenes_path.exists():
        try:
            scene_payload = json.loads(proj.scenes_path.read_text(encoding="utf-8"))
            scene_cuts = scene_payload.get("cuts_seconds", []) or scene_payload.get("cuts", [])
        except Exception:
            pass
    
    _report_progress(on_progress, 0.20, "Loading boundary graph")
    
    # Load boundary graph (key for quality)
    # Note: boundary_graph may not exist during pre-download analysis (expected)
    # It will be created after full analysis when video-based silence detection runs
    boundary_graph = None
    try:
        from .analysis_boundaries import load_boundary_graph
        boundary_graph = load_boundary_graph(proj)
        if boundary_graph is not None:
            _highlight_logger.info(f"[Highlights] Loaded boundary graph: {len(boundary_graph.start_boundaries)} starts, {len(boundary_graph.end_boundaries)} ends")
        else:
            # Debug level since this is expected during pre-download
            _highlight_logger.debug("[Highlights] Boundary graph not available yet (will use valley fallback)")
    except Exception as e:
        _highlight_logger.error(f"[Highlights] Failed to load boundary graph: {e}")
    
    _report_progress(on_progress, 0.25, "Configuring clip parameters")
    
    # Clip config
    clip_cfg_dict = highlights_cfg.get("clip", {})
    clip_cfg = ClipConfig(
        min_seconds=float(clip_cfg_dict.get("min_seconds", 12.0)),
        max_seconds=float(clip_cfg_dict.get("max_seconds", 60.0)),
        min_pre_seconds=float(clip_cfg_dict.get("min_pre_seconds", 2.0)),
        max_pre_seconds=float(clip_cfg_dict.get("max_pre_seconds", 12.0)),
        min_post_seconds=float(clip_cfg_dict.get("min_post_seconds", 4.0)),
        max_post_seconds=float(clip_cfg_dict.get("max_post_seconds", 28.0)),
    )
    
    duration_s = float(proj_data.get("video", {}).get("duration_seconds", hop_s * len(audio_scores)))
    
    # Overlap filtering config
    max_overlap_ratio = float(highlights_cfg.get("max_overlap_ratio", 0.0))
    max_overlap_seconds = float(highlights_cfg.get("max_overlap_seconds", 0.0))
    overlap_denom = str(highlights_cfg.get("overlap_denominator", "shorter"))
    
    _report_progress(on_progress, 0.30, "Shaping clip boundaries")
    
    # Build raw candidates
    raw_candidates: List[Dict[str, Any]] = []
    
    for rank, idx in enumerate(peak_idxs, start=1):
        if idx < 0 or idx >= len(combined_smoothed):
            continue
            
        bounds = shape_clip_bounds(
            peak_idx=idx,
            scores=combined_smoothed,
            hop_s=hop_s,
            duration_s=duration_s,
            clip_cfg=clip_cfg,
            scene_cuts=scene_cuts,
            snap_window_s=snap_window_s,
            boundary_graph=boundary_graph,
        )
        start_s = bounds["start_s"]
        end_s = bounds["end_s"]
        if end_s - start_s < max(1.0, clip_cfg.min_seconds * 0.5):
            continue
        
        # Check signals_used (based on non-zero term arrays)
        chat_used = bool(np.any(term_chat != 0))
        audio_events_used = bool(np.any(term_audio_events != 0))
        speech_used = bool(np.any(term_speech != 0))
        reaction_used = bool(np.any(term_reaction != 0))
        
        raw_candidates.append({
            "rank": rank,
            "peak_time_s": float(idx * hop_s),
            "start_s": float(start_s),
            "end_s": float(end_s),
            "score": float(combined_scores[idx]) if combined_scores is not None and idx < len(combined_scores) else 0.0,
            "used_boundary_graph": bounds.get("used_boundary_graph", False),
            "breakdown": {
                "audio": float(term_audio[idx]),
                "motion": float(term_motion[idx]),
                "chat": float(term_chat[idx]) if chat_used else 0.0,
                "audio_events": float(term_audio_events[idx]) if audio_events_used else 0.0,
                "speech": float(term_speech[idx]) if speech_used else 0.0,
                "reaction": float(term_reaction[idx]) if reaction_used else 0.0,
            },
            "raw_signals": {
                "audio": float(audio_scores[idx]),
                "motion": float(motion_scores[idx]),
                "chat": float(chat_scores[idx]) if chat_used else 0.0,
                "audio_events": float(audio_events_scores[idx]) if audio_events_used else 0.0,
                "speech": float(speech_scores[idx]) if speech_used else 0.0,
                "reaction": float(reaction_scores[idx]) if reaction_used else 0.0,
            },
        })
    
    _report_progress(on_progress, 0.50, "Filtering overlapping candidates")
    
    # Non-overlap filtering
    raw_candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    candidates: List[Dict[str, Any]] = []
    
    for cand in raw_candidates:
        if _check_overlap(
            cand["start_s"],
            cand["end_s"],
            candidates,
            max_overlap_ratio=max_overlap_ratio,
            max_overlap_seconds=max_overlap_seconds,
            denom=overlap_denom,
        ):
            cand["rank"] = len(candidates) + 1
            candidates.append(cand)
    
    _report_progress(on_progress, 0.65, f"Filtered to {len(candidates)} candidates")
    
    # LLM Semantic Scoring (optional, controlled by config)
    llm_semantic_used = False
    llm_semantic_scores: Dict[int, Dict[str, Any]] = {}
    llm_semantic_enabled = bool(highlights_cfg.get("llm_semantic_enabled", True))
    
    if not llm_semantic_enabled:
        _highlight_logger.info("[highlights_candidates] LLM semantic scoring disabled in config (llm_semantic_enabled=false)")
    elif llm_complete is None:
        _highlight_logger.info("[highlights_candidates] LLM semantic scoring skipped (no llm_complete function provided)")
    elif not candidates:
        _highlight_logger.info("[highlights_candidates] LLM semantic scoring skipped (no candidates)")
    
    if llm_semantic_enabled and llm_complete is not None and candidates:
        _report_progress(on_progress, 0.70, "Starting LLM semantic scoring")
        
        try:
            content_type = str(highlights_cfg.get("content_type", "gaming"))
            llm_semantic_scores = compute_llm_semantic_scores(
                candidates=candidates,
                proj=proj,
                llm_complete=llm_complete,
                max_candidates=int(highlights_cfg.get("llm_max_candidates", 15)),
                content_type=content_type,
            )
            
            if llm_semantic_scores:
                llm_semantic_used = True
                semantic_weight = float(highlights_cfg.get("llm_semantic_weight", 0.3))
                
                for cand in candidates:
                    rank = cand.get("rank", 0)
                    if rank in llm_semantic_scores:
                        llm_data = llm_semantic_scores[rank]
                        semantic_score = llm_data.get("semantic_score", 0.5)
                        signal_score = cand.get("score", 0.0)
                        semantic_z = (semantic_score - 0.5) * 4.0
                        blended_score = (1.0 - semantic_weight) * signal_score + semantic_weight * semantic_z
                        
                        cand["score_signal"] = signal_score
                        cand["score_semantic"] = semantic_score
                        cand["score"] = blended_score
                        cand["llm_reason"] = llm_data.get("reason", "")
                        cand["llm_quote"] = llm_data.get("best_quote", "")
                
                candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
                for i, cand in enumerate(candidates, start=1):
                    cand["rank"] = i
        except Exception as e:
            _highlight_logger.warning(f"LLM semantic scoring failed: {e}")
    
    _report_progress(on_progress, 0.85, "Checking LLM quality filter")
    
    # LLM Quality Filter (optional, more aggressive than semantic scoring)
    llm_filter_used = False
    llm_filter_stats: Dict[str, Any] = {}
    
    llm_filter_enabled = bool(highlights_cfg.get("llm_filter_enabled", True))
    
    if not llm_filter_enabled:
        _highlight_logger.info("[highlights_candidates] LLM quality filter disabled in config (llm_filter_enabled=false)")
    elif llm_complete is None:
        _highlight_logger.info("[highlights_candidates] LLM quality filter skipped (no llm_complete function provided)")
    elif not candidates:
        _highlight_logger.info("[highlights_candidates] LLM quality filter skipped (no candidates)")
    
    if llm_filter_enabled and llm_complete is not None and candidates:
        _report_progress(on_progress, 0.87, "Running LLM quality filter")
        
        try:
            min_quality = int(highlights_cfg.get("llm_filter_min_quality", 5))
            max_keep = highlights_cfg.get("llm_filter_max_keep")  # None = no limit
            if max_keep is not None:
                max_keep = int(max_keep)
            content_type = str(highlights_cfg.get("content_type", "gaming"))
            
            candidates, llm_filter_stats = compute_llm_filter(
                candidates=candidates,
                proj=proj,
                llm_complete=llm_complete,
                min_quality_score=min_quality,
                max_keep=max_keep,
                content_type=content_type,
            )
            
            if not llm_filter_stats.get("skipped", False):
                llm_filter_used = True
                _highlight_logger.info(
                    f"[highlights_candidates] LLM filter: kept {llm_filter_stats.get('kept', 0)}/{llm_filter_stats.get('total_input', 0)} candidates"
                )
        except Exception as e:
            _highlight_logger.warning(f"LLM quality filter failed: {e}")
    elif llm_complete is None:
        _highlight_logger.info("[highlights_candidates] LLM quality filter skipped (no llm_complete function)")
    elif not llm_filter_enabled:
        _highlight_logger.info("[highlights_candidates] LLM quality filter disabled in config")
    
    _report_progress(on_progress, 0.92, "Saving highlights.json")
    
    # Build and save payload
    payload: Dict[str, Any] = {
        "method": "dag_split_shaping_v1",
        "signals_used": {
            "boundary_graph": boundary_graph is not None,
            "llm_semantic": llm_semantic_used,
            "llm_filter": llm_filter_used,
        },
        "filtering": {
            "max_overlap_ratio": max_overlap_ratio,
            "max_overlap_seconds": max_overlap_seconds,
            "overlap_denominator": overlap_denom,
            "candidates_before_filter": len(raw_candidates),
            "candidates_after_filter": len(candidates),
        },
        "llm_filter_stats": llm_filter_stats if llm_filter_used else None,
        "candidates": candidates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Save highlights.json
    highlights_json_path = proj.analysis_dir / "highlights.json"
    highlights_json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    # Update project.json
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["highlights"] = {
            **payload,
            "features_npz": str(proj.highlights_features_path.relative_to(proj.project_dir)),
        }
    
    update_project(proj, _upd)
    
    _report_progress(on_progress, 1.0, "Done")
    
    return payload
