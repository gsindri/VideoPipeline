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
    
    # Get ranked candidates for start boundaries
    start_candidates = find_start_boundary_candidates(
        boundary_graph,
        target_s=peak_time - clip_cfg.min_pre_seconds,
        max_before_s=clip_cfg.max_pre_seconds - clip_cfg.min_pre_seconds,
        max_after_s=clip_cfg.min_pre_seconds,
        prefer_before=True,
    )
    
    # Get ranked candidates for end boundaries
    end_candidates = find_end_boundary_candidates(
        boundary_graph,
        target_s=peak_time + clip_cfg.min_post_seconds,
        max_before_s=clip_cfg.min_post_seconds,
        max_after_s=clip_cfg.max_post_seconds - clip_cfg.min_post_seconds,
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


def _build_llm_highlight_prompt(
    candidates: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    chat_context: Optional[str] = None,
) -> str:
    """Build prompt for LLM to semantically score highlight candidates."""
    import bisect
    
    # Pre-sort segments by start time for faster lookup
    sorted_segments = sorted(transcript_segments, key=lambda s: s.get("start", 0.0))
    seg_starts = [s.get("start", 0.0) for s in sorted_segments]
    
    # Get transcript text for each candidate's time window
    candidate_data = []
    for cand in candidates:
        start_s = cand.get("start_s", 0.0)
        end_s = cand.get("end_s", 0.0)
        
        # Use binary search to find relevant segments (much faster for large transcripts)
        # Find first segment that could overlap (starts before end_s)
        start_idx = bisect.bisect_left(seg_starts, start_s)
        # Go back a bit to catch segments that started before but extend into our window
        start_idx = max(0, start_idx - 5)
        
        relevant_text = []
        for i in range(start_idx, len(sorted_segments)):
            seg = sorted_segments[i]
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            
            # If segment starts after our window ends, we're done
            if seg_start > end_s:
                break
            
            # Check for overlap
            if seg_end >= start_s and seg_start <= end_s:
                relevant_text.append(seg.get("text", "").strip())
        
        candidate_data.append({
            "rank": cand.get("rank", 0),
            "start_s": round(start_s, 1),
            "end_s": round(end_s, 1),
            "duration_s": round(end_s - start_s, 1),
            "signal_score": round(cand.get("score", 0.0), 2),
            "transcript": " ".join(relevant_text)[:500],  # Limit length
            "breakdown": {
                k: round(v, 2) if isinstance(v, (int, float)) else v 
                for k, v in cand.get("breakdown", {}).items()
            },
        })
    
    prompt_data = {
        "task": "Score highlight candidates by semantic/entertainment value",
        "candidates": candidate_data,
    }
    
    if chat_context:
        prompt_data["chat_context"] = chat_context[:300]
    
    prompt = f"""Analyze these highlight candidates and score them by entertainment value.

Input:
{json.dumps(prompt_data, indent=2)}

For each candidate, evaluate the transcript content and assign a semantic_score (0.0-1.0).
Higher scores for: exciting reactions, funny moments, impressive plays, quotable phrases.
Lower scores for: mundane commentary, incomplete thoughts, boring content.

Output JSON array with one object per candidate:
[
  {{"rank": 1, "semantic_score": 0.8, "reason": "brief explanation", "best_quote": "short quotable phrase"}},
  ...
]

Respond with ONLY the JSON array:"""
    
    return prompt


def _parse_llm_highlight_scores(
    response_text: str,
    num_candidates: int,
) -> Dict[int, Dict[str, Any]]:
    """Parse LLM response into semantic scores by rank.
    
    Returns: Dict mapping rank -> {"semantic_score": float, "reason": str, "best_quote": str}
    """
    result = {}
    
    try:
        # Try to parse as JSON
        text = response_text.strip()
        
        # Handle markdown code blocks
        if "```" in text:
            match = _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, _re.DOTALL)
            if match:
                text = match.group(1).strip()
        
        # Try to find JSON array
        if not text.startswith("["):
            match = _re.search(r"\[.*\]", text, _re.DOTALL)
            if match:
                text = match.group(0)
        
        scores = json.loads(text)
        
        if isinstance(scores, list):
            for item in scores:
                if isinstance(item, dict) and "rank" in item:
                    rank = int(item["rank"])
                    result[rank] = {
                        "semantic_score": float(item.get("semantic_score", 0.5)),
                        "reason": str(item.get("reason", "")),
                        "best_quote": str(item.get("best_quote", "")),
                    }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        _highlight_logger.warning(f"Failed to parse LLM highlight scores: {e}")
    
    return result


def compute_llm_semantic_scores(
    candidates: List[Dict[str, Any]],
    proj: "Project",
    llm_complete: Callable[[str], str],
    *,
    max_candidates: int = 20,
) -> Dict[int, Dict[str, Any]]:
    """Use LLM to compute semantic scores for highlight candidates.
    
    Args:
        candidates: List of candidate dicts with start_s, end_s, score, breakdown
        proj: Project instance for accessing transcript
        llm_complete: Function that takes prompt string and returns response string
        max_candidates: Maximum candidates to send to LLM (for token limits)
        
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
    
    # Build prompt
    prompt = _build_llm_highlight_prompt(candidates_to_score, transcript_segments)
    
    try:
        prompt_len = len(prompt)
        _highlight_logger.info(f"Requesting LLM semantic scores for {len(candidates_to_score)} candidates (prompt: {prompt_len} chars)")
        
        import time as _t
        start = _t.time()
        response = llm_complete(prompt)
        elapsed = _t.time() - start
        
        _highlight_logger.info(f"LLM responded in {elapsed:.1f}s ({len(response)} chars)")
        scores = _parse_llm_highlight_scores(response, len(candidates_to_score))
        _highlight_logger.info(f"LLM returned semantic scores for {len(scores)} candidates")
        return scores
    except Exception as e:
        _highlight_logger.warning(f"LLM semantic scoring failed: {e}")
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
            on_progress(0.05)
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
            on_progress(0.25)
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
            on_progress(0.45)
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
            on_progress(0.6)
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
        on_progress(0.20)  # Done loading dependencies

    audio_data = load_npz(proj.audio_features_path)
    audio_scores = audio_data.get("scores")
    if audio_scores is None:
        raise ValueError("audio_features.npz missing scores")
    audio_scores = audio_scores.astype(np.float64)

    if on_progress:
        on_progress(0.25)

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
                    on_progress(0.65)
                
                # Create a sub-progress callback to scale audio events progress (0-0.85) into (0.65-0.85)
                def audio_events_progress(p: float) -> None:
                    if on_progress:
                        # Scale: 0.0 -> 0.65, 0.85 -> 0.82
                        scaled = 0.65 + p * 0.20
                        on_progress(min(0.85, scaled))
                
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
        on_progress(0.50)  # Done with signal fusion

    min_gap_frames = max(1, int(round(float(highlights_cfg.get("min_gap_seconds", 15.0)) / hop_s)))
    min_peak_score = float(highlights_cfg.get("min_peak_score", 0.0))
    peak_idxs = pick_top_peaks(
        scores_for_peaks,
        top_k=int(highlights_cfg.get("top", 20)),
        min_gap_frames=min_gap_frames,
        min_score=min_peak_score,
    )

    if on_progress:
        on_progress(0.55)  # Done picking peaks

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
        on_progress(0.70)  # Done shaping candidates

    # LLM Semantic Scoring (optional enhancement)
    llm_semantic_used = False
    llm_semantic_scores: Dict[int, Dict[str, Any]] = {}
    
    if llm_complete is not None and candidates:
        if on_progress:
            on_progress(0.75)  # Starting LLM scoring
        
        try:
            _highlight_logger.info(f"[highlights] Starting LLM semantic scoring for {len(candidates)} candidates...")
            llm_semantic_scores = compute_llm_semantic_scores(
                candidates=candidates,
                proj=proj,
                llm_complete=llm_complete,
                max_candidates=int(highlights_cfg.get("llm_max_candidates", 15)),
            )
            
            if on_progress:
                on_progress(0.90)  # LLM scoring done
            
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
