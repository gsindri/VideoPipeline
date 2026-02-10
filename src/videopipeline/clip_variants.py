"""Clip variant generation for context-aware clip shaping.

Generates multiple clip variants per highlight candidate using intelligent
boundary selection based on silence, sentences, scenes, chat activity,
and semantic chapters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .analysis_boundaries import (
    BoundaryConfig,
    BoundaryGraph,
    compute_boundary_graph,
    find_best_end_boundary,
    find_best_start_boundary,
    load_boundary_graph,
)
from .analysis_chat_boundaries import find_burst_in_range, find_nearest_valley, load_chat_boundaries
from .analysis_sentences import Sentence, get_sentences_in_range, load_sentences
from .project import Project, get_project_data, save_json, update_project, load_npz


@dataclass(frozen=True)
class VariantDurationConfig:
    """Duration constraints for a variant type."""
    min_s: float
    max_s: float


@dataclass(frozen=True)
class VariantGeneratorConfig:
    """Configuration for clip variant generation."""
    # Duration configs per variant type
    short: VariantDurationConfig = field(default_factory=lambda: VariantDurationConfig(16.0, 24.0))
    medium: VariantDurationConfig = field(default_factory=lambda: VariantDurationConfig(24.0, 40.0))
    long: VariantDurationConfig = field(default_factory=lambda: VariantDurationConfig(40.0, 75.0))

    # Pre/post peak constraints
    min_pre_s: float = 3.0
    max_pre_s: float = 20.0
    min_post_s: float = 5.0
    max_post_s: float = 30.0

    # Chat valley window for chat-centered variant
    chat_valley_window_s: float = 12.0

    # Chapter boundary constraints
    respect_chapters: bool = True  # If True, clamp variants to stay within chapters
    chapter_margin_s: float = 2.0  # Allow clips to extend slightly past chapter boundaries

    # ------------------------------------------------------------------
    # Step 5: "as-good-as-possible" shaping upgrades
    # ------------------------------------------------------------------
    # Reaction arc (voice excitement) variant
    enable_reaction_arc: bool = True
    reaction_arc_pre_search_s: float = 10.0
    reaction_arc_post_search_s: float = 14.0
    reaction_arc_min_s: float = 16.0
    reaction_arc_max_s: float = 65.0

    # Clean-cut variant using silence boundaries
    enable_clean_cut: bool = True
    clean_cut_min_s: float = 12.0
    clean_cut_max_s: float = 75.0

    # Attach per-variant signal summaries (helps director/UX/debug)
    enable_scores_summary: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VariantGeneratorConfig":
        """Build config from a profile dict."""
        d = d or {}
        short_d = d.get("short") or {}
        med_d = d.get("medium") or {}
        long_d = d.get("long") or {}
        ra_d = d.get("reaction_arc") or {}
        cc_d = d.get("clean_cut") or {}

        return cls(
            short=VariantDurationConfig(
                min_s=float(short_d.get("min_s", 16.0)),
                max_s=float(short_d.get("max_s", 24.0)),
            ),
            medium=VariantDurationConfig(
                min_s=float(med_d.get("min_s", 24.0)),
                max_s=float(med_d.get("max_s", 40.0)),
            ),
            long=VariantDurationConfig(
                min_s=float(long_d.get("min_s", 40.0)),
                max_s=float(long_d.get("max_s", 75.0)),
            ),
            min_pre_s=float(d.get("min_pre_s", 3.0)),
            max_pre_s=float(d.get("max_pre_s", 20.0)),
            min_post_s=float(d.get("min_post_s", 5.0)),
            max_post_s=float(d.get("max_post_s", 30.0)),
            chat_valley_window_s=float(d.get("chat_valley_window_s", 12.0)),
            respect_chapters=bool(d.get("respect_chapters", True)),
            chapter_margin_s=float(d.get("chapter_margin_s", 2.0)),
            enable_reaction_arc=bool(d.get("enable_reaction_arc", True)),
            reaction_arc_pre_search_s=float(ra_d.get("pre_search_s", d.get("reaction_arc_pre_search_s", 10.0))),
            reaction_arc_post_search_s=float(ra_d.get("post_search_s", d.get("reaction_arc_post_search_s", 14.0))),
            reaction_arc_min_s=float(ra_d.get("min_s", d.get("reaction_arc_min_s", 16.0))),
            reaction_arc_max_s=float(ra_d.get("max_s", d.get("reaction_arc_max_s", 65.0))),
            enable_clean_cut=bool(d.get("enable_clean_cut", True)),
            clean_cut_min_s=float(cc_d.get("min_s", d.get("clean_cut_min_s", 12.0))),
            clean_cut_max_s=float(cc_d.get("max_s", d.get("clean_cut_max_s", 75.0))),
            enable_scores_summary=bool(d.get("enable_scores_summary", True)),
        )


@dataclass
class ClipVariant:
    """A single clip variant for a candidate."""
    variant_id: str
    start_s: float
    end_s: float
    duration_s: float
    description: str = ""
    setup_text: str = ""  # First sentence(s) for context
    payoff_text: str = ""  # Most excited sentence
    scores_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
            "description": self.description,
            "setup_text": self.setup_text,
            "payoff_text": self.payoff_text,
            "scores_summary": self.scores_summary,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClipVariant":
        return cls(
            variant_id=str(d["variant_id"]),
            start_s=float(d["start_s"]),
            end_s=float(d["end_s"]),
            duration_s=float(d["duration_s"]),
            description=str(d.get("description", "")),
            setup_text=str(d.get("setup_text", "")),
            payoff_text=str(d.get("payoff_text", "")),
            scores_summary=d.get("scores_summary", {}),
        )


@dataclass
class CandidateVariants:
    """All variants for a single candidate."""
    candidate_rank: int
    candidate_peak_time_s: float
    variants: List[ClipVariant]
    candidate_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_rank": self.candidate_rank,
            "candidate_id": self.candidate_id,
            "candidate_peak_time_s": round(self.candidate_peak_time_s, 2),
            "variants": [v.to_dict() for v in self.variants],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateVariants":
        return cls(
            candidate_rank=int(d["candidate_rank"]),
            candidate_id=str(d.get("candidate_id", "")),
            candidate_peak_time_s=float(d["candidate_peak_time_s"]),
            variants=[ClipVariant.from_dict(v) for v in d.get("variants", [])],
        )

    def get_variant(self, variant_id: str) -> Optional[ClipVariant]:
        """Get a specific variant by ID."""
        for v in self.variants:
            if v.variant_id == variant_id:
                return v
        return None


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a range."""
    return max(min_val, min(max_val, value))


# ============================================================================
# Chapter Boundary Helpers
# ============================================================================

def _find_chapter_at_time(
    chapters: List[Tuple[float, float]],
    time_s: float,
) -> Optional[Tuple[float, float]]:
    """Find the chapter (start, end) containing a specific time."""
    for start, end in chapters:
        if start <= time_s < end:
            return (start, end)
    # If time is at video end, return last chapter
    if chapters and time_s >= chapters[-1][0]:
        return chapters[-1]
    return None


def _clamp_to_chapter(
    start_s: float,
    end_s: float,
    peak_time_s: float,
    chapters: List[Tuple[float, float]],
    margin_s: float = 2.0,
) -> Tuple[float, float]:
    """Clamp a clip to stay within its chapter boundaries.
    
    Uses the chapter containing the peak time as the reference.
    Allows a small margin to prevent awkward cuts right at boundaries.
    
    Args:
        start_s: Proposed clip start
        end_s: Proposed clip end
        peak_time_s: The highlight peak time (determines which chapter)
        chapters: List of (start, end) tuples for all chapters
        margin_s: Small margin to allow clips to extend past boundaries
        
    Returns:
        Tuple of (clamped_start, clamped_end)
    """
    if not chapters:
        return start_s, end_s
    
    # Find chapter containing the peak
    chapter = _find_chapter_at_time(chapters, peak_time_s)
    if not chapter:
        return start_s, end_s
    
    ch_start, ch_end = chapter
    
    # Clamp with margin
    clamped_start = max(start_s, ch_start - margin_s)
    clamped_end = min(end_s, ch_end + margin_s)
    
    # Ensure valid range (start < end)
    if clamped_start >= clamped_end:
        # Chapter is very short, just use chapter bounds
        clamped_start = ch_start
        clamped_end = ch_end
    
    return clamped_start, clamped_end


def _compute_setup_text(
    sentences: Optional[List[Sentence]],
    start_s: float,
    end_s: float,
    max_chars: int = 150,
) -> str:
    """Get setup text (first sentences in clip)."""
    if not sentences:
        return ""

    clip_sentences = get_sentences_in_range(sentences, start_s, end_s)
    if not clip_sentences:
        return ""

    # Take first sentences up to max_chars
    setup_parts: List[str] = []
    total_chars = 0

    for sent in clip_sentences[:3]:  # Max 3 sentences
        if total_chars + len(sent.text) > max_chars and setup_parts:
            break
        setup_parts.append(sent.text)
        total_chars += len(sent.text) + 1

    return " ".join(setup_parts)


def _compute_payoff_text(
    sentences: Optional[List[Sentence]],
    peak_time_s: float,
    window_s: float = 5.0,
) -> str:
    """Get payoff text (sentence near the peak moment)."""
    if not sentences:
        return ""

    # Find sentence closest to/containing the peak
    best_sent: Optional[Sentence] = None
    best_dist = float("inf")

    for sent in sentences:
        if sent.t0 <= peak_time_s <= sent.t1:
            return sent.text

        dist = min(abs(sent.t0 - peak_time_s), abs(sent.t1 - peak_time_s))
        if dist < best_dist and dist <= window_s:
            best_dist = dist
            best_sent = sent

    return best_sent.text if best_sent else ""


# =============================================================================
# Step 5: Signal summary + new variant helpers
# =============================================================================

def _mask_for_range(times: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Create boolean mask for time range."""
    times = np.asarray(times, dtype=np.float32)
    if times.size == 0:
        return np.zeros(0, dtype=bool)
    return (times >= float(start_s)) & (times <= float(end_s))


def _safe_stat_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size > 0 else 0.0


def _safe_stat_max(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size > 0 else 0.0


def _find_min_time(times: np.ndarray, values: np.ndarray, t0: float, t1: float) -> Optional[float]:
    """Return the time of the minimum value in [t0,t1], ignoring NaNs."""
    times = np.asarray(times, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if times.size == 0 or values.size == 0 or times.shape[0] != values.shape[0]:
        return None
    mask = (times >= float(t0)) & (times <= float(t1)) & np.isfinite(values)
    if not np.any(mask):
        return None
    sub_vals = values[mask]
    sub_times = times[mask]
    idx = int(np.argmin(sub_vals))
    return float(sub_times[idx])


def _compute_scores_summary(
    highlights_features: Dict[str, np.ndarray],
    *,
    start_s: float,
    end_s: float,
    peak_time_s: float,
) -> Dict[str, float]:
    """Compute per-clip signal summary stats from highlights_features.npz."""
    out: Dict[str, float] = {}
    if not highlights_features:
        return out

    times = highlights_features.get("times")
    if times is None:
        return out

    times = np.asarray(times, dtype=np.float32)
    mask = _mask_for_range(times, start_s, end_s)
    if mask.size == 0 or not np.any(mask):
        return out

    # Signal keys to summarize
    keys = [
        "combined_smoothed", "audio", "motion", "chat", "audio_events",
        "reaction_audio", "turn_rate", "overlap", "speech", "speech_fraction",
    ]

    for k in keys:
        arr = highlights_features.get(k)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[0] != times.shape[0]:
            continue
        out[f"{k}_mean"] = _safe_stat_mean(arr[mask])
        out[f"{k}_max"] = _safe_stat_max(arr[mask])

    # Values at/near the peak (helpful for sorting in UI)
    try:
        peak_idx = int(np.argmin(np.abs(times - float(peak_time_s))))
        for k in ["combined_smoothed", "reaction_audio", "turn_rate", "overlap", "chat", "audio_events", "speech_fraction"]:
            arr = highlights_features.get(k)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape[0] != times.shape[0]:
                continue
            out[f"{k}_at_peak"] = float(arr[peak_idx])
    except Exception:
        pass

    return out


def _generate_reaction_arc_variant(
    peak_time_s: float,
    duration_s: float,
    cfg: VariantGeneratorConfig,
    graph: Optional[BoundaryGraph],
    highlights_features: Optional[Dict[str, np.ndarray]],
) -> Optional[ClipVariant]:
    """Valley->valley cut around a vocal reaction peak.

    Uses the reaction_audio track from highlights_features.npz to create
    clips with a natural "build" and "release".
    """
    if not highlights_features:
        return None

    times = highlights_features.get("times")
    ra = highlights_features.get("reaction_audio")
    if times is None or ra is None:
        return None

    times = np.asarray(times, dtype=np.float32)
    ra = np.asarray(ra, dtype=np.float32)
    if times.size == 0 or ra.size == 0 or times.shape[0] != ra.shape[0]:
        return None

    pre_t0 = float(peak_time_s) - float(cfg.reaction_arc_pre_search_s)
    pre_t1 = float(peak_time_s) - float(cfg.min_pre_s)
    post_t0 = float(peak_time_s) + float(cfg.min_post_s)
    post_t1 = float(peak_time_s) + float(cfg.reaction_arc_post_search_s)

    start_t = _find_min_time(times, ra, pre_t0, pre_t1)
    end_t = _find_min_time(times, ra, post_t0, post_t1)

    if start_t is None or end_t is None:
        return None

    start_s = float(start_t)
    end_s = float(end_t)

    if end_s <= start_s:
        return None

    # Enforce duration bounds
    dur = end_s - start_s
    if dur < float(cfg.reaction_arc_min_s):
        end_s += float(cfg.reaction_arc_min_s) - dur
    elif dur > float(cfg.reaction_arc_max_s):
        excess = dur - float(cfg.reaction_arc_max_s)
        start_s += 0.5 * excess
        end_s -= 0.5 * excess

    # Snap to editor boundaries (if available)
    if graph:
        start_bp = find_best_start_boundary(graph, start_s, max_before_s=4.0, max_after_s=2.0, prefer_before=True)
        if start_bp:
            start_s = float(start_bp.time_s)
        end_bp = find_best_end_boundary(graph, end_s, max_before_s=2.0, max_after_s=4.0, prefer_after=True)
        if end_bp:
            end_s = float(end_bp.time_s)

    # Clamp to duration
    start_s = max(0.0, start_s)
    if duration_s > 0:
        end_s = min(float(duration_s), end_s)

    dur = end_s - start_s
    if dur < 8.0:
        return None

    return ClipVariant(
        variant_id="reaction_arc",
        start_s=start_s,
        end_s=end_s,
        duration_s=dur,
        description="Cuts on voice reaction arc (valley → peak → valley)",
    )


def _generate_clean_cut_variant(
    peak_time_s: float,
    duration_s: float,
    cfg: VariantGeneratorConfig,
    graph: Optional[BoundaryGraph],
) -> Optional[ClipVariant]:
    """Start on silence_end before peak; end on silence_start after peak."""
    if not graph:
        return None

    # Start: latest silence_end before peak with sufficient pre-context
    start_cands = [
        bp for bp in graph.start_boundaries
        if "silence_end" in (bp.sources or set())
        and float(bp.time_s) <= float(peak_time_s) - float(cfg.min_pre_s)
        and (float(peak_time_s) - float(bp.time_s)) <= float(cfg.max_pre_s)
    ]
    if not start_cands:
        return None

    start_bp = max(start_cands, key=lambda b: float(b.time_s))

    # End: earliest silence_start after peak with sufficient post-context
    end_cands = [
        bp for bp in graph.end_boundaries
        if "silence_start" in (bp.sources or set())
        and float(bp.time_s) >= float(peak_time_s) + float(cfg.min_post_s)
        and (float(bp.time_s) - float(peak_time_s)) <= float(cfg.max_post_s)
    ]
    if not end_cands:
        return None

    end_bp = min(end_cands, key=lambda b: float(b.time_s))

    start_s = float(start_bp.time_s)
    end_s = float(end_bp.time_s)
    dur = end_s - start_s

    if dur < float(cfg.clean_cut_min_s) or dur > float(cfg.clean_cut_max_s):
        return None

    start_s = max(0.0, start_s)
    if duration_s > 0:
        end_s = min(float(duration_s), end_s)

    dur = end_s - start_s
    if dur <= 0:
        return None

    return ClipVariant(
        variant_id="clean_cut",
        start_s=start_s,
        end_s=end_s,
        duration_s=dur,
        description="Starts/ends on natural silence boundaries (clean cut)",
    )


def _generate_duration_variant(
    peak_time_s: float,
    duration_s: float,
    duration_cfg: VariantDurationConfig,
    cfg: VariantGeneratorConfig,
    graph: Optional[BoundaryGraph],
    variant_id: str,
    description: str,
) -> ClipVariant:
    """Generate a variant with target duration."""
    target_duration = (duration_cfg.min_s + duration_cfg.max_s) / 2.0

    # Calculate ideal split around peak (slightly more post than pre)
    pre_ratio = 0.4
    ideal_pre = target_duration * pre_ratio
    ideal_post = target_duration * (1 - pre_ratio)

    # Apply constraints
    pre_s = _clamp(ideal_pre, cfg.min_pre_s, cfg.max_pre_s)
    post_s = _clamp(ideal_post, cfg.min_post_s, cfg.max_post_s)

    start_s = peak_time_s - pre_s
    end_s = peak_time_s + post_s

    # Try to snap to boundaries
    if graph:
        start_bp = find_best_start_boundary(
            graph, start_s,
            max_before_s=5.0,
            max_after_s=3.0,
            prefer_before=True,
        )
        if start_bp:
            new_start = start_bp.time_s
            # Ensure we don't violate duration constraints
            new_duration = end_s - new_start
            if duration_cfg.min_s <= new_duration <= duration_cfg.max_s:
                start_s = new_start

        end_bp = find_best_end_boundary(
            graph, end_s,
            max_before_s=3.0,
            max_after_s=5.0,
            prefer_after=True,
        )
        if end_bp:
            new_end = end_bp.time_s
            new_duration = new_end - start_s
            if duration_cfg.min_s <= new_duration <= duration_cfg.max_s:
                end_s = new_end

    # Ensure minimum duration
    duration = end_s - start_s
    if duration < duration_cfg.min_s:
        # Extend symmetrically
        needed = duration_cfg.min_s - duration
        start_s -= needed / 2
        end_s += needed / 2
    elif duration > duration_cfg.max_s:
        # Trim symmetrically
        excess = duration - duration_cfg.max_s
        start_s += excess / 2
        end_s -= excess / 2

    return ClipVariant(
        variant_id=variant_id,
        start_s=max(0.0, start_s),
        end_s=end_s,
        duration_s=end_s - max(0.0, start_s),
        description=description,
    )


def _generate_setup_first_variant(
    peak_time_s: float,
    cfg: VariantGeneratorConfig,
    graph: Optional[BoundaryGraph],
    sentences: Optional[List[Sentence]],
) -> Optional[ClipVariant]:
    """Generate variant that starts at best setup sentence."""
    if not sentences:
        return None

    # Find a good setup sentence before the peak
    setup_window_start = peak_time_s - cfg.max_pre_s
    setup_window_end = peak_time_s - cfg.min_pre_s

    setup_sentences = [
        s for s in sentences
        if setup_window_start <= s.t0 <= setup_window_end
    ]

    if not setup_sentences:
        return None

    # Pick the earliest sentence that's a reasonable setup
    setup_sent = setup_sentences[0]
    start_s = setup_sent.t0

    # End after the peak with some buffer
    end_s = peak_time_s + cfg.min_post_s

    # Snap end to boundary
    if graph:
        end_bp = find_best_end_boundary(
            graph, end_s,
            max_before_s=2.0,
            max_after_s=8.0,
            prefer_after=True,
        )
        if end_bp and end_bp.time_s > peak_time_s:
            end_s = end_bp.time_s

    duration = end_s - start_s

    # Check if duration is reasonable
    if duration < 15.0 or duration > 90.0:
        return None

    return ClipVariant(
        variant_id="setup_first",
        start_s=start_s,
        end_s=end_s,
        duration_s=duration,
        description="Starts at setup sentence for maximum context",
        setup_text=setup_sent.text,
    )


def _generate_punchline_first_variant(
    peak_time_s: float,
    cfg: VariantGeneratorConfig,
    graph: Optional[BoundaryGraph],
) -> ClipVariant:
    """Generate variant that starts close to peak (hooks fast)."""
    # Start just before peak
    start_s = peak_time_s - 3.0
    end_s = peak_time_s + cfg.min_post_s + 5.0  # Include reaction

    # Snap to boundaries
    if graph:
        start_bp = find_best_start_boundary(
            graph, start_s,
            max_before_s=2.0,
            max_after_s=2.0,
        )
        if start_bp:
            start_s = start_bp.time_s

        end_bp = find_best_end_boundary(
            graph, end_s,
            max_before_s=3.0,
            max_after_s=5.0,
        )
        if end_bp:
            end_s = end_bp.time_s

    return ClipVariant(
        variant_id="punchline_first",
        start_s=max(0.0, start_s),
        end_s=end_s,
        duration_s=end_s - max(0.0, start_s),
        description="Hooks fast, starts close to peak moment",
    )


def _generate_chat_centered_variant(
    peak_time_s: float,
    cfg: VariantGeneratorConfig,
    chat_valleys: Optional[List[float]],
    chat_bursts: Optional[List[float]],
) -> Optional[ClipVariant]:
    """Generate variant that spans from chat valley to valley through burst."""
    if not chat_valleys:
        return None

    # Find valley before peak
    valley_before = find_nearest_valley(
        chat_valleys,
        peak_time_s,
        max_distance_s=cfg.chat_valley_window_s,
        direction="before",
    )

    # Find valley after peak
    valley_after = find_nearest_valley(
        chat_valleys,
        peak_time_s,
        max_distance_s=cfg.chat_valley_window_s,
        direction="after",
    )

    if not valley_before or not valley_after:
        return None

    duration = valley_after - valley_before

    # Check reasonable bounds
    if duration < 12.0 or duration > 90.0:
        return None

    return ClipVariant(
        variant_id="chat_centered",
        start_s=valley_before,
        end_s=valley_after,
        duration_s=duration,
        description="Spans chat activity arc (valley → burst → valley)",
    )


def generate_variants_for_candidate(
    candidate: Dict[str, Any],
    *,
    cfg: VariantGeneratorConfig,
    graph: Optional[BoundaryGraph] = None,
    sentences: Optional[List[Sentence]] = None,
    chat_valleys: Optional[List[float]] = None,
    chat_bursts: Optional[List[float]] = None,
    chapters: Optional[List[Tuple[float, float]]] = None,
    highlights_features: Optional[Dict[str, np.ndarray]] = None,
    duration_s: float = 0.0,
) -> CandidateVariants:
    """Generate all variants for a single candidate.
    
    Args:
        candidate: Candidate dict with peak_time_s, start_s, end_s, etc.
        cfg: Variant generation configuration
        graph: Optional boundary graph
        sentences: Optional list of sentences
        chat_valleys: Optional list of chat valley times
        chat_bursts: Optional list of chat burst times
        chapters: Optional list of (start_s, end_s) chapter boundaries
        highlights_features: Optional dict of highlight signal arrays (for reaction_arc + scores_summary)
        duration_s: Total video duration
        
    Returns:
        CandidateVariants with 3-8 generated variants
    """
    peak_time_s = float(candidate.get("peak_time_s", 0))
    rank = int(candidate.get("rank", 0))

    variants: List[ClipVariant] = []

    # 1. Short variant (16-24s)
    short = _generate_duration_variant(
        peak_time_s, duration_s, cfg.short, cfg, graph,
        "short", "Quick highlight for shorts/reels (16-24s)",
    )
    variants.append(short)

    # 2. Medium variant (24-40s)
    medium = _generate_duration_variant(
        peak_time_s, duration_s, cfg.medium, cfg, graph,
        "medium", "Balanced clip with context (24-40s)",
    )
    variants.append(medium)

    # 3. Long variant (40-75s)
    long_var = _generate_duration_variant(
        peak_time_s, duration_s, cfg.long, cfg, graph,
        "long", "Extended clip with full story (40-75s)",
    )
    variants.append(long_var)

    # 4. Setup-first variant
    setup_var = _generate_setup_first_variant(peak_time_s, cfg, graph, sentences)
    if setup_var:
        variants.append(setup_var)

    # 5. Punchline-first variant
    punchline_var = _generate_punchline_first_variant(peak_time_s, cfg, graph)
    variants.append(punchline_var)

    # 6. Chat-centered variant
    chat_var = _generate_chat_centered_variant(
        peak_time_s, cfg, chat_valleys, chat_bursts,
    )
    if chat_var:
        variants.append(chat_var)

    # 7. Reaction arc (voice excitement) variant - Step 5
    if cfg.enable_reaction_arc:
        reaction_var = _generate_reaction_arc_variant(
            peak_time_s, duration_s, cfg, graph, highlights_features
        )
        if reaction_var:
            variants.append(reaction_var)

    # 8. Clean-cut silence variant - Step 5
    if cfg.enable_clean_cut:
        clean_var = _generate_clean_cut_variant(peak_time_s, duration_s, cfg, graph)
        if clean_var:
            variants.append(clean_var)

    # Post-process all variants
    for var in variants:
        # Apply chapter constraints (if enabled and chapters available)
        if cfg.respect_chapters and chapters:
            var.start_s, var.end_s = _clamp_to_chapter(
                var.start_s, var.end_s, peak_time_s,
                chapters, margin_s=cfg.chapter_margin_s,
            )
            var.duration_s = var.end_s - var.start_s
        
        # Add setup/payoff text
        if not var.setup_text:
            var.setup_text = _compute_setup_text(sentences, var.start_s, var.end_s)
        if not var.payoff_text:
            var.payoff_text = _compute_payoff_text(sentences, peak_time_s)

        # Attach signal summary (Step 5)
        if cfg.enable_scores_summary and highlights_features:
            try:
                var.scores_summary = _compute_scores_summary(
                    highlights_features,
                    start_s=var.start_s,
                    end_s=var.end_s,
                    peak_time_s=peak_time_s,
                )
            except Exception:
                pass

        # Clamp to video duration
        if duration_s > 0:
            var.start_s = max(0.0, var.start_s)
            var.end_s = min(duration_s, var.end_s)
            var.duration_s = var.end_s - var.start_s

    return CandidateVariants(
        candidate_rank=rank,
        candidate_id=str(candidate.get("candidate_id") or ""),
        candidate_peak_time_s=peak_time_s,
        variants=variants,
    )


def compute_clip_variants(
    proj: Project,
    *,
    cfg: VariantGeneratorConfig,
    top_n: int = 25,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute clip variants for top N candidates.
    
    Persists:
      - analysis/variants.json (canonical)
      - analysis/clip_variants.json (legacy alias for backward compatibility)
      - project.json -> analysis.variants (and legacy analysis.clip_variants)
    """
    if on_progress:
        on_progress(0.1)

    proj_data = get_project_data(proj)
    duration_s = float(proj_data.get("video", {}).get("duration_seconds", 0))

    # Get candidates
    candidates = proj_data.get("analysis", {}).get("highlights", {}).get("candidates", [])
    if not candidates:
        raise ValueError("No highlight candidates found. Run highlights analysis first.")

    # Load boundary graph
    graph = load_boundary_graph(proj)

    # Load sentences
    sentences = load_sentences(proj)

    # Load chat boundaries
    chat_data = load_chat_boundaries(proj)
    chat_valleys = chat_data.get("valleys", []) if chat_data else None
    chat_bursts = chat_data.get("bursts", []) if chat_data else None

    # Load chapters (if available and enabled)
    chapters: Optional[List[Tuple[float, float]]] = None
    if cfg.respect_chapters:
        chapters_path = proj.chapters_path
        if chapters_path.exists():
            try:
                ch_data = json.loads(chapters_path.read_text(encoding="utf-8"))
                chapters = [
                    (c["start_s"], c["end_s"])
                    for c in ch_data.get("chapters", [])
                ]
            except Exception:
                pass  # Chapters are optional, continue without them

    # Load highlight feature tracks (for reaction_arc + scores_summary) - Step 5
    highlights_features: Optional[Dict[str, np.ndarray]] = None
    if proj.highlights_features_path.exists():
        try:
            highlights_features = load_npz(proj.highlights_features_path)
        except Exception:
            highlights_features = None

    if on_progress:
        on_progress(0.2)

    # Generate variants for top N candidates
    all_variants: List[CandidateVariants] = []
    candidates_to_process = candidates[:top_n]
    total = len(candidates_to_process)

    for i, candidate in enumerate(candidates_to_process):
        cv = generate_variants_for_candidate(
            candidate,
            cfg=cfg,
            graph=graph,
            sentences=sentences,
            chat_valleys=chat_valleys,
            chat_bursts=chat_bursts,
            chapters=chapters,
            highlights_features=highlights_features,
            duration_s=duration_s,
        )
        all_variants.append(cv)

        if on_progress:
            on_progress(0.2 + 0.7 * ((i + 1) / total))

    # Build payload
    variants_path = proj.analysis_dir / "variants.json"
    legacy_variants_path = proj.analysis_dir / "clip_variants.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "short": {"min_s": cfg.short.min_s, "max_s": cfg.short.max_s},
            "medium": {"min_s": cfg.medium.min_s, "max_s": cfg.medium.max_s},
            "long": {"min_s": cfg.long.min_s, "max_s": cfg.long.max_s},
            "min_pre_s": cfg.min_pre_s,
            "max_pre_s": cfg.max_pre_s,
            "min_post_s": cfg.min_post_s,
            "max_post_s": cfg.max_post_s,
            "chat_valley_window_s": cfg.chat_valley_window_s,
            "respect_chapters": cfg.respect_chapters,
            "chapter_margin_s": cfg.chapter_margin_s,
            # Step 5 toggles
            "enable_reaction_arc": cfg.enable_reaction_arc,
            "reaction_arc": {
                "pre_search_s": cfg.reaction_arc_pre_search_s,
                "post_search_s": cfg.reaction_arc_post_search_s,
                "min_s": cfg.reaction_arc_min_s,
                "max_s": cfg.reaction_arc_max_s,
            },
            "enable_clean_cut": cfg.enable_clean_cut,
            "clean_cut": {
                "min_s": cfg.clean_cut_min_s,
                "max_s": cfg.clean_cut_max_s,
            },
            "enable_scores_summary": cfg.enable_scores_summary,
        },
        "top_n": top_n,
        "candidate_count": len(all_variants),
        "chapters_used": chapters is not None and len(chapters) > 0,
        "chapters_count": len(chapters) if chapters else 0,
        "candidates": [cv.to_dict() for cv in all_variants],
    }

    # Save variants.json (canonical)
    save_json(variants_path, payload)
    # Save legacy alias for backward compatibility
    try:
        save_json(legacy_variants_path, payload)
    except Exception:
        pass

    # Update project.json
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        meta = {
            "created_at": payload["created_at"],
            "config": payload["config"],
            "candidate_count": len(all_variants),
            "variants_json": str(variants_path.relative_to(proj.project_dir)),
            "legacy_clip_variants_json": str(legacy_variants_path.relative_to(proj.project_dir)),
        }
        d["analysis"]["variants"] = meta
        # Legacy key for older UIs
        d["analysis"]["clip_variants"] = meta

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload


def load_clip_variants(proj: Project) -> Optional[List[CandidateVariants]]:
    """Load cached clip variants if available.
    
    Prefers analysis/variants.json, falls back to legacy analysis/clip_variants.json.
    """
    for fname in ("variants.json", "clip_variants.json"):
        variants_path = proj.analysis_dir / fname
        if not variants_path.exists():
            continue
        data = json.loads(variants_path.read_text(encoding="utf-8"))
        return [CandidateVariants.from_dict(cv) for cv in data.get("candidates", [])]
    return None


def get_variants_for_candidate(
    proj: Project,
    candidate_rank: int,
) -> Optional[CandidateVariants]:
    """Get variants for a specific candidate by rank."""
    variants = load_clip_variants(proj)
    if not variants:
        return None

    for cv in variants:
        if cv.candidate_rank == candidate_rank:
            return cv

    return None


# ---------------------------------------------------------------------------
# Compatibility wrapper (tasks.py expects compute_variants)
# ---------------------------------------------------------------------------

def compute_variants(
    proj: Project,
    *,
    cfg: Dict[str, Any],
    llm_complete: Optional[Callable[[str], str]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute variants from a dict config (runner/task API).

    This is a thin wrapper over `compute_clip_variants` to match the DAG task
    signature. `llm_complete` is accepted for API compatibility but not used
    in this step (Step 6 can consume the summaries we attach).
    """
    _ = llm_complete  # reserved for Step 6
    cfg_obj = VariantGeneratorConfig.from_dict(cfg or {})
    top_n = int((cfg or {}).get("top_n", 25))
    return compute_clip_variants(
        proj,
        cfg=cfg_obj,
        top_n=top_n,
        on_progress=on_progress,
    )
