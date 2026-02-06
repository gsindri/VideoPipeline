"""Unified boundary graph for clip shaping.

Merges all boundary sources (scenes, VAD, silence, sentences, chat, chapters) into
a single "editor's cut grid" for intelligent clip boundary selection.

Boundary sources (in priority order for clip snapping):
  - Chapters: Semantic topic boundaries (high score, strong snapping)
  - Scenes: Visual shot cuts from motion detection
  - VAD: Speech start/end boundaries (robust even with constant game audio)
  - Silence: Natural pauses in audio (legacy; optional)
  - Sentences: Linguistic boundaries from transcription
  - Chat valleys: Periods of low chat activity
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from .analysis_chat_boundaries import load_chat_boundaries
from .analysis_chapters import load_chapters, get_chapter_boundaries
from .analysis_diarization import load_diarization, get_diarization_boundaries
from .analysis_sentences import load_sentences, get_sentence_boundaries
from .analysis_silence import load_silence_intervals, get_silence_boundaries
from .analysis_vad import load_vad_segments, get_vad_boundaries
from .project import Project, save_json, update_project


@dataclass(frozen=True)
class BoundaryConfig:
    """Configuration for boundary merging."""
    prefer_vad: bool = True
    vad_boundary_score: float = 1.0
    prefer_silence: bool = True
    prefer_sentences: bool = True
    prefer_scene_cuts: bool = True
    prefer_chat_valleys: bool = True
    prefer_chapters: bool = True  # Semantic chapter boundaries (strong boundaries)
    prefer_turn_boundaries: bool = True  # Speaker diarization turn boundaries
    # Snap tolerance: how close a boundary needs to be to be considered
    snap_tolerance_s: float = 1.0
    # Chapter boundaries get a higher score since they represent semantic topic shifts
    chapter_boundary_score: float = 2.0
    # Turn boundaries score - speaker changes are good clip boundaries
    turn_boundary_score: float = 1.3


@dataclass
class BoundaryPoint:
    """A single boundary point with source attribution."""
    time_s: float
    sources: Set[str] = field(default_factory=set)
    score: float = 1.0  # Higher = better boundary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": self.time_s,
            "sources": sorted(self.sources),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundaryPoint":
        return cls(
            time_s=float(d["time_s"]),
            sources=set(d.get("sources", [])),
            score=float(d.get("score", 1.0)),
        )


@dataclass
class BoundaryGraph:
    """Unified boundary graph combining all boundary sources."""
    start_boundaries: List[BoundaryPoint]  # Good places to start a clip
    end_boundaries: List[BoundaryPoint]    # Good places to end a clip
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_boundaries": [b.to_dict() for b in self.start_boundaries],
            "end_boundaries": [b.to_dict() for b in self.end_boundaries],
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundaryGraph":
        return cls(
            start_boundaries=[BoundaryPoint.from_dict(b) for b in d.get("start_boundaries", [])],
            end_boundaries=[BoundaryPoint.from_dict(b) for b in d.get("end_boundaries", [])],
            duration_s=float(d.get("duration_s", 0.0)),
        )


def _merge_boundaries(
    boundaries: List[float],
    source: str,
    existing: Dict[float, BoundaryPoint],
    tolerance_s: float,
    weight: float = 1.0,
) -> None:
    """Merge boundaries into existing dict, combining nearby points.
    
    Uses bisect for O(N log N) performance. Merges to the closest existing
    point within tolerance, and updates the time to a weighted average.
    
    Args:
        boundaries: List of boundary times to merge
        source: Source name for attribution
        existing: Dictionary of existing boundary points (modified in place)
        tolerance_s: Maximum distance to merge boundaries
        weight: Score weight for this source (higher = stronger boundary)
    """
    if not boundaries:
        return
    
    # Keep sorted list of existing times for fast lookup
    sorted_times = sorted(existing.keys())
    
    for t in boundaries:
        # Use bisect to find nearest candidates efficiently
        idx = bisect.bisect_left(sorted_times, t)
        
        # Check the closest candidates (at idx-1 and idx)
        best_match: Optional[float] = None
        best_dist = tolerance_s + 1  # Start beyond tolerance
        
        for check_idx in [idx - 1, idx]:
            if 0 <= check_idx < len(sorted_times):
                candidate = sorted_times[check_idx]
                dist = abs(candidate - t)
                if dist <= tolerance_s and dist < best_dist:
                    best_match = candidate
                    best_dist = dist
        
        if best_match is not None:
            # Merge with closest existing point
            bp = existing[best_match]
            bp.sources.add(source)
            
            # Update time to weighted average based on score
            old_weight = bp.score
            new_weight = weight
            total_weight = old_weight + new_weight
            new_time = (bp.time_s * old_weight + t * new_weight) / total_weight
            
            # If time changed significantly, we need to re-key the dict
            if abs(new_time - best_match) > 0.001:
                del existing[best_match]
                bp.time_s = new_time
                existing[new_time] = bp
                # Update sorted_times for subsequent lookups
                sorted_times.remove(best_match)
                bisect.insort(sorted_times, new_time)
            
            bp.score += weight
        else:
            # No match within tolerance, create new point
            existing[t] = BoundaryPoint(time_s=t, sources={source}, score=weight)
            bisect.insort(sorted_times, t)


def compute_boundary_graph(
    proj: Project,
    cfg: BoundaryConfig,
    *,
    scene_cuts: Optional[List[float]] = None,
) -> BoundaryGraph:
    """Compute unified boundary graph from all available sources.
    
    Args:
        proj: Project instance
        cfg: Boundary configuration
        scene_cuts: Optional list of scene cut times (from scenes.json)
        
    Returns:
        BoundaryGraph with merged start and end boundaries
    """
    # Get video duration with ffprobe fallback
    from .project import get_project_data
    from .ffmpeg import ffprobe_duration_seconds
    proj_data = get_project_data(proj)
    duration_s = float(proj_data.get("video", {}).get("duration_seconds", 0))
    
    # Fallback to ffprobe if project.json lacks duration
    if duration_s <= 0 and proj.audio_source.exists():
        try:
            duration_s = ffprobe_duration_seconds(proj.audio_source)
        except Exception:
            pass  # Leave as 0 if ffprobe fails

    start_points: Dict[float, BoundaryPoint] = {}
    end_points: Dict[float, BoundaryPoint] = {}

    tolerance = cfg.snap_tolerance_s
    
    # Always inject 0.0 as a valid start boundary and duration_s as end boundary
    # This ensures clip shaping always has safe fallback points
    start_points[0.0] = BoundaryPoint(time_s=0.0, sources={"video_start"}, score=0.5)
    if duration_s > 0:
        end_points[duration_s] = BoundaryPoint(time_s=duration_s, sources={"video_end"}, score=0.5)

    # 1. Scene cuts (both start and end boundaries)
    if cfg.prefer_scene_cuts and scene_cuts:
        _merge_boundaries(scene_cuts, "scene", start_points, tolerance)
        _merge_boundaries(scene_cuts, "scene", end_points, tolerance)

    # 2. VAD speech boundaries (robust even with constant game audio)
    if cfg.prefer_vad:
        speech_segments = load_vad_segments(proj)
        if speech_segments:
            vad_bounds = get_vad_boundaries(speech_segments)
            # Speech starts = strong start points (start at beginning of speech)
            _merge_boundaries(
                vad_bounds["speech_starts"],
                "speech_start",
                start_points,
                tolerance,
                weight=cfg.vad_boundary_score,
            )
            # Speech ends = strong end points (end at end of speech)
            _merge_boundaries(
                vad_bounds["speech_ends"],
                "speech_end",
                end_points,
                tolerance,
                weight=cfg.vad_boundary_score,
            )

    # 2b. Diarization turn boundaries (speaker changes are excellent clip boundaries)
    if cfg.prefer_turn_boundaries:
        diar_data = load_diarization(proj)
        if diar_data:
            diar_bounds = get_diarization_boundaries(diar_data)
            # Turn starts = strong start points (new speaker begins)
            _merge_boundaries(
                diar_bounds["turn_starts"],
                "turn_start",
                start_points,
                tolerance,
                weight=cfg.turn_boundary_score,
            )
            # Turn ends = strong end points (speaker finishes)
            _merge_boundaries(
                diar_bounds["turn_ends"],
                "turn_end",
                end_points,
                tolerance,
                weight=cfg.turn_boundary_score,
            )

    # 3. Silence boundaries (legacy; optional)
    if cfg.prefer_silence:
        silences = load_silence_intervals(proj)
        if silences:
            silence_bounds = get_silence_boundaries(silences)
            # Silence ends = good start points (start after quiet)
            _merge_boundaries(silence_bounds["silence_ends"], "silence_end", start_points, tolerance)
            # Silence starts = good end points (end before quiet)
            _merge_boundaries(silence_bounds["silence_starts"], "silence_start", end_points, tolerance)

    # 4. Sentence boundaries
    if cfg.prefer_sentences:
        sentences = load_sentences(proj)
        if sentences:
            sent_bounds = get_sentence_boundaries(sentences)
            # Sentence starts = good start points
            _merge_boundaries(sent_bounds["sentence_starts"], "sentence", start_points, tolerance)
            # Sentence ends = good end points
            _merge_boundaries(sent_bounds["sentence_ends"], "sentence", end_points, tolerance)

    # 5. Chat valleys (good for both start and end)
    if cfg.prefer_chat_valleys:
        chat_bounds = load_chat_boundaries(proj)
        if chat_bounds:
            # Use .get() and normalize floats for defensive schema handling
            valleys = chat_bounds.get("valleys") or []
            valleys = [float(v) for v in valleys]  # Normalize to float

            # Shift valleys into video time using the current sync offset.
            # Offset convention: video_time = chat_time + offset.
            try:
                from .project import get_chat_config

                chat_cfg = get_chat_config(proj)
                offset_ms = int(chat_cfg.get("sync_offset_ms", 0) or 0)
                offset_s = float(offset_ms) / 1000.0
            except Exception:
                offset_s = 0.0

            if abs(offset_s) > 1e-6:
                valleys = [v + offset_s for v in valleys]

            # Clamp to video duration if known (keeps merge stable).
            if duration_s > 0:
                valleys = [v for v in valleys if 0.0 <= v <= duration_s]
            _merge_boundaries(valleys, "chat_valley", start_points, tolerance)
            _merge_boundaries(valleys, "chat_valley", end_points, tolerance)

    # 6. Semantic chapter boundaries (strong boundaries for topic-aligned clips)
    if cfg.prefer_chapters:
        chapters = load_chapters(proj)
        if chapters:
            chapter_bounds = get_chapter_boundaries(chapters)
            # Chapter starts are excellent start points (topic begins here)
            # Use _merge_boundaries with higher weight for proper snap/merge handling
            # This avoids the float equality bug (if t in start_points)
            _merge_boundaries(
                chapter_bounds, "chapter", start_points, tolerance, 
                weight=cfg.chapter_boundary_score
            )
            # Chapter ends are good end points (topic concludes here)
            chapter_ends = [float(c.end_s) for c in chapters]  # Normalize to float
            _merge_boundaries(
                chapter_ends, "chapter", end_points, tolerance,
                weight=cfg.chapter_boundary_score
            )

    # 7. Synthetic fallback boundaries - ensure we have coverage every N seconds
    # This prevents "rough cut" warnings when natural boundaries are too sparse
    SYNTHETIC_INTERVAL_S = 5.0  # Generate synthetic boundaries every 5 seconds
    SYNTHETIC_SCORE = 0.3  # Lower score than natural boundaries
    
    if duration_s > 0:
        # Find gaps in start boundaries and fill with synthetic points
        start_times_set = set(start_points.keys())
        end_times_set = set(end_points.keys())
        
        t = 0.0
        while t <= duration_s:
            # Check if there's any boundary within SYNTHETIC_INTERVAL_S / 2
            has_nearby_start = any(abs(existing_t - t) < SYNTHETIC_INTERVAL_S / 2 for existing_t in start_times_set)
            has_nearby_end = any(abs(existing_t - t) < SYNTHETIC_INTERVAL_S / 2 for existing_t in end_times_set)
            
            if not has_nearby_start:
                start_points[t] = BoundaryPoint(time_s=t, sources={"synthetic"}, score=SYNTHETIC_SCORE)
                start_times_set.add(t)
            
            if not has_nearby_end:
                end_points[t] = BoundaryPoint(time_s=t, sources={"synthetic"}, score=SYNTHETIC_SCORE)
                end_times_set.add(t)
            
            t += SYNTHETIC_INTERVAL_S

    # Sort by time
    start_list = sorted(start_points.values(), key=lambda b: b.time_s)
    end_list = sorted(end_points.values(), key=lambda b: b.time_s)

    return BoundaryGraph(
        start_boundaries=start_list,
        end_boundaries=end_list,
        duration_s=duration_s,
    )


def compute_boundaries_analysis(
    proj: Project,
    *,
    cfg: BoundaryConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute unified boundaries and save to project.
    
    Persists:
      - analysis/boundaries.json
      - project.json -> analysis.boundaries section
    """
    # Helper for progress reporting with optional message
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)
    
    _report(0.1, "Loading scene cuts")

    # Load scene cuts if available
    scene_cuts: List[float] = []
    if proj.scenes_path.exists():
        try:
            scenes_data = json.loads(proj.scenes_path.read_text(encoding="utf-8"))
            scene_cuts = scenes_data.get("cuts_seconds", []) or scenes_data.get("cuts", [])
        except Exception:
            pass

    _report(0.3, "Computing boundary graph")

    # Compute boundary graph
    graph = compute_boundary_graph(proj, cfg, scene_cuts=scene_cuts)

    _report(0.8, "Saving boundaries.json")

    # Build payload
    boundaries_path = proj.analysis_dir / "boundaries.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "prefer_silence": cfg.prefer_silence,
            "prefer_sentences": cfg.prefer_sentences,
            "prefer_scene_cuts": cfg.prefer_scene_cuts,
            "prefer_chat_valleys": cfg.prefer_chat_valleys,
            "prefer_chapters": cfg.prefer_chapters,
            "snap_tolerance_s": cfg.snap_tolerance_s,
            "chapter_boundary_score": cfg.chapter_boundary_score,
        },
        "start_boundary_count": len(graph.start_boundaries),
        "end_boundary_count": len(graph.end_boundaries),
        "duration_s": graph.duration_s,
        **graph.to_dict(),
    }

    # Save boundaries.json
    save_json(boundaries_path, payload)

    # Update project.json
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["boundaries"] = {
            "created_at": payload["created_at"],
            "config": payload["config"],
            "start_boundary_count": len(graph.start_boundaries),
            "end_boundary_count": len(graph.end_boundaries),
            "boundaries_json": str(boundaries_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    _report(1.0, f"Done ({len(graph.start_boundaries)} starts, {len(graph.end_boundaries)} ends)")

    return payload


def load_boundary_graph(proj: Project) -> Optional[BoundaryGraph]:
    """Load cached boundary graph if available.
    
    Tries both boundary_graph.json (new format) and boundaries.json (legacy format),
    picking whichever is valid and newest. This ensures resilience against stale
    or corrupted files (e.g., from interrupted runs).
    """
    import logging
    import os
    
    log = logging.getLogger(__name__)
    
    new_path = proj.analysis_dir / "boundary_graph.json"
    legacy_path = proj.analysis_dir / "boundaries.json"
    
    candidates: list[tuple[Path, Optional[BoundaryGraph], float]] = []  # (path, graph, mtime_or_created_at)
    
    for path in [new_path, legacy_path]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            graph = BoundaryGraph.from_dict(data)
            
            # Determine freshness: prefer created_at timestamp, fall back to file mtime
            created_at_str = data.get("created_at")
            if created_at_str:
                try:
                    from datetime import datetime
                    # Parse ISO format timestamp
                    dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    freshness = dt.timestamp()
                except (ValueError, TypeError):
                    freshness = os.path.getmtime(path)
            else:
                freshness = os.path.getmtime(path)
            
            candidates.append((path, graph, freshness))
            log.debug(f"[load_boundary_graph] Loaded valid graph from {path.name}, freshness={freshness}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            log.warning(f"[load_boundary_graph] Failed to load {path.name}: {e}, will try fallback")
            continue
    
    if not candidates:
        return None
    
    # Pick the newest valid graph
    candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by freshness descending
    best_path, best_graph, _ = candidates[0]
    
    if len(candidates) > 1:
        log.info(f"[load_boundary_graph] Using {best_path.name} (newest of {len(candidates)} valid files)")
    
    return best_graph


def save_boundary_graph(proj: Project, graph: BoundaryGraph) -> None:
    """Save a boundary graph to disk.
    
    This is a simpler version of compute_boundaries_analysis that just
    saves the graph without full project.json updates.
    """
    boundaries_path = proj.analysis_dir / "boundary_graph.json"
    
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_boundary_count": len(graph.start_boundaries),
        "end_boundary_count": len(graph.end_boundaries),
        "duration_s": graph.duration_s,
        **graph.to_dict(),
    }
    
    save_json(boundaries_path, payload)


def find_start_boundary_candidates(
    graph: BoundaryGraph,
    target_s: float,
    *,
    max_before_s: float = 10.0,
    max_after_s: float = 3.0,
    prefer_before: bool = True,
) -> List[BoundaryPoint]:
    """Find ranked start boundary candidates near a target time.
    
    Args:
        graph: Boundary graph
        target_s: Target start time
        max_before_s: Maximum time before target to search
        max_after_s: Maximum time after target to search
        prefer_before: Prefer boundaries before target (for context)
        
    Returns:
        List of boundary points sorted by score (best first)
    """
    candidates: List[BoundaryPoint] = []
    min_t = target_s - max_before_s
    max_t = target_s + max_after_s

    for bp in graph.start_boundaries:
        if min_t <= bp.time_s <= max_t:
            candidates.append(bp)

    if not candidates:
        return []

    # Score candidates: prefer higher boundary score, closer to target, before target
    def score_candidate(bp: BoundaryPoint) -> float:
        dist = abs(bp.time_s - target_s)
        dist_score = 1.0 / (1.0 + dist)  # Closer is better

        position_bonus = 0.0
        if prefer_before and bp.time_s < target_s:
            position_bonus = 0.2
        elif not prefer_before and bp.time_s > target_s:
            position_bonus = 0.2

        return bp.score * dist_score + position_bonus

    candidates.sort(key=score_candidate, reverse=True)
    return candidates


def find_best_start_boundary(
    graph: BoundaryGraph,
    target_s: float,
    *,
    max_before_s: float = 10.0,
    max_after_s: float = 3.0,
    prefer_before: bool = True,
) -> Optional[BoundaryPoint]:
    """Find the best start boundary near a target time.
    
    Args:
        graph: Boundary graph
        target_s: Target start time
        max_before_s: Maximum time before target to search
        max_after_s: Maximum time after target to search
        prefer_before: Prefer boundaries before target (for context)
        
    Returns:
        Best boundary point, or None if none found
    """
    candidates = find_start_boundary_candidates(
        graph, target_s,
        max_before_s=max_before_s,
        max_after_s=max_after_s,
        prefer_before=prefer_before,
    )
    return candidates[0] if candidates else None


def find_end_boundary_candidates(
    graph: BoundaryGraph,
    target_s: float,
    *,
    max_before_s: float = 3.0,
    max_after_s: float = 10.0,
    prefer_after: bool = True,
) -> List[BoundaryPoint]:
    """Find ranked end boundary candidates near a target time.
    
    Args:
        graph: Boundary graph
        target_s: Target end time
        max_before_s: Maximum time before target to search
        max_after_s: Maximum time after target to search
        prefer_after: Prefer boundaries after target (for payoff)
        
    Returns:
        List of boundary points sorted by score (best first)
    """
    candidates: List[BoundaryPoint] = []
    min_t = target_s - max_before_s
    max_t = target_s + max_after_s

    for bp in graph.end_boundaries:
        if min_t <= bp.time_s <= max_t:
            candidates.append(bp)

    if not candidates:
        return []

    # Score candidates
    def score_candidate(bp: BoundaryPoint) -> float:
        dist = abs(bp.time_s - target_s)
        dist_score = 1.0 / (1.0 + dist)

        position_bonus = 0.0
        if prefer_after and bp.time_s > target_s:
            position_bonus = 0.2
        elif not prefer_after and bp.time_s < target_s:
            position_bonus = 0.2

        return bp.score * dist_score + position_bonus

    candidates.sort(key=score_candidate, reverse=True)
    return candidates


def find_best_end_boundary(
    graph: BoundaryGraph,
    target_s: float,
    *,
    max_before_s: float = 3.0,
    max_after_s: float = 10.0,
    prefer_after: bool = True,
) -> Optional[BoundaryPoint]:
    """Find the best end boundary near a target time.
    
    Args:
        graph: Boundary graph
        target_s: Target end time
        max_before_s: Maximum time before target to search
        max_after_s: Maximum time after target to search
        prefer_after: Prefer boundaries after target (for payoff)
        
    Returns:
        Best boundary point, or None if none found
    """
    candidates = find_end_boundary_candidates(
        graph, target_s,
        max_before_s=max_before_s,
        max_after_s=max_after_s,
        prefer_after=prefer_after,
    )
    return candidates[0] if candidates else None
