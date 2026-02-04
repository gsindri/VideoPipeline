"""Chat boundary detection for clip shaping.

Identifies chat activity valleys (low activity) and bursts (high activity)
as useful boundaries for clip creation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .peaks import moving_average, robust_z
from .project import Project, load_npz, save_json, update_project


@dataclass(frozen=True)
class ChatBoundaryConfig:
    """Configuration for chat boundary detection.
    
    Note: valley_threshold_z and burst_threshold_z are applied to z-scored
    chat activity. Negative z-scores indicate below-average activity.
    """
    valley_threshold_z: float = -0.5  # Z-score threshold for valleys
    burst_threshold_z: float = 1.5  # Z-score threshold for bursts
    min_valley_gap_s: float = 5.0  # Minimum gap between consecutive valleys
    min_burst_gap_s: float = 3.0  # Minimum gap between consecutive bursts


def _find_local_minima(
    scores: np.ndarray,
    *,
    threshold_z: float,
    min_gap_frames: int,
) -> List[int]:
    """Find local minima below a threshold with minimum spacing.

    Implementation notes:
    - First find *all* local minima below threshold.
    - Then apply a "best-first" non-maximum suppression so that, within
      min_gap_frames, we keep the deepest valley rather than the first one
      encountered in time order.
    """
    if len(scores) == 0:
        return []

    # Candidate local minima
    cands: List[int] = []
    for idx in range(len(scores)):
        if scores[idx] >= threshold_z:
            continue
        left = scores[idx - 1] if idx > 0 else np.inf
        right = scores[idx + 1] if idx < len(scores) - 1 else np.inf
        if scores[idx] <= left and scores[idx] <= right:
            cands.append(idx)

    if not cands:
        return []

    # Sort by depth (most negative first)
    cands_sorted = sorted(cands, key=lambda i: float(scores[i]))

    selected: List[int] = []
    blocked = np.zeros(len(scores), dtype=bool)
    radius = max(0, min_gap_frames - 1)

    for idx in cands_sorted:
        if blocked[idx]:
            continue
        selected.append(idx)
        # Block nearby indices
        lo = max(0, idx - radius)
        hi = min(len(scores), idx + radius + 1)
        blocked[lo:hi] = True

    selected.sort()
    return selected


def _find_local_maxima(
    scores: np.ndarray,
    *,
    threshold_z: float,
    min_gap_frames: int,
) -> List[int]:
    """Find local maxima above a threshold with minimum spacing.

    Uses the same best-first selection as _find_local_minima.
    """
    if len(scores) == 0:
        return []

    cands: List[int] = []
    for idx in range(len(scores)):
        if scores[idx] <= threshold_z:
            continue
        left = scores[idx - 1] if idx > 0 else -np.inf
        right = scores[idx + 1] if idx < len(scores) - 1 else -np.inf
        if scores[idx] >= left and scores[idx] >= right:
            cands.append(idx)

    if not cands:
        return []

    # Sort by height (largest first)
    cands_sorted = sorted(cands, key=lambda i: float(scores[i]), reverse=True)

    selected: List[int] = []
    blocked = np.zeros(len(scores), dtype=bool)
    radius = max(0, min_gap_frames - 1)

    for idx in cands_sorted:
        if blocked[idx]:
            continue
        selected.append(idx)
        # Block nearby indices
        lo = max(0, idx - radius)
        hi = min(len(scores), idx + radius + 1)
        blocked[lo:hi] = True

    selected.sort()
    return selected


def compute_chat_boundaries(
    chat_scores: np.ndarray,
    hop_s: float,
    cfg: ChatBoundaryConfig,
) -> Dict[str, List[float]]:
    """Compute chat valleys and bursts from chat scores.
    
    Args:
        chat_scores: Hop-aligned chat activity scores (z-scored)
        hop_s: Hop size in seconds
        cfg: Boundary detection configuration
        
    Returns:
        Dict with:
          - valleys: Times of low chat activity (good cut points)
          - bursts: Times of high chat activity (payoff indicators)
    """
    if len(chat_scores) == 0:
        return {"valleys": [], "bursts": []}

    min_valley_gap_frames = max(1, int(cfg.min_valley_gap_s / hop_s))
    min_burst_gap_frames = max(1, int(cfg.min_burst_gap_s / hop_s))

    # Find valleys (local minima below threshold)
    valley_idxs = _find_local_minima(
        chat_scores,
        threshold_z=cfg.valley_threshold_z,
        min_gap_frames=min_valley_gap_frames,
    )
    valleys = [float(idx * hop_s) for idx in valley_idxs]

    # Find bursts (local maxima above threshold)
    burst_idxs = _find_local_maxima(
        chat_scores,
        threshold_z=cfg.burst_threshold_z,
        min_gap_frames=min_burst_gap_frames,
    )
    bursts = [float(idx * hop_s) for idx in burst_idxs]

    return {"valleys": valleys, "bursts": bursts}


def compute_chat_boundaries_analysis(
    proj: Project,
    *,
    cfg: ChatBoundaryConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute chat boundaries and save to project.
    
    Requires chat_features.npz to exist.
    
    Persists:
      - analysis/chat_boundaries.json
      - project.json -> analysis.chat_boundaries section
    """
    chat_features_path = proj.chat_features_path
    if not chat_features_path.exists():
        raise FileNotFoundError("chat_features.npz not found. Run chat analysis first.")

    if on_progress:
        on_progress(0.1)

    # Load chat features
    data = load_npz(chat_features_path)
    chat_scores = data.get("scores_activity")
    if chat_scores is None:
        chat_scores = data.get("scores")
    if chat_scores is None:
        raise ValueError("chat_features.npz missing scores")

    hop_arr = data.get("hop_seconds")
    hop_s = float(hop_arr[0]) if hop_arr is not None and len(hop_arr) > 0 else 0.5

    if on_progress:
        on_progress(0.4)

    # Compute boundaries
    boundaries = compute_chat_boundaries(chat_scores.astype(np.float64), hop_s, cfg)

    if on_progress:
        on_progress(0.8)

    # Build payload
    boundaries_path = proj.analysis_dir / "chat_boundaries.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "valley_threshold_z": cfg.valley_threshold_z,
            "burst_threshold_z": cfg.burst_threshold_z,
            "min_valley_gap_s": cfg.min_valley_gap_s,
            "min_burst_gap_s": cfg.min_burst_gap_s,
        },
        "valleys": boundaries["valleys"],
        "bursts": boundaries["bursts"],
        "valley_count": len(boundaries["valleys"]),
        "burst_count": len(boundaries["bursts"]),
    }

    # Save chat_boundaries.json
    save_json(boundaries_path, payload)

    # Update project.json
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["chat_boundaries"] = {
            "created_at": payload["created_at"],
            "config": payload["config"],
            "valley_count": len(boundaries["valleys"]),
            "burst_count": len(boundaries["bursts"]),
            "boundaries_json": str(boundaries_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload


def load_chat_boundaries(proj: Project) -> Optional[Dict[str, List[float]]]:
    """Load cached chat boundaries if available."""
    boundaries_path = proj.analysis_dir / "chat_boundaries.json"
    if not boundaries_path.exists():
        return None

    data = json.loads(boundaries_path.read_text(encoding="utf-8"))
    return {
        "valleys": data.get("valleys", []),
        "bursts": data.get("bursts", []),
    }


def find_nearest_valley(
    valleys: List[float],
    time_s: float,
    *,
    max_distance_s: float = 15.0,
    direction: str = "both",  # "before", "after", or "both"
) -> Optional[float]:
    """Find the nearest valley to a given time.
    
    Args:
        valleys: List of valley timestamps
        time_s: Reference time
        max_distance_s: Maximum search distance
        direction: Search direction constraint
        
    Returns:
        Nearest valley time, or None if not found within constraints
    """
    if not valleys:
        return None

    best: Optional[float] = None
    best_dist = max_distance_s

    for valley in valleys:
        dist = valley - time_s

        if direction == "before" and dist > 0:
            continue
        if direction == "after" and dist < 0:
            continue

        abs_dist = abs(dist)
        if abs_dist < best_dist:
            best = valley
            best_dist = abs_dist

    return best


def find_burst_in_range(
    bursts: List[float],
    start_s: float,
    end_s: float,
) -> Optional[float]:
    """Find a chat burst within a time range.
    
    Returns the first burst in the range, or None if no bursts exist.
    Note: This does not consider burst intensity, just temporal ordering.
    """
    in_range = [b for b in bursts if start_s <= b <= end_s]
    return in_range[0] if in_range else None
