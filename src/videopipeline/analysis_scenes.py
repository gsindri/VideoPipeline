from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .project import Project, load_npz, save_json, update_project


def detect_scene_cuts(
    scores: np.ndarray,
    *,
    fps: float,
    threshold_z: float,
    min_scene_len_seconds: float,
) -> List[float]:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    cut_idxs = np.where(scores >= threshold_z)[0].astype(int)
    cuts: List[float] = []
    last_cut = 0.0
    for idx in cut_idxs:
        t = float(idx / fps)
        if t - last_cut < min_scene_len_seconds:
            continue
        cuts.append(t)
        last_cut = t
    return cuts


def compute_scene_analysis(
    proj: Project,
    *,
    threshold_z: float,
    min_scene_len_seconds: float,
    snap_window_seconds: float,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    # Helper for progress reporting with optional message
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)
    
    _report(0.1, "Loading motion features")
    
    motion_path = proj.motion_features_path
    if not motion_path.exists():
        raise FileNotFoundError("motion_features.npz not found; run motion analysis first")

    data = load_npz(motion_path)
    scores = data.get("scores")
    if scores is None:
        raise ValueError("motion_features.npz missing scores")
    fps_arr = data.get("fps")
    fps = float(fps_arr[0]) if fps_arr is not None and len(fps_arr) > 0 else 1.0
    
    _report(0.3, "Detecting scene cuts")

    cuts = detect_scene_cuts(
        scores.astype(float),
        fps=fps,
        threshold_z=threshold_z,
        min_scene_len_seconds=min_scene_len_seconds,
    )
    
    _report(0.7, f"Found {len(cuts)} scene cuts")

    payload = {
        "method": "motion_spike_threshold",
        "config": {
            "threshold_z": threshold_z,
            "min_scene_len_seconds": min_scene_len_seconds,
            "snap_window_seconds": snap_window_seconds,
        },
        "cuts_seconds": cuts,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    
    _report(0.9, "Saving scenes.json")

    save_json(proj.scenes_path, payload)

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["scenes"] = {
            **payload,
            "scenes_json": str(proj.scenes_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    _report(1.0, "Done")

    return payload
