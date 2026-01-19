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
    motion_path = proj.motion_features_path
    if not motion_path.exists():
        raise FileNotFoundError("motion_features.npz not found; run motion analysis first")

    data = load_npz(motion_path)
    scores = data.get("scores")
    if scores is None:
        raise ValueError("motion_features.npz missing scores")
    fps_arr = data.get("fps")
    fps = float(fps_arr[0]) if fps_arr is not None and len(fps_arr) > 0 else 1.0

    cuts = detect_scene_cuts(
        scores.astype(float),
        fps=fps,
        threshold_z=threshold_z,
        min_scene_len_seconds=min_scene_len_seconds,
    )

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

    save_json(proj.scenes_path, payload)

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["scenes"] = {
            **payload,
            "scenes_json": str(proj.scenes_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload
