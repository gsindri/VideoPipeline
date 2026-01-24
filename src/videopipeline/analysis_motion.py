from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .ffmpeg import ffprobe_duration_seconds, ffprobe_video_stream_info, stream_video_frames_gray
from .peaks import moving_average, robust_z
from .project import Project, save_npz, update_project


def compute_motion_analysis(
    proj: Project,
    *,
    sample_fps: float,
    scale_width: int,
    smooth_s: float,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    video_path = Path(proj.video_path)
    duration_s = ffprobe_duration_seconds(video_path)
    info = ffprobe_video_stream_info(video_path)
    src_w = int(info.get("width") or 0)
    src_h = int(info.get("height") or 0)
    if src_w <= 0 or src_h <= 0:
        raise ValueError("ffprobe returned invalid dimensions")

    scale_height = max(1, int(round(src_h * (scale_width / float(src_w)))))

    total_frames = max(1, int(duration_s * sample_fps))
    processed = 0

    prev: Optional[np.ndarray] = None
    diffs: list[float] = []
    frame_count = 0

    for frame in stream_video_frames_gray(video_path, fps=sample_fps, width=scale_width, height=scale_height):
        if prev is not None:
            diff = np.mean(np.abs(frame.astype(np.float32) - prev.astype(np.float32)))
            diffs.append(float(diff))
        else:
            # First frame has no previous frame to compare, use 0 diff
            # This ensures diffs[i] corresponds to time i/sample_fps
            diffs.append(0.0)
        prev = frame
        frame_count += 1
        processed += 1
        if on_progress and processed % 20 == 0:
            on_progress(min(0.95, processed / total_frames))

    x = np.array(diffs, dtype=np.float64)
    # Create explicit times array aligned with diffs
    times = np.arange(len(x)) / sample_fps
    smooth_frames = max(1, int(round(smooth_s * sample_fps)))
    xs = moving_average(x, smooth_frames) if len(x) > 0 else x
    scores = robust_z(xs) if len(xs) > 0 else xs

    save_npz(
        proj.motion_features_path,
        diffs=x,
        smoothed=xs,
        scores=scores,
        fps=np.array([sample_fps], dtype=np.float64),
        times=times,
    )

    payload = {
        "video": str(video_path),
        "duration_seconds": duration_s,
        "method": "ffmpeg_gray_frame_diff",
        "config": {
            "sample_fps": sample_fps,
            "scale_width": scale_width,
            "scale_height": scale_height,
            "smooth_seconds": smooth_s,
        },
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["motion"] = {
            **payload,
            "features_npz": str(proj.motion_features_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload
