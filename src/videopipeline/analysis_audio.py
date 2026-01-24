from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .audio_features import AudioFeatureConfig, audio_rms_db_timeline
from .ffmpeg import ffprobe_duration_seconds
from .peaks import moving_average, pick_top_peaks, robust_z
from .project import Project, save_npz, update_project


def compute_audio_analysis(
    proj: Project,
    *,
    sample_rate: int,
    hop_s: float,
    smooth_s: float,
    top: int,
    min_gap_s: float,
    pre_s: float,
    post_s: float,
    skip_start_s: float,
    min_clip_s: float = 3.0,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute audio RMS excitement timeline and top highlight candidates.

    Persists:
      - analysis/audio_features.npz (timeline_db, smoothed_db, scores)
      - project.json -> analysis.audio section
    """

    def _clamp_window(peak_time: float) -> tuple[float, float]:
        """Clamp a (peak_time - pre_s, peak_time + post_s) window into [0, duration].

        If the window hits an edge, we *shift* it to preserve the requested
        duration as much as possible (instead of shrinking and then being
        dropped by min_clip_s).
        """
        desired = max(0.0, float(pre_s + post_s))
        start = float(peak_time - pre_s)
        end = float(peak_time + post_s)

        # Shift right if we underflow.
        if start < 0.0:
            end = min(duration_s, end - start)
            start = 0.0

        # Shift left if we overflow.
        if end > duration_s:
            start = max(0.0, start - (end - duration_s))
            end = duration_s

        # If video is shorter than desired, just clamp.
        if desired > 0 and (end - start) > desired + 1e-6:
            end = min(duration_s, start + desired)
        return start, end
    video_path = Path(proj.video_path)
    duration_s = ffprobe_duration_seconds(video_path)

    # Use shared RMS timeline computation
    cfg = AudioFeatureConfig(sample_rate=sample_rate, hop_seconds=hop_s)
    timeline_db = audio_rms_db_timeline(video_path, cfg)
    
    if on_progress:
        on_progress(0.5)  # Mark audio processing complete

    x = np.array(timeline_db, dtype=np.float64)
    
    # Create explicit times array for consistent resampling
    times = np.arange(len(x)) * hop_s
    
    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    xs = moving_average(x, smooth_frames)
    scores = robust_z(xs)

    # Don't mutate the persisted score timeline (keep it plottable); apply
    # skip-start only for peak picking.
    scores_for_peaks = np.array(scores, dtype=np.float64, copy=True)
    skip_frames = int(round(skip_start_s / hop_s))
    if skip_frames > 0 and skip_frames < len(scores_for_peaks):
        scores_for_peaks[:skip_frames] = -np.inf

    min_gap_frames = max(1, int(round(min_gap_s / hop_s)))
    peak_idxs = pick_top_peaks(scores_for_peaks, top_k=top, min_gap_frames=min_gap_frames)

    candidates = []
    rank = 0
    for idx in peak_idxs:
        peak_time = float(idx * hop_s)
        start, end = _clamp_window(peak_time)
        if end - start < float(min_clip_s):
            continue
        rank += 1
        candidates.append(
            {
                "rank": rank,
                "peak_time_s": peak_time,
                "start_s": start,
                "end_s": end,
                "score": float(scores[idx]),
                "peak_db": float(x[idx]),
            }
        )

    save_npz(
        proj.audio_features_path,
        timeline_db=x,
        smoothed_db=xs,
        scores=scores,
        times=times,
        hop_seconds=np.array([hop_s], dtype=np.float64),
    )

    payload = {
        "video": str(video_path),
        "duration_seconds": duration_s,
        "method": "audio_rms_db_peaks",
        "config": {
            "sample_rate": sample_rate,
            "hop_seconds": hop_s,
            "smooth_seconds": smooth_s,
            "top": top,
            "min_gap_seconds": min_gap_s,
            "pre_seconds": pre_s,
            "post_seconds": post_s,
            "skip_start_seconds": skip_start_s,
            "min_clip_seconds": min_clip_s,
        },
        "candidates": candidates,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["audio"] = {
            **payload,
            "features_npz": str(proj.audio_features_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload
