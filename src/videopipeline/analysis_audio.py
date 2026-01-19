from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .ffmpeg import AudioStreamParams, ffprobe_duration_seconds, stream_audio_blocks_f32le
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
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute audio RMS excitement timeline and top highlight candidates.

    Persists:
      - analysis/audio_features.npz (timeline_db, smoothed_db, scores)
      - project.json -> analysis.audio section
    """
    video_path = Path(proj.video_path)
    duration_s = ffprobe_duration_seconds(video_path)

    hop_samples = int(sample_rate * hop_s)
    if hop_samples <= 0:
        raise ValueError("hop_s too small")

    params = AudioStreamParams(sample_rate=sample_rate, channels=1)

    total_frames = max(1, int(duration_s / hop_s))
    timeline_db = []
    processed = 0

    for block in stream_audio_blocks_f32le(video_path, params=params, block_samples=hop_samples):
        block64 = block.astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(block64 * block64)))
        db = float(20.0 * np.log10(rms + 1e-12))
        timeline_db.append(db)

        processed += 1
        if on_progress and processed % 20 == 0:
            on_progress(min(0.95, processed / total_frames))

    x = np.array(timeline_db, dtype=np.float64)
    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    xs = moving_average(x, smooth_frames)
    scores = robust_z(xs)

    skip_frames = int(round(skip_start_s / hop_s))
    if skip_frames > 0 and skip_frames < len(scores):
        scores[:skip_frames] = -np.inf

    min_gap_frames = max(1, int(round(min_gap_s / hop_s)))
    peak_idxs = pick_top_peaks(scores, top_k=top, min_gap_frames=min_gap_frames)

    candidates = []
    for rank, idx in enumerate(peak_idxs, start=1):
        peak_time = float(idx * hop_s)
        start = max(0.0, peak_time - pre_s)
        end = min(duration_s, peak_time + post_s)
        if end - start < 3.0:
            continue
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

    save_npz(proj.audio_features_path, timeline_db=x, smoothed_db=xs, scores=scores)

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
