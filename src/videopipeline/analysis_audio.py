from __future__ import annotations

import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .audio_features import AudioFeatureConfig, audio_rms_db_timeline
from .ffmpeg import ffprobe_duration_seconds
from .peaks import moving_average, pick_top_peaks, robust_z
from .project import Project, save_npz, update_project


# ============================================================================
# Standalone Audio RMS Analysis (for early processing during download)
# ============================================================================

def compute_audio_rms_from_file(
    audio_path: Path,
    *,
    sample_rate: int = 16000,
    hop_s: float = 0.5,
    smooth_s: float = 2.0,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute audio RMS timeline from an audio file (no Project required).
    
    This is used during URL download to run audio analysis in parallel with
    video download and transcription.
    
    Args:
        audio_path: Path to audio file (WAV, MP3, M4A, etc.)
        sample_rate: Sample rate for analysis
        hop_s: Hop size in seconds
        smooth_s: Smoothing window in seconds
        on_progress: Optional progress callback
        
    Returns:
        Dict with timeline_db, smoothed_db, scores, times, hop_seconds
        Can be saved to a project later using save_audio_rms_to_project()
    """
    start_time = _time.time()
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Helper for progress reporting with optional message
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)
    
    duration_s = ffprobe_duration_seconds(audio_path)
    
    _report(0.1, "Extracting audio RMS timeline")
    
    # Compute RMS timeline (works on any audio/video file)
    cfg = AudioFeatureConfig(sample_rate=sample_rate, hop_seconds=hop_s)
    timeline_db = audio_rms_db_timeline(audio_path, cfg)
    
    _report(0.7, "Smoothing and computing z-scores")
    
    x = np.array(timeline_db, dtype=np.float64)
    times = np.arange(len(x)) * hop_s
    
    # Smooth the timeline
    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    xs = moving_average(x, smooth_frames)
    scores = robust_z(xs)
    
    _report(0.9, "Building results")
    
    elapsed_seconds = _time.time() - start_time
    
    result = {
        "timeline_db": x.tolist(),
        "smoothed_db": xs.tolist(),
        "scores": scores.tolist(),
        "times": times.tolist(),
        "hop_seconds": hop_s,
        "sample_rate": sample_rate,
        "smooth_seconds": smooth_s,
        "duration_seconds": duration_s,
        "elapsed_seconds": elapsed_seconds,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    _report(1.0, "Done")
    
    return result


def save_audio_rms_to_project(
    proj: Project,
    audio_data: Dict[str, Any],
) -> None:
    """Save pre-computed audio RMS analysis to a project.
    
    Args:
        proj: Project instance
        audio_data: Audio analysis data from compute_audio_rms_from_file()
    """
    # Convert lists back to numpy arrays for npz saving
    timeline_db = np.array(audio_data["timeline_db"], dtype=np.float64)
    smoothed_db = np.array(audio_data["smoothed_db"], dtype=np.float64)
    scores = np.array(audio_data["scores"], dtype=np.float64)
    times = np.array(audio_data["times"], dtype=np.float64)
    hop_seconds = np.array([audio_data["hop_seconds"]], dtype=np.float64)
    
    save_npz(
        proj.audio_features_path,
        timeline_db=timeline_db,
        smoothed_db=smoothed_db,
        scores=scores,
        times=times,
        hop_seconds=hop_seconds,
    )
    
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["audio"] = {
            "video": str(proj.audio_source),  # May be audio file during early analysis
            "duration_seconds": audio_data["duration_seconds"],
            "method": "audio_rms_db_peaks",
            "config": {
                "sample_rate": audio_data["sample_rate"],
                "hop_seconds": audio_data["hop_seconds"],
                "smooth_seconds": audio_data["smooth_seconds"],
            },
            "features_npz": str(proj.audio_features_path.relative_to(proj.project_dir)),
            "elapsed_seconds": audio_data["elapsed_seconds"],
            "generated_at": audio_data["generated_at"],
        }
    
    update_project(proj, _upd)


# ============================================================================
# Project-based Audio Analysis
# ============================================================================

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

    start_time = _time.time()
    video_path = Path(proj.audio_source)  # Use audio_source for fallback during early analysis
    duration_s = ffprobe_duration_seconds(video_path)
    
    # Helper for progress reporting with optional message
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)

    # Use shared RMS timeline computation
    cfg = AudioFeatureConfig(sample_rate=sample_rate, hop_seconds=hop_s)
    timeline_db = audio_rms_db_timeline(video_path, cfg)
    
    _report(0.5, "Audio extraction complete, computing scores")

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

    elapsed_seconds = _time.time() - start_time
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
        "elapsed_seconds": elapsed_seconds,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["audio"] = {
            **payload,
            "features_npz": str(proj.audio_features_path.relative_to(proj.project_dir)),
            "elapsed_seconds": elapsed_seconds,
        }

    update_project(proj, _upd)

    _report(1.0, "Done")

    return payload
