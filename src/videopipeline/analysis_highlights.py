from __future__ import annotations

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


def resample_series(values: np.ndarray, *, src_hop_s: float, target_hop_s: float, target_len: int) -> np.ndarray:
    if src_hop_s <= 0 or target_hop_s <= 0:
        raise ValueError("hop size must be > 0")
    if target_len <= 0:
        return np.array([], dtype=np.float64)
    times_src = np.arange(len(values)) * src_hop_s
    times_tgt = np.arange(target_len) * target_hop_s
    if len(values) == 0:
        return np.zeros(target_len, dtype=np.float64)
    return np.interp(times_tgt, times_src, values).astype(np.float64)


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


@dataclass(frozen=True)
class ClipConfig:
    min_seconds: float
    max_seconds: float
    min_pre_seconds: float
    max_pre_seconds: float
    min_post_seconds: float
    max_post_seconds: float


def shape_clip_bounds(
    *,
    peak_idx: int,
    scores: np.ndarray,
    hop_s: float,
    duration_s: float,
    clip_cfg: ClipConfig,
    scene_cuts: Iterable[float],
    snap_window_s: float,
) -> Dict[str, float]:
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


def compute_highlights_analysis(
    proj: Project,
    *,
    audio_cfg: Dict[str, Any],
    motion_cfg: Dict[str, Any],
    scenes_cfg: Dict[str, Any],
    highlights_cfg: Dict[str, Any],
    include_chat: bool = True,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
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
            from .chat.features import compute_and_save_chat_features
            compute_and_save_chat_features(
                proj,
                hop_s=hop_s,
                smooth_s=float(highlights_cfg.get("chat_smooth_seconds", 3.0)),
                on_progress=None,
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

    audio_data = load_npz(proj.audio_features_path)
    audio_scores = audio_data.get("scores")
    if audio_scores is None:
        raise ValueError("audio_features.npz missing scores")
    audio_scores = audio_scores.astype(np.float64)

    motion_data = load_npz(proj.motion_features_path)
    motion_scores = motion_data.get("scores")
    if motion_scores is None:
        raise ValueError("motion_features.npz missing scores")
    motion_scores = motion_scores.astype(np.float64)
    motion_fps_arr = motion_data.get("fps")
    motion_fps = float(motion_fps_arr[0]) if motion_fps_arr is not None and len(motion_fps_arr) > 0 else 1.0
    motion_hop_s = 1.0 / motion_fps
    motion_resampled = resample_series(motion_scores, src_hop_s=motion_hop_s, target_hop_s=hop_s, target_len=len(audio_scores))

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
            )
            chat_used = True

    weights = highlights_cfg.get("weights", {})
    w_audio = float(weights.get("audio", 0.55))
    w_motion = float(weights.get("motion", 0.45))
    # When chat is available, use a default weight of 0.20 if not explicitly set to 0
    # This re-normalizes with chat contribution
    w_chat_cfg = float(weights.get("chat", 0.20))
    w_chat = w_chat_cfg if chat_used else 0.0

    # If chat is used with default weight and audio/motion are also default,
    # normalize weights to sum to 1.0
    w_total = w_audio + w_motion + w_chat
    if w_total > 0 and abs(w_total - 1.0) > 0.001:
        w_audio = w_audio / w_total
        w_motion = w_motion / w_total
        w_chat = w_chat / w_total

    combined_raw = (w_audio * audio_scores) + (w_motion * motion_resampled) + (w_chat * chat_scores)
    smooth_s = float(highlights_cfg.get("smooth_seconds", max(1.0, 2.0 * hop_s)))
    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    combined_smoothed = moving_average(combined_raw, smooth_frames) if len(combined_raw) > 0 else combined_raw
    combined_scores = robust_z(combined_smoothed) if len(combined_smoothed) > 0 else combined_smoothed

    skip_start_frames = int(round(float(highlights_cfg.get("skip_start_seconds", 10.0)) / hop_s))
    if skip_start_frames > 0 and skip_start_frames < len(combined_scores):
        combined_scores[:skip_start_frames] = -np.inf

    min_gap_frames = max(1, int(round(float(highlights_cfg.get("min_gap_seconds", 15.0)) / hop_s)))
    peak_idxs = pick_top_peaks(combined_scores, top_k=int(highlights_cfg.get("top", 20)), min_gap_frames=min_gap_frames)

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

    duration_s = float(proj_data.get("video", {}).get("duration_seconds", hop_s * len(audio_scores)))
    candidates: List[Dict[str, Any]] = []

    for rank, idx in enumerate(peak_idxs, start=1):
        bounds = shape_clip_bounds(
            peak_idx=idx,
            scores=combined_smoothed,
            hop_s=hop_s,
            duration_s=duration_s,
            clip_cfg=clip_cfg,
            scene_cuts=scene_cuts,
            snap_window_s=snap_window_s,
        )
        start_s = bounds["start_s"]
        end_s = bounds["end_s"]
        if end_s - start_s < max(1.0, clip_cfg.min_seconds * 0.5):
            continue
        candidates.append(
            {
                "rank": rank,
                "peak_time_s": float(idx * hop_s),
                "start_s": float(start_s),
                "end_s": float(end_s),
                "score": float(combined_scores[idx]),
                "breakdown": {
                    "audio": float(audio_scores[idx]),
                    "motion": float(motion_resampled[idx]),
                    "chat": float(chat_scores[idx]) if chat_used else 0.0,
                },
            }
        )

    save_npz(
        proj.highlights_features_path,
        combined_raw=combined_raw,
        combined_smoothed=combined_smoothed,
        combined_scores=combined_scores,
        audio_scores=audio_scores,
        motion_scores=motion_resampled,
        chat_scores=chat_scores,
        hop_seconds=np.array([hop_s], dtype=np.float64),
    )

    payload = {
        "method": "multi_signal_highlights_v2",
        "config": {
            "audio": audio_cfg,
            "motion": motion_cfg,
            "scenes": scenes_cfg,
            "highlights": highlights_cfg,
        },
        "weights": {"audio": w_audio, "motion": w_motion, "chat": w_chat},
        "candidates": candidates,
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
