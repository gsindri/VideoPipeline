"""Speaker diarization analysis (who speaks when).

This task runs *speaker diarization* over the project's audio and produces
`analysis/diarization.json`.

Why this matters for shorts:
  - Speaker change points are excellent clip boundaries (banter, callouts)
  - Overlapping speech is often a "hype" indicator (interruptions, shouting)
  - Turn rate is a useful highlight signal (rapid back-and-forth)

The diarization backend uses `pyannote-audio` (requires a Hugging Face token).
If pyannote isn't installed or no token is available, the task is treated as
optional and simply produces no output.

Output schema (diarization.json):

  {
    "generated_at": "...",
    "duration_seconds": 123.4,
    "speaker_count": 2,
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "speaker_segments": [
      {"speaker": "SPEAKER_00", "start_s": 1.2, "end_s": 3.4},
      ...
    ],
    "turn_boundaries": [5.0, 7.1, ...],
    "turn_starts": [1.2, 5.0, ...],
    "turn_ends": [3.4, 6.2, ...],
    "overlaps": [
      {"start_s": 10.2, "end_s": 10.8, "speakers": ["SPEAKER_00", "SPEAKER_01"]},
      ...
    ],
    "turn_rate": {
      "hop_seconds": 0.5,
      "window_seconds": 30.0,
      "times": [...],
      "turns_per_minute": [...]
    },
    "overlap_fraction": {
      "hop_seconds": 0.5,
      "times": [...],
      "fraction": [...]
    },
    "config": {...}
  }

The boundary graph task can consume `turn_starts`/`turn_ends` directly.
"""

from __future__ import annotations

import json
import logging
import math
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ffmpeg import ffprobe_duration_seconds
from .project import Project, save_json, update_project

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiarizationConfig:
    """Configuration for diarization.

    Notes:
      - `hf_token` is intentionally not persisted into output JSON; only whether
        it was set is recorded.
      - For most streamer content, setting `max_speakers` to something like 6
        keeps the model from inventing too many speakers.
    """

    enabled: bool = True

    # pyannote execution
    use_gpu: bool = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    hf_token: Optional[str] = None

    # perf (best-effort; depends on pyannote/torch version)
    segmentation_batch_size: Optional[int] = None
    embedding_batch_size: Optional[int] = None
    auto_batch_size_probe: bool = True
    probe_seconds: int = 60
    matmul_precision: str = "high"
    benchmark_backend: bool = True

    # post-processing
    merge_gap_seconds: float = 0.2

    # derived signals
    hop_seconds: float = 0.5
    turn_rate_window_seconds: float = 30.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiarizationConfig":
        # Interpret common keys from the speech config too.
        enabled = bool(d.get("enabled", d.get("diarize", True)))

        # If the pipeline is explicitly set to CPU, default to CPU for diarization too.
        device = str(d.get("device", "")).lower().strip()
        use_gpu = bool(d.get("use_gpu", True))
        if device == "cpu":
            use_gpu = False

        def _opt_int(x: Any) -> Optional[int]:
            if x is None:
                return None
            try:
                return int(x)
            except Exception:
                return None

        return cls(
            enabled=enabled,
            use_gpu=use_gpu,
            min_speakers=_opt_int(d.get("min_speakers", d.get("diarize_min_speakers"))),
            max_speakers=_opt_int(d.get("max_speakers", d.get("diarize_max_speakers"))),
            hf_token=d.get("hf_token") or None,
            segmentation_batch_size=_opt_int(d.get("segmentation_batch_size") or d.get("diarize_segmentation_batch_size")),
            embedding_batch_size=_opt_int(d.get("embedding_batch_size") or d.get("diarize_embedding_batch_size")),
            auto_batch_size_probe=bool(
                d.get("auto_batch_size_probe", d.get("diarize_auto_batch_size_probe", True))
            ),
            probe_seconds=int(d.get("probe_seconds", d.get("diarize_probe_seconds", 60)) or 60),
            matmul_precision=str(d.get("matmul_precision", d.get("diarize_matmul_precision", "high")) or "high"),
            benchmark_backend=bool(d.get("benchmark_backend", d.get("diarize_benchmark_backend", True))),
            merge_gap_seconds=float(d.get("merge_gap_seconds", 0.2)),
            hop_seconds=float(d.get("hop_seconds", 0.5)),
            turn_rate_window_seconds=float(d.get("turn_rate_window_seconds", 30.0)),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float_list(xs: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for x in xs:
        try:
            out.append(float(x))
        except Exception:
            continue
    return out


def _merge_adjacent_segments(
    segments: List[Dict[str, Any]],
    *,
    gap_s: float,
) -> List[Dict[str, Any]]:
    """Merge adjacent segments belonging to the same speaker.

    pyannote can emit small gaps/splits; merging keeps turn boundaries sane.
    """
    if not segments:
        return []

    segs = sorted(segments, key=lambda s: (float(s["start_s"]), float(s["end_s"])))
    merged: List[Dict[str, Any]] = []

    cur = dict(segs[0])
    cur["start_s"] = float(cur["start_s"])
    cur["end_s"] = float(cur["end_s"])

    for s in segs[1:]:
        spk = str(s.get("speaker"))
        st = float(s.get("start_s", 0.0))
        en = float(s.get("end_s", st))
        if spk == str(cur.get("speaker")) and st - float(cur["end_s"]) <= gap_s:
            cur["end_s"] = max(float(cur["end_s"]), en)
        else:
            merged.append(cur)
            cur = {"speaker": spk, "start_s": st, "end_s": en}

    merged.append(cur)
    return merged


def _turn_boundaries_from_segments(segments: List[Dict[str, Any]]) -> List[float]:
    """Compute boundaries where the active speaker changes."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: float(s["start_s"]))
    out: List[float] = []
    prev_spk = str(segs[0].get("speaker"))
    for s in segs[1:]:
        spk = str(s.get("speaker"))
        if spk != prev_spk:
            out.append(float(s.get("start_s", 0.0)))
        prev_spk = spk
    return out


def _compute_overlaps(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute overlap intervals where 2+ speakers are active."""
    events: List[Tuple[float, int, str]] = []  # (time, delta, speaker)
    for s in segments:
        spk = str(s.get("speaker"))
        st = float(s.get("start_s", 0.0))
        en = float(s.get("end_s", st))
        if en <= st:
            continue
        events.append((st, 1, spk))
        events.append((en, -1, spk))

    if not events:
        return []

    # Sort by time; for the same timestamp, process ends before starts.
    events.sort(key=lambda e: (e[0], e[1]))

    active_counts: Dict[str, int] = {}
    active_set: set[str] = set()
    overlaps: List[Dict[str, Any]] = []

    i = 0
    prev_t = events[0][0]

    def _active_speakers() -> List[str]:
        return sorted(active_set)

    while i < len(events):
        t = events[i][0]

        # Emit interval [prev_t, t) with current active set.
        if t > prev_t and len(active_set) >= 2:
            overlaps.append(
                {
                    "start_s": float(prev_t),
                    "end_s": float(t),
                    "speakers": _active_speakers(),
                }
            )

        # Consume all events at time t.
        while i < len(events) and events[i][0] == t:
            _, delta, spk = events[i]
            active_counts[spk] = active_counts.get(spk, 0) + delta
            if active_counts[spk] <= 0:
                active_counts.pop(spk, None)
                active_set.discard(spk)
            else:
                active_set.add(spk)
            i += 1

        prev_t = t

    # No need to emit tail beyond last event.
    return overlaps


def _turn_rate_timeline(
    turn_boundaries: Sequence[float],
    *,
    duration_s: float,
    hop_s: float,
    window_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a trailing-window turns-per-minute signal."""
    if duration_s <= 0 or hop_s <= 0 or window_s <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    n = int(math.ceil(duration_s / hop_s))
    times = (np.arange(n, dtype=np.float32) * float(hop_s)).astype(np.float32)
    rates = np.zeros((n,), dtype=np.float32)

    turns = sorted(float(t) for t in turn_boundaries)
    left = 0
    right = 0

    for i, t in enumerate(times):
        t_f = float(t)
        # advance right bound
        while right < len(turns) and turns[right] <= t_f:
            right += 1
        # advance left bound
        win_start = t_f - window_s
        while left < right and turns[left] <= win_start:
            left += 1
        cnt = right - left
        rates[i] = float(cnt) / float(window_s) * 60.0

    return times, rates


def _fraction_timeline_from_intervals(
    intervals: Sequence[Tuple[float, float]],
    *,
    duration_s: float,
    hop_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fraction-of-hop timeline for any set of time intervals."""
    if duration_s <= 0 or hop_s <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    n = int(math.ceil(duration_s / hop_s))
    times = (np.arange(n, dtype=np.float32) * float(hop_s)).astype(np.float32)
    frac = np.zeros((n,), dtype=np.float32)

    if not intervals:
        return times, frac

    for st, en in intervals:
        if en <= st:
            continue
        i0 = max(0, int(st / hop_s))
        i1 = min(n - 1, int(en / hop_s))
        for i in range(i0, i1 + 1):
            w0 = float(i) * hop_s
            w1 = w0 + hop_s
            overlap = max(0.0, min(en, w1) - max(st, w0))
            if overlap <= 0:
                continue
            frac[i] = min(1.0, frac[i] + overlap / hop_s)

    return times, frac


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_diarization_analysis(
    proj: Project,
    *,
    cfg: DiarizationConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Run diarization and persist `analysis/diarization.json`.

    Returns a payload dict (also written to disk). If diarization cannot run
    (missing pyannote or token), returns a short dict and produces no file.
    """
    if not cfg.enabled:
        logger.info("[diarization] Disabled via config.")
        return {"disabled": True}

    out_path = proj.analysis_dir / "diarization.json"

    # Lazy import so basic pipeline runs without pyannote.
    diarize_audio = None
    is_available = None
    get_hf_token = None
    try:
        from .transcription.diarization import diarize_audio as _da  # type: ignore
        from .transcription.diarization import is_diarization_available as _avail  # type: ignore
        from .transcription.diarization import get_hf_token as _get_token  # type: ignore

        diarize_audio = _da
        is_available = _avail
        get_hf_token = _get_token
    except Exception:
        # Fallback if the module isn't under transcription/.
        try:
            from .diarization import diarize_audio as _da  # type: ignore
            from .diarization import is_diarization_available as _avail  # type: ignore
            from .diarization import get_hf_token as _get_token  # type: ignore

            diarize_audio = _da
            is_available = _avail
            get_hf_token = _get_token
        except Exception:
            diarize_audio = None

    if diarize_audio is None or is_available is None:
        logger.warning("[diarization] Diarization module not available; skipping.")
        return {"unavailable": True, "reason": "diarization module missing"}

    if not is_available():
        logger.warning("[diarization] pyannote-audio not installed; skipping.")
        return {"unavailable": True, "reason": "pyannote-audio not installed"}

    # Prefer explicit token; otherwise read from env via helper.
    token = cfg.hf_token
    if get_hf_token is not None:
        token = get_hf_token(token)

    if not token:
        logger.warning("[diarization] No HF token available; skipping.")
        return {"unavailable": True, "reason": "missing HF token"}

    start_time = _time.time()

    audio_path = Path(proj.audio_source)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio source not found: {audio_path}")

    # Duration is used for timelines and sanity.
    try:
        duration_s = float(ffprobe_duration_seconds(audio_path))
    except Exception as exc:
        logger.warning("[diarization] Failed to get duration via ffprobe (%s): %s", audio_path, exc)
        duration_s = 0.0

    if on_progress:
        on_progress(0.02)

    # Run diarization (pyannote)
    diar = diarize_audio(
        audio_path,
        hf_token=token,
        use_gpu=cfg.use_gpu,
        min_speakers=cfg.min_speakers,
        max_speakers=cfg.max_speakers,
        segmentation_batch_size=cfg.segmentation_batch_size,
        embedding_batch_size=cfg.embedding_batch_size,
        auto_batch_size_probe=cfg.auto_batch_size_probe,
        probe_seconds=int(cfg.probe_seconds),
        matmul_precision=str(cfg.matmul_precision),
        benchmark_backend=bool(cfg.benchmark_backend),
        on_progress=on_progress,
    )

    diar_meta = getattr(diar, "meta", None)
    if not isinstance(diar_meta, dict):
        diar_meta = {}

    # Convert to simple dict segments.
    raw_segments: List[Dict[str, Any]] = []
    for seg in getattr(diar, "segments", []) or []:
        raw_segments.append(
            {
                "speaker": str(getattr(seg, "speaker", "")),
                "start_s": float(getattr(seg, "start", 0.0)),
                "end_s": float(getattr(seg, "end", 0.0)),
            }
        )

    segments = _merge_adjacent_segments(raw_segments, gap_s=float(cfg.merge_gap_seconds))

    speakers = list(getattr(diar, "speakers", []) or [])
    if not speakers:
        speakers = sorted({str(s.get("speaker")) for s in segments if s.get("speaker")})

    speaker_count = int(len(speakers))
    if duration_s <= 0 and segments:
        duration_s = float(max(float(s["end_s"]) for s in segments))

    turn_starts = [float(s["start_s"]) for s in segments]
    turn_ends = [float(s["end_s"]) for s in segments]
    turn_boundaries = _turn_boundaries_from_segments(segments)

    overlaps = _compute_overlaps(segments)
    overlap_intervals = [(float(o["start_s"]), float(o["end_s"])) for o in overlaps]

    # Derived timelines
    hop_s = float(cfg.hop_seconds)
    tr_times, tr = _turn_rate_timeline(
        turn_boundaries,
        duration_s=float(duration_s),
        hop_s=hop_s,
        window_s=float(cfg.turn_rate_window_seconds),
    )
    ov_times, ov_frac = _fraction_timeline_from_intervals(
        overlap_intervals,
        duration_s=float(duration_s),
        hop_s=hop_s,
    )

    elapsed_seconds = _time.time() - start_time

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
        "audio": str(audio_path),
        "duration_seconds": float(duration_s),
        "speaker_count": speaker_count,
        "speakers": speakers,
        "speaker_segments": segments,
        "turn_boundaries": _safe_float_list(turn_boundaries),
        "turn_starts": _safe_float_list(turn_starts),
        "turn_ends": _safe_float_list(turn_ends),
        "overlaps": overlaps,
        "turn_rate": {
            "hop_seconds": hop_s,
            "window_seconds": float(cfg.turn_rate_window_seconds),
            "times": tr_times.astype(np.float32).tolist(),
            "turns_per_minute": tr.astype(np.float32).tolist(),
        },
        "overlap_fraction": {
            "hop_seconds": hop_s,
            "times": ov_times.astype(np.float32).tolist(),
            "fraction": ov_frac.astype(np.float32).tolist(),
        },
        "config": {
            "enabled": True,
            "use_gpu": bool(cfg.use_gpu),
            "min_speakers": cfg.min_speakers,
            "max_speakers": cfg.max_speakers,
            "segmentation_batch_size": cfg.segmentation_batch_size,
            "embedding_batch_size": cfg.embedding_batch_size,
            "auto_batch_size_probe": bool(cfg.auto_batch_size_probe),
            "probe_seconds": int(cfg.probe_seconds),
            "matmul_precision": str(cfg.matmul_precision),
            "benchmark_backend": bool(cfg.benchmark_backend),
            "merge_gap_seconds": float(cfg.merge_gap_seconds),
            "hop_seconds": hop_s,
            "turn_rate_window_seconds": float(cfg.turn_rate_window_seconds),
            "hf_token_set": bool(token),
        },
    }

    if diar_meta:
        payload["backend_meta"] = diar_meta

    save_json(out_path, payload)

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        out: Dict[str, Any] = {
            "generated_at": payload["generated_at"],
            "elapsed_seconds": float(elapsed_seconds),
            "speaker_count": payload["speaker_count"],
            "segment_count": len(segments),
            "turn_count": len(turn_boundaries),
            "diarization_json": str(out_path.relative_to(proj.project_dir)),
        }

        # Include compact backend diagnostics for the UI (keep project.json reasonably small).
        if diar_meta:
            fp = diar_meta.get("device_fingerprint")
            if fp:
                out["device_fingerprint"] = fp
            batching = diar_meta.get("batching")
            if isinstance(batching, dict):
                out["batching"] = {
                    "segmentation_batch_size": batching.get("segmentation_batch_size"),
                    "embedding_batch_size": batching.get("embedding_batch_size"),
                    "configured": bool(batching.get("configured")),
                    "auto_probe_used": bool(batching.get("auto_probe_used")),
                    "auto_probe_from_cache": bool(batching.get("auto_probe_from_cache")),
                }
            timing = diar_meta.get("timing")
            if isinstance(timing, dict):
                out["timing"] = {
                    "load_pipeline_seconds": timing.get("load_pipeline_seconds"),
                    "probe_seconds": timing.get("probe_seconds"),
                    "run_seconds": timing.get("run_seconds"),
                    "fallback_decode_seconds": timing.get("fallback_decode_seconds"),
                    "convert_seconds": timing.get("convert_seconds"),
                    "total_seconds": timing.get("total_seconds"),
                }
            out["used_waveform_input"] = bool(diar_meta.get("used_waveform_input", False))

        d["analysis"]["diarization"] = out

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload


def load_diarization(proj: Project) -> Optional[Dict[str, Any]]:
    """Load cached diarization.json if available."""
    path = proj.analysis_dir / "diarization.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[diarization] Failed to load diarization.json: {e}")
        return None


def get_diarization_boundaries(diarization: Dict[str, Any]) -> Dict[str, List[float]]:
    """Extract boundary candidates from diarization.json."""
    # Prefer explicit lists if present.
    starts = diarization.get("turn_starts") or []
    ends = diarization.get("turn_ends") or []

    # Fallback to speaker_segments.
    if not starts or not ends:
        segs = diarization.get("speaker_segments") or []
        if segs:
            if not starts:
                starts = [s.get("start_s") for s in segs]
            if not ends:
                ends = [s.get("end_s") for s in segs]

    return {
        "turn_starts": _safe_float_list(starts),
        "turn_ends": _safe_float_list(ends),
    }
