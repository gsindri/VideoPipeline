"""Speaker diarization using pyannote-audio.

This module provides speaker diarization (identifying who speaks when)
using the pyannote-audio library. It can be used standalone or integrated
with Whisper transcription to add speaker labels to transcripts.

Requirements:
    pip install pyannote-audio

Note: pyannote models require accepting license on Hugging Face Hub
and providing an HF token. Set HF_TOKEN environment variable or
pass hf_token to functions.

Usage:
    from videopipeline.transcription.diarization import (
        diarize_audio,
        merge_diarization_with_transcript,
        is_diarization_available,
    )
    
    # Check if available
    if is_diarization_available():
        # Run diarization
        diarization = diarize_audio(audio_path, hf_token="your_token")
        
        # Merge with existing transcript
        result = merge_diarization_with_transcript(transcript_result, diarization)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .base import TranscriptResult, TranscriptSegment, TranscriptWord

_log = logging.getLogger(__name__)

# pyannote model id (keep stable; used in caching + tests)
_MODEL_ID = "pyannote/speaker-diarization-3.1"

# Lazy-loaded pipeline
_diarization_pipeline = None

# Cache auto-selected batch sizes per (model, device, torch) fingerprint.
_batch_size_cache: Dict[str, Tuple[int, int]] = {}
_batch_size_cache_loaded: bool = False
_warned_batch_instantiate: bool = False
_warned_batch_probe: bool = False
_perf_settings_applied: bool = False
_perf_settings_meta: Dict[str, Any] = {}

# Keep Windows DLL directory handles alive. If the returned handle is dropped,
# the directory is removed from the DLL search path.
_dll_dir_handles: List[Any] = []
_dll_dir_added: set[str] = set()


def _batch_cache_path() -> Path:
    return Path.home() / ".cache" / "videopipeline" / "diarization_batch_cache.json"


def _load_batch_cache() -> None:
    global _batch_size_cache_loaded
    if _batch_size_cache_loaded:
        return
    _batch_size_cache_loaded = True

    path = _batch_cache_path()
    if not path.exists():
        return
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                continue
            try:
                seg = int(v[0])
                emb = int(v[1])
            except Exception:
                continue
            if seg > 0 and emb > 0:
                _batch_size_cache[k] = (seg, emb)
    except Exception:
        return


def _save_batch_cache() -> None:
    path = _batch_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        payload = {k: [int(v[0]), int(v[1])] for k, v in _batch_size_cache.items()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        return


def _maybe_add_windows_ffmpeg_dll_dir() -> None:
    """Ensure FFmpeg shared DLLs are discoverable on Windows.

    TorchCodec (used by pyannote for file decoding) depends on FFmpeg DLLs.
    On Windows, having FFmpeg installed is not enough if the DLL directory is
    not discoverable by the current Python process.

    If the user sets FFMPEG_SHARED_BIN (e.g. C:\\Tools\\ffmpeg-shared\\bin),
    we add it to the DLL search path early.
    """
    if os.name != "nt":
        return

    add_dll = getattr(os, "add_dll_directory", None)
    if add_dll is None:
        return

    env = os.environ.get("FFMPEG_SHARED_BIN") or ""
    if not env:
        return

    for raw in env.split(os.pathsep):
        p = raw.strip().strip('"')
        if not p:
            continue
        if p in _dll_dir_added:
            continue
        try:
            if not Path(p).exists():
                _log.warning("FFMPEG_SHARED_BIN points to missing directory: %s", p)
                continue
            _dll_dir_handles.append(add_dll(p))
            _dll_dir_added.add(p)
        except Exception as exc:
            _log.warning("Failed to add DLL directory '%s': %s", p, exc)


def _is_rocm() -> bool:
    """Return True when running on ROCm/HIP backend (even though device name is 'cuda')."""
    try:
        import torch
    except Exception:
        return False
    return bool(getattr(getattr(torch, "version", None), "hip", None))


def _disable_rnn_dropout(root: Any) -> int:
    """Force dropout=0.0 on all RNN modules reachable from `root`.

    Why:
      On some ROCm/MIOpen builds, the dropout kernel (MIOpenDropoutHIP.cpp)
      fails to compile at runtime due to missing rocrand headers, leading to
      `miopenStatusUnknownError` during diarization.

    Notes:
      - This is safe for inference (dropout is a training-time regularizer).
      - We only touch torch.nn.RNNBase subclasses (LSTM/GRU/RNN), which have a
        `dropout` attribute even in eval mode.
    """
    try:
        import torch
    except ImportError:
        return 0

    nn = torch.nn

    seen: set[int] = set()
    stack: list[Any] = [root]
    patched = 0

    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        # If it's a torch module, patch it and don't descend into __dict__.
        if isinstance(obj, nn.Module):
            for m in obj.modules():
                if isinstance(m, nn.RNNBase) and float(getattr(m, "dropout", 0.0) or 0.0) > 0.0:
                    m.dropout = 0.0
                    patched += 1
            continue

        # Common containers.
        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set)):
            stack.extend(list(obj))
            continue

        # Generic object graph walk.
        try:
            d = vars(obj)
        except Exception:
            d = None
        if d:
            stack.extend(d.values())

    return patched


def _torch_device_fingerprint() -> str:
    """Return a stable-ish key for caching tuned settings for the current torch device."""
    try:
        import torch
    except Exception:
        return "torch:missing"

    ver = getattr(torch, "__version__", "unknown")

    # Prefer CUDA/ROCm (torch uses `cuda` device type for both).
    try:
        if torch.cuda.is_available():
            backend = "rocm" if _is_rocm() else "cuda"
            name = None
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = None
            dev = name or "cuda:0"
            return f"{backend}:{dev}:torch={ver}"
    except Exception:
        pass

    # Apple MPS
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return f"mps:torch={ver}"
    except Exception:
        pass

    return f"cpu:torch={ver}"


def _apply_torch_perf_settings(
    *,
    use_gpu: bool,
    matmul_precision: str,
    benchmark_backend: bool,
) -> Dict[str, Any]:
    """Best-effort performance knobs for inference."""
    global _perf_settings_applied
    if _perf_settings_applied:
        return dict(_perf_settings_meta)

    if not use_gpu:
        return {}

    try:
        import torch
    except Exception:
        return {}

    # Only apply once per process (global knobs).
    meta: Dict[str, Any] = {}

    try:
        fn = getattr(torch, "set_float32_matmul_precision", None)
        if callable(fn) and matmul_precision:
            fn(str(matmul_precision))
            meta["matmul_precision"] = str(matmul_precision)
    except Exception:
        # Non-fatal: keep defaults.
        pass

    if benchmark_backend:
        # CUDA uses cuDNN; ROCm uses MIOpen behind the same API surface.
        try:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                meta["cudnn_benchmark"] = True
        except Exception:
            pass

        # TF32 knobs (no-op on backends that don't support it).
        try:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                meta["cuda_matmul_allow_tf32"] = True
        except Exception:
            pass
        try:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
                meta["cudnn_allow_tf32"] = True
        except Exception:
            pass

    _perf_settings_applied = True
    _perf_settings_meta.clear()
    _perf_settings_meta.update(meta)
    return dict(meta)


def _maybe_empty_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _looks_like_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "out of memory" in msg:
        return True
    if "hiperroroutofmemory" in msg:
        return True
    if "cuda" in msg and "memory" in msg and "alloc" in msg:
        return True
    return False


def _configure_pyannote_batches(
    pipeline: Any,
    *,
    segmentation_batch_size: Optional[int],
    embedding_batch_size: Optional[int],
) -> bool:
    """Best-effort: configure pyannote batch sizes.

    pyannote's public API does not consistently expose a first-class way to set
    inference batch sizes across versions. We try a few strategies:
      1) direct attributes on known pipeline internals (common in pyannote>=3)
      2) shallow name-based scan of pipeline attributes
      3) `pipeline.instantiate(...)` with a couple common config shapes
    """
    global _warned_batch_instantiate

    if segmentation_batch_size is None and embedding_batch_size is None:
        return True

    def _try_set(obj: Any, bs: Optional[int]) -> bool:
        if obj is None or bs is None:
            return False
        bs_i = int(bs)

        for attr in ("batch_size", "_batch_size"):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, bs_i)
                    return True
                except Exception:
                    pass

        setter = getattr(obj, "set_batch_size", None)
        if callable(setter):
            try:
                setter(bs_i)
                return True
            except Exception:
                pass

        return False

    seg_ok = False
    emb_ok = False

    # Known common names first.
    if segmentation_batch_size is not None:
        for name in (
            "segmentation",
            "_segmentation",
            "segmentation_inference",
            "_segmentation_inference",
        ):
            if hasattr(pipeline, name):
                seg_ok = _try_set(getattr(pipeline, name, None), segmentation_batch_size)
                if seg_ok:
                    break

    if embedding_batch_size is not None:
        for name in (
            "embedding",
            "_embedding",
            "embedding_inference",
            "_embedding_inference",
        ):
            if hasattr(pipeline, name):
                emb_ok = _try_set(getattr(pipeline, name, None), embedding_batch_size)
                if emb_ok:
                    break

    # Shallow scan fallback (handles dict-backed pipelines).
    if not seg_ok or not emb_ok:
        try:
            for k, v in vars(pipeline).items():
                lk = str(k).lower()
                if not seg_ok and segmentation_batch_size is not None and "segment" in lk:
                    seg_ok = _try_set(v, segmentation_batch_size) or seg_ok
                if not emb_ok and embedding_batch_size is not None and "embed" in lk:
                    emb_ok = _try_set(v, embedding_batch_size) or emb_ok
        except Exception:
            pass

    if (segmentation_batch_size is None or seg_ok) and (embedding_batch_size is None or emb_ok):
        return True

    cfg: Dict[str, Any] = {}
    if segmentation_batch_size is not None:
        cfg.setdefault("segmentation", {})["batch_size"] = int(segmentation_batch_size)
    if embedding_batch_size is not None:
        cfg.setdefault("embedding", {})["batch_size"] = int(embedding_batch_size)

    # Try the most common nested configuration shape first, then a flatter fallback.
    flat: Dict[str, Any] = {}
    if segmentation_batch_size is not None:
        flat["segmentation_batch_size"] = int(segmentation_batch_size)
    if embedding_batch_size is not None:
        flat["embedding_batch_size"] = int(embedding_batch_size)

    candidates: List[Dict[str, Any]] = [cfg]
    if flat:
        candidates.append(flat)

    inst = getattr(pipeline, "instantiate", None)
    if not callable(inst):
        if not _warned_batch_instantiate:
            _log.info("pyannote pipeline has no instantiate(); cannot set batch sizes (using defaults).")
            _warned_batch_instantiate = True
        return False

    for c in candidates:
        try:
            inst(c)
            return True
        except Exception:
            continue

    if not _warned_batch_instantiate:
        _log.info("Failed to set diarization batch sizes via pipeline.instantiate(); using defaults.")
        _warned_batch_instantiate = True
    return False


def _autotune_batch_sizes(
    pipeline: Any,
    *,
    cache_key: str,
    probe_input: Dict[str, Any],
    diarization_params: Dict[str, Any],
    candidates: Sequence[int],
) -> Optional[Tuple[int, int]]:
    """Pick a batch size by probing on a short input and caching the result."""
    global _warned_batch_probe

    _load_batch_cache()
    cached = _batch_size_cache.get(cache_key)
    if cached is not None:
        return cached

    for bs in candidates:
        try:
            bs_i = int(bs)
        except Exception:
            continue
        if bs_i <= 0:
            continue

        if not _configure_pyannote_batches(
            pipeline,
            segmentation_batch_size=bs_i,
            embedding_batch_size=bs_i,
        ):
            return None

        try:
            pipeline(probe_input, **diarization_params)
            _batch_size_cache[cache_key] = (bs_i, bs_i)
            _save_batch_cache()
            _log.info("Auto-selected diarization batch_size=%d (segmentation+embedding).", bs_i)
            return (bs_i, bs_i)
        except Exception as exc:
            if not _looks_like_oom(exc) and not _warned_batch_probe:
                _log.info("Batch-size probe hit non-OOM error (will still retry smaller): %s", exc)
                _warned_batch_probe = True

            _maybe_empty_cuda_cache()
            continue

    _log.info("Batch-size auto probe failed; using defaults.")
    return None


def _decode_audio_with_ffmpeg(
    path: Path,
    *,
    sample_rate: int = 16000,
    block_seconds: int = 30,
    max_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Decode audio using ffmpeg and return a pyannote-compatible dict.

    This bypasses pyannote/torchcodec audio decoding, which can be broken on
    some Windows/PyTorch builds and cause runtime errors like:
        NameError: name 'AudioDecoder' is not defined
    """
    import numpy as np

    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is required for diarization waveform decoding. "
            "Install diarization extras: pip install -e '.[diarization]'"
        ) from e

    from ..ffmpeg import AudioStreamParams, stream_audio_blocks_f32le

    duration = None if max_seconds is None else float(max_seconds)

    blocks: list[np.ndarray] = []
    for block in stream_audio_blocks_f32le(
        path,
        params=AudioStreamParams(sample_rate=sample_rate, channels=1, dtype=np.float32),
        block_samples=int(sample_rate) * int(block_seconds),
        duration_seconds=duration,
        yield_partial=True,
        pad_final=False,
    ):
        blocks.append(block)

    if not blocks:
        raise RuntimeError(f"ffmpeg decoded no audio samples for: {path}")

    samples = np.concatenate(blocks, axis=0)
    waveform = torch.from_numpy(samples).reshape(1, -1).contiguous()
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _progress_cb(cb: Optional[Callable[..., Any]], frac: float, msg: Optional[str] = None) -> None:
    """Call progress callback with best-effort support for optional message."""
    if cb is None:
        return
    try:
        f = float(frac)
    except Exception:
        f = 0.0
    f = max(0.0, min(1.0, f))
    if msg is None:
        cb(f)
        return
    try:
        cb(f, msg)
    except TypeError:
        cb(f)


class _DiarizationProgressHook:
    """Best-effort bridge from pyannote progress hooks to our on_progress callback.

    pyannote pipelines accept a `hook=...` callable in many versions. The exact
    call signature varies between releases, so we accept `*args/**kwargs` and
    infer (stage, fraction) from common patterns.
    """

    _STAGE_WEIGHTS: Dict[str, float] = {
        "segmentation": 0.55,
        "embedding": 0.35,
        "clustering": 0.10,
    }

    def __init__(
        self,
        report: Callable[[float, Optional[str]], None],
        *,
        start: float,
        end: float,
        min_emit_interval_s: float = 0.2,
    ) -> None:
        self._report = report
        self._start = float(start)
        self._end = float(end)
        self._stage_frac: Dict[str, float] = {k: 0.0 for k in self._STAGE_WEIGHTS}
        self._last_emit_time = 0.0
        self._last_emit_pct = -1
        self._last_mapped = float(start)
        self._min_emit_interval_s = float(min_emit_interval_s)

    def _stage_key(self, raw: str) -> str:
        s = (raw or "").lower().strip()
        if "segment" in s:
            return "segmentation"
        if "embed" in s:
            return "embedding"
        if "cluster" in s:
            return "clustering"
        return "segmentation"

    def _parse(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        stage = ""

        for k in ("stage", "task", "step", "name", "component"):
            v = kwargs.get(k)
            if v is not None:
                stage = str(v)
                break

        if not stage and args and isinstance(args[0], str):
            stage = str(args[0])

        frac: Optional[float] = None

        # Common keyword patterns
        if "progress" in kwargs:
            try:
                frac = float(kwargs["progress"])
            except Exception:
                frac = None
        elif "fraction" in kwargs:
            try:
                frac = float(kwargs["fraction"])
            except Exception:
                frac = None
        elif ("completed" in kwargs or "current" in kwargs) and "total" in kwargs:
            cur = kwargs.get("completed", kwargs.get("current"))
            tot = kwargs.get("total")
            try:
                cur_f = float(cur)
                tot_f = float(tot)
                if tot_f > 0:
                    frac = cur_f / tot_f
            except Exception:
                frac = None

        # Common positional patterns: (stage, completed, total) or (completed, total)
        if frac is None and len(args) >= 3:
            try:
                cur_f = float(args[1])
                tot_f = float(args[2])
                if tot_f > 0:
                    frac = cur_f / tot_f
            except Exception:
                frac = None

        if frac is None and len(args) == 2:
            try:
                v = float(args[1])
                # Either already normalized or a "completed" count.
                if 0.0 <= v <= 1.0:
                    frac = v
            except Exception:
                frac = None

        # Last resort: pick any normalized float argument
        if frac is None:
            for a in args:
                try:
                    v = float(a)
                except Exception:
                    continue
                if 0.0 <= v <= 1.0:
                    frac = v
                    break

        if frac is None:
            return None

        frac = max(0.0, min(1.0, float(frac)))
        return (stage or "progress", frac)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        parsed = self._parse(args, kwargs)
        if parsed is None:
            return
        stage, frac = parsed
        key = self._stage_key(stage)
        self._stage_frac[key] = max(self._stage_frac.get(key, 0.0), float(frac))

        overall = 0.0
        for k, w in self._STAGE_WEIGHTS.items():
            overall += float(w) * float(self._stage_frac.get(k, 0.0))

        mapped = self._start + (self._end - self._start) * float(overall)
        mapped = max(float(self._start), min(float(self._end), float(mapped)))
        mapped = max(mapped, self._last_mapped)

        now = time.time()
        pct = int(mapped * 100)
        if pct == self._last_emit_pct and (now - self._last_emit_time) < self._min_emit_interval_s:
            return

        self._last_mapped = mapped
        self._last_emit_pct = pct
        self._last_emit_time = now
        self._report(mapped, key)


@dataclass
class DiarizationSegment:
    """A segment of speech attributed to a specific speaker."""
    speaker: str  # e.g., "SPEAKER_00", "SPEAKER_01"
    start: float  # Start time in seconds
    end: float    # End time in seconds
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass 
class DiarizationResult:
    """Result of speaker diarization."""
    segments: List[DiarizationSegment]
    speakers: List[str]  # Unique speaker labels
    duration_seconds: float
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def get_speaker_at_time(self, time_s: float) -> Optional[str]:
        """Get the speaker label at a specific time."""
        for seg in self.segments:
            if seg.start <= time_s < seg.end:
                return seg.speaker
        return None
    
    def get_dominant_speaker_in_range(
        self, 
        start_s: float, 
        end_s: float
    ) -> Optional[str]:
        """Get the speaker with most speaking time in a range."""
        speaker_times: Dict[str, float] = {}
        
        for seg in self.segments:
            # Check overlap
            overlap_start = max(seg.start, start_s)
            overlap_end = min(seg.end, end_s)
            
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaker_times[seg.speaker] = speaker_times.get(seg.speaker, 0.0) + duration
        
        if not speaker_times:
            return None
        
        return max(speaker_times.items(), key=lambda x: x[1])[0]


def _merge_adjacent_speaker_segments(
    segments: List[DiarizationSegment],
    *,
    gap_s: float = 0.2,
) -> List[DiarizationSegment]:
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: (float(s.start), float(s.end)))
    out: List[DiarizationSegment] = [segs[0]]
    for seg in segs[1:]:
        prev = out[-1]
        if seg.speaker == prev.speaker and float(seg.start) <= float(prev.end) + float(gap_s):
            out[-1] = DiarizationSegment(
                speaker=prev.speaker,
                start=float(prev.start),
                end=max(float(prev.end), float(seg.end)),
            )
        else:
            out.append(seg)
    return out


def _speaker_durations(segments: List[DiarizationSegment]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for s in segments:
        dur = float(s.end) - float(s.start)
        if dur <= 0:
            continue
        totals[str(s.speaker)] = totals.get(str(s.speaker), 0.0) + dur
    return totals


def _clean_short_segments(
    segments: List[DiarizationSegment],
    *,
    hard_min_s: float = 0.10,
    flicker_min_s: float = 0.25,
    merge_gap_s: float = 0.2,
) -> List[DiarizationSegment]:
    """Remove obvious diarization flicker (e.g., 20ms speaker flips).

    This is conservative: it only rewrites *very* short segments, either to the
    surrounding speaker (when flanked by the same speaker) or to the nearest
    neighbor for micro segments.
    """
    segs_in = [s for s in segments if float(s.end) > float(s.start)]
    if not segs_in:
        return []
    segs_in.sort(key=lambda s: (float(s.start), float(s.end)))

    out: List[DiarizationSegment] = []
    for i, seg in enumerate(segs_in):
        dur = float(seg.end) - float(seg.start)
        prev = out[-1] if out else None
        nxt = segs_in[i + 1] if i + 1 < len(segs_in) else None

        speaker = str(seg.speaker)
        if dur < float(flicker_min_s) and prev is not None and nxt is not None:
            if str(prev.speaker) == str(nxt.speaker) and speaker != str(prev.speaker):
                speaker = str(prev.speaker)

        if dur < float(hard_min_s) and speaker == str(seg.speaker):
            choices: List[Tuple[float, str]] = []
            if prev is not None:
                choices.append((abs(float(seg.start) - float(prev.end)), str(prev.speaker)))
            if nxt is not None:
                choices.append((abs(float(nxt.start) - float(seg.end)), str(nxt.speaker)))
            if choices:
                choices.sort(key=lambda t: t[0])
                speaker = choices[0][1]

        out.append(DiarizationSegment(speaker=speaker, start=float(seg.start), end=float(seg.end)))

    return _merge_adjacent_speaker_segments(out, gap_s=float(merge_gap_s))


def _collapse_to_max_speakers(
    segments: List[DiarizationSegment],
    *,
    max_speakers: int,
    merge_gap_s: float = 0.2,
) -> List[DiarizationSegment]:
    """Collapse minor speakers into the nearest dominant speakers by duration."""
    if max_speakers <= 0 or not segments:
        return segments

    totals = _speaker_durations(segments)
    if len(totals) <= max_speakers:
        return segments

    top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:max_speakers]
    keep = {k for k, _ in top}
    fallback = top[0][0] if top else segments[0].speaker

    segs = sorted(segments, key=lambda s: (float(s.start), float(s.end)))
    out: List[DiarizationSegment] = []

    # Precompute nearest kept speaker by scanning neighbors.
    for i, seg in enumerate(segs):
        if seg.speaker in keep:
            out.append(seg)
            continue

        prev_choice: Optional[Tuple[float, str]] = None
        for j in range(i - 1, -1, -1):
            sj = segs[j]
            if sj.speaker in keep:
                prev_choice = (abs(float(seg.start) - float(sj.end)), str(sj.speaker))
                break

        next_choice: Optional[Tuple[float, str]] = None
        for j in range(i + 1, len(segs)):
            sj = segs[j]
            if sj.speaker in keep:
                next_choice = (abs(float(sj.start) - float(seg.end)), str(sj.speaker))
                break

        chosen = fallback
        if prev_choice is not None and next_choice is not None:
            chosen = prev_choice[1] if prev_choice[0] <= next_choice[0] else next_choice[1]
        elif prev_choice is not None:
            chosen = prev_choice[1]
        elif next_choice is not None:
            chosen = next_choice[1]

        out.append(DiarizationSegment(speaker=str(chosen), start=float(seg.start), end=float(seg.end)))

    return _merge_adjacent_speaker_segments(out, gap_s=float(merge_gap_s))


def is_diarization_available() -> bool:
    """Check if pyannote-audio is installed and available."""
    _maybe_add_windows_ffmpeg_dll_dir()
    try:
        import pyannote.audio
        return True
    except ImportError:
        return False


def get_hf_token(hf_token: Optional[str] = None) -> Optional[str]:
    """Get Hugging Face token from argument or environment."""
    if hf_token:
        return hf_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_diarization_pipeline(
    hf_token: Optional[str] = None,
    use_gpu: bool = True,
) -> Any:
    """Load the pyannote diarization pipeline.
    
    Args:
        hf_token: Hugging Face token (required for model download)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Loaded Pipeline instance
        
    Raises:
        ImportError: If pyannote-audio is not installed
        ValueError: If no HF token provided
    """
    global _diarization_pipeline
    
    if _diarization_pipeline is not None:
        return _diarization_pipeline

    _maybe_add_windows_ffmpeg_dll_dir()

    # Keep logs clean: pyannote emits a couple of noisy, non-fatal warnings in
    # common inference scenarios (TF32 note + pooling std() edge case). We
    # filter them narrowly by message+module so real warnings still surface.
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*TensorFloat-32 \\(TF32\\) has been disabled.*",
        module=r"pyannote\\.audio\\.utils\\.reproducibility",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"std\\(\\): degrees of freedom is <= 0\\..*",
        module=r"pyannote\\.audio\\.models\\.blocks\\.pooling",
    )
    
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            "pyannote-audio is required for speaker diarization. "
            "Install with: pip install pyannote-audio"
        ) from e
    
    token = get_hf_token(hf_token)
    if not token:
        raise ValueError(
            "Hugging Face token required for pyannote models. "
            "Set HF_TOKEN environment variable or pass hf_token parameter. "
            "Get your token at https://huggingface.co/settings/tokens"
        )
    
    _log.info("Loading pyannote diarization pipeline...")

    # Use the latest speaker diarization model
    model_id = _MODEL_ID

    # Preflight Hugging Face access check. This avoids some upstream libraries
    # printing raw "gated repo" guidance directly to stdout/stderr on failure.
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id, token=token)
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("401", "403")) or "gated" in msg.lower() or "restricted" in msg.lower():
            raise RuntimeError(
                f"Cannot access Hugging Face model '{model_id}' (gated). "
                f"Visit https://hf.co/{model_id} to accept the model conditions and ensure your token has access."
            ) from e
        raise

    try:
        # pyannote-audio>=3.1 uses `token=...` (huggingface_hub API).
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except TypeError as e:
        # Older versions used `use_auth_token=...`.
        if "unexpected keyword argument 'token'" not in str(e):
            raise
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)

    # Work around ROCm/MIOpen dropout JIT failures in some builds by disabling
    # RNN dropout unconditionally for inference.
    if _is_rocm():
        try:
            n = _disable_rnn_dropout(pipeline)
            if n:
                _log.info(
                    "ROCm/HIP detected (torch device='cuda'); forced dropout=0.0 for %d RNN modules to avoid MIOpen dropout JIT failures",
                    n,
                )
        except Exception as exc:
            _log.warning("ROCm dropout workaround failed (continuing): %s", exc)
    
    # Move to GPU if available and requested
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                pipeline = pipeline.to(device)

                backend = "ROCm/HIP" if _is_rocm() else "CUDA"
                gpu_name = None
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = None

                msg = f"Diarization pipeline using {backend} GPU (torch device='{device.type}')"
                if gpu_name:
                    msg += f": {gpu_name}"
                _log.info(msg)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                pipeline = pipeline.to(torch.device("mps"))
                _log.info("Diarization pipeline using Apple MPS")
            else:
                _log.info("Diarization pipeline using CPU (no GPU available)")
        except Exception as e:
            _log.warning(f"Failed to move diarization to GPU: {e}, using CPU")
    
    _diarization_pipeline = pipeline
    return pipeline


def unload_diarization_pipeline() -> None:
    """Unload the diarization pipeline to free memory."""
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        del _diarization_pipeline
        _diarization_pipeline = None
        _log.info("Diarization pipeline unloaded")


def diarize_audio(
    audio_path: Path,
    *,
    hf_token: Optional[str] = None,
    use_gpu: bool = True,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    exclusive: bool = False,
    segmentation_batch_size: Optional[int] = None,
    embedding_batch_size: Optional[int] = None,
    auto_batch_size_probe: bool = True,
    probe_seconds: int = 60,
    matmul_precision: str = "high",
    benchmark_backend: bool = True,
    on_progress: Optional[Callable[[float], None]] = None,
) -> DiarizationResult:
    """Run speaker diarization on an audio file.
    
    Args:
        audio_path: Path to audio file (WAV recommended)
        hf_token: Hugging Face token for model access
        use_gpu: Whether to use GPU acceleration
        min_speakers: Minimum expected speakers (None for auto)
        max_speakers: Maximum expected speakers (None for auto)
        on_progress: Optional progress callback
        
    Returns:
        DiarizationResult with speaker segments
        
    Raises:
        ImportError: If pyannote-audio not installed
        ValueError: If no HF token available
        FileNotFoundError: If audio file doesn't exist
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    t_total0 = time.time()

    last_progress = 0.0

    def _report(frac: float, msg: Optional[str] = None) -> None:
        nonlocal last_progress
        try:
            f = float(frac)
        except Exception:
            f = 0.0
        f = max(0.0, min(1.0, f))
        prev = last_progress
        if f < prev:
            f = prev
        # Avoid spamming callbacks with unchanged fractional progress.
        if msg is None and f <= prev + 1e-9:
            return
        last_progress = f
        _progress_cb(on_progress, f, msg)

    perf_meta = _apply_torch_perf_settings(
        use_gpu=use_gpu,
        matmul_precision=str(matmul_precision),
        benchmark_backend=bool(benchmark_backend),
    )

    _report(0.05, "init")
    
    _log.info(f"Starting speaker diarization for: {audio_path.name}")
    
    t_load0 = time.time()
    pipeline = _load_diarization_pipeline(hf_token=hf_token, use_gpu=use_gpu)
    load_s = time.time() - t_load0
    
    _report(0.12, "pipeline loaded")
    
    # Build pipeline parameters
    params: Dict[str, Any] = {}
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers

    # Batch-size tuning (best-effort). Keep it symmetric unless explicitly set.
    seg_bs = segmentation_batch_size
    emb_bs = embedding_batch_size
    if seg_bs is None and emb_bs is not None:
        seg_bs = emb_bs
    if emb_bs is None and seg_bs is not None:
        emb_bs = seg_bs

    device_fp = _torch_device_fingerprint()
    cache_key = f"{_MODEL_ID}|{device_fp}"

    probe_elapsed_s = 0.0
    probe_used = False
    probe_from_cache = False
    candidates = (64, 32, 16, 8, 4, 2, 1)
    if (
        bool(auto_batch_size_probe)
        and use_gpu
        and seg_bs is None
        and emb_bs is None
        and probe_seconds > 0
        and not device_fp.startswith("cpu:")
        and not device_fp.startswith("torch:missing")
    ):
        _load_batch_cache()
        cached = _batch_size_cache.get(cache_key)
        if cached is not None:
            probe_used = True
            probe_from_cache = True
            seg_bs, emb_bs = cached
        else:
            probe_used = True
            t_probe0 = time.time()
            try:
                probe_input = _decode_audio_with_ffmpeg(audio_path, max_seconds=float(probe_seconds))
                tuned = _autotune_batch_sizes(
                    pipeline,
                    cache_key=cache_key,
                    probe_input=probe_input,
                    diarization_params=params,
                    candidates=candidates,
                )
                if tuned is not None:
                    seg_bs, emb_bs = tuned
            except Exception as exc:
                _log.info("Batch-size auto probe failed (continuing with defaults): %s", exc)
            probe_elapsed_s = time.time() - t_probe0

    batch_requested = seg_bs is not None or emb_bs is not None
    batch_configured = False
    if batch_requested:
        batch_configured = _configure_pyannote_batches(
            pipeline,
            segmentation_batch_size=seg_bs,
            embedding_batch_size=emb_bs,
        )

    _report(0.18, "running")

    # UI polish: when pyannote doesn't report granular progress, the UI can look
    # "stuck" (e.g., staying at ~20% until completion). Run a soft ticker that
    # advances progress based on a rough runtime estimate, capped at 92% until
    # the pipeline completes.
    ticker_stop = None
    if on_progress is not None:
        try:
            import threading

            est_total_s = 300.0
            try:
                from ..ffmpeg import ffprobe_duration_seconds

                dur = float(ffprobe_duration_seconds(audio_path))
                # Heuristic: diarization is typically faster than realtime on GPU.
                rate = 0.30 if use_gpu and not device_fp.startswith("cpu:") else 1.05
                est_total_s = max(10.0, dur * rate)
            except Exception:
                est_total_s = 300.0

            start_frac = last_progress
            end_frac = 0.92
            ticker_stop = threading.Event()

            def _ticker() -> None:
                t0 = time.time()
                while not ticker_stop.is_set():
                    time.sleep(0.5)
                    elapsed = time.time() - t0
                    if est_total_s <= 0:
                        continue
                    frac = start_frac + (end_frac - start_frac) * min(elapsed / est_total_s, 1.0)
                    _report(frac)

            threading.Thread(target=_ticker, daemon=True).start()
        except Exception:
            ticker_stop = None

    hook: Optional[_DiarizationProgressHook] = None
    if on_progress is not None:
        hook = _DiarizationProgressHook(_report, start=0.18, end=0.92)

    def _call_pipeline(inp: Any) -> Any:
        nonlocal hook
        if hook is None:
            return pipeline(inp, **params)
        try:
            return pipeline(inp, hook=hook, **params)
        except TypeError as exc:
            # Some pyannote versions/pipelines don't accept `hook=...`.
            if "hook" in str(exc) or "unexpected keyword argument" in str(exc):
                hook = None
                return pipeline(inp, **params)
            raise
    
    # Run diarization
    _log.info(f"Running diarization (min_speakers={min_speakers}, max_speakers={max_speakers})...")
    used_waveform_input = False
    fallback_decode_s = 0.0
    t_run0 = time.time()
    try:
        try:
            diarization = _call_pipeline(str(audio_path))
        except Exception as e:
            msg = str(e)
            if ("AudioDecoder" not in msg) and ("torchcodec" not in msg.lower()):
                raise
            _log.warning(
                "pyannote audio decoding failed (%s). Falling back to ffmpeg-decoded waveform input.",
                msg,
            )
            used_waveform_input = True
            t_dec0 = time.time()
            wav = _decode_audio_with_ffmpeg(audio_path)
            fallback_decode_s = time.time() - t_dec0
            diarization = _call_pipeline(wav)
    finally:
        if ticker_stop is not None:
            try:
                ticker_stop.set()
            except Exception:
                pass
    run_s = time.time() - t_run0
    
    _report(0.94, "post")
    
    # Convert pyannote output to our format.
    #
    # pyannote-audio v3 speaker diarization returns a `DiarizeOutput` dataclass by
    # default (unless `pipeline.legacy` is enabled), with the underlying
    # diarization annotations stored on:
    #   - speaker_diarization (may include overlaps)
    #   - exclusive_speaker_diarization (no overlaps, better for transcript merge)
    #
    segments: List[DiarizationSegment] = []
    speakers_set: set = set()

    annotation = diarization
    try:
        if hasattr(diarization, "speaker_diarization") and hasattr(diarization, "exclusive_speaker_diarization"):
            annotation = diarization.exclusive_speaker_diarization if exclusive else diarization.speaker_diarization
    except Exception:
        annotation = diarization

    if not hasattr(annotation, "itertracks"):
        raise TypeError(
            f"Unexpected diarization output type: {type(diarization).__name__} "
            f"(expected pyannote.core.Annotation or DiarizeOutput with .speaker_diarization)"
        )

    t_conv0 = time.time()
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        seg = DiarizationSegment(
            speaker=speaker,
            start=turn.start,
            end=turn.end,
        )
        segments.append(seg)
        speakers_set.add(speaker)
    convert_s = time.time() - t_conv0
    _report(0.98, "finalize")
    
    # Sort by start time
    segments.sort(key=lambda s: s.start)

    speaker_count_before = len({str(s.speaker) for s in segments})
    segments = _clean_short_segments(segments, hard_min_s=0.10, flicker_min_s=0.25, merge_gap_s=0.2)
    if max_speakers is not None:
        try:
            max_s = int(max_speakers)
        except Exception:
            max_s = 0
        if max_s > 0:
            segments = _collapse_to_max_speakers(segments, max_speakers=max_s, merge_gap_s=0.2)
    
    # Duration is max end-time (segments may overlap).
    duration = max((s.end for s in segments), default=0.0)
    
    speakers = sorted({str(s.speaker) for s in segments})
    _log.info(f"Diarization complete: found {len(speakers)} speakers, {len(segments)} segments")
    total_s = time.time() - t_total0
    _log.info(
        "Diarization timing: load=%.2fs probe=%.2fs run=%.2fs convert=%.2fs total=%.2fs (device=%s)",
        float(load_s),
        float(probe_elapsed_s),
        float(run_s),
        float(convert_s),
        float(total_s),
        device_fp,
    )
    
    _report(1.0)
    
    return DiarizationResult(
        segments=segments,
        speakers=speakers,
        duration_seconds=duration,
        meta={
            "model_id": _MODEL_ID,
            "device_fingerprint": device_fp,
            "perf": perf_meta,
            "postprocess": {
                "speaker_count_before": int(speaker_count_before),
                "speaker_count_after": int(len(speakers)),
                "hard_min_segment_seconds": 0.10,
                "flicker_min_segment_seconds": 0.25,
                "merge_gap_seconds": 0.2,
                "max_speakers_cap": max_speakers,
            },
            "batching": {
                "segmentation_batch_size": seg_bs,
                "embedding_batch_size": emb_bs,
                "configured": bool(batch_configured),
                "auto_probe_used": bool(probe_used),
                "auto_probe_from_cache": bool(probe_from_cache),
                "auto_probe_seconds": int(probe_seconds) if probe_used else 0,
                "auto_probe_candidates": list(candidates) if probe_used else [],
                "auto_probe_cache_key": cache_key if probe_used else "",
            },
            "timing": {
                "load_pipeline_seconds": float(load_s),
                "probe_seconds": float(probe_elapsed_s),
                "run_seconds": float(run_s),
                "fallback_decode_seconds": float(fallback_decode_s),
                "convert_seconds": float(convert_s),
                "total_seconds": float(total_s),
            },
            "used_waveform_input": bool(used_waveform_input),
        },
    )


def merge_diarization_with_transcript(
    transcript: TranscriptResult,
    diarization: DiarizationResult,
) -> TranscriptResult:
    """Merge speaker diarization results with a transcript.
    
    This assigns speaker labels to transcript segments and words
    based on timing overlap with diarization segments.
    
    Args:
        transcript: Original transcript (from Whisper or similar)
        diarization: Speaker diarization results
        
    Returns:
        New TranscriptResult with speaker labels added
    """
    _log.info(f"Merging diarization ({len(diarization.speakers)} speakers) with transcript ({len(transcript.segments)} segments)")
    
    new_segments: List[TranscriptSegment] = []
    
    for seg in transcript.segments:
        # Get dominant speaker for this segment
        speaker = diarization.get_dominant_speaker_in_range(seg.start, seg.end)
        
        # Also assign speakers to individual words if available
        new_words: Optional[List[TranscriptWord]] = None
        if seg.words:
            new_words = []
            for word in seg.words:
                # Get speaker at word midpoint
                word_mid = (word.start + word.end) / 2
                word_speaker = diarization.get_speaker_at_time(word_mid)
                new_words.append(TranscriptWord(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    probability=word.probability,
                    speaker=word_speaker or speaker,  # Fall back to segment speaker
                ))
        
        new_segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=new_words,
            speaker=speaker,
        ))
    
    return TranscriptResult(
        segments=new_segments,
        language=transcript.language,
        duration_seconds=transcript.duration_seconds,
        backend_used=transcript.backend_used,
        gpu_used=transcript.gpu_used,
        speakers=diarization.speakers,
        diarization_used=True,
    )


def transcribe_with_diarization(
    audio_path: Path,
    transcriber: Any,  # Transcriber instance
    *,
    hf_token: Optional[str] = None,
    use_gpu: bool = True,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> TranscriptResult:
    """Convenience function to transcribe and diarize in one call.
    
    Runs transcription and diarization in parallel-ish fashion
    (diarization after transcription), then merges results.
    
    Args:
        audio_path: Path to audio file
        transcriber: Transcriber instance to use
        hf_token: Hugging Face token
        use_gpu: Use GPU for diarization
        min_speakers: Min speakers hint
        max_speakers: Max speakers hint
        on_progress: Progress callback (0.0 to 1.0)
        
    Returns:
        TranscriptResult with speaker labels
    """
    def transcribe_progress(p: float) -> None:
        if on_progress:
            on_progress(p * 0.6)  # Transcription is 60% of work
    
    def diarize_progress(p: float) -> None:
        if on_progress:
            on_progress(0.6 + p * 0.4)  # Diarization is 40%
    
    # Step 1: Transcribe
    _log.info("Step 1: Transcribing audio...")
    transcript = transcriber.transcribe(audio_path, on_progress=transcribe_progress)
    
    # Step 2: Diarize
    _log.info("Step 2: Running speaker diarization...")
    diarization = diarize_audio(
        audio_path,
        hf_token=hf_token,
        use_gpu=use_gpu,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        on_progress=diarize_progress,
    )
    
    # Step 3: Merge
    _log.info("Step 3: Merging transcript with speaker labels...")
    result = merge_diarization_with_transcript(transcript, diarization)
    
    return result
