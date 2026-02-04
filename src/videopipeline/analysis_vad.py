"""Voice Activity Detection (VAD) analysis using Silero VAD.

This task detects *speech* segments in the video's audio. It is more robust than
FFmpeg silencedetect for streams/VODs because it can find speech even when the
game/music bed is never "silent".

Outputs (persisted to analysis/audio_vad.npz):
  - speech_segments: Nx2 float32 array of [start_s, end_s] for detected speech
  - non_speech_segments: Mx2 float32 array of [start_s, end_s] gaps between speech
  - speech_fraction: float32 timeline sampled at `hop_seconds` (0..1 fraction speech)
  - times: float32 timeline timestamps aligned with speech_fraction
  - hop_seconds: scalar float32

Project metadata is stored under project.json -> analysis.vad.
"""

from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .ffmpeg import _require_cmd, ffprobe_duration_seconds
from .project import Project, load_npz, save_npz, update_project
from .utils import subprocess_flags as _subprocess_flags

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VadConfig:
    """Configuration for Silero VAD.

    Notes:
    - Silero VAD expects 8 kHz or 16 kHz audio for best results.
    - For long VODs, we run it in a streaming mode to avoid loading full audio into RAM.
    """

    enabled: bool = True
    sample_rate: int = 16000  # 8000 or 16000
    hop_seconds: float = 0.5  # timeline resolution for speech_fraction

    # Streaming VADIterator knobs
    threshold: float = 0.5
    min_silence_duration_ms: int = 250
    speech_pad_ms: int = 60

    # Post-filtering
    min_speech_duration_ms: int = 250

    # Model loading
    use_onnx: bool = False
    opset_version: int = 16
    torch_threads: int = 1
    device: str = "cpu"  # "cpu" or "cuda" (if you have it)


@dataclass(frozen=True)
class SpeechSegment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_tuple(self) -> Tuple[float, float]:
        return (float(self.start_s), float(self.end_s))


def _window_size_samples(sample_rate: int) -> int:
    # Silero examples use 512 (16 kHz) and 256 (8 kHz).
    if sample_rate == 16000:
        return 512
    if sample_rate == 8000:
        return 256
    raise ValueError("Silero VAD streaming expects sample_rate of 8000 or 16000")


def _load_silero_vad_iterator(cfg: VadConfig):
    """Load Silero VAD model + VADIterator.

    Preference order:
      1) pip package `silero-vad` (import name: silero_vad)
      2) torch.hub load from snakers4/silero-vad (requires internet or a cached repo)

    Returns:
      (model, VADIterator_class, source_name)
    """
    import torch

    # Keep VAD from hogging all cores; let the DAG parallelize instead.
    if cfg.torch_threads and cfg.torch_threads > 0:
        try:
            torch.set_num_threads(int(cfg.torch_threads))
        except Exception:
            pass

    # 1) pip package
    try:
        from silero_vad import load_silero_vad, VADIterator  # type: ignore

        model = load_silero_vad(onnx=cfg.use_onnx, opset_version=cfg.opset_version)
        return model, VADIterator, "pip:silero-vad"
    except Exception as e:
        logger.debug(f"[audio_vad] pip load failed: {e}")

    # 2) torch.hub
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=cfg.use_onnx,
            opset_version=cfg.opset_version,
        )
        # utils tuple: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
        VADIterator = utils[3]
        return model, VADIterator, "torchhub:snakers4/silero-vad"
    except Exception as e:
        raise RuntimeError(
            "Failed to load Silero VAD.\n"
            "Install `silero-vad` (recommended) or ensure torch.hub can load snakers4/silero-vad.\n"
            "Tip: `pip install silero-vad torchaudio`."
        ) from e


def _iter_audio_frames_ffmpeg(
    media_path: Path,
    *,
    sample_rate: int,
    window_size_samples: int,
    duration_s: float,
    on_progress: Optional[Callable[[float], None]] = None,
):
    """Yield mono float32 frames in [-1, 1] from any media file via FFmpeg."""
    _require_cmd("ffmpeg")

    # Decode to signed 16-bit PCM, mono, resampled.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-sn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_subprocess_flags(),
    )
    assert proc.stdout is not None

    bytes_per_sample = 2  # s16le
    frame_bytes = window_size_samples * bytes_per_sample
    block_bytes = frame_bytes * 200  # ~6.4s at 16k/512

    expected_total = max(1.0, float(duration_s)) * sample_rate * bytes_per_sample
    read_total = 0
    last_progress_emit = 0.0

    buf = b""
    try:
        while True:
            chunk = proc.stdout.read(block_bytes)
            if not chunk:
                break
            buf += chunk
            read_total += len(chunk)

            # Emit rough progress based on bytes read.
            if on_progress:
                p = min(0.95, 0.05 + 0.9 * (read_total / expected_total))
                # Avoid spamming progress callbacks.
                if p - last_progress_emit >= 0.01:
                    last_progress_emit = p
                    on_progress(p)

            # Consume full frames from buffer.
            while len(buf) >= frame_bytes:
                frame = buf[:frame_bytes]
                buf = buf[frame_bytes:]
                x_i16 = np.frombuffer(frame, dtype=np.int16)
                # Convert to float32 torch-friendly range [-1, 1]
                x = x_i16.astype(np.float32) / 32768.0
                yield x
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        stderr = b""
        try:
            if proc.stderr is not None:
                stderr = proc.stderr.read()
                proc.stderr.close()
        except Exception:
            pass
        rc = proc.wait()
        if rc != 0:
            msg = stderr.decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"FFmpeg audio decode failed (rc={rc}): {msg}")


def _segments_from_vad(
    media_path: Path,
    *,
    cfg: VadConfig,
    duration_s: float,
    on_progress: Optional[Callable[[float], None]] = None,
) -> List[SpeechSegment]:
    """Run Silero streaming VAD and return speech segments."""
    if cfg.sample_rate not in (8000, 16000):
        raise ValueError("VadConfig.sample_rate must be 8000 or 16000 for Silero VAD streaming")

    window_size = _window_size_samples(cfg.sample_rate)

    model, VADIterator, source = _load_silero_vad_iterator(cfg)
    logger.info(f"[audio_vad] Using Silero VAD ({source}), sr={cfg.sample_rate}")

    # Move model if possible.
    try:
        import torch

        if cfg.device and cfg.device != "cpu":
            model = model.to(cfg.device)
    except Exception:
        pass

    # Instantiate iterator; API varies slightly across versions.
    try:
        vad_iter = VADIterator(
            model,
            threshold=float(cfg.threshold),
            sampling_rate=int(cfg.sample_rate),
            min_silence_duration_ms=int(cfg.min_silence_duration_ms),
            speech_pad_ms=int(cfg.speech_pad_ms),
        )
    except TypeError:
        try:
            vad_iter = VADIterator(model, sampling_rate=int(cfg.sample_rate))
        except TypeError:
            vad_iter = VADIterator(model)

    current_start_sample: Optional[int] = None
    segments_samples: List[Tuple[int, int]] = []

    # Stream frames from ffmpeg -> VADIterator
    for x in _iter_audio_frames_ffmpeg(
        media_path,
        sample_rate=cfg.sample_rate,
        window_size_samples=window_size,
        duration_s=duration_s,
        on_progress=on_progress,
    ):
        import torch

        xt = torch.from_numpy(x)
        event = vad_iter(xt, return_seconds=False)

        if not event:
            continue

        if "start" in event:
            start_samp = int(event["start"])
            if current_start_sample is None or start_samp < current_start_sample:
                current_start_sample = start_samp

        if "end" in event:
            end_samp = int(event["end"])
            if current_start_sample is None:
                continue
            if end_samp > current_start_sample:
                segments_samples.append((current_start_sample, end_samp))
            current_start_sample = None

    # Reset states after each audio stream.
    try:
        vad_iter.reset_states()
    except Exception:
        try:
            model.reset_states()
        except Exception:
            pass

    # Close open segment at EOF
    if current_start_sample is not None:
        end_samp = int(round(duration_s * cfg.sample_rate))
        if end_samp > current_start_sample:
            segments_samples.append((current_start_sample, end_samp))

    # Convert to seconds and post-filter
    segments: List[SpeechSegment] = []
    min_len_s = float(cfg.min_speech_duration_ms) / 1000.0

    for a, b in segments_samples:
        start_s = max(0.0, float(a) / cfg.sample_rate)
        end_s = min(duration_s if duration_s > 0 else float(b) / cfg.sample_rate, float(b) / cfg.sample_rate)
        if end_s - start_s < min_len_s:
            continue
        segments.append(SpeechSegment(start_s=start_s, end_s=end_s))

    # Merge overlaps / sort
    segments.sort(key=lambda s: s.start_s)
    merged: List[SpeechSegment] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg.start_s <= prev.end_s + 1e-6:
            merged[-1] = SpeechSegment(start_s=prev.start_s, end_s=max(prev.end_s, seg.end_s))
        else:
            merged.append(seg)

    return merged


def _invert_segments(segments: List[SpeechSegment], duration_s: float) -> List[SpeechSegment]:
    """Return non-speech segments as the complement of speech segments."""
    if duration_s <= 0:
        return []

    gaps: List[SpeechSegment] = []
    t = 0.0
    for seg in segments:
        if seg.start_s > t:
            gaps.append(SpeechSegment(start_s=t, end_s=seg.start_s))
        t = max(t, seg.end_s)
    if t < duration_s:
        gaps.append(SpeechSegment(start_s=t, end_s=duration_s))
    return gaps


def _speech_fraction_timeline(
    speech: List[SpeechSegment],
    *,
    duration_s: float,
    hop_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a compact timeline of 'fraction of the window that is speech'."""
    if duration_s <= 0 or hop_s <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    n = int(math.ceil(duration_s / hop_s))
    times = (np.arange(n, dtype=np.float32) * float(hop_s)).astype(np.float32)
    frac = np.zeros((n,), dtype=np.float32)

    if not speech:
        return times, frac

    for seg in speech:
        i0 = max(0, int(seg.start_s / hop_s))
        i1 = min(n - 1, int(seg.end_s / hop_s))
        for i in range(i0, i1 + 1):
            w0 = float(i) * hop_s
            w1 = w0 + hop_s
            overlap = max(0.0, min(seg.end_s, w1) - max(seg.start_s, w0))
            if overlap <= 0:
                continue
            frac[i] = min(1.0, frac[i] + overlap / hop_s)

    return times, frac


def compute_audio_vad_analysis(
    proj: Project,
    *,
    cfg: VadConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute Silero VAD speech segments and persist to analysis/audio_vad.npz."""
    if not cfg.enabled:
        logger.info("[audio_vad] Disabled via config.")
        return {"disabled": True}

    media_path = Path(proj.audio_source)
    if not media_path.exists():
        raise FileNotFoundError(f"Audio source not found: {media_path}")

    # Duration used for EOF handling + timeline size.
    try:
        duration_s = float(ffprobe_duration_seconds(media_path))
    except Exception as exc:
        logger.warning("[audio_vad] Failed to get duration via ffprobe (%s): %s", media_path, exc)
        duration_s = 0.0
    
    # Helper for progress reporting with optional message
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)

    _report(0.02, "Initializing Silero VAD")

    speech_segments = _segments_from_vad(
        media_path,
        cfg=cfg,
        duration_s=duration_s,
        on_progress=on_progress,
    )

    non_speech_segments = _invert_segments(speech_segments, duration_s)

    times, speech_frac = _speech_fraction_timeline(
        speech_segments,
        duration_s=duration_s,
        hop_s=float(cfg.hop_seconds),
    )

    seg_arr = np.array([s.to_tuple() for s in speech_segments], dtype=np.float32)
    gap_arr = np.array([s.to_tuple() for s in non_speech_segments], dtype=np.float32)

    out_path = getattr(proj, "audio_vad_path", proj.analysis_dir / "audio_vad.npz")

    save_npz(
        out_path,
        speech_segments=seg_arr,
        non_speech_segments=gap_arr,
        times=times.astype(np.float32),
        speech_fraction=speech_frac.astype(np.float32),
        hop_seconds=np.array([float(cfg.hop_seconds)], dtype=np.float32),
        duration_seconds=np.array([float(duration_s)], dtype=np.float32),
        sample_rate=np.array([int(cfg.sample_rate)], dtype=np.int32),
    )

    speech_total = float(sum(s.duration_s for s in speech_segments))
    speech_ratio = float(speech_total / duration_s) if duration_s > 0 else 0.0

    payload: Dict[str, Any] = {
        "video": str(media_path),
        "duration_seconds": duration_s,
        "method": "silero_vad_streaming",
        "config": {
            "sample_rate": int(cfg.sample_rate),
            "hop_seconds": float(cfg.hop_seconds),
            "threshold": float(cfg.threshold),
            "min_silence_duration_ms": int(cfg.min_silence_duration_ms),
            "speech_pad_ms": int(cfg.speech_pad_ms),
            "min_speech_duration_ms": int(cfg.min_speech_duration_ms),
            "use_onnx": bool(cfg.use_onnx),
            "opset_version": int(cfg.opset_version),
            "torch_threads": int(cfg.torch_threads),
            "device": str(cfg.device),
        },
        "speech_segments_count": int(len(speech_segments)),
        "speech_seconds": speech_total,
        "speech_ratio": speech_ratio,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["vad"] = {
            **payload,
            "features_npz": str(out_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    _report(1.0, f"Done ({len(speech_segments)} speech segments, {speech_ratio*100:.0f}% speech)")

    return payload


def load_vad_segments(proj: Project) -> Optional[List[SpeechSegment]]:
    """Load cached VAD speech segments if available."""
    path = getattr(proj, "audio_vad_path", proj.analysis_dir / "audio_vad.npz")
    if not path.exists():
        legacy = proj.analysis_dir / "vad_features.npz"
        if not legacy.exists():
            return None
        path = legacy

    data = load_npz(path)
    seg = data.get("speech_segments")
    if seg is None:
        return None
    seg = np.array(seg, dtype=np.float32)

    out: List[SpeechSegment] = []
    if seg.size == 0:
        return out
    for row in seg.reshape(-1, 2):
        out.append(SpeechSegment(start_s=float(row[0]), end_s=float(row[1])))
    return out


def get_vad_boundaries(speech_segments: List[SpeechSegment]) -> Dict[str, List[float]]:
    """Extract boundary timestamps from VAD speech segments."""
    return {
        "speech_starts": [s.start_s for s in speech_segments],
        "speech_ends": [s.end_s for s in speech_segments],
    }
