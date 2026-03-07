from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np
from .ffmpeg import _require_cmd, ffprobe_duration_seconds
from .project import Project, save_json, update_project
from .utils import subprocess_flags as _subprocess_flags


@dataclass(frozen=True)
class AudioDecodeConfig:
    sample_rate: int = 16000
    chunk_seconds: float = 30.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AudioDecodeConfig":
        d = d or {}
        allowed = {"enabled", "sample_rate", "chunk_seconds"}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"Unknown audio_decode config keys: {sorted(unknown)}")

        cfg = cls(
            sample_rate=int(d.get("sample_rate", 16000)),
            chunk_seconds=float(d.get("chunk_seconds", 30.0)),
        )
        if cfg.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0 (got {cfg.sample_rate})")
        if cfg.chunk_seconds <= 0:
            raise ValueError(f"chunk_seconds must be > 0 (got {cfg.chunk_seconds})")
        return cfg


def _source_signature(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def load_audio_decode_index(proj: Project) -> Optional[Dict[str, Any]]:
    index_path = getattr(proj, "audio_decode_index_path", proj.analysis_dir / "audio_decode" / "index.json")
    if not index_path.exists():
        return None
    try:
        import json

        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_decoded_audio_path(proj: Project) -> Optional[Path]:
    wav_path = getattr(proj, "audio_decode_wav_path", proj.analysis_dir / "audio_decode" / "mono_16k.wav")
    index_path = getattr(proj, "audio_decode_index_path", proj.analysis_dir / "audio_decode" / "index.json")
    if wav_path.exists() and index_path.exists():
        return wav_path
    return None


def stream_decoded_audio_blocks_f32(
    wav_path: Path,
    *,
    block_samples: int,
    expected_sample_rate: Optional[int] = None,
    yield_partial: bool = False,
    pad_final: bool = False,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Iterator[np.ndarray]:
    """Stream float32 mono audio blocks from a decoded PCM16 WAV file."""
    if block_samples <= 0:
        raise ValueError("block_samples must be > 0")

    import wave

    wav_path = Path(wav_path)
    with wave.open(str(wav_path), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
        sample_rate = int(wf.getframerate())
        total_frames = int(wf.getnframes())

        if channels != 1:
            raise RuntimeError(f"decoded WAV must be mono (got channels={channels})")
        if sample_width != 2:
            raise RuntimeError(f"decoded WAV must be PCM16 (got sample_width={sample_width})")
        if expected_sample_rate is not None and int(expected_sample_rate) != sample_rate:
            raise RuntimeError(
                f"decoded WAV sample rate mismatch: expected {expected_sample_rate}, got {sample_rate}"
            )

        done = 0
        last_emit = 0.0
        while True:
            raw = wf.readframes(block_samples)
            if not raw:
                break

            n = len(raw) // 2
            if n <= 0:
                break

            if n < block_samples and not yield_partial:
                break

            x = np.frombuffer(raw[: n * 2], dtype="<i2").astype(np.float32) / 32768.0

            if n < block_samples and pad_final:
                x = np.pad(x, (0, block_samples - n), mode="constant")

            done += n
            if on_progress and total_frames > 0:
                p = min(0.95, 0.05 + 0.9 * (float(done) / float(total_frames)))
                if p - last_emit >= 0.01:
                    last_emit = p
                    on_progress(p)

            yield x

            if n < block_samples:
                break


def _decode_to_wav(source_path: Path, wav_path: Path, sample_rate: int) -> None:
    _require_cmd("ffmpeg")
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(wav_path),
    ]
    subprocess.check_call(cmd, **_subprocess_flags())


def compute_audio_decode_analysis(
    proj: Project,
    *,
    cfg: AudioDecodeConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Decode source media once to mono 16k WAV + index/timebase metadata."""
    source_path = Path(proj.audio_source)
    if not source_path.exists():
        raise FileNotFoundError(f"Audio source not found: {source_path}")

    out_dir = getattr(proj, "audio_decode_dir", proj.analysis_dir / "audio_decode")
    wav_path = getattr(proj, "audio_decode_wav_path", out_dir / "mono_16k.wav")
    index_path = getattr(proj, "audio_decode_index_path", out_dir / "index.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)

    source_sig = _source_signature(source_path)
    existing = load_audio_decode_index(proj)
    if existing and wav_path.exists():
        if (
            int(existing.get("sample_rate", 0)) == int(cfg.sample_rate)
            and float(existing.get("chunk_seconds", 0.0)) == float(cfg.chunk_seconds)
            and existing.get("source_signature") == source_sig
        ):
            _report(1.0, "audio decode already available")
            return existing

    _report(0.05, "decoding audio to mono 16k")
    _decode_to_wav(source_path, wav_path, cfg.sample_rate)
    _report(0.80, "building audio decode index")

    import wave

    with wave.open(str(wav_path), "rb") as wf:
        total_samples = int(wf.getnframes())
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())

    duration_s = 0.0
    try:
        duration_s = float(ffprobe_duration_seconds(wav_path))
    except Exception:
        if sr > 0:
            duration_s = float(total_samples) / float(sr)

    chunk_samples = max(1, int(round(cfg.chunk_seconds * sr)))
    chunks = []
    start = 0
    idx = 0
    while start < total_samples:
        end = min(total_samples, start + chunk_samples)
        chunks.append(
            {
                "index": idx,
                "start_sample": int(start),
                "end_sample": int(end),
                "start_s": float(start / sr),
                "end_s": float(end / sr),
            }
        )
        idx += 1
        start = end

    payload: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "source_signature": source_sig,
        "decoded_wav": str(wav_path.relative_to(proj.project_dir)),
        "sample_rate": sr,
        "channels": channels,
        "duration_seconds": float(duration_s),
        "total_samples": int(total_samples),
        "chunk_seconds": float(cfg.chunk_seconds),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "config": {
            "sample_rate": int(cfg.sample_rate),
            "chunk_seconds": float(cfg.chunk_seconds),
        },
    }

    save_json(index_path, payload)

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["audio_decode"] = {
            "created_at": payload["created_at"],
            "sample_rate": payload["sample_rate"],
            "duration_seconds": payload["duration_seconds"],
            "chunk_seconds": payload["chunk_seconds"],
            "chunk_count": payload["chunk_count"],
            "index_json": str(index_path.relative_to(proj.project_dir)),
            "decoded_wav": payload["decoded_wav"],
        }

    update_project(proj, _upd)

    _report(1.0, "done")
    return payload
