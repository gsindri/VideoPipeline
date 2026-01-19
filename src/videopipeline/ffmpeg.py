from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np


def _require_cmd(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        raise RuntimeError(
            f"Required executable '{cmd}' not found in PATH. "
            "Install ffmpeg/ffprobe and ensure they are available on PATH."
        )
    return path


def ffprobe_duration_seconds(video_path: Path) -> float:
    _require_cmd("ffprobe")
    video_path = Path(video_path)

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return float(out)
    except ValueError as e:
        raise RuntimeError(f"ffprobe returned non-numeric duration: {out!r}") from e


def ffprobe_streams(video_path: Path) -> dict:
    """Return ffprobe JSON for streams/format (handy for future work)."""
    _require_cmd("ffprobe")
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    import json

    return json.loads(out)


def _read_exact(stream: BinaryIO, nbytes: int) -> bytes:
    """Read exactly nbytes unless EOF occurs."""
    chunks: list[bytes] = []
    total = 0
    while total < nbytes:
        chunk = stream.read(nbytes - total)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class AudioStreamParams:
    sample_rate: int = 16000
    channels: int = 1
    dtype: np.dtype = np.float32  # f32le => float32 little-endian


def stream_audio_blocks_f32le(
    video_path: Path,
    *,
    params: AudioStreamParams,
    block_samples: int,
) -> Iterator[np.ndarray]:
    """
    Stream mono float32 PCM blocks from ffmpeg without loading whole audio into memory.

    Yields numpy arrays of length block_samples. The final partial block is discarded.
    """
    _require_cmd("ffmpeg")
    video_path = Path(video_path)

    if block_samples <= 0:
        raise ValueError("block_samples must be > 0")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(params.channels),
        "-ar",
        str(params.sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    bytes_per_sample = np.dtype(params.dtype).itemsize
    block_bytes = block_samples * bytes_per_sample * params.channels

    try:
        while True:
            raw = _read_exact(proc.stdout, block_bytes)
            if not raw:
                break
            if len(raw) < block_bytes:
                break
            block = np.frombuffer(raw, dtype=params.dtype)
            yield block
    except Exception:
        proc.kill()
        raise
    finally:
        try:
            proc.stdout.close()  # type: ignore[union-attr]
        except Exception:
            pass

        ret = proc.wait()
        err = b""
        try:
            err = proc.stderr.read()  # type: ignore[union-attr]
            proc.stderr.close()  # type: ignore[union-attr]
        except Exception:
            pass

        if ret != 0:
            msg = err.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed (exit={ret}). {msg}")
