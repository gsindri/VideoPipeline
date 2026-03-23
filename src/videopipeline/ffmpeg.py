from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

import numpy as np

from .utils import subprocess_flags as _subprocess_flags


def _prepend_dir_to_path(dir_path: Path) -> None:
    """Prepend a directory to PATH for current process if not already present."""
    p = str(dir_path)
    current = os.environ.get("PATH", "")
    parts = [x for x in current.split(os.pathsep) if x] if current else []
    if p in parts:
        return
    os.environ["PATH"] = p + (os.pathsep + current if current else "")


def _windows_cmd_fallback_path(cmd: str) -> str | None:
    """Best-effort Windows fallback discovery for ffmpeg/ffprobe executables.

    Some launch contexts may not hydrate user PATH/FFMPEG_SHARED_BIN into the
    process environment. We still want a known local install to work.
    """
    if os.name != "nt":
        return None

    exe_name = f"{cmd}.exe"
    candidates: list[Path] = []

    # Prefer explicit env var when present.
    raw = os.environ.get("FFMPEG_SHARED_BIN", "")
    if raw:
        for entry in raw.split(os.pathsep):
            entry = entry.strip().strip('"')
            if entry:
                candidates.append(Path(entry))

    # Common install location used by this repo setup scripts.
    candidates.append(Path(r"C:\Tools\ffmpeg-shared\bin"))

    for directory in candidates:
        exe = directory / exe_name
        if exe.exists():
            _prepend_dir_to_path(directory)
            return str(exe)
    return None


def _require_cmd(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        path = _windows_cmd_fallback_path(cmd)
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
    out = subprocess.check_output(cmd, text=True, **_subprocess_flags()).strip()
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
    out = subprocess.check_output(cmd, text=True, **_subprocess_flags())
    import json

    return json.loads(out)


def ffprobe_video_stream_info(video_path: Path) -> dict:
    _require_cmd("ffprobe")
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-print_format",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True, **_subprocess_flags())
    import json

    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no video streams")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_ffprobe_fps(stream.get("avg_frame_rate") or "") or _parse_ffprobe_fps(stream.get("r_frame_rate") or "")
    if not fps:
        fps = 30.0
    return {"width": width, "height": height, "fps": float(fps)}


def _parse_ffprobe_fps(rate: str) -> float | None:
    if not rate:
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
        except ValueError:
            return None
        if den_f == 0:
            return None
        return num_f / den_f
    try:
        return float(rate)
    except ValueError:
        return None


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


def extract_video_frame_jpeg(
    video_path: Path,
    *,
    output_path: Path,
    time_seconds: float,
    width: Optional[int] = None,
    quality: int = 4,
) -> Path:
    """Extract a single video frame to a JPEG file."""
    _require_cmd("ffmpeg")
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if quality < 2:
        quality = 2
    elif quality > 31:
        quality = 31

    time_seconds = max(0.0, float(time_seconds))

    vf_parts: list[str] = []
    if width is not None:
        vf_parts.append(f"scale={int(width)}:-2")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{time_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        str(quality),
    ]
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    cmd.extend(["-y", str(output_path)])
    subprocess.run(cmd, check=True, **_subprocess_flags())
    return output_path


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
    duration_seconds: float | None = None,
    yield_partial: bool = False,
    pad_final: bool = False,
) -> Iterator[np.ndarray]:
    """
    Stream mono float32 PCM blocks from ffmpeg without loading whole audio into memory.

    Yields numpy arrays of length block_samples.

    By default the final partial block is discarded (to preserve older behavior).
    If `yield_partial=True`, the last short block (if any) is yielded.
    If `pad_final=True` as well, that final block is zero-padded up to
    `block_samples`.
    """
    _require_cmd("ffmpeg")
    video_path = Path(video_path)

    if block_samples <= 0:
        raise ValueError("block_samples must be > 0")

    if duration_seconds is not None:
        try:
            duration_seconds = float(duration_seconds)
        except Exception as exc:
            raise ValueError(f"duration_seconds must be a float, got: {duration_seconds!r}") from exc
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be > 0 when provided")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        *(["-t", str(duration_seconds)] if duration_seconds is not None else []),
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
        **_subprocess_flags(),
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
                if not yield_partial:
                    break

                # Guard against odd byte counts (shouldn't happen, but be safe).
                nbytes = len(raw) - (len(raw) % bytes_per_sample)
                if nbytes <= 0:
                    break
                raw = raw[:nbytes]

                block = np.frombuffer(raw, dtype=params.dtype)
                if pad_final and block.size < (block_samples * params.channels):
                    pad = (block_samples * params.channels) - block.size
                    block = np.pad(block, (0, pad), mode="constant")
                yield block
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


def stream_video_frames_gray(
    video_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    _require_cmd("ffmpeg")
    video_path = Path(video_path)

    if fps <= 0:
        raise ValueError("fps must be > 0")
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be > 0")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        **_subprocess_flags(),
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    frame_bytes = width * height

    try:
        while True:
            raw = _read_exact(proc.stdout, frame_bytes)
            if not raw:
                break
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
            yield frame
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
