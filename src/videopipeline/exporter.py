from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .ffmpeg import _require_cmd


@dataclass(frozen=True)
class ExportSpec:
    video_path: Path
    start_s: float
    end_s: float
    output_path: Path

    # Template + output
    template: str = "vertical_blur"
    width: int = 1080
    height: int = 1920
    fps: int = 30

    # Encode
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: int = 20
    acodec: str = "aac"
    abitrate: str = "128k"

    # Optional extras
    subtitles_ass: Optional[Path] = None
    normalize_audio: bool = False


def _escape_path_for_ffmpeg_filter(p: Path) -> str:
    # The subtitles filter expects a string where ':' and '\\' have special meaning.
    # For local paths, simplest is to use absolute path and escape backslashes/colons.
    s = str(p)
    s = s.replace('\\', '\\\\')
    s = s.replace(':', '\\:')
    return s


def filtergraph_for_template(template: str, width: int, height: int) -> str:
    """Return ffmpeg filtergraph for a given layout template."""
    template = template.lower().strip()

    if template == "original":
        # No layout change.
        return "null"

    if template == "vertical_blur":
        # Gaming-friendly vertical: blurred background + full-width 16:9 foreground.
        # Background fills 9:16, then we overlay the 16:9 gameplay centered.
        # Note: boxblur values are intentionally modest for speed.
        return (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:1[bg];"
            f"[0:v]scale={width}:-2:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1"
        )

    if template == "vertical_crop_center":
        # Aggressive: crop center to 9:16 (can cut off HUD/text in many games).
        return (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )

    raise ValueError(f"Unknown template: {template}")


def build_ffmpeg_command(spec: ExportSpec) -> list[str]:
    _require_cmd("ffmpeg")

    duration = max(0.01, float(spec.end_s - spec.start_s))
    if duration <= 0.01:
        raise ValueError("end_s must be > start_s")

    vf = filtergraph_for_template(spec.template, spec.width, spec.height)

    # Optional subtitles burned-in
    if spec.subtitles_ass is not None:
        subs = _escape_path_for_ffmpeg_filter(spec.subtitles_ass)
        if vf == "null":
            vf = f"subtitles='{subs}'"
        else:
            vf = f"{vf},subtitles='{subs}'"

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{spec.start_s:.3f}",
        "-i",
        str(spec.video_path),
        "-t",
        f"{duration:.3f}",
    ]

    # Filters & frame rate
    if vf != "null":
        cmd += ["-vf", vf]

    cmd += ["-r", str(spec.fps)]

    # Audio normalization (optional)
    if spec.normalize_audio:
        cmd += ["-af", "loudnorm=I=-16:LRA=11:TP=-1.5"]

    # Encoding
    cmd += [
        "-c:v",
        spec.vcodec,
        "-preset",
        spec.preset,
        "-crf",
        str(spec.crf),
        "-c:a",
        spec.acodec,
        "-b:a",
        spec.abitrate,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]

    # Progress output for parsers
    cmd += ["-progress", "pipe:1", "-nostats"]

    # Output
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd.append(str(spec.output_path))

    return cmd


def run_ffmpeg_export(
    spec: ExportSpec,
    *,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> None:
    """Run ffmpeg export, optionally reporting progress (0..1)."""
    cmd = build_ffmpeg_command(spec)
    duration = max(0.01, float(spec.end_s - spec.start_s))

    if on_progress:
        on_progress(0.0, "starting")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    out_time_ms = 0

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            # ffmpeg -progress emits key=value
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()

                if k == "out_time_ms":
                    try:
                        out_time_ms = int(v)
                        frac = min(1.0, max(0.0, (out_time_ms / 1_000_000.0) / duration))
                        if on_progress:
                            on_progress(frac, "encoding")
                    except ValueError:
                        pass
                elif k == "progress" and v == "end":
                    if on_progress:
                        on_progress(1.0, "done")

        ret = proc.wait()
        if ret != 0:
            err = proc.stderr.read().strip()
            raise RuntimeError(f"ffmpeg export failed (exit={ret}). {err}")
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.stderr.close()
        except Exception:
            pass
