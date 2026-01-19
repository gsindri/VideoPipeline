from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .ffmpeg import _require_cmd, ffprobe_video_stream_info
from .layouts import RectNorm


@dataclass(frozen=True)
class HookTextSpec:
    enabled: bool
    duration_seconds: float = 2.0
    text: Optional[str] = None
    font: str = "auto"
    fontsize: int = 64
    y: int = 120


@dataclass(frozen=True)
class LayoutPipSpec:
    position: str = "top_left"
    margin: int = 40
    width_fraction: float = 0.28
    border_px: int = 6


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
    layout_facecam: Optional[RectNorm] = None
    layout_pip: Optional[LayoutPipSpec] = None
    hook_text: Optional[HookTextSpec] = None


def _escape_path_for_ffmpeg_filter(p: Path) -> str:
    # The subtitles filter expects a string where ':' and '\\' have special meaning.
    # For local paths, simplest is to use absolute path and escape backslashes/colons.
    s = str(p)
    s = s.replace('\\', '\\\\')
    s = s.replace(':', '\\:')
    return s


def filtergraph_for_template(
    template: str,
    width: int,
    height: int,
    *,
    layout_facecam: Optional[RectNorm] = None,
    source_width: Optional[int] = None,
    source_height: Optional[int] = None,
    pip_spec: Optional[LayoutPipSpec] = None,
) -> str:
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

    if template == "vertical_streamer_pip":
        if layout_facecam is None:
            return filtergraph_for_template("vertical_blur", width, height)
        if source_width is None or source_height is None:
            raise ValueError("source_width/source_height required for facecam layouts")

        pip = pip_spec or LayoutPipSpec()
        face_px = layout_facecam.to_pixels(width=source_width, height=source_height)
        pip_w = max(1, int(round(width * float(pip.width_fraction))))
        pip_h = max(1, int(round(pip_w * (face_px.h / max(1, face_px.w)))))
        margin = int(pip.margin)

        pos = pip.position.lower()
        if pos not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
            pos = "top_left"
        x = margin if "left" in pos else max(margin, width - pip_w - margin)
        y = margin if "top" in pos else max(margin, height - pip_h - margin)

        border_px = max(0, int(pip.border_px))
        face_label = "fc"
        if border_px > 0:
            face_label = "fcbox"

        face_chain = (
            f"[0:v]crop={face_px.w}:{face_px.h}:{face_px.x}:{face_px.y},"
            f"scale={pip_w}:-2"
        )
        if border_px > 0:
            face_chain += f",pad=iw+{border_px * 2}:ih+{border_px * 2}:{border_px}:{border_px}:color=black@0.75"
        face_chain += f"[{face_label}]"

        return (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:1[bg];"
            f"[0:v]scale={width}:-2:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1[base];"
            f"{face_chain};"
            f"[base][{face_label}]overlay={x}:{y}:shortest=1"
        )

    if template == "vertical_streamer_split":
        if layout_facecam is None:
            return filtergraph_for_template("vertical_blur", width, height)
        if source_width is None or source_height is None:
            raise ValueError("source_width/source_height required for facecam layouts")

        face_px = layout_facecam.to_pixels(width=source_width, height=source_height)
        top_h = max(1, int(round(height * 0.72)))
        bottom_h = max(1, height - top_h)
        return (
            f"[0:v]scale={width}:-2:force_original_aspect_ratio=decrease,"
            f"crop={width}:{top_h}[game];"
            f"[0:v]crop={face_px.w}:{face_px.h}:{face_px.x}:{face_px.y},"
            f"scale={width}:{bottom_h}:force_original_aspect_ratio=increase,"
            f"crop={width}:{bottom_h}[face];"
            f"[game][face]vstack=inputs=2"
        )

    raise ValueError(f"Unknown template: {template}")


def _escape_drawtext_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _escape_drawtext_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _resolve_hook_font(font: str) -> Optional[Path]:
    if not font or font.lower() == "auto":
        candidates = []
        if os.name == "nt":
            candidates = [
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("C:/Windows/Fonts/arial.ttf"),
            ]
        for cand in candidates:
            if cand.exists():
                return cand
        return None
    p = Path(font)
    return p if p.exists() else None


def build_ffmpeg_command(spec: ExportSpec) -> list[str]:
    _require_cmd("ffmpeg")

    duration = max(0.01, float(spec.end_s - spec.start_s))
    if duration <= 0.01:
        raise ValueError("end_s must be > start_s")

    source_width = None
    source_height = None
    if spec.layout_facecam is not None or spec.template in {"vertical_streamer_pip", "vertical_streamer_split"}:
        info = ffprobe_video_stream_info(spec.video_path)
        source_width = int(info.get("width") or 0)
        source_height = int(info.get("height") or 0)

    vf = filtergraph_for_template(
        spec.template,
        spec.width,
        spec.height,
        layout_facecam=spec.layout_facecam,
        source_width=source_width,
        source_height=source_height,
        pip_spec=spec.layout_pip,
    )

    # Optional subtitles burned-in
    if spec.subtitles_ass is not None:
        subs = _escape_path_for_ffmpeg_filter(spec.subtitles_ass)
        if vf == "null":
            vf = f"subtitles='{subs}'"
        else:
            vf = f"{vf},subtitles='{subs}'"

    if spec.hook_text is not None and spec.hook_text.enabled:
        hook = spec.hook_text
        if hook.text:
            font_path = _resolve_hook_font(hook.font)
            if font_path is not None:
                text = _escape_drawtext_text(hook.text)
                fontfile = _escape_drawtext_value(str(font_path))
                enable = f"lt(t\\,{float(hook.duration_seconds):.2f})"
                draw = (
                    "drawtext="
                    f"fontfile='{fontfile}':"
                    f"text='{text}':"
                    f"fontsize={int(hook.fontsize)}:"
                    "fontcolor=white:borderw=3:bordercolor=black@0.8:"
                    "x=(w-text_w)/2:"
                    f"y={int(hook.y)}:"
                    f"enable='{enable}'"
                )
                if vf == "null":
                    vf = draw
                else:
                    vf = f\"{vf},{draw}\"

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
