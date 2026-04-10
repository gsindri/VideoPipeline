from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from .ffmpeg import _require_cmd, ffprobe_video_stream_info
from .layouts import RectNorm
from .utils import subprocess_flags as _subprocess_flags

logger = logging.getLogger(__name__)


DEFAULT_LAYOUT_PRESET = "proof_overlay"
LAYOUT_PRESET_TEMPLATE_MAP = {
    "original": "original",
    "proof_overlay": "vertical_blur",
    "speaker_broll": "vertical_streamer_pip",
    "reaction_stack": "vertical_streamer_split",
    "single_subject_punch": "vertical_crop_center",
}
LAYOUT_PRESET_ALIASES = {
    "original": "original",
    "proof_overlay": "proof_overlay",
    "proof-overlay": "proof_overlay",
    "vertical_blur": "proof_overlay",
    "speaker_broll": "speaker_broll",
    "speaker-broll": "speaker_broll",
    "vertical_streamer_pip": "speaker_broll",
    "reaction_stack": "reaction_stack",
    "reaction-stack": "reaction_stack",
    "vertical_streamer_split": "reaction_stack",
    "single_subject_punch": "single_subject_punch",
    "single-subject-punch": "single_subject_punch",
    "vertical_crop_center": "single_subject_punch",
}
LAYOUT_PRESETS = tuple(LAYOUT_PRESET_TEMPLATE_MAP.keys())


def normalize_layout_preset(value: str | None, *, default: str = DEFAULT_LAYOUT_PRESET) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_")
    if not raw:
        return default
    normalized = LAYOUT_PRESET_ALIASES.get(raw, raw)
    return normalized if normalized in LAYOUT_PRESET_TEMPLATE_MAP else default


def layout_preset_to_template(value: str | None, *, default: str = DEFAULT_LAYOUT_PRESET) -> str:
    preset = normalize_layout_preset(value, default=default)
    return LAYOUT_PRESET_TEMPLATE_MAP[preset]


class ExportCancelledError(RuntimeError):
    """Raised when an export is cancelled while ffmpeg is running."""


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
class CameraPlanKeyframe:
    at_s: float
    focus_x: float = 0.5
    focus_y: float = 0.5
    zoom: float = 1.0


@dataclass(frozen=True)
class ExportSpec:
    video_path: Path
    start_s: float
    end_s: float
    output_path: Path

    # Template + output
    template: str = "vertical_blur"
    layout_preset: Optional[str] = None
    caption_theme: str = "clean"
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
    camera_plan: Optional[Sequence[CameraPlanKeyframe]] = None
    hook_text: Optional[HookTextSpec] = None


def _escape_path_for_ffmpeg_filter(p: Path) -> str:
    # The subtitles filter expects a string where ':' and '\\' have special meaning.
    # For local paths, simplest is to use absolute path and escape backslashes/colons.
    s = str(p)
    s = s.replace('\\', '\\\\')
    s = s.replace(':', '\\:')
    return s


def measure_loudness(
    video_path: Path,
    start_s: float,
    end_s: float,
    *,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> Optional[dict[str, float]]:
    """Measure audio loudness using ffmpeg loudnorm filter (first pass).

    Returns dict with measured values for second pass, or None on failure.
    """
    _require_cmd("ffmpeg")
    duration = end_s - start_s

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5:print_format=json",
        "-f", "null",
        "-",
    ]

    if on_progress:
        on_progress(0.0, "measuring loudness")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            **_subprocess_flags(),
        )
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            logger.warning("ffmpeg loudnorm (pass 1) failed (exit=%s): %s", result.returncode, err[:800])
            return None
        # loudnorm outputs JSON to stderr
        stderr = result.stderr

        # Find JSON block in output (it's embedded in other ffmpeg output)
        json_match = re.search(r'\{[^{}]*"input_i"[^{}]*\}', stderr, re.DOTALL)
        if not json_match:
            logger.warning("ffmpeg loudnorm (pass 1) did not emit parseable JSON. stderr: %s", (stderr or "")[:800])
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse loudnorm JSON: %s", exc)
            return None

        # Extract the values we need for second pass
        measured = {
            "input_i": float(data.get("input_i", -24)),
            "input_tp": float(data.get("input_tp", -2)),
            "input_lra": float(data.get("input_lra", 7)),
            "input_thresh": float(data.get("input_thresh", -34)),
            "target_offset": float(data.get("target_offset", 0)),
        }

        if on_progress:
            on_progress(1.0, "loudness measured")

        return measured
    except Exception as exc:
        logger.warning("Loudness measurement failed, falling back to single-pass loudnorm: %s", exc, exc_info=True)
        return None


def filtergraph_for_template(
    template: str,
    width: int,
    height: int,
    *,
    layout_facecam: Optional[RectNorm] = None,
    source_width: Optional[int] = None,
    source_height: Optional[int] = None,
    pip_spec: Optional[LayoutPipSpec] = None,
    camera_plan: Optional[Sequence[CameraPlanKeyframe]] = None,
) -> str:
    """Return ffmpeg filtergraph for a given layout template."""
    template = layout_preset_to_template(template)

    if template == "original":
        # No layout change.
        return "null"

    if template == "vertical_blur":
        fg_chain = "[0:v]"
        if camera_plan:
            fg_chain += f"{_camera_plan_crop_filter(camera_plan, mode='source_aspect', target_width=width, target_height=height)},"
        # Gaming-friendly vertical: blurred background + full-width 16:9 foreground.
        # Background fills 9:16, then we overlay the 16:9 gameplay centered.
        # Note: boxblur values are intentionally modest for speed.
        return (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:1[bg];"
            f"{fg_chain}scale={width}:-2:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1"
        )

    if template == "vertical_crop_center":
        if camera_plan:
            crop_filter = _camera_plan_crop_filter(
                camera_plan,
                mode="target_aspect",
                target_width=width,
                target_height=height,
            )
            return f"[0:v]{crop_filter},scale={width}:{height}"
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

        fg_chain = "[0:v]"
        if camera_plan:
            fg_chain += f"{_camera_plan_crop_filter(camera_plan, mode='source_aspect', target_width=width, target_height=height)},"

        return (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:1[bg];"
            f"{fg_chain}scale={width}:-2:force_original_aspect_ratio=decrease[fg];"
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
        game_chain = "[0:v]"
        if camera_plan:
            game_chain += f"{_camera_plan_crop_filter(camera_plan, mode='source_aspect', target_width=width, target_height=height)},"
        return (
            f"{game_chain}scale={width}:-2:force_original_aspect_ratio=decrease,"
            f"crop={width}:{top_h}[game];"
            f"[0:v]crop={face_px.w}:{face_px.h}:{face_px.x}:{face_px.y},"
            f"scale={width}:{bottom_h}:force_original_aspect_ratio=increase,"
            f"crop={width}:{bottom_h}[face];"
            f"[game][face]vstack=inputs=2"
        )

    raise ValueError(f"Unknown template: {template}")


def _escape_drawtext_text(text: str) -> str:
    # Keep overlay text as a single line; line breaks often cause filter parse surprises.
    text = str(text).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("%", "\\%")
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


def _parse_numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value == value:
            return None
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _first_numeric(raw_item: dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in raw_item:
            continue
        parsed = _parse_numeric(raw_item.get(key))
        if parsed is not None:
            return parsed
    return None


def normalize_camera_plan(
    value: Any,
    *,
    duration_s: float | None = None,
    max_keyframes: int = 8,
) -> tuple[CameraPlanKeyframe, ...]:
    if not isinstance(value, list):
        return ()

    normalized: dict[int, CameraPlanKeyframe] = {}
    clip_duration = max(0.0, float(duration_s)) if duration_s is not None else None

    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue

        at_s = _first_numeric(raw_item, ("at_s", "atSeconds", "at", "time_s", "timeSeconds", "t"))
        if at_s is None or at_s < 0.0:
            continue
        if clip_duration is not None:
            at_s = min(at_s, clip_duration)

        focus_x = _first_numeric(raw_item, ("focus_x", "focusX", "center_x", "centerX", "x"))
        focus_y = _first_numeric(raw_item, ("focus_y", "focusY", "center_y", "centerY", "y"))
        zoom = _first_numeric(raw_item, ("zoom", "scale"))

        key = int(round(at_s * 1000.0))
        normalized[key] = CameraPlanKeyframe(
            at_s=float(at_s),
            focus_x=_clamp(0.5 if focus_x is None else focus_x, 0.0, 1.0),
            focus_y=_clamp(0.5 if focus_y is None else focus_y, 0.0, 1.0),
            zoom=_clamp(1.0 if zoom is None else zoom, 1.0, 4.0),
        )

    ordered = [normalized[key] for key in sorted(normalized.keys())[:max_keyframes]]
    return tuple(ordered)


def camera_plan_to_dicts(
    value: Optional[Sequence[CameraPlanKeyframe] | Iterable[CameraPlanKeyframe]],
) -> list[dict[str, float]]:
    if not value:
        return []
    return [
        {
            "at_s": float(frame.at_s),
            "focus_x": float(frame.focus_x),
            "focus_y": float(frame.focus_y),
            "zoom": float(frame.zoom),
        }
        for frame in value
    ]


def _fmt_expr_number(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _camera_plan_value_expr(camera_plan: Sequence[CameraPlanKeyframe], field: str) -> str:
    frames = list(camera_plan)
    if not frames:
        raise ValueError("camera plan is empty")
    if len(frames) == 1:
        return _fmt_expr_number(float(getattr(frames[0], field)))

    expr = _fmt_expr_number(float(getattr(frames[-1], field)))
    for idx in range(len(frames) - 2, -1, -1):
        left = frames[idx]
        right = frames[idx + 1]
        left_value = _fmt_expr_number(float(getattr(left, field)))
        right_value = _fmt_expr_number(float(getattr(right, field)))
        left_at = _fmt_expr_number(left.at_s)
        right_at = _fmt_expr_number(right.at_s)
        delta = max(1e-6, float(right.at_s - left.at_s))
        interp = (
            f"({left_value}+({right_value}-{left_value})*clip((t-{left_at})/{_fmt_expr_number(delta)},0,1))"
        )
        expr = f"if(lte(t,{right_at}),{interp},{expr})"
    return expr


def _camera_plan_crop_filter(
    camera_plan: Sequence[CameraPlanKeyframe],
    *,
    mode: str,
    target_width: int,
    target_height: int,
) -> str:
    if not camera_plan:
        return ""

    focus_x_expr = _camera_plan_value_expr(camera_plan, "focus_x")
    focus_y_expr = _camera_plan_value_expr(camera_plan, "focus_y")
    zoom_expr = _camera_plan_value_expr(camera_plan, "zoom")

    if mode == "source_aspect":
        crop_w_expr = f"(iw/({zoom_expr}))"
        crop_h_expr = f"(ih/({zoom_expr}))"
    elif mode == "target_aspect":
        target_ar = _fmt_expr_number(float(target_width) / max(1.0, float(target_height)))
        crop_w_expr = f"if(gte(iw/ih,{target_ar}),ih*{target_ar}/({zoom_expr}),iw/({zoom_expr}))"
        crop_h_expr = f"if(gte(iw/ih,{target_ar}),ih/({zoom_expr}),iw/{target_ar}/({zoom_expr}))"
    else:
        raise ValueError(f"Unknown camera plan crop mode: {mode}")

    x_expr = f"clip(iw*({focus_x_expr})-({crop_w_expr})/2,0,iw-({crop_w_expr}))"
    y_expr = f"clip(ih*({focus_y_expr})-({crop_h_expr})/2,0,ih-({crop_h_expr}))"
    return f"crop=w='{crop_w_expr}':h='{crop_h_expr}':x='{x_expr}':y='{y_expr}'"


def _build_video_filtergraph(spec: ExportSpec) -> str:
    resolved_template = layout_preset_to_template(spec.layout_preset or spec.template)
    source_width = None
    source_height = None
    if spec.layout_facecam is not None or resolved_template in {"vertical_streamer_pip", "vertical_streamer_split"}:
        info = ffprobe_video_stream_info(spec.video_path)
        source_width = int(info.get("width") or 0)
        source_height = int(info.get("height") or 0)

    vf = filtergraph_for_template(
        resolved_template,
        spec.width,
        spec.height,
        layout_facecam=spec.layout_facecam,
        source_width=source_width,
        source_height=source_height,
        pip_spec=spec.layout_pip,
        camera_plan=tuple(spec.camera_plan or ()),
    )

    # Optional subtitles burned-in.
    if spec.subtitles_ass is not None:
        subs = _escape_path_for_ffmpeg_filter(spec.subtitles_ass)
        if vf == "null":
            vf = f"subtitles='{subs}'"
        else:
            vf = f"{vf},subtitles='{subs}'"

    # Optional hook text overlay.
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
                    "expansion=none:"
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
                    vf = f"{vf},{draw}"

    return vf


def _append_video_filters(cmd: list[str], vf: str, *, map_audio: bool) -> None:
    if vf == "null":
        return
    use_complex = ";" in vf
    if use_complex:
        # Label final output via an explicit pass-through filter. Appending
        # "[vout]" directly to the previous filter can be misparsed when the
        # preceding filter has quoted expressions (e.g. drawtext enable=...).
        cmd += ["-filter_complex", f"{vf},null[vout]", "-map", "[vout]"]
        if map_audio:
            cmd += ["-map", "0:a?"]
        return
    cmd += ["-vf", vf]


def _preflight_filtergraph(spec: ExportSpec) -> None:
    """Validate that the generated video filter graph parses before full encode."""
    _require_cmd("ffmpeg")
    vf = _build_video_filtergraph(spec)
    if vf == "null":
        return

    duration = max(0.2, min(1.5, float(spec.end_s - spec.start_s)))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-ss",
        f"{spec.start_s:.3f}",
        "-i",
        str(spec.video_path),
        "-t",
        f"{duration:.3f}",
    ]
    _append_video_filters(cmd, vf, map_audio=False)
    cmd += ["-an", "-frames:v", "1", "-f", "null", "-"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        **_subprocess_flags(),
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg filtergraph validation failed (exit={result.returncode}). {err[:1200]}")


def build_ffmpeg_command(spec: ExportSpec) -> list[str]:
    _require_cmd("ffmpeg")

    duration = max(0.01, float(spec.end_s - spec.start_s))
    if duration <= 0.01:
        raise ValueError("end_s must be > start_s")

    vf = _build_video_filtergraph(spec)

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
    _append_video_filters(cmd, vf, map_audio=True)

    cmd += ["-r", str(spec.fps)]

    # Audio normalization (optional) - supports two-pass with measured values
    if spec.normalize_audio:
        if hasattr(spec, '_loudness_measured') and spec._loudness_measured:
            m = spec._loudness_measured
            cmd += [
                "-af",
                f"loudnorm=I=-16:LRA=11:TP=-1.5:"
                f"measured_I={m['input_i']}:"
                f"measured_LRA={m['input_lra']}:"
                f"measured_TP={m['input_tp']}:"
                f"measured_thresh={m['input_thresh']}:"
                f"offset={m['target_offset']}:"
                f"linear=true"
            ]
        else:
            # Fallback to single-pass
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
    two_pass_loudnorm: bool = True,
    check_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Run ffmpeg export, optionally reporting progress (0..1).

    Args:
        spec: Export specification
        on_progress: Callback for progress updates (fraction, status)
        two_pass_loudnorm: If True and normalize_audio is enabled, use two-pass
            loudness normalization for more accurate results. Adds ~30-50% time.
        check_cancel: Optional callback returning True when the export should stop.
    """
    duration = max(0.01, float(spec.end_s - spec.start_s))

    if check_cancel and check_cancel():
        raise ExportCancelledError("export cancelled")

    if on_progress:
        on_progress(0.0, "validating filter graph")
    _preflight_filtergraph(spec)

    if check_cancel and check_cancel():
        raise ExportCancelledError("export cancelled")

    # Two-pass loudness normalization: measure first, then encode with measured values
    if spec.normalize_audio and two_pass_loudnorm:
        if on_progress:
            on_progress(0.0, "measuring loudness (pass 1/2)")

        measured = measure_loudness(
            spec.video_path,
            spec.start_s,
            spec.end_s,
        )

        if measured:
            # Attach measured values to spec for build_ffmpeg_command
            # Using object.__setattr__ since ExportSpec is frozen
            object.__setattr__(spec, '_loudness_measured', measured)
        else:
            logger.warning(
                "Two-pass loudnorm requested but pass 1 failed; using single-pass loudnorm for this export."
            )

        if check_cancel and check_cancel():
            raise ExportCancelledError("export cancelled")

    cmd = build_ffmpeg_command(spec)

    if on_progress:
        status = "encoding (pass 2/2)" if (spec.normalize_audio and two_pass_loudnorm) else "encoding"
        on_progress(0.0, status)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        **_subprocess_flags(),
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    out_time_ms = 0
    cancel_event = threading.Event()
    monitor_stop = threading.Event()
    monitor_thread: Optional[threading.Thread] = None

    def _terminate_proc() -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass

    def _monitor_cancel() -> None:
        while not monitor_stop.wait(0.1):
            try:
                should_cancel = bool(check_cancel and check_cancel())
            except Exception:
                should_cancel = False
            if not should_cancel:
                continue
            cancel_event.set()
            _terminate_proc()
            return

    if check_cancel:
        monitor_thread = threading.Thread(target=_monitor_cancel, daemon=True)
        monitor_thread.start()

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
        if cancel_event.is_set() or (check_cancel and check_cancel()):
            raise ExportCancelledError("export cancelled")
        if ret != 0:
            err = proc.stderr.read().strip()
            raise RuntimeError(f"ffmpeg export failed (exit={ret}). {err}")
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.stderr.close()
        except Exception:
            pass
