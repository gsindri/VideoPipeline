from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class SubtitleSegment:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class AssStyleSpec:
    fontname: str = "DejaVu Sans"
    fontsize: int = 60
    margin_l: int = 80
    margin_r: int = 80
    margin_v: int = 200
    outline: int = 3
    shadow: int = 1
    alignment: int = 2
    bold: int = -1
    italic: int = 0
    border_style: int = 1
    primary_colour: str = "&H00FFFFFF"
    secondary_colour: str = "&H000000FF"
    outline_colour: str = "&H00000000"
    back_colour: str = "&H64000000"


DEFAULT_CAPTION_THEME = "clean"
CAPTION_THEME_ALIASES = {
    "default": DEFAULT_CAPTION_THEME,
    "classic": DEFAULT_CAPTION_THEME,
    "clean": DEFAULT_CAPTION_THEME,
    "impact": "impact",
    "punch": "impact",
    "boxed": "boxed",
    "highlight_box": "boxed",
}
CAPTION_THEMES = ("clean", "impact", "boxed")


def normalize_caption_theme(theme: str | None, *, default: str = DEFAULT_CAPTION_THEME) -> str:
    raw = str(theme or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return default
    normalized = CAPTION_THEME_ALIASES.get(raw, raw)
    return normalized if normalized in CAPTION_THEMES else default


def _scale_style_value(value: int, *, playres_y: int) -> int:
    return max(1, int(round(int(value) * (float(playres_y) / 1920.0))))


def caption_theme_style(theme: str | None, *, playres_y: int = 1920) -> AssStyleSpec:
    normalized = normalize_caption_theme(theme)
    base = AssStyleSpec(
        fontsize=_scale_style_value(60, playres_y=playres_y),
        margin_l=_scale_style_value(88, playres_y=playres_y),
        margin_r=_scale_style_value(88, playres_y=playres_y),
        margin_v=_scale_style_value(220, playres_y=playres_y),
        outline=_scale_style_value(4, playres_y=playres_y),
        shadow=0,
    )

    if normalized == "impact":
        return AssStyleSpec(
            fontname=base.fontname,
            fontsize=_scale_style_value(70, playres_y=playres_y),
            margin_l=base.margin_l,
            margin_r=base.margin_r,
            margin_v=_scale_style_value(240, playres_y=playres_y),
            outline=_scale_style_value(5, playres_y=playres_y),
            shadow=_scale_style_value(1, playres_y=playres_y),
            alignment=base.alignment,
            bold=-1,
            italic=0,
            border_style=1,
            primary_colour="&H00FFFFFF",
            secondary_colour="&H000000FF",
            outline_colour="&H00000000",
            back_colour="&H32000000",
        )

    if normalized == "boxed":
        return AssStyleSpec(
            fontname=base.fontname,
            fontsize=_scale_style_value(58, playres_y=playres_y),
            margin_l=_scale_style_value(96, playres_y=playres_y),
            margin_r=_scale_style_value(96, playres_y=playres_y),
            margin_v=_scale_style_value(250, playres_y=playres_y),
            outline=0,
            shadow=0,
            alignment=base.alignment,
            bold=-1,
            italic=0,
            border_style=3,
            primary_colour="&H00FFFFFF",
            secondary_colour="&H000000FF",
            outline_colour="&H20000000",
            back_colour="&H64000000",
        )

    return base


def _ass_time(seconds: float) -> str:
    # ASS format: h:mm:ss.cs (centiseconds)
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60.0
    cs = int(round((s - int(s)) * 100))
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def write_ass(
    segments: Iterable[SubtitleSegment],
    out_path: Path,
    *,
    playres_x: int = 1080,
    playres_y: int = 1920,
    theme: str = DEFAULT_CAPTION_THEME,
    fontname: str | None = None,
    fontsize: int | None = None,
    margin_l: int | None = None,
    margin_r: int | None = None,
    margin_v: int | None = None,
    outline: int | None = None,
    shadow: int | None = None,
    alignment: int | None = None,
) -> Path:
    """Write a basic ASS subtitle file suitable for burned-in captions.

    Notes:
    - alignment=2 => bottom-center
    - margins help keep text out of platform UI safezones
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    style = caption_theme_style(theme, playres_y=playres_y)
    fontname = fontname or style.fontname
    fontsize = int(fontsize if fontsize is not None else style.fontsize)
    margin_l = int(margin_l if margin_l is not None else style.margin_l)
    margin_r = int(margin_r if margin_r is not None else style.margin_r)
    margin_v = int(margin_v if margin_v is not None else style.margin_v)
    outline = int(outline if outline is not None else style.outline)
    shadow = int(shadow if shadow is not None else style.shadow)
    alignment = int(alignment if alignment is not None else style.alignment)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {playres_x}
PlayResY: {playres_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},{style.primary_colour},{style.secondary_colour},{style.outline_colour},{style.back_colour},{style.bold},{style.italic},0,0,100,100,0,0,{style.border_style},{outline},{shadow},{alignment},{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines: List[str] = [header]
    for seg in segments:
        text = seg.text.replace("\n", " ").strip()
        if not text:
            continue
        # ASS line breaks use \N
        text = text.replace("\\", "\\\\")
        start = _ass_time(seg.start_s)
        end = _ass_time(seg.end_s)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
