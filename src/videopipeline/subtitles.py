from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class SubtitleSegment:
    start_s: float
    end_s: float
    text: str


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
    fontname: str = "DejaVu Sans",
    fontsize: int = 60,
    margin_l: int = 80,
    margin_r: int = 80,
    margin_v: int = 200,
    outline: int = 3,
    shadow: int = 1,
    alignment: int = 2,
) -> Path:
    """Write a basic ASS subtitle file suitable for burned-in captions.

    Notes:
    - alignment=2 => bottom-center
    - margins help keep text out of platform UI safezones
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {playres_x}
PlayResY: {playres_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},{margin_l},{margin_r},{margin_v},1

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
