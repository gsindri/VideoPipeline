from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .ffmpeg import ffprobe_video_stream_info


@dataclass(frozen=True)
class RectNorm:
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        for name, val in {"x": self.x, "y": self.y, "w": self.w, "h": self.h}.items():
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be numeric")
        if self.w <= 0 or self.h <= 0:
            raise ValueError("w and h must be > 0")
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("x and y must be within [0, 1]")
        if not (0.0 <= self.w <= 1.0 and 0.0 <= self.h <= 1.0):
            raise ValueError("w and h must be within [0, 1]")
        if self.x + self.w > 1.0 + 1e-6 or self.y + self.h > 1.0 + 1e-6:
            raise ValueError("rect must fit within [0, 1] bounds")

    def to_dict(self) -> Dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "w": float(self.w), "h": float(self.h)}

    def to_pixels(self, *, width: int, height: int) -> "RectPixel":
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        x_px = int(round(self.x * width))
        y_px = int(round(self.y * height))
        w_px = max(1, int(round(self.w * width)))
        h_px = max(1, int(round(self.h * height)))
        if x_px + w_px > width:
            w_px = max(1, width - x_px)
        if y_px + h_px > height:
            h_px = max(1, height - y_px)
        return RectPixel(x=x_px, y=y_px, w=w_px, h=h_px)


@dataclass(frozen=True)
class RectPixel:
    x: int
    y: int
    w: int
    h: int


def rect_norm_from_dict(data: Dict[str, Any]) -> RectNorm:
    return RectNorm(
        x=float(data.get("x")),
        y=float(data.get("y")),
        w=float(data.get("w")),
        h=float(data.get("h")),
    )


def rect_norm_to_pixels(video_path: Path, rect: RectNorm) -> RectPixel:
    info = ffprobe_video_stream_info(video_path)
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("invalid source dimensions")
    return rect.to_pixels(width=width, height=height)


def get_facecam_rect(layout_data: Dict[str, Any]) -> Optional[RectNorm]:
    facecam = layout_data.get("facecam")
    if not isinstance(facecam, dict):
        return None
    try:
        return rect_norm_from_dict(facecam)
    except (TypeError, ValueError):
        return None
