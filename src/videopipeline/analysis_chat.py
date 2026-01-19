from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from .ffmpeg import ffprobe_duration_seconds
from .peaks import moving_average, robust_z
from .project import Project, save_npz, update_project


_TS_KEYS = (
    "timestamp",
    "time",
    "ts",
    "t",
    "offset",
    "offset_seconds",
    "seconds",
    "timestamp_ms",
    "time_ms",
    "offset_ms",
)


def _parse_hhmmss(val: str) -> Optional[float]:
    parts = val.strip().split(":")
    if len(parts) < 2:
        return None
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        return None
    sec = 0.0
    for p in parts_f:
        sec = sec * 60.0 + p
    return sec


def _parse_timestamp(val: Any, key: str) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        seconds = float(val)
        if "ms" in key or seconds > 1e7:
            seconds /= 1000.0
        return seconds
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            seconds = float(val)
            if "ms" in key or seconds > 1e7:
                seconds /= 1000.0
            return seconds
        except ValueError:
            pass
        hhmmss = _parse_hhmmss(val)
        if hhmmss is not None:
            return hhmmss
        try:
            dt = datetime.fromisoformat(val)
        except ValueError:
            return None
        return dt.timestamp()
    return None


def _extract_timestamps(messages: Iterable[Dict[str, Any]]) -> list[float]:
    times: list[float] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for key in _TS_KEYS:
            if key in msg:
                t = _parse_timestamp(msg.get(key), key)
                if t is not None:
                    times.append(float(t))
                    break
    return times


def _load_messages(path: Path) -> list[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    if isinstance(data, dict):
        for key in ("messages", "chat", "items"):
            if key in data and isinstance(data[key], list):
                return [m for m in data[key] if isinstance(m, dict)]
    return []


def compute_chat_analysis(
    proj: Project,
    *,
    chat_path: Path,
    hop_s: float,
    smooth_s: float,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    chat_path = Path(chat_path)
    if not chat_path.exists():
        raise FileNotFoundError(f"Chat JSON not found: {chat_path}")

    messages = _load_messages(chat_path)
    timestamps = _extract_timestamps(messages)

    duration_s = ffprobe_duration_seconds(proj.video_path)
    if hop_s <= 0:
        raise ValueError("hop_s must be > 0")

    n = int(duration_s / hop_s) + 1
    counts = np.zeros(n, dtype=np.float64)
    for t in timestamps:
        if t < 0 or t > duration_s:
            continue
        idx = int(t / hop_s)
        counts[idx] += 1.0

    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    smoothed = moving_average(counts, smooth_frames) if len(counts) > 0 else counts
    scores = robust_z(smoothed) if len(smoothed) > 0 else smoothed

    save_npz(
        proj.chat_features_path,
        counts=counts,
        smoothed=smoothed,
        scores=scores,
        hop_seconds=np.array([hop_s], dtype=np.float64),
    )

    payload = {
        "method": "chat_message_rate",
        "config": {
            "hop_seconds": hop_s,
            "smooth_seconds": smooth_s,
            "messages_total": len(messages),
        },
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["chat"] = {
            **payload,
            "features_npz": str(proj.chat_features_path.relative_to(proj.project_dir)),
            "raw_json": str(proj.chat_raw_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload
