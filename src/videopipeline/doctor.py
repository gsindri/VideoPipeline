from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .utils import subprocess_flags as _subprocess_flags


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    checks: Dict[str, Dict[str, object]]


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _version(cmd: str) -> str:
    try:
        out = subprocess.check_output([cmd, "-version"], text=True, stderr=subprocess.STDOUT, **_subprocess_flags())
        return out.splitlines()[0].strip()
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def run_doctor() -> DoctorReport:
    checks: Dict[str, Dict[str, object]] = {}

    ffmpeg_path = _which("ffmpeg")
    ffprobe_path = _which("ffprobe")

    checks["ffmpeg"] = {
        "found": ffmpeg_path is not None,
        "path": ffmpeg_path,
        "version": _version("ffmpeg") if ffmpeg_path else None,
    }
    checks["ffprobe"] = {
        "found": ffprobe_path is not None,
        "path": ffprobe_path,
        "version": _version("ffprobe") if ffprobe_path else None,
    }

    whisper_spec = importlib.util.find_spec("faster_whisper")
    if whisper_spec is not None:
        checks["faster_whisper"] = {"installed": True}
    else:
        checks["faster_whisper"] = {
            "installed": False,
            "note": "Install with: pip install -e '.[whisper]'",
        }

    ok = bool(checks["ffmpeg"]["found"] and checks["ffprobe"]["found"])
    return DoctorReport(ok=ok, checks=checks)
