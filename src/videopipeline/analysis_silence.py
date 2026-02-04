"""Silence detection using FFmpeg silencedetect.

Detects silence intervals for use as natural cut points in clip shaping.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .ffmpeg import _require_cmd
from .project import Project, save_json, update_project
from .utils import subprocess_flags as _subprocess_flags

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SilenceConfig:
    """Configuration for silence detection."""
    noise_db: float = -30.0  # Silence threshold in dB
    min_duration: float = 0.3  # Minimum silence duration in seconds


@dataclass
class SilenceInterval:
    """A detected silence interval."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, float]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SilenceInterval":
        return cls(start=float(d["start"]), end=float(d["end"]))


def parse_silencedetect_output(stderr: str, duration_s: float = 0.0) -> List[SilenceInterval]:
    """Parse FFmpeg silencedetect filter output.
    
    Example output lines:
        [silencedetect @ 0x...] silence_start: 123.456
        [silencedetect @ 0x...] silence_end: 124.789 | silence_duration: 1.333
    
    Args:
        stderr: FFmpeg stderr output
        duration_s: Video duration (used to close final open interval at EOF)
    """
    silences: List[SilenceInterval] = []
    current_start: Optional[float] = None

    # Pattern for silence_start
    start_pattern = re.compile(r"silence_start:\s*([0-9.]+)")
    # Pattern for silence_end
    end_pattern = re.compile(r"silence_end:\s*([0-9.]+)")

    for line in stderr.splitlines():
        start_match = start_pattern.search(line)
        if start_match:
            current_start = float(start_match.group(1))
            continue

        end_match = end_pattern.search(line)
        if end_match and current_start is not None:
            end_time = float(end_match.group(1))
            silences.append(SilenceInterval(start=current_start, end=end_time))
            current_start = None

    # Handle EOF case: if silence started but never ended, close with duration
    if current_start is not None and duration_s > 0 and current_start < duration_s:
        silences.append(SilenceInterval(start=current_start, end=duration_s))

    return silences


def detect_silence(
    video_path: Path,
    *,
    cfg: SilenceConfig,
    duration_s: float = 0.0,
    on_progress: Optional[Callable[[float], None]] = None,
) -> List[SilenceInterval]:
    """Run FFmpeg silencedetect filter on video.
    
    Args:
        video_path: Path to the video file
        cfg: Silence detection configuration
        duration_s: Video duration (for handling EOF silence)
        on_progress: Optional progress callback
        
    Returns:
        List of SilenceInterval objects
    """
    _require_cmd("ffmpeg")
    
    # Helper for progress reporting with optional message
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)

    _report(0.1, "Starting FFmpeg silence detection")

    # Build FFmpeg command with silencedetect filter
    # -vn skips video decoding for faster processing
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # Skip video decoding
        "-sn",  # Skip subtitles
        "-af", f"silencedetect=noise={cfg.noise_db}dB:d={cfg.min_duration}",
        "-f", "null",
        "-",
    ]

    _report(0.2, "Running FFmpeg silencedetect filter")

    # Run FFmpeg - silencedetect output goes to stderr
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        **_subprocess_flags(),
    )

    _report(0.8, "Parsing silence intervals")

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg silencedetect failed with code {result.returncode}: {result.stderr[:500]}"
        )

    # Parse the stderr output (pass duration for EOF handling)
    silences = parse_silencedetect_output(result.stderr, duration_s=duration_s)

    _report(1.0, f"Found {len(silences)} silence intervals")

    return silences


def compute_silence_analysis(
    proj: Project,
    *,
    cfg: SilenceConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Detect silence intervals and save to project.
    
    Persists:
      - analysis/silence.json
      - project.json -> analysis.silence section
    """
    from .ffmpeg import ffprobe_duration_seconds
    
    video_path = Path(proj.audio_source)  # Use audio_source for fallback during early analysis
    
    # Get duration for EOF handling
    try:
        duration_s = ffprobe_duration_seconds(video_path)
    except Exception as exc:
        logger.warning("[silence] Failed to get duration via ffprobe (%s): %s", video_path, exc)
        duration_s = 0.0
    
    # Helper for progress reporting with optional message
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)

    _report(0.05, "Starting silence detection")

    silences = detect_silence(video_path, cfg=cfg, duration_s=duration_s, on_progress=on_progress)

    # Build payload
    silence_path = proj.analysis_dir / "silence.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "noise_db": cfg.noise_db,
            "min_duration": cfg.min_duration,
        },
        "silences": [s.to_dict() for s in silences],
        "count": len(silences),
    }

    # Save silence.json
    save_json(silence_path, payload)

    # Update project.json
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["silence"] = {
            "created_at": payload["created_at"],
            "config": payload["config"],
            "count": len(silences),
            "silence_json": str(silence_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    _report(1.0, "Done")

    return payload


def load_silence_intervals(proj: Project) -> Optional[List[SilenceInterval]]:
    """Load cached silence intervals if available."""
    silence_path = proj.analysis_dir / "silence.json"
    if not silence_path.exists():
        return None

    data = json.loads(silence_path.read_text(encoding="utf-8"))
    return [SilenceInterval.from_dict(s) for s in data.get("silences", [])]


def get_silence_boundaries(silences: List[SilenceInterval]) -> Dict[str, List[float]]:
    """Extract boundary timestamps from silence intervals.
    
    Returns:
        Dict with:
          - silence_starts: Times where silence begins (good end points)
          - silence_ends: Times where silence ends (good start points)
    """
    return {
        "silence_starts": [s.start for s in silences],
        "silence_ends": [s.end for s in silences],
    }
