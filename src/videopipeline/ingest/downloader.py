"""yt-dlp based video downloader with progress hooks.

Downloads videos from supported URLs (YouTube, Twitch, etc.) with:
- Progress callbacks for UI updates
- Automatic info.json metadata storage
- Optional preview proxy generation for browser playback
- Adaptive concurrency for HLS streams (Twitch optimization)

.. deprecated::
    This module is largely superseded by the newer ingest modules:
    - `ingest.ytdlp_runner` for URL downloads
    - `ingest.models` for data models
    - `ingest.tuner` for adaptive concurrency
    - `ingest.postprocess` for post-processing
    
    Consider using those modules instead for new code.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from ..ffmpeg import _require_cmd
from ..utils import subprocess_flags as _subprocess_flags, utc_iso as _utc_iso


# ---------------------------------------------------------------------------
# Speed Mode / Adaptive Concurrency
# ---------------------------------------------------------------------------

class SpeedMode(str, Enum):
    """Download speed mode for HLS fragment concurrency."""
    BALANCED = "balanced"      # N=8, conservative
    FAST = "fast"              # N=16, good for most connections
    AGGRESSIVE = "aggressive"  # N=32, may trigger throttling
    AUTO = "auto"              # Adaptive based on tuning history


# Default concurrency values per mode
SPEED_MODE_N: dict[SpeedMode, int] = {
    SpeedMode.BALANCED: 8,
    SpeedMode.FAST: 16,
    SpeedMode.AGGRESSIVE: 32,
    SpeedMode.AUTO: 16,  # Starting value for auto
}


def _tuning_file_path() -> Path:
    """Get path to the yt-dlp tuning state file."""
    if os.name == "nt":
        base = os.getenv("APPDATA")
        if base:
            return Path(base) / "VideoPipeline" / "ytdlp_tuning.json"
        return Path.home() / "AppData" / "Roaming" / "VideoPipeline" / "ytdlp_tuning.json"
    return Path.home() / ".config" / "videopipeline" / "ytdlp_tuning.json"


def _load_tuning() -> dict[str, Any]:
    """Load the tuning state file."""
    path = _tuning_file_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_tuning(data: dict[str, Any]) -> None:
    """Save the tuning state file."""
    path = _tuning_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_domain(url: str) -> str:
    """Extract domain from URL for tuning lookup."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        return netloc if netloc else "unknown"
    except Exception:
        return "unknown"


def _load_best_n(domain: str, default: int = 16) -> int:
    """Load the best N value for a domain from tuning history."""
    tuning = _load_tuning()
    site_data = tuning.get(domain, {})
    return site_data.get("N", default)


def _save_best_n(domain: str, n: int, ok: bool) -> None:
    """Save N value and success status for a domain."""
    tuning = _load_tuning()
    if domain not in tuning:
        tuning[domain] = {}
    tuning[domain]["N"] = n
    tuning[domain]["last_ok"] = ok
    tuning[domain]["last_updated"] = _utc_iso()
    _save_tuning(tuning)


def _looks_like_throttle(error: Exception) -> bool:
    """Check if an error looks like throttling/rate limiting."""
    error_str = str(error).lower()
    throttle_indicators = [
        "429",
        "403",
        "too many requests",
        "rate limit",
        "throttl",
        "temporary failure",
        "fragment",
        "unable to download",
        "giving up",
        "retries",
    ]
    return any(indicator in error_str for indicator in throttle_indicators)


@dataclass
class DownloadOptions:
    """Options for URL download."""
    
    # Output settings
    output_dir: Optional[Path] = None
    
    # Download behavior
    no_playlist: bool = True
    best_quality: bool = True
    
    # Speed / concurrency settings
    speed_mode: SpeedMode = SpeedMode.AUTO
    
    # Post-processing
    create_preview: bool = True  # Create H.264/AAC proxy for browser
    preview_height: int = 720    # Preview resolution
    preview_crf: int = 28        # Preview quality (higher = smaller file)


@dataclass
class DownloadResult:
    """Result of a URL download."""
    
    # Paths
    video_path: Path
    info_json_path: Optional[Path] = None
    preview_path: Optional[Path] = None
    
    # Metadata from yt-dlp
    title: str = ""
    url: str = ""
    extractor: str = ""
    video_id: str = ""
    duration_seconds: float = 0.0
    
    # Status
    created_at: str = field(default_factory=_utc_iso)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "info_json_path": str(self.info_json_path) if self.info_json_path else None,
            "preview_path": str(self.preview_path) if self.preview_path else None,
            "title": self.title,
            "url": self.url,
            "extractor": self.extractor,
            "video_id": self.video_id,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
        }


ProgressCallback = Callable[[float, str], None]


def _default_downloads_dir() -> Path:
    """Get default downloads directory."""
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA")
        if base:
            return Path(base) / "VideoPipeline" / "Workspace" / "downloads"
        return Path.home() / "AppData" / "Local" / "VideoPipeline" / "Workspace" / "downloads"
    return Path.home() / ".videopipeline" / "downloads"


def _sanitize_filename(name: str, max_length: int = 100) -> str:
    """Make a filename safe for Windows and other filesystems."""
    # Remove or replace problematic characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'[\x00-\x1f]', '', name)
    name = name.strip('. ')
    
    if len(name) > max_length:
        name = name[:max_length].strip()
    
    return name or "video"


def _needs_preview(video_path: Path) -> bool:
    """Check if video needs a browser-friendly preview.
    
    Returns True if the video is not H.264/AAC in MP4 container.
    """
    try:
        ffprobe = _require_cmd("ffprobe")
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30,
            **_subprocess_flags(),
        )
        
        if result.returncode != 0:
            return True  # If we can't probe, assume we need preview
        
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        
        video_codec = None
        audio_codec = None
        
        for stream in streams:
            codec_type = stream.get("codec_type")
            codec_name = stream.get("codec_name", "").lower()
            
            if codec_type == "video" and video_codec is None:
                video_codec = codec_name
            elif codec_type == "audio" and audio_codec is None:
                audio_codec = codec_name
        
        # Browser-friendly: H.264 video + AAC audio
        is_h264 = video_codec in ("h264", "avc", "avc1")
        is_aac = audio_codec in ("aac", "mp4a") or audio_codec is None  # No audio is ok
        
        # Also check container (should be mp4)
        is_mp4 = video_path.suffix.lower() in (".mp4", ".m4v")
        
        return not (is_h264 and is_aac and is_mp4)
        
    except Exception:
        return True  # If anything fails, create preview to be safe


def _create_preview(
    source_path: Path,
    preview_path: Path,
    height: int = 720,
    crf: int = 28,
    on_progress: Optional[ProgressCallback] = None,
) -> bool:
    """Create a browser-friendly H.264/AAC preview.
    
    Attempts hardware acceleration (AMF > NVENC > QSV > CPU) for faster encoding.
    
    Returns True on success.
    """
    try:
        ffmpeg = _require_cmd("ffmpeg")
        
        if on_progress:
            on_progress(0.95, "Creating browser preview...")
        
        # Scale to target height while maintaining aspect ratio
        # -2 ensures width is divisible by 2 (required for H.264)
        scale_filter = f"scale=-2:{height}"
        
        # Try hardware encoders in order: AMF (AMD) > NVENC (NVIDIA) > QSV (Intel) > CPU
        encoder_configs = [
            # AMD AMF
            {
                "codec": "h264_amf",
                "opts": ["-quality", "speed", "-rc", "vbr_latency", "-qp_i", str(crf), "-qp_p", str(crf)],
            },
            # NVIDIA NVENC
            {
                "codec": "h264_nvenc",
                "opts": ["-preset", "p4", "-rc", "vbr", "-cq", str(crf)],
            },
            # Intel Quick Sync
            {
                "codec": "h264_qsv",
                "opts": ["-preset", "fast", "-global_quality", str(crf)],
            },
            # CPU fallback
            {
                "codec": "libx264",
                "opts": ["-preset", "veryfast", "-crf", str(crf)],
            },
        ]
        
        for config in encoder_configs:
            cmd = [
                ffmpeg, "-y",
                "-i", str(source_path),
                "-c:v", config["codec"],
                *config["opts"],
                "-vf", scale_filter,
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(preview_path),
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                **_subprocess_flags(),
            )
            
            if result.returncode == 0 and preview_path.exists():
                return True
            
            # Clean up partial file before trying next encoder
            if preview_path.exists():
                try:
                    preview_path.unlink()
                except Exception:
                    pass
        
        return False
        
    except Exception:
        return False


def download_url(
    url: str,
    options: Optional[DownloadOptions] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> DownloadResult:
    """Download a video from a URL using yt-dlp.
    
    .. deprecated::
        Use `videopipeline.ingest.ytdlp_runner.download_url` instead.
    
    Args:
        url: The URL to download from
        options: Download options (uses defaults if None)
        on_progress: Callback for progress updates (fraction, message)
    
    Returns:
        DownloadResult with paths and metadata
    
    Raises:
        ImportError: If yt-dlp is not installed
        RuntimeError: If download fails
    """
    warnings.warn(
        "videopipeline.ingest.downloader.download_url is deprecated. "
        "Use videopipeline.ingest.ytdlp_runner.download_url instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise ImportError("yt-dlp is required for URL downloads. Install with: pip install yt-dlp")
    
    options = options or DownloadOptions()
    output_dir = options.output_dir or _default_downloads_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    domain = _get_domain(url)
    
    # Determine initial concurrency based on speed mode
    if options.speed_mode == SpeedMode.AUTO:
        current_n = _load_best_n(domain, default=SPEED_MODE_N[SpeedMode.FAST])
    else:
        current_n = SPEED_MODE_N[options.speed_mode]
    
    min_n = 2
    max_attempts = 3
    
    # Progress hook for yt-dlp
    download_progress = {"phase": "extracting", "last_percent": 0.0}
    
    def progress_hook(d: dict[str, Any]) -> None:
        status = d.get("status")
        
        if status == "downloading":
            download_progress["phase"] = "downloading"
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            
            if total and total > 0:
                percent = downloaded / total
                # Map to 0-0.9 range (leave room for post-processing)
                mapped = percent * 0.9
                download_progress["last_percent"] = mapped
                
                if on_progress:
                    speed = d.get("speed")
                    speed_str = f" ({speed/1024/1024:.1f} MB/s)" if speed else ""
                    n_str = f" [N={current_n}]" if current_n > 1 else ""
                    on_progress(mapped, f"Downloading... {percent:.0%}{speed_str}{n_str}")
            else:
                if on_progress:
                    on_progress(download_progress["last_percent"], "Downloading...")
                    
        elif status == "finished":
            download_progress["phase"] = "postprocessing"
            if on_progress:
                on_progress(0.92, "Download complete. Processing...")
    
    # Retry loop with backoff
    last_error: Optional[Exception] = None
    info: Optional[dict[str, Any]] = None
    
    for attempt in range(max_attempts):
        # Configure yt-dlp with adaptive concurrency
        ydl_opts: dict[str, Any] = {
            "outtmpl": str(output_dir / "%(title).100s [%(id)s].%(ext)s"),
            "noplaylist": options.no_playlist,
            "restrictfilenames": True,  # Windows-safe filenames
            "progress_hooks": [progress_hook],
            "writeinfojson": True,
            "quiet": True,
            "no_warnings": True,
            # HLS optimization for Twitch and other streaming sites
            "hls_prefer_native": True,
            "concurrent_fragment_downloads": current_n,
        }
        
        # Format selection
        if options.best_quality:
            # Best video+audio, prefer mp4
            ydl_opts["format"] = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"
            ydl_opts["merge_output_format"] = "mp4"
        else:
            # Just best combined format
            ydl_opts["format"] = "best[ext=mp4]/best"
        
        if on_progress:
            if attempt == 0:
                on_progress(0.0, f"Extracting video info... (N={current_n})")
            else:
                on_progress(0.0, f"Retrying with N={current_n}...")
        
        # Download
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            
            # Success! Save the working N value for AUTO mode
            if options.speed_mode == SpeedMode.AUTO:
                _save_best_n(domain, current_n, ok=True)
            break
            
        except Exception as e:
            last_error = e
            
            # Check if it looks like throttling
            if _looks_like_throttle(e) and current_n > min_n:
                # Back off: halve concurrency
                old_n = current_n
                current_n = max(min_n, current_n // 2)
                
                if options.speed_mode == SpeedMode.AUTO:
                    _save_best_n(domain, current_n, ok=False)
                
                if on_progress:
                    on_progress(0.0, f"Throttled at N={old_n}, backing off to N={current_n}...")
                continue
            else:
                # Not throttle-related or can't back off further
                raise RuntimeError(f"Download failed: {e}")
    
    if not info:
        raise RuntimeError(f"Download failed after {max_attempts} attempts: {last_error}")
    
    if on_progress:
        on_progress(0.93, "Finding downloaded files...")
    
    # Find the downloaded video file
    video_id = info.get("id", "unknown")
    title = info.get("title", "video")
    
    # yt-dlp may have downloaded to various locations; find the newest mp4/mkv/webm
    video_exts = {".mp4", ".mkv", ".webm", ".m4v", ".avi", ".mov"}
    candidates = []
    for ext in video_exts:
        candidates.extend(output_dir.glob(f"*{ext}"))
    
    if not candidates:
        raise RuntimeError("No video file found after download")
    
    # Sort by modification time, newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    video_path = candidates[0]
    
    # Find info.json
    info_json_path = None
    possible_info_paths = [
        video_path.with_suffix(".info.json"),
        output_dir / f"{video_path.stem}.info.json",
    ]
    for p in possible_info_paths:
        if p.exists():
            info_json_path = p
            break
    
    # Build result
    result = DownloadResult(
        video_path=video_path,
        info_json_path=info_json_path,
        title=title,
        url=url,
        extractor=info.get("extractor", ""),
        video_id=video_id,
        duration_seconds=float(info.get("duration", 0) or 0),
    )
    
    # Create preview if needed
    if options.create_preview and _needs_preview(video_path):
        if on_progress:
            on_progress(0.95, "Creating browser-friendly preview...")
        
        preview_path = video_path.parent / f"{video_path.stem}_preview.mp4"
        
        if _create_preview(
            video_path,
            preview_path,
            height=options.preview_height,
            crf=options.preview_crf,
            on_progress=on_progress,
        ):
            result.preview_path = preview_path
    
    if on_progress:
        on_progress(1.0, "Download complete!")
    
    return result
