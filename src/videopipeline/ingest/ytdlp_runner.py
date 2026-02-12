"""yt-dlp download runner with adaptive concurrency.

This is the main entry point for downloading videos with:
- Site-specific policies
- Adaptive fragment concurrency for HLS
- Automatic retry with backoff on throttling
- Progress hooks for UI updates
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .models import (
    IngestRequest,
    IngestResult,
    QualityCap,
    SiteType,
    SpeedMode,
    SPEED_MODE_N,
)
from .policy import classify_url_heuristic, get_format_selector, get_policy, probe_url
from .postprocess import postprocess_download
from .tuner import (
    calculate_backoff_n,
    extract_domain,
    get_domain_tuning,
    looks_like_throttle,
    update_domain_tuning,
)


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]  # Returns True if cancelled


class DownloadCancelled(Exception):
    """Raised when download is cancelled by user."""
    pass


class _NoopYtDlpLogger:
    def debug(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


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


def download_url(
    url: str,
    request: Optional[IngestRequest] = None,
    output_dir: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
    check_cancel: Optional[CancelCallback] = None,
) -> IngestResult:
    """Download a video from a URL using yt-dlp.
    
    This is the main entry point with all the smart features:
    - Site detection and policy selection
    - Adaptive concurrency for HLS (Twitch)
    - Automatic retry with backoff on throttling
    - Post-processing (remux, preview)
    
    Args:
        url: The URL to download
        request: Download options (uses defaults if None)
        output_dir: Override output directory
        on_progress: Callback for progress updates (fraction, message)
        check_cancel: Callback that returns True if download should be cancelled
    
    Returns:
        IngestResult with paths and metadata
    
    Raises:
        ImportError: If yt-dlp is not installed
        RuntimeError: If download fails after all retries
        DownloadCancelled: If download was cancelled via check_cancel
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise ImportError("yt-dlp is required for URL downloads. Install with: pip install yt-dlp")
    
    request = request or IngestRequest(url=url)
    output_dir = output_dir or _default_downloads_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Classify the URL
    if on_progress:
        on_progress(0.0, "Detecting site...")
    
    site_type = classify_url_heuristic(url)
    policy = get_policy(site_type)
    domain = extract_domain(url)
    
    # Step 2: Determine concurrency
    if policy.supports_fragment_concurrency:
        if request.speed_mode == SpeedMode.AUTO:
            tuning = get_domain_tuning(domain)
            current_n = tuning.N
            min_n = tuning.min_N
        else:
            current_n = SPEED_MODE_N.get(request.speed_mode, 8)
            min_n = 2
    else:
        current_n = 1
        min_n = 1
    
    max_attempts = 3
    last_error: Optional[Exception] = None
    info: Optional[dict[str, Any]] = None
    
    # Progress tracking
    download_progress = {"phase": "init", "last_percent": 0.0, "cancelled": False}
    partial_files: list[Path] = []  # Track partial downloads for cleanup
    
    def progress_hook(d: dict[str, Any]) -> None:
        # Check for cancellation first
        if check_cancel and check_cancel():
            download_progress["cancelled"] = True
            # Track partial file for cleanup
            filename = d.get("filename")
            if filename:
                partial_files.append(Path(filename))
            # Raise yt-dlp's expected exception to abort download
            from yt_dlp.utils import DownloadCancelled as YtDlpCancelled
            raise YtDlpCancelled("Download cancelled by user")
        
        status = d.get("status")
        
        if status == "downloading":
            download_progress["phase"] = "downloading"
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            
            if total and total > 0:
                percent = downloaded / total
                # Map to 0-0.7 range (leave room for post-processing)
                mapped = percent * 0.7
                download_progress["last_percent"] = mapped
                
                if on_progress:
                    speed = d.get("speed")
                    eta = d.get("eta")
                    
                    parts = [f"Downloading... {percent:.0%}"]
                    if speed:
                        parts.append(f"{speed/1024/1024:.1f} MB/s")
                    if eta is not None:
                        # Format ETA as human-readable time
                        try:
                            eta_int = int(eta)
                            if eta_int < 60:
                                parts.append(f"ETA {eta_int}s")
                            elif eta_int < 3600:
                                mins, secs = divmod(eta_int, 60)
                                parts.append(f"ETA {mins}m {secs}s")
                            else:
                                hours, remainder = divmod(eta_int, 3600)
                                mins, secs = divmod(remainder, 60)
                                parts.append(f"ETA {hours}h {mins}m")
                        except (ValueError, TypeError):
                            pass  # Skip malformed ETA

                    # HLS fragments (Twitch): show fragment progress when available.
                    frag_idx = d.get("fragment_index")
                    frag_cnt = d.get("fragment_count")
                    if frag_cnt:
                        try:
                            frag_cnt_i = int(frag_cnt)
                            frag_idx_i = int(frag_idx) + 1 if frag_idx is not None else None
                            if frag_idx_i is not None and frag_cnt_i > 0:
                                parts.append(f"frag {frag_idx_i}/{frag_cnt_i}")
                        except (ValueError, TypeError):
                            pass
                    if current_n > 1:
                        parts.append(f"[N={current_n}]")
                    
                    on_progress(mapped, " | ".join(parts))
                    
        elif status == "finished":
            download_progress["phase"] = "postprocessing"
            if on_progress:
                on_progress(0.72, "Download complete. Processing...")
    
    # Step 3: Download with retry loop
    for attempt in range(max_attempts):
        # Build yt-dlp options
        ydl_opts: dict[str, Any] = {
            "outtmpl": str(output_dir / "%(title).100s [%(id)s].%(ext)s"),
            "noplaylist": request.no_playlist,
            "restrictfilenames": True,
            "progress_hooks": [progress_hook],
            "writeinfojson": True,
            # Best-effort: download the platform thumbnail as a sidecar file.
            # This is used by Studio as a "video icon" when available.
            "writethumbnail": True,
            "write_all_thumbnails": False,
            # Ensure yt-dlp doesn't write directly to stdout/stderr (prevents
            # progress lines from interleaving with our app logs).
            "logger": _NoopYtDlpLogger(),
            "quiet": True,
            "no_warnings": True,
            # Prevent yt-dlp's built-in progress output from interleaving with our logs.
            # We rely on progress_hooks + on_progress for UI and structured logging.
            "noprogress": True,
        }
        
        # Apply site policy
        if policy.use_hls_native:
            ydl_opts["hls_prefer_native"] = True
        
        if policy.supports_fragment_concurrency and current_n > 1:
            ydl_opts["concurrent_fragment_downloads"] = current_n
        
        # Format selection based on quality cap
        ydl_opts["format"] = get_format_selector(request.quality_cap.value)
        ydl_opts["merge_output_format"] = "mp4"
        
        # Status message
        if on_progress:
            if attempt == 0:
                badge = f"Detected: {policy.display_name}"
                if current_n > 1:
                    badge += f" — Auto concurrency {current_n}"
                on_progress(0.02, badge)
            else:
                on_progress(0.02, f"Retrying with N={current_n}...")
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            
            # Success! Update tuner
            if policy.supports_fragment_concurrency and request.speed_mode == SpeedMode.AUTO:
                update_domain_tuning(domain, current_n, "ok")
            
            break
            
        except Exception as e:
            last_error = e
            
            # Check for cancellation (yt-dlp wraps our exception)
            if download_progress.get("cancelled") or "cancelled" in str(e).lower():
                # Clean up partial files
                for pf in partial_files:
                    try:
                        if pf.exists():
                            pf.unlink()
                    except Exception:
                        pass
                # Also clean up any .part files in output_dir
                for part_file in output_dir.glob("*.part*"):
                    try:
                        part_file.unlink()
                    except Exception:
                        pass
                for ytdl_file in output_dir.glob("*.ytdl"):
                    try:
                        ytdl_file.unlink()
                    except Exception:
                        pass
                raise DownloadCancelled("Download cancelled by user")
            
            # Check for throttling
            if looks_like_throttle(e) and current_n > min_n:
                old_n = current_n
                current_n = calculate_backoff_n(current_n, min_n)
                
                if request.speed_mode == SpeedMode.AUTO:
                    update_domain_tuning(domain, current_n, "throttled")
                
                if on_progress:
                    on_progress(0.02, f"Throttled at N={old_n}, backing off to N={current_n}...")
                continue
            
            # Not throttling or can't back off further
            raise RuntimeError(f"Download failed: {e}")
    
    if not info:
        raise RuntimeError(f"Download failed after {max_attempts} attempts: {last_error}")
    
    # Step 4: Find downloaded files
    if on_progress:
        on_progress(0.75, "Finding downloaded files...")
    
    video_id = info.get("id", "unknown")
    title = info.get("title", "video")
    
    video_exts = {".mp4", ".mkv", ".webm", ".m4v", ".avi", ".mov", ".ts"}
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
    possible_paths = [
        video_path.with_suffix(".info.json"),
        output_dir / f"{video_path.stem}.info.json",
    ]
    for p in possible_paths:
        if p.exists():
            info_json_path = p
            break

    # Find downloaded thumbnail (yt-dlp sidecar). This may be .webp/.jpg/.png.
    thumbnail_path = None
    for ext in (".webp", ".jpg", ".jpeg", ".png"):
        p = video_path.with_suffix(ext)
        if p.exists():
            thumbnail_path = p
            break
    
    # Step 5: Post-process
    if on_progress:
        on_progress(0.78, "Post-processing...")
    
    postprocess_result = None
    preview_path = None
    
    if request.create_preview:
        def pp_progress(frac: float, msg: str) -> None:
            # Map postprocess progress to 0.78-0.98
            if on_progress:
                mapped = 0.78 + frac * 0.2
                on_progress(mapped, msg)
        
        postprocess_result = postprocess_download(
            video_path,
            create_preview_if_needed=True,
            on_progress=pp_progress,
        )
        
        if postprocess_result.preview_path:
            preview_path = postprocess_result.preview_path
        
        # Update video_path if it was remuxed
        if postprocess_result.remuxed:
            video_path = postprocess_result.source_path
    
    if on_progress:
        on_progress(1.0, "Download complete!")
    
    return IngestResult(
        video_path=video_path,
        info_json_path=info_json_path,
        preview_path=preview_path,
        thumbnail_path=thumbnail_path,
        title=title,
        url=url,
        extractor=info.get("extractor", ""),
        video_id=video_id,
        duration_seconds=float(info.get("duration", 0) or 0),
        site_type=site_type,
        postprocess=postprocess_result,
    )


# Re-export for backward compatibility with old downloader.py
__all__ = [
    "download_url",
    "download_audio_only",
    "IngestRequest",
    "IngestResult",
    "SpeedMode",
    "QualityCap",
    "SiteType",
    "SPEED_MODE_N",
    "probe_url",
]


def download_audio_only(
    url: str,
    output_dir: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Optional[Path]:
    """Download only the audio track from a URL using yt-dlp.
    
    This is much faster than downloading full video and is useful for
    starting transcription while video download is still in progress.
    
    Args:
        url: The URL to download audio from
        output_dir: Override output directory
        on_progress: Callback for progress updates (fraction, message)
    
    Returns:
        Path to downloaded audio file, or None if failed
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return None
    
    output_dir = output_dir or _default_downloads_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if on_progress:
        on_progress(0.0, "Downloading audio track...", {})
    
    def progress_hook(d: dict[str, Any]) -> None:
        status = d.get("status")
        if status == "downloading" and on_progress:
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            speed = d.get("speed")  # bytes per second
            if total and total > 0:
                percent = downloaded / total
                extra = {
                    "total_bytes": total,
                    "downloaded_bytes": downloaded,
                    "speed": speed,
                }
                frag_idx = d.get("fragment_index")
                frag_cnt = d.get("fragment_count")
                if frag_cnt:
                    try:
                        frag_cnt_i = int(frag_cnt)
                        frag_idx_i = int(frag_idx) + 1 if frag_idx is not None else None
                        if frag_idx_i is not None and frag_cnt_i > 0:
                            extra["fragment"] = f"{frag_idx_i}/{frag_cnt_i}"
                    except (ValueError, TypeError):
                        pass
                on_progress(percent * 0.95, f"Downloading audio... {percent:.0%}", extra)
        elif status == "finished" and on_progress:
            on_progress(0.98, "Audio download complete", {})
    
    # Get site policy for concurrency settings
    site_type = classify_url_heuristic(url)
    policy = get_policy(site_type)
    
    ydl_opts: dict[str, Any] = {
        "outtmpl": str(output_dir / "%(title).80s_audio_[%(id)s].%(ext)s"),
        "restrictfilenames": True,
        "progress_hooks": [progress_hook],
        # Ensure yt-dlp doesn't write directly to stdout/stderr (prevents progress
        # lines from interleaving with our app logs).
        "logger": _NoopYtDlpLogger(),
        "quiet": True,
        "no_warnings": True,
        # Prevent yt-dlp's built-in progress output from interleaving with our logs.
        # We rely on progress_hooks + on_progress for UI and structured logging.
        "noprogress": True,
        # Audio-only settings optimized for speed:
        # - "worstaudio" is smallest but often opus/webm which Whisper handles fine
        # - Skip FFmpeg extraction entirely to avoid the conversion overhead
        # - Whisper can decode most audio formats directly (opus, m4a, mp3, wav, etc.)
        "format": "worstaudio[ext=m4a]/worstaudio[ext=mp3]/worstaudio[ext=opus]/worstaudio/worst",
        # NO postprocessors - skip FFmpeg conversion entirely for speed
        # Whisper can handle opus/m4a/webm audio directly
    }
    
    # Enable concurrent downloads - use higher concurrency for audio since it's smaller
    if policy.supports_fragment_concurrency:
        # Use more aggressive concurrency for audio - it's smaller so less likely to throttle
        ydl_opts["concurrent_fragment_downloads"] = 16
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        
        if not info:
            return None
        
        # Find the audio file - check all common audio/container formats
        # Note: yt-dlp may download audio in video containers like .mp4 or .webm
        video_id = info.get("id", "unknown")
        audio_exts = [".m4a", ".mp4", ".mp3", ".opus", ".ogg", ".webm", ".wav", ".aac", ".mkv"]
        audio_candidates = []
        
        for ext in audio_exts:
            audio_candidates.extend(output_dir.glob(f"*_audio_*{video_id}*{ext}"))
        
        if audio_candidates:
            # Return most recently modified
            audio_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if on_progress:
                on_progress(1.0, "Audio ready for transcription")
            return audio_candidates[0]
        
        return None
        
    except Exception as e:
        if on_progress:
            on_progress(0.0, f"Audio download failed: {e}")
        return None
