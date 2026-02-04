"""URL classification and site policy selection."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

from .models import ProbeResult, SitePolicy, SiteType, SITE_POLICIES


def classify_url_heuristic(url: str) -> SiteType:
    """Classify URL by hostname and path patterns (instant, no network).
    
    This is Layer 1 classification - fast but may be overridden by yt-dlp probe.
    """
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()
    except Exception:
        return SiteType.GENERIC
    
    # Twitch
    if "twitch.tv" in host:
        # Twitch VOD: /videos/<id>
        if "/videos/" in path:
            return SiteType.TWITCH_VOD
        # Twitch clip: /clip/ or clips.twitch.tv
        if "/clip/" in path or "clips.twitch.tv" in host:
            return SiteType.TWITCH_CLIP
        # Could be a channel page with VOD - let yt-dlp figure it out
        return SiteType.TWITCH_VOD
    
    # YouTube
    if "youtube.com" in host or "youtu.be" in host:
        return SiteType.YOUTUBE
    
    # YouTube other domains
    if "googlevideo.com" in host:
        return SiteType.YOUTUBE
    
    return SiteType.GENERIC


def get_policy(site_type: SiteType) -> SitePolicy:
    """Get the download policy for a site type."""
    return SITE_POLICIES.get(site_type, SITE_POLICIES[SiteType.GENERIC])


def probe_url(url: str, use_ytdlp: bool = True) -> ProbeResult:
    """Probe a URL to detect site type and metadata.
    
    Args:
        url: The URL to probe
        use_ytdlp: Whether to use yt-dlp for authoritative detection (slower but more accurate)
    
    Returns:
        ProbeResult with site type, policy, and metadata
    """
    # Layer 1: Heuristic classification (instant)
    site_type = classify_url_heuristic(url)
    policy = get_policy(site_type)
    
    result = ProbeResult(
        url=url,
        site_type=site_type,
        policy=policy,
        detected_by="heuristic",
    )
    
    if not use_ytdlp:
        return result
    
    # Layer 2: yt-dlp probe (authoritative)
    try:
        from yt_dlp import YoutubeDL
        
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        
        if info:
            result.title = info.get("title", "")
            result.duration_seconds = float(info.get("duration", 0) or 0)
            result.extractor = info.get("extractor", "") or info.get("ie_key", "")
            result.video_id = info.get("id", "")
            result.is_live = bool(info.get("is_live", False))
            result.detected_by = "ytdlp"
            
            # Refine site type based on extractor
            extractor_lower = result.extractor.lower()
            
            if "twitch" in extractor_lower:
                if "clip" in extractor_lower:
                    result.site_type = SiteType.TWITCH_CLIP
                else:
                    result.site_type = SiteType.TWITCH_VOD
                result.policy = get_policy(result.site_type)
            
            elif "youtube" in extractor_lower:
                result.site_type = SiteType.YOUTUBE
                result.policy = get_policy(result.site_type)
    
    except ImportError:
        # yt-dlp not installed, stick with heuristic
        pass
    except Exception:
        # Probe failed, stick with heuristic
        pass
    
    return result


def get_format_selector(quality_cap: str) -> str:
    """Get yt-dlp format selector for a quality cap.
    
    Args:
        quality_cap: One of "source", "1080", "720", "480"
    
    Returns:
        yt-dlp format string
    """
    if quality_cap == "source":
        # Best video+audio, prefer mp4
        return "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"
    
    height = int(quality_cap)
    
    # Best video up to height + best audio, with fallbacks
    return (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}]/"
        "best"
    )
