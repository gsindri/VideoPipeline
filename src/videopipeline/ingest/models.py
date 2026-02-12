"""Data models for URL ingest."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..utils import utc_iso as _utc_iso


class SiteType(str, Enum):
    """Detected site type from URL."""
    TWITCH_VOD = "twitch_vod"
    TWITCH_CLIP = "twitch_clip"
    YOUTUBE = "youtube"
    GENERIC = "generic"


class SpeedMode(str, Enum):
    """Download speed mode for HLS fragment concurrency."""
    AUTO = "auto"              # Adaptive based on tuning history
    CONSERVATIVE = "conservative"  # N=4, very safe
    BALANCED = "balanced"      # N=8, conservative
    FAST = "fast"              # N=16, good for most connections
    AGGRESSIVE = "aggressive"  # N=32, may trigger throttling


class QualityCap(str, Enum):
    """Maximum quality to download."""
    SOURCE = "source"      # Best available
    P1080 = "1080"         # Cap at 1080p
    P720 = "720"           # Cap at 720p
    P480 = "480"           # Cap at 480p


# Default concurrency values per speed mode
SPEED_MODE_N: dict[SpeedMode, int] = {
    SpeedMode.AUTO: 16,        # Starting value for auto
    SpeedMode.CONSERVATIVE: 4,
    SpeedMode.BALANCED: 8,
    SpeedMode.FAST: 16,
    SpeedMode.AGGRESSIVE: 32,
}


@dataclass
class SitePolicy:
    """Policy for downloading from a specific site type."""
    
    site_type: SiteType
    display_name: str
    
    # HLS-specific settings
    use_hls_native: bool = False
    supports_fragment_concurrency: bool = False
    default_concurrency: int = 1
    
    # Behavior hints
    notes: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "site_type": self.site_type.value,
            "display_name": self.display_name,
            "use_hls_native": self.use_hls_native,
            "supports_fragment_concurrency": self.supports_fragment_concurrency,
            "default_concurrency": self.default_concurrency,
            "notes": self.notes,
        }


# Pre-defined policies
SITE_POLICIES: dict[SiteType, SitePolicy] = {
    SiteType.TWITCH_VOD: SitePolicy(
        site_type=SiteType.TWITCH_VOD,
        display_name="Twitch VOD (HLS)",
        use_hls_native=True,
        supports_fragment_concurrency=True,
        default_concurrency=16,
        notes="Twitch downloads may throttle. Auto mode adapts fragment concurrency automatically.",
    ),
    SiteType.TWITCH_CLIP: SitePolicy(
        site_type=SiteType.TWITCH_CLIP,
        display_name="Twitch Clip",
        use_hls_native=True,
        supports_fragment_concurrency=True,
        default_concurrency=8,
        notes="Clips are short, downloads quickly.",
    ),
    SiteType.YOUTUBE: SitePolicy(
        site_type=SiteType.YOUTUBE,
        display_name="YouTube",
        use_hls_native=False,
        supports_fragment_concurrency=False,
        default_concurrency=1,
        notes="YouTube uses DASH, not HLS fragments.",
    ),
    SiteType.GENERIC: SitePolicy(
        site_type=SiteType.GENERIC,
        display_name="Generic Site",
        use_hls_native=False,
        supports_fragment_concurrency=False,
        default_concurrency=1,
        notes="Using default yt-dlp settings.",
    ),
}


@dataclass
class ProbeResult:
    """Result of probing a URL."""
    
    url: str
    site_type: SiteType
    policy: SitePolicy
    
    # From yt-dlp extract_info
    title: str = ""
    duration_seconds: float = 0.0
    extractor: str = ""
    video_id: str = ""
    is_live: bool = False
    
    # Detection details
    detected_by: str = "heuristic"  # "heuristic" or "ytdlp"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "site_type": self.site_type.value,
            "policy": self.policy.to_dict(),
            "title": self.title,
            "duration_seconds": self.duration_seconds,
            "extractor": self.extractor,
            "video_id": self.video_id,
            "is_live": self.is_live,
            "detected_by": self.detected_by,
            "display_badge": f"Detected: {self.policy.display_name}",
        }


@dataclass
class IngestRequest:
    """Request to download a URL."""
    
    url: str
    speed_mode: SpeedMode = SpeedMode.AUTO
    quality_cap: QualityCap = QualityCap.SOURCE
    no_playlist: bool = True
    create_preview: bool = True
    auto_open: bool = True


@dataclass
class PostprocessResult:
    """Result of postprocessing a downloaded file."""
    
    source_path: Path
    preview_path: Optional[Path] = None
    
    # What was done
    remuxed: bool = False
    preview_created: bool = False
    
    # Codec info
    source_video_codec: str = ""
    source_audio_codec: str = ""
    source_container: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "preview_path": str(self.preview_path) if self.preview_path else None,
            "remuxed": self.remuxed,
            "preview_created": self.preview_created,
            "source_video_codec": self.source_video_codec,
            "source_audio_codec": self.source_audio_codec,
            "source_container": self.source_container,
        }


@dataclass
class IngestResult:
    """Complete result of URL ingestion."""
    
    # Paths
    video_path: Path
    info_json_path: Optional[Path] = None
    preview_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    
    # Metadata from yt-dlp
    title: str = ""
    url: str = ""
    extractor: str = ""
    video_id: str = ""
    duration_seconds: float = 0.0
    
    # Site detection
    site_type: SiteType = SiteType.GENERIC
    
    # Postprocess info
    postprocess: Optional[PostprocessResult] = None
    
    # Status
    created_at: str = field(default_factory=_utc_iso)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "info_json_path": str(self.info_json_path) if self.info_json_path else None,
            "preview_path": str(self.preview_path) if self.preview_path else None,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "title": self.title,
            "url": self.url,
            "extractor": self.extractor,
            "video_id": self.video_id,
            "duration_seconds": self.duration_seconds,
            "site_type": self.site_type.value,
            "postprocess": self.postprocess.to_dict() if self.postprocess else None,
            "created_at": self.created_at,
        }
