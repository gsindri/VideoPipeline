"""Ingest module for downloading videos from URLs.

This module provides smart URL download with:
- Site detection (Twitch, YouTube, etc.)
- Adaptive concurrency for HLS streams
- Automatic throttle backoff
- Post-processing (remux, preview generation)
"""

# Keep old imports working
from .downloader import DownloadOptions, DownloadResult
from .models import (
    SITE_POLICIES,
    SPEED_MODE_N,
    IngestRequest,
    IngestResult,
    PostprocessResult,
    ProbeResult,
    QualityCap,
    SitePolicy,
    SiteType,
    SpeedMode,
)
from .policy import (
    classify_url_heuristic,
    get_format_selector,
    get_format_selectors,
    get_policy,
    probe_url,
)
from .postprocess import (
    create_preview,
    needs_preview,
    needs_remux,
    postprocess_download,
    probe_video,
    remux_to_mp4,
)
from .tuner import (
    DomainTuning,
    calculate_backoff_n,
    extract_domain,
    get_domain_tuning,
    looks_like_throttle,
    tuning_file_path,
    update_domain_tuning,
)
from .ytdlp_runner import DownloadCancelled, download_url

__all__ = [
    # Main entry points
    "download_url",
    "probe_url",
    # Exceptions
    "DownloadCancelled",
    # Models
    "IngestRequest",
    "IngestResult",
    "PostprocessResult",
    "ProbeResult",
    "QualityCap",
    "SitePolicy",
    "SiteType",
    "SpeedMode",
    "DomainTuning",
    "SITE_POLICIES",
    "SPEED_MODE_N",
    # Policy
    "classify_url_heuristic",
    "get_format_selector",
    "get_format_selectors",
    "get_policy",
    # Postprocess
    "create_preview",
    "needs_preview",
    "needs_remux",
    "postprocess_download",
    "probe_video",
    "remux_to_mp4",
    # Tuner
    "calculate_backoff_n",
    "extract_domain",
    "get_domain_tuning",
    "looks_like_throttle",
    "tuning_file_path",
    "update_domain_tuning",
    # Legacy
    "DownloadOptions",
    "DownloadResult",
]
