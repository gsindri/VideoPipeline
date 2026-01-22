"""Tests for the refactored ingest module (Step 6.5)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from videopipeline.ingest.models import (
    IngestRequest,
    IngestResult,
    PostprocessResult,
    ProbeResult,
    QualityCap,
    SitePolicy,
    SiteType,
    SpeedMode,
    SITE_POLICIES,
    SPEED_MODE_N,
)
from videopipeline.ingest.policy import (
    classify_url_heuristic,
    get_format_selector,
    get_policy,
)
from videopipeline.ingest.tuner import (
    DomainTuning,
    calculate_backoff_n,
    extract_domain,
    get_domain_tuning,
    looks_like_throttle,
    update_domain_tuning,
)
from videopipeline.ingest.postprocess import (
    needs_preview,
    needs_remux,
)


# =============================================================================
# URL Classification Tests
# =============================================================================

class TestClassifyUrlHeuristic:
    """Tests for URL classification by hostname/path."""

    def test_twitch_vod(self):
        """Twitch VOD URLs are detected correctly."""
        assert classify_url_heuristic("https://www.twitch.tv/videos/2663368937") == SiteType.TWITCH_VOD
        assert classify_url_heuristic("https://twitch.tv/videos/123456") == SiteType.TWITCH_VOD

    def test_twitch_clip(self):
        """Twitch clip URLs are detected correctly."""
        assert classify_url_heuristic("https://clips.twitch.tv/SomeClipName") == SiteType.TWITCH_CLIP
        assert classify_url_heuristic("https://www.twitch.tv/channel/clip/ClipName") == SiteType.TWITCH_CLIP

    def test_youtube(self):
        """YouTube URLs are detected correctly."""
        assert classify_url_heuristic("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == SiteType.YOUTUBE
        assert classify_url_heuristic("https://youtube.com/watch?v=abc123") == SiteType.YOUTUBE
        assert classify_url_heuristic("https://youtu.be/abc123") == SiteType.YOUTUBE

    def test_generic(self):
        """Unknown URLs are classified as generic."""
        assert classify_url_heuristic("https://example.com/video.mp4") == SiteType.GENERIC
        assert classify_url_heuristic("https://vimeo.com/123456") == SiteType.GENERIC

    def test_invalid_url(self):
        """Invalid URLs default to generic."""
        assert classify_url_heuristic("not a url") == SiteType.GENERIC
        assert classify_url_heuristic("") == SiteType.GENERIC


class TestGetPolicy:
    """Tests for site policy retrieval."""

    def test_twitch_vod_policy(self):
        """Twitch VOD has correct policy settings."""
        policy = get_policy(SiteType.TWITCH_VOD)
        assert policy.site_type == SiteType.TWITCH_VOD
        assert policy.use_hls_native is True
        assert policy.supports_fragment_concurrency is True
        assert policy.default_concurrency == 16

    def test_youtube_policy(self):
        """YouTube has correct policy settings."""
        policy = get_policy(SiteType.YOUTUBE)
        assert policy.site_type == SiteType.YOUTUBE
        assert policy.use_hls_native is False
        assert policy.supports_fragment_concurrency is False

    def test_generic_policy(self):
        """Generic sites have conservative policy."""
        policy = get_policy(SiteType.GENERIC)
        assert policy.site_type == SiteType.GENERIC
        assert policy.supports_fragment_concurrency is False


class TestGetFormatSelector:
    """Tests for format selector generation."""

    def test_source_quality(self):
        """Source quality returns best format."""
        selector = get_format_selector("source")
        assert "bestvideo" in selector
        assert "bestaudio" in selector

    def test_1080p_quality(self):
        """1080p quality caps at 1080."""
        selector = get_format_selector("1080")
        assert "height<=1080" in selector

    def test_720p_quality(self):
        """720p quality caps at 720."""
        selector = get_format_selector("720")
        assert "height<=720" in selector

    def test_480p_quality(self):
        """480p quality caps at 480."""
        selector = get_format_selector("480")
        assert "height<=480" in selector


# =============================================================================
# Tuner Tests
# =============================================================================

class TestDomainTuning:
    """Tests for domain tuning dataclass."""

    def test_defaults(self):
        """Default tuning values."""
        tuning = DomainTuning()
        assert tuning.N == 16
        assert tuning.min_N == 2
        assert tuning.max_N == 32

    def test_to_dict_from_dict(self):
        """Round-trip through dict."""
        original = DomainTuning(N=8, min_N=2, max_N=24, last_result="ok")
        restored = DomainTuning.from_dict(original.to_dict())
        assert restored.N == 8
        assert restored.last_result == "ok"


class TestExtractDomain:
    """Tests for domain extraction."""

    def test_twitch_domain(self):
        """Extracts Twitch domain."""
        assert extract_domain("https://www.twitch.tv/videos/123") == "www.twitch.tv"
        assert extract_domain("https://clips.twitch.tv/abc") == "clips.twitch.tv"

    def test_youtube_domain(self):
        """Extracts YouTube domain."""
        assert extract_domain("https://www.youtube.com/watch?v=abc") == "www.youtube.com"
        assert extract_domain("https://youtu.be/abc") == "youtu.be"

    def test_invalid_url(self):
        """Returns 'unknown' for invalid URLs."""
        assert extract_domain("not a url") == "unknown"
        assert extract_domain("") == "unknown"


class TestCalculateBackoffN:
    """Tests for concurrency backoff calculation."""

    def test_halves_concurrency(self):
        """Backoff halves the concurrency."""
        assert calculate_backoff_n(16, min_n=2) == 8
        assert calculate_backoff_n(8, min_n=2) == 4
        assert calculate_backoff_n(4, min_n=2) == 2

    def test_respects_minimum(self):
        """Backoff respects minimum."""
        assert calculate_backoff_n(2, min_n=2) == 2
        assert calculate_backoff_n(3, min_n=2) == 2


class TestLooksLikeThrottle:
    """Tests for throttle detection."""

    def test_429_error(self):
        """HTTP 429 is detected."""
        assert looks_like_throttle(Exception("HTTP Error 429: Too Many Requests")) is True

    def test_403_error(self):
        """HTTP 403 is detected."""
        assert looks_like_throttle(Exception("HTTP Error 403: Forbidden")) is True

    def test_rate_limit(self):
        """Rate limit messages are detected."""
        assert looks_like_throttle(Exception("Rate limit exceeded")) is True

    def test_fragment_retries(self):
        """Fragment retry messages are detected."""
        assert looks_like_throttle(Exception("fragment 5: giving up after 10 retries")) is True

    def test_regular_error(self):
        """Regular errors are not throttling."""
        assert looks_like_throttle(Exception("Video unavailable")) is False


class TestTuningPersistence:
    """Tests for tuning state persistence."""

    def test_get_domain_tuning_twitch(self):
        """Twitch domains get Twitch defaults."""
        tuning = get_domain_tuning("www.twitch.tv")
        assert tuning.N == 16
        assert tuning.min_N == 2
        assert tuning.max_N == 32

    def test_get_domain_tuning_generic(self):
        """Unknown domains get conservative defaults."""
        tuning = get_domain_tuning("unknown-site.com")
        assert tuning.N == 4
        assert tuning.max_N == 16

    def test_update_and_load(self, tmp_path: Path):
        """Can save and load tuning state."""
        from videopipeline.ingest.tuner import tuning_file_path

        with patch("videopipeline.ingest.tuner.tuning_file_path", return_value=tmp_path / "tuning.json"):
            update_domain_tuning("test.com", 12, "ok")
            tuning = get_domain_tuning("test.com")
            assert tuning.N == 12
            assert tuning.last_result == "ok"


# =============================================================================
# Postprocess Tests
# =============================================================================

class TestNeedsRemux:
    """Tests for remux detection."""

    def test_mpegts_needs_remux(self):
        """MPEG-TS needs remux."""
        probe = {"container": ".ts", "format_name": "mpegts", "video_codec": "h264", "audio_codec": "aac"}
        assert needs_remux(probe) is True

    def test_mkv_h264_needs_remux(self):
        """MKV with H.264 can be remuxed."""
        probe = {"container": ".mkv", "format_name": "matroska", "video_codec": "h264", "audio_codec": "aac"}
        assert needs_remux(probe) is True

    def test_mp4_no_remux(self):
        """MP4 doesn't need remux."""
        probe = {"container": ".mp4", "format_name": "mp4", "video_codec": "h264", "audio_codec": "aac"}
        assert needs_remux(probe) is False

    def test_error_no_remux(self):
        """Error probe returns no remux needed."""
        probe = {"error": "ffprobe failed"}
        assert needs_remux(probe) is False


class TestNeedsPreview:
    """Tests for preview detection."""

    def test_h264_mp4_no_preview(self):
        """H.264/AAC MP4 doesn't need preview."""
        probe = {"container": ".mp4", "video_codec": "h264", "audio_codec": "aac"}
        assert needs_preview(probe) is False

    def test_hevc_needs_preview(self):
        """HEVC needs preview."""
        probe = {"container": ".mp4", "video_codec": "hevc", "audio_codec": "aac"}
        assert needs_preview(probe) is True

    def test_webm_needs_preview(self):
        """WebM needs preview."""
        probe = {"container": ".webm", "video_codec": "vp9", "audio_codec": "opus"}
        assert needs_preview(probe) is True

    def test_error_needs_preview(self):
        """Error probe assumes preview needed."""
        probe = {"error": "ffprobe failed"}
        assert needs_preview(probe) is True


# =============================================================================
# Model Tests
# =============================================================================

class TestSpeedMode:
    """Tests for SpeedMode enum."""

    def test_values(self):
        """SpeedMode has correct values."""
        assert SpeedMode.AUTO.value == "auto"
        assert SpeedMode.CONSERVATIVE.value == "conservative"
        assert SpeedMode.BALANCED.value == "balanced"
        assert SpeedMode.FAST.value == "fast"
        assert SpeedMode.AGGRESSIVE.value == "aggressive"

    def test_speed_mode_n_mapping(self):
        """Speed modes map to correct N values."""
        assert SPEED_MODE_N[SpeedMode.CONSERVATIVE] == 4
        assert SPEED_MODE_N[SpeedMode.BALANCED] == 8
        assert SPEED_MODE_N[SpeedMode.FAST] == 16
        assert SPEED_MODE_N[SpeedMode.AGGRESSIVE] == 32


class TestQualityCap:
    """Tests for QualityCap enum."""

    def test_values(self):
        """QualityCap has correct values."""
        assert QualityCap.SOURCE.value == "source"
        assert QualityCap.P1080.value == "1080"
        assert QualityCap.P720.value == "720"
        assert QualityCap.P480.value == "480"


class TestIngestRequest:
    """Tests for IngestRequest dataclass."""

    def test_defaults(self):
        """Default request values."""
        req = IngestRequest(url="https://example.com")
        assert req.speed_mode == SpeedMode.AUTO
        assert req.quality_cap == QualityCap.SOURCE
        assert req.no_playlist is True
        assert req.create_preview is True
        assert req.auto_open is True


class TestIngestResult:
    """Tests for IngestResult dataclass."""

    def test_to_dict(self, tmp_path: Path):
        """to_dict returns proper structure."""
        result = IngestResult(
            video_path=tmp_path / "video.mp4",
            title="Test Video",
            url="https://twitch.tv/videos/123",
            site_type=SiteType.TWITCH_VOD,
        )
        d = result.to_dict()
        assert d["title"] == "Test Video"
        assert d["site_type"] == "twitch_vod"


class TestProbeResult:
    """Tests for ProbeResult dataclass."""

    def test_to_dict(self):
        """to_dict returns proper structure including badge."""
        policy = get_policy(SiteType.TWITCH_VOD)
        result = ProbeResult(
            url="https://twitch.tv/videos/123",
            site_type=SiteType.TWITCH_VOD,
            policy=policy,
            title="Test Stream",
        )
        d = result.to_dict()
        assert d["site_type"] == "twitch_vod"
        assert "Twitch VOD" in d["display_badge"]
        assert d["policy"]["supports_fragment_concurrency"] is True
