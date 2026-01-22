"""Tests for the URL ingest module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from videopipeline.ingest.downloader import (
    DownloadOptions,
    DownloadResult,
    SpeedMode,
    SPEED_MODE_N,
    _default_downloads_dir,
    _get_domain,
    _looks_like_throttle,
    _needs_preview,
    _sanitize_filename,
)


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_removes_invalid_chars(self):
        """Invalid characters are replaced with underscores."""
        result = _sanitize_filename('video:test<>"|?*')
        # Each invalid char gets replaced with underscore
        assert '<' not in result
        assert '>' not in result
        assert ':' not in result
        assert '"' not in result
        assert '|' not in result
        assert '?' not in result
        assert '*' not in result

    def test_truncates_long_names(self):
        """Long filenames are truncated."""
        long_name = 'a' * 200
        result = _sanitize_filename(long_name, max_length=50)
        assert len(result) <= 50

    def test_handles_empty_string(self):
        """Empty string returns 'video'."""
        assert _sanitize_filename('') == 'video'
        assert _sanitize_filename('   ') == 'video'

    def test_strips_trailing_dots_and_spaces(self):
        """Trailing dots and spaces are stripped."""
        assert _sanitize_filename('test...') == 'test'
        assert _sanitize_filename('test   ') == 'test'


class TestDefaultDownloadsDir:
    """Tests for default downloads directory."""

    def test_returns_path(self):
        """Returns a Path object."""
        result = _default_downloads_dir()
        assert isinstance(result, Path)
        assert 'downloads' in str(result).lower()


class TestDownloadOptions:
    """Tests for DownloadOptions dataclass."""

    def test_defaults(self):
        """Default options are sensible."""
        opts = DownloadOptions()
        assert opts.no_playlist is True
        assert opts.best_quality is True
        assert opts.create_preview is True
        assert opts.preview_height == 720
        assert opts.speed_mode == SpeedMode.AUTO

    def test_custom_options(self):
        """Custom options are preserved."""
        opts = DownloadOptions(
            no_playlist=False,
            best_quality=False,
            create_preview=False,
            preview_height=480,
            speed_mode=SpeedMode.FAST,
        )
        assert opts.no_playlist is False
        assert opts.best_quality is False
        assert opts.create_preview is False
        assert opts.preview_height == 480
        assert opts.speed_mode == SpeedMode.FAST


class TestDownloadResult:
    """Tests for DownloadResult dataclass."""

    def test_to_dict(self, tmp_path: Path):
        """to_dict returns proper structure."""
        result = DownloadResult(
            video_path=tmp_path / 'video.mp4',
            info_json_path=tmp_path / 'video.info.json',
            preview_path=tmp_path / 'video_preview.mp4',
            title='Test Video',
            url='https://youtube.com/watch?v=abc123',
            extractor='youtube',
            video_id='abc123',
            duration_seconds=120.5,
        )

        d = result.to_dict()
        assert d['title'] == 'Test Video'
        assert d['url'] == 'https://youtube.com/watch?v=abc123'
        assert d['extractor'] == 'youtube'
        assert d['video_id'] == 'abc123'
        assert d['duration_seconds'] == 120.5
        assert 'video.mp4' in d['video_path']

    def test_to_dict_optional_fields(self, tmp_path: Path):
        """to_dict handles None optional fields."""
        result = DownloadResult(
            video_path=tmp_path / 'video.mp4',
        )

        d = result.to_dict()
        assert d['info_json_path'] is None
        assert d['preview_path'] is None


class TestNeedsPreview:
    """Tests for codec detection."""

    def test_mp4_without_ffprobe(self, tmp_path: Path):
        """Returns True if ffprobe fails (safe default)."""
        video = tmp_path / 'test.mp4'
        video.write_bytes(b'fake video data')

        # Without ffprobe available, should return True (create preview to be safe)
        with patch('videopipeline.ingest.downloader._require_cmd') as mock_cmd:
            mock_cmd.side_effect = FileNotFoundError('ffprobe not found')
            result = _needs_preview(video)
            assert result is True

    def test_webm_needs_preview(self, tmp_path: Path):
        """WebM files need preview."""
        video = tmp_path / 'test.webm'
        video.write_bytes(b'fake video data')

        # WebM container = needs preview regardless of codec
        result = _needs_preview(video)
        assert result is True  # Non-mp4 container


class TestMockDownload:
    """Tests for download function with mocked yt-dlp."""

    def test_download_import_error(self):
        """Raises ImportError if yt-dlp not installed."""
        from videopipeline.ingest.downloader import download_url

        with patch.dict('sys.modules', {'yt_dlp': None}):
            with patch('builtins.__import__', side_effect=ImportError('No module named yt_dlp')):
                with pytest.raises(ImportError, match='yt-dlp'):
                    download_url('https://example.com/video')

    def test_download_calls_yt_dlp(self, tmp_path: Path):
        """download_url calls yt-dlp correctly."""
        from videopipeline.ingest.downloader import download_url

        # Create a fake output file
        output_file = tmp_path / 'Test Video [abc123].mp4'
        output_file.write_bytes(b'fake video content')

        # Mock yt-dlp
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {
            'id': 'abc123',
            'title': 'Test Video',
            'extractor': 'youtube',
            'duration': 120,
        }

        mock_ydl_class = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', mock_ydl_class):
            with patch('videopipeline.ingest.downloader._default_downloads_dir', return_value=tmp_path):
                with patch('videopipeline.ingest.downloader._needs_preview', return_value=False):
                    result = download_url('https://youtube.com/watch?v=abc123')

        assert result.title == 'Test Video'
        assert result.video_id == 'abc123'
        assert result.extractor == 'youtube'
        assert result.duration_seconds == 120


class TestProgressCallbacks:
    """Tests for progress callback behavior."""

    def test_progress_callback_called(self, tmp_path: Path):
        """Progress callback is invoked during download."""
        from videopipeline.ingest.downloader import download_url

        progress_calls = []

        def on_progress(frac, msg):
            progress_calls.append((frac, msg))

        output_file = tmp_path / 'video.mp4'
        output_file.write_bytes(b'fake video')

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {
            'id': 'test',
            'title': 'Test',
        }
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            with patch('videopipeline.ingest.downloader._default_downloads_dir', return_value=tmp_path):
                with patch('videopipeline.ingest.downloader._needs_preview', return_value=False):
                    download_url('https://example.com', on_progress=on_progress)

        # Should have at least the initial "Extracting video info..." call
        assert len(progress_calls) > 0
        # AUTO mode shows N value in the message
        assert 'Extracting video info' in progress_calls[0][1]


class TestSpeedMode:
    """Tests for SpeedMode enum and concurrency values."""

    def test_speed_mode_values(self):
        """SpeedMode has expected values."""
        assert SpeedMode.BALANCED.value == 'balanced'
        assert SpeedMode.FAST.value == 'fast'
        assert SpeedMode.AGGRESSIVE.value == 'aggressive'
        assert SpeedMode.AUTO.value == 'auto'

    def test_speed_mode_n_mapping(self):
        """Each speed mode has a concurrency value."""
        assert SPEED_MODE_N[SpeedMode.BALANCED] == 8
        assert SPEED_MODE_N[SpeedMode.FAST] == 16
        assert SPEED_MODE_N[SpeedMode.AGGRESSIVE] == 32
        assert SPEED_MODE_N[SpeedMode.AUTO] == 16  # Default starting value


class TestGetDomain:
    """Tests for domain extraction."""

    def test_youtube_domain(self):
        """Extracts YouTube domain."""
        assert _get_domain('https://www.youtube.com/watch?v=abc123') == 'www.youtube.com'
        assert _get_domain('https://youtube.com/watch?v=abc123') == 'youtube.com'

    def test_twitch_domain(self):
        """Extracts Twitch domain."""
        assert _get_domain('https://www.twitch.tv/videos/123456') == 'www.twitch.tv'
        assert _get_domain('https://clips.twitch.tv/some-clip') == 'clips.twitch.tv'

    def test_invalid_url(self):
        """Returns 'unknown' for invalid URLs."""
        assert _get_domain('not a url') == 'unknown'
        assert _get_domain('') == 'unknown'


class TestLooksLikeThrottle:
    """Tests for throttle detection."""

    def test_429_error(self):
        """HTTP 429 is detected as throttling."""
        error = Exception('HTTP Error 429: Too Many Requests')
        assert _looks_like_throttle(error) is True

    def test_403_error(self):
        """HTTP 403 is detected as throttling."""
        error = Exception('HTTP Error 403: Forbidden')
        assert _looks_like_throttle(error) is True

    def test_rate_limit_error(self):
        """Rate limit messages are detected."""
        error = Exception('Rate limit exceeded, try again later')
        assert _looks_like_throttle(error) is True

    def test_retry_error(self):
        """Retry messages are detected."""
        error = Exception('fragment 5: giving up after 10 retries')
        assert _looks_like_throttle(error) is True

    def test_regular_error(self):
        """Regular errors are not detected as throttling."""
        error = Exception('Video unavailable')
        assert _looks_like_throttle(error) is False

    def test_network_error(self):
        """Network errors are not throttling."""
        error = Exception('Connection refused')
        assert _looks_like_throttle(error) is False


class TestTuningState:
    """Tests for tuning state persistence."""

    def test_load_save_best_n(self, tmp_path: Path):
        """Can save and load best N values."""
        from videopipeline.ingest.downloader import _load_best_n, _save_best_n, _tuning_file_path

        with patch('videopipeline.ingest.downloader._tuning_file_path', return_value=tmp_path / 'tuning.json'):
            # Initially returns default
            assert _load_best_n('www.twitch.tv', default=16) == 16

            # Save a value
            _save_best_n('www.twitch.tv', 8, ok=True)

            # Now returns saved value
            assert _load_best_n('www.twitch.tv', default=16) == 8

            # Different domain returns default
            assert _load_best_n('youtube.com', default=16) == 16

    def test_tuning_file_structure(self, tmp_path: Path):
        """Tuning file has correct JSON structure."""
        from videopipeline.ingest.downloader import _save_best_n, _load_tuning

        tuning_file = tmp_path / 'tuning.json'

        with patch('videopipeline.ingest.downloader._tuning_file_path', return_value=tuning_file):
            _save_best_n('www.twitch.tv', 12, ok=True)
            _save_best_n('youtube.com', 8, ok=False)

            # Check file contents
            data = json.loads(tuning_file.read_text())
            assert 'www.twitch.tv' in data
            assert data['www.twitch.tv']['N'] == 12
            assert data['www.twitch.tv']['last_ok'] is True
            assert 'youtube.com' in data
            assert data['youtube.com']['N'] == 8
            assert data['youtube.com']['last_ok'] is False
