"""Tests for metadata module."""

from pathlib import Path

import pytest

from videopipeline.metadata import (
    build_metadata,
    derive_hook_text,
    _clean_text,
    _pick_hook_from_segments,
    _fmt_time,
)
from videopipeline.subtitles import SubtitleSegment


class TestCleanText:
    def test_removes_extra_whitespace(self):
        assert _clean_text("  hello   world  ") == "hello world"

    def test_handles_newlines(self):
        assert _clean_text("hello\n  world") == "hello world"

    def test_empty_string(self):
        assert _clean_text("   ") == ""


class TestFmtTime:
    def test_zero_seconds(self):
        assert _fmt_time(0) == "00:00"

    def test_simple_time(self):
        assert _fmt_time(65) == "01:05"

    def test_negative_clamped(self):
        assert _fmt_time(-10) == "00:00"

    def test_large_time(self):
        assert _fmt_time(3661) == "61:01"


class TestPickHookFromSegments:
    def test_picks_good_segment(self):
        segments = [
            SubtitleSegment(start_s=0.0, end_s=1.0, text="Hi"),  # too short
            SubtitleSegment(start_s=1.0, end_s=5.0, text="This is a longer sentence that works well"),
        ]
        result = _pick_hook_from_segments(segments)
        assert result == "This is a longer sentence that works well"

    def test_empty_list(self):
        assert _pick_hook_from_segments([]) is None

    def test_all_short_segments(self):
        segments = [
            SubtitleSegment(start_s=0.0, end_s=1.0, text="Hi"),
            SubtitleSegment(start_s=1.0, end_s=2.0, text="OK"),
        ]
        assert _pick_hook_from_segments(segments) is None

    def test_truncates_long_hook(self):
        long_text = "word " * 50  # longer than 100 chars, has enough words
        segments = [SubtitleSegment(start_s=0.0, end_s=10.0, text=long_text)]
        result = _pick_hook_from_segments(segments)
        assert result is not None
        assert len(result) == 100


class TestDeriveHookText:
    def test_uses_segments_if_available(self):
        selection = {"rank": 1, "start_s": 0.0}
        segments = [
            SubtitleSegment(start_s=0.0, end_s=5.0, text="This is a great hook text here"),
        ]
        result = derive_hook_text(selection, segments)
        assert result == "This is a great hook text here"

    def test_fallback_to_rank(self):
        selection = {"candidate_rank": 5, "start_s": 10.0}
        result = derive_hook_text(selection, None)
        assert result == "Highlight #5"

    def test_fallback_to_time(self):
        selection = {"start_s": 125.0}
        result = derive_hook_text(selection, None)
        assert result == "Clip @ 02:05"


class TestBuildMetadata:
    def test_basic_metadata(self):
        selection = {
            "id": "abc123",
            "start_s": 10.0,
            "end_s": 40.0,
            "candidate_rank": 1,
            "candidate_score": 0.85,
            "candidate_peak_time_s": 25.0,
        }
        result = build_metadata(
            selection=selection,
            output_path=Path("/tmp/test.mp4"),
            template="vertical_blur",
            with_captions=True,
        )
        assert result["template"] == "vertical_blur"
        assert result["with_captions"] is True
        assert result["selection"]["id"] == "abc123"
        assert "#vertical" in result["hashtags"]

    def test_ai_metadata_priority(self):
        """AI metadata should override defaults."""
        selection = {
            "id": "abc123",
            "start_s": 10.0,
            "end_s": 40.0,
            "title": "Original Title",
        }
        ai_metadata = {
            "title": "AI Generated Title",
            "description": "AI generated description for this clip",
            "tags": ["gaming", "highlights", "epic"],
            "hook": "This is insane!",
        }
        result = build_metadata(
            selection=selection,
            output_path=Path("/tmp/test.mp4"),
            template="vertical_blur",
            with_captions=False,
            ai_metadata=ai_metadata,
        )
        assert result["title"] == "AI Generated Title"
        assert result["caption"] == "AI generated description for this clip"
        assert "#gaming" in result["hashtags"]
        assert result["ai_generated"]["hook"] == "This is insane!"

    def test_ai_metadata_partial(self):
        """AI metadata with only some fields should use fallbacks for others."""
        selection = {
            "id": "abc123",
            "start_s": 10.0,
            "end_s": 40.0,
            "title": "Original Title",
        }
        ai_metadata = {
            "title": "AI Title",
            # no description, tags, or hook
        }
        result = build_metadata(
            selection=selection,
            output_path=Path("/tmp/test.mp4"),
            template="horizontal",
            with_captions=False,
            ai_metadata=ai_metadata,
        )
        assert result["title"] == "AI Title"
        # caption should be derived from derive_hook_text since no AI description
        assert result["caption"] is not None
        # hashtags should be defaults since no AI tags
        assert "#gaming" in result["hashtags"]
        # ai_generated section should be present
        assert "ai_generated" in result

    def test_hashtags_get_hash_prefix(self):
        """Tags without # should get # added."""
        selection = {"id": "abc123", "start_s": 10.0, "end_s": 40.0}
        ai_metadata = {"tags": ["gaming", "#already_has"]}
        result = build_metadata(
            selection=selection,
            output_path=Path("/tmp/test.mp4"),
            template="vertical_blur",
            with_captions=False,
            ai_metadata=ai_metadata,
        )
        assert "#gaming" in result["hashtags"]
        assert "#already_has" in result["hashtags"]

    def test_vertical_tag_not_duplicated(self):
        """#vertical should not be duplicated if already in tags."""
        selection = {"id": "abc123", "start_s": 10.0, "end_s": 40.0}
        ai_metadata = {"tags": ["vertical", "gaming"]}
        result = build_metadata(
            selection=selection,
            output_path=Path("/tmp/test.mp4"),
            template="vertical_blur",
            with_captions=False,
            ai_metadata=ai_metadata,
        )
        vertical_count = result["hashtags"].count("#vertical")
        assert vertical_count == 1

    def test_transcript_snippet_included(self):
        """Transcript snippet should be included if segments provided."""
        selection = {"id": "abc123", "start_s": 10.0, "end_s": 40.0}
        segments = [
            SubtitleSegment(start_s=0.0, end_s=5.0, text="This is a good transcript snippet here"),
        ]
        result = build_metadata(
            selection=selection,
            output_path=Path("/tmp/test.mp4"),
            template="vertical_blur",
            with_captions=True,
            segments=segments,
        )
        assert "transcript_snippet" in result
        assert result["transcript_snippet"] == "This is a good transcript snippet here"

    def test_platform_hints(self):
        """Platform hints should always be present."""
        selection = {"id": "abc123", "start_s": 10.0, "end_s": 40.0}
        result = build_metadata(
            selection=selection,
            output_path=Path("/tmp/test.mp4"),
            template="vertical_blur",
            with_captions=False,
        )
        hints = result["platform_hints"]
        assert hints["shorts_max_seconds"] == 60
        assert hints["tiktok_max_seconds"] == 60
        assert "safe_zone_top_px" in hints
        assert "safe_zone_bottom_px" in hints
