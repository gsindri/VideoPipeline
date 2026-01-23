"""Tests for analysis_silence module."""

import pytest

from videopipeline.analysis_silence import (
    SilenceConfig,
    SilenceInterval,
    parse_silencedetect_output,
)


class TestParseSilencedetectOutput:
    """Tests for parsing FFmpeg silencedetect output."""

    def test_basic_parsing(self):
        """Test parsing standard silencedetect output."""
        output = """
[silencedetect @ 0x12345] silence_start: 10.5
[silencedetect @ 0x12345] silence_end: 12.3 | silence_duration: 1.8
[silencedetect @ 0x12345] silence_start: 25.0
[silencedetect @ 0x12345] silence_end: 26.5 | silence_duration: 1.5
"""
        intervals = parse_silencedetect_output(output)
        
        assert len(intervals) == 2
        assert intervals[0].start == pytest.approx(10.5)
        assert intervals[0].end == pytest.approx(12.3)
        assert intervals[0].duration == pytest.approx(1.8)
        assert intervals[1].start == pytest.approx(25.0)
        assert intervals[1].end == pytest.approx(26.5)
        assert intervals[1].duration == pytest.approx(1.5)

    def test_empty_output(self):
        """Test parsing when no silence is detected."""
        output = "Some random FFmpeg output with no silence"
        intervals = parse_silencedetect_output(output)
        assert intervals == []

    def test_unclosed_silence(self):
        """Test parsing when silence_start has no matching silence_end."""
        output = """
[silencedetect @ 0x12345] silence_start: 10.5
[silencedetect @ 0x12345] silence_end: 12.3 | silence_duration: 1.8
[silencedetect @ 0x12345] silence_start: 99.0
"""
        intervals = parse_silencedetect_output(output)
        
        # Should only have 1 complete interval
        assert len(intervals) == 1
        assert intervals[0].start == pytest.approx(10.5)

    def test_scientific_notation(self):
        """Test parsing scientific notation in timestamps."""
        output = """
[silencedetect @ 0x12345] silence_start: 15.0
[silencedetect @ 0x12345] silence_end: 20.0 | silence_duration: 5.0
"""
        intervals = parse_silencedetect_output(output)
        
        assert len(intervals) == 1
        assert intervals[0].start == pytest.approx(15.0)
        assert intervals[0].end == pytest.approx(20.0)


class TestSilenceConfig:
    """Tests for SilenceConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        cfg = SilenceConfig()
        assert cfg.noise_db == -30.0
        assert cfg.min_duration == 0.3

    def test_custom_values(self):
        """Test custom configuration values."""
        cfg = SilenceConfig(noise_db=-40.0, min_duration=0.5)
        assert cfg.noise_db == -40.0
        assert cfg.min_duration == 0.5


class TestSilenceInterval:
    """Tests for SilenceInterval dataclass."""

    def test_duration_property(self):
        """Test duration calculation."""
        interval = SilenceInterval(start=10.0, end=12.0)
        assert interval.duration == pytest.approx(2.0)

    def test_to_dict(self):
        """Test dictionary conversion."""
        interval = SilenceInterval(start=10.0, end=12.0)
        d = interval.to_dict()
        
        assert d["start"] == 10.0
        assert d["end"] == 12.0
