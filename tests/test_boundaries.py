"""Tests for analysis_boundaries module."""

import pytest

from videopipeline.analysis_boundaries import (
    BoundaryConfig,
    BoundaryPoint,
    BoundaryGraph,
)


class TestBoundaryConfig:
    """Tests for BoundaryConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        cfg = BoundaryConfig()
        assert cfg.prefer_silence is True
        assert cfg.prefer_sentences is True
        assert cfg.prefer_scene_cuts is True
        assert cfg.prefer_chat_valleys is True
        assert cfg.snap_tolerance_s == 1.0

    def test_custom_values(self):
        """Test custom configuration values."""
        cfg = BoundaryConfig(
            prefer_silence=False,
            snap_tolerance_s=2.0,
        )
        assert cfg.prefer_silence is False
        assert cfg.snap_tolerance_s == 2.0


class TestBoundaryPoint:
    """Tests for BoundaryPoint dataclass."""

    def test_basic_creation(self):
        """Test basic boundary point creation."""
        point = BoundaryPoint(
            time_s=10.5,
            sources={"silence", "sentence_end"},
            score=0.85,
        )
        
        assert point.time_s == pytest.approx(10.5)
        assert point.score == pytest.approx(0.85)
        assert "silence" in point.sources
        assert "sentence_end" in point.sources

    def test_to_dict(self):
        """Test dictionary conversion."""
        point = BoundaryPoint(
            time_s=10.5,
            sources={"silence"},
            score=0.85,
        )
        d = point.to_dict()
        
        assert d["time_s"] == 10.5
        assert d["score"] == 0.85
        assert "silence" in d["sources"]


class TestBoundaryGraph:
    """Tests for BoundaryGraph dataclass."""

    def test_basic_creation(self):
        """Test basic graph creation."""
        start_points = [
            BoundaryPoint(time_s=10.0, score=0.8, sources={"silence"}),
            BoundaryPoint(time_s=20.0, score=0.9, sources={"sentence_end"}),
        ]
        end_points = [
            BoundaryPoint(time_s=15.0, score=0.7, sources={"silence"}),
        ]
        graph = BoundaryGraph(start_boundaries=start_points, end_boundaries=end_points, duration_s=30.0)
        
        assert len(graph.start_boundaries) == 2
        assert len(graph.end_boundaries) == 1

    def test_to_dict(self):
        """Test dictionary conversion."""
        start_points = [
            BoundaryPoint(time_s=10.0, score=0.8, sources={"silence"}),
        ]
        graph = BoundaryGraph(start_boundaries=start_points, end_boundaries=[], duration_s=30.0)
        d = graph.to_dict()
        
        assert len(d["start_boundaries"]) == 1
        assert d["start_boundaries"][0]["time_s"] == 10.0
        assert d["duration_s"] == 30.0
