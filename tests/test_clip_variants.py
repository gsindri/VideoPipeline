"""Tests for clip_variants module."""

import pytest

from videopipeline.clip_variants import (
    VariantGeneratorConfig,
    VariantDurationConfig,
    ClipVariant,
    CandidateVariants,
)


class TestVariantDurationConfig:
    """Tests for VariantDurationConfig dataclass."""

    def test_basic_creation(self):
        """Test basic duration config creation."""
        cfg = VariantDurationConfig(min_s=16.0, max_s=24.0)
        assert cfg.min_s == 16.0
        assert cfg.max_s == 24.0


class TestVariantGeneratorConfig:
    """Tests for VariantGeneratorConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        cfg = VariantGeneratorConfig()
        assert cfg.short.min_s == 16.0
        assert cfg.short.max_s == 24.0
        assert cfg.medium.min_s == 24.0
        assert cfg.medium.max_s == 40.0
        assert cfg.long.min_s == 40.0
        assert cfg.long.max_s == 75.0

    def test_custom_short(self):
        """Test custom short duration values."""
        cfg = VariantGeneratorConfig(
            short=VariantDurationConfig(10.0, 20.0)
        )
        assert cfg.short.min_s == 10.0
        assert cfg.short.max_s == 20.0


class TestClipVariant:
    """Tests for ClipVariant dataclass."""

    def test_basic_creation(self):
        """Test basic variant creation."""
        variant = ClipVariant(
            variant_id="short_A",
            start_s=10.0,
            end_s=25.0,
            duration_s=15.0,
            description="Short clip variant",
        )
        
        assert variant.variant_id == "short_A"
        assert variant.start_s == 10.0
        assert variant.end_s == 25.0
        assert variant.duration_s == 15.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        variant = ClipVariant(
            variant_id="short_A",
            start_s=10.0,
            end_s=25.0,
            duration_s=15.0,
            description="Test variant",
            setup_text="Setup context",
            payoff_text="Big moment",
        )
        d = variant.to_dict()
        
        assert d["variant_id"] == "short_A"
        assert d["start_s"] == 10.0
        assert d["end_s"] == 25.0
        assert d["duration_s"] == 15.0
        assert d["description"] == "Test variant"
        assert d["setup_text"] == "Setup context"
        assert d["payoff_text"] == "Big moment"

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "variant_id": "medium_B",
            "start_s": 20.0,
            "end_s": 50.0,
            "duration_s": 30.0,
            "description": "Medium clip",
        }
        variant = ClipVariant.from_dict(d)
        
        assert variant.variant_id == "medium_B"
        assert variant.start_s == 20.0
        assert variant.duration_s == 30.0


class TestCandidateVariants:
    """Tests for CandidateVariants dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        variants = CandidateVariants(
            candidate_rank=1,
            candidate_peak_time_s=120.0,
            variants=[
                ClipVariant(
                    variant_id="short_A",
                    start_s=10.0,
                    end_s=25.0,
                    duration_s=15.0,
                )
            ],
        )
        d = variants.to_dict()
        
        assert d["candidate_rank"] == 1
        assert d["candidate_peak_time_s"] == 120.0
        assert len(d["variants"]) == 1
        assert d["variants"][0]["variant_id"] == "short_A"
