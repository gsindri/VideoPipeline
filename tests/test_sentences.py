"""Tests for analysis_sentences module."""

import pytest

from videopipeline.analysis_sentences import (
    SentenceConfig,
    Sentence,
)


class TestSentence:
    """Tests for Sentence dataclass."""

    def test_duration_property(self):
        """Test duration calculation."""
        sentence = Sentence(
            t0=10.0,
            t1=15.0,
            text="Hello world",
        )
        assert sentence.duration == pytest.approx(5.0)

    def test_to_dict(self):
        """Test dictionary conversion."""
        sentence = Sentence(
            t0=10.0,
            t1=12.0,
            text="Hello world",
        )
        d = sentence.to_dict()
        
        assert d["t0"] == 10.0
        assert d["t1"] == 12.0
        assert d["text"] == "Hello world"


class TestSentenceConfig:
    """Tests for SentenceConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        cfg = SentenceConfig()
        assert cfg.max_sentence_words == 30
        assert cfg.sentence_end_chars == ".!?"

    def test_custom_values(self):
        """Test custom configuration values."""
        cfg = SentenceConfig(max_sentence_words=20, sentence_end_chars=".!?;")
        assert cfg.max_sentence_words == 20
        assert cfg.sentence_end_chars == ".!?;"
