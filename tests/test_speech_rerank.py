"""Tests for speech analysis and candidate reranking."""
import numpy as np
import pytest

from videopipeline.analysis_speech_features import (
    SpeechFeatureConfig,
    compute_lexical_excitement,
    _count_exclamations_questions,
    _uppercase_ratio,
    _count_repeated_chars,
    _count_reaction_phrases,
)
from videopipeline.analysis_transcript import (
    FullTranscript,
    TranscriptSegment,
    TranscriptWord,
)
from videopipeline.rerank_candidates import (
    RerankConfig,
    extract_hook_text,
    extract_quote_text,
    _is_punchy,
    _has_payoff_word,
    _get_feature_value_in_range,
)


# ============================================================================
# Transcript slicing tests
# ============================================================================


def test_transcript_get_text_in_range():
    """Test extracting text from transcript within a time range."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="Hello world"),
        TranscriptSegment(start=2.5, end=4.0, text="This is a test"),
        TranscriptSegment(start=5.0, end=7.0, text="Another segment"),
    ]
    transcript = FullTranscript(segments=segments, duration_seconds=10.0)
    
    # Get text spanning multiple segments
    text = transcript.get_text_in_range(1.0, 5.5)
    assert "Hello world" in text
    assert "This is a test" in text
    assert "Another segment" in text
    
    # Get text from single segment
    text = transcript.get_text_in_range(0.5, 1.5)
    assert "Hello world" in text
    assert "This is a test" not in text
    
    # No overlap
    text = transcript.get_text_in_range(8.0, 9.0)
    assert text == ""


def test_transcript_get_segments_in_range():
    """Test getting segments that overlap with a time range."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="First"),
        TranscriptSegment(start=3.0, end=5.0, text="Second"),
        TranscriptSegment(start=6.0, end=8.0, text="Third"),
    ]
    transcript = FullTranscript(segments=segments, duration_seconds=10.0)
    
    # Get overlapping segments
    result = transcript.get_segments_in_range(1.5, 4.0)
    assert len(result) == 2
    assert result[0].text == "First"
    assert result[1].text == "Second"
    
    # Single segment
    result = transcript.get_segments_in_range(6.5, 7.5)
    assert len(result) == 1
    assert result[0].text == "Third"


def test_transcript_get_words_in_range():
    """Test getting words that fall within a time range."""
    words = [
        TranscriptWord(word="Hello", start=0.0, end=0.5),
        TranscriptWord(word="world", start=0.6, end=1.0),
        TranscriptWord(word="test", start=1.5, end=2.0),
    ]
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="Hello world test", words=words),
    ]
    transcript = FullTranscript(segments=segments, duration_seconds=3.0)
    
    # Get words in range
    result = transcript.get_words_in_range(0.0, 0.8)
    assert len(result) == 2
    assert result[0].word == "Hello"
    assert result[1].word == "world"
    
    # Partial overlap
    result = transcript.get_words_in_range(1.0, 1.8)
    assert len(result) == 1
    assert result[0].word == "test"


# ============================================================================
# Lexical excitement tests
# ============================================================================


def test_count_exclamations_questions():
    """Test counting exclamation and question marks."""
    assert _count_exclamations_questions("Hello!") == (1, 0)
    assert _count_exclamations_questions("What?") == (0, 1)
    assert _count_exclamations_questions("OMG!!! What?") == (3, 1)
    assert _count_exclamations_questions("Normal text") == (0, 0)


def test_uppercase_ratio():
    """Test calculating uppercase ratio."""
    assert _uppercase_ratio("hello") == 0.0
    assert _uppercase_ratio("HELLO") == 1.0
    assert _uppercase_ratio("Hello") == pytest.approx(0.2, rel=0.01)
    assert _uppercase_ratio("123") == 0.0  # No letters


def test_count_repeated_chars():
    """Test counting repeated character patterns."""
    assert _count_repeated_chars("nooooo") == 1
    assert _count_repeated_chars("whaaaat") == 1
    assert _count_repeated_chars("yessss nooooo") == 2
    assert _count_repeated_chars("normal") == 0


def test_count_reaction_phrases():
    """Test counting reaction phrases."""
    phrases = ["no way", "oh my god", "bro", "what"]
    
    assert _count_reaction_phrases("no way that happened", phrases) == 1
    assert _count_reaction_phrases("bro what", phrases) == 2
    assert _count_reaction_phrases("oh my god bro no way", phrases) == 3
    assert _count_reaction_phrases("hello world", phrases) == 0


def test_compute_lexical_excitement():
    """Test computing lexical excitement score."""
    phrases = ["no way", "bro", "what"]
    cfg = SpeechFeatureConfig(reaction_phrases=phrases)
    
    # High excitement text
    score_high = compute_lexical_excitement("NO WAY!!! WHAT?!", phrases, cfg)
    
    # Low excitement text
    score_low = compute_lexical_excitement("hello world", phrases, cfg)
    
    assert score_high > score_low
    assert score_high > 0
    assert score_low == 0


# ============================================================================
# Hook extraction tests
# ============================================================================


def test_is_punchy():
    """Test checking if text is punchy/hook-worthy."""
    phrases = ["no way", "bro", "what"]
    
    assert _is_punchy("NO WAY!", phrases) is True
    assert _is_punchy("What?", phrases) is True
    assert _is_punchy("bro that was crazy", phrases) is True
    assert _is_punchy("Short", phrases) is True  # Short text
    assert _is_punchy("This is a very long boring sentence without any excitement at all which makes it not punchy", phrases) is False


def test_extract_hook_text():
    """Test extracting hook text from transcript."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="NO WAY! That was insane!"),
        TranscriptSegment(start=3.0, end=5.0, text="I can't believe it."),
    ]
    transcript = FullTranscript(segments=segments, duration_seconds=6.0)
    cfg = RerankConfig(
        hook_max_chars=60,
        hook_window_seconds=4.0,
        reaction_phrases=["no way", "insane"],
    )
    
    hook = extract_hook_text(transcript, 0.0, 5.0, cfg)
    assert hook is not None
    assert "NO WAY" in hook or "insane" in hook.lower()


def test_extract_hook_text_fallback():
    """Test hook extraction fallback to first words."""
    segments = [
        TranscriptSegment(start=0.0, end=3.0, text="This is a normal sentence without excitement."),
    ]
    transcript = FullTranscript(segments=segments, duration_seconds=5.0)
    cfg = RerankConfig(
        hook_max_chars=30,
        hook_window_seconds=4.0,
        reaction_phrases=["no way"],
    )
    
    hook = extract_hook_text(transcript, 0.0, 4.0, cfg)
    # Should fallback to first few words
    assert hook is not None
    assert len(hook) <= 60


def test_extract_hook_text_empty():
    """Test hook extraction with no matching segments."""
    transcript = FullTranscript(segments=[], duration_seconds=5.0)
    cfg = RerankConfig()
    
    hook = extract_hook_text(transcript, 0.0, 4.0, cfg)
    assert hook is None


# ============================================================================
# Quote extraction tests
# ============================================================================


def test_has_payoff_word():
    """Test checking for payoff words."""
    assert _has_payoff_word("That was a clutch play!") is True
    assert _has_payoff_word("We won the game!") is True
    assert _has_payoff_word("I'm dead") is True
    assert _has_payoff_word("Hello world") is False


def test_extract_quote_text():
    """Test extracting best quote from transcript."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="Setting up the play."),
        TranscriptSegment(start=2.5, end=5.0, text="That was the most insane clutch ever!"),
        TranscriptSegment(start=6.0, end=8.0, text="Okay moving on."),
    ]
    transcript = FullTranscript(segments=segments, duration_seconds=10.0)
    cfg = RerankConfig(
        quote_max_chars=120,
        reaction_phrases=["insane", "clutch"],
    )
    
    quote = extract_quote_text(transcript, 0.0, 8.0, cfg)
    assert quote is not None
    # Should pick the most exciting sentence
    assert "insane" in quote.lower() or "clutch" in quote.lower()


def test_extract_quote_text_empty():
    """Test quote extraction with no segments."""
    transcript = FullTranscript(segments=[], duration_seconds=5.0)
    cfg = RerankConfig()
    
    quote = extract_quote_text(transcript, 0.0, 4.0, cfg)
    assert quote is None


# ============================================================================
# Rerank weighting tests
# ============================================================================


def test_get_feature_value_in_range_max():
    """Test getting max feature value in range."""
    feature = np.array([1.0, 2.0, 5.0, 3.0, 1.0])
    hop_s = 1.0
    
    value = _get_feature_value_in_range(feature, hop_s, 1.5, 3.5, "max")
    assert value == 5.0


def test_get_feature_value_in_range_mean():
    """Test getting mean feature value in range."""
    feature = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    hop_s = 1.0
    
    value = _get_feature_value_in_range(feature, hop_s, 1.0, 4.0, "mean")
    assert value == pytest.approx(8.0 / 3.0, rel=0.01)  # (2+4+2)/3


def test_get_feature_value_in_range_edge_cases():
    """Test feature value extraction edge cases."""
    feature = np.array([1.0, 2.0, 3.0])
    hop_s = 1.0
    
    # Range outside array
    value = _get_feature_value_in_range(feature, hop_s, 5.0, 6.0, "max")
    assert value == 3.0  # Clamps to last element
    
    # Empty array
    empty = np.array([])
    value = _get_feature_value_in_range(empty, hop_s, 0.0, 1.0, "max")
    assert value == 0.0


def test_rerank_config_defaults():
    """Test RerankConfig has sensible defaults."""
    cfg = RerankConfig()
    
    assert cfg.enabled is True
    assert cfg.weights["highlight"] == 0.55
    assert cfg.weights["reaction"] == 0.25
    assert cfg.weights["speech"] == 0.20
    assert cfg.hook_max_chars == 60
    assert cfg.hook_window_seconds == 4.0
    assert cfg.quote_max_chars == 120


def test_rerank_weights_sum():
    """Test that default rerank weights sum to 1."""
    cfg = RerankConfig()
    total = sum(cfg.weights.values())
    assert total == pytest.approx(1.0, rel=0.01)


# ============================================================================
# Serialization tests
# ============================================================================


def test_transcript_segment_roundtrip():
    """Test TranscriptSegment serialization roundtrip."""
    words = [
        TranscriptWord(word="Hello", start=0.0, end=0.5, probability=0.95),
        TranscriptWord(word="world", start=0.6, end=1.0, probability=0.98),
    ]
    segment = TranscriptSegment(start=0.0, end=1.0, text="Hello world", words=words)
    
    d = segment.to_dict()
    restored = TranscriptSegment.from_dict(d)
    
    assert restored.start == segment.start
    assert restored.end == segment.end
    assert restored.text == segment.text
    assert len(restored.words) == len(segment.words)
    assert restored.words[0].word == "Hello"


def test_full_transcript_roundtrip():
    """Test FullTranscript serialization roundtrip."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="First segment"),
        TranscriptSegment(start=3.0, end=5.0, text="Second segment"),
    ]
    transcript = FullTranscript(
        segments=segments,
        language="en",
        duration_seconds=10.0,
    )
    
    d = transcript.to_dict()
    restored = FullTranscript.from_dict(d)
    
    assert restored.language == transcript.language
    assert restored.duration_seconds == transcript.duration_seconds
    assert len(restored.segments) == len(transcript.segments)
    assert restored.segments[0].text == "First segment"
