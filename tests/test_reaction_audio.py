"""Tests for reaction audio analysis module."""

import pytest


def test_reaction_audio_config_from_dict():
    """Config parsing from dict should coerce types and allow 'enabled' key."""
    from videopipeline.analysis_reaction_audio import ReactionAudioConfig

    cfg = ReactionAudioConfig.from_dict(
        {
            "enabled": True,
            "sample_rate": "16000",
            "hop_seconds": "0.25",
            "smooth_seconds": 2.0,
        }
    )

    assert cfg.sample_rate == 16000
    assert cfg.hop_seconds == 0.25
    assert cfg.smooth_seconds == 2.0


def test_reaction_audio_config_defaults():
    """Empty dict should use defaults."""
    from videopipeline.analysis_reaction_audio import ReactionAudioConfig

    cfg = ReactionAudioConfig.from_dict({})
    assert cfg.sample_rate == 16000
    assert cfg.hop_seconds == 0.5
    assert cfg.smooth_seconds == 1.5


def test_reaction_audio_config_unknown_keys_raise():
    """Unknown keys should fail fast to avoid silently ignoring config."""
    from videopipeline.analysis_reaction_audio import ReactionAudioConfig

    with pytest.raises(ValueError, match="Unknown reaction_audio config keys"):
        ReactionAudioConfig.from_dict({"sample_rate": 16000, "bogus": 1})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sample_rate", 0),
        ("hop_seconds", 0.0),
        ("smooth_seconds", 0.0),
    ],
)
def test_reaction_audio_config_invalid_values_raise(key: str, value: object):
    """Non-positive numeric config values should raise."""
    from videopipeline.analysis_reaction_audio import ReactionAudioConfig

    with pytest.raises(ValueError):
        ReactionAudioConfig.from_dict({key: value})

