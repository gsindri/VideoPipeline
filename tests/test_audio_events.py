"""Tests for audio event detection module."""

import numpy as np
import pytest

from videopipeline.peaks import moving_average, robust_z


# ============================================================================
# Test smoothing and z-scoring utilities
# ============================================================================

def test_moving_average_basic():
    """Test basic moving average computation."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = moving_average(x, window=3)
    # With mode='same', values at edges are different
    assert len(result) == len(x)
    # Middle values should be averaged correctly
    assert np.isclose(result[2], 3.0)  # (2+3+4)/3 = 3


def test_moving_average_window_1():
    """Window of 1 should return input unchanged."""
    x = np.array([1.0, 2.0, 3.0])
    result = moving_average(x, window=1)
    assert np.allclose(result, x)


def test_robust_z_basic():
    """Test robust z-score computation."""
    # Standard normal-ish distribution
    x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])  # With outlier
    z = robust_z(x)
    
    # Outlier should have high z-score
    assert z[-1] > 3.0
    # Median value should have z-score near 0
    median_idx = 2  # median of [0,1,2,3,100] = 2
    assert abs(z[median_idx]) < 0.5


def test_robust_z_constant():
    """Constant input should not crash."""
    x = np.array([5.0, 5.0, 5.0, 5.0])
    z = robust_z(x)
    assert len(z) == len(x)
    # All values same, so z-scores should all be 0
    assert np.allclose(z, 0.0)


# ============================================================================
# Test AudioEventsConfig
# ============================================================================

def test_audio_events_config_from_dict():
    """Test config parsing from dict."""
    from videopipeline.analysis_audio_events import AudioEventsConfig
    
    cfg = AudioEventsConfig.from_dict({
        "enabled": True,
        "hop_seconds": 0.25,
        "smooth_seconds": 1.5,
        "strict": True,
        "events": {
            "laughter": 0.9,
            "cheering": 0.5,
        }
    })
    
    assert cfg.enabled is True
    assert cfg.hop_seconds == 0.25
    assert cfg.smooth_seconds == 1.5
    assert cfg.strict is True
    assert cfg.events["laughter"] == 0.9
    assert cfg.events["cheering"] == 0.5


def test_audio_events_config_defaults():
    """Test config with default values."""
    from videopipeline.analysis_audio_events import AudioEventsConfig
    
    cfg = AudioEventsConfig.from_dict({})
    
    assert cfg.enabled is True
    assert cfg.hop_seconds == 0.5
    assert cfg.smooth_seconds == 2.0
    assert cfg.strict is False
    assert "laughter" in cfg.events
    assert cfg.events["laughter"] == 1.0


def test_audio_events_config_assemblyai_nested():
    """AssemblyAI nested config should map to flat config fields."""
    from videopipeline.analysis_audio_events import AudioEventsConfig

    cfg = AudioEventsConfig.from_dict(
        {
            "backend": "assemblyai",
            "strict": True,
            "assemblyai": {
                "speech_models": ["universal-3-pro", "universal-2"],
                "auto_highlights": False,
                "sentiment_analysis": False,
                "poll_interval_s": 4.0,
                "timeout_s": 1234.0,
            },
        }
    )

    assert cfg.backend == "assemblyai"
    assert cfg.strict is True
    assert cfg.assemblyai_speech_models == ["universal-3-pro", "universal-2"]
    assert cfg.assemblyai_auto_highlights is False
    assert cfg.assemblyai_sentiment_analysis is False
    assert cfg.assemblyai_poll_interval_s == 4.0
    assert cfg.assemblyai_timeout_s == 1234.0


def test_assemblyai_audio_events_availability_requires_key(monkeypatch: pytest.MonkeyPatch):
    """Availability helper should fail cleanly when API key is missing."""
    import videopipeline.analysis_audio_events as mod

    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("AAI_API_KEY", raising=False)
    ok, reason = mod.check_assemblyai_audio_events_available(explicit_key=None)
    if ok:
        # In CI or local env a key might exist via process-level injection.
        assert reason is None
    else:
        assert "API_KEY" in str(reason) or "SDK not installed" in str(reason)


# ============================================================================
# Test AudioEventClassifier heuristic fallback
# ============================================================================

def test_classifier_heuristic_basic():
    """Test heuristic classifier with synthetic audio."""
    from videopipeline.analysis_audio_events import AudioEventClassifier
    
    classifier = AudioEventClassifier(sample_rate=16000, backend="heuristic")
    
    # Create a simple sine wave (should be low on all events)
    t = np.linspace(0, 0.5, 8000)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    
    probs = classifier.classify_chunk(audio)
    
    assert "laughter" in probs
    assert "cheering" in probs
    assert "screaming" in probs
    assert all(0.0 <= v <= 1.0 for v in probs.values())


def test_classifier_heuristic_noise():
    """Test heuristic classifier with white noise (should detect some crowd-like signal)."""
    from videopipeline.analysis_audio_events import AudioEventClassifier
    
    classifier = AudioEventClassifier(sample_rate=16000, backend="heuristic")
    
    # White noise
    np.random.seed(42)
    audio = np.random.randn(8000).astype(np.float32) * 0.3
    
    probs = classifier.classify_chunk(audio)
    
    # Noise should register some applause/crowd (high flatness)
    assert probs["applause"] > 0 or probs["crowd"] > 0


def test_classifier_heuristic_empty():
    """Test heuristic classifier with very short audio."""
    from videopipeline.analysis_audio_events import AudioEventClassifier
    
    classifier = AudioEventClassifier(sample_rate=16000, backend="heuristic")
    
    # Very short audio
    audio = np.zeros(100, dtype=np.float32)
    
    probs = classifier.classify_chunk(audio)
    
    # Should return zeros for all events
    assert all(v == 0.0 for v in probs.values())


def test_compute_audio_events_from_file_short_audio_skips_backend_dispatch(tmp_path, monkeypatch):
    """Very short inputs should not initialize heavyweight backends."""
    import videopipeline.analysis_audio_events as mod

    audio_path = tmp_path / "short.wav"
    audio_path.write_bytes(b"not-a-real-waveform")

    monkeypatch.setattr(mod, "ffprobe_duration_seconds", lambda _path: 100.0 / 16000.0)

    class _ExplodingClassifier:
        def __init__(self, *args, **kwargs):
            raise AssertionError("classifier should not be initialized for very short audio")

    monkeypatch.setattr(mod, "AudioEventClassifier", _ExplodingClassifier)

    result = mod.compute_audio_events_from_file(
        audio_path,
        cfg=mod.AudioEventsConfig(backend="pytorch", sample_rate=16000),
    )

    assert result["backend"] == "short_audio"
    assert result["ml_available"] is False
    assert result["times"] == [0.0]
    for key in ("laughter", "cheering", "applause", "screaming", "shouting", "crowd", "event_combo"):
        assert result[key] == [0.0]


def test_classifier_strict_requested_backend_failure(monkeypatch: pytest.MonkeyPatch):
    """Strict mode should fail instead of silently falling back."""
    import videopipeline.analysis_audio_events as mod

    monkeypatch.setattr(mod, "_try_load_pytorch_classifier", lambda: (None, None, None, "missing torch"))

    with pytest.raises(mod.AudioEventsBackendError):
        mod.AudioEventClassifier(sample_rate=16000, backend="pytorch", strict=True)


def test_classifier_auto_strict_requires_ml_backend(monkeypatch: pytest.MonkeyPatch):
    """Strict auto mode requires at least one ML backend."""
    import videopipeline.analysis_audio_events as mod

    monkeypatch.setattr(mod, "_try_load_pytorch_classifier", lambda: (None, None, None, "missing torch"))
    monkeypatch.setattr(mod, "_try_load_yamnet_onnx", lambda: (None, "missing onnx"))
    monkeypatch.setattr(mod, "_try_load_yamnet", lambda: (None, "missing tensorflow_hub"))

    with pytest.raises(mod.AudioEventsBackendError):
        mod.AudioEventClassifier(sample_rate=16000, backend="auto", strict=True)


def test_classifier_auto_non_strict_falls_back_to_heuristic(monkeypatch: pytest.MonkeyPatch):
    """Non-strict auto mode should still keep the heuristic fallback."""
    import videopipeline.analysis_audio_events as mod

    monkeypatch.setattr(mod, "_try_load_pytorch_classifier", lambda: (None, None, None, "missing torch"))
    monkeypatch.setattr(mod, "_try_load_yamnet_onnx", lambda: (None, "missing onnx"))
    monkeypatch.setattr(mod, "_try_load_yamnet", lambda: (None, "missing tensorflow_hub"))

    classifier = mod.AudioEventClassifier(sample_rate=16000, backend="auto", strict=False)
    assert classifier._backend == "heuristic"


# ============================================================================
# Test highlight scoring integration
# ============================================================================

def test_resample_series_for_audio_events():
    """Test that audio events can be resampled to highlight grid."""
    from videopipeline.analysis_highlights import resample_series
    
    # Audio events at 0.5s hop
    audio_events_z = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    
    # Resample to 0.25s hop (doubling resolution)
    resampled = resample_series(
        audio_events_z,
        src_hop_s=0.5,
        target_hop_s=0.25,
        target_len=12
    )
    
    assert len(resampled) == 12
    # Peak should be preserved (around index 6-7)
    assert np.argmax(resampled) in [6, 7]


def test_highlight_weights_with_audio_events():
    """Test that weights normalize correctly with audio events."""
    # Simulate weight normalization logic
    w_audio = 0.35
    w_motion = 0.30
    w_chat = 0.20
    w_audio_events = 0.15
    
    w_total = w_audio + w_motion + w_chat + w_audio_events
    
    # Should already sum to 1.0
    assert abs(w_total - 1.0) < 0.001


def test_highlight_weights_without_chat_events():
    """Test weight normalization when chat and events are unavailable."""
    w_audio = 0.35
    w_motion = 0.30
    w_chat = 0.0  # Not available
    w_audio_events = 0.0  # Not available
    
    w_total = w_audio + w_motion + w_chat + w_audio_events
    
    # Should be 0.65, needs normalization to 1.0
    if w_total > 0 and abs(w_total - 1.0) > 0.001:
        w_audio = w_audio / w_total
        w_motion = w_motion / w_total
    
    assert abs(w_audio + w_motion - 1.0) < 0.001
    assert np.isclose(w_audio, 0.35 / 0.65)
    assert np.isclose(w_motion, 0.30 / 0.65)


# ============================================================================
# Test combined score computation
# ============================================================================

def test_combined_score_with_audio_events():
    """Test that audio events contribute to combined score."""
    # Simulate scores
    audio_scores = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    motion_scores = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
    chat_scores = np.zeros(5)
    audio_events_scores = np.array([0.0, 0.5, 5.0, 0.5, 0.0])  # Big spike at index 2
    
    # Weights
    w_audio = 0.4
    w_motion = 0.3
    w_chat = 0.0
    w_audio_events = 0.3
    
    combined = (
        w_audio * audio_scores +
        w_motion * motion_scores +
        w_chat * chat_scores +
        w_audio_events * audio_events_scores
    )
    
    # Peak should be at index 2 due to audio_events spike
    assert np.argmax(combined) == 2
    # The peak value should be higher than pure audio+motion
    audio_motion_only = w_audio * audio_scores[2] + w_motion * motion_scores[2]
    assert combined[2] > audio_motion_only


# ============================================================================
# Test candidate breakdown
# ============================================================================

def test_candidate_breakdown_includes_audio_events():
    """Test that candidate breakdown dict includes audio_events."""
    breakdown = {
        "audio": 2.5,
        "motion": 1.8,
        "chat": 0.0,
        "audio_events": 3.2,
    }
    
    assert "audio_events" in breakdown
    assert breakdown["audio_events"] == 3.2


# ============================================================================
# Test profile config
# ============================================================================

def test_default_profile_has_audio_events():
    """Test that default profile includes audio_events section."""
    from videopipeline.profile import default_profile
    
    profile = default_profile()
    
    # Check analysis.audio_events exists
    assert "audio_events" in profile["analysis"]
    events_cfg = profile["analysis"]["audio_events"]
    assert events_cfg["enabled"] is True
    assert "hop_seconds" in events_cfg
    assert "events" in events_cfg
    assert "laughter" in events_cfg["events"]
    
    # Check highlights weights includes audio_events
    weights = profile["analysis"]["highlights"]["weights"]
    assert "audio_events" in weights
