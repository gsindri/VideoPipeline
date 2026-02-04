from __future__ import annotations

from typing import Any, Dict, Optional

import pytest


def test_configure_pyannote_batches_uses_instantiate_nested_config() -> None:
    from videopipeline.transcription import diarization as dia

    calls: list[Dict[str, Any]] = []

    class FakePipeline:
        def instantiate(self, cfg: Dict[str, Any]) -> None:
            calls.append(cfg)

    ok = dia._configure_pyannote_batches(
        FakePipeline(),
        segmentation_batch_size=8,
        embedding_batch_size=4,
    )

    assert ok is True
    assert calls
    assert calls[0] == {"segmentation": {"batch_size": 8}, "embedding": {"batch_size": 4}}


def test_configure_pyannote_batches_returns_false_without_instantiate() -> None:
    from videopipeline.transcription import diarization as dia

    class FakePipeline:
        pass

    ok = dia._configure_pyannote_batches(
        FakePipeline(),
        segmentation_batch_size=8,
        embedding_batch_size=None,
    )

    assert ok is False


def test_autotune_batch_sizes_retries_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    from videopipeline.transcription import diarization as dia

    # Ensure a clean cache for this test.
    monkeypatch.setattr(dia, "_batch_size_cache", {})

    class FakePipeline:
        def __init__(self) -> None:
            self.instantiate_calls: list[Dict[str, Any]] = []
            self.bs: Optional[int] = None

        def instantiate(self, cfg: Dict[str, Any]) -> None:
            self.instantiate_calls.append(cfg)
            bs = None
            if isinstance(cfg.get("segmentation"), dict):
                bs = cfg["segmentation"].get("batch_size")
            if bs is None:
                bs = cfg.get("segmentation_batch_size")
            self.bs = int(bs) if bs is not None else None

        def __call__(self, _: Dict[str, Any], **__: Any) -> object:
            if self.bs is None:
                raise RuntimeError("missing batch size")
            if self.bs > 8:
                raise RuntimeError("CUDA out of memory")
            return object()

    p = FakePipeline()

    tuned = dia._autotune_batch_sizes(
        p,
        cache_key="model|device",
        probe_input={"waveform": object(), "sample_rate": 16000},
        diarization_params={},
        candidates=(32, 16, 8, 4),
    )
    assert tuned == (8, 8)
    assert len(p.instantiate_calls) == 3  # 32 -> 16 -> 8

    tuned2 = dia._autotune_batch_sizes(
        p,
        cache_key="model|device",
        probe_input={"waveform": object(), "sample_rate": 16000},
        diarization_params={},
        candidates=(32, 16, 8, 4),
    )
    assert tuned2 == (8, 8)
    assert len(p.instantiate_calls) == 3  # cached; no additional calls

