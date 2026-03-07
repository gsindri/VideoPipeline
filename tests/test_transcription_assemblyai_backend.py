from __future__ import annotations

from pathlib import Path

import pytest

from videopipeline.transcription import factory
from videopipeline.transcription.base import (
    BackendNotAvailableError,
    BaseTranscriber,
    TranscriberConfig,
    TranscriptResult,
    TranscriptSegment,
)


class _DummyAssemblyAITranscriber(BaseTranscriber):
    @property
    def backend_name(self) -> str:
        return "assemblyai"

    @property
    def gpu_available(self) -> bool:
        return False

    def _load_model(self):  # type: ignore[no-untyped-def]
        return object()

    def transcribe(self, audio_path: Path, *, on_progress=None) -> TranscriptResult:  # type: ignore[no-untyped-def]
        return TranscriptResult(
            segments=[TranscriptSegment(start=0.0, end=1.0, text="hello from cloud")],
            language="en",
            duration_seconds=1.0,
            backend_used=self.backend_name,
            gpu_used=False,
            speakers=["A"],
            diarization_used=True,
        )


def test_factory_explicit_assemblyai_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "_get_assemblyai_class", lambda: _DummyAssemblyAITranscriber)

    cfg = TranscriberConfig(backend="assemblyai", strict=True)
    transcriber = factory.get_transcriber(cfg)
    assert transcriber.backend_name == "assemblyai"


def test_factory_auto_does_not_pick_assemblyai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "_get_nemo_asr_class", lambda: None)
    monkeypatch.setattr(factory, "_get_openai_whisper_class", lambda: None)
    monkeypatch.setattr(factory, "_get_faster_whisper_class", lambda: None)
    monkeypatch.setattr(factory, "_get_whispercpp_class", lambda: None)
    monkeypatch.setattr(factory, "_get_assemblyai_class", lambda: _DummyAssemblyAITranscriber)

    cfg = TranscriberConfig(backend="auto", use_gpu=False)
    with pytest.raises(BackendNotAvailableError):
        factory.get_transcriber(cfg)


def test_available_backends_exposes_assemblyai_keys() -> None:
    backends = factory.get_available_backends()
    assert "assemblyai" in backends
    assert "assemblyai_gpu" in backends
