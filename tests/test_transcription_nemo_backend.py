from __future__ import annotations

from pathlib import Path

import pytest

from videopipeline.transcription.base import BaseTranscriber, TranscriberConfig, TranscriptResult, TranscriptSegment
from videopipeline.transcription import factory
from videopipeline.transcription import nemo_asr_backend as nemo_mod


class _DummyNemoTranscriber(BaseTranscriber):
    @property
    def backend_name(self) -> str:
        return "nemo_asr"

    @property
    def gpu_available(self) -> bool:
        return True

    def _load_model(self):  # type: ignore[no-untyped-def]
        return object()

    def transcribe(self, audio_path: Path, *, on_progress=None) -> TranscriptResult:  # type: ignore[no-untyped-def]
        return TranscriptResult(
            segments=[TranscriptSegment(start=0.0, end=1.0, text="hello")],
            language="en",
            duration_seconds=1.0,
            backend_used=self.backend_name,
            gpu_used=True,
        )


def test_factory_auto_prefers_nemo_for_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "_get_nemo_asr_class", lambda: _DummyNemoTranscriber)
    monkeypatch.setattr(factory, "_get_openai_whisper_class", lambda: None)
    monkeypatch.setattr(factory, "_get_faster_whisper_class", lambda: None)
    monkeypatch.setattr(factory, "_get_whispercpp_class", lambda: None)

    cfg = TranscriberConfig(backend="auto", use_gpu=True)
    transcriber = factory.get_transcriber(cfg)
    assert transcriber.backend_name == "nemo_asr"


def test_available_backends_exposes_nemo_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_mod, "is_available", lambda: True)
    monkeypatch.setattr(nemo_mod, "check_gpu_available", lambda: True)

    backends = factory.get_available_backends()
    assert "nemo_asr" in backends
    assert "nemo_asr_gpu" in backends


def test_nemo_retries_without_word_timestamps_on_windows_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(nemo_mod, "is_available", lambda: True)

    cfg = TranscriberConfig(
        backend="nemo_asr",
        model="small.en",
        use_gpu=False,
        word_timestamps=True,
    )
    transcriber = nemo_mod.NemoASRTranscriber(cfg)

    # Avoid loading real NeMo model in unit test.
    monkeypatch.setattr(transcriber, "ensure_model_loaded", lambda: object())

    calls: list[bool] = []

    def _fake_run_transcribe(_model, _audio_path, *, with_hyp):  # type: ignore[no-untyped-def]
        calls.append(bool(with_hyp))
        if with_hyp:
            raise RuntimeError(
                "[WinError 32] The process cannot access the file because it is being used by another process"
            )
        return "retry succeeded"

    monkeypatch.setattr(transcriber, "_run_transcribe", _fake_run_transcribe)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"dummy")

    result = transcriber.transcribe(audio_path)
    assert calls == [True, False]
    assert result.segments
    assert result.segments[0].text == "retry succeeded"
