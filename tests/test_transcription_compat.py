from __future__ import annotations

import sys
import types

import pytest


def _make_package(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    # Mark as package so `import x.y` works.
    mod.__path__ = []  # type: ignore[attr-defined]
    return mod


def test_whisper_triton_stub_supports_kernel_launch_syntax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure our stub matches Triton's `kernel[grid](...)` calling convention."""
    # Arrange: a dummy `whisper` package with no `triton_ops` submodule.
    monkeypatch.setitem(sys.modules, "whisper", _make_package("whisper"))
    monkeypatch.delitem(sys.modules, "whisper.triton_ops", raising=False)

    from videopipeline.transcription import openai_whisper_backend as mod

    # Act: run the patch (safe to call more than once).
    mod._patch_whisper_triton_import()

    # Assert: the stub exists and fails with RuntimeError (not TypeError).
    triton_ops = sys.modules.get("whisper.triton_ops")
    assert triton_ops is not None
    with pytest.raises(RuntimeError):
        triton_ops.dtw_kernel[None]()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        triton_ops.median_filter_cuda[None]()  # type: ignore[attr-defined]


def test_pyannote_pipeline_from_pretrained_prefers_token_kw(monkeypatch: pytest.MonkeyPatch) -> None:
    """pyannote-audio>=3.1 switched from `use_auth_token` to `token`."""
    from videopipeline.transcription import diarization as dia

    dia.unload_diarization_pipeline()

    calls: dict[str, object] = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, *, token: str | None = None):
            calls["model_id"] = model_id
            calls["token"] = token
            return object()

    pyannote_pkg = _make_package("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = Pipeline  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)

    dia._load_diarization_pipeline(hf_token="abc123", use_gpu=False)

    assert calls["model_id"] == "pyannote/speaker-diarization-3.1"
    assert calls["token"] == "abc123"


def test_pyannote_pipeline_from_pretrained_falls_back_to_use_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Older pyannote versions only accept `use_auth_token`."""
    from videopipeline.transcription import diarization as dia

    dia.unload_diarization_pipeline()

    calls: dict[str, object] = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, *, use_auth_token: str | None = None):
            calls["model_id"] = model_id
            calls["use_auth_token"] = use_auth_token
            return object()

    pyannote_pkg = _make_package("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = Pipeline  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)

    dia._load_diarization_pipeline(hf_token="abc123", use_gpu=False)

    assert calls["model_id"] == "pyannote/speaker-diarization-3.1"
    assert calls["use_auth_token"] == "abc123"

