from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_run_transcript_propagates_strict_and_runtime_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from videopipeline.analysis import tasks as task_mod
    import videopipeline.analysis_transcript as transcript_mod

    captured: dict[str, object] = {}

    def _fake_compute_transcript_analysis(
        proj, *, cfg, on_progress=None, force=False, source_audio_path=None
    ):  # type: ignore[no-untyped-def]
        captured["cfg"] = cfg
        return {"ok": True}

    monkeypatch.setattr(transcript_mod, "compute_transcript_analysis", _fake_compute_transcript_analysis)

    proj = SimpleNamespace(analysis_dir=tmp_path)
    cfg = {
        "speech": {
            "backend": "nemo_asr",
            "model_size": "small.en",
            "language": "en",
            "sample_rate": 22050,
            "device": "cuda",
            "compute_type": "float16",
            "vad_filter": True,
            "word_timestamps": True,
            "use_gpu": True,
            "threads": 4,
            "n_processors": 2,
            "strict": True,
            "verbose": True,
        }
    }

    task_mod._run_transcript(proj, cfg, on_progress=None)

    tc = captured["cfg"]
    assert getattr(tc, "backend") == "nemo_asr"
    assert getattr(tc, "sample_rate") == 22050
    assert getattr(tc, "threads") == 4
    assert getattr(tc, "n_processors") == 2
    assert getattr(tc, "strict") is True


def test_run_transcript_defaults_strict_false(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from videopipeline.analysis import tasks as task_mod
    import videopipeline.analysis_transcript as transcript_mod

    captured: dict[str, object] = {}

    def _fake_compute_transcript_analysis(
        proj, *, cfg, on_progress=None, force=False, source_audio_path=None
    ):  # type: ignore[no-untyped-def]
        captured["cfg"] = cfg
        return {"ok": True}

    monkeypatch.setattr(transcript_mod, "compute_transcript_analysis", _fake_compute_transcript_analysis)

    proj = SimpleNamespace(analysis_dir=tmp_path)
    task_mod._run_transcript(proj, {"speech": {}}, on_progress=None)

    assert getattr(captured["cfg"], "strict") is False
