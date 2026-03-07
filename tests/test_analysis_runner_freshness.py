from __future__ import annotations

from pathlib import Path

from videopipeline.analysis.artifacts import save_artifact_state
from videopipeline.analysis.runner import AnalysisRunner
from videopipeline.analysis.tasks import Task
from videopipeline.project import Project


def _noop_task_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
    return None


def _make_project(tmp_path: Path) -> Project:
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir(parents=True, exist_ok=True)
    proj = Project(project_dir=proj_dir, video_path=tmp_path / "video.mp4")
    proj.analysis_dir.mkdir(parents=True, exist_ok=True)
    return proj


def test_should_rerun_when_audio_config_changes(tmp_path: Path) -> None:
    proj = _make_project(tmp_path)
    proj.audio_features_path.write_bytes(b"present")

    old_cfg = {"audio": {"hop_seconds": 0.5, "smooth_seconds": 3.0}}
    new_cfg = {"audio": {"hop_seconds": 1.0, "smooth_seconds": 3.0}}
    save_artifact_state("audio_features", proj.analysis_dir, config=old_cfg, task_version="1.0")

    runner = AnalysisRunner(proj, config=new_cfg)
    task = Task(
        name="audio_features",
        requires=set(),
        produces={"audio_features"},
        run=_noop_task_run,
        version="1.0",
    )

    should_run, reason = runner._should_run_task(task)
    assert should_run is True
    assert "stale outputs" in reason


def test_should_rerun_when_task_version_changes(tmp_path: Path) -> None:
    proj = _make_project(tmp_path)
    proj.audio_features_path.write_bytes(b"present")

    cfg = {"audio": {"hop_seconds": 0.5}}
    save_artifact_state("audio_features", proj.analysis_dir, config=cfg, task_version="1.0")

    runner = AnalysisRunner(proj, config=cfg)
    task = Task(
        name="audio_features",
        requires=set(),
        produces={"audio_features"},
        run=_noop_task_run,
        version="2.0",
    )

    should_run, reason = runner._should_run_task(task)
    assert should_run is True
    assert "stale outputs" in reason


def test_should_rerun_when_metadata_is_missing(tmp_path: Path) -> None:
    proj = _make_project(tmp_path)
    proj.audio_features_path.write_bytes(b"present")

    runner = AnalysisRunner(proj, config={"audio": {"hop_seconds": 0.5}})
    task = Task(
        name="audio_features",
        requires=set(),
        produces={"audio_features"},
        run=_noop_task_run,
        version="1.0",
    )

    should_run, reason = runner._should_run_task(task)
    assert should_run is True
    assert "stale outputs" in reason


def test_transcript_freshness_uses_speech_config(tmp_path: Path) -> None:
    proj = _make_project(tmp_path)
    proj.transcript_path.write_text("{}", encoding="utf-8")

    old_cfg = {"speech": {"backend": "auto", "model_size": "small"}}
    new_cfg = {"speech": {"backend": "nemo_asr", "model_size": "small"}}
    save_artifact_state("transcript", proj.analysis_dir, config=old_cfg, task_version="1.0")

    runner = AnalysisRunner(proj, config=new_cfg)
    task = Task(
        name="transcript",
        requires=set(),
        produces={"transcript"},
        run=_noop_task_run,
        version="1.0",
    )

    should_run, reason = runner._should_run_task(task)
    assert should_run is True
    assert "stale outputs" in reason
