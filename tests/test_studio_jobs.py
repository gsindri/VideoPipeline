from __future__ import annotations

import time
from pathlib import Path

from videopipeline.exporter import ExportCancelledError
from videopipeline.project import Project
from videopipeline.studio.jobs import JobManager


def test_start_export_preserves_cancelled_terminal_state(tmp_path: Path, monkeypatch):
    manager = JobManager()

    def fake_run_ffmpeg_export(spec, *, on_progress=None, two_pass_loudnorm=True, check_cancel=None):
        if on_progress:
            on_progress(0.1, "encoding")
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if check_cancel and check_cancel():
                raise ExportCancelledError("cancelled")
            time.sleep(0.01)
        raise AssertionError("export did not observe cancellation")

    monkeypatch.setattr("videopipeline.studio.jobs.run_ffmpeg_export", fake_run_ffmpeg_export)
    monkeypatch.setattr("videopipeline.studio.jobs.get_project_data", lambda proj: {"layout": {}})
    monkeypatch.setattr("videopipeline.studio.jobs.get_facecam_rect", lambda layout: None)
    monkeypatch.setattr("videopipeline.studio.jobs.build_metadata", lambda **kwargs: {"title": "x"})
    monkeypatch.setattr(
        "videopipeline.studio.jobs.write_metadata",
        lambda path, metadata: path.write_text("ok", encoding="utf-8"),
    )
    monkeypatch.setattr("videopipeline.studio.jobs.record_export", lambda *args, **kwargs: None)

    proj_dir = tmp_path / "proj"
    video_dir = proj_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "video.mp4"
    video_path.write_bytes(b"video")
    proj = Project(project_dir=proj_dir, video_path=video_path)

    job = manager.start_export(
        proj=proj,
        selection={"id": "sel1", "start_s": 0.0, "end_s": 1.0},
        export_dir=proj.exports_dir,
        with_captions=False,
        template="vertical_blur",
        width=1080,
        height=1920,
        fps=30,
        crf=20,
        preset="veryfast",
        normalize_audio=False,
    )

    time.sleep(0.1)
    assert manager.cancel(job.id) is True

    deadline = time.time() + 2.0
    while time.time() < deadline:
        current = manager.get(job.id)
        if current is not None and current.status == "cancelled":
            break
        time.sleep(0.01)

    current = manager.get(job.id)
    assert current is not None
    assert current.cancel_requested is True
    assert current.status == "cancelled"
    assert not any(proj.exports_dir.glob("*.mp4"))
