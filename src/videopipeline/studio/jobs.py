from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..exporter import ExportSpec, run_ffmpeg_export
from ..project import Project, record_export
from ..subtitles import write_ass
from ..transcribe import TranscribeConfig, WhisperNotInstalledError, load_transcript_json, save_transcript_json, transcribe_segment


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    kind: str
    created_at: str = field(default_factory=_utc_iso)
    status: str = "queued"  # queued|running|succeeded|failed
    progress: float = 0.0
    message: str = ""
    result: Dict[str, Any] = field(default_factory=dict)

    # SSE event stream
    events: "queue.Queue[str]" = field(default_factory=lambda: queue.Queue(maxsize=1000))


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def create(self, kind: str) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job
        self._emit(job, {"type": "job_created", "job": self._public(job)})
        return job

    def _public(self, job: Job) -> Dict[str, Any]:
        return {
            "id": job.id,
            "kind": job.kind,
            "created_at": job.created_at,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "result": job.result,
        }

    def _emit(self, job: Job, payload: Dict[str, Any]) -> None:
        try:
            job.events.put_nowait(json.dumps(payload))
        except queue.Full:
            # Drop if client is slow; next update will catch up.
            pass

    def _set(self, job: Job, *, status: Optional[str] = None, progress: Optional[float] = None, message: Optional[str] = None, result: Optional[Dict[str, Any]] = None) -> None:
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = max(0.0, min(1.0, float(progress)))
        if message is not None:
            job.message = message
        if result is not None:
            job.result = result
        self._emit(job, {"type": "job_update", "job": self._public(job)})

    def start_export(
        self,
        *,
        proj: Project,
        selection: Dict[str, Any],
        export_dir: Path,
        with_captions: bool,
        template: str,
        width: int,
        height: int,
        fps: int,
        crf: int,
        preset: str,
        normalize_audio: bool,
        whisper_cfg: Optional[TranscribeConfig] = None,
    ) -> Job:
        job = self.create("export")

        def runner() -> None:
            self._set(job, status="running", progress=0.0, message="starting")
            try:
                video_path = Path(proj.video_path)
                sel_id = selection["id"]
                start_s = float(selection["start_s"])
                end_s = float(selection["end_s"])

                export_dir.mkdir(parents=True, exist_ok=True)
                out_path = export_dir / f"{sel_id}_{template}_{width}x{height}.mp4"

                subtitles_ass = None
                if with_captions:
                    self._set(job, message="transcribing")
                    # Cache transcript per selection
                    tjson = proj.analysis_dir / "transcripts" / f"{sel_id}_{int(start_s)}_{int(end_s)}.json"
                    if tjson.exists():
                        segs = load_transcript_json(tjson)
                    else:
                        if whisper_cfg is None:
                            whisper_cfg_local = TranscribeConfig()
                        else:
                            whisper_cfg_local = whisper_cfg

                        segs = transcribe_segment(video_path, start_s=start_s, end_s=end_s, cfg=whisper_cfg_local)
                        save_transcript_json(tjson, segs, whisper_cfg_local)

                    self._set(job, message="rendering captions")
                    ass_path = proj.analysis_dir / "subtitles" / f"{sel_id}.ass"
                    subtitles_ass = write_ass(segs, ass_path, playres_x=width, playres_y=height)

                spec = ExportSpec(
                    video_path=video_path,
                    start_s=start_s,
                    end_s=end_s,
                    output_path=out_path,
                    template=template,
                    width=width,
                    height=height,
                    fps=fps,
                    crf=crf,
                    preset=preset,
                    subtitles_ass=subtitles_ass,
                    normalize_audio=normalize_audio,
                )

                def on_prog(frac: float, msg: str) -> None:
                    self._set(job, progress=frac, message=msg)

                run_ffmpeg_export(spec, on_progress=on_prog)

                record_export(
                    proj,
                    selection_id=sel_id,
                    output_path=out_path,
                    template=template,
                    with_captions=with_captions,
                    status="succeeded",
                )

                self._set(job, status="succeeded", progress=1.0, message="done", result={"output": str(out_path)})
            except WhisperNotInstalledError as e:
                self._set(job, status="failed", message=str(e), result={})
            except Exception as e:
                self._set(job, status="failed", message=f"{type(e).__name__}: {e}", result={})

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        return job


JOB_MANAGER = JobManager()
