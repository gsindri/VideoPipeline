from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..exporter import (
    ExportCancelledError,
    ExportSpec,
    HookTextSpec,
    LayoutPipSpec,
    layout_preset_to_template,
    normalize_layout_preset,
    run_ffmpeg_export,
)
from ..layouts import get_facecam_rect
from ..metadata import build_metadata, derive_hook_text, write_metadata
from ..project import Project, get_project_data, record_export
from ..subtitles import DEFAULT_CAPTION_THEME, SubtitleSegment, normalize_caption_theme, write_ass
from ..transcribe import (
    TranscribeConfig,
    WhisperNotInstalledError,
    load_transcript_json,
    save_transcript_json,
    transcribe_segment,
)
from ..utils import PreventSleep
from ..utils import utc_iso as _utc_iso


def with_prevent_sleep(reason: str = "VideoPipeline job") -> Callable:
    """Decorator to wrap a function with sleep prevention.

    Use this on job runner functions to prevent Windows from sleeping
    during long operations like analysis or export.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with PreventSleep(reason):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@dataclass
class Job:
    id: str
    kind: str
    created_at: str = field(default_factory=_utc_iso)
    status: str = "queued"  # queued|running|succeeded|failed|cancelled
    progress: float = 0.0
    message: str = ""
    result: Dict[str, Any] = field(default_factory=dict)

    # Timing fields
    started_at: Optional[float] = field(default=None)  # time.time() when job started running

    # Cancellation flag - threads should check this periodically
    _cancel_requested: bool = field(default=False, repr=False)

    # SSE event stream
    events: "queue.Queue[str]" = field(default_factory=lambda: queue.Queue(maxsize=1000))

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def elapsed_seconds(self) -> Optional[float]:
        """Return elapsed seconds since job started, or None if not started."""
        if self.started_at is None:
            return None
        return time.time() - self.started_at


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
            "started_at": job.started_at,
            "elapsed_seconds": job.elapsed_seconds,
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
        if job.status in {"succeeded", "failed", "cancelled"}:
            if status is None or status != job.status:
                return
        if status is not None:
            job.status = status
            # Record start time when job begins running
            if status == "running" and job.started_at is None:
                job.started_at = time.time()
        if progress is not None:
            job.progress = max(0.0, min(1.0, float(progress)))
        if message is not None:
            job.message = message
        if result is not None:
            # Merge incremental result updates so callers can update a subset of keys
            # without accidentally dropping previously-populated fields.
            try:
                job.result.update(result)
            except Exception:
                job.result = result
        self._emit(job, {"type": "job_update", "job": self._public(job)})

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a job. Returns True if job was found and cancellable."""
        job = self.get(job_id)
        if not job:
            return False
        if job.status not in ("queued", "running"):
            return False  # Already finished
        job._cancel_requested = True
        self._set(job, status="cancelled", message="Cancelled by user")
        return True

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
        caption_theme: str = DEFAULT_CAPTION_THEME,
        whisper_cfg: Optional[TranscribeConfig] = None,
        hook_cfg: Optional[Dict[str, Any]] = None,
        pip_cfg: Optional[Dict[str, Any]] = None,
        director_results: Optional[list[Dict[str, Any]]] = None,
    ) -> Job:
        job = self.create("export")

        @with_prevent_sleep("Exporting video")
        def runner() -> None:
            out_path: Optional[Path] = None
            self._set(job, status="running", progress=0.0, message="starting")
            try:
                if job.cancel_requested:
                    raise ExportCancelledError("cancelled")

                video_path = Path(proj.video_path)
                sel_id = selection["id"]
                start_s = float(selection["start_s"])
                end_s = float(selection["end_s"])
                layout_preset = normalize_layout_preset(template or selection.get("template"))
                caption_theme_normalized = normalize_caption_theme(caption_theme)

                export_dir.mkdir(parents=True, exist_ok=True)
                out_path = export_dir / f"{sel_id}_{layout_preset}_{width}x{height}.mp4"

                proj_data = get_project_data(proj)
                facecam = get_facecam_rect(proj_data.get("layout", {}))

                subtitles_ass = None
                segments: Optional[list[SubtitleSegment]] = None
                if with_captions:
                    if job.cancel_requested:
                        raise ExportCancelledError("cancelled")
                    self._set(job, message="transcribing")
                    # Cache transcript per selection
                    tjson = proj.analysis_dir / "transcripts" / f"{sel_id}_{int(start_s)}_{int(end_s)}.json"
                    if tjson.exists():
                        segments = load_transcript_json(tjson)
                    else:
                        if whisper_cfg is None:
                            whisper_cfg_local = TranscribeConfig()
                        else:
                            whisper_cfg_local = whisper_cfg

                        segments = transcribe_segment(video_path, start_s=start_s, end_s=end_s, cfg=whisper_cfg_local)
                        save_transcript_json(tjson, segments, whisper_cfg_local)

                    self._set(job, message="rendering captions")
                    ass_path = proj.analysis_dir / "subtitles" / f"{sel_id}.ass"
                    subtitles_ass = write_ass(
                        segments or [],
                        ass_path,
                        playres_x=width,
                        playres_y=height,
                        theme=caption_theme_normalized,
                    )
                else:
                    tjson = proj.analysis_dir / "transcripts" / f"{sel_id}_{int(start_s)}_{int(end_s)}.json"
                    if tjson.exists():
                        segments = load_transcript_json(tjson)

                hook_spec = None
                if hook_cfg and bool(hook_cfg.get("enabled", False)):
                    hook_text = hook_cfg.get("text")
                    # Check AI director results for this selection's hook
                    if not hook_text and director_results:
                        candidate_rank = selection.get("candidate_rank") or selection.get("rank")
                        if candidate_rank is not None:
                            for dr in director_results:
                                if dr.get("candidate_rank") == candidate_rank:
                                    hook_text = dr.get("hook")
                                    break
                    # Fall back to candidate hook_text or derived hook
                    if not hook_text:
                        hook_text = selection.get("hook_text")
                    if not hook_text:
                        hook_text = derive_hook_text(selection, segments)
                    if hook_text:
                        hook_spec = HookTextSpec(
                            enabled=True,
                            duration_seconds=float(hook_cfg.get("duration_seconds", 2.0)),
                            text=str(hook_text),
                            font=str(hook_cfg.get("font", "auto")),
                            fontsize=int(hook_cfg.get("fontsize", 64)),
                            y=int(hook_cfg.get("y", 120)),
                        )

                spec = ExportSpec(
                    video_path=video_path,
                    start_s=start_s,
                    end_s=end_s,
                    output_path=out_path,
                    template=template,
                    layout_preset=layout_preset,
                    caption_theme=caption_theme_normalized,
                    width=width,
                    height=height,
                    fps=fps,
                    crf=crf,
                    preset=preset,
                    subtitles_ass=subtitles_ass,
                    normalize_audio=normalize_audio,
                    layout_facecam=facecam,
                    layout_pip=LayoutPipSpec(**pip_cfg) if pip_cfg else None,
                    hook_text=hook_spec,
                )

                def on_prog(frac: float, msg: str) -> None:
                    if job.cancel_requested:
                        return
                    self._set(job, progress=frac, message=msg)

                run_ffmpeg_export(spec, on_progress=on_prog, check_cancel=lambda: job.cancel_requested)
                if job.cancel_requested:
                    raise ExportCancelledError("cancelled")

                # Look up AI metadata for this selection
                ai_metadata = None
                if director_results:
                    candidate_rank = selection.get("rank") or selection.get("candidate_rank")
                    if candidate_rank is not None:
                        for dr in director_results:
                            if dr.get("candidate_rank") == candidate_rank:
                                ai_metadata = dr
                                break

                metadata = build_metadata(
                    selection=selection,
                    output_path=out_path,
                    template=layout_preset,
                    with_captions=with_captions,
                    segments=segments,
                    ai_metadata=ai_metadata,
                    caption_theme=caption_theme_normalized,
                    render_template=layout_preset_to_template(layout_preset),
                )
                write_metadata(out_path.with_suffix(".metadata.json"), metadata)

                record_export(
                    proj,
                    selection_id=sel_id,
                    output_path=out_path,
                    template=layout_preset,
                    with_captions=with_captions,
                    caption_theme=caption_theme_normalized,
                    render_template=layout_preset_to_template(layout_preset),
                    status="succeeded",
                )

                self._set(job, status="succeeded", progress=1.0, message="done", result={"output": str(out_path)})
            except ExportCancelledError:
                if out_path is not None:
                    for path in (out_path, out_path.with_suffix(".metadata.json")):
                        try:
                            path.unlink(missing_ok=True)
                        except Exception:
                            pass
                self._set(job, status="cancelled", message="cancelled", result={})
            except WhisperNotInstalledError as e:
                if job.cancel_requested:
                    self._set(job, status="cancelled", message="cancelled", result={})
                else:
                    self._set(job, status="failed", message=str(e), result={})
            except Exception as e:
                if job.cancel_requested:
                    self._set(job, status="cancelled", message="cancelled", result={})
                else:
                    self._set(job, status="failed", message=f"{type(e).__name__}: {e}", result={})

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        return job

    def start_export_batch(
        self,
        *,
        proj: Project,
        selections: list[Dict[str, Any]],
        export_dir: Path,
        with_captions: bool,
        template: str,
        width: int,
        height: int,
        fps: int,
        crf: int,
        preset: str,
        normalize_audio: bool,
        caption_theme: str = DEFAULT_CAPTION_THEME,
        whisper_cfg: Optional[TranscribeConfig] = None,
        hook_cfg: Optional[Dict[str, Any]] = None,
        pip_cfg: Optional[Dict[str, Any]] = None,
        director_results: Optional[list[Dict[str, Any]]] = None,
    ) -> Job:
        job = self.create("export_batch")

        @with_prevent_sleep("Exporting batch")
        def runner() -> None:
            total = max(1, len(selections))
            self._set(job, status="running", progress=0.0, message=f"exporting 0/{total}")
            try:
                for idx, selection in enumerate(selections, start=1):
                    if job.cancel_requested:
                        self._set(job, status="cancelled", message="cancelled", result={})
                        return

                    subjob = self.start_export(
                        proj=proj,
                        selection=selection,
                        export_dir=export_dir,
                        with_captions=with_captions,
                        template=template or selection.get("template") or "vertical_blur",
                        caption_theme=caption_theme,
                        width=width,
                        height=height,
                        fps=fps,
                        crf=crf,
                        preset=preset,
                        normalize_audio=normalize_audio,
                        whisper_cfg=whisper_cfg,
                        hook_cfg=hook_cfg,
                        pip_cfg=pip_cfg,
                        director_results=director_results,
                    )

                    while True:
                        time.sleep(0.2)
                        job_child = self.get(subjob.id)
                        if job_child is None:
                            break
                        if job_child.status in {"succeeded", "failed", "cancelled"}:
                            break
                        if job.cancel_requested:
                            self.cancel(subjob.id)
                        frac = (idx - 1 + job_child.progress) / total
                        self._set(job, progress=frac, message=f"exporting {idx}/{total}")

                    job_child = self.get(subjob.id)
                    if job_child and job_child.status == "failed":
                        raise RuntimeError(job_child.message)
                    if job_child and job_child.status == "cancelled":
                        self._set(job, status="cancelled", message="cancelled", result={})
                        return

                    self._set(job, progress=idx / total, message=f"exporting {idx}/{total}")

                self._set(job, status="succeeded", progress=1.0, message="done", result={"count": len(selections)})
            except WhisperNotInstalledError as e:
                self._set(job, status="failed", message=str(e), result={})
            except Exception as e:
                self._set(job, status="failed", message=f"{type(e).__name__}: {e}", result={})

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        return job


JOB_MANAGER = JobManager()
