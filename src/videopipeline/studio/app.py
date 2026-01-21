from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..analysis_audio import compute_audio_analysis
from ..analysis_highlights import compute_highlights_analysis
from ..layouts import RectNorm
from ..project import create_or_load_project, get_project_data, load_npz, set_layout_facecam, Project
from ..profile import load_profile
from ..publisher.accounts import AccountStore
from ..publisher.jobs import PublishJobStore
from ..publisher.queue import PublishWorker
from .jobs import JOB_MANAGER
from .range import ranged_file_response
from .home import list_recent_projects, windows_open_video_dialog, check_video_exists


class StudioContext:
    """Holds the active project context for Studio."""

    def __init__(
        self,
        video_path: Optional[Path] = None,
        profile_path: Optional[Path] = None,
    ):
        self.profile = load_profile(profile_path)
        self.profile_path = profile_path
        self._project: Optional[Project] = None

        if video_path is not None:
            self._project = create_or_load_project(video_path)

    @property
    def project(self) -> Optional[Project]:
        return self._project

    @property
    def has_project(self) -> bool:
        return self._project is not None

    def open_project(self, video_path: Path) -> Project:
        """Open or create a project for the given video."""
        self._project = create_or_load_project(video_path)
        return self._project

    def close_project(self) -> None:
        """Close the current project."""
        self._project = None

    def require_project(self) -> Project:
        """Return the current project or raise HTTPException if none."""
        if self._project is None:
            raise HTTPException(status_code=400, detail="no_active_project")
        return self._project


def create_app(
    *,
    video_path: Optional[Path] = None,
    profile_path: Optional[Path] = None,
) -> FastAPI:
    ctx = StudioContext(video_path=video_path, profile_path=profile_path)

    app = FastAPI(title="VideoPipeline Studio")
    account_store = AccountStore()
    publish_store = PublishJobStore()
    publish_worker = PublishWorker(job_store=publish_store, account_store=account_store)
    publish_worker.start()

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/api/profile")
    def api_profile() -> JSONResponse:
        return JSONResponse(ctx.profile)

    # -------------------------------------------------------------------------
    # Home endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/project")
    def api_project() -> JSONResponse:
        if not ctx.has_project:
            return JSONResponse({"active": False})
        proj = ctx.require_project()
        data = get_project_data(proj)
        return JSONResponse({"active": True, "project": data})

    @app.get("/api/home/recent_projects")
    def api_home_recent_projects() -> JSONResponse:
        projects = list_recent_projects()
        return JSONResponse({"projects": projects})

    @app.post("/api/home/open_project")
    def api_home_open_project(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        video_path = body.get("video_path")
        if not video_path:
            raise HTTPException(status_code=400, detail="video_path_required")

        video_path = Path(video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="video_not_found")

        proj = ctx.open_project(video_path)
        return JSONResponse({"active": True, "project": get_project_data(proj)})

    @app.post("/api/home/open_video")
    def api_home_open_video(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        video_path = body.get("video_path")
        if not video_path:
            raise HTTPException(status_code=400, detail="video_path_required")

        video_path = Path(video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="video_not_found")

        proj = ctx.open_project(video_path)
        return JSONResponse({"active": True, "project": get_project_data(proj)})

    @app.post("/api/home/open_dialog")
    def api_home_open_dialog():
        if sys.platform != "win32":
            return JSONResponse({"video_path": None, "error": "not_windows"})
        result = windows_open_video_dialog()
        return JSONResponse({"video_path": result})

    @app.post("/api/home/close_project")
    def api_home_close_project():
        ctx.close_project()
        return JSONResponse({"active": False})

    # -------------------------------------------------------------------------
    # Publisher endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/publisher/accounts")
    def api_publisher_accounts() -> JSONResponse:
        accounts = [acct.to_dict() for acct in account_store.list()]
        return JSONResponse(accounts)

    @app.post("/api/publisher/publish")
    def api_publisher_publish(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        account_id = str(body.get("account_id") or "")
        file_path = Path(body.get("file_path") or "")
        metadata_path = Path(body.get("metadata_path") or "")
        account = account_store.get(account_id)
        if not account:
            raise HTTPException(status_code=404, detail="account_not_found")
        if not file_path.exists() or not metadata_path.exists():
            raise HTTPException(status_code=404, detail="missing_file_or_metadata")
        job = publish_worker.queue_job(
            platform=account.platform,
            account_id=account.id,
            file_path=file_path,
            metadata_path=metadata_path,
        )
        return JSONResponse({"job_id": job.id})

    @app.get("/api/publisher/jobs")
    def api_publisher_jobs() -> JSONResponse:
        jobs = [job.to_dict() for job in publish_store.list_jobs()]
        return JSONResponse(jobs)

    @app.get("/api/publisher/jobs/stream")
    def api_publisher_jobs_stream() -> StreamingResponse:
        def event_stream():
            last_seen: dict[str, str] = {}
            while True:
                jobs = publish_store.list_jobs()
                for job in jobs:
                    payload = json.dumps(job.to_dict())
                    if last_seen.get(job.id) != job.updated_at:
                        last_seen[job.id] = job.updated_at
                        yield f"data: {payload}\n\n"
                time.sleep(1.0)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/layout")
    def api_layout() -> JSONResponse:
        proj = ctx.require_project()
        proj_data = get_project_data(proj)
        return JSONResponse(proj_data.get("layout", {}))

    @app.post("/api/layout/facecam")
    def api_layout_facecam(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        proj = ctx.require_project()
        try:
            rect = RectNorm(
                x=float(body.get("x")),
                y=float(body.get("y")),
                w=float(body.get("w")),
                h=float(body.get("h")),
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"invalid_rect: {e}")

        data = set_layout_facecam(proj, rect=rect.to_dict())
        return JSONResponse(data.get("layout", {}))

    @app.post("/api/analyze/audio")
    def api_analyze_audio(body: Dict[str, Any] = Body(default={})):  # type: ignore[valid-type]
        proj = ctx.require_project()
        cfg = ctx.profile.get("analysis", {}).get("audio", {})
        cfg = {**cfg, **(body or {})}

        job = JOB_MANAGER.create("analyze_audio")

        def runner() -> None:
            JOB_MANAGER._set(job, status="running", progress=0.0, message="analyzing audio")  # type: ignore[attr-defined]

            def on_prog(frac: float) -> None:
                JOB_MANAGER._set(job, progress=frac, message="analyzing audio")  # type: ignore[attr-defined]

            try:
                result = compute_audio_analysis(
                    proj,
                    sample_rate=int(cfg.get("sample_rate", 16000)),
                    hop_s=float(cfg.get("hop_seconds", 0.5)),
                    smooth_s=float(cfg.get("smooth_seconds", 3.0)),
                    top=int(cfg.get("top", 12)),
                    min_gap_s=float(cfg.get("min_gap_seconds", 20.0)),
                    pre_s=float(cfg.get("pre_seconds", 8.0)),
                    post_s=float(cfg.get("post_seconds", 22.0)),
                    skip_start_s=float(cfg.get("skip_start_seconds", 10.0)),
                    on_progress=on_prog,
                )
                JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result=result)  # type: ignore[attr-defined]
            except Exception as e:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}")  # type: ignore[attr-defined]

        import threading

        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse({"job_id": job.id})

    @app.post("/api/analyze/highlights")
    def api_analyze_highlights(body: Dict[str, Any] = Body(default={})):  # type: ignore[valid-type]
        proj = ctx.require_project()
        analysis_cfg = ctx.profile.get("analysis", {})
        body = body or {}
        audio_cfg = {**analysis_cfg.get("audio", {}), **(body.get("audio") or {})}
        motion_cfg = {**analysis_cfg.get("motion", {}), **(body.get("motion") or {})}
        scenes_cfg = {**analysis_cfg.get("scenes", {}), **(body.get("scenes") or {})}
        highlights_cfg = {**analysis_cfg.get("highlights", {}), **(body.get("highlights") or {})}

        job = JOB_MANAGER.create("analyze_highlights")

        def runner() -> None:
            JOB_MANAGER._set(job, status="running", progress=0.0, message="analyzing highlights")  # type: ignore[attr-defined]

            def on_prog(frac: float) -> None:
                JOB_MANAGER._set(job, progress=frac, message="analyzing highlights")  # type: ignore[attr-defined]

            try:
                result = compute_highlights_analysis(
                    proj,
                    audio_cfg=audio_cfg,
                    motion_cfg=motion_cfg,
                    scenes_cfg=scenes_cfg,
                    highlights_cfg=highlights_cfg,
                    include_chat=True,
                    on_progress=on_prog,
                )
                JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result=result)  # type: ignore[attr-defined]
            except Exception as e:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}")  # type: ignore[attr-defined]

        import threading

        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse({"job_id": job.id})

    @app.get("/api/audio/timeline")
    def api_audio_timeline(max_points: int = 2000) -> JSONResponse:
        proj = ctx.require_project()
        if not proj.audio_features_path.exists():
            return JSONResponse({"ok": False, "reason": "no_audio_analysis"}, status_code=404)

        data = load_npz(proj.audio_features_path)
        scores = data.get("scores")
        if scores is None:
            return JSONResponse({"ok": False, "reason": "missing_scores"}, status_code=404)

        scores = scores.astype(float)
        n = len(scores)
        if n <= 0:
            return JSONResponse({"ok": False, "reason": "empty"}, status_code=404)

        step = max(1, int(n / max(1, int(max_points))))
        idx = np.arange(0, n, step)
        down = scores[idx]
        if not np.isfinite(down).all():
            down = np.where(np.isfinite(down), down, 0.0)

        proj_data = get_project_data(proj)
        hop_s = float(
            proj_data.get("analysis", {})
            .get("audio", {})
            .get("config", {})
            .get("hop_seconds", ctx.profile.get("analysis", {}).get("audio", {}).get("hop_seconds", 0.5))
        )

        return JSONResponse({"ok": True, "hop_seconds": hop_s, "indices": idx.tolist(), "scores": down.tolist()})

    @app.get("/api/highlights/timeline")
    def api_highlights_timeline(max_points: int = 2000) -> JSONResponse:
        proj = ctx.require_project()
        if not proj.highlights_features_path.exists():
            return JSONResponse({"ok": False, "reason": "no_highlights_analysis"}, status_code=404)

        data = load_npz(proj.highlights_features_path)
        scores = data.get("combined_scores")
        if scores is None:
            return JSONResponse({"ok": False, "reason": "missing_scores"}, status_code=404)

        scores = scores.astype(float)
        n = len(scores)
        if n <= 0:
            return JSONResponse({"ok": False, "reason": "empty"}, status_code=404)

        step = max(1, int(n / max(1, int(max_points))))
        idx = np.arange(0, n, step)
        down = scores[idx]
        if not np.isfinite(down).all():
            down = np.where(np.isfinite(down), down, 0.0)

        proj_data = get_project_data(proj)
        hop_s = float(
            proj_data.get("analysis", {})
            .get("highlights", {})
            .get("config", {})
            .get("audio", {})
            .get("hop_seconds", ctx.profile.get("analysis", {}).get("audio", {}).get("hop_seconds", 0.5))
        )

        return JSONResponse({"ok": True, "hop_seconds": hop_s, "indices": idx.tolist(), "scores": down.tolist()})

    @app.get("/video")
    async def video(request: Request):
        proj = ctx.require_project()
        ext = Path(proj.video_path).suffix.lower()
        media_type = "video/mp4" if ext in {".mp4", ".m4v", ".mov"} else "application/octet-stream"
        return ranged_file_response(request, Path(proj.video_path), media_type=media_type)

    @app.post("/api/selections")
    def api_add_selection(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        from ..project import add_selection
        proj = ctx.require_project()
        start_s = float(body.get("start_s"))
        end_s = float(body.get("end_s"))
        title = str(body.get("title") or "")
        notes = str(body.get("notes") or "")
        template = str(body.get("template") or ctx.profile.get("export", {}).get("template", "vertical_blur"))

        add_selection(proj, start_s=start_s, end_s=end_s, title=title, notes=notes, template=template)
        return JSONResponse(get_project_data(proj))

    @app.post("/api/selections/from_candidates")
    def api_selections_from_candidates(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        from ..project import add_selection_from_candidate
        proj = ctx.require_project()
        top = int(body.get("top", 10))
        template = str(body.get("template") or ctx.profile.get("export", {}).get("template", "vertical_blur"))

        proj_data = get_project_data(proj)
        highlights = proj_data.get("analysis", {}).get("highlights", {})
        candidates = highlights.get("candidates") or []
        if not candidates:
            raise HTTPException(status_code=404, detail="no_candidates")

        created_ids = []
        for cand in candidates[: max(0, top)]:
            title = cand.get("title") or ""
            created_ids.append(add_selection_from_candidate(proj, candidate=cand, template=template, title=title))

        return JSONResponse({"project": get_project_data(proj), "created_ids": created_ids})

    @app.patch("/api/selections/{selection_id}")
    def api_patch_selection(selection_id: str, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        from ..project import update_selection
        proj = ctx.require_project()

        patch: Dict[str, Any] = {}
        for k in ["start_s", "end_s", "title", "notes", "template"]:
            if k in body:
                patch[k] = body[k]
        try:
            update_selection(proj, selection_id, patch)
        except KeyError:
            raise HTTPException(status_code=404, detail="selection_not_found")
        return JSONResponse(get_project_data(proj))

    @app.delete("/api/selections/{selection_id}")
    def api_delete_selection(selection_id: str):
        from ..project import remove_selection
        proj = ctx.require_project()

        remove_selection(proj, selection_id)
        return JSONResponse(get_project_data(proj))

    @app.post("/api/export")
    def api_export(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        proj = ctx.require_project()
        sel_id = str(body.get("selection_id"))
        proj_data = get_project_data(proj)
        selection = next((s for s in proj_data.get("selections", []) if s.get("id") == sel_id), None)
        if not selection:
            raise HTTPException(status_code=404, detail="selection_not_found")

        export_cfg = {**ctx.profile.get("export", {}), **(body.get("export") or {})}
        cap_cfg = {**ctx.profile.get("captions", {}), **(body.get("captions") or {})}
        hook_cfg = {**ctx.profile.get("overlay", {}).get("hook_text", {}), **(body.get("hook_text") or {})}
        pip_cfg = {**ctx.profile.get("layout", {}).get("pip", {}), **(body.get("pip") or {})}

        with_captions = bool(body.get("with_captions", cap_cfg.get("enabled", False)))

        whisper_cfg = None
        if with_captions:
            from ..transcribe import TranscribeConfig

            whisper_cfg = TranscribeConfig(
                model_size=str(cap_cfg.get("model_size", "small")),
                language=cap_cfg.get("language"),
                device=str(cap_cfg.get("device", "cpu")),
                compute_type=str(cap_cfg.get("compute_type", "int8")),
            )

        job = JOB_MANAGER.start_export(
            proj=proj,
            selection=selection,
            export_dir=proj.exports_dir,
            with_captions=with_captions,
            template=str(export_cfg.get("template", selection.get("template") or "vertical_blur")),
            width=int(export_cfg.get("width", 1080)),
            height=int(export_cfg.get("height", 1920)),
            fps=int(export_cfg.get("fps", 30)),
            crf=int(export_cfg.get("crf", 20)),
            preset=str(export_cfg.get("preset", "veryfast")),
            normalize_audio=bool(export_cfg.get("normalize_audio", False)),
            whisper_cfg=whisper_cfg,
            hook_cfg=hook_cfg,
            pip_cfg=pip_cfg,
        )

        return JSONResponse({"job_id": job.id})

    @app.post("/api/export/batch")
    def api_export_batch(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        proj = ctx.require_project()
        selection_ids = body.get("selection_ids") or []
        export_cfg = {**ctx.profile.get("export", {}), **(body.get("export") or {})}
        cap_cfg = {**ctx.profile.get("captions", {}), **(body.get("captions") or {})}
        hook_cfg = {**ctx.profile.get("overlay", {}).get("hook_text", {}), **(body.get("hook_text") or {})}
        pip_cfg = {**ctx.profile.get("layout", {}).get("pip", {}), **(body.get("pip") or {})}
        with_captions = bool(body.get("with_captions", cap_cfg.get("enabled", False)))

        whisper_cfg = None
        if with_captions:
            from ..transcribe import TranscribeConfig

            whisper_cfg = TranscribeConfig(
                model_size=str(cap_cfg.get("model_size", "small")),
                language=cap_cfg.get("language"),
                device=str(cap_cfg.get("device", "cpu")),
                compute_type=str(cap_cfg.get("compute_type", "int8")),
            )

        proj_data = get_project_data(proj)
        selections = proj_data.get("selections", [])
        if selection_ids:
            selections = [s for s in selections if s.get("id") in selection_ids]
        if not selections:
            raise HTTPException(status_code=404, detail="no_selections")

        job = JOB_MANAGER.start_export_batch(
            proj=proj,
            selections=selections,
            export_dir=proj.exports_dir,
            with_captions=with_captions,
            template=str(export_cfg.get("template", "vertical_blur")),
            width=int(export_cfg.get("width", 1080)),
            height=int(export_cfg.get("height", 1920)),
            fps=int(export_cfg.get("fps", 30)),
            crf=int(export_cfg.get("crf", 20)),
            preset=str(export_cfg.get("preset", "veryfast")),
            normalize_audio=bool(export_cfg.get("normalize_audio", False)),
            whisper_cfg=whisper_cfg,
            hook_cfg=hook_cfg,
            pip_cfg=pip_cfg,
        )

        return JSONResponse({"job_id": job.id})

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        job = JOB_MANAGER.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")
        return JSONResponse(
            {
                "id": job.id,
                "kind": job.kind,
                "created_at": job.created_at,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "result": job.result,
            }
        )

    @app.get("/api/jobs/{job_id}/events")
    def api_job_events(job_id: str):
        job = JOB_MANAGER.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")

        def event_stream():
            # Initial snapshot
            yield f"data: {json.dumps({'type': 'job_update', 'job': {'id': job.id, 'kind': job.kind, 'created_at': job.created_at, 'status': job.status, 'progress': job.progress, 'message': job.message, 'result': job.result}})}\n\n"
            while True:
                try:
                    payload = job.events.get(timeout=15)
                    yield f"data: {payload}\n\n"
                except Exception:
                    yield ": keep-alive\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
