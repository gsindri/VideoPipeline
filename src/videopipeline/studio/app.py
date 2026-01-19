from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..analysis_audio import compute_audio_analysis
from ..analysis_highlights import compute_highlights_analysis
from ..project import Project, add_selection, create_or_load_project, get_project_data, load_npz
from ..profile import load_profile
from .jobs import JOB_MANAGER
from .range import ranged_file_response


def create_app(
    *,
    video_path: Path,
    profile_path: Optional[Path] = None,
) -> FastAPI:
    proj = create_or_load_project(video_path)
    profile = load_profile(profile_path)

    app = FastAPI(title="VideoPipeline Studio")

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/api/profile")
    def api_profile() -> JSONResponse:
        return JSONResponse(profile)

    @app.get("/api/project")
    def api_project() -> JSONResponse:
        return JSONResponse(get_project_data(proj))

    @app.post("/api/analyze/audio")
    def api_analyze_audio(body: Dict[str, Any] = Body(default={})):  # type: ignore[valid-type]
        cfg = profile.get("analysis", {}).get("audio", {})
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
        analysis_cfg = profile.get("analysis", {})
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

        proj_data = get_project_data(proj)
        hop_s = float(
            proj_data.get("analysis", {})
            .get("audio", {})
            .get("config", {})
            .get("hop_seconds", profile.get("analysis", {}).get("audio", {}).get("hop_seconds", 0.5))
        )

        return JSONResponse({"ok": True, "hop_seconds": hop_s, "indices": idx.tolist(), "scores": down.tolist()})

    @app.get("/api/highlights/timeline")
    def api_highlights_timeline(max_points: int = 2000) -> JSONResponse:
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

        proj_data = get_project_data(proj)
        hop_s = float(
            proj_data.get("analysis", {})
            .get("highlights", {})
            .get("config", {})
            .get("audio", {})
            .get("hop_seconds", profile.get("analysis", {}).get("audio", {}).get("hop_seconds", 0.5))
        )

        return JSONResponse({"ok": True, "hop_seconds": hop_s, "indices": idx.tolist(), "scores": down.tolist()})

    @app.get("/video")
    async def video(request: Request):
        ext = Path(proj.video_path).suffix.lower()
        media_type = "video/mp4" if ext in {".mp4", ".m4v", ".mov"} else "application/octet-stream"
        return ranged_file_response(request, Path(proj.video_path), media_type=media_type)

    @app.post("/api/selections")
    def api_add_selection(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        start_s = float(body.get("start_s"))
        end_s = float(body.get("end_s"))
        title = str(body.get("title") or "")
        notes = str(body.get("notes") or "")
        template = str(body.get("template") or profile.get("export", {}).get("template", "vertical_blur"))

        add_selection(proj, start_s=start_s, end_s=end_s, title=title, notes=notes, template=template)
        return JSONResponse(get_project_data(proj))

    @app.patch("/api/selections/{selection_id}")
    def api_patch_selection(selection_id: str, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        from ..project import update_selection

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

        remove_selection(proj, selection_id)
        return JSONResponse(get_project_data(proj))

    @app.post("/api/export")
    def api_export(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        sel_id = str(body.get("selection_id"))
        proj_data = get_project_data(proj)
        selection = next((s for s in proj_data.get("selections", []) if s.get("id") == sel_id), None)
        if not selection:
            raise HTTPException(status_code=404, detail="selection_not_found")

        export_cfg = {**profile.get("export", {}), **(body.get("export") or {})}
        cap_cfg = {**profile.get("captions", {}), **(body.get("captions") or {})}

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
