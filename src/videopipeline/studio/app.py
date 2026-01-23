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
from ..analysis_audio_events import AudioEventsConfig, compute_audio_events_analysis, load_audio_events_features
from ..analysis_highlights import compute_highlights_analysis
from ..analysis_transcript import TranscriptConfig, compute_transcript_analysis, load_transcript
from ..analysis_speech_features import SpeechFeatureConfig, compute_speech_features, load_speech_features
from ..analysis_reaction_audio import ReactionAudioConfig, compute_reaction_audio_features
from ..rerank_candidates import RerankConfig, compute_reranked_candidates
from ..analysis_silence import SilenceConfig, compute_silence_analysis
from ..analysis_sentences import SentenceConfig, compute_sentences_analysis
from ..analysis_chat_boundaries import ChatBoundaryConfig, compute_chat_boundaries_analysis
from ..analysis_boundaries import BoundaryConfig, compute_boundaries_analysis
from ..clip_variants import VariantGeneratorConfig, VariantDurationConfig, compute_clip_variants, load_clip_variants
from ..ai.director import DirectorConfig, compute_director_analysis
from ..layouts import RectNorm
from ..project import create_or_load_project, get_project_data, load_npz, set_layout_facecam, Project
from ..profile import load_profile
from ..publisher.accounts import AccountStore
from ..publisher.jobs import PublishJobStore
from ..publisher.queue import PublishWorker
from .jobs import JOB_MANAGER
from .range import ranged_file_response
from .home import list_recent_projects, windows_open_video_dialog, check_video_exists
from .publisher_api import create_publisher_router
from ..ingest import (
    IngestRequest,
    IngestResult,
    QualityCap,
    SpeedMode,
    SiteType,
    probe_url,
)
from ..ingest.ytdlp_runner import download_url


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

    # Mount publisher API router
    def get_exports_dir():
        if ctx.has_project:
            return ctx.require_project().exports_dir
        return None

    publisher_router = create_publisher_router(
        get_exports_dir=get_exports_dir,
        account_store=account_store,
        job_store=publish_store,
    )
    app.include_router(publisher_router)

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/api/health")
    def api_health() -> JSONResponse:
        return JSONResponse({"ok": True})

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

    @app.delete("/api/home/project/{project_id}")
    def api_home_delete_project(project_id: str) -> JSONResponse:
        """Delete a project and optionally its video file.
        
        Query params:
            delete_video: bool - Also delete the video file (default False)
        """
        import shutil
        from ..project import default_projects_root
        
        projects_root = default_projects_root()
        project_dir = projects_root / project_id
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="project_not_found")
        
        # Load project info before deleting
        project_json = project_dir / "project.json"
        video_path = None
        if project_json.exists():
            try:
                data = json.loads(project_json.read_text(encoding="utf-8"))
                video_path = data.get("video", {}).get("path")
            except Exception:
                pass
        
        # Close project if it's currently open
        if ctx.project and ctx.project.project_id == project_id:
            ctx.close_project()
        
        # Delete the project directory
        try:
            shutil.rmtree(project_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete project: {e}")
        
        return JSONResponse({
            "deleted": True,
            "project_id": project_id,
            "video_path": video_path,
        })

    @app.get("/api/home/videos")
    def api_home_videos() -> JSONResponse:
        """Unified list of all videos (projects + downloads merged).
        
        Returns videos sorted by most recently modified, with project status.
        """
        from ..ingest.ytdlp_runner import _default_downloads_dir
        from ..project import default_projects_root
        
        # Build lookup of video_path -> project info
        projects_by_path: Dict[str, Dict[str, Any]] = {}
        projects = list_recent_projects(limit=100)
        for p in projects:
            vp = p.get("video_path", "")
            if vp:
                # Normalize path for comparison
                projects_by_path[str(Path(vp).resolve())] = p
        
        # Scan downloads directory
        downloads_dir = _default_downloads_dir()
        video_exts = {".mp4", ".mkv", ".webm", ".m4v", ".avi", ".mov", ".ts"}
        videos: List[Dict[str, Any]] = []
        seen_paths: set = set()
        
        if downloads_dir.exists():
            for ext in video_exts:
                for f in downloads_dir.glob(f"*{ext}"):
                    resolved = str(f.resolve())
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    
                    try:
                        st = f.stat()
                        info_json = f.with_suffix(".info.json")
                        info = {}
                        if info_json.exists():
                            try:
                                info = json.loads(info_json.read_text(encoding="utf-8"))
                            except Exception:
                                pass
                        
                        # Check if this video has a project
                        project_info = projects_by_path.get(resolved)
                        
                        videos.append({
                            "path": str(f),
                            "filename": f.name,
                            "title": info.get("title", f.stem),
                            "size_bytes": st.st_size,
                            "mtime": st.st_mtime,
                            "duration_seconds": info.get("duration", project_info.get("duration_seconds", 0) if project_info else 0),
                            "url": info.get("webpage_url", ""),
                            "extractor": info.get("extractor", ""),
                            # Project info
                            "has_project": project_info is not None,
                            "project_id": project_info.get("project_id") if project_info else None,
                            "selections_count": project_info.get("selections_count", 0) if project_info else 0,
                            "exports_count": project_info.get("exports_count", 0) if project_info else 0,
                        })
                    except Exception:
                        continue
        
        # Sort by mtime descending
        videos.sort(key=lambda v: v["mtime"], reverse=True)
        
        return JSONResponse({"videos": videos[:30]})

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
    # URL Ingest endpoints
    # -------------------------------------------------------------------------

    @app.post("/api/ingest/probe")
    def api_ingest_probe(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        """Probe a URL to detect site type and metadata.
        
        Body:
            url: str - The URL to probe
            use_ytdlp: bool - Whether to use yt-dlp for detailed probe (default True)
        
        Returns:
            ProbeResult with site_type, policy, title, duration, etc.
        """
        url = str(body.get("url") or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="url_required")

        if not (url.startswith("http://") or url.startswith("https://")):
            raise HTTPException(status_code=400, detail="invalid_url")

        use_ytdlp = bool(body.get("use_ytdlp", True))

        try:
            result = probe_url(url, use_ytdlp=use_ytdlp)
            return JSONResponse(result.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Probe failed: {e}")

    @app.post("/api/ingest/url")
    def api_ingest_url(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        """Download a video from a URL using yt-dlp.
        
        Body:
            url: str - The URL to download
            options: dict - Optional download options
                no_playlist: bool (default True)
                create_preview: bool (default True)
                speed_mode: str - One of: auto, conservative, balanced, fast, aggressive
                quality_cap: str - One of: source, 1080, 720, 480
            auto_open: bool - Automatically open as project when done (default True)
        
        Returns job_id for tracking progress.
        """
        import threading

        url = str(body.get("url") or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="url_required")

        if not (url.startswith("http://") or url.startswith("https://")):
            raise HTTPException(status_code=400, detail="invalid_url")

        opts = body.get("options") or {}
        auto_open = bool(body.get("auto_open", True))

        # Parse speed mode
        speed_mode_str = str(opts.get("speed_mode", "auto")).lower()
        try:
            speed_mode = SpeedMode(speed_mode_str)
        except ValueError:
            speed_mode = SpeedMode.AUTO

        # Parse quality cap
        quality_cap_str = str(opts.get("quality_cap", "source")).lower()
        try:
            quality_cap = QualityCap(quality_cap_str)
        except ValueError:
            quality_cap = QualityCap.SOURCE

        request = IngestRequest(
            url=url,
            speed_mode=speed_mode,
            quality_cap=quality_cap,
            no_playlist=bool(opts.get("no_playlist", True)),
            create_preview=bool(opts.get("create_preview", True)),
            auto_open=auto_open,
        )

        job = JOB_MANAGER.create("download_url")

        def runner() -> None:
            import concurrent.futures
            
            # Shared state for progress tracking
            current_chat_status = {"status": "pending", "message": "Waiting..."}
            
            JOB_MANAGER._set(
                job, 
                status="running", 
                progress=0.0, 
                message="Starting download...",
                result={"video_status": "downloading", "chat_status": "pending"}
            )

            video_result = None
            chat_result = {"status": "skipped", "message": "Not a Twitch VOD"}
            chat_error = None

            def on_video_progress(frac: float, msg: str) -> None:
                # Video download is 0-90%, chat is 90-100%
                # Build combined message
                chat_info = ""
                if current_chat_status["status"] == "downloading":
                    chat_info = " | 💬 Chat: downloading..."
                elif current_chat_status["status"] == "success":
                    chat_info = " | 💬 Chat: ✓ done"
                elif current_chat_status["status"] == "failed":
                    chat_info = f" | 💬 Chat: ✗ failed"
                elif current_chat_status["status"] == "pending" and "twitch.tv/videos/" in url.lower():
                    chat_info = " | 💬 Chat: pending"
                
                JOB_MANAGER._set(
                    job, 
                    progress=frac * 0.9, 
                    message=f"{msg}{chat_info}",
                    result={
                        "video_status": "downloading",
                        "video_progress": frac,
                        "chat_status": current_chat_status["status"],
                        "chat_message": current_chat_status.get("message", ""),
                    }
                )

            def download_video():
                nonlocal video_result
                video_result = download_url(url, request=request, on_progress=on_video_progress)
                return video_result

            def download_chat():
                nonlocal chat_result, chat_error, current_chat_status
                # Only download chat for Twitch VODs
                if "twitch.tv/videos/" not in url.lower():
                    chat_result = {"status": "skipped", "message": "Not a Twitch VOD"}
                    current_chat_status = chat_result
                    return chat_result
                
                try:
                    from ..chat.downloader import download_chat as dl_chat, find_twitch_downloader_cli
                    
                    cli_path = find_twitch_downloader_cli()
                    if not cli_path:
                        chat_result = {"status": "skipped", "message": "TwitchDownloaderCLI not found"}
                        current_chat_status = chat_result
                        return chat_result
                    
                    # Extract video ID from URL
                    import re
                    match = re.search(r'twitch\.tv/videos/(\d+)', url)
                    if not match:
                        chat_result = {"status": "skipped", "message": "Could not extract video ID"}
                        current_chat_status = chat_result
                        return chat_result
                    
                    video_id = match.group(1)
                    
                    # Download to temp location first, will move to project later
                    from ..ingest.ytdlp_runner import _default_downloads_dir
                    chat_temp_path = _default_downloads_dir() / f"chat_{video_id}.json"
                    
                    # Update status to downloading
                    current_chat_status["status"] = "downloading"
                    current_chat_status["message"] = "Downloading Twitch chat..."
                    
                    dl_chat(url, chat_temp_path)
                    
                    chat_result = {
                        "status": "success",
                        "message": "Chat downloaded",
                        "temp_path": str(chat_temp_path),
                        "video_id": video_id,
                    }
                    current_chat_status["status"] = "success"
                    current_chat_status["message"] = "Chat downloaded"
                    return chat_result
                    
                except Exception as e:
                    chat_error = str(e)
                    chat_result = {"status": "failed", "message": str(e)}
                    current_chat_status["status"] = "failed"
                    current_chat_status["message"] = str(e)
                    return chat_result

            try:
                # Run video and chat downloads in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    video_future = executor.submit(download_video)
                    chat_future = executor.submit(download_chat)
                    
                    # Wait for both to complete
                    video_future.result()  # This will raise if video download failed
                    chat_future.result()   # Chat failures are non-fatal

                result_dict = video_result.to_dict()
                result_dict["chat"] = chat_result

                # Auto-open the project if requested
                if auto_open:
                    # Use preview if available, otherwise original
                    video_to_open = video_result.preview_path or video_result.video_path
                    proj = ctx.open_project(video_to_open)
                    
                    # Store the source URL for chat download later
                    from ..project import set_source_url
                    set_source_url(proj, url)
                    
                    # If chat was downloaded, import it into the project
                    if chat_result.get("status") == "success" and chat_result.get("temp_path"):
                        JOB_MANAGER._set(job, progress=0.95, message="Importing chat into project...")
                        try:
                            from ..chat.downloader import import_chat_to_project
                            import_chat_to_project(proj, Path(chat_result["temp_path"]))
                            chat_result["imported"] = True
                        except Exception as e:
                            chat_result["import_error"] = str(e)
                    
                    result_dict["project"] = get_project_data(proj)
                    result_dict["auto_opened"] = True
                    result_dict["chat"] = chat_result

                # Build final message
                final_msg = "Download complete!"
                if chat_result.get("imported"):
                    final_msg = "Download complete! Chat imported."
                elif chat_result.get("import_error"):
                    final_msg = f"Download complete. Chat import failed: {chat_result.get('import_error')}"
                elif chat_result.get("status") == "failed":
                    final_msg = f"Download complete. Chat download failed: {chat_result.get('message', 'unknown')}"
                elif chat_result.get("status") == "skipped":
                    final_msg = f"Download complete. {chat_result.get('message', 'Chat skipped.')}"
                
                JOB_MANAGER._set(
                    job,
                    status="succeeded",
                    progress=1.0,
                    message=final_msg,
                    result=result_dict,
                )

            except ImportError as e:
                JOB_MANAGER._set(job, status="failed", message=str(e))
            except Exception as e:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}")

        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse({"job_id": job.id})

    @app.get("/api/ingest/recent_downloads")
    def api_ingest_recent_downloads() -> JSONResponse:
        """List recent downloads from the downloads folder."""
        from ..ingest.ytdlp_runner import _default_downloads_dir

        downloads_dir = _default_downloads_dir()
        if not downloads_dir.exists():
            return JSONResponse({"downloads": []})

        # Find video files
        video_exts = {".mp4", ".mkv", ".webm", ".m4v", ".avi", ".mov", ".ts"}
        files = []
        for ext in video_exts:
            files.extend(downloads_dir.glob(f"*{ext}"))

        # Sort by modification time
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        downloads = []
        for f in files[:20]:  # Limit to 20 most recent
            try:
                st = f.stat()
                info_json = f.with_suffix(".info.json")
                info = {}
                if info_json.exists():
                    try:
                        info = json.loads(info_json.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                downloads.append({
                    "path": str(f),
                    "filename": f.name,
                    "size_bytes": st.st_size,
                    "mtime": st.st_mtime,
                    "title": info.get("title", f.stem),
                    "url": info.get("webpage_url", ""),
                    "extractor": info.get("extractor", ""),
                    "duration_seconds": info.get("duration", 0),
                })
            except Exception:
                continue

        return JSONResponse({"downloads": downloads})

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
        audio_events_cfg = {**analysis_cfg.get("audio_events", {}), **(body.get("audio_events") or {})}

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
                    audio_events_cfg=audio_events_cfg,
                    include_chat=True,
                    include_audio_events=bool(audio_events_cfg.get("enabled", True)),
                    on_progress=on_prog,
                )
                JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result=result)  # type: ignore[attr-defined]
            except Exception as e:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}")  # type: ignore[attr-defined]

        import threading

        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse({"job_id": job.id})

    @app.post("/api/analyze/audio_events")
    def api_analyze_audio_events(body: Dict[str, Any] = Body(default={})):  # type: ignore[valid-type]
        """Run audio event detection (laughter/cheer/shout).
        
        This runs a lightweight audio event classifier to detect semantic
        audio events like laughter, cheering, applause, screaming, etc.
        
        The results can be used in highlight scoring as a first-class signal.
        """
        proj = ctx.require_project()
        analysis_cfg = ctx.profile.get("analysis", {})
        body = body or {}

        # Get configuration from profile, allow overrides from body
        events_cfg_dict = {**analysis_cfg.get("audio_events", {}), **(body or {})}

        job = JOB_MANAGER.create("analyze_audio_events")

        def runner() -> None:
            JOB_MANAGER._set(job, status="running", progress=0.0, message="Detecting audio events...")

            def on_prog(frac: float) -> None:
                JOB_MANAGER._set(job, progress=frac, message="Detecting audio events...")

            try:
                cfg = AudioEventsConfig.from_dict(events_cfg_dict)
                result = compute_audio_events_analysis(proj, cfg=cfg, on_progress=on_prog)
                JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result=result)
            except Exception as e:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}")

        import threading
        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse({"job_id": job.id})

    @app.post("/api/analyze/speech")
    def api_analyze_speech(body: Dict[str, Any] = Body(default={})):  # type: ignore[valid-type]
        """Run full speech analysis pipeline: transcription + speech features + reaction audio + rerank.
        
        This is the main endpoint for speech-based clip enhancement.
        It runs:
        1. Whisper transcription (if not cached)
        2. Speech feature extraction (lexical excitement, speech rate)
        3. Reaction audio features (acoustic cues)
        4. Candidate reranking with hook/quote extraction
        """
        proj = ctx.require_project()
        analysis_cfg = ctx.profile.get("analysis", {})
        body = body or {}

        # Get configurations from profile, allow overrides from body
        speech_cfg = {**analysis_cfg.get("speech", {}), **(body.get("speech") or {})}
        reaction_cfg = {**analysis_cfg.get("reaction_audio", {}), **(body.get("reaction_audio") or {})}
        rerank_cfg_dict = {**analysis_cfg.get("rerank", {}), **(body.get("rerank") or {})}

        job = JOB_MANAGER.create("analyze_speech")

        def runner() -> None:
            import threading as _threading
            JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting speech analysis...")

            try:
                # Step 1: Transcription (if needed)
                if speech_cfg.get("enabled", True) and not proj.transcript_path.exists():
                    JOB_MANAGER._set(job, progress=0.05, message="Transcribing audio with Whisper...")
                    
                    transcript_config = TranscriptConfig(
                        model_size=str(speech_cfg.get("model_size", "small")),
                        language=speech_cfg.get("language"),
                        device=str(speech_cfg.get("device", "cpu")),
                        compute_type=str(speech_cfg.get("compute_type", "int8")),
                        vad_filter=bool(speech_cfg.get("vad_filter", True)),
                        word_timestamps=bool(speech_cfg.get("word_timestamps", True)),
                    )
                    
                    def on_transcript_progress(frac: float) -> None:
                        JOB_MANAGER._set(job, progress=0.05 + 0.35 * frac, message="Transcribing audio...")
                    
                    compute_transcript_analysis(proj, cfg=transcript_config, on_progress=on_transcript_progress)

                # Step 2: Speech features (if transcript exists)
                if speech_cfg.get("enabled", True) and proj.transcript_path.exists() and not proj.speech_features_path.exists():
                    JOB_MANAGER._set(job, progress=0.4, message="Extracting speech features...")
                    
                    # Get reaction phrases from config
                    hook_cfg = rerank_cfg_dict.get("hook", {})
                    phrases = hook_cfg.get("phrases", [])
                    
                    speech_feature_config = SpeechFeatureConfig(
                        hop_seconds=float(speech_cfg.get("hop_seconds", 0.5)),
                        reaction_phrases=phrases if phrases else None,
                    )
                    
                    def on_speech_progress(frac: float) -> None:
                        JOB_MANAGER._set(job, progress=0.4 + 0.15 * frac, message="Extracting speech features...")
                    
                    compute_speech_features(proj, cfg=speech_feature_config, on_progress=on_speech_progress)

                # Step 3: Reaction audio features
                if reaction_cfg.get("enabled", True) and not proj.reaction_audio_features_path.exists():
                    JOB_MANAGER._set(job, progress=0.55, message="Analyzing reaction audio...")
                    
                    reaction_audio_config = ReactionAudioConfig(
                        sample_rate=int(reaction_cfg.get("sample_rate", 16000)),
                        hop_seconds=float(reaction_cfg.get("hop_seconds", 0.5)),
                        smooth_seconds=float(reaction_cfg.get("smooth_seconds", 1.5)),
                    )
                    
                    def on_reaction_progress(frac: float) -> None:
                        JOB_MANAGER._set(job, progress=0.55 + 0.25 * frac, message="Analyzing reaction audio...")
                    
                    compute_reaction_audio_features(proj, cfg=reaction_audio_config, on_progress=on_reaction_progress)

                # Step 4: Rerank candidates (if highlights exist)
                proj_data = get_project_data(proj)
                has_candidates = bool(proj_data.get("analysis", {}).get("highlights", {}).get("candidates"))
                
                if rerank_cfg_dict.get("enabled", True) and has_candidates:
                    JOB_MANAGER._set(job, progress=0.8, message="Reranking candidates...")
                    
                    # Build rerank config
                    hook_cfg = rerank_cfg_dict.get("hook", {})
                    quote_cfg = rerank_cfg_dict.get("quote", {})
                    weights = rerank_cfg_dict.get("weights", {})
                    
                    rerank_config = RerankConfig(
                        enabled=True,
                        weights=weights if weights else None,
                        hook_max_chars=int(hook_cfg.get("max_chars", 60)),
                        hook_window_seconds=float(hook_cfg.get("window_seconds", 4.0)),
                        quote_max_chars=int(quote_cfg.get("max_chars", 120)),
                        reaction_phrases=hook_cfg.get("phrases", []),
                    )
                    
                    def on_rerank_progress(frac: float) -> None:
                        JOB_MANAGER._set(job, progress=0.8 + 0.15 * frac, message="Reranking candidates...")
                    
                    result = compute_reranked_candidates(proj, cfg=rerank_config, on_progress=on_rerank_progress)
                else:
                    result = {"message": "No candidates to rerank. Run highlights analysis first."}

                JOB_MANAGER._set(
                    job,
                    status="succeeded",
                    progress=1.0,
                    message="Speech analysis complete!",
                    result=result,
                )

            except ImportError as e:
                JOB_MANAGER._set(job, status="failed", message=f"Missing dependency: {e}")
            except Exception as e:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}")

        import threading
        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse({"job_id": job.id})

    @app.post("/api/analyze/context_titles")
    def api_analyze_context_titles(body: Dict[str, Any] = Body(default={})):  # type: ignore[valid-type]
        """Run context-aware clip shaping + AI Director pipeline.
        
        This runs:
        1. Silence detection (if not cached)
        2. Sentence boundary extraction (if not cached)
        3. Chat boundary detection (if chat available, not cached)
        4. Unified boundary graph
        5. Clip variant generation
        6. AI Director for variant selection + metadata
        """
        proj = ctx.require_project()
        analysis_cfg = ctx.profile.get("analysis", {})
        context_cfg = ctx.profile.get("context", {})
        ai_cfg = ctx.profile.get("ai", {}).get("director", {})
        body = body or {}

        # Merge body overrides
        context_cfg = {**context_cfg, **(body.get("context") or {})}
        ai_cfg = {**ai_cfg, **(body.get("ai") or {})}

        job = JOB_MANAGER.create("analyze_context_titles")

        def runner() -> None:
            import threading as _threading
            JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting context analysis...")

            try:
                # Step 1: Silence detection
                silence_path = proj.analysis_dir / "silence.json"
                if not silence_path.exists():
                    JOB_MANAGER._set(job, progress=0.05, message="Detecting silence intervals...")
                    silence_cfg = SilenceConfig(
                        noise_db=float(context_cfg.get("silence_noise_db", -30.0)),
                        min_duration=float(context_cfg.get("silence_min_duration", 0.3)),
                    )
                    compute_silence_analysis(proj, cfg=silence_cfg)

                # Step 2: Sentence boundaries (requires transcript)
                sentences_path = proj.analysis_dir / "sentences.json"
                if proj.transcript_path.exists() and not sentences_path.exists():
                    JOB_MANAGER._set(job, progress=0.15, message="Extracting sentence boundaries...")
                    sentence_cfg = SentenceConfig(
                        max_sentence_words=int(context_cfg.get("max_sentence_words", 30)),
                    )
                    compute_sentences_analysis(proj, cfg=sentence_cfg)

                # Step 3: Chat boundaries (if chat available)
                chat_boundaries_path = proj.analysis_dir / "chat_boundaries.json"
                if proj.chat_features_path.exists() and not chat_boundaries_path.exists():
                    JOB_MANAGER._set(job, progress=0.25, message="Detecting chat valleys/bursts...")
                    chat_boundary_cfg = ChatBoundaryConfig(
                        valley_window_s=float(context_cfg.get("boundaries", {}).get("chat_valley_window_s", 12.0)),
                    )
                    compute_chat_boundaries_analysis(proj, cfg=chat_boundary_cfg)

                # Step 4: Unified boundaries
                boundaries_path = proj.analysis_dir / "boundaries.json"
                JOB_MANAGER._set(job, progress=0.35, message="Building boundary graph...")
                boundary_prefs = context_cfg.get("boundaries", {})
                boundary_cfg = BoundaryConfig(
                    prefer_silence=bool(boundary_prefs.get("prefer_silence", True)),
                    prefer_sentences=bool(boundary_prefs.get("prefer_sentences", True)),
                    prefer_scene_cuts=bool(boundary_prefs.get("prefer_scene_cuts", True)),
                    prefer_chat_valleys=bool(boundary_prefs.get("prefer_chat_valleys", True)),
                )
                compute_boundaries_analysis(proj, cfg=boundary_cfg)

                # Step 5: Clip variants
                JOB_MANAGER._set(job, progress=0.45, message="Generating clip variants...")
                variants_cfg_dict = context_cfg.get("variants", {})
                short_cfg = variants_cfg_dict.get("short", {})
                medium_cfg = variants_cfg_dict.get("medium", {})
                long_cfg = variants_cfg_dict.get("long", {})
                
                variant_cfg = VariantGeneratorConfig(
                    short=VariantDurationConfig(
                        min_s=float(short_cfg.get("min_s", 16)),
                        max_s=float(short_cfg.get("max_s", 24)),
                    ),
                    medium=VariantDurationConfig(
                        min_s=float(medium_cfg.get("min_s", 24)),
                        max_s=float(medium_cfg.get("max_s", 40)),
                    ),
                    long=VariantDurationConfig(
                        min_s=float(long_cfg.get("min_s", 40)),
                        max_s=float(long_cfg.get("max_s", 75)),
                    ),
                    chat_valley_window_s=float(boundary_prefs.get("chat_valley_window_s", 12.0)),
                )
                
                top_n = int(context_cfg.get("top_n", 25))
                
                def on_variant_progress(frac: float) -> None:
                    JOB_MANAGER._set(job, progress=0.45 + 0.25 * frac, message="Generating clip variants...")
                
                compute_clip_variants(proj, cfg=variant_cfg, top_n=top_n, on_progress=on_variant_progress)

                # Step 6: AI Director
                JOB_MANAGER._set(job, progress=0.70, message="Running AI Director...")
                director_cfg = DirectorConfig(
                    enabled=bool(ai_cfg.get("enabled", True)),
                    engine=str(ai_cfg.get("engine", "llama_cpp_server")),
                    endpoint=str(ai_cfg.get("endpoint", "http://127.0.0.1:11435")),
                    model_name=str(ai_cfg.get("model_name", "local-gguf-vulkan")),
                    timeout_s=float(ai_cfg.get("timeout_s", 30)),
                    max_tokens=int(ai_cfg.get("max_tokens", 256)),
                    temperature=float(ai_cfg.get("temperature", 0.2)),
                    platform=str(ai_cfg.get("platform", "shorts")),
                    fallback_to_rules=bool(ai_cfg.get("fallback_to_rules", True)),
                )
                
                def on_director_progress(frac: float) -> None:
                    JOB_MANAGER._set(job, progress=0.70 + 0.25 * frac, message="Running AI Director...")
                
                result = compute_director_analysis(proj, cfg=director_cfg, top_n=top_n, on_progress=on_director_progress)

                JOB_MANAGER._set(
                    job,
                    status="succeeded",
                    progress=1.0,
                    message="Context + Titles analysis complete!",
                    result=result,
                )

            except Exception as e:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}")

        import threading
        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse({"job_id": job.id})

    @app.post("/api/analyze/full")
    def api_analyze_full(body: Dict[str, Any] = Body(default={})):  # type: ignore[valid-type]
        """Run the complete analysis DAG with parallel orchestration.
        
        This runs all analysis stages efficiently by parallelizing independent stages:
        
        Stage 1 (parallel):
          - Audio energy analysis
          - Motion/scenes analysis  
          - Audio events detection (7.3)
          - Whisper transcription (if enabled)
          - Chat features (if chat available)
        
        Stage 2 (after Stage 1):
          - Highlight scoring (combines all Stage 1 signals)
        
        Stage 3 (after highlights):
          - Speech features + reaction audio
          - Candidate reranking
        
        Stage 4 (after reranking, optional):
          - Context/boundaries analysis
          - Clip variants
          - AI Director
        
        Progress is streamed via SSE job updates as each sub-stage completes.
        """
        import threading
        import concurrent.futures

        proj = ctx.require_project()
        analysis_cfg = ctx.profile.get("analysis", {})
        context_cfg = ctx.profile.get("context", {})
        ai_cfg = ctx.profile.get("ai", {}).get("director", {})
        body = body or {}

        # Merge overrides
        audio_cfg = {**analysis_cfg.get("audio", {}), **(body.get("audio") or {})}
        motion_cfg = {**analysis_cfg.get("motion", {}), **(body.get("motion") or {})}
        scenes_cfg = {**analysis_cfg.get("scenes", {}), **(body.get("scenes") or {})}
        highlights_cfg = {**analysis_cfg.get("highlights", {}), **(body.get("highlights") or {})}
        audio_events_cfg = {**analysis_cfg.get("audio_events", {}), **(body.get("audio_events") or {})}
        speech_cfg = {**analysis_cfg.get("speech", {}), **(body.get("speech") or {})}
        reaction_cfg = {**analysis_cfg.get("reaction_audio", {}), **(body.get("reaction_audio") or {})}
        rerank_cfg_dict = {**analysis_cfg.get("rerank", {}), **(body.get("rerank") or {})}
        context_cfg = {**context_cfg, **(body.get("context") or {})}
        ai_cfg = {**ai_cfg, **(body.get("ai") or {})}

        # Options
        include_speech = bool(body.get("include_speech", True))
        include_context = bool(body.get("include_context", True))
        include_director = bool(body.get("include_director", True))

        job = JOB_MANAGER.create("analyze_full")

        def runner() -> None:
            from ..analysis_motion import compute_motion_analysis
            from ..analysis_scenes import compute_scene_analysis

            JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting parallel analysis DAG...")
            
            completed_stages = []
            errors = []

            # Stage 1: Run independent analyses in parallel
            def run_audio():
                if not proj.audio_features_path.exists():
                    hop_s = float(audio_cfg.get("hop_seconds", 0.5))
                    compute_audio_analysis(
                        proj,
                        sample_rate=int(audio_cfg.get("sample_rate", 16000)),
                        hop_s=hop_s,
                        smooth_s=float(audio_cfg.get("smooth_seconds", 3.0)),
                        top=int(audio_cfg.get("top", 12)),
                        min_gap_s=float(audio_cfg.get("min_gap_seconds", 20.0)),
                        pre_s=float(audio_cfg.get("pre_seconds", 8.0)),
                        post_s=float(audio_cfg.get("post_seconds", 22.0)),
                        skip_start_s=float(audio_cfg.get("skip_start_seconds", 10.0)),
                    )
                return "audio"

            def run_motion():
                if not proj.motion_features_path.exists():
                    compute_motion_analysis(
                        proj,
                        sample_fps=float(motion_cfg.get("sample_fps", 3.0)),
                        scale_width=int(motion_cfg.get("scale_width", 160)),
                        smooth_s=float(motion_cfg.get("smooth_seconds", 2.5)),
                    )
                return "motion"

            def run_scenes():
                if bool(scenes_cfg.get("enabled", True)) and not proj.scenes_path.exists():
                    compute_scene_analysis(
                        proj,
                        threshold_z=float(scenes_cfg.get("threshold_z", 3.5)),
                        min_scene_len_seconds=float(scenes_cfg.get("min_scene_len_seconds", 1.2)),
                        snap_window_seconds=float(scenes_cfg.get("snap_window_seconds", 1.0)),
                    )
                return "scenes"

            def run_audio_events():
                if bool(audio_events_cfg.get("enabled", True)) and not proj.audio_events_features_path.exists():
                    cfg = AudioEventsConfig.from_dict(audio_events_cfg)
                    compute_audio_events_analysis(proj, cfg=cfg)
                return "audio_events"

            def run_transcript():
                if include_speech and speech_cfg.get("enabled", True) and not proj.transcript_path.exists():
                    transcript_config = TranscriptConfig(
                        model_size=str(speech_cfg.get("model_size", "small")),
                        language=speech_cfg.get("language"),
                        device=str(speech_cfg.get("device", "cpu")),
                        compute_type=str(speech_cfg.get("compute_type", "int8")),
                        vad_filter=bool(speech_cfg.get("vad_filter", True)),
                        word_timestamps=bool(speech_cfg.get("word_timestamps", True)),
                    )
                    compute_transcript_analysis(proj, cfg=transcript_config)
                return "transcript"

            def run_chat_features():
                if proj.chat_features_path.exists():
                    return "chat_features_cached"
                chat_db_path = proj.analysis_dir / "chat.sqlite"
                if chat_db_path.exists():
                    from ..chat.features import compute_and_save_chat_features
                    hop_s = float(audio_cfg.get("hop_seconds", 0.5))
                    smooth_s = float(highlights_cfg.get("chat_smooth_seconds", 3.0))
                    compute_and_save_chat_features(proj, hop_s=hop_s, smooth_s=smooth_s)
                return "chat_features"

            # Build stage 1 task list with display names
            stage1_tasks = [
                (run_audio, "run_audio", "audio"),
                (run_motion, "run_motion", "motion"),
                (run_scenes, "run_scenes", "scenes"),
                (run_audio_events, "run_audio_events", "audio_events"),
                (run_chat_features, "run_chat_features", "chat"),
            ]
            if include_speech:
                stage1_tasks.append((run_transcript, "run_transcript", "transcript"))

            total_tasks = len(stage1_tasks)
            task_names = [t[2] for t in stage1_tasks]
            pending_tasks = set(task_names)
            
            JOB_MANAGER._set(
                job, 
                progress=0.05, 
                message=f"Stage 1: Starting {total_tasks} parallel tasks: {', '.join(task_names)}",
                result={"stage": 1, "pending": list(pending_tasks), "completed": [], "failed": []}
            )

            # Run Stage 1 in parallel using ThreadPoolExecutor
            stage1_completed = []
            stage1_failed = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(task[0]): (task[1], task[2]) for task in stage1_tasks}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    func_name, display_name = futures[future]
                    try:
                        result = future.result()
                        completed_stages.append(result)
                        stage1_completed.append(display_name)
                        pending_tasks.discard(display_name)
                        progress = 0.05 + 0.30 * ((i + 1) / total_tasks)
                        JOB_MANAGER._set(
                            job, 
                            progress=progress, 
                            message=f"Stage 1: ✓ {display_name} ({len(stage1_completed)}/{total_tasks})",
                            result={
                                "stage": 1, 
                                "pending": list(pending_tasks), 
                                "completed": stage1_completed,
                                "failed": stage1_failed,
                            }
                        )
                    except Exception as e:
                        err_msg = f"{display_name}: {e}"
                        errors.append(err_msg)
                        stage1_failed.append(display_name)
                        pending_tasks.discard(display_name)
                        progress = 0.05 + 0.30 * ((i + 1) / total_tasks)
                        JOB_MANAGER._set(
                            job,
                            progress=progress,
                            message=f"Stage 1: ✗ {display_name} failed ({len(stage1_completed)}/{total_tasks} ok)",
                            result={
                                "stage": 1,
                                "pending": list(pending_tasks),
                                "completed": stage1_completed,
                                "failed": stage1_failed,
                            }
                        )

            # Stage 2: Combine signals into highlight scores
            JOB_MANAGER._set(job, progress=0.35, message="Stage 2: Computing highlight scores...")
            try:
                compute_highlights_analysis(
                    proj,
                    audio_cfg=audio_cfg,
                    motion_cfg=motion_cfg,
                    scenes_cfg=scenes_cfg,
                    highlights_cfg=highlights_cfg,
                    audio_events_cfg=audio_events_cfg,
                    include_chat=True,
                    include_audio_events=bool(audio_events_cfg.get("enabled", True)),
                )
                completed_stages.append("highlights")
            except Exception as e:
                errors.append(f"highlights: {e}")

            # Stage 3: Speech features + reranking (if speech enabled)
            if include_speech and speech_cfg.get("enabled", True):
                # Speech features
                if proj.transcript_path.exists() and not proj.speech_features_path.exists():
                    JOB_MANAGER._set(job, progress=0.45, message="Stage 3: Extracting speech features...")
                    try:
                        hook_cfg = rerank_cfg_dict.get("hook", {})
                        phrases = hook_cfg.get("phrases", [])
                        speech_feature_config = SpeechFeatureConfig(
                            hop_seconds=float(speech_cfg.get("hop_seconds", 0.5)),
                            reaction_phrases=phrases if phrases else None,
                        )
                        compute_speech_features(proj, cfg=speech_feature_config)
                        completed_stages.append("speech_features")
                    except Exception as e:
                        errors.append(f"speech_features: {e}")

                # Reaction audio
                if reaction_cfg.get("enabled", True) and not proj.reaction_audio_features_path.exists():
                    JOB_MANAGER._set(job, progress=0.55, message="Stage 3: Analyzing reaction audio...")
                    try:
                        reaction_audio_config = ReactionAudioConfig(
                            sample_rate=int(reaction_cfg.get("sample_rate", 16000)),
                            hop_seconds=float(reaction_cfg.get("hop_seconds", 0.5)),
                            smooth_seconds=float(reaction_cfg.get("smooth_seconds", 1.5)),
                        )
                        compute_reaction_audio_features(proj, cfg=reaction_audio_config)
                        completed_stages.append("reaction_audio")
                    except Exception as e:
                        errors.append(f"reaction_audio: {e}")

                # Rerank
                proj_data = get_project_data(proj)
                has_candidates = bool(proj_data.get("analysis", {}).get("highlights", {}).get("candidates"))
                if rerank_cfg_dict.get("enabled", True) and has_candidates:
                    JOB_MANAGER._set(job, progress=0.65, message="Stage 3: Reranking candidates...")
                    try:
                        hook_cfg = rerank_cfg_dict.get("hook", {})
                        quote_cfg = rerank_cfg_dict.get("quote", {})
                        weights = rerank_cfg_dict.get("weights", {})
                        rerank_config = RerankConfig(
                            enabled=True,
                            weights=weights if weights else None,
                            hook_max_chars=int(hook_cfg.get("max_chars", 60)),
                            hook_window_seconds=float(hook_cfg.get("window_seconds", 4.0)),
                            quote_max_chars=int(quote_cfg.get("max_chars", 120)),
                            reaction_phrases=hook_cfg.get("phrases", []),
                        )
                        compute_reranked_candidates(proj, cfg=rerank_config)
                        completed_stages.append("rerank")
                    except Exception as e:
                        errors.append(f"rerank: {e}")

            # Stage 4: Context + AI Director (optional)
            if include_context:
                JOB_MANAGER._set(job, progress=0.70, message="Stage 4: Computing context boundaries...")
                try:
                    # Silence detection
                    silence_path = proj.analysis_dir / "silence.json"
                    if not silence_path.exists():
                        silence_cfg = SilenceConfig(
                            noise_db=float(context_cfg.get("silence_noise_db", -30.0)),
                            min_duration=float(context_cfg.get("silence_min_duration", 0.3)),
                        )
                        compute_silence_analysis(proj, cfg=silence_cfg)

                    # Sentence boundaries
                    sentences_path = proj.analysis_dir / "sentences.json"
                    if proj.transcript_path.exists() and not sentences_path.exists():
                        sentence_cfg = SentenceConfig(
                            max_sentence_words=int(context_cfg.get("max_sentence_words", 30)),
                        )
                        compute_sentences_analysis(proj, cfg=sentence_cfg)

                    # Chat boundaries
                    chat_boundaries_path = proj.analysis_dir / "chat_boundaries.json"
                    if proj.chat_features_path.exists() and not chat_boundaries_path.exists():
                        chat_boundary_cfg = ChatBoundaryConfig(
                            valley_window_s=float(context_cfg.get("boundaries", {}).get("chat_valley_window_s", 12.0)),
                        )
                        compute_chat_boundaries_analysis(proj, cfg=chat_boundary_cfg)

                    # Unified boundaries
                    boundary_prefs = context_cfg.get("boundaries", {})
                    boundary_cfg = BoundaryConfig(
                        prefer_silence=bool(boundary_prefs.get("prefer_silence", True)),
                        prefer_sentences=bool(boundary_prefs.get("prefer_sentences", True)),
                        prefer_scene_cuts=bool(boundary_prefs.get("prefer_scene_cuts", True)),
                        prefer_chat_valleys=bool(boundary_prefs.get("prefer_chat_valleys", True)),
                    )
                    compute_boundaries_analysis(proj, cfg=boundary_cfg)
                    completed_stages.append("boundaries")

                    # Clip variants
                    JOB_MANAGER._set(job, progress=0.80, message="Stage 4: Generating clip variants...")
                    variants_cfg_dict = context_cfg.get("variants", {})
                    short_cfg = variants_cfg_dict.get("short", {})
                    medium_cfg = variants_cfg_dict.get("medium", {})
                    long_cfg = variants_cfg_dict.get("long", {})
                    variant_cfg = VariantGeneratorConfig(
                        short=VariantDurationConfig(
                            min_s=float(short_cfg.get("min_s", 16)),
                            max_s=float(short_cfg.get("max_s", 24)),
                        ),
                        medium=VariantDurationConfig(
                            min_s=float(medium_cfg.get("min_s", 24)),
                            max_s=float(medium_cfg.get("max_s", 40)),
                        ),
                        long=VariantDurationConfig(
                            min_s=float(long_cfg.get("min_s", 40)),
                            max_s=float(long_cfg.get("max_s", 75)),
                        ),
                        chat_valley_window_s=float(boundary_prefs.get("chat_valley_window_s", 12.0)),
                    )
                    top_n = int(context_cfg.get("top_n", 25))
                    compute_clip_variants(proj, cfg=variant_cfg, top_n=top_n)
                    completed_stages.append("clip_variants")

                except Exception as e:
                    errors.append(f"context: {e}")

            # AI Director (if enabled)
            if include_director and ai_cfg.get("enabled", True):
                JOB_MANAGER._set(job, progress=0.90, message="Stage 4: Running AI Director...")
                try:
                    director_cfg = DirectorConfig(
                        enabled=True,
                        engine=str(ai_cfg.get("engine", "llama_cpp_server")),
                        endpoint=str(ai_cfg.get("endpoint", "http://127.0.0.1:11435")),
                        model_name=str(ai_cfg.get("model_name", "local-gguf-vulkan")),
                        timeout_s=float(ai_cfg.get("timeout_s", 30)),
                        max_tokens=int(ai_cfg.get("max_tokens", 256)),
                        temperature=float(ai_cfg.get("temperature", 0.2)),
                        platform=str(ai_cfg.get("platform", "shorts")),
                        fallback_to_rules=bool(ai_cfg.get("fallback_to_rules", True)),
                    )
                    top_n = int(context_cfg.get("top_n", 25))
                    compute_director_analysis(proj, cfg=director_cfg, top_n=top_n)
                    completed_stages.append("director")
                except Exception as e:
                    errors.append(f"director: {e}")

            # Final result
            final_proj_data = get_project_data(proj)
            result = {
                "completed_stages": completed_stages,
                "errors": errors,
                "candidates_count": len(final_proj_data.get("analysis", {}).get("highlights", {}).get("candidates", [])),
                "signals_used": final_proj_data.get("analysis", {}).get("highlights", {}).get("signals_used", {}),
            }

            if errors:
                JOB_MANAGER._set(
                    job,
                    status="succeeded",  # Partial success
                    progress=1.0,
                    message=f"Analysis complete with {len(errors)} errors",
                    result=result,
                )
            else:
                JOB_MANAGER._set(
                    job,
                    status="succeeded",
                    progress=1.0,
                    message="Full analysis complete!",
                    result=result,
                )

        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse({"job_id": job.id})

    @app.get("/api/clip_variants/{rank}")
    def api_get_clip_variants(rank: int) -> JSONResponse:
        """Get clip variants for a specific candidate by rank."""
        proj = ctx.require_project()
        variants = load_clip_variants(proj)
        if not variants:
            return JSONResponse({"ok": False, "reason": "no_variants"}, status_code=404)

        for cv in variants:
            if cv.candidate_rank == rank:
                return JSONResponse({
                    "ok": True,
                    "candidate_rank": cv.candidate_rank,
                    "candidate_peak_time_s": cv.candidate_peak_time_s,
                    "variants": [v.to_dict() for v in cv.variants],
                })

        return JSONResponse({"ok": False, "reason": "candidate_not_found"}, status_code=404)

    @app.get("/api/speech/timeline")
    def api_speech_timeline(max_points: int = 2000) -> JSONResponse:
        """Get speech feature timeline for visualization."""
        proj = ctx.require_project()
        if not proj.speech_features_path.exists():
            return JSONResponse({"ok": False, "reason": "no_speech_analysis"}, status_code=404)

        data = load_npz(proj.speech_features_path)
        speech_score = data.get("speech_score")
        if speech_score is None:
            return JSONResponse({"ok": False, "reason": "missing_scores"}, status_code=404)

        speech_score = speech_score.astype(float)
        n = len(speech_score)
        if n <= 0:
            return JSONResponse({"ok": False, "reason": "empty"}, status_code=404)

        step = max(1, int(n / max(1, int(max_points))))
        idx = np.arange(0, n, step)
        down = speech_score[idx]
        if not np.isfinite(down).all():
            down = np.where(np.isfinite(down), down, 0.0)

        hop_arr = data.get("hop_seconds")
        hop_s = float(hop_arr[0]) if hop_arr is not None and len(hop_arr) > 0 else 0.5

        return JSONResponse({
            "ok": True,
            "hop_seconds": hop_s,
            "indices": idx.tolist(),
            "speech_score": down.tolist(),
        })

    @app.get("/api/reaction/timeline")
    def api_reaction_timeline(max_points: int = 2000) -> JSONResponse:
        """Get reaction audio feature timeline for visualization."""
        proj = ctx.require_project()
        if not proj.reaction_audio_features_path.exists():
            return JSONResponse({"ok": False, "reason": "no_reaction_analysis"}, status_code=404)

        data = load_npz(proj.reaction_audio_features_path)
        reaction_score = data.get("reaction_score")
        if reaction_score is None:
            return JSONResponse({"ok": False, "reason": "missing_scores"}, status_code=404)

        reaction_score = reaction_score.astype(float)
        n = len(reaction_score)
        if n <= 0:
            return JSONResponse({"ok": False, "reason": "empty"}, status_code=404)

        step = max(1, int(n / max(1, int(max_points))))
        idx = np.arange(0, n, step)
        down = reaction_score[idx]
        if not np.isfinite(down).all():
            down = np.where(np.isfinite(down), down, 0.0)

        hop_arr = data.get("hop_seconds")
        hop_s = float(hop_arr[0]) if hop_arr is not None and len(hop_arr) > 0 else 0.5

        return JSONResponse({
            "ok": True,
            "hop_seconds": hop_s,
            "indices": idx.tolist(),
            "reaction_score": down.tolist(),
        })

    @app.get("/api/audio_events/timeline")
    def api_audio_events_timeline(max_points: int = 2000) -> JSONResponse:
        """Get audio events timeline for visualization.
        
        Returns the combined event score (laughter + cheering + etc.) z-scored,
        plus individual event curves if requested.
        """
        proj = ctx.require_project()
        if not proj.audio_events_features_path.exists():
            return JSONResponse({"ok": False, "reason": "no_audio_events_analysis"}, status_code=404)

        data = load_npz(proj.audio_events_features_path)
        event_combo_z = data.get("event_combo_z")
        if event_combo_z is None:
            return JSONResponse({"ok": False, "reason": "missing_scores"}, status_code=404)

        event_combo_z = event_combo_z.astype(float)
        n = len(event_combo_z)
        if n <= 0:
            return JSONResponse({"ok": False, "reason": "empty"}, status_code=404)

        step = max(1, int(n / max(1, int(max_points))))
        idx = np.arange(0, n, step)
        combo_down = event_combo_z[idx]
        if not np.isfinite(combo_down).all():
            combo_down = np.where(np.isfinite(combo_down), combo_down, 0.0)

        # Also downsample individual event z-scores
        laughter_z = data.get("laughter_z")
        cheering_z = data.get("cheering_z")
        screaming_z = data.get("screaming_z")

        result = {
            "ok": True,
            "hop_seconds": float(data.get("hop_seconds", [0.5])[0]),
            "indices": idx.tolist(),
            "event_combo_z": combo_down.tolist(),
        }

        if laughter_z is not None:
            laughter_down = laughter_z.astype(float)[idx]
            laughter_down = np.where(np.isfinite(laughter_down), laughter_down, 0.0)
            result["laughter_z"] = laughter_down.tolist()

        if cheering_z is not None:
            cheering_down = cheering_z.astype(float)[idx]
            cheering_down = np.where(np.isfinite(cheering_down), cheering_down, 0.0)
            result["cheering_z"] = cheering_down.tolist()

        if screaming_z is not None:
            screaming_down = screaming_z.astype(float)[idx]
            screaming_down = np.where(np.isfinite(screaming_down), screaming_down, 0.0)
            result["screaming_z"] = screaming_down.tolist()

        return JSONResponse(result)

    @app.get("/api/audio_events/status")
    def api_audio_events_status() -> JSONResponse:
        """Get audio events analysis status including backend availability.
        
        Returns comprehensive status:
        - backend_selected: Which backend was used (tensorflow/onnx/heuristic)
        - available_backends: List of backends that could be used
        - unavailable_backends: Dict of unavailable backends with reasons
        - model_loaded: Whether an ML model is active
        - notes: Human-readable status message
        """
        proj = ctx.require_project()
        proj_data = get_project_data(proj)

        audio_events_analysis = proj_data.get("analysis", {}).get("audio_events", {})
        features_available = proj.audio_events_features_path.exists()
        
        # Check available backends
        from ..analysis_audio_events import _try_load_yamnet, _try_load_yamnet_onnx, _try_load_onnx_directml
        
        available_backends = []
        unavailable_backends = {}
        
        # Check TF Hub
        _, tf_err = _try_load_yamnet()
        if tf_err is None:
            available_backends.append("tensorflow")
        else:
            unavailable_backends["tensorflow"] = tf_err
        
        # Check ONNX DirectML (preferred on Windows)
        _, dml_err = _try_load_onnx_directml()
        if dml_err is None:
            available_backends.append("onnx_directml")
        else:
            unavailable_backends["onnx_directml"] = dml_err
        
        # Check ONNX CPU
        _, onnx_err = _try_load_yamnet_onnx()
        if onnx_err is None:
            available_backends.append("onnx_cpu")
        else:
            unavailable_backends["onnx_cpu"] = onnx_err
        
        # Heuristic is always available
        available_backends.append("heuristic")
        
        # Determine notes based on status
        backend_used = audio_events_analysis.get("backend", "unknown") if features_available else None
        ml_available = audio_events_analysis.get("ml_available", False) if features_available else False
        
        if backend_used == "onnx_directml":
            notes = "Using GPU via DirectML (optimal)"
        elif backend_used == "tensorflow":
            notes = "Using TensorFlow Hub YAMNet"
        elif backend_used == "onnx_cpu":
            notes = "Using ONNX CPU (consider DirectML for GPU)"
        elif backend_used == "heuristic":
            if "onnx_directml" in available_backends or "onnx_cpu" in available_backends:
                notes = "Using acoustic heuristics (ML models available but not used)"
            else:
                notes = "Using acoustic heuristics (no ML model installed)"
        elif not features_available:
            notes = "Audio events not yet analyzed"
        else:
            notes = "Unknown backend"

        return JSONResponse({
            "features_available": features_available,
            "backend_selected": backend_used,
            "available_backends": available_backends,
            "unavailable_backends": unavailable_backends,
            "model_loaded": ml_available,
            "notes": notes,
            "peaks": audio_events_analysis.get("peaks", {}) if features_available else {},
            "config": audio_events_analysis.get("config", {}) if features_available else {},
            "generated_at": audio_events_analysis.get("generated_at") if features_available else None,
        })

    # -------------------------------------------------------------------------
    # Chat Replay endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/chat/status")
    def api_chat_status() -> JSONResponse:
        """Get chat status for the current project.
        
        Returns:
            - available: Whether chat is available
            - enabled: Whether chat is enabled in project config
            - sync_offset_ms: Current sync offset
            - message_count: Number of chat messages
            - source_url: Source URL used for chat
        """
        proj = ctx.require_project()
        from ..project import get_chat_config, get_source_url

        chat_config = get_chat_config(proj)
        source_url = get_source_url(proj) or chat_config.get("source_url", "")

        chat_available = proj.chat_db_path.exists()
        message_count = 0
        duration_ms = 0

        if chat_available:
            try:
                from ..chat.store import ChatStore
                store = ChatStore(proj.chat_db_path)
                meta = store.get_all_meta()
                message_count = meta.message_count
                duration_ms = meta.duration_ms
                store.close()
            except Exception:
                pass

        return JSONResponse({
            "available": chat_available,
            "enabled": chat_config.get("enabled", False),
            "sync_offset_ms": chat_config.get("sync_offset_ms", 0),
            "message_count": message_count,
            "duration_ms": duration_ms,
            "source_url": source_url,
        })

    @app.post("/api/chat/set_source_url")
    def api_chat_set_source_url(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        """Set the source URL for chat download.
        
        Body:
            source_url: str - The URL to use for chat download
            platform: str - Optional platform hint (twitch, youtube, etc.)
        """
        proj = ctx.require_project()
        from ..project import set_source_url

        source_url = str(body.get("source_url") or "").strip()
        if not source_url:
            raise HTTPException(status_code=400, detail="source_url_required")

        platform = body.get("platform")
        set_source_url(proj, source_url, platform=platform)

        return JSONResponse({"ok": True, "source_url": source_url})

    @app.post("/api/chat/set_offset")
    def api_chat_set_offset(body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        """Set the chat sync offset.
        
        Body:
            sync_offset_ms: int - Offset in milliseconds (negative = chat earlier, positive = chat later)
        """
        proj = ctx.require_project()
        from ..project import set_chat_config

        sync_offset_ms = int(body.get("sync_offset_ms", 0))
        set_chat_config(proj, sync_offset_ms=sync_offset_ms)

        return JSONResponse({"ok": True, "sync_offset_ms": sync_offset_ms})

    @app.post("/api/chat/download")
    def api_chat_download(body: Dict[str, Any] = Body(default={})):  # type: ignore[valid-type]
        """Download chat from source URL.
        
        Body:
            source_url: str - Optional URL override. If not provided, uses project's source URL.
        
        Returns job_id for tracking progress.
        """
        import threading

        proj = ctx.require_project()
        from ..project import get_source_url, set_chat_config

        # Get source URL from body or project
        source_url = str(body.get("source_url") or "").strip()
        if not source_url:
            source_url = get_source_url(proj) or ""

        if not source_url:
            raise HTTPException(status_code=400, detail="source_url_required")

        if not (source_url.startswith("http://") or source_url.startswith("https://")):
            raise HTTPException(status_code=400, detail="invalid_url")

        job = JOB_MANAGER.create("download_chat")

        def runner() -> None:
            JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting chat download...")

            def on_progress(frac: float, msg: str) -> None:
                JOB_MANAGER._set(job, progress=frac * 0.5, message=msg)

            try:
                from ..chat.downloader import download_chat, ChatDownloadError
                from ..chat.normalize import load_and_normalize
                from ..chat.store import ChatStore, ChatMeta
                from ..chat.features import compute_and_save_chat_features

                # Download chat
                JOB_MANAGER._set(job, progress=0.05, message="Downloading chat replay...")
                result = download_chat(source_url, proj.chat_raw_path, on_progress=on_progress)

                # Load and normalize
                JOB_MANAGER._set(job, progress=0.5, message="Normalizing chat messages...")
                messages, detected_format = load_and_normalize(proj.chat_raw_path)

                # Store in SQLite
                JOB_MANAGER._set(job, progress=0.6, message="Storing chat messages...")
                store = ChatStore(proj.chat_db_path)
                store.initialize()
                store.clear_messages()
                count = store.insert_messages(messages)

                # Save metadata
                store.set_all_meta(ChatMeta(
                    source_url=source_url,
                    platform=result.platform,
                    video_id=result.video_id,
                    message_count=count,
                    duration_ms=result.duration_ms,
                    downloaded_at=__import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    downloader_version=result.downloader_version,
                    raw_file=str(proj.chat_raw_path),
                ))
                store.close()

                # Compute features
                JOB_MANAGER._set(job, progress=0.7, message="Computing chat features...")
                hop_s = float(ctx.profile.get("analysis", {}).get("audio", {}).get("hop_seconds", 0.5))
                smooth_s = float(ctx.profile.get("analysis", {}).get("highlights", {}).get("chat_smooth_seconds", 3.0))

                def on_feature_progress(frac: float) -> None:
                    JOB_MANAGER._set(job, progress=0.7 + 0.25 * frac, message="Computing chat features...")

                compute_and_save_chat_features(proj, hop_s=hop_s, smooth_s=smooth_s, on_progress=on_feature_progress)

                # Update project config
                set_chat_config(proj, enabled=True, source_url=source_url)

                JOB_MANAGER._set(
                    job,
                    status="succeeded",
                    progress=1.0,
                    message=f"Downloaded {count} chat messages",
                    result={
                        "message_count": count,
                        "platform": result.platform,
                        "source_url": source_url,
                    },
                )

            except ChatDownloadError as e:
                JOB_MANAGER._set(job, status="failed", message=str(e))
            except Exception as e:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}")

        threading.Thread(target=runner, daemon=True).start()
        return JSONResponse({"job_id": job.id})

    @app.post("/api/chat/clear")
    def api_chat_clear() -> JSONResponse:
        """Clear chat data for the current project."""
        proj = ctx.require_project()
        from ..project import set_chat_config

        # Remove chat files
        if proj.chat_db_path.exists():
            proj.chat_db_path.unlink()
        if proj.chat_raw_path.exists():
            proj.chat_raw_path.unlink()
        if proj.chat_features_path.exists():
            proj.chat_features_path.unlink()

        # Update config
        set_chat_config(proj, enabled=False)

        return JSONResponse({"ok": True})

    @app.get("/api/chat/timeline")
    def api_chat_timeline(max_points: int = 2000) -> JSONResponse:
        """Get chat spike timeline for visualization."""
        proj = ctx.require_project()
        if not proj.chat_features_path.exists():
            return JSONResponse({"ok": False, "reason": "no_chat_analysis"}, status_code=404)

        data = load_npz(proj.chat_features_path)
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

        hop_arr = data.get("hop_seconds")
        hop_s = float(hop_arr[0]) if hop_arr is not None and len(hop_arr) > 0 else 0.5

        return JSONResponse({
            "ok": True,
            "hop_seconds": hop_s,
            "indices": idx.tolist(),
            "scores": down.tolist(),
        })

    @app.get("/api/chat/messages")
    def api_chat_messages(
        start_ms: int = 0,
        end_ms: int = 60000,
        limit: int = 500,
    ) -> JSONResponse:
        """Get chat messages in a time range.
        
        Query params:
            start_ms: Start time in milliseconds (video time)
            end_ms: End time in milliseconds (video time)
            limit: Maximum messages to return
        
        Returns messages with sync_offset_ms already applied.
        """
        proj = ctx.require_project()

        if not proj.chat_db_path.exists():
            return JSONResponse({"ok": False, "reason": "no_chat"}, status_code=404)

        from ..chat.store import ChatStore
        from ..project import get_chat_config

        chat_config = get_chat_config(proj)
        offset_ms = chat_config.get("sync_offset_ms", 0)

        store = ChatStore(proj.chat_db_path)
        try:
            messages = store.get_messages(start_ms, end_ms, offset_ms=offset_ms, limit=limit)
        finally:
            store.close()

        return JSONResponse({
            "ok": True,
            "sync_offset_ms": offset_ms,
            "messages": [m.to_dict() for m in messages],
        })

    @app.get("/api/transcript")
    def api_transcript() -> JSONResponse:
        """Get the full transcript for the current project."""
        proj = ctx.require_project()
        transcript = load_transcript(proj)
        if transcript is None:
            return JSONResponse({"ok": False, "reason": "no_transcript"}, status_code=404)

        return JSONResponse({
            "ok": True,
            "language": transcript.language,
            "duration_seconds": transcript.duration_seconds,
            "segment_count": len(transcript.segments),
            "segments": [s.to_dict() for s in transcript.segments],
        })

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

        # Auto-fill hook text from AI director or candidate if enabled and text is blank
        if hook_cfg.get("enabled") and not hook_cfg.get("text"):
            # First try AI director results
            director_results = proj_data.get("analysis", {}).get("director", {}).get("results", [])
            candidate_rank = selection.get("candidate_rank")
            if candidate_rank is not None:
                ai_result = next((r for r in director_results if r.get("candidate_rank") == candidate_rank), None)
                if ai_result and ai_result.get("hook"):
                    hook_cfg = {**hook_cfg, "text": ai_result.get("hook")}
                else:
                    # Fallback to candidate hook_text
                    candidates = proj_data.get("analysis", {}).get("highlights", {}).get("candidates", [])
                    matching_cand = next((c for c in candidates if c.get("rank") == candidate_rank), None)
                    if matching_cand and matching_cand.get("hook_text"):
                        hook_cfg = {**hook_cfg, "text": matching_cand.get("hook_text")}

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

        # Load AI director results for hook text lookups
        director_results = proj_data.get("director_results", [])

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
            director_results=director_results,
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
