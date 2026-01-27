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
from ..analysis_chapters import ChapterConfig, compute_chapters_analysis
from ..analysis_highlights import compute_highlights_analysis
from ..analysis_transcript import TranscriptConfig, compute_transcript_analysis, load_transcript
from ..analysis import run_analysis, BUNDLE_FULL
from ..analysis_speech_features import SpeechFeatureConfig, compute_speech_features, load_speech_features
from ..analysis_reaction_audio import ReactionAudioConfig, compute_reaction_audio_features
from ..enrich_candidates import EnrichConfig, enrich_candidates
from ..analysis_silence import SilenceConfig, compute_silence_analysis
from ..analysis_sentences import SentenceConfig, compute_sentences_analysis
from ..analysis_chat_boundaries import ChatBoundaryConfig, compute_chat_boundaries_analysis
from ..analysis_boundaries import BoundaryConfig, compute_boundaries_analysis
from ..clip_variants import VariantGeneratorConfig, VariantDurationConfig, compute_clip_variants, load_clip_variants
from ..ai.director import DirectorConfig, compute_director_analysis
from ..ai.helpers import get_llm_complete_fn
from ..layouts import RectNorm
from ..project import create_or_load_project, get_project_data, load_npz, set_layout_facecam, Project, utc_now_iso
from ..profile import load_profile
from ..publisher.accounts import AccountStore
from ..publisher.jobs import PublishJobStore
from ..publisher.queue import PublishWorker
from .jobs import JOB_MANAGER, with_prevent_sleep
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

    @app.get("/api/system/info")
    def api_system_info() -> JSONResponse:
        """Get system information including available transcription backends."""
        from ..transcription import get_available_backends
        
        backends = get_available_backends()
        
        # Recommended: openai_whisper with GPU (supports AMD ROCm + NVIDIA CUDA)
        if backends.get("openai_whisper_gpu"):
            recommended = "openai_whisper"
        elif backends.get("openai_whisper"):
            recommended = "openai_whisper"
        elif backends.get("faster_whisper"):
            recommended = "faster_whisper"
        else:
            recommended = "whispercpp"
        
        return JSONResponse({
            "transcription": {
                "backends": backends,
                "recommended": recommended,
                "gpu_available": backends.get("openai_whisper_gpu") or backends.get("faster_whisper_gpu") or backends.get("whispercpp_gpu"),
            },
        })

    @app.get("/api/system/gpu")
    def api_system_gpu() -> JSONResponse:
        """Get GPU memory status and optionally clear cache."""
        import gc
        
        result: Dict[str, Any] = {
            "available": False,
            "device_name": None,
            "memory": None,
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                result["available"] = True
                result["device_name"] = torch.cuda.get_device_name(0)
                result["device_count"] = torch.cuda.device_count()
                
                # Memory stats in GB
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
                
                result["memory"] = {
                    "allocated_gb": round(allocated, 3),
                    "reserved_gb": round(reserved, 3),
                    "max_allocated_gb": round(max_allocated, 3),
                }
        except ImportError:
            result["error"] = "PyTorch not installed"
        except Exception as e:
            result["error"] = str(e)
        
        return JSONResponse(result)

    @app.post("/api/system/gpu/clear")
    def api_system_gpu_clear() -> JSONResponse:
        """Force clear GPU memory cache."""
        import gc
        
        result: Dict[str, Any] = {
            "success": False,
            "before": None,
            "after": None,
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                # Get memory before
                before_allocated = torch.cuda.memory_allocated(0) / 1024**3
                before_reserved = torch.cuda.memory_reserved(0) / 1024**3
                result["before"] = {
                    "allocated_gb": round(before_allocated, 3),
                    "reserved_gb": round(before_reserved, 3),
                }
                
                # Force Python garbage collection first
                gc.collect()
                
                # Clear CUDA/ROCm cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                
                # Get memory after
                after_allocated = torch.cuda.memory_allocated(0) / 1024**3
                after_reserved = torch.cuda.memory_reserved(0) / 1024**3
                result["after"] = {
                    "allocated_gb": round(after_allocated, 3),
                    "reserved_gb": round(after_reserved, 3),
                }
                
                result["success"] = True
                result["freed_gb"] = round(before_reserved - after_reserved, 3)
            else:
                result["error"] = "No GPU available"
        except ImportError:
            result["error"] = "PyTorch not installed"
        except Exception as e:
            result["error"] = str(e)
        
        return JSONResponse(result)

    @app.get("/api/system/llm")
    def api_system_llm() -> JSONResponse:
        """Get LLM server status."""
        import subprocess
        import urllib.request
        
        result: Dict[str, Any] = {
            "running": False,
            "endpoint": None,
            "model": None,
        }
        
        # Get configured endpoint from profile
        ai_cfg = ctx.profile.get("ai", {}).get("director", {})
        endpoint = ai_cfg.get("endpoint", "http://127.0.0.1:11435")
        result["endpoint"] = endpoint
        
        # Check if server is running
        try:
            req = urllib.request.Request(f"{endpoint}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    result["running"] = True
                    
                    # Try to get model info
                    try:
                        import json
                        req2 = urllib.request.Request(f"{endpoint}/v1/models")
                        with urllib.request.urlopen(req2, timeout=2.0) as resp2:
                            data = json.loads(resp2.read().decode('utf-8'))
                            if data.get("data"):
                                result["model"] = data["data"][0].get("id", "unknown")
                    except:
                        pass
        except:
            pass
        
        # Check process info
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5
            )
            if "llama-server.exe" in proc.stdout:
                # Parse PID and memory from CSV output
                import csv
                import io
                reader = csv.reader(io.StringIO(proc.stdout.strip()))
                for row in reader:
                    if len(row) >= 5 and row[0] == '"llama-server.exe"':
                        result["pid"] = int(row[1].strip('"'))
                        mem_str = row[4].strip('"').replace(',', '').replace(' K', '')
                        result["memory_mb"] = int(mem_str) // 1024
                        break
        except:
            pass
        
        return JSONResponse(result)

    @app.post("/api/system/llm/stop")
    def api_system_llm_stop() -> JSONResponse:
        """Stop the LLM server to free GPU memory."""
        import subprocess
        
        result: Dict[str, Any] = {
            "success": False,
            "was_running": False,
        }
        
        # Check if it was running first
        try:
            import urllib.request
            ai_cfg = ctx.profile.get("ai", {}).get("director", {})
            endpoint = ai_cfg.get("endpoint", "http://127.0.0.1:11435")
            req = urllib.request.Request(f"{endpoint}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                result["was_running"] = resp.status == 200
        except:
            pass
        
        # Kill the process
        try:
            proc = subprocess.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                capture_output=True, text=True, timeout=10
            )
            result["success"] = proc.returncode == 0 or "not found" in proc.stderr.lower()
            if proc.returncode == 0:
                result["message"] = "LLM server stopped"
            else:
                result["message"] = proc.stderr.strip() or "No server to stop"
        except Exception as e:
            result["error"] = str(e)
        
        return JSONResponse(result)

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
        
        # Add chat AI analysis status
        chat_ai_status = None
        if proj.chat_db_path.exists():
            try:
                from ..chat.store import ChatStore
                store = ChatStore(proj.chat_db_path)
                
                # Check for laugh tokens (AI-learned emotes)
                laugh_tokens_json = store.get_meta("laugh_tokens_json", "[]")
                laugh_tokens = json.loads(laugh_tokens_json) if laugh_tokens_json else []
                laugh_source = store.get_meta("laugh_tokens_source", "")
                laugh_updated = store.get_meta("laugh_tokens_updated_at", "")
                
                # Get LLM-learned tokens specifically (vs seed tokens)
                llm_learned_json = store.get_meta("laugh_tokens_llm_learned", "[]")
                llm_learned = json.loads(llm_learned_json) if llm_learned_json else []
                
                # Get message counts
                total_messages = store.get_message_count()
                
                # Get additional stats from project.json if available
                chat_analysis = data.get("analysis", {}).get("chat", {})
                newly_learned_count = chat_analysis.get("newly_learned_count", 0)
                newly_learned_tokens = chat_analysis.get("newly_learned_tokens", [])
                loaded_from_global = chat_analysis.get("loaded_from_global", 0)
                
                chat_ai_status = {
                    "has_chat": total_messages > 0,
                    "message_count": total_messages,
                    "laugh_tokens": laugh_tokens,
                    "laugh_tokens_count": len(laugh_tokens),
                    "llm_learned_tokens": llm_learned,
                    "llm_learned_count": len(llm_learned),
                    "newly_learned_count": newly_learned_count,
                    "newly_learned_tokens": newly_learned_tokens,
                    "loaded_from_global": loaded_from_global,
                    "laugh_source": laugh_source,  # "llm" or "seed" or ""
                    "laugh_updated_at": laugh_updated,
                    "ai_analyzed": laugh_source.startswith("llm") if laugh_source else False,
                }
                store.close()
            except Exception as e:
                import traceback
                print(f"[chat_ai_status] Error reading chat DB: {e}")
                traceback.print_exc()
                chat_ai_status = {"has_chat": False, "error": f"Failed to read chat database: {e}"}
        else:
            chat_ai_status = {"has_chat": False}
        
        data["chat_ai_status"] = chat_ai_status
        return JSONResponse({"active": True, "project": data})

    @app.post("/api/project/reset_analysis")
    def api_project_reset_analysis(body: Dict[str, Any] = Body(default={})) -> JSONResponse:
        """Reset all analysis data for the current project.
        
        This deletes all analysis files (audio, motion, highlights, transcript, etc.)
        but keeps the project, video, selections, and exports intact.
        
        Body (optional):
            keep_chat: bool - If True, keep chat data (default False)
            keep_transcript: bool - If True, keep transcript (default False)
        """
        import shutil
        
        proj = ctx.require_project()
        keep_chat = body.get("keep_chat", False)
        keep_transcript = body.get("keep_transcript", False)
        
        deleted_files = []
        
        # List of analysis files to delete
        analysis_files = [
            proj.audio_features_path,
            proj.motion_features_path,
            proj.highlights_features_path,
            proj.scenes_path,
            proj.chapters_path,
            proj.speech_features_path,
            proj.reaction_audio_features_path,
            proj.audio_events_features_path,
            proj.analysis_dir / "silence.json",
            proj.analysis_dir / "sentences.json",
            proj.analysis_dir / "boundaries.json",
            proj.analysis_dir / "chat_boundaries.json",
            proj.analysis_dir / "clip_variants.json",
            proj.analysis_dir / "director_results.json",
        ]
        
        # Add chat files unless keeping
        if not keep_chat:
            analysis_files.extend([
                proj.chat_features_path,
                proj.chat_raw_path,
                proj.chat_db_path,
            ])
        
        # Add transcript files unless keeping
        if not keep_transcript:
            analysis_files.append(proj.transcript_path)
        
        # Delete each file
        for fpath in analysis_files:
            if fpath.exists():
                try:
                    fpath.unlink()
                    deleted_files.append(fpath.name)
                except Exception as e:
                    pass  # Continue on error
        
        # Clear analysis data from project.json
        proj_json_path = proj.project_json_path
        if proj_json_path.exists():
            import json
            with open(proj_json_path, "r", encoding="utf-8") as f:
                proj_data = json.load(f)
            
            # Clear analysis section but preserve other data
            if "analysis" in proj_data:
                del proj_data["analysis"]
            
            with open(proj_json_path, "w", encoding="utf-8") as f:
                json.dump(proj_data, f, indent=2)
        
        return JSONResponse({
            "reset": True,
            "deleted_files": deleted_files,
            "kept_chat": keep_chat,
            "kept_transcript": keep_transcript,
        })

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

    @app.delete("/api/home/video")
    def api_home_delete_video(body: Dict[str, Any] = Body(...)) -> JSONResponse:
        """Delete a video file and optionally its project.
        
        Body:
            video_path: str - Path to the video file
            delete_project: bool - Also delete the associated project (default True)
        """
        import shutil
        from ..project import default_projects_root, project_dir_for_video
        
        video_path = body.get("video_path")
        delete_project = body.get("delete_project", True)
        
        if not video_path:
            raise HTTPException(status_code=400, detail="video_path_required")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="video_not_found")
        
        deleted_files = []
        project_deleted = False
        
        # Close project if this video is currently open
        if ctx.project and Path(ctx.project.video_path).resolve() == video_path.resolve():
            ctx.close_project()
        
        # Delete associated project if requested
        if delete_project:
            try:
                project_dir = project_dir_for_video(video_path)
                if project_dir.exists():
                    shutil.rmtree(project_dir)
                    project_deleted = True
            except Exception:
                pass  # Project may not exist, continue
        
        # Delete the video file
        try:
            video_path.unlink()
            deleted_files.append(str(video_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete video: {e}")
        
        # Delete associated metadata files (.info.json, .description, etc.)
        for ext in [".info.json", ".description", ".jpg", ".webp", ".png"]:
            meta_file = video_path.with_suffix(ext)
            if meta_file.exists():
                try:
                    meta_file.unlink()
                    deleted_files.append(str(meta_file))
                except Exception:
                    pass
        
        return JSONResponse({
            "deleted": True,
            "video_path": str(video_path),
            "deleted_files": deleted_files,
            "project_deleted": project_deleted,
        })

    @app.post("/api/home/favorite")
    def api_home_toggle_favorite(body: Dict[str, Any] = Body(...)) -> JSONResponse:
        """Toggle favorite status for a video/project.
        
        Body:
            video_path: str - Path to the video file
            favorite: bool - Set favorite status (if omitted, toggles current state)
        """
        from ..project import project_dir_for_video, load_json, save_json
        
        video_path = body.get("video_path")
        if not video_path:
            raise HTTPException(status_code=400, detail="video_path_required")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="video_not_found")
        
        # Get or create project directory
        project_dir = project_dir_for_video(video_path)
        project_json_path = project_dir / "project.json"
        
        # Load existing project data or create minimal structure
        if project_json_path.exists():
            data = load_json(project_json_path)
        else:
            # Create minimal project just for favorite status
            project_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "project_id": project_dir.name,
                "created_at": utc_now_iso(),
                "video": {"path": str(video_path)},
            }
        
        # Toggle or set favorite
        current_favorite = data.get("favorite", False)
        new_favorite = body.get("favorite", not current_favorite)
        data["favorite"] = bool(new_favorite)
        
        save_json(project_json_path, data)
        
        return JSONResponse({
            "video_path": str(video_path),
            "favorite": data["favorite"],
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
                            "favorite": project_info.get("favorite", False) if project_info else False,
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
        
        # Parse whisper verbose option (defaults to True)
        whisper_verbose = bool(opts.get("whisper_verbose", True))

        job = JOB_MANAGER.create("download_url")

        @with_prevent_sleep("Downloading video")
        def runner() -> None:
            import concurrent.futures
            import logging
            log = logging.getLogger("videopipeline.studio")
            
            # Shared state for progress tracking
            current_chat_status = {"status": "pending", "message": "Waiting..."}
            transcript_status = {"status": "pending", "progress": 0, "audio_path": None}
            
            # Cancellation support - raise this exception from callbacks to abort
            class CancelledError(Exception):
                pass
            
            def check_cancel():
                """Check if job was cancelled, raise CancelledError if so."""
                if job.cancel_requested:
                    raise CancelledError("Job cancelled by user")
            
            JOB_MANAGER._set(
                job, 
                status="running", 
                progress=0.0, 
                message="Starting download...",
                result={"video_status": "downloading", "chat_status": "pending", "transcript_status": "pending", "transcript_progress": 0}
            )

            video_result = None
            chat_result = {"status": "skipped", "message": "Not a Twitch VOD"}
            chat_error = None
            
            # Track early audio analysis status (initialized early so on_video_progress can access)
            audio_rms_status = {"status": "pending", "progress": 0}
            audio_events_status = {"status": "pending", "progress": 0}

            def on_video_progress(frac: float, msg: str) -> None:
                # Check for cancellation
                check_cancel()
                
                # Video download is 0-90%, chat is 90-100%
                # Build combined message for main status line
                chat_info = ""
                if current_chat_status["status"] == "downloading":
                    chat_pct = current_chat_status.get("progress", 0)
                    chat_pct_str = f"{int(chat_pct * 100)}%" if chat_pct > 0 else ""
                    chat_info = f" | 💬 Chat: {chat_pct_str}" if chat_pct_str else " | 💬 Chat: starting..."
                elif current_chat_status["status"] == "success":
                    chat_info = " | 💬 Chat: ✓ done"
                elif current_chat_status["status"] == "failed":
                    err = str(current_chat_status.get("message", "") or "")
                    err = err.replace("\n", " ").replace("\r", " ").strip()
                    if len(err) > 120:
                        err = err[:117] + "..."
                    chat_info = f" | 💬 Chat: ✗ failed ({err})" if err else " | 💬 Chat: ✗ failed"
                elif current_chat_status["status"] == "pending" and "twitch.tv/videos/" in url.lower():
                    chat_info = " | 💬 Chat: pending"
                
                # Build transcript info for message
                transcript_info = ""
                ts = transcript_status.get("status", "pending")
                if ts == "downloading_audio":
                    transcript_info = " | 🎙️ Audio: downloading..."
                elif ts == "transcribing":
                    tp = transcript_status.get("progress", 0)
                    transcript_info = f" | 🎙️ Transcribing: {int(tp * 100)}%"
                elif ts == "complete":
                    transcript_info = " | 🎙️ Transcript: ✓"
                elif ts == "failed":
                    transcript_info = " | 🎙️ Transcript: ✗"
                
                # Build audio RMS info for message
                audio_rms_info = ""
                rms_stat = audio_rms_status.get("status", "pending")
                if rms_stat == "analyzing":
                    rp = audio_rms_status.get("progress", 0)
                    audio_rms_info = f" | 📊 RMS: {int(rp * 100)}%"
                elif rms_stat == "complete":
                    audio_rms_info = " | 📊 RMS: ✓"
                elif rms_stat == "failed":
                    audio_rms_info = " | 📊 RMS: ✗"
                
                # Build audio events info for message
                audio_events_info = ""
                events_stat = audio_events_status.get("status", "pending")
                if events_stat == "analyzing":
                    ep = audio_events_status.get("progress", 0)
                    audio_events_info = f" | 🎭 Events: {int(ep * 100)}%"
                elif events_stat == "complete":
                    audio_events_info = " | 🎭 Events: ✓"
                elif events_stat == "failed":
                    audio_events_info = " | 🎭 Events: ✗"
                
                JOB_MANAGER._set(
                    job, 
                    progress=frac * 0.9, 
                    message=f"{msg}{chat_info}{transcript_info}{audio_rms_info}{audio_events_info}",
                    result={
                        "video_status": "downloading",
                        "video_progress": frac,
                        "chat_status": current_chat_status["status"],
                        "chat_progress": current_chat_status.get("progress", 0),
                        "chat_message": current_chat_status.get("message", ""),
                        "chat_messages_count": current_chat_status.get("messages_count", 0),
                        "transcript_status": transcript_status.get("status", "pending"),
                        "transcript_progress": transcript_status.get("progress", 0),
                        "audio_progress": transcript_status.get("audio_progress", 0),
                        "audio_total_bytes": transcript_status.get("audio_total_bytes"),
                        "audio_downloaded_bytes": transcript_status.get("audio_downloaded_bytes"),
                        "audio_speed": transcript_status.get("audio_speed"),
                        "audio_rms_status": audio_rms_status.get("status", "pending"),
                        "audio_rms_progress": audio_rms_status.get("progress", 0),
                        "audio_events_status": audio_events_status.get("status", "pending"),
                        "audio_events_progress": audio_events_status.get("progress", 0),
                    }
                )

            def download_video():
                nonlocal video_result
                video_result = download_url(url, request=request, on_progress=on_video_progress)
                return video_result

            def download_chat():
                nonlocal chat_result, chat_error, current_chat_status
                import logging
                log = logging.getLogger("videopipeline.studio")
                log.info(f"[CHAT DEBUG] Starting chat download for URL: {url}")
                
                # Only download chat for Twitch VODs
                if "twitch.tv/videos/" not in url.lower():
                    log.info(f"[CHAT DEBUG] Skipped: not a Twitch VOD")
                    chat_result = {"status": "skipped", "message": "Not a Twitch VOD"}
                    current_chat_status = chat_result
                    return chat_result
                
                try:
                    from ..chat.downloader import download_chat as dl_chat, find_twitch_downloader_cli
                    
                    cli_path = find_twitch_downloader_cli()
                    log.info(f"[CHAT DEBUG] TwitchDownloaderCLI path: {cli_path}")
                    if not cli_path:
                        chat_result = {"status": "skipped", "message": "TwitchDownloaderCLI not found"}
                        current_chat_status = chat_result
                        return chat_result
                    
                    # Extract video ID from URL
                    import re
                    match = re.search(r'twitch\.tv/videos/(\d+)', url)
                    if not match:
                        log.info(f"[CHAT DEBUG] Could not extract video ID from URL")
                        chat_result = {"status": "skipped", "message": "Could not extract video ID"}
                        current_chat_status = chat_result
                        return chat_result
                    
                    video_id = match.group(1)
                    log.info(f"[CHAT DEBUG] Extracted video ID: {video_id}")
                    
                    # Download to temp location first, will move to project later
                    # Use job-unique filename to avoid overwrite prompts on repeat runs
                    from ..ingest.ytdlp_runner import _default_downloads_dir
                    chat_temp_path = _default_downloads_dir() / f"chat_{video_id}_{job.id}.json"
                    log.info(f"[CHAT DEBUG] Chat temp path: {chat_temp_path}")
                    
                    # Update status to downloading
                    current_chat_status["status"] = "downloading"
                    current_chat_status["message"] = "Downloading Twitch chat..."
                    current_chat_status["progress"] = 0
                    current_chat_status["messages_count"] = 0
                    
                    def on_chat_progress(frac: float, msg: str) -> None:
                        # Check for cancellation
                        check_cancel()
                        
                        current_chat_status["message"] = msg
                        current_chat_status["progress"] = frac
                        
                        # Try to extract message count from status text
                        import re as re_mod
                        count_match = re_mod.search(r"(\d+)\s*messages?", msg, re_mod.IGNORECASE)
                        if count_match:
                            current_chat_status["messages_count"] = int(count_match.group(1))
                        
                        # Update job progress if video is done (progress > 90%)
                        # Chat download is 90-95%, import is 95-100%
                        JOB_MANAGER._set(
                            job,
                            progress=0.90 + frac * 0.05,
                            message=f"Chat: {msg}",
                            result={
                                "video_status": "complete",
                                "chat_status": "downloading",
                                "chat_progress": frac,
                                "chat_message": msg,
                                "chat_messages_count": current_chat_status.get("messages_count", 0),
                                "transcript_status": transcript_status.get("status", "pending"),
                                "transcript_progress": transcript_status.get("progress", 0),
                            }
                        )
                    
                    log.info(f"[CHAT DEBUG] Calling dl_chat...")
                    
                    def check_cancel_chat() -> bool:
                        return job.cancel_requested
                    
                    dl_chat(url, chat_temp_path, on_progress=on_chat_progress, check_cancel=check_cancel_chat)
                    log.info(f"[CHAT DEBUG] dl_chat completed")
                    
                    # Check if file was created
                    if chat_temp_path.exists():
                        file_size = chat_temp_path.stat().st_size
                        log.info(f"[CHAT DEBUG] Chat file created: {file_size} bytes")
                    else:
                        log.warning(f"[CHAT DEBUG] Chat file NOT created!")
                    
                    chat_result = {
                        "status": "success",
                        "message": "Chat downloaded",
                        "temp_path": str(chat_temp_path),
                        "video_id": video_id,
                    }
                    current_chat_status["status"] = "success"
                    current_chat_status["message"] = "Chat downloaded"
                    log.info(f"[CHAT DEBUG] chat_result: {chat_result}")
                    return chat_result
                    
                except Exception as e:
                    import traceback
                    log.error(f"[CHAT DEBUG] Exception: {e}\n{traceback.format_exc()}")
                    chat_error = str(e)
                    chat_result = {"status": "failed", "message": str(e)}
                    current_chat_status["status"] = "failed"
                    current_chat_status["message"] = str(e)
                    return chat_result

            # Helper to clean up downloaded files on cancellation
            def cleanup_downloads():
                try:
                    if video_result and video_result.video_path:
                        vp = Path(video_result.video_path)
                        if vp.exists():
                            vp.unlink()
                    if video_result and video_result.preview_path:
                        pp = Path(video_result.preview_path)
                        if pp.exists():
                            pp.unlink()
                    if chat_result.get("temp_path"):
                        cp = Path(chat_result["temp_path"])
                        if cp.exists():
                            cp.unlink()
                    # Also clean up temp audio file if it exists
                    audio_path = transcript_status.get("audio_path")
                    if audio_path:
                        ap = Path(audio_path)
                        if ap.exists():
                            ap.unlink()
                except Exception:
                    pass  # Best effort cleanup

            # Track parallel transcription (transcript_status already initialized above)
            transcript_result = {"status": "skipped"}
            
            def download_audio_for_transcript():
                """Download audio track for early transcription start."""
                nonlocal transcript_status
                import logging
                log = logging.getLogger("videopipeline.studio")
                
                # Check if transcription is enabled
                speech_cfg = ctx.profile.get("analysis", {}).get("speech", {})
                if not speech_cfg.get("enabled", True):
                    log.info("[AUDIO] Transcription disabled, skipping audio download")
                    transcript_status["status"] = "disabled"
                    return None
                
                try:
                    from ..ingest.ytdlp_runner import download_audio_only, _default_downloads_dir
                    
                    log.info("[AUDIO] Starting audio-only download for early transcription...")
                    transcript_status["status"] = "downloading_audio"
                    
                    def on_audio_progress(frac: float, msg: str, extra: dict = None) -> None:
                        # Check for cancellation
                        check_cancel()
                        
                        # Store actual audio download progress (0-100%) separately
                        transcript_status["audio_progress"] = frac
                        # Overall transcript progress: audio download is 0-10%, transcription is 10-100%
                        transcript_status["progress"] = frac * 0.1
                        if extra:
                            if extra.get("total_bytes"):
                                transcript_status["audio_total_bytes"] = extra["total_bytes"]
                            if extra.get("downloaded_bytes"):
                                transcript_status["audio_downloaded_bytes"] = extra["downloaded_bytes"]
                            if extra.get("speed"):
                                transcript_status["audio_speed"] = extra["speed"]
                    
                    audio_path = download_audio_only(url, on_progress=on_audio_progress)
                    
                    if audio_path and audio_path.exists():
                        log.info(f"[AUDIO] Audio downloaded: {audio_path}")
                        transcript_status["audio_path"] = str(audio_path)
                        transcript_status["status"] = "audio_ready"
                        return audio_path
                    else:
                        log.warning("[AUDIO] Audio download returned no file")
                        transcript_status["status"] = "audio_failed"
                        return None
                        
                except Exception as e:
                    log.warning(f"[AUDIO] Audio download failed: {e}")
                    transcript_status["status"] = "audio_failed"
                    return None
            
            def run_early_transcription(audio_path: Path):
                """Run transcription on downloaded audio while video downloads."""
                nonlocal transcript_result, transcript_status
                import logging
                log = logging.getLogger("videopipeline.studio")
                
                try:
                    from ..analysis_transcript import TranscriptConfig, compute_transcript_analysis_from_audio
                    
                    speech_cfg = ctx.profile.get("analysis", {}).get("speech", {})
                    transcript_config = TranscriptConfig(
                        backend=str(speech_cfg.get("backend", "auto")),
                        model_size=str(speech_cfg.get("model_size", "small.en")),
                        language=speech_cfg.get("language"),
                        sample_rate=int(speech_cfg.get("sample_rate", 16000)),
                        device=str(speech_cfg.get("device", "cpu")),
                        compute_type=str(speech_cfg.get("compute_type", "int8")),
                        vad_filter=bool(speech_cfg.get("vad_filter", True)),
                        word_timestamps=bool(speech_cfg.get("word_timestamps", True)),
                        use_gpu=bool(speech_cfg.get("use_gpu", False)),
                        threads=int(speech_cfg.get("threads", 0)),
                        n_processors=int(speech_cfg.get("n_processors", 1)),
                        strict=bool(speech_cfg.get("strict", False)),
                        verbose=whisper_verbose,  # From download options
                    )
                    
                    log.info(f"[TRANSCRIPT] Starting early transcription from audio...")
                    transcript_status["status"] = "transcribing"
                    
                    def on_transcript_progress(frac: float) -> None:
                        # Check for cancellation
                        check_cancel()
                        
                        # Transcript is 10-100% of the transcript phase
                        transcript_status["progress"] = 0.1 + frac * 0.9
                    
                    result = compute_transcript_analysis_from_audio(
                        audio_path,
                        cfg=transcript_config,
                        on_progress=on_transcript_progress,
                    )
                    
                    transcript_status["status"] = "complete"
                    transcript_status["progress"] = 1.0
                    transcript_result = {"status": "success", "data": result}
                    log.info(f"[TRANSCRIPT] Early transcription complete: {result.get('segment_count', 0)} segments")
                    return result
                    
                except ImportError:
                    # compute_transcript_analysis_from_audio doesn't exist yet
                    log.info("[TRANSCRIPT] Early transcription not available, will run after video download")
                    transcript_status["status"] = "deferred"
                    return None
                except Exception as e:
                    log.warning(f"[TRANSCRIPT] Early transcription failed: {e}")
                    transcript_status["status"] = "failed"
                    transcript_result = {"status": "failed", "error": str(e)}
                    return None
            
            # Track early project for DAG-based analysis
            early_proj = None
            early_dag_result = None
            early_dag_status = {"status": "pending", "progress": 0, "message": ""}
            
            def run_early_dag_analysis(audio_path: Path, chat_temp_path: Optional[Path] = None):
                """Create project early and run pre_download DAG bundle."""
                nonlocal early_proj, early_dag_result, early_dag_status, audio_rms_status, audio_events_status, transcript_status
                import logging
                import shutil
                log = logging.getLogger("videopipeline.studio")
                
                try:
                    from ..project import create_project_early, set_source_url
                    from ..analysis import run_analysis, BUNDLE_PRE_DOWNLOAD
                    
                    # Extract content ID from URL
                    import re
                    content_id = url  # Fallback to full URL
                    # Try to extract video ID for cleaner project ID
                    twitch_match = re.search(r'twitch\.tv/videos/(\d+)', url)
                    youtube_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', url)
                    if twitch_match:
                        content_id = f"twitch_{twitch_match.group(1)}"
                    elif youtube_match:
                        content_id = f"youtube_{youtube_match.group(1)}"
                    
                    log.info(f"[DAG] Creating early project for content_id: {content_id}")
                    early_dag_status["status"] = "creating_project"
                    early_dag_status["message"] = "Creating project..."
                    
                    # Create project early
                    early_proj = create_project_early(
                        content_id,
                        source_url=url,
                        audio_path=audio_path,
                    )
                    set_source_url(early_proj, url)
                    
                    log.info(f"[DAG] Early project created at: {early_proj.project_dir}")
                    
                    # Copy audio to project for analysis
                    project_audio_path = early_proj.analysis_dir / "audio_source.m4a"
                    early_proj.analysis_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(audio_path, project_audio_path)
                    log.info(f"[DAG] Copied audio to project: {project_audio_path}")
                    
                    # Import chat if available
                    if chat_temp_path and chat_temp_path.exists():
                        try:
                            from ..chat.downloader import import_chat_to_project
                            import_chat_to_project(early_proj, chat_temp_path)
                            log.info(f"[DAG] Imported chat to early project")
                        except Exception as e:
                            log.warning(f"[DAG] Chat import failed: {e}")
                    
                    # Build config from profile
                    analysis_cfg = ctx.profile.get("analysis", {})
                    dag_config = {
                        "audio": analysis_cfg.get("audio", {}),
                        "silence": analysis_cfg.get("silence", {}),
                        "speech": analysis_cfg.get("speech", {}),
                        "sentences": analysis_cfg.get("sentences", {}),
                        "speech_features": analysis_cfg.get("speech_features", {}),
                        "highlights": analysis_cfg.get("highlights", {}),
                        "audio_events": analysis_cfg.get("audio_events", {}),
                        "include_chat": True,
                        "include_audio_events": bool(analysis_cfg.get("audio_events", {}).get("enabled", True)),
                    }
                    
                    # Add verbose transcript setting
                    if whisper_verbose:
                        dag_config["speech"]["verbose"] = True
                    
                    log.info(f"[DAG] Starting pre_download bundle analysis...")
                    early_dag_status["status"] = "analyzing"
                    
                    def on_dag_progress(frac: float, msg: str) -> None:
                        check_cancel()
                        early_dag_status["progress"] = frac
                        early_dag_status["message"] = msg
                        
                        # Update sub-status based on what DAG is doing
                        if "transcript" in msg.lower():
                            transcript_status["status"] = "transcribing"
                            transcript_status["progress"] = frac
                        elif "audio_features" in msg.lower():
                            audio_rms_status["status"] = "analyzing"
                            audio_rms_status["progress"] = frac
                        elif "audio_events" in msg.lower():
                            audio_events_status["status"] = "analyzing"
                            audio_events_status["progress"] = frac
                    
                    result = run_analysis(
                        early_proj,
                        bundle="pre_download",
                        config=dag_config,
                        on_progress=on_dag_progress,
                    )
                    
                    early_dag_status["status"] = "complete"
                    early_dag_status["progress"] = 1.0
                    early_dag_result = result
                    
                    # Update sub-statuses
                    transcript_status["status"] = "complete"
                    transcript_status["progress"] = 1.0
                    audio_rms_status["status"] = "complete"
                    audio_rms_status["progress"] = 1.0
                    if analysis_cfg.get("audio_events", {}).get("enabled", True):
                        audio_events_status["status"] = "complete"
                        audio_events_status["progress"] = 1.0
                    
                    log.info(f"[DAG] Pre-download analysis complete: {len(result.tasks_run)} tasks in {result.total_elapsed_seconds:.1f}s")
                    return result
                    
                except Exception as e:
                    import traceback
                    log.error(f"[DAG] Early analysis failed: {e}\n{traceback.format_exc()}")
                    early_dag_status["status"] = "failed"
                    early_dag_status["message"] = str(e)
                    return None

            try:
                # Run video, chat, and audio downloads in parallel
                # Audio finishes first -> start DAG analysis while video continues
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                    video_future = executor.submit(download_video)
                    chat_future = executor.submit(download_chat)
                    audio_future = executor.submit(download_audio_for_transcript)
                    
                    dag_future = None
                    audio_path_for_transcript = None
                    chat_temp_for_dag = None
                    
                    # Wait for all to complete, checking for cancellation
                    # and starting analysis tasks when audio is ready
                    while True:
                        if job.cancel_requested:
                            # Job was cancelled - clean up and exit
                            cleanup_downloads()
                            return
                        
                        # Check if audio and chat are done and we haven't started DAG yet
                        audio_done = audio_future and audio_future.done()
                        chat_done_early = chat_future and chat_future.done()
                        
                        if audio_done and dag_future is None:
                            try:
                                audio_path_for_transcript = audio_future.result()
                                
                                # Get chat temp path if available
                                if chat_done_early:
                                    try:
                                        chat_result_early = chat_future.result()
                                        if chat_result_early and chat_result_early.get("status") == "success":
                                            chat_temp_for_dag = Path(chat_result_early.get("temp_path", ""))
                                    except Exception:
                                        pass
                                
                                if audio_path_for_transcript:
                                    log.info(f"[DOWNLOAD] Audio ready, starting DAG analysis...")
                                    # Use the new DAG-based early analysis
                                    dag_future = executor.submit(
                                        run_early_dag_analysis,
                                        audio_path_for_transcript,
                                        chat_temp_for_dag if chat_temp_for_dag and chat_temp_for_dag.exists() else None,
                                    )
                                else:
                                    log.info(f"[DOWNLOAD] Audio download skipped or failed")
                            except Exception as e:
                                log.warning(f"[DOWNLOAD] Audio download error: {e}")
                        
                        # Check if all required futures are done
                        video_done = video_future.done()
                        chat_done = chat_future.done()
                        dag_done = dag_future is None or dag_future.done()
                        
                        if video_done and chat_done and dag_done:
                            break
                        
                        # Update UI with current analysis progress while waiting
                        # This is important when video finishes before analysis tasks
                        if video_done and not dag_done:
                            # Build status parts
                            msg_parts = ["Download complete!"]
                            
                            # Chat status
                            chat_stat = current_chat_status.get("status", "pending")
                            if chat_stat == "success":
                                msg_parts.append("💬 ✓")
                            elif chat_stat == "downloading":
                                msg_parts.append("💬 ...")
                            elif chat_stat == "failed":
                                msg_parts.append("💬 ✗")
                            
                            # DAG analysis status
                            dag_stat = early_dag_status.get("status", "pending")
                            if dag_stat == "analyzing":
                                dp = early_dag_status.get("progress", 0)
                                dm = early_dag_status.get("message", "analyzing")
                                msg_parts.append(f"🔬 {int(dp * 100)}% {dm}")
                            elif dag_stat == "complete":
                                msg_parts.append("🔬 ✓")
                            elif dag_stat == "failed":
                                msg_parts.append("🔬 ✗")
                            
                            # Keep overall progress at 90% until all analysis tasks finish
                            JOB_MANAGER._set(
                                job,
                                progress=0.90,
                                message=" | ".join(msg_parts),
                                result={
                                    "video_status": "complete",
                                    "video_progress": 1.0,
                                    "chat_status": current_chat_status["status"],
                                    "chat_progress": current_chat_status.get("progress", 0),
                                    "chat_message": current_chat_status.get("message", ""),
                                    "chat_messages_count": current_chat_status.get("messages_count", 0),
                                    "dag_status": early_dag_status.get("status", "pending"),
                                    "dag_progress": early_dag_status.get("progress", 0),
                                    "dag_message": early_dag_status.get("message", ""),
                                }
                            )
                        
                        # Small sleep to avoid busy-waiting
                        import time
                        time.sleep(0.5)
                    
                    # Get results (will raise if video download failed or cancelled)
                    video_future.result()  # Raises CancelledError if cancelled
                    
                    # Chat failures are non-fatal, but propagate cancellation
                    try:
                        chat_future.result()
                    except CancelledError:
                        raise  # Re-raise cancellation
                    except Exception:
                        pass  # Chat failures are non-fatal
                    
                    # Get DAG analysis result if available
                    if dag_future:
                        try:
                            early_dag_result = dag_future.result()
                            if early_dag_result:
                                log.info(f"[DOWNLOAD] Early DAG analysis completed successfully")
                        except CancelledError:
                            raise  # Re-raise cancellation
                        except Exception as e:
                            log.warning(f"[DOWNLOAD] Early DAG analysis error: {e}")
                    
                    # Clean up temp audio file (DAG copied it to project)
                    if audio_path_for_transcript and Path(audio_path_for_transcript).exists():
                        try:
                            Path(audio_path_for_transcript).unlink()
                            log.info(f"[DOWNLOAD] Cleaned up temp audio file")
                        except Exception:
                            pass

                # Check cancellation again after downloads complete
                if job.cancel_requested:
                    cleanup_downloads()
                    return

                result_dict = video_result.to_dict()
                result_dict["chat"] = chat_result

                # Auto-open the project if requested
                if auto_open:
                    import logging
                    log = logging.getLogger("videopipeline.studio")
                    
                    # Use preview if available, otherwise original
                    video_to_open = video_result.preview_path or video_result.video_path
                    
                    # If we created an early project during DAG analysis, use it and update video path
                    if early_proj:
                        from ..project import set_project_video
                        set_project_video(early_proj, Path(video_to_open))
                        proj = early_proj
                        ctx._project = proj  # Update context to use this project
                        log.info(f"[DAG] Updated early project with video: {video_to_open}")
                    else:
                        # No early project - create one now (fallback path)
                        proj = ctx.open_project(video_to_open)
                    log.info(f"[DOWNLOAD] Project ready: {proj.project_dir}")
                    
                    # Store the source URL for chat download later
                    from ..project import set_source_url
                    set_source_url(proj, url)
                    
                    # If chat was downloaded and NOT already imported by DAG, import it
                    chat_already_imported = early_proj is not None and chat_temp_for_dag is not None
                    log.info(f"[CHAT DEBUG] Checking chat_result: status={chat_result.get('status')}, already_imported={chat_already_imported}")
                    if chat_result.get("status") == "success" and chat_result.get("temp_path") and not chat_already_imported:
                        chat_temp = Path(chat_result["temp_path"])
                        log.info(f"[CHAT DEBUG] Chat temp file exists: {chat_temp.exists()}")
                        if chat_temp.exists():
                            log.info(f"[CHAT DEBUG] Chat temp file size: {chat_temp.stat().st_size} bytes")
                        
                        JOB_MANAGER._set(
                            job, 
                            progress=0.95, 
                            message="Importing chat into project...",
                            result={
                                "video_status": "complete",
                                "chat_status": "importing",
                                "chat_message": "Reading chat file...",
                                "dag_status": early_dag_status.get("status", "pending"),
                                "dag_progress": early_dag_status.get("progress", 0),
                            }
                        )
                        try:
                            from ..chat.downloader import import_chat_to_project
                            
                            # Update progress during import
                            JOB_MANAGER._set(
                                job,
                                progress=0.96,
                                message="Importing chat: parsing messages...",
                                result={
                                    "video_status": "complete",
                                    "chat_status": "importing",
                                    "chat_message": "Parsing messages...",
                                    "dag_status": early_dag_status.get("status", "pending"),
                                    "dag_progress": early_dag_status.get("progress", 0),
                                }
                            )
                            
                            log.info(f"[CHAT DEBUG] Calling import_chat_to_project...")
                            import_chat_to_project(proj, chat_temp)
                            log.info(f"[CHAT DEBUG] import_chat_to_project completed successfully")
                            
                            # Verify the chat.sqlite was created
                            chat_db = proj.chat_db_path
                            log.info(f"[CHAT DEBUG] chat_db_path: {chat_db}, exists: {chat_db.exists()}")
                            
                            # Now compute chat features with LLM
                            JOB_MANAGER._set(
                                job,
                                progress=0.97,
                                message="Learning chat emotes with AI...",
                                result={
                                    "video_status": "complete",
                                    "chat_status": "ai_learning",
                                    "chat_message": "Learning channel-specific emotes...",
                                    "transcript_status": transcript_status.get("status", "pending"),
                                    "transcript_progress": transcript_status.get("progress", 0),
                                }
                            )
                            
                            try:
                                from ..chat.features import compute_and_save_chat_features
                                
                                hop_s = float(ctx.profile.get("analysis", {}).get("audio", {}).get("hop_seconds", 0.5))
                                smooth_s = float(ctx.profile.get("analysis", {}).get("highlights", {}).get("chat_smooth_seconds", 3.0))
                                
                                # Get LLM for laugh emote learning
                                ai_cfg = ctx.profile.get("ai", {}).get("director", {})
                                
                                def on_llm_status(msg: str) -> None:
                                    JOB_MANAGER._set(job, message=msg)
                                
                                llm_complete_fn = get_llm_complete_fn(ai_cfg, proj.analysis_dir, on_status=on_llm_status)
                                
                                def on_feature_status(msg: str) -> None:
                                    JOB_MANAGER._set(job, message=msg)
                                
                                # Extract channel info from URL for global emote persistence
                                from ..chat.emote_db import get_channel_for_project
                                channel_info = get_channel_for_project(proj, source_url=url)
                                channel_id = channel_info[0] if channel_info else None
                                platform = channel_info[1] if channel_info else "twitch"
                                
                                compute_and_save_chat_features(
                                    proj,
                                    hop_s=hop_s,
                                    smooth_s=smooth_s,
                                    on_status=on_feature_status,
                                    llm_complete=llm_complete_fn,
                                    channel_id=channel_id,
                                    platform=platform,
                                )
                                log.info(f"[CHAT DEBUG] Chat features computed")
                            except Exception as e:
                                log.warning(f"[CHAT DEBUG] Chat feature computation failed: {e}")
                                # Non-fatal - features can be computed later during analysis
                            
                            JOB_MANAGER._set(
                                job,
                                progress=0.99,
                                message="Chat imported successfully",
                                result={
                                    "video_status": "complete",
                                    "chat_status": "complete",
                                    "chat_message": "Done!",
                                    "transcript_status": transcript_status.get("status", "pending"),
                                    "transcript_progress": transcript_status.get("progress", 0),
                                }
                            )
                            chat_result["imported"] = True
                            
                            # Record successful chat download in project
                            from ..project import set_chat_config
                            set_chat_config(proj, download_status="success")
                            
                            # Clean up temp file after successful import
                            try:
                                chat_temp.unlink()
                                log.info(f"[CHAT DEBUG] Cleaned up temp file")
                            except Exception:
                                pass
                        except Exception as e:
                            import traceback
                            log.error(f"[CHAT DEBUG] Import exception: {e}\n{traceback.format_exc()}")
                            chat_result["import_error"] = str(e)
                            # Record failed import status
                            from ..project import set_chat_config
                            set_chat_config(proj, download_status="failed", download_error=str(e))
                    elif chat_already_imported:
                        log.info(f"[CHAT DEBUG] Chat already imported by DAG analysis")
                        chat_result["imported"] = True
                        from ..project import set_chat_config
                        set_chat_config(proj, download_status="success")
                    else:
                        log.info(f"[CHAT DEBUG] Skipping import - condition not met")
                        # Record the chat download status
                        from ..project import set_chat_config
                        chat_status = chat_result.get("status", "unknown")
                        if chat_status == "failed":
                            set_chat_config(proj, download_status="failed", download_error=chat_result.get("message", "Download failed"))
                        elif chat_status == "skipped":
                            set_chat_config(proj, download_status="skipped")
                    
                    # Early analysis is already saved to project by DAG runner
                    # Just log what was computed
                    if early_dag_result:
                        log.info(f"[DOWNLOAD] DAG analysis completed: {len(early_dag_result.tasks_run)} tasks")
                    
                    result_dict["project"] = get_project_data(proj)
                    result_dict["auto_opened"] = True
                    result_dict["chat"] = chat_result
                    
                    # Include DAG analysis status in result
                    if early_dag_result:
                        result_dict["dag_analysis"] = {
                            "status": "complete",
                            "tasks_run": len(early_dag_result.tasks_run),
                            "elapsed_seconds": early_dag_result.total_elapsed_seconds,
                        }
                    elif early_dag_status.get("status") == "failed":
                        result_dict["dag_analysis"] = {"status": "failed", "error": early_dag_status.get("message")}

                # Build final message
                final_msg = "Download complete!"
                if early_dag_result:
                    final_msg = f"Download complete! Analysis ready ({len(early_dag_result.tasks_run)} tasks)."
                if chat_result.get("imported"):
                    final_msg += " Chat imported."
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

            except CancelledError:
                # Job was cancelled via callback - clean up
                log.info("[DOWNLOAD] Job cancelled by user")
                cleanup_downloads()
                JOB_MANAGER._set(job, status="cancelled", message="Cancelled by user")
            except ImportError as e:
                if not job.cancel_requested:
                    JOB_MANAGER._set(job, status="failed", message=str(e))
            except Exception as e:
                if not job.cancel_requested:
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

        # Normalize motion_weight_mode into actual numeric weight
        motion_mode = str(highlights_cfg.get("motion_weight_mode", "off")).lower()
        motion_weight = {
            "off": 0.0,
            "low": 0.15,
            "normal": 0.35,
            "high": 0.8,
        }.get(motion_mode, 0.0)
        highlights_cfg.setdefault("weights", {})["motion"] = motion_weight

        job = JOB_MANAGER.create("analyze_highlights")

        def runner() -> None:
            JOB_MANAGER._set(job, status="running", progress=0.0, message="analyzing highlights")  # type: ignore[attr-defined]

            def on_prog(frac: float) -> None:
                JOB_MANAGER._set(job, progress=frac, message="analyzing highlights")  # type: ignore[attr-defined]

            try:
                # Use DAG runner for full analysis with proper dependency ordering
                dag_config = {
                    "audio": audio_cfg,
                    "motion": motion_cfg,
                    "scenes": scenes_cfg,
                    "highlights": highlights_cfg,
                    "audio_events": audio_events_cfg,
                    "include_chat": True,
                    "include_audio_events": bool(audio_events_cfg.get("enabled", True)),
                }
                
                def on_dag_progress(frac: float, msg: str) -> None:
                    on_prog(frac)
                
                result = run_analysis(
                    proj,
                    bundle="full",
                    config=dag_config,
                    on_progress=on_dag_progress,
                    upgrade_mode=True,  # Re-shape when boundaries improve
                )
                JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result={"tasks_run": len(result.tasks_run)})  # type: ignore[attr-defined]
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
        # Support both "enrich" and legacy "rerank" config keys
        enrich_cfg_dict = {
            **analysis_cfg.get("enrich", analysis_cfg.get("rerank", {})),
            **(body.get("enrich") or body.get("rerank") or {})
        }

        job = JOB_MANAGER.create("analyze_speech")

        def runner() -> None:
            import threading as _threading
            JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting speech analysis...")

            try:
                # Step 1: Transcription (if needed)
                if speech_cfg.get("enabled", True) and not proj.transcript_path.exists():
                    JOB_MANAGER._set(job, progress=0.05, message="Transcribing audio with Whisper...")
                    
                    transcript_config = TranscriptConfig(
                        backend=str(speech_cfg.get("backend", "auto")),
                        model_size=str(speech_cfg.get("model_size", "small")),
                        language=speech_cfg.get("language"),
                        device=str(speech_cfg.get("device", "cpu")),
                        compute_type=str(speech_cfg.get("compute_type", "int8")),
                        vad_filter=bool(speech_cfg.get("vad_filter", True)),
                        word_timestamps=bool(speech_cfg.get("word_timestamps", True)),
                        use_gpu=bool(speech_cfg.get("use_gpu", False)),
                        threads=int(speech_cfg.get("threads", 0)),
                        n_processors=int(speech_cfg.get("n_processors", 1)),
                        strict=bool(speech_cfg.get("strict", False)),
                    )
                    
                    def on_transcript_progress(frac: float) -> None:
                        JOB_MANAGER._set(job, progress=0.05 + 0.35 * frac, message="Transcribing audio...")
                    
                    compute_transcript_analysis(proj, cfg=transcript_config, on_progress=on_transcript_progress)

                # Step 2: Speech features (if transcript exists)
                if speech_cfg.get("enabled", True) and proj.transcript_path.exists() and not proj.speech_features_path.exists():
                    JOB_MANAGER._set(job, progress=0.4, message="Extracting speech features...")
                    
                    # Get reaction phrases from config
                    hook_cfg = enrich_cfg_dict.get("hook", {})
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

                # Step 4: Enrich candidates (if highlights exist)
                proj_data = get_project_data(proj)
                has_candidates = bool(proj_data.get("analysis", {}).get("highlights", {}).get("candidates"))
                
                if enrich_cfg_dict.get("enabled", True) and has_candidates:
                    JOB_MANAGER._set(job, progress=0.8, message="Enriching candidates...")
                    
                    # Build enrich config (hook/quote text extraction only - no score fusion)
                    hook_cfg = enrich_cfg_dict.get("hook", {})
                    quote_cfg = enrich_cfg_dict.get("quote", {})
                    
                    enrich_config = EnrichConfig(
                        enabled=True,
                        hook_max_chars=int(hook_cfg.get("max_chars", 60)),
                        hook_window_seconds=float(hook_cfg.get("window_seconds", 4.0)),
                        quote_max_chars=int(quote_cfg.get("max_chars", 120)),
                        reaction_phrases=hook_cfg.get("phrases", []),
                    )
                    
                    def on_enrich_progress(frac: float) -> None:
                        JOB_MANAGER._set(job, progress=0.8 + 0.15 * frac, message="Enriching candidates...")
                    
                    result = enrich_candidates(proj, cfg=enrich_config, on_progress=on_enrich_progress)
                else:
                    result = {"message": "No candidates to enrich. Run highlights analysis first."}

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
                    JOB_MANAGER._set(job, progress=0.20, message="Detecting chat valleys/bursts...")
                    chat_boundary_cfg = ChatBoundaryConfig(
                        min_valley_gap_s=float(context_cfg.get("boundaries", {}).get("chat_valley_window_s", 5.0)),
                    )
                    compute_chat_boundaries_analysis(proj, cfg=chat_boundary_cfg)

                # Step 3.5: Semantic chapters (if enabled, requires transcript)
                chapters_cfg_dict = context_cfg.get("chapters", {})
                chapters_enabled = bool(chapters_cfg_dict.get("enabled", True))
                chapters_path = proj.chapters_path
                if chapters_enabled and proj.transcript_path.exists() and not chapters_path.exists():
                    JOB_MANAGER._set(job, progress=0.25, message="Detecting semantic chapters...")
                    try:
                        chapter_cfg = ChapterConfig(
                            min_chapter_len_s=float(chapters_cfg_dict.get("min_chapter_len_s", 60.0)),
                            max_chapter_len_s=float(chapters_cfg_dict.get("max_chapter_len_s", 900.0)),
                            embedding_model=str(chapters_cfg_dict.get("embedding_model", "all-mpnet-base-v2")),
                            changepoint_method=str(chapters_cfg_dict.get("changepoint_method", "pelt")),
                            changepoint_penalty=float(chapters_cfg_dict.get("changepoint_penalty", 10.0)),
                            snap_to_silence_window_s=float(chapters_cfg_dict.get("snap_to_silence_window_s", 10.0)),
                            llm_labeling=bool(chapters_cfg_dict.get("llm_labeling", True)),
                            llm_endpoint=str(ai_cfg.get("endpoint", "http://127.0.0.1:11435")),
                            llm_model_name=str(ai_cfg.get("model_name", "local-gguf-vulkan")),
                            llm_timeout_s=float(ai_cfg.get("timeout_s", 30.0)),
                            max_chars_per_chapter=int(chapters_cfg_dict.get("max_chars_per_chapter", 6000)),
                        )
                        
                        def on_chapters_progress(frac: float) -> None:
                            JOB_MANAGER._set(job, progress=0.25 + 0.10 * frac, message="Detecting semantic chapters...")
                        
                        compute_chapters_analysis(proj, cfg=chapter_cfg, on_progress=on_chapters_progress)
                    except Exception as e:
                        # Chapters are optional - continue if they fail
                        import logging
                        logging.getLogger(__name__).warning(f"Semantic chapters failed (non-fatal): {e}")

                # Step 4: Unified boundaries
                boundaries_path = proj.analysis_dir / "boundaries.json"
                JOB_MANAGER._set(job, progress=0.38, message="Building boundary graph...")
                boundary_prefs = context_cfg.get("boundaries", {})
                boundary_cfg = BoundaryConfig(
                    prefer_silence=bool(boundary_prefs.get("prefer_silence", True)),
                    prefer_sentences=bool(boundary_prefs.get("prefer_sentences", True)),
                    prefer_scene_cuts=bool(boundary_prefs.get("prefer_scene_cuts", True)),
                    prefer_chat_valleys=bool(boundary_prefs.get("prefer_chat_valleys", True)),
                    prefer_chapters=bool(boundary_prefs.get("prefer_chapters", True)),
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
        # Support both "enrich" and legacy "rerank" config keys
        enrich_cfg_dict = {
            **analysis_cfg.get("enrich", analysis_cfg.get("rerank", {})),
            **(body.get("enrich") or body.get("rerank") or {})
        }
        context_cfg = {**context_cfg, **(body.get("context") or {})}
        ai_cfg = {**ai_cfg, **(body.get("ai") or {})}

        # Normalize motion_weight_mode into actual numeric weight
        motion_mode = str(highlights_cfg.get("motion_weight_mode", "off")).lower()
        motion_weight = {
            "off": 0.0,
            "low": 0.15,
            "normal": 0.35,
            "high": 0.8,
        }.get(motion_mode, 0.0)
        highlights_cfg.setdefault("weights", {})["motion"] = motion_weight
        
        # Skip motion/scenes analysis entirely if motion weight is off
        include_motion = motion_mode != "off"

        # Options
        include_speech = bool(body.get("include_speech", True))
        include_context = bool(body.get("include_context", True))
        include_director = bool(body.get("include_director", True))

        job = JOB_MANAGER.create("analyze_full")

        @with_prevent_sleep("Full analysis running")
        def runner() -> None:
            import time as _time
            import threading
            import logging
            from ..analysis_motion import compute_motion_analysis
            from ..analysis_scenes import compute_scene_analysis
            
            log = logging.getLogger("videopipeline.studio")

            JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting parallel analysis DAG...")
            
            completed_stages = []
            errors = []
            stage_times: Dict[str, float] = {}  # stage_name -> elapsed seconds
            current_stage: Dict[str, Any] = {}  # For in-progress tracking
            
            # Task-level progress for parallel Stage 1 tasks
            # Key: task name, Value: {"progress": 0.0-1.0, "message": "..."}
            task_progress: Dict[str, Dict[str, Any]] = {}
            task_progress_lock = threading.Lock()
            
            # These will be populated before tasks run
            pending_tasks: set = set()
            task_start_times: Dict[str, float] = {}
            stage1_completed: list = []
            stage1_failed: list = []
            stage1_task_times: Dict[str, Any] = {}
            
            def update_task_progress(task_name: str, progress: float, message: str = "") -> None:
                """Update progress for a specific task (thread-safe)."""
                with task_progress_lock:
                    task_progress[task_name] = {"progress": progress, "message": message}
                # Trigger a job update so frontend sees the progress
                JOB_MANAGER._set(
                    job,
                    result=update_timing_result({
                        "stage": 1,
                        "pending": list(pending_tasks),
                        "completed": list(stage1_completed),
                        "failed": list(stage1_failed),
                        "task_times": dict(stage1_task_times),
                        "task_start_times": {k: v for k, v in task_start_times.items() if k in pending_tasks},
                        "task_progress": dict(task_progress),
                    })
                )
            
            def check_cancelled() -> bool:
                """Check if job was cancelled. Returns True if cancelled."""
                if job.cancel_requested:
                    current_stage.clear()
                    return True
                return False

            def update_timing_result(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                """Build result dict with timing info."""
                result = {
                    "stage_times": stage_times,
                    "current_stage": current_stage,
                    "task_progress": dict(task_progress),
                }
                if extra:
                    result.update(extra)
                return result

            # Stage 1: Run independent analyses in parallel
            # Each function returns a tuple: (stage_name, was_cached)
            def run_audio():
                if proj.audio_features_path.exists():
                    return ("audio", True)  # cached
                    
                def audio_progress(p: float) -> None:
                    if p < 0.5:
                        update_task_progress("audio", p, "analyzing audio...")
                    else:
                        pct = int((p - 0.5) * 200)
                        update_task_progress("audio", p, f"computing peaks {pct}%")
                        
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
                    on_progress=audio_progress,
                )
                return ("audio", False)  # computed

            def run_motion():
                if proj.motion_features_path.exists():
                    return ("motion", True)  # cached
                    
                def motion_progress(p: float) -> None:
                    pct = int(p * 100)
                    update_task_progress("motion", p, f"analyzing frames {pct}%")
                    
                compute_motion_analysis(
                    proj,
                    sample_fps=float(motion_cfg.get("sample_fps", 3.0)),
                    scale_width=int(motion_cfg.get("scale_width", 160)),
                    smooth_s=float(motion_cfg.get("smooth_seconds", 2.5)),
                    on_progress=motion_progress,
                )
                return ("motion", False)  # computed

            def run_audio_events():
                if not bool(audio_events_cfg.get("enabled", True)):
                    return ("audio_events", True)  # disabled = treat as cached/skipped
                if proj.audio_events_features_path.exists():
                    return ("audio_events", True)  # cached
                    
                def audio_events_progress(p: float) -> None:
                    pct = int(p * 100)
                    update_task_progress("audio_events", p, f"detecting events {pct}%")
                    
                cfg = AudioEventsConfig.from_dict(audio_events_cfg)
                compute_audio_events_analysis(proj, cfg=cfg, on_progress=audio_events_progress)
                return ("audio_events", False)  # computed

            def run_transcript():
                if not (include_speech and speech_cfg.get("enabled", True)):
                    return ("transcript", True)  # disabled
                if proj.transcript_path.exists():
                    return ("transcript", True)  # cached
                    
                # Progress callback for transcript - maps internal 0-1 to descriptive messages
                def transcript_progress(p: float) -> None:
                    if p < 0.1:
                        msg = "extracting audio..."
                    elif p < 0.2:
                        msg = "loading model..."
                    elif p < 0.9:
                        # Map 0.2-0.9 to percentage
                        pct = int((p - 0.2) / 0.7 * 100)
                        msg = f"transcribing {pct}%"
                    else:
                        msg = "saving..."
                    update_task_progress("transcript", p, msg)
                
                transcript_config = TranscriptConfig(
                    backend=str(speech_cfg.get("backend", "auto")),
                    model_size=str(speech_cfg.get("model_size", "small")),
                    language=speech_cfg.get("language"),
                    device=str(speech_cfg.get("device", "cpu")),
                    compute_type=str(speech_cfg.get("compute_type", "int8")),
                    vad_filter=bool(speech_cfg.get("vad_filter", True)),
                    word_timestamps=bool(speech_cfg.get("word_timestamps", True)),
                    use_gpu=bool(speech_cfg.get("use_gpu", False)),
                    threads=int(speech_cfg.get("threads", 0)),
                    n_processors=int(speech_cfg.get("n_processors", 1)),
                    strict=bool(speech_cfg.get("strict", False)),
                )
                compute_transcript_analysis(proj, cfg=transcript_config, on_progress=transcript_progress)
                return ("transcript", False)  # computed

            def run_reaction_audio():
                """Run reaction audio analysis (can run in parallel - no dependencies)."""
                if not (include_speech and reaction_cfg.get("enabled", True)):
                    return ("reaction_audio", True)  # disabled
                if proj.reaction_audio_features_path.exists():
                    return ("reaction_audio", True)  # cached
                    
                def reaction_progress(p: float) -> None:
                    pct = int(p * 100)
                    update_task_progress("reaction_audio", p, f"analyzing reactions {pct}%")
                    
                reaction_audio_config = ReactionAudioConfig(
                    sample_rate=int(reaction_cfg.get("sample_rate", 16000)),
                    hop_seconds=float(reaction_cfg.get("hop_seconds", 0.5)),
                    smooth_seconds=float(reaction_cfg.get("smooth_seconds", 1.5)),
                )
                compute_reaction_audio_features(proj, cfg=reaction_audio_config, on_progress=reaction_progress)
                return ("reaction_audio", False)  # computed

            def run_chat_download_retry():
                """Retry chat download if it previously failed."""
                from ..project import get_chat_config, get_source_url, set_chat_config
                
                chat_db_path = proj.analysis_dir / "chat.sqlite"
                
                # If chat.sqlite already exists, nothing to retry
                if chat_db_path.exists():
                    return ("chat_retry", True)  # skip - already have chat
                
                # Check the project's chat download status
                chat_cfg = get_chat_config(proj)
                download_status = chat_cfg.get("download_status")
                
                # Only retry if the previous download failed
                if download_status != "failed":
                    return ("chat_retry", True)  # skip - wasn't failed
                
                # Get the source URL for retry
                source_url = get_source_url(proj)
                if not source_url or "twitch.tv/videos/" not in source_url.lower():
                    return ("chat_retry", True)  # skip - not a Twitch VOD
                
                log.info(f"[chat] Previous download failed, retrying for: {source_url}")
                update_task_progress("chat", 0.0, "Retrying chat download...")
                
                try:
                    from ..chat.downloader import download_chat as dl_chat, import_chat_to_project, find_twitch_downloader_cli
                    
                    cli_path = find_twitch_downloader_cli()
                    if not cli_path:
                        log.warning("[chat] TwitchDownloaderCLI not found for retry")
                        return ("chat_retry", True)  # skip
                    
                    # Download to temp path
                    import tempfile
                    import re
                    match = re.search(r'twitch\.tv/videos/(\d+)', source_url)
                    if not match:
                        return ("chat_retry", True)
                    
                    video_id = match.group(1)
                    chat_temp_path = Path(tempfile.gettempdir()) / f"chat_retry_{video_id}.json"
                    
                    def on_chat_progress(frac: float, msg: str) -> None:
                        update_task_progress("chat", frac * 0.8, f"retry: {msg}")
                    
                    dl_chat(source_url, chat_temp_path, on_progress=on_chat_progress)
                    
                    # Import to project
                    if chat_temp_path.exists():
                        update_task_progress("chat", 0.85, "Importing chat...")
                        import_chat_to_project(proj, chat_temp_path)
                        
                        # Clean up temp
                        try:
                            chat_temp_path.unlink()
                        except Exception:
                            pass
                        
                        # Update status to success
                        set_chat_config(proj, download_status="success")
                        log.info("[chat] Retry successful!")
                        return ("chat_retry", False)  # computed (downloaded)
                    else:
                        log.warning("[chat] Retry produced no file")
                        return ("chat_retry", True)  # skip
                        
                except Exception as e:
                    log.warning(f"[chat] Retry failed: {e}")
                    # Update error message but keep status as failed
                    set_chat_config(proj, download_error=f"Retry failed: {e}")
                    return ("chat_retry", True)  # skip - retry also failed

            def run_chat_features():
                chat_db_path = proj.analysis_dir / "chat.sqlite"
                
                # If chat.sqlite doesn't exist, try to retry download if previous attempt failed
                if not chat_db_path.exists():
                    retry_result = run_chat_download_retry()
                    # Check again if retry succeeded
                    if not chat_db_path.exists():
                        return ("chat", "skipped")  # no chat data = skip (not cached)
                
                # Check if we should force re-learn (cached but only has seed tokens)
                force_relearn = False
                if proj.chat_features_path.exists():
                    from ..chat.store import ChatStore
                    try:
                        store = ChatStore(chat_db_path)
                        laugh_source = store.get_meta("laugh_tokens_source", "")
                        store.close()
                        # If we have features but they were only from seeds, and LLM is enabled, re-learn
                        ai_cfg_check = ctx.profile.get("ai", {}).get("director", {})
                        if laugh_source == "seed" and ai_cfg_check.get("enabled", True):
                            log.info("[chat] Cached features used seeds only, will try LLM this time")
                            force_relearn = True
                        elif laugh_source.startswith("llm"):
                            return ("chat", True)  # Already LLM-analyzed, skip
                        elif laugh_source:
                            return ("chat", True)  # Has some source, skip
                    except Exception:
                        pass
                    
                    if not force_relearn:
                        return ("chat", True)  # cached
                
                from ..chat.features import compute_and_save_chat_features
                hop_s = float(audio_cfg.get("hop_seconds", 0.5))
                smooth_s = float(highlights_cfg.get("chat_smooth_seconds", 3.0))
                
                # Get LLM for laugh emote learning (with auto-start)
                llm_complete_fn = get_llm_complete_fn(ai_cfg, proj.analysis_dir)
                
                # Get channel info from source URL for global emote persistence
                from ..project import get_source_url
                from ..chat.emote_db import get_channel_for_project
                source_url = get_source_url(proj)
                channel_info = get_channel_for_project(proj, source_url=source_url)
                channel_id = channel_info[0] if channel_info else None
                platform = channel_info[1] if channel_info else "twitch"
                
                compute_and_save_chat_features(
                    proj,
                    hop_s=hop_s,
                    smooth_s=smooth_s,
                    llm_complete=llm_complete_fn,
                    force_relearn_laugh=force_relearn,
                    channel_id=channel_id,
                    platform=platform,
                )
                return ("chat", False)  # computed

            # Build stage 1 task list with display names
            # Stage 1: Parallel input analysis (no dependencies)
            #   - audio, motion, audio_events, chat, transcript, reaction_audio
            # Stage 1.5 (after transcript): speech_features (depends on transcript)
            # Stage 2: scenes (depends on motion)
            # Stage 3: highlights (combines ALL signals including speech + reaction)
            # Stage 4: rerank, chapters, boundaries, clip_variants, director
            stage1_tasks = [
                (run_audio, "run_audio", "audio"),
                (run_audio_events, "run_audio_events", "audio_events"),
                (run_chat_features, "run_chat_features", "chat"),
            ]
            # Only include motion if motion weight is not "off"
            if include_motion:
                stage1_tasks.append((run_motion, "run_motion", "motion"))
            if include_speech:
                stage1_tasks.append((run_transcript, "run_transcript", "transcript"))
                # reaction_audio has no dependencies - can run in parallel
                stage1_tasks.append((run_reaction_audio, "run_reaction_audio", "reaction_audio"))

            total_tasks = len(stage1_tasks)
            task_names = [t[2] for t in stage1_tasks]
            pending_tasks.update(task_names)
            
            # Track start times for parallel tasks
            for name in task_names:
                task_start_times[name] = _time.time()
            
            current_stage = {"name": "stage1", "started_at": _time.time()}
            JOB_MANAGER._set(
                job, 
                progress=0.05, 
                message=f"Stage 1: Starting {total_tasks} parallel tasks: {', '.join(task_names)}",
                result=update_timing_result({
                    "stage": 1, 
                    "pending": list(pending_tasks), 
                    "completed": [], 
                    "failed": [],
                    "task_start_times": task_start_times,
                })
            )

            # Run Stage 1 in parallel using ThreadPoolExecutor
            stage1_cached = []  # Track which stages were cached

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(task[0]): (task[1], task[2]) for task in stage1_tasks}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    func_name, display_name = futures[future]
                    elapsed = _time.time() - task_start_times[display_name]
                    try:
                        stage_name, was_cached = future.result()
                        completed_stages.append(stage_name)
                        stage1_completed.append(display_name)
                        pending_tasks.discard(display_name)
                        
                        # Handle different cache statuses:
                        # - True: cached (have data from previous run)
                        # - "skipped": no data available (e.g., chat download failed)
                        # - False: computed fresh
                        if was_cached == "skipped":
                            stage1_cached.append(display_name)
                            stage1_task_times[display_name] = "skipped"
                            stage_times[display_name] = "skipped"
                            msg = f"Stage 1: ○ {display_name} (skipped) ({len(stage1_completed)}/{total_tasks})"
                        elif was_cached:
                            stage1_cached.append(display_name)
                            stage1_task_times[display_name] = "cached"
                            stage_times[display_name] = "cached"
                            msg = f"Stage 1: ✓ {display_name} (cached) ({len(stage1_completed)}/{total_tasks})"
                        else:
                            stage1_task_times[display_name] = elapsed
                            stage_times[display_name] = elapsed
                            msg = f"Stage 1: ✓ {display_name} ({len(stage1_completed)}/{total_tasks}) [{elapsed:.1f}s]"
                        
                        progress = 0.05 + 0.30 * ((i + 1) / total_tasks)
                        JOB_MANAGER._set(
                            job, 
                            progress=progress, 
                            message=msg,
                            result=update_timing_result({
                                "stage": 1, 
                                "pending": list(pending_tasks), 
                                "completed": stage1_completed,
                                "cached": stage1_cached,
                                "failed": stage1_failed,
                                "task_times": stage1_task_times,
                                "task_start_times": {k: v for k, v in task_start_times.items() if k in pending_tasks},
                            })
                        )
                    except Exception as e:
                        err_msg = f"{display_name}: {e}"
                        errors.append(err_msg)
                        stage1_failed.append(display_name)
                        stage1_task_times[display_name] = elapsed
                        stage_times[display_name] = elapsed
                        pending_tasks.discard(display_name)
                        progress = 0.05 + 0.30 * ((i + 1) / total_tasks)
                        JOB_MANAGER._set(
                            job,
                            progress=progress,
                            message=f"Stage 1: ✗ {display_name} failed ({len(stage1_completed)}/{total_tasks} ok)",
                            result=update_timing_result({
                                "stage": 1,
                                "pending": list(pending_tasks),
                                "completed": stage1_completed,
                                "cached": stage1_cached,
                                "failed": stage1_failed,
                                "task_times": stage1_task_times,
                                "task_start_times": {k: v for k, v in task_start_times.items() if k in pending_tasks},
                            })
                        )

            # Stage 1.5: Scenes analysis (depends on motion -> must run AFTER Stage 1)
            # Only run if motion analysis is enabled (scenes detection uses motion data)
            if check_cancelled():
                return
                
            if include_motion and bool(scenes_cfg.get("enabled", True)):
                if proj.scenes_path.exists():
                    completed_stages.append("scenes")
                    stage_times["scenes"] = "cached"
                else:
                    scenes_start = _time.time()
                    current_stage = {"name": "scenes", "started_at": scenes_start}
                    JOB_MANAGER._set(
                        job,
                        progress=0.36,
                        message="Stage 1.5: Detecting scene cuts...",
                        result=update_timing_result()
                    )
                    try:
                        # Safety net: if motion file somehow isn't there, compute it now
                        if not proj.motion_features_path.exists():
                            compute_motion_analysis(
                                proj,
                                sample_fps=float(motion_cfg.get("sample_fps", 3.0)),
                                scale_width=int(motion_cfg.get("scale_width", 160)),
                                smooth_s=float(motion_cfg.get("smooth_seconds", 2.5)),
                            )
                            if "motion" not in completed_stages:
                                completed_stages.append("motion")

                        compute_scene_analysis(
                            proj,
                            threshold_z=float(scenes_cfg.get("threshold_z", 3.5)),
                            min_scene_len_seconds=float(scenes_cfg.get("min_scene_len_seconds", 1.2)),
                            snap_window_seconds=float(scenes_cfg.get("snap_window_seconds", 1.0)),
                        )
                        completed_stages.append("scenes")
                        stage_times["scenes"] = _time.time() - scenes_start
                    except Exception as e:
                        errors.append(f"scenes: {e}")
                        stage_times["scenes"] = _time.time() - scenes_start

            # Stage 1.6: Speech features (depends on transcript -> must run AFTER Stage 1)
            # This runs before highlights so speech/lexical excitement can inform peak selection
            if check_cancelled():
                return
                
            if include_speech and speech_cfg.get("enabled", True):
                if proj.transcript_path.exists() and not proj.speech_features_path.exists():
                    speech_feat_start = _time.time()
                    current_stage = {"name": "speech_features", "started_at": speech_feat_start}
                    JOB_MANAGER._set(job, progress=0.37, message="Stage 1.6: Extracting speech features...", result=update_timing_result())
                    try:
                        hook_cfg = enrich_cfg_dict.get("hook", {})
                        phrases = hook_cfg.get("phrases", [])
                        speech_feature_config = SpeechFeatureConfig(
                            hop_seconds=float(speech_cfg.get("hop_seconds", 0.5)),
                            reaction_phrases=phrases if phrases else None,
                        )
                        compute_speech_features(proj, cfg=speech_feature_config)
                        completed_stages.append("speech_features")
                        stage_times["speech_features"] = _time.time() - speech_feat_start
                    except Exception as e:
                        errors.append(f"speech_features: {e}")
                        stage_times["speech_features"] = _time.time() - speech_feat_start
                elif proj.speech_features_path.exists():
                    completed_stages.append("speech_features")
                    stage_times["speech_features"] = "cached"

            # Stage 2: Combine signals into highlight scores (now includes speech + reaction!)
            if check_cancelled():
                return
                
            highlights_start = _time.time()
            current_stage = {"name": "highlights", "started_at": highlights_start}
            JOB_MANAGER._set(job, progress=0.40, message="Stage 2: Computing highlight scores (with speech+reaction)...", result=update_timing_result())
            
            # Set up LLM client for semantic scoring (optional enhancement)
            llm_complete_fn_highlights = None
            if ai_cfg.get("enabled", True) and highlights_cfg.get("llm_semantic_enabled", True):
                llm_complete_fn_highlights = get_llm_complete_fn(ai_cfg, proj.analysis_dir)
            
            try:
                def on_highlights_progress(p: float) -> None:
                    # Map 0-1 progress to stage 2 range (0.40-0.65)
                    scaled = 0.40 + p * 0.25
                    if p < 0.25:
                        msg = "Stage 2: Loading signals..."
                    elif p < 0.65:
                        msg = "Stage 2: Computing audio events (may take a while)..."
                    elif p < 0.85:
                        msg = "Stage 2: Fusing signals and picking candidates..."
                    elif p < 0.92:
                        msg = "Stage 2: LLM semantic scoring..."
                    else:
                        msg = "Stage 2: Finalizing highlights..."
                    JOB_MANAGER._set(job, progress=scaled, message=msg, result=update_timing_result())
                
                # Use DAG runner for highlights with proper boundary graph ordering
                dag_config = {
                    "audio": audio_cfg,
                    "motion": motion_cfg,
                    "scenes": scenes_cfg,
                    "highlights": highlights_cfg,
                    "audio_events": audio_events_cfg,
                    "include_chat": True,
                    "include_audio_events": bool(audio_events_cfg.get("enabled", True)),
                    "_llm_complete": llm_complete_fn_highlights,
                }
                
                def on_dag_progress(frac: float, msg: str) -> None:
                    on_highlights_progress(frac)
                
                run_analysis(
                    proj,
                    targets={"highlights_candidates", "boundary_graph"},
                    config=dag_config,
                    on_progress=on_dag_progress,
                    llm_complete=llm_complete_fn_highlights,
                    upgrade_mode=True,
                )
                completed_stages.append("highlights")
                stage_times["highlights"] = _time.time() - highlights_start
            except Exception as e:
                errors.append(f"highlights: {e}")
                stage_times["highlights"] = _time.time() - highlights_start

            # Stage 3: Enrichment (speech_features and reaction_audio now run in Stage 1/1.6)
            if check_cancelled():
                return
                
            if include_speech and speech_cfg.get("enabled", True):
                # Enrich - always recompute since it depends on current highlights
                proj_data = get_project_data(proj)
                has_candidates = bool(proj_data.get("analysis", {}).get("highlights", {}).get("candidates"))
                if enrich_cfg_dict.get("enabled", True) and has_candidates:
                    enrich_start = _time.time()
                    current_stage = {"name": "enrich", "started_at": enrich_start}
                    JOB_MANAGER._set(job, progress=0.65, message="Stage 3: Enriching candidates...", result=update_timing_result())
                    try:
                        hook_cfg = enrich_cfg_dict.get("hook", {})
                        quote_cfg = enrich_cfg_dict.get("quote", {})
                        enrich_config = EnrichConfig(
                            enabled=True,
                            hook_max_chars=int(hook_cfg.get("max_chars", 60)),
                            hook_window_seconds=float(hook_cfg.get("window_seconds", 4.0)),
                            quote_max_chars=int(quote_cfg.get("max_chars", 120)),
                            reaction_phrases=hook_cfg.get("phrases", []),
                        )
                        enrich_candidates(proj, cfg=enrich_config)
                        completed_stages.append("enrich")
                        stage_times["enrich"] = _time.time() - enrich_start
                    except Exception as e:
                        errors.append(f"enrich: {e}")
                        stage_times["enrich"] = _time.time() - enrich_start

            # Stage 4: Context + AI Director (optional)
            if check_cancelled():
                return
                
            if include_context:
                boundaries_start = _time.time()
                current_stage = {"name": "boundaries", "started_at": boundaries_start}
                JOB_MANAGER._set(job, progress=0.70, message="Stage 4: Computing context boundaries...", result=update_timing_result())
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
                            min_valley_gap_s=float(context_cfg.get("boundaries", {}).get("chat_valley_window_s", 5.0)),
                        )
                        compute_chat_boundaries_analysis(proj, cfg=chat_boundary_cfg)

                    # Semantic chapters (if enabled and transcript available)
                    chapters_cfg_dict = context_cfg.get("chapters", {})
                    chapters_enabled = bool(chapters_cfg_dict.get("enabled", True))
                    if chapters_enabled and proj.transcript_path.exists() and not proj.chapters_path.exists():
                        chapters_start = _time.time()
                        current_stage = {"name": "chapters", "started_at": chapters_start}
                        JOB_MANAGER._set(job, progress=0.72, message="Stage 4: Detecting semantic chapters...", result=update_timing_result())
                        try:
                            chapter_cfg = ChapterConfig(
                                min_chapter_len_s=float(chapters_cfg_dict.get("min_chapter_len_s", 60.0)),
                                max_chapter_len_s=float(chapters_cfg_dict.get("max_chapter_len_s", 900.0)),
                                embedding_model=str(chapters_cfg_dict.get("embedding_model", "all-mpnet-base-v2")),
                                changepoint_method=str(chapters_cfg_dict.get("changepoint_method", "pelt")),
                                changepoint_penalty=float(chapters_cfg_dict.get("changepoint_penalty", 10.0)),
                                snap_to_silence_window_s=float(chapters_cfg_dict.get("snap_to_silence_window_s", 10.0)),
                                llm_labeling=bool(chapters_cfg_dict.get("llm_labeling", True)),
                                llm_endpoint=str(ai_cfg.get("endpoint", "http://127.0.0.1:11435")),
                                llm_model_name=str(ai_cfg.get("model_name", "local-gguf-vulkan")),
                                llm_timeout_s=float(ai_cfg.get("timeout_s", 30.0)),
                                max_chars_per_chapter=int(chapters_cfg_dict.get("max_chars_per_chapter", 6000)),
                            )
                            compute_chapters_analysis(proj, cfg=chapter_cfg)
                            completed_stages.append("chapters")
                            stage_times["chapters"] = _time.time() - chapters_start
                        except Exception as e:
                            # Chapters are optional - continue if they fail
                            import logging
                            logging.getLogger(__name__).warning(f"Semantic chapters failed (non-fatal): {e}")
                            errors.append(f"chapters: {e}")
                            stage_times["chapters"] = _time.time() - chapters_start
                    elif proj.chapters_path.exists():
                        completed_stages.append("chapters")
                        stage_times["chapters"] = "cached"

                    # Unified boundaries
                    boundary_prefs = context_cfg.get("boundaries", {})
                    boundary_cfg = BoundaryConfig(
                        prefer_silence=bool(boundary_prefs.get("prefer_silence", True)),
                        prefer_sentences=bool(boundary_prefs.get("prefer_sentences", True)),
                        prefer_scene_cuts=bool(boundary_prefs.get("prefer_scene_cuts", True)),
                        prefer_chat_valleys=bool(boundary_prefs.get("prefer_chat_valleys", True)),
                        prefer_chapters=bool(boundary_prefs.get("prefer_chapters", True)),
                    )
                    compute_boundaries_analysis(proj, cfg=boundary_cfg)
                    completed_stages.append("boundaries")
                    stage_times["boundaries"] = _time.time() - boundaries_start

                    # Clip variants
                    variants_start = _time.time()
                    current_stage = {"name": "clip_variants", "started_at": variants_start}
                    JOB_MANAGER._set(job, progress=0.80, message="Stage 4: Generating clip variants...", result=update_timing_result())
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
                    stage_times["clip_variants"] = _time.time() - variants_start

                except Exception as e:
                    errors.append(f"context: {e}")
                    # Record partial times if we fail partway
                    if "boundaries" not in stage_times:
                        stage_times["boundaries"] = _time.time() - boundaries_start

            # AI Director (if enabled)
            if check_cancelled():
                return
                
            if include_director and ai_cfg.get("enabled", True):
                director_start = _time.time()
                current_stage = {"name": "director", "started_at": director_start}
                JOB_MANAGER._set(job, progress=0.90, message="Stage 4: Running AI Director...", result=update_timing_result())
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
                    stage_times["director"] = _time.time() - director_start
                except Exception as e:
                    errors.append(f"director: {e}")
                    stage_times["director"] = _time.time() - director_start

            # Final result
            current_stage = {}  # Clear current stage since we're done
            final_proj_data = get_project_data(proj)
            result = {
                "completed_stages": completed_stages,
                "errors": errors,
                "candidates_count": len(final_proj_data.get("analysis", {}).get("highlights", {}).get("candidates", [])),
                "signals_used": final_proj_data.get("analysis", {}).get("highlights", {}).get("signals_used", {}),
                "stage_times": stage_times,
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
            - ai_status: AI analysis status (laugh tokens, etc.)
        """
        proj = ctx.require_project()
        from ..project import get_chat_config, get_source_url

        chat_config = get_chat_config(proj)
        source_url = get_source_url(proj) or chat_config.get("source_url", "")

        chat_available = proj.chat_db_path.exists()
        message_count = 0
        duration_ms = 0
        ai_status = {"has_chat": False}
        
        # Check director status
        director_path = proj.analysis_dir / "ai_director.json"
        director_status = {"analyzed": False}
        if director_path.exists():
            try:
                director_data = json.loads(director_path.read_text(encoding="utf-8"))
                director_status = {
                    "analyzed": True,
                    "candidates_count": len(director_data.get("results", [])),
                    "model": director_data.get("config", {}).get("model_name", "unknown"),
                    "analyzed_at": director_data.get("created_at", ""),
                    "llm_available": director_data.get("llm_available", False),
                }
            except Exception:
                pass

        if chat_available:
            try:
                from ..chat.store import ChatStore
                store = ChatStore(proj.chat_db_path)
                meta = store.get_all_meta()
                message_count = meta.message_count
                duration_ms = meta.duration_ms
                
                # Get AI analysis status
                laugh_tokens_json = store.get_meta("laugh_tokens_json", "[]")
                laugh_tokens = json.loads(laugh_tokens_json) if laugh_tokens_json else []
                laugh_source = store.get_meta("laugh_tokens_source", "")
                laugh_updated = store.get_meta("laugh_tokens_updated_at", "")
                llm_learned_json = store.get_meta("laugh_tokens_llm_learned", "[]")
                llm_learned = json.loads(llm_learned_json) if llm_learned_json else []
                
                # Get additional stats from project.json if available
                proj_data = get_project_data(proj)
                chat_analysis = proj_data.get("analysis", {}).get("chat", {})
                newly_learned_count = chat_analysis.get("newly_learned_count", 0)
                newly_learned_tokens = chat_analysis.get("newly_learned_tokens", [])
                loaded_from_global = chat_analysis.get("loaded_from_global", 0)
                
                ai_status = {
                    "has_chat": message_count > 0,
                    "laugh_tokens": laugh_tokens,
                    "laugh_tokens_count": len(laugh_tokens),
                    "llm_learned_tokens": llm_learned,
                    "llm_learned_count": len(llm_learned),
                    "newly_learned_count": newly_learned_count,
                    "newly_learned_tokens": newly_learned_tokens,
                    "loaded_from_global": loaded_from_global,
                    "laugh_source": laugh_source,
                    "laugh_updated_at": laugh_updated,
                    "ai_analyzed": laugh_source.startswith("llm") if laugh_source else False,
                    # Add reason why LLM wasn't used
                    "llm_skip_reason": "" if laugh_source.startswith("llm") else (
                        "Not yet analyzed" if not laugh_source else
                        "LLM not available during analysis" if laugh_source == "seed" else
                        f"Source: {laugh_source}"
                    ),
                }
                
                store.close()
            except Exception as e:
                ai_status = {"has_chat": False, "error": str(e)}

        return JSONResponse({
            "available": chat_available,
            "enabled": chat_config.get("enabled", False),
            "sync_offset_ms": chat_config.get("sync_offset_ms", 0),
            "message_count": message_count,
            "duration_ms": duration_ms,
            "source_url": source_url,
            "ai_status": ai_status,
            "director_status": director_status,
        })

    @app.post("/api/chat/relearn_ai")
    def api_chat_relearn_ai():
        """Re-learn chat emotes with AI (LLM).
        
        This clears the cached laugh tokens and re-analyzes the chat
        using the LLM to learn channel-specific emotes.
        """
        proj = ctx.require_project()
        
        if not proj.chat_db_path.exists():
            raise HTTPException(status_code=400, detail="No chat data available")
        
        # Clear the chat features to force re-analysis
        if proj.chat_features_path.exists():
            proj.chat_features_path.unlink()
        
        # Clear cached laugh tokens from the store
        from ..chat.store import ChatStore
        store = ChatStore(proj.chat_db_path)
        store.set_meta("laugh_tokens_json", "[]")
        store.set_meta("laugh_tokens_source", "")
        store.set_meta("laugh_tokens_llm_learned", "[]")
        store.close()
        
        return JSONResponse({"ok": True, "message": "Cleared chat AI cache. Run Analyze (Full) to re-learn with AI."})

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
            
            def check_cancel() -> bool:
                return job.cancel_requested

            try:
                from ..chat.downloader import download_chat, ChatDownloadError, ChatDownloadCancelled
                from ..chat.normalize import load_and_normalize
                from ..chat.store import ChatStore, ChatMeta
                from ..chat.features import compute_and_save_chat_features

                # Download chat
                JOB_MANAGER._set(job, progress=0.05, message="Downloading chat replay...")
                result = download_chat(source_url, proj.chat_raw_path, on_progress=on_progress, check_cancel=check_cancel)
                
                # Check if cancelled after download
                if job.cancel_requested:
                    JOB_MANAGER._set(job, status="cancelled", message="Download cancelled")
                    return
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

                # Compute features (with optional LLM-based laugh emote learning)
                JOB_MANAGER._set(job, progress=0.7, message="Computing chat features...")
                hop_s = float(ctx.profile.get("analysis", {}).get("audio", {}).get("hop_seconds", 0.5))
                smooth_s = float(ctx.profile.get("analysis", {}).get("highlights", {}).get("chat_smooth_seconds", 3.0))
                
                # Get LLM config for laugh emote learning (with auto-start)
                ai_cfg = ctx.profile.get("ai", {}).get("director", {})
                
                def on_server_status(msg: str) -> None:
                    JOB_MANAGER._set(job, message=msg)
                
                llm_complete_fn = get_llm_complete_fn(ai_cfg, proj.analysis_dir, on_status=on_server_status)

                def on_feature_progress(frac: float) -> None:
                    JOB_MANAGER._set(job, progress=0.7 + 0.25 * frac)
                
                def on_feature_status(msg: str) -> None:
                    JOB_MANAGER._set(job, message=msg)

                # Extract channel info from source URL for global emote persistence
                from ..chat.emote_db import get_channel_for_project
                channel_info = get_channel_for_project(proj, source_url=source_url)
                channel_id = channel_info[0] if channel_info else None
                platform = channel_info[1] if channel_info else "twitch"

                compute_and_save_chat_features(
                    proj,
                    hop_s=hop_s,
                    smooth_s=smooth_s,
                    on_progress=on_feature_progress,
                    on_status=on_feature_status,
                    llm_complete=llm_complete_fn,
                    channel_id=channel_id,
                    platform=platform,
                )

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

            except ChatDownloadCancelled:
                JOB_MANAGER._set(job, status="cancelled", message="Chat download cancelled")
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

    @app.post("/api/jobs/{job_id}/cancel")
    def api_job_cancel(job_id: str):
        """Cancel a running or queued job."""
        success = JOB_MANAGER.cancel(job_id)
        if not success:
            job = JOB_MANAGER.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="job_not_found")
            raise HTTPException(status_code=400, detail="job_not_cancellable")
        return JSONResponse({"cancelled": True})

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
