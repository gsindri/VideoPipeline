from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional

import numpy as np

from .ffmpeg import ffprobe_duration_seconds


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint_file(
    path: Path,
    *,
    head_bytes: int = 2 * 1024 * 1024,
    tail_bytes: int = 2 * 1024 * 1024,
) -> str:
    """Compute a fast, content-informed fingerprint for large media.

    Reads:
      - file size
      - first `head_bytes`
      - last `tail_bytes` (if present)

    This avoids hashing the entire file while remaining stable across renames.
    """
    path = Path(path)
    st = path.stat()
    size = st.st_size

    h = hashlib.sha256()
    h.update(str(size).encode("utf-8"))

    with path.open("rb") as f:
        head = f.read(head_bytes)
        h.update(head)

        if size > tail_bytes:
            try:
                f.seek(max(0, size - tail_bytes))
                tail = f.read(tail_bytes)
                h.update(tail)
            except OSError:
                # Some file-like sources may not seek well; ignore tail.
                pass

    return h.hexdigest()


def get_outputs_dir() -> Path:
    """Get the outputs directory (parent of projects/)."""
    return Path("outputs")


def default_projects_root() -> Path:
    return get_outputs_dir() / "projects"


def project_dir_for_video(video_path: Path, projects_root: Optional[Path] = None) -> Path:
    projects_root = projects_root or default_projects_root()
    pid = fingerprint_file(video_path)
    return projects_root / pid


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    """Save JSON to file atomically (temp file + replace).

    Notes on Windows:
    - os.replace() fails with WinError 5/32 if another thread/process is replacing
      the same target at the same time.
    - We pair this with a higher-level lock in update_project(), and add a small
      retry loop here to survive transient file locks (e.g., antivirus).
    """
    import tempfile

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, indent=2, ensure_ascii=False)

    # Write to temp file in same directory (ensures same filesystem for atomic replace)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=path.stem + "_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)

        # os.replace() is atomic, but can transiently fail on Windows if the
        # destination is being swapped by another thread/process.
        for attempt in range(10):
            try:
                os.replace(tmp_path, str(path))
                break
            except PermissionError:
                if os.name == "nt" and attempt < 9:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise
    finally:
        # If replace succeeded, tmp_path no longer exists; if it failed, clean up.
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


@dataclass
class Project:
    project_dir: Path
    video_path: Path

    @property
    def video_dir(self) -> Path:
        """Directory for project-local video files (source + optional preview)."""
        return self.project_dir / "video"

    @property
    def preview_video_path(self) -> Path:
        """Path to a browser-friendly preview video inside the project, if present."""
        return self.video_dir / "preview.mp4"

    @property
    def project_json_path(self) -> Path:
        return self.project_dir / "project.json"

    @property
    def analysis_dir(self) -> Path:
        return self.project_dir / "analysis"

    @property
    def audio_features_path(self) -> Path:
        return self.analysis_dir / "audio_features.npz"

    @property
    def audio_vad_path(self) -> Path:
        """Path to the computed voice activity detection (VAD) features."""
        return self.analysis_dir / "audio_vad.npz"

    @property
    def motion_features_path(self) -> Path:
        return self.analysis_dir / "motion_features.npz"

    @property
    def chat_features_path(self) -> Path:
        return self.analysis_dir / "chat_features.npz"

    @property
    def highlights_features_path(self) -> Path:
        return self.analysis_dir / "highlights_features.npz"

    @property
    def scenes_path(self) -> Path:
        return self.analysis_dir / "scenes.json"

    @property
    def chapters_path(self) -> Path:
        return self.analysis_dir / "chapters.json"

    @property
    def chat_raw_path(self) -> Path:
        return self.analysis_dir / "chat_raw.json"

    @property
    def chat_db_path(self) -> Path:
        return self.analysis_dir / "chat.sqlite"

    @property
    def transcript_path(self) -> Path:
        return self.analysis_dir / "transcript_full.json"

    @property
    def speech_features_path(self) -> Path:
        return self.analysis_dir / "speech_features.npz"

    @property
    def reaction_audio_features_path(self) -> Path:
        return self.analysis_dir / "reaction_audio_features.npz"

    @property
    def audio_events_features_path(self) -> Path:
        return self.analysis_dir / "audio_events_features.npz"

    @property
    def exports_dir(self) -> Path:
        return self.project_dir / "exports"

    @property
    def inputs_dir(self) -> Path:
        """Directory for raw input files (audio, chat) copied into the project."""
        return self.analysis_dir / "inputs"

    @property
    def audio_raw_path(self) -> Path:
        """Path to store downloaded audio file within the project.
        
        When audio is downloaded before video (e.g., during URL download),
        it's copied here so all analysis can use it without external references.
        
        Note: This returns the expected path for a new audio file.
        Use find_audio_raw() to find an existing audio file with any extension.
        """
        return self.inputs_dir / "audio.m4a"

    def find_audio_raw(self) -> Optional[Path]:
        """Find downloaded audio file in the inputs directory.
        
        Searches for audio files with common extensions (m4a, mp3, opus, wav, webm, ogg, mp4).
        
        Returns:
            Path to audio file if found, None otherwise
        """
        if not self.inputs_dir.exists():
            return None
        
        # Include mp4 because yt-dlp sometimes downloads audio as .mp4 container
        audio_extensions = {".m4a", ".mp3", ".opus", ".wav", ".webm", ".ogg", ".aac", ".flac", ".mp4"}
        for f in self.inputs_dir.iterdir():
            if f.is_file() and f.name.startswith("audio") and f.suffix.lower() in audio_extensions:
                return f
        return None

    @property
    def audio_source(self) -> Path:
        """Get the path to use for audio extraction.
        
        Priority order:
        1. Audio file in inputs_dir - Audio file stored in project (best: self-contained)
        2. video_path - Extract audio from video (normal case)
        3. early_audio_path from project.json - Legacy temp file reference
        4. video_path as fallback
        
        This allows early analysis during download when only audio is available.
        """
        # 1. Prefer audio file stored in project (self-contained)
        audio_raw = self.find_audio_raw()
        if audio_raw and audio_raw.exists():
            return audio_raw
        
        # 2. If video exists, use it (normal case)
        if self.video_path.exists():
            return self.video_path
        
        # 3. Check for early audio path in project.json (legacy/temp reference)
        if self.project_json_path.exists():
            try:
                data = load_json(self.project_json_path)
                early_audio = data.get("video", {}).get("early_audio_path")
                if early_audio:
                    early_path = Path(early_audio)
                    if early_path.exists():
                        return early_path
            except Exception:
                pass
        
        # 4. Fall back to video_path (may not exist, but callers will handle)
        return self.video_path


def find_project_for_video(video_path: Path, projects_root: Optional[Path] = None) -> Optional[Path]:
    """Find an existing project directory that contains a given video file.
    
    This handles the case where a project was created via URL download (using content_id hash)
    and we later want to open it by video path.
    
    Checks:
    1. If video_path is already inside a project's video/ folder -> return that project
    2. Scan all projects for matching video path in project.json
    
    Args:
        video_path: Path to the video file
        projects_root: Optional custom projects directory
        
    Returns:
        Path to the project directory if found, None otherwise
    """
    video_path = Path(video_path).expanduser().resolve()
    projects_root = projects_root or default_projects_root()
    
    if not projects_root.exists():
        return None
    
    # Check if video_path is already inside a project's video/ folder
    # e.g., outputs/projects/<hash>/video/video.mp4
    try:
        for parent in video_path.parents:
            if parent.parent == projects_root:
                # This is a project directory
                if (parent / "project.json").exists():
                    return parent
    except Exception:
        pass
    
    # Scan all projects to find one that references this video
    for proj_dir in projects_root.iterdir():
        if not proj_dir.is_dir():
            continue
        
        project_json = proj_dir / "project.json"
        if not project_json.exists():
            continue
        
        try:
            data = load_json(project_json)
            stored_path = data.get("video", {}).get("path")
            if stored_path and Path(stored_path).resolve() == video_path:
                return proj_dir
        except Exception:
            continue
    
    return None


def create_or_load_project(video_path: Path, projects_root: Optional[Path] = None) -> Project:
    video_path = Path(video_path).expanduser().resolve()
    
    # First, check if there's already a project for this video
    # (handles projects created via URL download with different hash)
    existing_pdir = find_project_for_video(video_path, projects_root)
    if existing_pdir:
        return Project(project_dir=existing_pdir, video_path=video_path)
    
    # No existing project found, create new one based on file fingerprint
    pdir = project_dir_for_video(video_path, projects_root)
    pdir.mkdir(parents=True, exist_ok=True)

    proj = Project(project_dir=pdir, video_path=video_path)

    if not proj.project_json_path.exists():
        duration_s = ffprobe_duration_seconds(video_path)
        st = video_path.stat()
        initial = {
            "project_id": pdir.name,
            "created_at": utc_now_iso(),
            "video": {
                "path": str(video_path),
                "size_bytes": st.st_size,
                "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
                "duration_seconds": duration_s,
            },
            "analysis": {},
            "layout": {},
            "selections": [],
            "exports": [],
        }
        save_json(proj.project_json_path, initial)

    return proj


def create_project_early(
    content_id: str,
    *,
    source_url: Optional[str] = None,
    audio_path: Optional[Path] = None,
    duration_seconds: Optional[float] = None,
    projects_root: Optional[Path] = None,
) -> Project:
    """Create a project early, before video download completes.
    
    This allows running audio-based analysis (transcript, silence, etc.)
    while the video is still downloading.
    
    Args:
        content_id: Unique identifier for the content (e.g., video ID from URL)
        source_url: Original URL being downloaded
        audio_path: Path to downloaded audio file (optional, for duration detection)
        duration_seconds: Known duration (optional, from metadata)
        projects_root: Custom projects directory
        
    Returns:
        Project with placeholder video path
    """
    import hashlib
    import shutil
    
    projects_root = projects_root or default_projects_root()
    
    # Create project ID from content_id hash
    pid = hashlib.sha256(content_id.encode()).hexdigest()
    pdir = projects_root / pid
    pdir.mkdir(parents=True, exist_ok=True)
    
    # Check if this is a re-download (project exists but video doesn't)
    # If so, clear stale analysis outputs so tasks re-run
    video_dir = pdir / "video"
    video_file = video_dir / "video.mp4"
    analysis_dir = pdir / "analysis"
    if analysis_dir.exists() and not video_file.exists():
        # Stale analysis from previous incomplete download - clear it
        import logging
        logging.getLogger(__name__).info(f"Clearing stale analysis for re-download: {pdir.name[:12]}...")
        shutil.rmtree(analysis_dir, ignore_errors=True)
    
    # Placeholder video path (will be updated when video downloads)
    placeholder_video = pdir / "video_pending"
    
    proj = Project(project_dir=pdir, video_path=placeholder_video)
    
    if not proj.project_json_path.exists():
        # Try to get duration from audio if available
        if duration_seconds is None and audio_path and audio_path.exists():
            try:
                duration_seconds = ffprobe_duration_seconds(audio_path)
            except Exception:
                pass
        
        initial = {
            "project_id": pdir.name,
            "created_at": utc_now_iso(),
            "source_url": source_url,
            "video": {
                "path": None,  # Not yet known
                "preview_path": None,
                "status": "downloading",
                "duration_seconds": duration_seconds,
                "early_audio_path": str(audio_path) if audio_path else None,
            },
            "analysis": {},
            "layout": {},
            "selections": [],
            "exports": [],
        }
        save_json(proj.project_json_path, initial)
    
    return proj


def set_project_video(proj: Project, video_path: Path, *, preview_path: Optional[Path] = None) -> None:
    """Update a project with the final video path after download completes.
    
    This copies (or symlinks on Unix) the video into the project's video/ directory
    for self-contained storage.
    
    Args:
        proj: Project to update
        video_path: Path to downloaded video file
        preview_path: Optional browser-friendly preview/proxy video
    """
    import shutil
    import os
    
    video_path = Path(video_path).expanduser().resolve()
    
    # Create video directory in project
    video_dir = proj.video_dir
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Destination path - always use "video.mp4" for consistency
    dest_path = video_dir / "video.mp4"
    
    # Copy the video to the project directory if not already there
    if not dest_path.exists() or not dest_path.samefile(video_path):
        # On Windows, just copy; on Unix we could symlink
        if os.name == "nt":
            shutil.copy2(video_path, dest_path)
        else:
            # Try symlink first, fall back to copy
            try:
                if dest_path.exists():
                    dest_path.unlink()
                dest_path.symlink_to(video_path)
            except OSError:
                shutil.copy2(video_path, dest_path)

    preview_dest: Optional[Path] = None
    if preview_path is not None:
        try:
            pp = Path(preview_path).expanduser().resolve()
            if pp.exists():
                preview_dest = video_dir / "preview.mp4"
                # Copy the preview to the project directory if not already there
                same = False
                try:
                    if preview_dest.exists():
                        same = preview_dest.samefile(pp)
                except Exception:
                    same = False
                if not preview_dest.exists() or not same:
                    if os.name == "nt":
                        shutil.copy2(pp, preview_dest)
                    else:
                        try:
                            if preview_dest.exists():
                                preview_dest.unlink()
                            preview_dest.symlink_to(pp)
                        except OSError:
                            shutil.copy2(pp, preview_dest)
        except Exception:
            preview_dest = None
    
    # Update the project's video_path to point to the project-local copy
    proj.video_path = dest_path
    
    # Get video info
    try:
        duration_s = ffprobe_duration_seconds(dest_path)
    except Exception:
        duration_s = None
    
    st = dest_path.stat()
    
    def _upd(d: Dict[str, Any]) -> None:
        d["video"] = {
            "path": str(dest_path),
            "preview_path": str(preview_dest) if preview_dest and preview_dest.exists() else None,
            "status": "ready",
            "size_bytes": st.st_size,
            "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
            "duration_seconds": duration_s or d.get("video", {}).get("duration_seconds"),
        }
    
    update_project(proj, _upd)


@contextmanager
def _file_lock(lock_file: Path, *, timeout_s: float = 10.0, poll_s: float = 0.02, stale_s: float = 5.0) -> Generator[None, None, None]:
    """Very small file-based lock.

    We use this for project.json updates because many analysis tasks run in parallel
    and update_project() does a read-modify-write cycle. On Windows, concurrent
    os.replace() calls can raise PermissionError (WinError 5).
    
    The lock times out quickly (10s default) and considers locks stale after 5s,
    because save_json/load_json should be very fast operations.
    """
    lock_file = Path(lock_file).expanduser().resolve()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    acquired = False
    my_pid = os.getpid()

    while time.time() - start < timeout_s:
        try:
            # Create lock file exclusively.
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{time.time()}:{my_pid}\n".encode("utf-8"))
            finally:
                os.close(fd)
            acquired = True
            break
        except (FileExistsError, OSError) as e:
            # OSError errno 17 is EEXIST on some platforms
            if isinstance(e, OSError) and e.errno not in (17, None):
                # Some other OS error, wait and retry
                time.sleep(poll_s)
                continue
            
            # Lock file exists - check if it's stale
            try:
                content = lock_file.read_text(encoding="utf-8").strip()
                lock_time = float(content.split(":")[0])
                if time.time() - lock_time > stale_s:
                    # Stale lock - forcibly remove it
                    try:
                        lock_file.unlink()
                    except OSError:
                        pass
                    continue
            except Exception:
                # Can't read lock file (corrupt, being written, etc.) - remove it
                try:
                    lock_file.unlink()
                except OSError:
                    pass
                continue

            time.sleep(poll_s)

    if not acquired:
        # Last-ditch attempt: force remove any lock and don't use locking
        # This is better than failing the entire task
        try:
            lock_file.unlink()
        except OSError:
            pass
        # Log warning but continue without lock
        import logging
        logging.getLogger("videopipeline.project").warning(
            f"Lock timeout, proceeding without lock: {lock_file}"
        )
        yield
        return

    try:
        yield
    finally:
        try:
            lock_file.unlink()
        except OSError:
            pass


def update_project(proj: Project, updater) -> Dict[str, Any]:
    """Safely update project.json (read-modify-write) with a lock.

    This prevents concurrent task threads from clobbering each other's writes and
    avoids WinError 5 on Windows when multiple os.replace() calls happen at once.
    """
    lock_file = proj.project_dir / ".locks" / "project_json.lock"
    with _file_lock(lock_file):  # Use defaults: timeout_s=10, stale_s=5
        data = load_json(proj.project_json_path)
        updater(data)
        save_json(proj.project_json_path, data)
        return data


# Backwards-compatible alias (some modules import this name)
def update_project_json(proj: Project, updater) -> Dict[str, Any]:
    return update_project(proj, updater)


def get_project_data(proj: Project) -> Dict[str, Any]:
    return load_json(proj.project_json_path)


def add_selection(
    proj: Project,
    *,
    start_s: float,
    end_s: float,
    title: str = "",
    notes: str = "",
    template: str = "vertical_blur",
) -> Dict[str, Any]:
    import uuid

    sel_id = uuid.uuid4().hex

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("selections", [])
        d["selections"].append(
            {
                "id": sel_id,
                "created_at": utc_now_iso(),
                "start_s": float(start_s),
                "end_s": float(end_s),
                "title": title,
                "notes": notes,
                "template": template,
            }
        )

    return update_project(proj, _upd)


def add_selection_from_candidate(
    proj: Project,
    *,
    candidate: Dict[str, Any],
    template: str,
    title: str = "",
) -> str:
    import uuid

    sel_id = uuid.uuid4().hex

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("selections", [])
        d["selections"].append(
            {
                "id": sel_id,
                "created_at": utc_now_iso(),
                "start_s": float(candidate.get("start_s")),
                "end_s": float(candidate.get("end_s")),
                "title": title,
                "notes": "",
                "template": template,
                "candidate_rank": candidate.get("rank"),
                "candidate_score": candidate.get("score"),
                "candidate_peak_time_s": candidate.get("peak_time_s"),
            }
        )

    update_project(proj, _upd)
    return sel_id


def set_layout_facecam(proj: Project, *, rect: Dict[str, float]) -> Dict[str, Any]:
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("layout", {})
        d["layout"]["facecam"] = rect

    return update_project(proj, _upd)


def update_selection(proj: Project, selection_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    def _upd(d: Dict[str, Any]) -> None:
        sels = d.get("selections", [])
        for s in sels:
            if s.get("id") == selection_id:
                s.update(patch)
                s["updated_at"] = utc_now_iso()
                return
        raise KeyError(f"Selection not found: {selection_id}")

    return update_project(proj, _upd)


def remove_selection(proj: Project, selection_id: str) -> Dict[str, Any]:
    def _upd(d: Dict[str, Any]) -> None:
        before = d.get("selections", [])
        after = [s for s in before if s.get("id") != selection_id]
        d["selections"] = after

    return update_project(proj, _upd)


def record_export(
    proj: Project,
    *,
    selection_id: str,
    output_path: Path,
    template: str,
    with_captions: bool,
    status: str,
) -> Dict[str, Any]:
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("exports", [])
        d["exports"].append(
            {
                "created_at": utc_now_iso(),
                "selection_id": selection_id,
                "output": str(output_path),
                "template": template,
                "with_captions": bool(with_captions),
                "status": status,
            }
        )

    return update_project(proj, _upd)


# ---------------------------------------------------------------------------
# Source URL and Chat Config Management
# ---------------------------------------------------------------------------


def get_source_url(proj: Project) -> Optional[str]:
    """Get the source URL for the project (if set)."""
    data = get_project_data(proj)
    source = data.get("source", {})
    return source.get("source_url")


def set_source_url(proj: Project, source_url: str, *, platform: Optional[str] = None) -> Dict[str, Any]:
    """Set the source URL for a project (for chat download, etc.).

    Args:
        proj: Project instance
        source_url: URL of the original video (Twitch VOD, YouTube, etc.)
        platform: Optional platform hint (twitch, youtube, etc.)
    """
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("source", {})
        d["source"]["source_url"] = source_url
        if platform:
            d["source"]["platform"] = platform

    return update_project(proj, _upd)


def get_chat_config(proj: Project) -> Dict[str, Any]:
    """Get chat configuration for the project."""
    data = get_project_data(proj)
    return data.get("chat", {"enabled": False, "sync_offset_ms": 0})


def set_chat_config(
    proj: Project,
    *,
    enabled: Optional[bool] = None,
    sync_offset_ms: Optional[int] = None,
    sync_offset_source: Optional[str] = None,
    sync_offset_confidence: Optional[float] = None,
    sync_offset_method: Optional[str] = None,
    sync_offset_updated_at: Optional[str] = None,
    source_url: Optional[str] = None,
    download_status: Optional[str] = None,
    download_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Update chat configuration.

    Args:
        proj: Project instance
        enabled: Whether chat is enabled
        sync_offset_ms: Chat sync offset in milliseconds
        sync_offset_source: "auto" or "manual" (optional)
        sync_offset_confidence: Confidence score for auto sync (optional)
        sync_offset_method: Method identifier for auto sync (optional)
        sync_offset_updated_at: ISO timestamp of last offset update (optional)
        source_url: Source URL used for chat download
        download_status: Status of chat download: 'success', 'failed', 'skipped', or None
        download_error: Error message if download failed
    """
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("chat", {"enabled": False, "sync_offset_ms": 0})
        if enabled is not None:
            d["chat"]["enabled"] = bool(enabled)
        if sync_offset_ms is not None:
            d["chat"]["sync_offset_ms"] = int(sync_offset_ms)
        if sync_offset_source is not None:
            d["chat"]["sync_offset_source"] = str(sync_offset_source)
        if sync_offset_confidence is not None:
            d["chat"]["sync_offset_confidence"] = float(sync_offset_confidence)
        if sync_offset_method is not None:
            d["chat"]["sync_offset_method"] = str(sync_offset_method)
        if sync_offset_updated_at is not None:
            d["chat"]["sync_offset_updated_at"] = str(sync_offset_updated_at)
        if source_url is not None:
            d["chat"]["source_url"] = source_url
        if download_status is not None:
            d["chat"]["download_status"] = download_status
        if download_error is not None:
            d["chat"]["download_error"] = download_error
        elif download_status == "success":
            # Clear error on success
            d["chat"].pop("download_error", None)

    return update_project(proj, _upd)
