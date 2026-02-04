"""Publisher API router for Studio.

Provides endpoints for:
- Listing accounts
- Listing exports from active project
- Queuing publish jobs
- Job management (list, retry, cancel)
- SSE job progress streaming
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ..publisher.accounts import AccountStore
from ..publisher.jobs import PublishJobStore
from ..utils import utc_iso as _utc_iso


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class ExportInfo:
    """Represents an export file with its metadata."""

    def __init__(
        self,
        *,
        export_id: str,
        mp4_path: Path,
        metadata_path: Optional[Path],
        metadata: Dict[str, Any],
        duration_seconds: float,
        created_at: str,
        file_size_bytes: int,
    ):
        self.export_id = export_id
        self.mp4_path = mp4_path
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.duration_seconds = duration_seconds
        self.created_at = created_at
        self.file_size_bytes = file_size_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_id": self.export_id,
            "mp4_path": str(self.mp4_path),
            "mp4_filename": self.mp4_path.name,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "metadata": self.metadata,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
            "file_size_bytes": self.file_size_bytes,
            "title": self.metadata.get("title", ""),
            "description": self.metadata.get("description", ""),
            "template": self.metadata.get("template", ""),
        }


def scan_project_exports(exports_dir: Path) -> List[ExportInfo]:
    """Scan the exports directory for mp4 files with optional metadata.json."""
    exports = []

    if not exports_dir.exists():
        return exports

    for mp4_path in exports_dir.glob("*.mp4"):
        # Look for matching metadata.json
        metadata_path = mp4_path.with_suffix(".json")
        if not metadata_path.exists():
            # Try _metadata.json pattern
            metadata_path = mp4_path.parent / f"{mp4_path.stem}_metadata.json"

        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        else:
            metadata_path = None

        # Get file stats
        try:
            st = mp4_path.stat()
            file_size = st.st_size
            created_at = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            file_size = 0
            created_at = _utc_iso()

        # Duration from metadata or estimate
        duration_seconds = float(metadata.get("duration_seconds", 0))

        export_id = mp4_path.stem

        exports.append(
            ExportInfo(
                export_id=export_id,
                mp4_path=mp4_path,
                metadata_path=metadata_path,
                metadata=metadata,
                duration_seconds=duration_seconds,
                created_at=created_at,
                file_size_bytes=file_size,
            )
        )

    # Sort by creation time (newest first)
    exports.sort(key=lambda e: e.created_at, reverse=True)
    return exports


def is_safe_export_path(file_path: Path, exports_dir: Path) -> bool:
    """Check that the file path is inside the exports directory.
    
    This prevents publishing arbitrary files on the system.
    """
    try:
        file_path = file_path.resolve()
        exports_dir = exports_dir.resolve()
        return file_path.is_relative_to(exports_dir)
    except (ValueError, OSError):
        return False


def create_publisher_router(
    *,
    get_exports_dir: Callable[[], Optional[Path]],
    account_store: AccountStore,
    job_store: PublishJobStore,
) -> APIRouter:
    """Create the publisher API router.
    
    Args:
        get_exports_dir: Callable that returns the current project's exports directory,
                         or None if no project is active.
        account_store: The account store instance.
        job_store: The publish job store instance.
    """
    router = APIRouter(prefix="/api/publisher", tags=["publisher"])

    def require_exports_dir() -> Path:
        """Get exports dir or raise 400 if no project active."""
        exports_dir = get_exports_dir()
        if exports_dir is None:
            raise HTTPException(status_code=400, detail="no_active_project")
        return exports_dir

    # -------------------------------------------------------------------------
    # Accounts
    # -------------------------------------------------------------------------

    @router.get("/accounts")
    def get_accounts() -> JSONResponse:
        """List all connected accounts."""
        accounts = [acct.to_dict() for acct in account_store.list()]
        return JSONResponse({"accounts": accounts})

    # -------------------------------------------------------------------------
    # Exports
    # -------------------------------------------------------------------------

    @router.get("/exports")
    def get_exports() -> JSONResponse:
        """List all exports from the active project's exports directory."""
        exports_dir = require_exports_dir()
        exports = scan_project_exports(exports_dir)
        return JSONResponse({
            "exports": [e.to_dict() for e in exports],
            "exports_dir": str(exports_dir),
        })

    # -------------------------------------------------------------------------
    # Queue publish
    # -------------------------------------------------------------------------

    @router.post("/queue")
    def queue_publish(body: Dict[str, Any] = Body(...)) -> JSONResponse:  # type: ignore[valid-type]
        """Queue a single publish job.
        
        Body:
            account_id: str - The account to publish to
            export_id: str - The export ID (mp4 filename stem) to publish
            options: dict - Optional overrides (privacy, title_override, description_override, etc.)
        """
        exports_dir = require_exports_dir()

        account_id = str(body.get("account_id") or "")
        export_id = str(body.get("export_id") or "")
        options = body.get("options") or {}

        # Validate account
        account = account_store.get(account_id)
        if not account:
            raise HTTPException(status_code=404, detail="account_not_found")

        # Find export
        mp4_path = exports_dir / f"{export_id}.mp4"
        if not mp4_path.exists():
            raise HTTPException(status_code=404, detail="export_not_found")

        # Safety check
        if not is_safe_export_path(mp4_path, exports_dir):
            raise HTTPException(status_code=400, detail="invalid_export_path")

        # Find or create metadata
        metadata_path = mp4_path.with_suffix(".json")
        if not metadata_path.exists():
            metadata_path = exports_dir / f"{export_id}_metadata.json"

        if not metadata_path.exists():
            # Create a minimal metadata file
            metadata = {
                "title": options.get("title_override") or export_id,
                "description": options.get("description_override") or "",
                "privacy": options.get("privacy", "private"),
            }
            metadata_path = mp4_path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            # Load and potentially update metadata with overrides
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if options.get("title_override"):
                metadata["title"] = options["title_override"]
            if options.get("description_override"):
                metadata["description"] = options["description_override"]
            if options.get("privacy"):
                metadata["privacy"] = options["privacy"]
            if options.get("hashtags_append"):
                existing = metadata.get("description", "")
                metadata["description"] = f"{existing}\n\n{options['hashtags_append']}".strip()
            # Write updated metadata to a job-specific file to preserve original
            job_metadata_path = exports_dir / f"{export_id}_{uuid.uuid4().hex[:8]}_job.json"
            job_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            metadata_path = job_metadata_path

        # Create job
        job_id = uuid.uuid4().hex
        job = job_store.create_job(
            job_id=job_id,
            platform=account.platform,
            account_id=account.id,
            file_path=str(mp4_path),
            metadata_path=str(metadata_path),
        )

        return JSONResponse({"job_id": job.id, "status": job.status})

    @router.post("/queue_batch")
    def queue_publish_batch(body: Dict[str, Any] = Body(...)) -> JSONResponse:  # type: ignore[valid-type]
        """Queue multiple publish jobs.
        
        Body:
            account_ids: list[str] - Account IDs to publish to
            export_ids: list[str] - Export IDs to publish
            options: dict - Shared options (privacy, etc.)
            stagger_seconds: int - Seconds between scheduled jobs (default 0)
        
        Creates one job per (export, account) combination.
        If stagger_seconds > 0, jobs are scheduled with increasing delays.
        """
        exports_dir = require_exports_dir()

        account_ids = body.get("account_ids") or []
        export_ids = body.get("export_ids") or []
        options = body.get("options") or {}
        stagger_seconds = int(body.get("stagger_seconds", 0))

        if not account_ids:
            raise HTTPException(status_code=400, detail="account_ids_required")
        if not export_ids:
            raise HTTPException(status_code=400, detail="export_ids_required")

        # Validate all accounts first
        accounts = []
        for aid in account_ids:
            account = account_store.get(aid)
            if not account:
                raise HTTPException(status_code=404, detail=f"account_not_found: {aid}")
            accounts.append(account)

        # Validate all exports
        exports = []
        for eid in export_ids:
            mp4_path = exports_dir / f"{eid}.mp4"
            if not mp4_path.exists():
                raise HTTPException(status_code=404, detail=f"export_not_found: {eid}")
            if not is_safe_export_path(mp4_path, exports_dir):
                raise HTTPException(status_code=400, detail=f"invalid_export_path: {eid}")
            exports.append((eid, mp4_path))

        # Create jobs
        created_jobs = []
        job_index = 0

        for export_id, mp4_path in exports:
            # Find or create metadata for this export
            metadata_path = mp4_path.with_suffix(".json")
            if not metadata_path.exists():
                metadata_path = exports_dir / f"{export_id}_metadata.json"

            base_metadata: Dict[str, Any] = {}
            if metadata_path.exists():
                try:
                    base_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

            for account in accounts:
                # Create job-specific metadata
                metadata = {**base_metadata}
                if options.get("title_override"):
                    metadata["title"] = options["title_override"]
                if options.get("description_override"):
                    metadata["description"] = options["description_override"]
                if options.get("privacy"):
                    metadata["privacy"] = options["privacy"]
                if options.get("hashtags_append"):
                    existing = metadata.get("description", "")
                    metadata["description"] = f"{existing}\n\n{options['hashtags_append']}".strip()

                # Ensure title exists
                if not metadata.get("title"):
                    metadata["title"] = export_id

                # Write job metadata
                job_metadata_path = exports_dir / f"{export_id}_{uuid.uuid4().hex[:8]}_job.json"
                job_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                # Create job
                job_id = uuid.uuid4().hex
                job = job_store.create_job(
                    job_id=job_id,
                    platform=account.platform,
                    account_id=account.id,
                    file_path=str(mp4_path),
                    metadata_path=str(job_metadata_path),
                )

                created_jobs.append({"job_id": job.id, "export_id": export_id, "account_id": account.id})
                job_index += 1

        return JSONResponse({
            "jobs": created_jobs,
            "total": len(created_jobs),
        })

    # -------------------------------------------------------------------------
    # Jobs
    # -------------------------------------------------------------------------

    @router.get("/jobs")
    def get_jobs(limit: int = 100, project_only: bool = True) -> JSONResponse:
        """List publish jobs.
        
        If project_only=True (default), filters to jobs whose file_path
        is inside the active project's exports directory.
        """
        all_jobs = job_store.list_jobs(limit=limit)

        if project_only:
            exports_dir = get_exports_dir()
            if exports_dir:
                exports_dir_str = str(exports_dir.resolve())
                all_jobs = [
                    j for j in all_jobs
                    if j.file_path.startswith(exports_dir_str)
                ]

        return JSONResponse({
            "jobs": [j.to_dict() for j in all_jobs],
        })

    @router.post("/jobs/{job_id}/retry")
    def retry_job(job_id: str) -> JSONResponse:
        """Retry a failed or canceled job."""
        try:
            job = job_store.retry(job_id)
            return JSONResponse({"job": job.to_dict()})
        except KeyError:
            raise HTTPException(status_code=404, detail="job_not_found")

    @router.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> JSONResponse:
        """Cancel a queued or running job."""
        try:
            job = job_store.cancel(job_id)
            return JSONResponse({"job": job.to_dict()})
        except KeyError:
            raise HTTPException(status_code=404, detail="job_not_found")

    @router.get("/jobs/stream")
    def jobs_stream() -> StreamingResponse:
        """SSE stream of job updates.
        
        Emits job updates whenever a job's updated_at changes.
        Useful for real-time progress tracking in the UI.
        """
        def event_stream():
            last_seen: Dict[str, str] = {}

            while True:
                jobs = job_store.list_jobs(limit=50)
                updates = []

                for job in jobs:
                    if last_seen.get(job.id) != job.updated_at:
                        last_seen[job.id] = job.updated_at
                        updates.append(job.to_dict())

                if updates:
                    payload = json.dumps({"type": "jobs_update", "jobs": updates})
                    yield f"data: {payload}\n\n"
                else:
                    # Send keepalive
                    yield ": keepalive\n\n"

                time.sleep(1.0)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
