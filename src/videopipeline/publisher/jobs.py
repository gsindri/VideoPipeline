from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .state import publisher_db_path

_UNSET = object()


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class PublishJob:
    id: str
    created_at: str
    updated_at: str
    status: str
    platform: str
    account_id: str
    file_path: str
    metadata_path: str
    progress: float
    attempts: int
    last_error: Optional[str]
    remote_id: Optional[str]
    remote_url: Optional[str]
    resume_json: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "platform": self.platform,
            "account_id": self.account_id,
            "file_path": self.file_path,
            "metadata_path": self.metadata_path,
            "progress": self.progress,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "remote_id": self.remote_id,
            "remote_url": self.remote_url,
            "resume_json": self.resume_json,
        }

    def resume_state(self) -> dict[str, Any]:
        if not self.resume_json:
            return {}
        try:
            return json.loads(self.resume_json)
        except json.JSONDecodeError:
            return {}


class PublishJobStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or publisher_db_path()
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS publish_jobs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT,
                    platform TEXT,
                    account_id TEXT,
                    file_path TEXT,
                    metadata_path TEXT,
                    progress REAL,
                    attempts INTEGER,
                    last_error TEXT,
                    remote_id TEXT,
                    remote_url TEXT,
                    resume_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS publish_dedup (
                    platform TEXT,
                    account_id TEXT,
                    sha256 TEXT,
                    remote_id TEXT,
                    remote_url TEXT,
                    created_at TEXT,
                    PRIMARY KEY (platform, account_id, sha256)
                )
                """
            )

    def create_job(
        self,
        *,
        job_id: str,
        platform: str,
        account_id: str,
        file_path: str,
        metadata_path: str,
    ) -> PublishJob:
        now = _utc_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO publish_jobs (
                    id, created_at, updated_at, status, platform, account_id,
                    file_path, metadata_path, progress, attempts, last_error,
                    remote_id, remote_url, resume_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    now,
                    now,
                    "queued",
                    platform,
                    account_id,
                    file_path,
                    metadata_path,
                    0.0,
                    0,
                    None,
                    None,
                    None,
                    None,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> PublishJob:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM publish_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise KeyError(f"job_not_found: {job_id}")
        return PublishJob(**dict(row))

    def list_jobs(self, limit: int = 100) -> list[PublishJob]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM publish_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [PublishJob(**dict(row)) for row in rows]

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        attempts: Optional[int] = None,
        last_error: Any = _UNSET,
        remote_id: Any = _UNSET,
        remote_url: Any = _UNSET,
        resume_json: Any = _UNSET,
    ) -> PublishJob:
        updates = []
        values: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if progress is not None:
            updates.append("progress = ?")
            values.append(float(progress))
        if attempts is not None:
            updates.append("attempts = ?")
            values.append(int(attempts))
        if last_error is not _UNSET:
            updates.append("last_error = ?")
            values.append(last_error)
        if remote_id is not _UNSET:
            updates.append("remote_id = ?")
            values.append(remote_id)
        if remote_url is not _UNSET:
            updates.append("remote_url = ?")
            values.append(remote_url)
        if resume_json is not _UNSET:
            updates.append("resume_json = ?")
            values.append(resume_json)
        updates.append("updated_at = ?")
        values.append(_utc_iso())
        values.append(job_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE publish_jobs SET {', '.join(updates)} WHERE id = ?", values)
        return self.get_job(job_id)

    def claim_next(self, *, backoff_fn) -> Optional[PublishJob]:
        with self._lock:
            jobs = self._queued_jobs()
            now = datetime.now(timezone.utc)
            for job in jobs:
                if job.attempts > 0:
                    delay = timedelta(seconds=backoff_fn(job.attempts))
                    updated = _parse_iso(job.updated_at)
                    if updated + delay > now:
                        continue
                return self.update_job(job.id, status="running")
        return None

    def _queued_jobs(self) -> list[PublishJob]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM publish_jobs WHERE status = ? ORDER BY created_at",
                ("queued",),
            ).fetchall()
        return [PublishJob(**dict(row)) for row in rows]

    def mark_dedup(self, platform: str, account_id: str, sha256: str, remote_id: str, remote_url: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO publish_dedup (
                    platform, account_id, sha256, remote_id, remote_url, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (platform, account_id, sha256, remote_id, remote_url, _utc_iso()),
            )

    def lookup_dedup(self, platform: str, account_id: str, sha256: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM publish_dedup WHERE platform = ? AND account_id = ? AND sha256 = ?
                """,
                (platform, account_id, sha256),
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def retry(self, job_id: str) -> PublishJob:
        return self.update_job(job_id, status="queued", last_error=None, progress=0.0)

    def cancel(self, job_id: str) -> PublishJob:
        return self.update_job(job_id, status="canceled")
