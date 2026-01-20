from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .accounts import AccountStore
from .jobs import PublishJob, PublishJobStore
from .presets import AccountPreset, apply_presets
from .sanitize import sanitize_metadata
from .secrets import load_tokens
from .state import logs_dir
from .connectors import get_connector


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _backoff_seconds(attempts: int) -> int:
    return min(300, max(1, 2 ** min(attempts, 8)))


@dataclass
class PublishResult:
    remote_id: str
    remote_url: Optional[str] = None


class PublishWorker:
    def __init__(
        self,
        *,
        job_store: Optional[PublishJobStore] = None,
        account_store: Optional[AccountStore] = None,
    ) -> None:
        self.job_store = job_store or PublishJobStore()
        self.account_store = account_store or AccountStore()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def queue_job(self, *, platform: str, account_id: str, file_path: Path, metadata_path: Path) -> PublishJob:
        job_id = uuid.uuid4().hex
        return self.job_store.create_job(
            job_id=job_id,
            platform=platform,
            account_id=account_id,
            file_path=str(file_path),
            metadata_path=str(metadata_path),
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def run_loop(self) -> None:
        while not self._stop.is_set():
            job = self.job_store.claim_next(backoff_fn=_backoff_seconds)
            if not job:
                time.sleep(1.0)
                continue
            self._run_job(job)

    def run_once(self, job_id: Optional[str] = None) -> Optional[PublishJob]:
        if job_id:
            job = self.job_store.get_job(job_id)
            if job.status != "queued":
                return job
            job = self.job_store.update_job(job.id, status="running")
        else:
            job = self.job_store.claim_next(backoff_fn=_backoff_seconds)
        if not job:
            return None
        self._run_job(job)
        return self.job_store.get_job(job.id)

    def _run_job(self, job: PublishJob) -> None:
        log_path = logs_dir() / f"publish_{job.id}.log"
        try:
            account = self.account_store.get(job.account_id)
            if not account:
                raise RuntimeError(f"account_not_found: {job.account_id}")

            metadata = json.loads(Path(job.metadata_path).read_text(encoding="utf-8"))
            presets = AccountPreset.from_dict(account.presets)
            metadata = apply_presets(metadata, presets)
            metadata = sanitize_metadata(job.platform, metadata)

            file_path = Path(job.file_path)
            sha = _sha256(file_path)
            dedup = self.job_store.lookup_dedup(job.platform, job.account_id, sha)
            if dedup:
                self.job_store.update_job(
                    job.id,
                    status="succeeded",
                    progress=1.0,
                    remote_id=dedup.get("remote_id"),
                    remote_url=dedup.get("remote_url"),
                )
                return

            tokens = load_tokens(job.platform, job.account_id)
            if not tokens:
                raise RuntimeError("missing_tokens")

            connector = get_connector(job.platform, account=account, tokens=tokens)
            connector.validate_media(file_path, metadata)

            def on_progress(frac: float) -> None:
                self.job_store.update_job(job.id, progress=frac)

            def on_resume(state: dict[str, Any]) -> None:
                self.job_store.update_job(job.id, resume_json=json.dumps(state))

            result = connector.publish(
                file_path=file_path,
                metadata=metadata,
                resume_state=job.resume_state(),
                on_progress=on_progress,
                on_resume=on_resume,
            )
            self.job_store.update_job(
                job.id,
                status="succeeded",
                progress=1.0,
                remote_id=result.remote_id,
                remote_url=result.remote_url,
                last_error=None,
            )
            self.job_store.mark_dedup(job.platform, job.account_id, sha, result.remote_id, result.remote_url)
        except Exception as exc:
            attempts = job.attempts + 1
            status = "queued" if attempts < 5 else "failed"
            self.job_store.update_job(
                job.id,
                status=status,
                attempts=attempts,
                last_error=f"{type(exc).__name__}: {exc}",
            )
            log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
