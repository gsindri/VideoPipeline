from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


ProgressCallback = Callable[[float], None]
ResumeCallback = Callable[[dict[str, Any]], None]


@dataclass
class PublishResult:
    remote_id: str
    remote_url: str | None = None


class Connector(Protocol):
    platform: str

    def validate_media(self, file_path: Path, metadata: dict[str, Any]) -> None:
        ...

    def create_upload_session(
        self,
        *,
        file_path: Path,
        metadata: dict[str, Any],
        resume_state: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    def upload(
        self,
        *,
        file_path: Path,
        metadata: dict[str, Any],
        session: dict[str, Any],
        on_progress: ProgressCallback,
        on_resume: ResumeCallback,
    ) -> dict[str, Any]:
        ...

    def finalize_or_poll(
        self,
        *,
        session: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    def publish(
        self,
        *,
        file_path: Path,
        metadata: dict[str, Any],
        resume_state: dict[str, Any],
        on_progress: ProgressCallback,
        on_resume: ResumeCallback,
    ) -> PublishResult:
        session = self.create_upload_session(
            file_path=file_path,
            metadata=metadata,
            resume_state=resume_state,
        )
        on_resume(session)
        upload_result = self.upload(
            file_path=file_path,
            metadata=metadata,
            session=session,
            on_progress=on_progress,
            on_resume=on_resume,
        )
        final = self.finalize_or_poll(session=session, metadata=metadata)
        remote_id = str(final.get("remote_id") or upload_result.get("remote_id") or "")
        remote_url = final.get("remote_url") or upload_result.get("remote_url")
        if not remote_id:
            raise RuntimeError("missing_remote_id")
        return PublishResult(remote_id=remote_id, remote_url=remote_url)
