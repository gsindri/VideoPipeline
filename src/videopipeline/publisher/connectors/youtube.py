from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials

from ..accounts import Account


class YouTubeConnector:
    platform = "youtube"

    def __init__(self, *, account: Account, tokens: dict[str, Any]) -> None:
        self.account = account
        self.tokens = tokens

    def _credentials(self) -> Credentials:
        creds = Credentials(
            token=self.tokens.get("access_token"),
            refresh_token=self.tokens.get("refresh_token"),
            token_uri=self.tokens.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=self.tokens.get("client_id"),
            client_secret=self.tokens.get("client_secret"),
            scopes=self.tokens.get("scopes"),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self.tokens["access_token"] = creds.token
        return creds

    def _session(self) -> AuthorizedSession:
        return AuthorizedSession(self._credentials())

    def validate_media(self, file_path: Path, metadata: dict[str, Any]) -> None:
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))
        if file_path.stat().st_size <= 0:
            raise ValueError("empty_file")
        if not metadata.get("title"):
            raise ValueError("missing_title")

    def create_upload_session(
        self,
        *,
        file_path: Path,
        metadata: dict[str, Any],
        resume_state: dict[str, Any],
    ) -> dict[str, Any]:
        if resume_state.get("session_url"):
            return resume_state

        privacy = metadata.get("privacy") or "private"
        status: dict[str, Any] = {"privacyStatus": privacy}
        publish_at = metadata.get("publish_at")
        if publish_at and privacy == "private":
            status["publishAt"] = publish_at

        body = {
            "snippet": {
                "title": metadata.get("title") or "",
                "description": metadata.get("description") or metadata.get("caption") or "",
                "tags": metadata.get("hashtags") or [],
                "categoryId": metadata.get("category_id", "20"),
            },
            "status": status,
        }
        total_bytes = file_path.stat().st_size
        session = self._session()
        resp = session.post(
            "https://www.googleapis.com/upload/youtube/v3/videos",
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                "X-Upload-Content-Type": "video/*",
                "X-Upload-Content-Length": str(total_bytes),
                "Content-Type": "application/json; charset=UTF-8",
            },
            data=json.dumps(body),
        )
        if resp.status_code not in {200, 201}:
            raise RuntimeError(f"youtube_init_failed: {resp.status_code} {resp.text}")
        session_url = resp.headers.get("Location")
        if not session_url:
            raise RuntimeError("youtube_missing_session_url")
        return {
            "session_url": session_url,
            "chunk_size": metadata.get("chunk_size") or 8 * 1024 * 1024,
            "next_byte": 0,
            "total_bytes": total_bytes,
        }

    def upload(
        self,
        *,
        file_path: Path,
        metadata: dict[str, Any],
        session: dict[str, Any],
        on_progress,
        on_resume,
    ) -> dict[str, Any]:
        session_url = session["session_url"]
        chunk_size = int(session.get("chunk_size") or 8 * 1024 * 1024)
        total_bytes = int(session.get("total_bytes") or file_path.stat().st_size)
        start = int(session.get("next_byte") or 0)
        client = self._session()

        with file_path.open("rb") as handle:
            if start:
                handle.seek(start)
            while start < total_bytes:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                end = start + len(chunk) - 1
                headers = {
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {start}-{end}/{total_bytes}",
                }
                resp = client.put(session_url, data=chunk, headers=headers)
                if resp.status_code in {200, 201}:
                    data = resp.json()
                    return {
                        "remote_id": data.get("id"),
                        "remote_url": f"https://youtu.be/{data.get('id')}",
                    }
                if resp.status_code == 308:
                    range_header = resp.headers.get("Range")
                    if range_header and "-" in range_header:
                        last = int(range_header.split("-")[-1])
                        start = last + 1
                    else:
                        start = end + 1
                    session["next_byte"] = start
                    on_resume(session)
                    on_progress(min(1.0, start / total_bytes))
                    continue
                raise RuntimeError(f"youtube_upload_failed: {resp.status_code} {resp.text}")
        raise RuntimeError("youtube_upload_incomplete")

    def finalize_or_poll(self, *, session: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        return {}
