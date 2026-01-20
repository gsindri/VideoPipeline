from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from ..accounts import Account
from ..secrets import store_tokens


AUTH_BASE = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
API_BASE = "https://open.tiktokapis.com"


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def build_pkce_pair() -> tuple[str, str]:
    verifier = _base64url(secrets.token_bytes(32))
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def build_authorize_url(*, client_key: str, redirect_uri: str, scopes: str, state: str, code_challenge: str) -> str:
    return (
        f"{AUTH_BASE}?client_key={client_key}"
        f"&scope={scopes}"
        f"&response_type=code"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
        f"&code_challenge={code_challenge}"
        f"&code_challenge_method=S256"
    )


def exchange_code(
    *,
    client_key: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    resp = requests.post(
        TOKEN_URL,
        data={
            "client_key": client_key,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"tiktok_token_failed: {resp.status_code} {resp.text}")
    return resp.json()


def refresh_token(*, client_key: str, client_secret: str, refresh_token: str) -> dict[str, Any]:
    resp = requests.post(
        TOKEN_URL,
        data={
            "client_key": client_key,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"tiktok_refresh_failed: {resp.status_code} {resp.text}")
    return resp.json()


@dataclass
class TikTokMode:
    mode: str
    endpoint: str


class TikTokConnector:
    platform = "tiktok"

    def __init__(self, *, account: Account, tokens: dict[str, Any]) -> None:
        self.account = account
        self.tokens = tokens

    def _ensure_token(self) -> str:
        access = self.tokens.get("access_token")
        expires_at = self.tokens.get("expires_at")
        if expires_at and time.time() > float(expires_at) - 30:
            refreshed = refresh_token(
                client_key=self.tokens["client_key"],
                client_secret=self.tokens["client_secret"],
                refresh_token=self.tokens["refresh_token"],
            )
            refreshed["client_key"] = self.tokens["client_key"]
            refreshed["client_secret"] = self.tokens["client_secret"]
            refreshed["expires_at"] = time.time() + float(refreshed.get("expires_in", 0))
            store_tokens(self.platform, self.account.id, refreshed)
            self.tokens = refreshed
            access = refreshed.get("access_token")
        if not access:
            raise RuntimeError("missing_access_token")
        return str(access)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._ensure_token()}"}

    def validate_media(self, file_path: Path, metadata: dict[str, Any]) -> None:
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))
        if file_path.stat().st_size <= 0:
            raise ValueError("empty_file")

    def _mode(self, metadata: dict[str, Any]) -> TikTokMode:
        mode = (metadata.get("tiktok_mode") or self.account.metadata.get("tiktok_mode") or "inbox").lower()
        if mode == "direct":
            return TikTokMode(mode="direct", endpoint="/v2/post/publish/video/init/")
        return TikTokMode(mode="inbox", endpoint="/v2/post/publish/inbox/video/init/")

    def create_upload_session(
        self,
        *,
        file_path: Path,
        metadata: dict[str, Any],
        resume_state: dict[str, Any],
    ) -> dict[str, Any]:
        if resume_state.get("upload_url") and resume_state.get("publish_id"):
            return resume_state

        mode = self._mode(metadata)
        size = file_path.stat().st_size
        chunk_size = int(metadata.get("chunk_size") or 8 * 1024 * 1024)
        total_chunks = max(1, int((size + chunk_size - 1) / chunk_size))
        payload = {
            "post_info": {"title": metadata.get("caption") or metadata.get("title") or "VideoPipeline upload"},
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": size,
                "chunk_size": chunk_size,
                "total_chunk_count": total_chunks,
            },
        }

        resp = requests.post(
            f"{API_BASE}{mode.endpoint}",
            headers=self._headers() | {"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30,
        )
        if resp.status_code != 200:
            if mode.mode == "direct":
                metadata = dict(metadata)
                metadata["tiktok_mode"] = "inbox"
                return self.create_upload_session(
                    file_path=file_path,
                    metadata=metadata,
                    resume_state=resume_state,
                )
            raise RuntimeError(f"tiktok_init_failed: {resp.status_code} {resp.text}")
        data = resp.json().get("data") or {}
        return {
            "mode": mode.mode,
            "upload_url": data.get("upload_url"),
            "publish_id": data.get("publish_id"),
            "chunk_size": chunk_size,
            "total_bytes": size,
            "next_byte": 0,
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
        upload_url = session.get("upload_url")
        if not upload_url:
            raise RuntimeError("missing_upload_url")
        chunk_size = int(session.get("chunk_size") or 8 * 1024 * 1024)
        total_bytes = int(session.get("total_bytes") or file_path.stat().st_size)
        start = int(session.get("next_byte") or 0)

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
                resp = requests.put(upload_url, data=chunk, headers=headers, timeout=60)
                if resp.status_code not in {200, 201, 202, 206}:
                    raise RuntimeError(f"tiktok_upload_failed: {resp.status_code} {resp.text}")
                start = end + 1
                session["next_byte"] = start
                on_resume(session)
                on_progress(min(1.0, start / total_bytes))
        return {"remote_id": session.get("publish_id")}

    def finalize_or_poll(self, *, session: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        if session.get("mode") != "direct":
            return {"remote_id": session.get("publish_id")}

        publish_id = session.get("publish_id")
        if not publish_id:
            raise RuntimeError("missing_publish_id")

        for _ in range(6):
            resp = requests.post(
                f"{API_BASE}/v2/post/publish/status/fetch/",
                headers=self._headers() | {"Content-Type": "application/json"},
                data=json.dumps({"publish_id": publish_id}),
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"tiktok_status_failed: {resp.status_code} {resp.text}")
            data = resp.json().get("data") or {}
            if data.get("status") in {"SUCCESS", "SUCCESSFUL", "FINISHED"}:
                return {"remote_id": publish_id, "remote_url": data.get("share_url")}
            time.sleep(5)
        return {"remote_id": publish_id}
