from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from .accounts import Account
from .connectors.tiktok import refresh_token as refresh_tiktok_token
from .secrets import load_tokens, store_tokens


@dataclass(frozen=True)
class PublishAccountAuthStatus:
    ready: bool
    has_tokens: bool
    auth_state: str
    auth_error: str | None = None


def _format_auth_error(platform: str, exc: Exception) -> str:
    message = str(exc).strip()
    lowered = message.lower()
    if platform == "youtube" and "invalid_grant" in lowered:
        return "Google refresh token was rejected (invalid_grant). Reconnect the YouTube account."
    if platform == "tiktok" and "invalid_grant" in lowered:
        return "TikTok refresh token was rejected (invalid_grant). Reconnect the TikTok account."
    return message or f"{platform}_auth_failed"


def _youtube_auth_status(account: Account, tokens: dict[str, Any]) -> PublishAccountAuthStatus:
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    client_id = str(tokens.get("client_id") or "").strip()
    client_secret = str(tokens.get("client_secret") or "").strip()
    if not refresh_token or not client_id or not client_secret:
        return PublishAccountAuthStatus(
            ready=False,
            has_tokens=True,
            auth_state="needs_reauth",
            auth_error="YouTube account is missing refresh credentials. Reconnect the account.",
        )

    creds = Credentials(
        token=tokens.get("access_token"),
        refresh_token=refresh_token,
        token_uri=tokens.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=client_id,
        client_secret=client_secret,
        scopes=tokens.get("scopes"),
    )
    try:
        creds.refresh(Request())
    except RefreshError as exc:
        return PublishAccountAuthStatus(
            ready=False,
            has_tokens=True,
            auth_state="needs_reauth",
            auth_error=_format_auth_error(account.platform, exc),
        )
    except Exception as exc:
        return PublishAccountAuthStatus(
            ready=False,
            has_tokens=True,
            auth_state="token_check_failed",
            auth_error=_format_auth_error(account.platform, exc),
        )

    updated = dict(tokens)
    updated["access_token"] = creds.token
    updated["token_uri"] = creds.token_uri
    updated["client_id"] = creds.client_id
    updated["client_secret"] = creds.client_secret
    updated["scopes"] = list(creds.scopes or tokens.get("scopes") or [])
    store_tokens(account.platform, account.id, updated)
    return PublishAccountAuthStatus(ready=True, has_tokens=True, auth_state="ready")


def _tiktok_auth_status(account: Account, tokens: dict[str, Any]) -> PublishAccountAuthStatus:
    access_token = str(tokens.get("access_token") or "").strip()
    expires_at_raw = tokens.get("expires_at")
    expires_at = float(expires_at_raw) if expires_at_raw is not None else None
    if access_token and (expires_at is None or expires_at > time.time() + 60):
        return PublishAccountAuthStatus(ready=True, has_tokens=True, auth_state="ready")

    refresh = str(tokens.get("refresh_token") or "").strip()
    client_key = str(tokens.get("client_key") or "").strip()
    client_secret = str(tokens.get("client_secret") or "").strip()
    if not refresh or not client_key or not client_secret:
        return PublishAccountAuthStatus(
            ready=False,
            has_tokens=True,
            auth_state="needs_reauth",
            auth_error="TikTok account is missing refresh credentials. Reconnect the account.",
        )

    try:
        refreshed = refresh_tiktok_token(
            client_key=client_key,
            client_secret=client_secret,
            refresh_token=refresh,
        )
    except Exception as exc:
        auth_state = "needs_reauth" if "invalid_grant" in str(exc).lower() else "token_check_failed"
        return PublishAccountAuthStatus(
            ready=False,
            has_tokens=True,
            auth_state=auth_state,
            auth_error=_format_auth_error(account.platform, exc),
        )

    updated = dict(tokens)
    updated.update(refreshed)
    updated["client_key"] = client_key
    updated["client_secret"] = client_secret
    updated["expires_at"] = time.time() + float(refreshed.get("expires_in", 0))
    store_tokens(account.platform, account.id, updated)
    return PublishAccountAuthStatus(ready=True, has_tokens=True, auth_state="ready")


def get_publish_account_auth(account: Account) -> PublishAccountAuthStatus:
    try:
        tokens = load_tokens(account.platform, account.id)
    except Exception as exc:
        return PublishAccountAuthStatus(
            ready=False,
            has_tokens=False,
            auth_state="token_check_failed",
            auth_error=_format_auth_error(account.platform, exc),
        )

    if not tokens:
        return PublishAccountAuthStatus(
            ready=False,
            has_tokens=False,
            auth_state="missing_tokens",
            auth_error="No stored credentials are available for this account.",
        )

    if account.platform == "youtube":
        return _youtube_auth_status(account, tokens)
    if account.platform == "tiktok":
        return _tiktok_auth_status(account, tokens)
    return PublishAccountAuthStatus(ready=True, has_tokens=True, auth_state="ready")
