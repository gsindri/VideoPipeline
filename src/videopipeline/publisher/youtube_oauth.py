from __future__ import annotations

from typing import Any, Iterable

YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_FORCE_SSL_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
YOUTUBE_DELETE_ALLOWED_SCOPES = (
    YOUTUBE_FORCE_SSL_SCOPE,
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtubepartner",
)
YOUTUBE_DEFAULT_SCOPES = (
    YOUTUBE_UPLOAD_SCOPE,
    YOUTUBE_FORCE_SSL_SCOPE,
)


def normalize_google_oauth_scopes(scopes: Any) -> list[str]:
    items: list[str]
    if scopes is None:
        items = []
    elif isinstance(scopes, str):
        items = []
        for chunk in scopes.split(","):
            items.extend(part.strip() for part in chunk.split())
    elif isinstance(scopes, Iterable):
        items = [str(part).strip() for part in scopes]
    else:
        items = [str(scopes).strip()]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def default_youtube_oauth_scopes() -> list[str]:
    return list(YOUTUBE_DEFAULT_SCOPES)


def merge_google_oauth_scopes(*scope_sets: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for scope_set in scope_sets:
        for scope in normalize_google_oauth_scopes(scope_set):
            if scope in seen:
                continue
            seen.add(scope)
            merged.append(scope)
    return merged


def youtube_scopes_allow_delete(scopes: Any) -> bool:
    normalized = set(normalize_google_oauth_scopes(scopes))
    return bool(normalized.intersection(YOUTUBE_DELETE_ALLOWED_SCOPES))
