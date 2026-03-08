from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

_YOUTUBE_ID_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)")
_TWITCH_VOD_ID_RE = re.compile(r"twitch\.tv/videos/(\d+)")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if parsed.query:
        return f"{parsed.scheme.lower()}://{host}{path}?{parsed.query}"
    return f"{parsed.scheme.lower()}://{host}{path}"


def _extract_content_key(url: str) -> str:
    normalized = _normalize_url(url)
    if not normalized:
        return ""
    yt_match = _YOUTUBE_ID_RE.search(normalized)
    if yt_match:
        return f"youtube_{yt_match.group(1)}"
    twitch_match = _TWITCH_VOD_ID_RE.search(normalized)
    if twitch_match:
        return f"twitch_{twitch_match.group(1)}"
    return normalized


def default_source_inbox_paths() -> list[Path]:
    paths: list[Path] = []

    env_path = str(os.environ.get("VP_SOURCE_INBOX") or "").strip()
    if env_path:
        paths.append(Path(env_path))

    cwd = Path.cwd()
    paths.append(cwd / "sources" / "inbox.local.json")

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            paths.append(Path(appdata) / "VideoPipeline" / "source_inbox.json")
    else:
        paths.append(Path.home() / ".videopipeline" / "source_inbox.json")

    seen = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def resolve_source_inbox_path(path: Optional[Path] = None, *, create_default: bool = False) -> Optional[Path]:
    if path is not None:
        return Path(path)
    candidates = default_source_inbox_paths()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if create_default and candidates:
        return candidates[0]
    return None


def load_source_inbox(path: Optional[Path] = None) -> tuple[Optional[Path], list[Dict[str, Any]]]:
    resolved = resolve_source_inbox_path(path)
    if resolved is None:
        return None, []
    if not resolved.exists():
        return resolved, []
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("source inbox must be a list")
    return resolved, [item for item in data if isinstance(item, dict)]


def save_source_inbox(path: Path, entries: list[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp.replace(path)


def list_source_inbox_entries(
    *,
    path: Optional[Path] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> tuple[Optional[Path], list[Dict[str, Any]]]:
    resolved, entries = load_source_inbox(path)
    if status is not None:
        target = str(status).strip().lower()
        entries = [item for item in entries if str(item.get("status") or "").strip().lower() == target]
    entries.sort(key=lambda item: str(item.get("added_at") or ""), reverse=True)
    if limit is not None and limit >= 0:
        entries = entries[: int(limit)]
    return resolved, entries


def add_source_inbox_entry(
    *,
    url: str,
    title: Optional[str] = None,
    notes: Optional[str] = None,
    priority: Optional[float] = None,
    tags: Optional[list[str]] = None,
    added_by: Optional[str] = None,
    source_id: Optional[str] = None,
    source_label: Optional[str] = None,
    path: Optional[Path] = None,
) -> tuple[Path, Dict[str, Any], bool]:
    resolved = resolve_source_inbox_path(path, create_default=True)
    if resolved is None:
        raise ValueError("no_source_inbox_path")

    _, entries = load_source_inbox(resolved)
    normalized_url = _normalize_url(url)
    if not normalized_url:
        raise ValueError("invalid_url")

    for entry in entries:
        if (
            _normalize_url(str(entry.get("url") or "")) == normalized_url
            and str(entry.get("status") or "").strip().lower() == "pending"
        ):
            return resolved, entry, False

    clean_tags = [str(item).strip() for item in list(tags or []) if str(item).strip()]
    entry = {
        "inbox_id": uuid.uuid4().hex,
        "url": normalized_url,
        "content_key": _extract_content_key(normalized_url) or None,
        "title": str(title or "").strip() or None,
        "notes": str(notes or "").strip() or None,
        "priority": float(priority) if priority is not None else 5.0,
        "tags": clean_tags,
        "added_by": str(added_by or "").strip() or None,
        "source_id": str(source_id or "").strip() or None,
        "source_label": str(source_label or "").strip() or None,
        "status": "pending",
        "added_at": _utc_now_iso(),
        "selected_at": None,
        "project_id": None,
        "selection_notes": None,
    }
    entries.append(entry)
    save_source_inbox(resolved, entries)
    return resolved, entry, True


def update_source_inbox_entry(
    inbox_id: str,
    *,
    status: str,
    project_id: Optional[str] = None,
    selection_notes: Optional[str] = None,
    path: Optional[Path] = None,
) -> tuple[Path, Dict[str, Any]]:
    resolved = resolve_source_inbox_path(path, create_default=True)
    if resolved is None:
        raise ValueError("no_source_inbox_path")

    _, entries = load_source_inbox(resolved)
    target_id = str(inbox_id or "").strip()
    if not target_id:
        raise ValueError("missing_inbox_id")

    updated: Optional[Dict[str, Any]] = None
    for entry in entries:
        if str(entry.get("inbox_id") or "").strip() != target_id:
            continue
        entry["status"] = str(status or "").strip().lower() or "pending"
        if project_id is not None:
            entry["project_id"] = str(project_id).strip() or None
        if selection_notes is not None:
            entry["selection_notes"] = str(selection_notes).strip() or None
        if entry["status"] in {"selected", "dismissed", "processed"}:
            entry["selected_at"] = _utc_now_iso()
        updated = entry
        break

    if updated is None:
        raise KeyError(target_id)

    save_source_inbox(resolved, entries)
    return resolved, updated
