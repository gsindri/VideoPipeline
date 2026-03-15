from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlsplit

import requests
import yaml

from .project import default_projects_root
from .source_inbox import list_source_inbox_entries

_YOUTUBE_ID_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)")
_TWITCH_VOD_ID_RE = re.compile(r"twitch\.tv/videos/(\d+)")
_TWITCH_DURATION_RE = re.compile(
    r"^(?:(?P<days>\d+)d)?(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?$"
)

_TWITCH_OAUTH_URL = "https://id.twitch.tv/oauth2/token"
_TWITCH_HELIX_URL = "https://api.twitch.tv/helix"
_TWITCH_HTTP_TIMEOUT_S = 10.0
_TWITCH_TOKEN_LOCK = threading.Lock()
_TWITCH_TOKEN_CACHE: dict[str, Any] = {"access_token": "", "expires_at": 0.0}
_TWITCH_USER_CACHE: dict[str, dict[str, str]] = {}
_SCOUT_PROBE_SCHEMA_VERSION = 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if parsed != parsed:
        return float(default)
    return float(parsed)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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


def extract_content_key(url: str) -> str:
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


def default_watchlist_paths() -> list[Path]:
    paths: list[Path] = []

    env_path = str(os.environ.get("VP_SOURCE_WATCHLIST") or "").strip()
    if env_path:
        paths.append(Path(env_path))

    cwd = Path.cwd()
    paths.append(cwd / "sources" / "watchlist.local.yaml")
    paths.append(cwd / "sources" / "watchlist.local.yml")
    paths.append(cwd / "sources" / "watchlist.yaml")
    paths.append(cwd / "sources" / "watchlist.yml")

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            base = Path(appdata) / "VideoPipeline"
            paths.append(base / "source_watchlist.local.yaml")
            paths.append(base / "source_watchlist.local.yml")
            paths.append(base / "source_watchlist.yaml")
            paths.append(base / "source_watchlist.yml")
    else:
        base = Path.home() / ".videopipeline"
        paths.append(base / "source_watchlist.local.yaml")
        paths.append(base / "source_watchlist.local.yml")
        paths.append(base / "source_watchlist.yaml")
        paths.append(base / "source_watchlist.yml")

    seen = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def resolve_watchlist_path(path: Optional[Path] = None) -> Optional[Path]:
    if path is not None:
        return Path(path)
    for candidate in default_watchlist_paths():
        if candidate.exists():
            return candidate
    return None


def load_source_watchlist(path: Optional[Path] = None) -> tuple[Optional[Path], Dict[str, Any]]:
    resolved = resolve_watchlist_path(path)
    if resolved is None:
        return None, {"shadow_mode": True, "sources": []}

    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("source watchlist must be a mapping")
    raw_sources = data.get("sources") or []
    if not isinstance(raw_sources, list):
        raise ValueError("source watchlist 'sources' must be a list")
    data["sources"] = [item for item in raw_sources if isinstance(item, dict)]
    if "shadow_mode" not in data:
        data["shadow_mode"] = True
    return resolved, data


def _timestamp_to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    try:
        ts = float(value)
    except Exception:
        return None
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _parse_iso_ts(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt.timestamp()


def _age_hours(published_at: Optional[str], *, now_ts: float) -> Optional[float]:
    if not published_at:
        return None
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except Exception:
        return None
    return max(0.0, (now_ts - dt.timestamp()) / 3600.0)


def _match_any(text: str, words: list[str]) -> list[str]:
    hay = text.lower()
    matches: list[str] = []
    for word in words:
        token = str(word or "").strip().lower()
        if not token:
            continue
        if token in hay:
            matches.append(token)
    return matches


def _twitch_client_id() -> str:
    return str(
        os.environ.get("TWITCH_CLIENT_ID")
        or os.environ.get("TWITCH_API_CLIENT_ID")
        or ""
    ).strip()


def _twitch_client_secret() -> str:
    return str(
        os.environ.get("TWITCH_CLIENT_SECRET")
        or os.environ.get("TWITCH_API_CLIENT_SECRET")
        or ""
    ).strip()


def _twitch_static_access_token() -> str:
    return str(
        os.environ.get("TWITCH_APP_ACCESS_TOKEN")
        or os.environ.get("TWITCH_ACCESS_TOKEN")
        or ""
    ).strip()


def twitch_api_configured() -> bool:
    client_id = _twitch_client_id()
    if not client_id:
        return False
    return bool(_twitch_static_access_token() or _twitch_client_secret())


def _twitch_source_login(source: Dict[str, Any]) -> str:
    return str(
        source.get("channel_login")
        or source.get("twitch_login")
        or source.get("login")
        or ""
    ).strip().lower()


def _twitch_source_user_id(source: Dict[str, Any]) -> str:
    return str(
        source.get("channel_id")
        or source.get("user_id")
        or source.get("twitch_user_id")
        or ""
    ).strip()


def source_preflight_issue(source: Dict[str, Any]) -> Optional[str]:
    provider = str(source.get("provider") or "").strip().lower()
    source_url = str(source.get("url") or "").strip()
    if provider in {"twitch_helix", "twitch_api"}:
        if twitch_api_configured():
            if not (_twitch_source_user_id(source) or _twitch_source_login(source)):
                return "channel_login or user_id is required for twitch_helix sources"
            return None
        if source_url and yt_dlp_available():
            return None
        return "TWITCH_CLIENT_ID plus TWITCH_CLIENT_SECRET or TWITCH_APP_ACCESS_TOKEN is required"
    if provider and source_url and not yt_dlp_available():
        return "yt-dlp is required for source scouting"
    return None


def _load_yt_dlp() -> Any:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required for source scouting") from exc
    return YoutubeDL


def yt_dlp_available() -> bool:
    try:
        _load_yt_dlp()
    except RuntimeError:
        return False
    return True


def _parse_twitch_duration_seconds(value: Any) -> Optional[float]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    match = _TWITCH_DURATION_RE.match(raw)
    if not match:
        return None
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    total = (days * 86400) + (hours * 3600) + (minutes * 60) + seconds
    return float(total or 0)


def _twitch_get_access_token(force_refresh: bool = False) -> tuple[str, str]:
    client_id = _twitch_client_id()
    if not client_id:
        raise RuntimeError("TWITCH_CLIENT_ID is not set")

    static_token = _twitch_static_access_token()
    if static_token:
        return client_id, static_token

    client_secret = _twitch_client_secret()
    if not client_secret:
        raise RuntimeError("TWITCH_CLIENT_SECRET is not set")

    now = time.time()
    with _TWITCH_TOKEN_LOCK:
        cached_token = str(_TWITCH_TOKEN_CACHE.get("access_token") or "").strip()
        expires_at = float(_TWITCH_TOKEN_CACHE.get("expires_at") or 0.0)
        if not force_refresh and cached_token and expires_at > now + 60.0:
            return client_id, cached_token

        resp = requests.post(
            _TWITCH_OAUTH_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=_TWITCH_HTTP_TIMEOUT_S,
        )
        detail = ""
        if not resp.ok:
            try:
                payload = resp.json()
                detail = str(
                    payload.get("message")
                    or payload.get("error_description")
                    or payload.get("error")
                    or ""
                ).strip()
            except Exception:
                detail = resp.text.strip()
            raise RuntimeError(
                f"Twitch token request failed ({resp.status_code})"
                + (f": {detail}" if detail else "")
            )

        payload = resp.json()
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            raise RuntimeError("Twitch token response did not include access_token")
        expires_in = max(0, _safe_int(payload.get("expires_in"), 0))
        _TWITCH_TOKEN_CACHE["access_token"] = access_token
        _TWITCH_TOKEN_CACHE["expires_at"] = now + float(expires_in)
        return client_id, access_token


def _twitch_api_get(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    client_id, access_token = _twitch_get_access_token(force_refresh=force_refresh)
    resp = requests.get(
        f"{_TWITCH_HELIX_URL}/{path.lstrip('/')}",
        headers={
            "Client-ID": client_id,
            "Authorization": f"Bearer {access_token}",
        },
        params=params,
        timeout=_TWITCH_HTTP_TIMEOUT_S,
    )
    if resp.status_code == 401 and not force_refresh and not _twitch_static_access_token():
        with _TWITCH_TOKEN_LOCK:
            _TWITCH_TOKEN_CACHE["access_token"] = ""
            _TWITCH_TOKEN_CACHE["expires_at"] = 0.0
        return _twitch_api_get(path, params=params, force_refresh=True)
    if not resp.ok:
        detail = ""
        try:
            payload = resp.json()
            detail = str(payload.get("message") or payload.get("error") or "").strip()
        except Exception:
            detail = resp.text.strip()
        raise RuntimeError(
            f"Twitch API {path} failed ({resp.status_code})"
            + (f": {detail}" if detail else "")
        )
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Twitch API {path} returned a non-object response")
    return payload


def _resolve_twitch_user(source: Dict[str, Any]) -> dict[str, str]:
    user_id = _twitch_source_user_id(source)
    if user_id:
        login = _twitch_source_login(source)
        return {
            "id": user_id,
            "login": login,
            "display_name": str(source.get("label") or login or user_id).strip(),
        }

    login = _twitch_source_login(source)
    if not login:
        raise RuntimeError("channel_login or user_id is required for twitch_helix sources")
    cached = _TWITCH_USER_CACHE.get(login)
    if cached:
        return cached

    payload = _twitch_api_get("users", params={"login": login})
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Twitch user not found: {login}")
    row = rows[0] if isinstance(rows[0], dict) else {}
    resolved = {
        "id": str(row.get("id") or "").strip(),
        "login": str(row.get("login") or login).strip().lower(),
        "display_name": str(row.get("display_name") or login).strip(),
    }
    if not resolved["id"]:
        raise RuntimeError(f"Twitch user lookup returned no id for: {login}")
    _TWITCH_USER_CACHE[login] = resolved
    return resolved


def _fetch_twitch_helix_entries(source: Dict[str, Any], *, limit: int) -> list[Dict[str, Any]]:
    preflight_issue = source_preflight_issue(source)
    if preflight_issue:
        raise RuntimeError(preflight_issue)

    user = _resolve_twitch_user(source)
    video_type = str(source.get("video_type") or "archive").strip().lower() or "archive"
    payload = _twitch_api_get(
        "videos",
        params={
            "user_id": user["id"],
            "type": video_type,
            "first": max(1, min(int(limit), 100)),
        },
    )

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []

    out: list[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        video_id = str(row.get("id") or "").strip()
        candidate_url = str(row.get("url") or "").strip()
        if not candidate_url and video_id.isdigit():
            candidate_url = f"https://www.twitch.tv/videos/{video_id}"
        if not candidate_url.startswith("http"):
            continue
        duration_s = _parse_twitch_duration_seconds(row.get("duration"))
        published_at = str(
            row.get("published_at") or row.get("created_at") or ""
        ).strip() or None
        out.append(
            {
                "url": candidate_url,
                "title": str(row.get("title") or "").strip(),
                "duration_seconds": duration_s or 0.0,
                "published_at": published_at,
                "view_count": _safe_int(row.get("view_count"), 0),
                "channel_id": user["id"],
                "channel_name": str(
                    row.get("user_name") or user.get("display_name") or user.get("login") or ""
                ).strip(),
                "video_id": video_id or None,
                "is_live": False,
                "platform": "twitch",
                "fetch_mode": "twitch_helix",
            }
        )
        if len(out) >= limit:
            break
    return out


def fetch_source_entries(source: Dict[str, Any], *, limit: int) -> list[Dict[str, Any]]:
    provider = str(source.get("provider") or "").strip().lower()
    if provider in {"twitch_helix", "twitch_api"} and twitch_api_configured():
        return _fetch_twitch_helix_entries(source, limit=limit)

    YoutubeDL = _load_yt_dlp()

    source_url = str(source.get("url") or "").strip()
    if not source_url:
        return []

    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": max(1, int(limit)),
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source_url, download=False)

    raw_entries = info.get("entries") if isinstance(info, dict) else None
    if not isinstance(raw_entries, list):
        raw_entries = [info] if isinstance(info, dict) else []

    out: list[Dict[str, Any]] = []
    yt_dlp_fetch_mode = "yt_dlp_fallback" if provider in {"twitch_helix", "twitch_api"} else "yt_dlp"
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        candidate_url = str(
            item.get("webpage_url")
            or item.get("original_url")
            or item.get("url")
            or ""
        ).strip()
        if candidate_url and not candidate_url.startswith("http"):
            video_id = str(item.get("id") or "").strip()
            ie_key = str(item.get("ie_key") or item.get("extractor_key") or "").lower()
            if "youtube" in ie_key and video_id:
                candidate_url = f"https://www.youtube.com/watch?v={video_id}"
            elif "twitch" in ie_key and video_id.isdigit():
                candidate_url = f"https://www.twitch.tv/videos/{video_id}"
        if not candidate_url.startswith("http"):
            continue
        published_at = _timestamp_to_iso(item.get("timestamp") or item.get("release_timestamp"))
        out.append(
            {
                "url": candidate_url,
                "title": str(item.get("title") or "").strip(),
                "duration_seconds": _safe_float(item.get("duration"), 0.0),
                "published_at": published_at,
                "view_count": _safe_int(item.get("view_count"), 0),
                "channel_id": str(item.get("channel_id") or item.get("uploader_id") or "").strip(),
                "channel_name": str(item.get("channel") or item.get("uploader") or "").strip(),
                "video_id": str(item.get("id") or "").strip(),
                "is_live": bool(item.get("is_live")) or str(item.get("live_status") or "").lower() == "is_live",
                "platform": str(source.get("platform") or item.get("extractor_key") or "").strip().lower() or None,
                "fetch_mode": yt_dlp_fetch_mode,
            }
        )
        if len(out) >= limit:
            break
    return out


def _infer_source_fetch_mode(source: Dict[str, Any]) -> Optional[str]:
    provider = str(source.get("provider") or "").strip().lower()
    source_url = str(source.get("url") or "").strip()
    if provider in {"twitch_helix", "twitch_api"}:
        if twitch_api_configured():
            return "twitch_helix"
        if source_url and yt_dlp_available():
            return "yt_dlp_fallback"
        return None
    if source_url and yt_dlp_available():
        return "yt_dlp"
    return None


def build_project_history(projects_root: Optional[Path] = None) -> Dict[str, Any]:
    root = projects_root or default_projects_root()
    processed_urls: set[str] = set()
    processed_content_keys: set[str] = set()
    source_stats: Dict[str, Dict[str, int]] = {}

    if not root.exists():
        return {
            "processed_urls": processed_urls,
            "processed_content_keys": processed_content_keys,
            "source_stats": source_stats,
        }

    for pdir in root.iterdir():
        if not pdir.is_dir():
            continue
        pjson = pdir / "project.json"
        if not pjson.exists():
            continue
        try:
            data = json.loads(pjson.read_text(encoding="utf-8"))
        except Exception:
            continue

        source_block = data.get("source") or {}
        source_url = str(source_block.get("source_url") or data.get("source_url") or "").strip()
        if source_url:
            processed_urls.add(_normalize_url(source_url))
            processed_content_keys.add(extract_content_key(source_url))

        scout_block = source_block.get("scout") or {}
        source_id = str(scout_block.get("source_id") or "").strip()
        if not source_id:
            continue

        highlights = ((data.get("analysis") or {}).get("highlights") or {})
        candidates = [item for item in (highlights.get("candidates") or []) if isinstance(item, dict)]
        director = ((data.get("analysis") or {}).get("director") or {})
        exports = [item for item in (data.get("exports") or []) if isinstance(item, dict)]

        stats = source_stats.setdefault(
            source_id,
            {
                "project_count": 0,
                "projects_with_candidates": 0,
                "projects_with_director_picks": 0,
                "projects_with_exports": 0,
            },
        )
        stats["project_count"] += 1
        if candidates:
            stats["projects_with_candidates"] += 1
        if _safe_int(director.get("pick_count"), 0) > 0:
            stats["projects_with_director_picks"] += 1
        if any(str(item.get("status") or "").strip().lower() == "succeeded" for item in exports):
            stats["projects_with_exports"] += 1

    return {
        "processed_urls": processed_urls,
        "processed_content_keys": processed_content_keys,
        "source_stats": source_stats,
    }


def _safe_rating(value: Any) -> Optional[int]:
    try:
        rating = int(value)
    except Exception:
        return None
    if rating < 1 or rating > 5:
        return None
    return rating


def _weighted_average(parts: list[tuple[Optional[float], float]]) -> Optional[float]:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in parts:
        if value is None or weight <= 0:
            continue
        total_weight += float(weight)
        total_value += float(value) * float(weight)
    if total_weight <= 0:
        return None
    return total_value / total_weight


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _mean_top(values: list[float], *, limit: int) -> float:
    positives = sorted((float(item) for item in values if float(item) > 0.0), reverse=True)
    if not positives:
        return 0.0
    top = positives[: max(1, int(limit))]
    return sum(top) / float(len(top))


def _feature_values(features: Dict[str, Any], key: str) -> list[float]:
    raw = features.get(key)
    if raw is None:
        return []
    return [float(item) for item in list(raw)]


def _probe_cache_dir() -> Path:
    if os.name == "nt":
        base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or str(Path.home() / "AppData" / "Local")
        )
        return Path(base) / "VideoPipeline" / "scout_probe"
    return Path.home() / ".videopipeline" / "scout_probe"


def _probe_cache_key(url: str, content_key: Optional[str] = None) -> str:
    raw = str(content_key or "").strip() or extract_content_key(url)
    if not raw:
        raw = hashlib.sha1(str(url or "").encode("utf-8")).hexdigest()
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw).strip("._-") or "candidate"


def _probe_cache_paths(url: str, *, content_key: Optional[str] = None) -> Dict[str, Path]:
    cache_root = _probe_cache_dir() / _probe_cache_key(url, content_key=content_key)
    return {
        "root": cache_root,
        "summary": cache_root / "summary.json",
        "raw": cache_root / "chat.json",
        "db": cache_root / "chat.sqlite",
    }


def _load_scout_probe_config(watchlist: Dict[str, Any]) -> Dict[str, Any]:
    raw = watchlist.get("scout_probe")
    if raw is None:
        raw = watchlist.get("chat_probe")
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "shortlist": max(2, _safe_int(raw.get("shortlist"), 30)),
        "min_candidates": max(2, _safe_int(raw.get("min_candidates"), 2)),
        "rerank_weight": max(0.0, min(0.5, _safe_float(raw.get("rerank_weight"), 0.24))),
        "ttl_hours": max(1.0, _safe_float(raw.get("ttl_hours"), 18.0)),
        "hop_s": max(2.0, _safe_float(raw.get("hop_s"), 15.0)),
        "smooth_s": max(4.0, _safe_float(raw.get("smooth_s"), 45.0)),
        "peak_threshold": max(0.5, _safe_float(raw.get("peak_threshold"), 2.25)),
        "laugh_peak_threshold": max(0.5, _safe_float(raw.get("laugh_peak_threshold"), 1.75)),
        "emote_peak_threshold": max(0.5, _safe_float(raw.get("emote_peak_threshold"), 1.75)),
    }


def _load_cached_probe_summary(path: Path, *, now_ts: float, max_age_s: float) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    generated_ts = _parse_iso_ts(payload.get("generated_at"))
    if generated_ts is None:
        return None
    age_s = max(0.0, now_ts - generated_ts)
    if age_s > max_age_s:
        return None
    out = dict(payload)
    out["cached"] = True
    out["cache_age_hours"] = round(age_s / 3600.0, 2)
    return out


def _save_probe_summary(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    serializable["schema_version"] = _SCOUT_PROBE_SCHEMA_VERSION
    serializable["cached"] = False
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    return serializable


def _candidate_supports_chat_probe(candidate: Dict[str, Any]) -> bool:
    url = str(candidate.get("url") or "").strip()
    if not url.startswith("http"):
        return False
    selection_mode = str(candidate.get("selection_mode") or "").strip().lower()
    if selection_mode and selection_mode != "watchlist":
        return False
    platform = str(candidate.get("platform") or "").strip().lower()
    return platform in {"", "twitch", "youtube"}


def _store_probe_messages(
    *,
    db_path: Path,
    url: str,
    raw_path: Path,
    platform: str,
    video_id: Optional[str],
    channel_name: Optional[str],
) -> Dict[str, Any]:
    from .chat.normalize import load_chat_data, normalize_chat_messages
    from .chat.store import ChatMeta, ChatStore

    raw_data = load_chat_data(raw_path)
    messages = normalize_chat_messages(raw_data)
    if not messages:
        raise RuntimeError("no chat messages found")
    duration_ms = max((int(msg.t_ms) for msg in messages), default=0)

    with ChatStore(db_path) as store:
        store.initialize()
        store.clear_messages()
        inserted = store.insert_messages(messages)
        store.set_all_meta(
            ChatMeta(
                source_url=url,
                platform=platform,
                video_id=str(video_id or "").strip(),
                channel=str(channel_name or "").strip().lower(),
                channel_id=str(channel_name or "").strip().lower(),
                channel_name=str(channel_name or "").strip(),
                message_count=inserted,
                duration_ms=duration_ms,
            )
        )
    return {
        "message_count": int(len(messages)),
        "duration_s": max(0.0, duration_ms / 1000.0),
    }


def _compute_chat_probe_summary(
    *,
    candidate: Dict[str, Any],
    probe_config: Dict[str, Any],
    raw_path: Path,
    db_path: Path,
) -> Dict[str, Any]:
    from .chat.features import compute_chat_features
    from .chat.store import ChatStore

    store_meta = _store_probe_messages(
        db_path=db_path,
        url=str(candidate.get("url") or "").strip(),
        raw_path=raw_path,
        platform=str(candidate.get("platform") or "twitch").strip().lower() or "twitch",
        video_id=str(candidate.get("video_id") or "").strip() or None,
        channel_name=str(candidate.get("channel_name") or candidate.get("source_label") or "").strip() or None,
    )
    duration_s = max(
        1.0,
        _safe_float(candidate.get("duration_seconds"), 0.0),
        _safe_float(store_meta.get("duration_s"), 0.0),
    )

    with ChatStore(db_path) as store:
        features = compute_chat_features(
            store,
            duration_s=duration_s,
            hop_s=_safe_float(probe_config.get("hop_s"), 15.0),
            smooth_s=_safe_float(probe_config.get("smooth_s"), 45.0),
        )

    scores = _feature_values(features, "scores")
    laugh_scores = _feature_values(features, "scores_laugh")
    emote_scores = _feature_values(features, "scores_emote")
    messages_total = max(0, _safe_int(features.get("messages_total"), store_meta.get("message_count", 0)))

    peak_threshold = _safe_float(probe_config.get("peak_threshold"), 2.25)
    laugh_threshold = _safe_float(probe_config.get("laugh_peak_threshold"), 1.75)
    emote_threshold = _safe_float(probe_config.get("emote_peak_threshold"), 1.75)

    peak_count = sum(1 for item in scores if item >= peak_threshold)
    laugh_peak_count = sum(1 for item in laugh_scores if item >= laugh_threshold)
    emote_peak_count = sum(1 for item in emote_scores if item >= emote_threshold)
    top_composite_mean = _mean_top(scores, limit=6)
    top_laugh_mean = _mean_top(laugh_scores, limit=4)
    top_emote_mean = _mean_top(emote_scores, limit=4)
    messages_per_minute = (messages_total * 60.0 / duration_s) if duration_s > 0 else 0.0

    activity_signal = _clamp01(top_composite_mean / 4.0)
    peak_signal = _clamp01(peak_count / 6.0)
    laugh_signal = max(_clamp01(top_laugh_mean / 4.0), _clamp01(laugh_peak_count / 4.0))
    emote_signal = max(_clamp01(top_emote_mean / 4.0), _clamp01(emote_peak_count / 4.0))
    chatter_signal = _clamp01(messages_per_minute / 220.0)

    probe_score = _weighted_average(
        [
            (activity_signal, 0.35),
            (peak_signal, 0.25),
            (laugh_signal, 0.20),
            (emote_signal, 0.12),
            (chatter_signal, 0.08),
        ]
    )
    if probe_score is None:
        probe_score = 0.0

    return {
        "status": "ok",
        "method": "chat_excitement_probe",
        "score": round(probe_score, 4),
        "messages_total": messages_total,
        "duration_s": round(duration_s, 2),
        "messages_per_minute": round(messages_per_minute, 2),
        "peak_count": int(peak_count),
        "laugh_peak_count": int(laugh_peak_count),
        "emote_peak_count": int(emote_peak_count),
        "top_composite_mean": round(top_composite_mean, 3),
        "top_laugh_mean": round(top_laugh_mean, 3),
        "top_emote_mean": round(top_emote_mean, 3),
        "signals": {
            "activity": round(activity_signal, 4),
            "peaks": round(peak_signal, 4),
            "laughter": round(laugh_signal, 4),
            "emotes": round(emote_signal, 4),
            "chatter": round(chatter_signal, 4),
        },
    }


def _probe_single_candidate_chat(
    candidate: Dict[str, Any],
    *,
    probe_config: Dict[str, Any],
    now_ts: float,
) -> Dict[str, Any]:
    ttl_s = max(3600.0, _safe_float(probe_config.get("ttl_hours"), 18.0) * 3600.0)
    paths = _probe_cache_paths(
        str(candidate.get("url") or "").strip(),
        content_key=str(candidate.get("content_key") or "").strip() or None,
    )
    cached = _load_cached_probe_summary(paths["summary"], now_ts=now_ts, max_age_s=ttl_s)
    if cached is not None:
        return cached

    try:
        from .chat.downloader import download_chat, is_chat_download_available
    except Exception as exc:
        payload = {
            "status": "error",
            "generated_at": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
            "error": f"chat probe import failed: {exc}",
        }
        return _save_probe_summary(paths["summary"], payload)

    if not is_chat_download_available():
        payload = {
            "status": "error",
            "generated_at": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
            "error": "chat download tools unavailable",
        }
        return _save_probe_summary(paths["summary"], payload)

    try:
        paths["root"].mkdir(parents=True, exist_ok=True)
        if paths["raw"].exists():
            paths["raw"].unlink()
        if paths["db"].exists():
            paths["db"].unlink()
        download_chat(str(candidate.get("url") or "").strip(), paths["raw"])
        summary = _compute_chat_probe_summary(
            candidate=candidate,
            probe_config=probe_config,
            raw_path=paths["raw"],
            db_path=paths["db"],
        )
        summary["generated_at"] = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()
        return _save_probe_summary(paths["summary"], summary)
    except Exception as exc:
        payload = {
            "status": "error",
            "generated_at": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        return _save_probe_summary(paths["summary"], payload)


def _apply_chat_probe_rerank(
    candidates: list[Dict[str, Any]],
    *,
    watchlist: Dict[str, Any],
    now_ts: float,
    probe_candidates_fn: Optional[Callable[..., Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    probe_config = _load_scout_probe_config(watchlist)
    meta = {
        "enabled": bool(probe_config.get("enabled", True)),
        "eligible_candidates": 0,
        "shortlist_count": 0,
        "used_count": 0,
        "status": "disabled",
    }

    for item in candidates:
        item.setdefault("base_score", round(_safe_float(item.get("score"), 0.0), 4))

    if not meta["enabled"]:
        return meta

    eligible = [item for item in candidates if _candidate_supports_chat_probe(item)]
    meta["eligible_candidates"] = len(eligible)
    if len(eligible) < max(2, _safe_int(probe_config.get("min_candidates"), 2)):
        meta["status"] = "not_enough_candidates"
        return meta

    shortlist = eligible[: max(2, _safe_int(probe_config.get("shortlist"), 4))]
    meta["shortlist_count"] = len(shortlist)
    if len(shortlist) < 2:
        meta["status"] = "not_enough_candidates"
        return meta

    if probe_candidates_fn is None:
        def _default_probe(
            selected: list[Dict[str, Any]],
            *,
            probe_config: Dict[str, Any],
            now_ts: float,
        ) -> Dict[str, Dict[str, Any]]:
            return {
                str(item.get("url") or "").strip(): _probe_single_candidate_chat(
                    item,
                    probe_config=probe_config,
                    now_ts=now_ts,
                )
                for item in selected
            }

        probe_candidates_fn = _default_probe

    probe_results = probe_candidates_fn(shortlist, probe_config=probe_config, now_ts=now_ts) or {}
    rerank_weight = _safe_float(probe_config.get("rerank_weight"), 0.24)
    used_count = 0
    for item in shortlist:
        key = str(item.get("url") or "").strip()
        probe = probe_results.get(key)
        if not isinstance(probe, dict):
            continue
        item["chat_probe"] = probe
        if str(probe.get("status") or "").strip().lower() != "ok":
            error = str(probe.get("error") or "").strip()
            if error:
                item.setdefault("reasons", []).append(f"chat probe unavailable: {error}")
            continue
        probe_score = _clamp01(probe.get("score"))
        base_score = _safe_float(item.get("base_score"), item.get("score"))
        score_adjustment = (probe_score - 0.5) * rerank_weight
        item["score"] = round(base_score + score_adjustment, 4)
        item.setdefault("reasons", []).append(
            "chat probe compared multiple VODs: "
            f"score={probe_score:.2f} peaks={_safe_int(probe.get('peak_count'), 0)} "
            f"laugh_peaks={_safe_int(probe.get('laugh_peak_count'), 0)}"
        )
        probe["score_adjustment"] = round(score_adjustment, 4)
        probe["base_score"] = round(base_score, 4)
        probe["rerank_weight"] = round(rerank_weight, 4)
        used_count += 1

    meta["used_count"] = used_count
    meta["status"] = "applied" if used_count > 0 else "unavailable"
    return meta


def _positive_rating_score(rating: Optional[int]) -> Optional[float]:
    if rating is None:
        return None
    return (float(rating) - 1.0) / 4.0


def _inverse_rating_score(rating: Optional[int]) -> Optional[float]:
    if rating is None:
        return None
    return (5.0 - float(rating)) / 4.0


def _source_profile_input(source: Dict[str, Any]) -> Dict[str, Any]:
    profile = source.get("profile") or {}
    if not isinstance(profile, dict):
        profile = {}
    judgments = profile.get("judgments") or {}
    if isinstance(judgments, dict):
        merged = dict(profile)
        for key, value in judgments.items():
            merged.setdefault(key, value)
        return merged
    return dict(profile)


def build_source_profile(source: Dict[str, Any], history: Dict[str, Any]) -> Dict[str, Any]:
    source_id = str(source.get("id") or "").strip()
    source_stats = ((history.get("source_stats") or {}).get(source_id) or {}) if source_id else {}
    projects = _safe_int(source_stats.get("project_count"), 0)
    candidate_projects = _safe_int(source_stats.get("projects_with_candidates"), 0)
    director_projects = _safe_int(source_stats.get("projects_with_director_picks"), 0)
    export_projects = _safe_int(source_stats.get("projects_with_exports"), 0)

    candidate_rate = (candidate_projects / projects) if projects else None
    director_rate = (director_projects / projects) if projects else None
    export_rate = (export_projects / projects) if projects else None
    objective_score = _weighted_average(
        [
            (candidate_rate, 0.35),
            (director_rate, 0.35),
            (export_rate, 0.30),
        ]
    )
    confidence = min(1.0, (projects / 6.0)) if projects else 0.0

    profile_cfg = _source_profile_input(source)
    clip_density_rating = _safe_rating(profile_cfg.get("clip_density_rating"))
    style_fit_rating = _safe_rating(profile_cfg.get("style_fit_rating"))
    saturation_rating = _safe_rating(profile_cfg.get("saturation_rating"))
    rights_risk_rating = _safe_rating(profile_cfg.get("rights_risk_rating"))
    judgment_score = _weighted_average(
        [
            (_positive_rating_score(clip_density_rating), 0.35),
            (_positive_rating_score(style_fit_rating), 0.25),
            (_inverse_rating_score(saturation_rating), 0.20),
            (_inverse_rating_score(rights_risk_rating), 0.20),
        ]
    )
    composite_signal = _weighted_average(
        [
            (objective_score, 0.55),
            (judgment_score, 0.45),
        ]
    )

    band = "unscored"
    if rights_risk_rating is not None and rights_risk_rating >= 5:
        band = "blocked"
    elif composite_signal is not None:
        if composite_signal >= 0.75:
            band = "high"
        elif composite_signal >= 0.55:
            band = "medium"
        else:
            band = "low"
        if rights_risk_rating is not None and rights_risk_rating >= 4 and band == "high":
            band = "medium"
        if saturation_rating is not None and saturation_rating >= 5 and band == "high":
            band = "medium"

    reasons: list[str] = []
    if objective_score is not None:
        reasons.append(f"history objective_score={objective_score:.2f}")
    else:
        reasons.append("history objective_score unavailable")
    if projects:
        reasons.append(f"history confidence={confidence:.2f} from {projects} project(s)")
    if export_rate is not None:
        reasons.append(f"export success rate={export_rate:.2f}")
    if clip_density_rating is not None:
        reasons.append(f"clip density rated {clip_density_rating}/5")
    if style_fit_rating is not None:
        reasons.append(f"style fit rated {style_fit_rating}/5")
    if saturation_rating is not None:
        reasons.append(f"saturation rated {saturation_rating}/5")
    if rights_risk_rating is not None:
        reasons.append(f"rights risk rated {rights_risk_rating}/5")
    if rights_risk_rating is not None and rights_risk_rating >= 4:
        reasons.append("higher rights risk requires extra caution")
    if saturation_rating is not None and saturation_rating >= 4:
        reasons.append("high saturation means stronger competition")

    return {
        "category": str(profile_cfg.get("category") or "").strip() or None,
        "metrics": {
            "project_count": projects,
            "projects_with_candidates": candidate_projects,
            "projects_with_director_picks": director_projects,
            "projects_with_exports": export_projects,
            "candidate_hit_rate": round(candidate_rate, 4) if candidate_rate is not None else None,
            "director_pick_rate": round(director_rate, 4) if director_rate is not None else None,
            "export_success_rate": round(export_rate, 4) if export_rate is not None else None,
            "objective_score": round(objective_score, 4) if objective_score is not None else None,
            "confidence": round(confidence, 4),
        },
        "judgments": {
            "operator_priority": _safe_float(source.get("priority"), 1.0),
            "clip_density_rating": clip_density_rating,
            "style_fit_rating": style_fit_rating,
            "saturation_rating": saturation_rating,
            "rights_risk_rating": rights_risk_rating,
            "judgment_score": round(judgment_score, 4) if judgment_score is not None else None,
            "rated_by": str(profile_cfg.get("rated_by") or "").strip() or None,
            "rated_at": str(profile_cfg.get("rated_at") or "").strip() or None,
            "notes": str(profile_cfg.get("notes") or "").strip() or None,
        },
        "recommendation": {
            "band": band,
            "composite_signal": round(composite_signal, 4) if composite_signal is not None else None,
            "reasons": reasons,
        },
    }


def _score_candidate(
    *,
    source: Dict[str, Any],
    entry: Dict[str, Any],
    history: Dict[str, Any],
    now_ts: float,
    source_profile: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = _normalize_url(str(entry.get("url") or ""))
    if not url:
        return None, "missing_url"

    if bool(entry.get("is_live")):
        return None, "is_live"

    content_key = extract_content_key(url)
    processed_urls = history.get("processed_urls") or set()
    processed_content_keys = history.get("processed_content_keys") or set()
    if url in processed_urls or (content_key and content_key in processed_content_keys):
        return None, "already_processed"

    title = str(entry.get("title") or "").strip()
    title_lower = title.lower()
    include_words = [str(item).strip() for item in list(source.get("title_include_any") or []) if str(item).strip()]
    exclude_words = [str(item).strip() for item in list(source.get("title_exclude_any") or []) if str(item).strip()]
    include_matches = _match_any(title_lower, include_words)
    exclude_matches = _match_any(title_lower, exclude_words)
    if exclude_matches:
        return None, "title_excluded"

    min_duration_s = max(0.0, _safe_float(source.get("min_duration_s"), 0.0))
    max_duration_s = max(0.0, _safe_float(source.get("max_duration_s"), 0.0))
    if max_duration_s > 0 and max_duration_s < min_duration_s:
        max_duration_s = min_duration_s
    duration_s = _safe_float(entry.get("duration_seconds"), 0.0)
    if min_duration_s > 0 and duration_s > 0 and duration_s < min_duration_s:
        return None, "too_short"
    if max_duration_s > 0 and duration_s > 0 and duration_s > max_duration_s:
        return None, "too_long"

    recent_hours = max(1.0, _safe_float(source.get("recent_hours"), 72.0))
    age_hours = _age_hours(str(entry.get("published_at") or "").strip() or None, now_ts=now_ts)
    if age_hours is None:
        recency_score = 0.5
    else:
        recency_score = max(0.0, min(1.0, 1.0 - (age_hours / recent_hours)))

    if duration_s <= 0 or (min_duration_s <= 0 and max_duration_s <= 0):
        duration_score = 0.5
    elif min_duration_s <= duration_s <= max_duration_s:
        duration_score = 1.0
    else:
        duration_score = 0.1

    if include_words:
        title_score = 0.4
        if include_matches:
            title_score = min(1.0, 0.4 + (0.2 * len(include_matches)))
    else:
        title_score = 0.6

    priority_raw = _safe_float(source.get("priority"), 1.0)
    priority_score = max(0.0, min(1.0, priority_raw / 5.0))

    history_score = 0.5
    source_id = str(source.get("id") or "").strip()
    source_stats = ((history.get("source_stats") or {}).get(source_id) or {}) if source_id else {}
    profile_metrics = ((source_profile or {}).get("metrics") or {}) if isinstance(source_profile, dict) else {}
    profile_recommendation = (
        ((source_profile or {}).get("recommendation") or {}) if isinstance(source_profile, dict) else {}
    )
    objective_score = profile_metrics.get("objective_score")
    if objective_score is not None:
        history_score = max(0.0, min(1.0, _safe_float(objective_score, 0.5)))
    elif source_stats:
        projects = max(1, _safe_int(source_stats.get("project_count"), 1))
        history_score = min(
            1.0,
            (
                (_safe_int(source_stats.get("projects_with_candidates"), 0) / projects) * 0.35
                + (_safe_int(source_stats.get("projects_with_director_picks"), 0) / projects) * 0.35
                + (_safe_int(source_stats.get("projects_with_exports"), 0) / projects) * 0.30
            ),
        )

    score = (
        (priority_score * 0.30)
        + (duration_score * 0.20)
        + (recency_score * 0.20)
        + (title_score * 0.15)
        + (history_score * 0.15)
    )
    reasons = [
        f"source priority {priority_raw:g}",
        f"duration_score={duration_score:.2f}",
        f"recency_score={recency_score:.2f}",
        f"title_score={title_score:.2f}",
        f"history_score={history_score:.2f}",
    ]
    if include_matches:
        reasons.append(f"title matched preferred words: {', '.join(include_matches)}")
    if age_hours is not None:
        reasons.append(f"age_hours={age_hours:.1f}")
    band = str(profile_recommendation.get("band") or "").strip()
    if band and band != "unscored":
        reasons.append(f"source profile band={band}")
    judgments = ((source_profile or {}).get("judgments") or {}) if isinstance(source_profile, dict) else {}
    clip_density_rating = judgments.get("clip_density_rating")
    if clip_density_rating is not None:
        reasons.append(f"clip density rating={clip_density_rating}/5")
    saturation_rating = judgments.get("saturation_rating")
    if saturation_rating is not None:
        reasons.append(f"saturation rating={saturation_rating}/5")
    rights_risk_rating = judgments.get("rights_risk_rating")
    if rights_risk_rating is not None:
        reasons.append(f"rights risk rating={rights_risk_rating}/5")

    return (
        {
            "url": url,
            "content_key": content_key or None,
            "source_id": source_id or None,
            "source_label": str(source.get("label") or source_id or source.get("url") or "").strip(),
            "source_url": str(source.get("url") or "").strip(),
            "source_fetch_mode": str(entry.get("fetch_mode") or "").strip() or None,
            "platform": str(source.get("platform") or entry.get("platform") or "").strip() or None,
            "score": round(score, 4),
            "title": title,
            "duration_seconds": duration_s or None,
            "published_at": str(entry.get("published_at") or "").strip() or None,
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
            "view_count": _safe_int(entry.get("view_count"), 0) or None,
            "channel_id": str(entry.get("channel_id") or "").strip() or None,
            "channel_name": str(entry.get("channel_name") or "").strip() or None,
            "video_id": str(entry.get("video_id") or "").strip() or None,
            "tags": [str(item).strip() for item in list(source.get("tags") or []) if str(item).strip()],
            "reasons": reasons,
            "history": {
                "source_id": source_id or None,
                "source_project_count": _safe_int(source_stats.get("project_count"), 0),
                "projects_with_candidates": _safe_int(source_stats.get("projects_with_candidates"), 0),
                "projects_with_director_picks": _safe_int(source_stats.get("projects_with_director_picks"), 0),
                "projects_with_exports": _safe_int(source_stats.get("projects_with_exports"), 0),
            },
            "source_profile": source_profile,
        },
        None,
    )


def build_source_scout_report(
    *,
    watchlist_path: Optional[Path] = None,
    per_source: int = 20,
    limit: int = 60,
    now_ts: Optional[float] = None,
    fetch_entries_fn: Optional[Callable[..., list[Dict[str, Any]]]] = None,
    probe_candidates_fn: Optional[Callable[..., Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    resolved_path, watchlist = load_source_watchlist(watchlist_path)
    now_ts = float(now_ts if now_ts is not None else time.time())
    history = build_project_history()
    inbox_path, inbox_entries = list_source_inbox_entries(status="pending")

    per_source = max(1, int(per_source))
    limit = max(1, int(limit))
    fetch_entries_fn = fetch_entries_fn or fetch_source_entries

    enabled_sources = [
        item for item in (watchlist.get("sources") or []) if bool(item.get("enabled", True))
    ]
    source_profiles: Dict[str, Dict[str, Any]] = {}
    for source in enabled_sources:
        source_id = str(source.get("id") or "").strip()
        if not source_id:
            continue
        source_profiles[source_id] = build_source_profile(source, history)

    source_reports: list[Dict[str, Any]] = []
    candidates: list[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}

    for entry in inbox_entries:
        url = _normalize_url(str(entry.get("url") or ""))
        if not url:
            skipped["manual_missing_url"] = skipped.get("manual_missing_url", 0) + 1
            continue
        content_key = extract_content_key(url)
        processed_urls = history.get("processed_urls") or set()
        processed_content_keys = history.get("processed_content_keys") or set()
        if url in processed_urls or (content_key and content_key in processed_content_keys):
            skipped["already_processed"] = skipped.get("already_processed", 0) + 1
            continue
        priority_raw = max(1.0, _safe_float(entry.get("priority"), 5.0))
        source_id = str(entry.get("source_id") or "").strip() or "manual_inbox"
        source_stats = ((history.get("source_stats") or {}).get(source_id) or {})
        history_score = 0.5
        if source_stats:
            projects = max(1, _safe_int(source_stats.get("project_count"), 1))
            history_score = min(
                1.0,
                (
                    (_safe_int(source_stats.get("projects_with_candidates"), 0) / projects) * 0.35
                    + (_safe_int(source_stats.get("projects_with_director_picks"), 0) / projects) * 0.35
                    + (_safe_int(source_stats.get("projects_with_exports"), 0) / projects) * 0.30
                ),
            )
        score = min(1.2, 0.95 + min(0.20, priority_raw * 0.03) + (history_score * 0.05))
        reasons = [
            f"manual inbox priority {priority_raw:g}",
            "user/app explicitly queued this URL",
        ]
        notes = str(entry.get("notes") or "").strip()
        if notes:
            reasons.append(f"notes: {notes}")
        added_by = str(entry.get("added_by") or "").strip()
        if added_by:
            reasons.append(f"added_by={added_by}")
        tags = [str(item).strip() for item in list(entry.get("tags") or []) if str(item).strip()]
        candidates.append(
            {
                "url": url,
                "content_key": content_key or None,
                "source_id": None if source_id == "manual_inbox" else source_id,
                "source_label": str(entry.get("source_label") or "Manual Inbox").strip() or "Manual Inbox",
                "source_url": None,
                "platform": None,
                "score": round(score, 4),
                "title": str(entry.get("title") or "").strip(),
                "duration_seconds": None,
                "published_at": None,
                "age_hours": None,
                "view_count": None,
                "channel_id": None,
                "channel_name": None,
                "video_id": None,
                "tags": tags,
                "reasons": reasons,
                "history": {
                    "source_id": None if source_id == "manual_inbox" else source_id,
                    "source_project_count": _safe_int(source_stats.get("project_count"), 0),
                    "projects_with_candidates": _safe_int(source_stats.get("projects_with_candidates"), 0),
                    "projects_with_director_picks": _safe_int(source_stats.get("projects_with_director_picks"), 0),
                    "projects_with_exports": _safe_int(source_stats.get("projects_with_exports"), 0),
                },
                "selection_mode": "manual_inbox",
                "inbox_id": str(entry.get("inbox_id") or "").strip() or None,
                "added_at": str(entry.get("added_at") or "").strip() or None,
                "added_by": added_by or None,
                "notes": notes or None,
                "source_priority": priority_raw,
                "source_fetch_mode": None,
            }
        )

    for source in enabled_sources:
        source_id = str(source.get("id") or "").strip()
        source_limit = max(1, min(per_source, _safe_int(source.get("max_candidates"), per_source)))
        source_report = {
            "id": source_id or None,
            "label": str(source.get("label") or source_id or source.get("url") or "").strip(),
            "url": str(source.get("url") or "").strip(),
            "platform": str(source.get("platform") or "").strip() or None,
            "fetch_mode": _infer_source_fetch_mode(source),
            "priority": _safe_float(source.get("priority"), 1.0),
            "recent_hours": _safe_float(source.get("recent_hours"), 72.0),
            "max_candidates": source_limit,
            "tags": [str(item).strip() for item in list(source.get("tags") or []) if str(item).strip()],
            "status": "ok",
            "error": None,
            "profile": source_profiles.get(source_id) if source_id else None,
        }
        try:
            raw_entries = fetch_entries_fn(source, limit=source_limit)
        except Exception as exc:
            source_report["status"] = "error"
            source_report["error"] = f"{type(exc).__name__}: {exc}"
            source_reports.append(source_report)
            continue

        observed_fetch_modes = {
            str(item.get("fetch_mode") or "").strip()
            for item in raw_entries
            if isinstance(item, dict) and str(item.get("fetch_mode") or "").strip()
        }
        if len(observed_fetch_modes) == 1:
            source_report["fetch_mode"] = next(iter(observed_fetch_modes))
        elif len(observed_fetch_modes) > 1:
            source_report["fetch_mode"] = "mixed"

        source_count = 0
        for entry in raw_entries:
            candidate, skip_reason = _score_candidate(
                source=source,
                entry=entry,
                history=history,
                now_ts=now_ts,
                source_profile=source_profiles.get(source_id) if source_id else None,
            )
            if candidate is None:
                if skip_reason:
                    skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
                continue
            candidate.setdefault("selection_mode", "watchlist")
            candidate.setdefault("inbox_id", None)
            candidate.setdefault("added_at", None)
            candidate.setdefault("added_by", None)
            candidate.setdefault("notes", None)
            candidate["source_priority"] = source_report["priority"]
            candidates.append(candidate)
            source_count += 1
        source_report["eligible_candidates"] = source_count
        source_reports.append(source_report)

    candidates.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            -_safe_float(item.get("age_hours"), 0.0),
            str(item.get("title") or ""),
        ),
        reverse=True,
    )

    probe_meta = _apply_chat_probe_rerank(
        candidates,
        watchlist=watchlist,
        now_ts=now_ts,
        probe_candidates_fn=probe_candidates_fn,
    )

    candidates.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            -_safe_float(item.get("age_hours"), 0.0),
            str(item.get("title") or ""),
        ),
        reverse=True,
    )

    ranked: list[Dict[str, Any]] = []
    for idx, item in enumerate(candidates[:limit], start=1):
        ranked.append({"rank": idx, **item})

    return {
        "meta": {
            "generated_at": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
            "watchlist_path": str(resolved_path) if resolved_path is not None else None,
            "shadow_mode": bool(watchlist.get("shadow_mode", True)),
            "enabled_source_count": len(enabled_sources),
            "candidate_count": len(ranked),
            "chat_probe": probe_meta,
        },
        "strategy": {
            "mode": "shadow" if bool(watchlist.get("shadow_mode", True)) else "auto_ready",
            "per_source": per_source,
            "limit": limit,
            "ranking_factors": [
                "manual inbox priority",
                "source priority",
                "duration fit",
                "recency",
                "title cues",
                "source hit history",
                "chat excitement probe across shortlisted VODs",
            ],
            "chat_probe": _load_scout_probe_config(watchlist),
            "source_profile_model": {
                "objective_metrics": [
                    "candidate_hit_rate",
                    "director_pick_rate",
                    "export_success_rate",
                ],
                "manual_judgments": [
                    "clip_density_rating",
                    "style_fit_rating",
                    "saturation_rating",
                    "rights_risk_rating",
                ],
            },
            "dedupe": {
                "processed_url_count": len(history.get("processed_urls") or set()),
                "processed_content_count": len(history.get("processed_content_keys") or set()),
            },
            "manual_inbox": {
                "inbox_path": str(inbox_path) if inbox_path is not None else None,
                "pending_count": len(inbox_entries),
            },
        },
        "sources": source_reports,
        "skipped": skipped,
        "recommended": ranked[0] if ranked else None,
        "candidates": ranked,
    }
