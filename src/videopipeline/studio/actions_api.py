from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from ..analysis import run_analysis
from ..ai.helpers import get_llm_complete_fn
from ..ingest import IngestRequest, QualityCap, SpeedMode
from ..ingest.ytdlp_runner import DownloadCancelled, download_url
from ..project import (
    Project,
    create_project_early,
    default_projects_root,
    get_chat_config,
    get_project_data,
    save_json,
    set_project_status,
    set_source_url,
)
from .jobs import JOB_MANAGER, with_prevent_sleep


_PROJECT_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_YOUTUBE_ID_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)")
_TWITCH_VOD_ID_RE = re.compile(r"twitch\.tv/videos/(\d+)")


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bearer_token_from_request(request: Request) -> Optional[str]:
    raw = (request.headers.get("authorization") or "").strip()
    if not raw:
        return None
    parts = raw.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0], parts[1]
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def _request_actor_key(request: Request) -> str:
    token = _bearer_token_from_request(request)
    if token:
        return f"t:{_hash_key(token)}"
    # Best-effort fallback for local/no-auth setups.
    host = getattr(getattr(request, "client", None), "host", None) or "unknown"
    return f"ip:{host}"


def _clamp_int(value: Any, *, default: int, min_v: int, max_v: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = int(default)
    return max(min_v, min(max_v, n))


def _extract_content_id(url: str) -> str:
    """Mirror Studio's URL->content_id behavior for deterministic project IDs."""
    u = url.strip()
    m = _TWITCH_VOD_ID_RE.search(u)
    if m:
        return f"twitch_{m.group(1)}"
    m = _YOUTUBE_ID_RE.search(u)
    if m:
        return f"youtube_{m.group(1)}"
    return u


def _validate_project_id(project_id: str) -> str:
    pid = (project_id or "").strip().lower()
    if not _PROJECT_ID_RE.match(pid):
        raise HTTPException(status_code=400, detail="invalid_project_id")
    return pid


def _load_project(project_id: str) -> Tuple[Project, Dict[str, Any]]:
    pid = _validate_project_id(project_id)
    project_dir = default_projects_root() / pid
    project_json = project_dir / "project.json"
    if not project_json.exists():
        raise HTTPException(status_code=404, detail="project_not_found")

    data = json.loads(project_json.read_text(encoding="utf-8"))
    video_path_raw = (data.get("video", {}) or {}).get("path")
    video_path: Path
    if video_path_raw:
        video_path = Path(str(video_path_raw))
    else:
        # Fallback for partially-ingested projects.
        candidate = project_dir / "video" / "video.mp4"
        video_path = candidate if candidate.exists() else (project_dir / "video_pending")

    return Project(project_dir=project_dir, video_path=video_path), data


def _is_private_ip(hostname: str) -> bool:
    import ipaddress

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _normalize_host(h: str) -> str:
    """Lowercase, strip trailing dots, strip leading 'www.'."""
    h = h.strip().lower().rstrip(".")
    if h.startswith("www."):
        h = h[4:]
    return h


def _validate_ingest_url(url: str, *, allow_domains: set[str]) -> str:
    u = (url or "").strip()
    if not u:
        raise HTTPException(status_code=400, detail="url_required")

    parsed = urlsplit(u)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="invalid_url_scheme")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise HTTPException(status_code=400, detail="invalid_url_host")

    if hostname in {"localhost"} or hostname.endswith(".local"):
        raise HTTPException(status_code=400, detail="invalid_url_host")
    if _is_private_ip(hostname):
        raise HTTPException(status_code=400, detail="invalid_url_host")

    norm = _normalize_host(hostname)
    allowed = False
    for dom in allow_domains:
        nd = _normalize_host(dom)
        if not nd:
            continue
        if norm == nd or norm.endswith("." + nd):
            allowed = True
            break
    if not allowed:
        raise HTTPException(status_code=400, detail="url_domain_not_allowed")

    return u


@dataclass
class _TokenBucket:
    capacity: float
    refill_per_s: float
    tokens: float
    updated_at: float

    @classmethod
    def new(cls, *, capacity: float, refill_per_s: float) -> "_TokenBucket":
        now = time.monotonic()
        return cls(capacity=capacity, refill_per_s=refill_per_s, tokens=capacity, updated_at=now)

    def consume(self, amount: float = 1.0) -> Tuple[bool, float]:
        now = time.monotonic()
        elapsed = max(0.0, now - self.updated_at)
        if self.refill_per_s > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_s)
        self.updated_at = now

        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0

        if self.refill_per_s <= 0:
            return False, 1.0

        deficit = amount - self.tokens
        retry_after_s = deficit / self.refill_per_s
        return False, max(0.05, retry_after_s)


class _RateLimiter:
    def __init__(
        self,
        *,
        per_actor_capacity: float,
        per_actor_refill_per_s: float,
        global_capacity: float,
        global_refill_per_s: float,
    ) -> None:
        import threading

        self._lock = threading.Lock()
        self._per_actor: Dict[str, _TokenBucket] = {}
        self._global = _TokenBucket.new(capacity=global_capacity, refill_per_s=global_refill_per_s)
        self._per_actor_capacity = per_actor_capacity
        self._per_actor_refill_per_s = per_actor_refill_per_s

    def allow(self, actor_key: str) -> Tuple[bool, float]:
        with self._lock:
            bucket = self._per_actor.get(actor_key)
            if bucket is None:
                bucket = _TokenBucket.new(
                    capacity=self._per_actor_capacity, refill_per_s=self._per_actor_refill_per_s
                )
                self._per_actor[actor_key] = bucket

            ok_g, ra_g = self._global.consume(1.0)
            ok_a, ra_a = bucket.consume(1.0)
            if ok_g and ok_a:
                return True, 0.0

            # Roll-forward: no refunds (simpler, discourages thundering herd).
            return False, max(ra_g, ra_a)


class _IdempotencyCache:
    def __init__(self, *, ttl_s: float) -> None:
        import threading

        self._ttl_s = float(ttl_s)
        self._lock = threading.Lock()
        self._items: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}

    def get(self, actor_key: str, client_request_id: str) -> Optional[Dict[str, Any]]:
        if not client_request_id:
            return None
        key = (actor_key, client_request_id)
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            created_at, payload = item
            if now - created_at > self._ttl_s:
                self._items.pop(key, None)
                return None
            return dict(payload)

    def set(self, actor_key: str, client_request_id: str, payload: Dict[str, Any]) -> None:
        if not client_request_id:
            return
        key = (actor_key, client_request_id)
        with self._lock:
            self._items[key] = (time.monotonic(), dict(payload))


def create_actions_router(*, profile: Dict[str, Any]) -> APIRouter:
    router = APIRouter(prefix="/api/actions", tags=["actions"])

    _default_domains = {
        "twitch.tv",
        "youtube.com",
        "youtu.be",
    }
    _extra = os.environ.get("VP_ACTIONS_ALLOW_DOMAINS", "").strip()
    allow_domains: set[str] = set(_default_domains)
    if _extra:
        for d in _extra.replace(",", " ").split():
            d = _normalize_host(d)
            if d:
                allow_domains.add(d)
    # Normalise all entries once at startup.
    allow_domains = {_normalize_host(d) for d in allow_domains if _normalize_host(d)}

    def _env_float(name: str, default: float) -> float:
        raw = (os.environ.get(name) or "").strip()
        if raw == "":
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    per_actor_capacity = _env_float("VP_ACTIONS_RL_PER_ACTOR_CAPACITY", 60.0)
    per_actor_refill_per_s = _env_float("VP_ACTIONS_RL_PER_ACTOR_REFILL_PER_S", 1.0)
    global_capacity = _env_float("VP_ACTIONS_RL_GLOBAL_CAPACITY", 120.0)
    global_refill_per_s = _env_float("VP_ACTIONS_RL_GLOBAL_REFILL_PER_S", 2.0)

    limiter = _RateLimiter(
        per_actor_capacity=per_actor_capacity,
        per_actor_refill_per_s=per_actor_refill_per_s,
        global_capacity=global_capacity,
        global_refill_per_s=global_refill_per_s,
    )
    idem = _IdempotencyCache(ttl_s=_env_float("VP_ACTIONS_IDEMPOTENCY_TTL_S", 1800.0))
    last_run_by_actor: Dict[str, str] = {}

    def _rate_limit(request: Request) -> str:
        actor_key = _request_actor_key(request)
        ok, retry_after_s = limiter.allow(actor_key)
        if not ok:
            raise HTTPException(
                status_code=429,
                detail="rate_limited",
                headers={"Retry-After": str(int(retry_after_s + 0.999))},
            )
        return actor_key

    @router.get("/health", openapi_extra={"x-openai-isConsequential": False})
    def health(request: Request):
        _rate_limit(request)
        return JSONResponse({"ok": True})

    @router.get("/openapi.json", openapi_extra={"x-openai-isConsequential": False})
    def actions_openapi(request: Request):
        _rate_limit(request)
        def _server_url() -> str:
            # Prefer proxy headers when Studio is exposed via a TLS tunnel (ngrok/Cloudflare).
            xf_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
            xf_host = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
            host = (xf_host or request.headers.get("host") or request.url.netloc or "").strip()
            scheme = (xf_proto or request.url.scheme or "http").strip()
            if host:
                return f"{scheme}://{host}"
            return str(request.base_url).rstrip("/")

        # Keep this intentionally small and Actions-focused.
        spec: Dict[str, Any] = {
            "openapi": "3.1.0",
            "info": {
                "title": "VideoPipeline Actions API",
                "version": "1.0.0",
                "description": "Operator API for orchestrating VideoPipeline Studio via ChatGPT Actions.",
            },
            "servers": [{"url": _server_url()}],
            "components": {
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                    }
                },
                "schemas": {
                    "Error": {
                        "type": "object",
                        "properties": {"detail": {"type": "string"}},
                        "additionalProperties": True,
                    },
                    "IngestUrlRequest": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "options": {"type": "object", "additionalProperties": True},
                            "client_request_id": {"type": "string"},
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                    "IngestUrlResponse": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string"},
                            "project_id": {"type": "string"},
                        },
                        "required": ["job_id", "project_id"],
                        "additionalProperties": False,
                    },
                    "AnalyzeFullRequest": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "overrides": {"type": "object", "additionalProperties": True},
                            "client_request_id": {"type": "string"},
                        },
                        "required": ["project_id"],
                        "additionalProperties": False,
                    },
                    "AnalyzeFullResponse": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string"},
                            "project_id": {"type": "string"},
                        },
                        "required": ["job_id", "project_id"],
                        "additionalProperties": False,
                    },
                    "ExportBatchRequest": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "selection_ids": {"type": "array", "items": {"type": "string"}},
                            "top": {"type": "integer"},
                            "export": {"type": "object", "additionalProperties": True},
                            "captions": {"type": "object", "additionalProperties": True},
                            "hook_text": {"type": "object", "additionalProperties": True},
                            "pip": {"type": "object", "additionalProperties": True},
                            "with_captions": {"type": "boolean"},
                            "client_request_id": {"type": "string"},
                        },
                        "required": ["project_id"],
                        "additionalProperties": False,
                    },
                    "ExportBatchResponse": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string"},
                            "project_id": {"type": "string"},
                            "created_selection_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["job_id", "project_id"],
                        "additionalProperties": True,
                    },
                    "RunIngestAnalyzeRequest": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "analyze_overrides": {"type": "object", "additionalProperties": True},
                            "client_request_id": {"type": "string"},
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                    "RunIngestAnalyzeResponse": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string"},
                            "run_id": {"type": "string"},
                            "project_id": {"type": "string"},
                        },
                        "required": ["job_id", "run_id", "project_id"],
                        "additionalProperties": False,
                    },
                    "RunFullExportTopRequest": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "top": {"type": "integer"},
                            "export": {"type": "object", "additionalProperties": True},
                            "captions": {"type": "object", "additionalProperties": True},
                            "hook_text": {"type": "object", "additionalProperties": True},
                            "pip": {"type": "object", "additionalProperties": True},
                            "analyze_overrides": {"type": "object", "additionalProperties": True},
                            "client_request_id": {"type": "string"},
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                    "RunFullExportTopResponse": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string"},
                            "run_id": {"type": "string"},
                            "project_id": {"type": "string"},
                        },
                        "required": ["job_id", "run_id", "project_id"],
                        "additionalProperties": False,
                    },
                    "RunsLastResponse": {
                        "type": "object",
                        "properties": {
                            "run_id": {"type": "string"},
                            "job_id": {"type": "string"},
                            "project_id": {"type": "string"},
                            "status": {"type": "string"},
                            "progress": {"type": "number"},
                            "current_step": {"type": "string"},
                        },
                        "required": ["run_id", "job_id", "project_id", "status", "progress", "current_step"],
                        "additionalProperties": True,
                    },
                    "ProjectsRecentResponse": {
                        "type": "object",
                        "properties": {
                            "projects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "project_id": {"type": "string"},
                                        "created_at": {"type": "string"},
                                        "source_url": {"type": "string"},
                                        "last_run_id": {"type": "string"},
                                        "status": {"type": "string"},
                                    },
                                    "required": ["project_id"],
                                    "additionalProperties": True,
                                },
                            }
                        },
                        "required": ["projects"],
                        "additionalProperties": False,
                    },
                    "JobCancelResponse": {
                        "type": "object",
                        "properties": {
                            "ok": {"type": "boolean"},
                            "job_id": {"type": "string"},
                            "status": {"type": "string"},
                        },
                        "required": ["ok", "job_id", "status"],
                        "additionalProperties": False,
                    },
                    "JobStatusResponse": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "kind": {"type": "string"},
                            "created_at": {"type": "string"},
                            "status": {"type": "string"},
                            "progress": {"type": "number"},
                            "message": {"type": "string"},
                            "result": {"type": "object", "additionalProperties": True},
                            "poll_after_ms": {"type": "integer"},
                        },
                        "required": ["id", "kind", "created_at", "status", "progress", "message", "result", "poll_after_ms"],
                        "additionalProperties": False,
                    },
                    "AnnotationsRequest": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "annotations": {"type": "object", "additionalProperties": True},
                        },
                        "required": ["project_id", "annotations"],
                        "additionalProperties": False,
                    },
                    "AnnotationsResponse": {
                        "type": "object",
                        "properties": {
                            "ok": {"type": "boolean"},
                            "project_id": {"type": "string"},
                        },
                        "required": ["ok", "project_id"],
                        "additionalProperties": False,
                    },
                    "ResultsSummaryResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "DiagnosticsResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                },
            },
            "security": [{"bearerAuth": []}],
            "paths": {
                "/api/actions/health": {
                    "get": {
                        "summary": "Health check",
                        "operationId": "vp_actions_health",
                        "x-openai-isConsequential": False,
                        "responses": {"200": {"description": "OK"}},
                    }
                },
                "/api/actions/ingest_url": {
                    "post": {
                        "summary": "Ingest a video URL",
                        "operationId": "vp_actions_ingest_url",
                        "x-openai-isConsequential": True,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/IngestUrlRequest"},
                                    "examples": {
                                        "twitch_vod": {
                                            "summary": "Twitch VOD",
                                            "value": {
                                                "url": "https://twitch.tv/videos/1637616602",
                                                "client_request_id": "ingest-test-001",
                                            },
                                        },
                                        "twitch_vod_www": {
                                            "summary": "Twitch VOD (www)",
                                            "value": {
                                                "url": "https://www.twitch.tv/videos/1637616602",
                                                "client_request_id": "ingest-test-002",
                                            },
                                        },
                                        "twitch_clip": {
                                            "summary": "Twitch clip",
                                            "value": {
                                                "url": "https://clips.twitch.tv/SomeClipSlug",
                                                "client_request_id": "ingest-test-003",
                                            },
                                        },
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Job created",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/IngestUrlResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/analyze_full": {
                    "post": {
                        "summary": "Run full analysis",
                        "operationId": "vp_actions_analyze_full",
                        "x-openai-isConsequential": True,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AnalyzeFullRequest"},
                                    "examples": {
                                        "basic": {
                                            "summary": "Analyze with defaults",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "client_request_id": "analyze-001",
                                            },
                                        },
                                        "with_overrides": {
                                            "summary": "Analyze with profile overrides",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "overrides": {"highlights": {"top_n": 10}},
                                                "client_request_id": "analyze-002",
                                            },
                                        },
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Job created",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AnalyzeFullResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/run_ingest_analyze": {
                    "post": {
                        "summary": "Run ingest + analyze as a single workflow",
                        "operationId": "vp_actions_run_ingest_analyze",
                        "x-openai-isConsequential": False,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RunIngestAnalyzeRequest"},
                                    "examples": {
                                        "twitch_vod": {
                                            "summary": "Ingest + analyze a Twitch VOD",
                                            "value": {
                                                "url": "https://twitch.tv/videos/1637616602",
                                                "analyze_overrides": {"highlights": {"top_n": 10}},
                                                "client_request_id": "run-001",
                                            },
                                        }
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Run created",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/RunIngestAnalyzeResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/run_full_export_top": {
                    "post": {
                        "summary": "Run ingest + analyze + export top N as a single workflow",
                        "operationId": "vp_actions_run_full_export_top",
                        "x-openai-isConsequential": True,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RunFullExportTopRequest"},
                                    "examples": {
                                        "top_5": {
                                            "summary": "Full run with export top 5",
                                            "value": {
                                                "url": "https://twitch.tv/videos/1637616602",
                                                "top": 5,
                                                "client_request_id": "run-002",
                                            },
                                        }
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Run created",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/RunFullExportTopResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/export_batch": {
                    "post": {
                        "summary": "Render a batch of clips",
                        "operationId": "vp_actions_export_batch",
                        "x-openai-isConsequential": True,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ExportBatchRequest"},
                                    "examples": {
                                        "top_5": {
                                            "summary": "Export top 5 highlights",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "top": 5,
                                                "client_request_id": "export-001",
                                            },
                                        },
                                        "specific_clips": {
                                            "summary": "Export specific selections",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "selection_ids": ["sel-001", "sel-002"],
                                                "with_captions": True,
                                                "client_request_id": "export-002",
                                            },
                                        },
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Job created",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/ExportBatchResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/jobs/{job_id}": {
                    "get": {
                        "summary": "Get job status",
                        "operationId": "vp_actions_job_status",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {
                                "name": "job_id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Job status",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/JobStatusResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/jobs/{job_id}/cancel": {
                    "post": {
                        "summary": "Cancel a job",
                        "operationId": "vp_actions_job_cancel",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {
                                "name": "job_id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Cancellation requested",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/JobCancelResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/runs/last": {
                    "get": {
                        "summary": "Get the most recent run",
                        "operationId": "vp_actions_runs_last",
                        "x-openai-isConsequential": False,
                        "responses": {
                            "200": {
                                "description": "Last run",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/RunsLastResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/projects/recent": {
                    "get": {
                        "summary": "List recent projects",
                        "operationId": "vp_actions_projects_recent",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "integer"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Recent projects",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/ProjectsRecentResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/results/summary": {
                    "get": {
                        "summary": "Fetch a compact results summary",
                        "operationId": "vp_actions_results_summary",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "project_id", "in": "query", "required": True, "schema": {"type": "string"}},
                            {"name": "top_n", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "snippet_chars", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chat_lines", "in": "query", "required": False, "schema": {"type": "integer"}},
                        ],
                        "responses": {
                            "200": {
                                "description": "Summary",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/ResultsSummaryResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/annotations": {
                    "post": {
                        "summary": "Store model annotations",
                        "operationId": "vp_actions_annotations",
                        "x-openai-isConsequential": False,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AnnotationsRequest"},
                                    "examples": {
                                        "clip_notes": {
                                            "summary": "Add clip annotations",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "annotations": {
                                                    "clip_notes": [
                                                        {"selection_id": "sel-001", "note": "Great team fight moment"},
                                                    ],
                                                },
                                            },
                                        },
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Saved",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AnnotationsResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/diagnostics": {
                    "get": {
                        "summary": "Diagnostics",
                        "operationId": "vp_actions_diagnostics",
                        "x-openai-isConsequential": False,
                        "responses": {
                            "200": {
                                "description": "Diagnostics",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/DiagnosticsResponse"}}
                                },
                            }
                        },
                    }
                },
            },
        }
        return JSONResponse(spec)

    def _set_project_last_run(
        proj: Project,
        *,
        run_id: str,
        preset: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Persist a small pointer in project.json so Actions can list/resume runs across restarts."""
        from ..project import update_project, utc_now_iso

        def _upd(d: Dict[str, Any]) -> None:
            d.setdefault("analysis", {})
            d["analysis"].setdefault("actions", {})
            d["analysis"]["actions"]["last_run_id"] = str(run_id)
            d["analysis"]["actions"]["last_preset"] = str(preset)
            d["analysis"]["actions"]["last_run_status"] = str(status)
            d["analysis"]["actions"]["last_run_error"] = str(error) if error else None
            d["analysis"]["actions"]["last_run_updated_at"] = utc_now_iso()

        try:
            update_project(proj, _upd)
        except Exception:
            # Best-effort only.
            pass

    def _run_step_download_url(
        *,
        job: Any,
        proj: Project,
        project_id: str,
        url: str,
        options: Dict[str, Any],
        run_job: Optional[Any] = None,
        run_progress_base: float = 0.0,
        run_progress_span: float = 0.33,
    ) -> None:
        """Run the Actions download_url job body synchronously (used by run orchestrators)."""
        # Parse speed mode
        speed_mode_str = str(options.get("speed_mode", "auto")).lower()
        try:
            speed_mode = SpeedMode(speed_mode_str)
        except ValueError:
            speed_mode = SpeedMode.AUTO

        # Parse quality cap
        quality_cap_str = str(options.get("quality_cap", "source")).lower()
        try:
            quality_cap = QualityCap(quality_cap_str)
        except ValueError:
            quality_cap = QualityCap.SOURCE

        ingest_req = IngestRequest(
            url=url,
            speed_mode=speed_mode,
            quality_cap=quality_cap,
            no_playlist=bool(options.get("no_playlist", True)),
            create_preview=bool(options.get("create_preview", True)),
            auto_open=False,  # Actions should not affect Studio UI state.
        )

        def _set_run(frac: float, msg: str) -> None:
            if run_job is None or run_job.cancel_requested:
                return
            JOB_MANAGER._set(
                run_job,
                progress=run_progress_base + run_progress_span * max(0.0, min(1.0, float(frac))),
                message=msg,
                result={"current_step": "ingest", "active_job_id": job.id},
            )

        def _video_ready() -> bool:
            try:
                return bool((proj.video_dir / "video.mp4").exists())
            except Exception:
                return False

        try:
            JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting download...", result={"project_id": project_id, "url": url})
            _set_run(0.0, "Ingest: starting...")

            def check_cancel() -> bool:
                return bool(job.cancel_requested or (run_job is not None and run_job.cancel_requested))

            def on_prog(frac: float, msg: str) -> None:
                if job.cancel_requested:
                    return
                f = max(0.0, min(0.9, float(frac) * 0.9))
                JOB_MANAGER._set(
                    job,
                    progress=f,
                    message=msg,
                    result={"project_id": project_id, "url": url},
                )
                _set_run(f, f"Ingest: {msg}")

            video_result = download_url(url, request=ingest_req, on_progress=on_prog, check_cancel=check_cancel)

            # Persist into the project directory.
            from ..project import set_project_video

            src = Path(video_result.video_path)
            prev = Path(video_result.preview_path) if getattr(video_result, "preview_path", None) else None
            set_project_video(proj, src, preview_path=prev)
            set_source_url(proj, url)

            # Optional chat download/import (best-effort).
            chat_info: Dict[str, Any] = {"status": "skipped"}
            try:
                from ..chat.downloader import ChatDownloadCancelled, download_chat, import_chat_to_project

                chat_out = proj.chat_raw_path

                def on_chat(frac: float, msg: str) -> None:
                    if job.cancel_requested:
                        return
                    f = 0.9 + 0.09 * max(0.0, min(1.0, float(frac)))
                    JOB_MANAGER._set(
                        job,
                        progress=f,
                        message=f"Chat: {msg}",
                        result={"project_id": project_id, "url": url},
                    )
                    _set_run(f, f"Chat: {msg}")

                chat_res = download_chat(url, chat_out, on_progress=on_chat, check_cancel=check_cancel)
                import_chat_to_project(proj, chat_out)
                chat_info = {"status": "success", **(chat_res.to_dict())}
            except (DownloadCancelled, ChatDownloadCancelled):
                raise DownloadCancelled()
            except Exception as e:
                # Non-fatal.
                chat_info = {"status": "failed", "error": f"{type(e).__name__}: {e}"}

            JOB_MANAGER._set(
                job,
                status="succeeded",
                progress=1.0,
                message="done",
                result={
                    "project_id": project_id,
                    "url": url,
                    "video": video_result.to_dict(),
                    "chat": chat_info,
                },
            )
            _set_run(1.0, "Ingest: done")
        except DownloadCancelled:
            if not _video_ready():
                try:
                    set_project_status(
                        proj,
                        status="download_failed",
                        status_error="cancelled",
                        video_status="download_failed",
                        video_error="cancelled",
                    )
                except Exception:
                    pass
            if not job.cancel_requested:
                JOB_MANAGER._set(job, status="cancelled", message="cancelled")
            _set_run(1.0, "Ingest: cancelled")
        except Exception as e:
            if not _video_ready():
                err = f"{type(e).__name__}: {e}"
                try:
                    set_project_status(
                        proj,
                        status="download_failed",
                        status_error=err,
                        video_status="download_failed",
                        video_error=err,
                    )
                except Exception:
                    pass
            if not job.cancel_requested:
                JOB_MANAGER._set(
                    job,
                    status="failed",
                    message=f"{type(e).__name__}: {e}",
                    result={"project_id": project_id, "url": url},
                )
            _set_run(1.0, f"Ingest: failed ({type(e).__name__})")

    def _run_step_analyze_full(
        *,
        job: Any,
        project_id: str,
        overrides: Dict[str, Any],
        run_job: Optional[Any] = None,
        run_progress_base: float = 0.33,
        run_progress_span: float = 0.33,
    ) -> None:
        """Run the Actions analyze_full job body synchronously (used by run orchestrators)."""

        class _CancelledError(Exception):
            pass

        proj, _proj_data = _load_project(project_id)
        if not Path(proj.video_path).exists():
            JOB_MANAGER._set(job, status="failed", message="video_not_ready", result={"project_id": project_id})
            return

        def _set_run(frac: float, msg: str) -> None:
            if run_job is None or run_job.cancel_requested:
                return
            JOB_MANAGER._set(
                run_job,
                progress=run_progress_base + run_progress_span * max(0.0, min(1.0, float(frac))),
                message=msg,
                result={"current_step": "analyze", "active_job_id": job.id},
            )

        try:
            JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting analysis...", result={"project_id": project_id})
            _set_run(0.0, "Analyze: starting...")

            try:
                set_project_status(proj, status="analyzing", status_error=None)
            except Exception:
                pass

            analysis_cfg = profile.get("analysis", {}) or {}
            speech_cfg = dict(analysis_cfg.get("speech", {}) or {})

            # Allow a few common overrides without expanding surface area too much.
            if "diarize" in overrides:
                speech_cfg["diarize"] = bool(overrides.get("diarize"))
            if "whisper_verbose" in overrides:
                speech_cfg["verbose"] = bool(overrides.get("whisper_verbose"))

            dag_config: Dict[str, Any] = {
                **analysis_cfg,
                "speech": speech_cfg,
                "include_chat": True,
                "include_audio_events": bool(analysis_cfg.get("audio_events", {}).get("enabled", True)),
            }

            llm_complete_fn = None
            try:
                ai_cfg = profile.get("ai", {}).get("director", {}) or {}
                highlights_cfg = dag_config.get("highlights", {}) or {}
                llm_needed = bool(highlights_cfg.get("llm_semantic_enabled", True)) or bool(
                    highlights_cfg.get("llm_filter_enabled", False)
                )
                if ai_cfg.get("enabled", True) and llm_needed:
                    llm_complete_fn = get_llm_complete_fn(ai_cfg, proj.analysis_dir)
            except Exception:
                llm_complete_fn = None

            def on_progress(frac: float, msg: str) -> None:
                if job.cancel_requested or (run_job is not None and run_job.cancel_requested):
                    raise _CancelledError("cancelled")
                f = max(0.0, min(1.0, float(frac)))
                JOB_MANAGER._set(job, progress=f, message=msg, result={"project_id": project_id})
                _set_run(f, f"Analyze: {msg}")

            result = run_analysis(
                proj,
                bundle="full",
                config=dag_config,
                on_progress=on_progress,
                llm_complete=llm_complete_fn,
                upgrade_mode=True,
            )

            try:
                ok = bool(getattr(result, "success", True))
                err_msg = getattr(result, "error", None)
                if ok:
                    set_project_status(proj, status="complete", status_error=None)
                else:
                    set_project_status(proj, status="analysis_failed", status_error=str(err_msg or "analysis failed"))
            except Exception:
                pass

            JOB_MANAGER._set(
                job,
                status="succeeded",
                progress=1.0,
                message="done",
                result={
                    "project_id": project_id,
                    "success": bool(getattr(result, "success", True)),
                    "error": getattr(result, "error", None),
                    "tasks_run": len(getattr(result, "tasks_run", []) or []),
                    "elapsed_seconds": getattr(result, "total_elapsed_seconds", None),
                    "missing_targets": sorted(getattr(result, "missing_targets", set()) or []),
                },
            )
            _set_run(1.0, "Analyze: done")
        except _CancelledError:
            JOB_MANAGER._set(job, status="cancelled", message="cancelled", result={"project_id": project_id})
            _set_run(1.0, "Analyze: cancelled")
        except Exception as e:
            try:
                set_project_status(proj, status="analysis_failed", status_error=f"{type(e).__name__}: {e}")
            except Exception:
                pass
            if not job.cancel_requested:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}", result={"project_id": project_id})
            _set_run(1.0, f"Analyze: failed ({type(e).__name__})")

    def _run_step_export_batch_top(
        *,
        job: Any,
        project_id: str,
        top: int,
        export_cfg: Dict[str, Any],
        cap_cfg: Dict[str, Any],
        hook_cfg: Dict[str, Any],
        pip_cfg: Dict[str, Any],
        run_job: Optional[Any] = None,
        run_progress_base: float = 0.66,
        run_progress_span: float = 0.34,
    ) -> None:
        """Run the Actions export_batch job body synchronously (used by run orchestrators)."""
        proj, proj_data = _load_project(project_id)
        if not Path(proj.video_path).exists():
            JOB_MANAGER._set(job, status="failed", message="video_not_ready", result={"project_id": project_id})
            return

        def _set_run(frac: float, msg: str) -> None:
            if run_job is None or run_job.cancel_requested:
                return
            JOB_MANAGER._set(
                run_job,
                progress=run_progress_base + run_progress_span * max(0.0, min(1.0, float(frac))),
                message=msg,
                result={"current_step": "export", "active_job_id": job.id},
            )

        # Merge configs (profile defaults overridden by request).
        export_cfg = {**(profile.get("export", {}) or {}), **(export_cfg or {})}
        cap_cfg = {**(profile.get("captions", {}) or {}), **(cap_cfg or {})}
        hook_cfg = {**(profile.get("overlay", {}).get("hook_text", {}) or {}), **(hook_cfg or {})}
        pip_cfg = {**(profile.get("layout", {}).get("pip", {}) or {}), **(pip_cfg or {})}
        with_captions = bool(cap_cfg.get("enabled", False))

        whisper_cfg = None
        if with_captions:
            from ..transcribe import TranscribeConfig

            whisper_cfg = TranscribeConfig(
                model_size=str(cap_cfg.get("model_size", "small")),
                language=cap_cfg.get("language"),
                device=str(cap_cfg.get("device", "cpu")),
                compute_type=str(cap_cfg.get("compute_type", "int8")),
            )

        # Create selections from top candidates (persisted in project.json).
        from ..project import add_selection_from_candidate

        top_n = _clamp_int(top, default=10, min_v=1, max_v=30)
        template = str(export_cfg.get("template", "vertical_blur"))
        candidates = ((proj_data.get("analysis", {}) or {}).get("highlights", {}) or {}).get("candidates") or []
        if not candidates:
            JOB_MANAGER._set(job, status="failed", message="no_candidates", result={"project_id": project_id})
            return

        created_ids: list[str] = []
        for cand in candidates[:top_n]:
            if not isinstance(cand, dict):
                continue
            title = str(cand.get("title") or "")
            sel_id = add_selection_from_candidate(proj, candidate=cand, template=template, title=title)
            created_ids.append(sel_id)

        proj_data = get_project_data(proj)
        selections = list(proj_data.get("selections") or [])
        sel_set = set(created_ids)
        selections = [s for s in selections if str(s.get("id")) in sel_set]
        if not selections:
            JOB_MANAGER._set(job, status="failed", message="no_selections", result={"project_id": project_id, "created_selection_ids": created_ids})
            return

        director_results = (proj_data.get("analysis", {}) or {}).get("director", {}).get("results") or []

        JOB_MANAGER._set(job, status="running", progress=0.0, message="starting", result={"project_id": project_id, "created_selection_ids": created_ids})
        _set_run(0.0, "Export: starting...")

        try:
            total = max(1, len(selections))
            outputs: list[str] = []
            JOB_MANAGER._set(job, message=f"exporting 0/{total}")

            for idx, selection in enumerate(selections, start=1):
                if job.cancel_requested or (run_job is not None and run_job.cancel_requested):
                    JOB_MANAGER._set(job, status="cancelled", message="cancelled")
                    _set_run(1.0, "Export: cancelled")
                    return

                subjob = JOB_MANAGER.start_export(
                    proj=proj,
                    selection=selection,
                    export_dir=proj.exports_dir,
                    with_captions=with_captions,
                    template=str(export_cfg.get("template", selection.get("template") or "vertical_blur")),
                    width=int(export_cfg.get("width", 1080)),
                    height=int(export_cfg.get("height", 1920)),
                    fps=int(export_cfg.get("fps", 30)),
                    crf=int(export_cfg.get("crf", 20)),
                    preset=str(export_cfg.get("preset", "veryfast")),
                    normalize_audio=bool(export_cfg.get("normalize_audio", False)),
                    whisper_cfg=whisper_cfg,
                    hook_cfg=hook_cfg,
                    pip_cfg=pip_cfg,
                    director_results=director_results,
                )

                # Wait for subjob completion and mirror progress.
                while True:
                    time.sleep(0.2)
                    sj = JOB_MANAGER.get(subjob.id)
                    if sj is None:
                        break
                    if sj.status in {"succeeded", "failed", "cancelled"}:
                        break
                    frac = (idx - 1 + sj.progress) / total
                    JOB_MANAGER._set(job, progress=frac, message=f"exporting {idx}/{total}", result={"active_subjob_id": subjob.id})
                    _set_run(frac, f"Export: exporting {idx}/{total}")

                    if job.cancel_requested or (run_job is not None and run_job.cancel_requested):
                        try:
                            JOB_MANAGER.cancel(subjob.id)
                        except Exception:
                            pass
                        JOB_MANAGER._set(job, status="cancelled", message="cancelled", result={"active_subjob_id": subjob.id})
                        _set_run(1.0, "Export: cancelled")
                        return

                sj = JOB_MANAGER.get(subjob.id)
                if sj and sj.status == "failed":
                    raise RuntimeError(sj.message)
                if sj and sj.status == "succeeded":
                    out = (sj.result or {}).get("output")
                    if out:
                        outputs.append(str(out))

                JOB_MANAGER._set(job, progress=idx / total, message=f"exporting {idx}/{total}", result={"active_subjob_id": subjob.id})
                _set_run(idx / total, f"Export: exporting {idx}/{total}")

            # Write a small manifest for Actions.
            manifest_path = proj.exports_dir / f"actions_export_manifest_{int(time.time())}.json"
            manifest = {
                "project_id": project_id,
                "created_selection_ids": created_ids,
                "outputs": outputs,
                "export": {
                    "template": str(export_cfg.get("template", "vertical_blur")),
                    "width": int(export_cfg.get("width", 1080)),
                    "height": int(export_cfg.get("height", 1920)),
                    "fps": int(export_cfg.get("fps", 30)),
                    "crf": int(export_cfg.get("crf", 20)),
                    "preset": str(export_cfg.get("preset", "veryfast")),
                    "with_captions": bool(with_captions),
                },
                "generated_at": time.time(),
            }
            save_json(manifest_path, manifest)

            JOB_MANAGER._set(
                job,
                status="succeeded",
                progress=1.0,
                message="done",
                result={
                    "project_id": project_id,
                    "created_selection_ids": created_ids,
                    "outputs": outputs,
                    "manifest_path": str(manifest_path),
                },
            )
            _set_run(1.0, "Export: done")
        except Exception as e:
            if not job.cancel_requested:
                JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}", result={"project_id": project_id, "created_selection_ids": created_ids})
            _set_run(1.0, f"Export: failed ({type(e).__name__})")

    def _start_run_job(
        *,
        actor_key: str,
        preset: str,
        proj: Project,
        project_id: str,
        url: str,
        analyze_overrides: Dict[str, Any],
        export_top: Optional[int] = None,
        export_cfg: Optional[Dict[str, Any]] = None,
        cap_cfg: Optional[Dict[str, Any]] = None,
        hook_cfg: Optional[Dict[str, Any]] = None,
        pip_cfg: Optional[Dict[str, Any]] = None,
    ) -> Any:
        run_job = JOB_MANAGER.create("run")

        steps: Dict[str, Any] = {}
        JOB_MANAGER._set(
            run_job,
            message="queued",
            result={
                "run_id": run_job.id,
                "project_id": project_id,
                "preset": preset,
                "url": url,
                "current_step": "queued",
                "active_job_id": None,
                "steps": steps,
            },
        )
        last_run_by_actor[actor_key] = run_job.id
        _set_project_last_run(proj, run_id=run_job.id, preset=preset, status="queued", error=None)

        def _set_steps(*, current_step: str, active_job_id: Optional[str] = None) -> None:
            if run_job.cancel_requested:
                return
            JOB_MANAGER._set(
                run_job,
                result={
                    "steps": steps,
                    "current_step": current_step,
                    "active_job_id": active_job_id,
                },
            )

        def _finish_run(status: str, message: str, *, error: Optional[str] = None) -> None:
            # Avoid overwriting cancellation.
            if run_job.cancel_requested and status != "cancelled":
                return
            JOB_MANAGER._set(run_job, status=status, progress=1.0 if status in {"succeeded", "failed", "cancelled"} else run_job.progress, message=message, result={"current_step": status})
            _set_project_last_run(proj, run_id=run_job.id, preset=preset, status=status, error=error)

        @with_prevent_sleep("Running workflow (Actions)")
        def runner() -> None:
            try:
                if run_job.cancel_requested:
                    _finish_run("cancelled", "cancelled")
                    return

                JOB_MANAGER._set(run_job, status="running", progress=0.0, message="Starting run...", result={"current_step": "starting"})
                _set_project_last_run(proj, run_id=run_job.id, preset=preset, status="running", error=None)

                # Step 1: ingest
                ingest_job = JOB_MANAGER.create("download_url")
                steps["ingest"] = {"job_id": ingest_job.id, "status": "running"}
                _set_steps(current_step="ingest", active_job_id=ingest_job.id)
                _run_step_download_url(
                    job=ingest_job,
                    proj=proj,
                    project_id=project_id,
                    url=url,
                    options={},
                    run_job=run_job,
                    run_progress_base=0.0,
                    run_progress_span=0.5 if export_top is None else 0.34,
                )
                steps["ingest"]["status"] = ingest_job.status
                _set_steps(current_step="ingest", active_job_id=ingest_job.id)

                if ingest_job.status != "succeeded":
                    if ingest_job.status == "cancelled" or run_job.cancel_requested:
                        _finish_run("cancelled", "cancelled")
                    else:
                        _finish_run("failed", "ingest_failed", error=ingest_job.message)
                    return

                if run_job.cancel_requested:
                    try:
                        JOB_MANAGER.cancel(ingest_job.id)
                    except Exception:
                        pass
                    _finish_run("cancelled", "cancelled")
                    return

                # Step 2: analyze
                analyze_job = JOB_MANAGER.create("analyze_full")
                steps["analyze"] = {"job_id": analyze_job.id, "status": "running"}
                _set_steps(current_step="analyze", active_job_id=analyze_job.id)
                _run_step_analyze_full(
                    job=analyze_job,
                    project_id=project_id,
                    overrides=analyze_overrides,
                    run_job=run_job,
                    run_progress_base=0.5 if export_top is None else 0.34,
                    run_progress_span=0.5 if export_top is None else 0.33,
                )
                steps["analyze"]["status"] = analyze_job.status
                _set_steps(current_step="analyze", active_job_id=analyze_job.id)

                if analyze_job.status != "succeeded":
                    if analyze_job.status == "cancelled" or run_job.cancel_requested:
                        _finish_run("cancelled", "cancelled")
                    else:
                        _finish_run("failed", "analyze_failed", error=analyze_job.message)
                    return

                # Optional Step 3: export top N
                if export_top is not None:
                    if run_job.cancel_requested:
                        try:
                            JOB_MANAGER.cancel(analyze_job.id)
                        except Exception:
                            pass
                        _finish_run("cancelled", "cancelled")
                        return

                    export_job = JOB_MANAGER.create("export_batch")
                    steps["export"] = {"job_id": export_job.id, "status": "running"}
                    _set_steps(current_step="export", active_job_id=export_job.id)
                    _run_step_export_batch_top(
                        job=export_job,
                        project_id=project_id,
                        top=int(export_top),
                        export_cfg=export_cfg or {},
                        cap_cfg=cap_cfg or {},
                        hook_cfg=hook_cfg or {},
                        pip_cfg=pip_cfg or {},
                        run_job=run_job,
                        run_progress_base=0.67,
                        run_progress_span=0.33,
                    )
                    steps["export"]["status"] = export_job.status
                    _set_steps(current_step="export", active_job_id=export_job.id)

                    if export_job.status != "succeeded":
                        if export_job.status == "cancelled" or run_job.cancel_requested:
                            _finish_run("cancelled", "cancelled")
                        else:
                            _finish_run("failed", "export_failed", error=export_job.message)
                        return

                    JOB_MANAGER._set(
                        run_job,
                        result={
                            "summary": {
                                "created_selection_ids": (export_job.result or {}).get("created_selection_ids") or [],
                                "manifest_path": (export_job.result or {}).get("manifest_path"),
                                "outputs": (export_job.result or {}).get("outputs") or [],
                            }
                        },
                    )

                _finish_run("succeeded", "done")
            except Exception as e:
                if not run_job.cancel_requested:
                    _finish_run("failed", f"{type(e).__name__}: {e}", error=f"{type(e).__name__}: {e}")

        import threading

        threading.Thread(target=runner, daemon=True).start()
        return run_job

    @router.post("/run_ingest_analyze", openapi_extra={"x-openai-isConsequential": False})
    def run_ingest_analyze(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        url = _validate_ingest_url(str(body.get("url") or ""), allow_domains=allow_domains)
        analyze_overrides = body.get("analyze_overrides") or body.get("overrides") or {}
        if analyze_overrides and not isinstance(analyze_overrides, dict):
            raise HTTPException(status_code=400, detail="invalid_analyze_overrides")

        content_id = _extract_content_id(url)
        proj = create_project_early(content_id, source_url=url)
        project_id = proj.project_dir.name

        run_job = _start_run_job(
            actor_key=actor_key,
            preset="ingest_analyze",
            proj=proj,
            project_id=project_id,
            url=url,
            analyze_overrides=analyze_overrides or {},
        )

        payload = {"job_id": run_job.id, "run_id": run_job.id, "project_id": project_id}
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.post("/run_full_export_top", openapi_extra={"x-openai-isConsequential": True})
    def run_full_export_top(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        url = _validate_ingest_url(str(body.get("url") or ""), allow_domains=allow_domains)
        analyze_overrides = body.get("analyze_overrides") or body.get("overrides") or {}
        if analyze_overrides and not isinstance(analyze_overrides, dict):
            raise HTTPException(status_code=400, detail="invalid_analyze_overrides")

        top_n = _clamp_int(body.get("top"), default=5, min_v=1, max_v=30)

        export_cfg = body.get("export") or {}
        cap_cfg = body.get("captions") or {}
        hook_cfg = body.get("hook_text") or {}
        pip_cfg = body.get("pip") or {}
        if not isinstance(export_cfg, dict) or not isinstance(cap_cfg, dict) or not isinstance(hook_cfg, dict) or not isinstance(pip_cfg, dict):
            raise HTTPException(status_code=400, detail="invalid_export_config")

        content_id = _extract_content_id(url)
        proj = create_project_early(content_id, source_url=url)
        project_id = proj.project_dir.name

        run_job = _start_run_job(
            actor_key=actor_key,
            preset="full_export_top",
            proj=proj,
            project_id=project_id,
            url=url,
            analyze_overrides=analyze_overrides or {},
            export_top=top_n,
            export_cfg=export_cfg,
            cap_cfg=cap_cfg,
            hook_cfg=hook_cfg,
            pip_cfg=pip_cfg,
        )

        payload = {"job_id": run_job.id, "run_id": run_job.id, "project_id": project_id}
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.post("/jobs/{job_id}/cancel", openapi_extra={"x-openai-isConsequential": False})
    def job_cancel(request: Request, job_id: str):
        _rate_limit(request)
        job = JOB_MANAGER.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")

        # Best-effort: if this is a run, try to cancel the currently active sub-job.
        if job.kind == "run":
            active = None
            try:
                active = (job.result or {}).get("active_job_id")
                if not active:
                    steps = (job.result or {}).get("steps") or {}
                    cur = (job.result or {}).get("current_step")
                    if cur and isinstance(steps, dict) and isinstance(steps.get(cur), dict):
                        active = steps[cur].get("job_id")
            except Exception:
                active = None
            if active:
                try:
                    JOB_MANAGER.cancel(str(active))
                except Exception:
                    pass

        ok = JOB_MANAGER.cancel(job_id)
        if not ok:
            raise HTTPException(status_code=400, detail="job_not_cancellable")
        return JSONResponse({"ok": True, "job_id": job_id, "status": "cancel_requested"})

    @router.get("/runs/last", openapi_extra={"x-openai-isConsequential": False})
    def runs_last(request: Request):
        actor_key = _rate_limit(request)
        run_id = last_run_by_actor.get(actor_key)

        def _job_to_payload(j: Any) -> Dict[str, Any]:
            return {
                "run_id": j.id,
                "job_id": j.id,
                "project_id": (j.result or {}).get("project_id"),
                "status": j.status,
                "progress": j.progress,
                "current_step": (j.result or {}).get("current_step") or "",
            }

        if run_id:
            j = JOB_MANAGER.get(run_id)
            if j and j.kind == "run":
                return JSONResponse(_job_to_payload(j))

        # Fallback: pick most recent run job in memory (useful for single-user setups).
        try:
            with JOB_MANAGER._lock:  # type: ignore[attr-defined]
                runs = [j for j in JOB_MANAGER._jobs.values() if getattr(j, "kind", None) == "run"]  # type: ignore[attr-defined]
            if runs:
                runs.sort(key=lambda j: str(getattr(j, "created_at", "")))
                return JSONResponse(_job_to_payload(runs[-1]))
        except Exception:
            pass

        raise HTTPException(status_code=404, detail="no_runs")

    @router.get("/projects/recent", openapi_extra={"x-openai-isConsequential": False})
    def projects_recent(request: Request, limit: int = 20):
        _rate_limit(request)
        limit = _clamp_int(limit, default=20, min_v=1, max_v=50)

        projects_root = default_projects_root()
        if not projects_root.exists():
            return JSONResponse({"projects": []})

        items: list[Dict[str, Any]] = []
        for pdir in projects_root.iterdir():
            if not pdir.is_dir():
                continue
            pjson = pdir / "project.json"
            if not pjson.exists():
                continue
            try:
                data = json.loads(pjson.read_text(encoding="utf-8"))
                src = (data.get("source", {}) or {}).get("source_url") or data.get("source_url")
                actions = ((data.get("analysis", {}) or {}).get("actions", {}) or {})
                items.append(
                    {
                        "project_id": data.get("project_id", pdir.name),
                        "created_at": data.get("created_at") or "",
                        "source_url": src or "",
                        "last_run_id": actions.get("last_run_id") or "",
                        "status": data.get("status") or "",
                        "_mtime": pjson.stat().st_mtime,
                    }
                )
            except Exception:
                continue

        items.sort(key=lambda it: float(it.get("_mtime") or 0.0), reverse=True)
        out = []
        for it in items[:limit]:
            it.pop("_mtime", None)
            out.append(it)
        return JSONResponse({"projects": out})

    @router.post("/ingest_url", openapi_extra={"x-openai-isConsequential": True})
    def ingest_url(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        url = _validate_ingest_url(str(body.get("url") or ""), allow_domains=allow_domains)
        opts = body.get("options") or {}

        content_id = _extract_content_id(url)
        proj = create_project_early(content_id, source_url=url)
        project_id = proj.project_dir.name

        # Parse speed mode
        speed_mode_str = str(opts.get("speed_mode", "auto")).lower()
        try:
            speed_mode = SpeedMode(speed_mode_str)
        except ValueError:
            speed_mode = SpeedMode.AUTO

        # Parse quality cap
        quality_cap_str = str(opts.get("quality_cap", "source")).lower()
        try:
            quality_cap = QualityCap(quality_cap_str)
        except ValueError:
            quality_cap = QualityCap.SOURCE

        ingest_req = IngestRequest(
            url=url,
            speed_mode=speed_mode,
            quality_cap=quality_cap,
            no_playlist=bool(opts.get("no_playlist", True)),
            create_preview=bool(opts.get("create_preview", True)),
            auto_open=False,  # Actions should not affect Studio UI state.
        )

        job = JOB_MANAGER.create("download_url")
        JOB_MANAGER._set(job, message="queued", result={"project_id": project_id, "url": url})

        @with_prevent_sleep("Downloading video (Actions)")
        def runner() -> None:
            def _video_ready() -> bool:
                try:
                    return bool((proj.video_dir / "video.mp4").exists())
                except Exception:
                    return False

            try:
                JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting download...")

                def check_cancel() -> bool:
                    return bool(job.cancel_requested)

                def on_prog(frac: float, msg: str) -> None:
                    if job.cancel_requested:
                        return
                    JOB_MANAGER._set(
                        job,
                        progress=max(0.0, min(0.9, float(frac) * 0.9)),
                        message=msg,
                        result={"project_id": project_id, "url": url},
                    )

                video_result = download_url(url, request=ingest_req, on_progress=on_prog, check_cancel=check_cancel)

                # Persist into the project directory.
                from ..project import set_project_video

                src = Path(video_result.video_path)
                prev = Path(video_result.preview_path) if getattr(video_result, "preview_path", None) else None
                set_project_video(proj, src, preview_path=prev)
                set_source_url(proj, url)

                # Optional chat download/import (best-effort).
                chat_info: Dict[str, Any] = {"status": "skipped"}
                try:
                    from ..chat.downloader import ChatDownloadCancelled, download_chat, import_chat_to_project

                    chat_out = proj.chat_raw_path

                    def on_chat(frac: float, msg: str) -> None:
                        if job.cancel_requested:
                            return
                        JOB_MANAGER._set(
                            job,
                            progress=0.9 + 0.09 * max(0.0, min(1.0, float(frac))),
                            message=f"Chat: {msg}",
                            result={"project_id": project_id, "url": url},
                        )

                    chat_res = download_chat(url, chat_out, on_progress=on_chat, check_cancel=check_cancel)
                    import_chat_to_project(proj, chat_out)
                    chat_info = {"status": "success", **(chat_res.to_dict())}
                except (DownloadCancelled, ChatDownloadCancelled):
                    raise DownloadCancelled()
                except Exception as e:
                    # Non-fatal.
                    chat_info = {"status": "failed", "error": f"{type(e).__name__}: {e}"}

                JOB_MANAGER._set(
                    job,
                    status="succeeded",
                    progress=1.0,
                    message="done",
                    result={
                        "project_id": project_id,
                        "url": url,
                        "video": video_result.to_dict(),
                        "chat": chat_info,
                    },
                )
            except DownloadCancelled:
                if not _video_ready():
                    try:
                        set_project_status(
                            proj,
                            status="download_failed",
                            status_error="cancelled",
                            video_status="download_failed",
                            video_error="cancelled",
                        )
                    except Exception:
                        pass
                if not job.cancel_requested:
                    JOB_MANAGER._set(job, status="cancelled", message="cancelled")
            except Exception as e:
                if not _video_ready():
                    err = f"{type(e).__name__}: {e}"
                    try:
                        set_project_status(
                            proj,
                            status="download_failed",
                            status_error=err,
                            video_status="download_failed",
                            video_error=err,
                        )
                    except Exception:
                        pass
                if not job.cancel_requested:
                    JOB_MANAGER._set(
                        job,
                        status="failed",
                        message=f"{type(e).__name__}: {e}",
                        result={"project_id": project_id, "url": url},
                    )

        import threading

        threading.Thread(target=runner, daemon=True).start()

        payload = {"job_id": job.id, "project_id": project_id}
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.post("/analyze_full", openapi_extra={"x-openai-isConsequential": True})
    def analyze_full(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        project_id = _validate_project_id(str(body.get("project_id") or ""))
        overrides = body.get("overrides") or {}

        proj, _proj_data = _load_project(project_id)
        if not Path(proj.video_path).exists():
            raise HTTPException(status_code=409, detail="video_not_ready")

        job = JOB_MANAGER.create("analyze_full")
        JOB_MANAGER._set(job, message="queued", result={"project_id": project_id})

        @with_prevent_sleep("Analyzing video (Actions)")
        def runner() -> None:
            try:
                JOB_MANAGER._set(job, status="running", progress=0.0, message="Starting analysis...")
                try:
                    set_project_status(proj, status="analyzing", status_error=None)
                except Exception:
                    pass

                analysis_cfg = profile.get("analysis", {}) or {}
                speech_cfg = dict(analysis_cfg.get("speech", {}) or {})

                # Allow a few common overrides without expanding surface area too much.
                if "diarize" in overrides:
                    speech_cfg["diarize"] = bool(overrides.get("diarize"))
                if "whisper_verbose" in overrides:
                    speech_cfg["verbose"] = bool(overrides.get("whisper_verbose"))

                dag_config: Dict[str, Any] = {
                    **analysis_cfg,
                    "speech": speech_cfg,
                    "include_chat": True,
                    "include_audio_events": bool(analysis_cfg.get("audio_events", {}).get("enabled", True)),
                }

                llm_complete_fn = None
                try:
                    ai_cfg = profile.get("ai", {}).get("director", {}) or {}
                    highlights_cfg = dag_config.get("highlights", {}) or {}
                    llm_needed = bool(highlights_cfg.get("llm_semantic_enabled", True)) or bool(
                        highlights_cfg.get("llm_filter_enabled", False)
                    )
                    if ai_cfg.get("enabled", True) and llm_needed:
                        llm_complete_fn = get_llm_complete_fn(ai_cfg, proj.analysis_dir)
                except Exception:
                    llm_complete_fn = None

                def on_progress(frac: float, msg: str) -> None:
                    if job.cancel_requested:
                        return
                    JOB_MANAGER._set(
                        job,
                        progress=max(0.0, min(1.0, float(frac))),
                        message=msg,
                        result={"project_id": project_id},
                    )

                result = run_analysis(
                    proj,
                    bundle="full",
                    config=dag_config,
                    on_progress=on_progress,
                    llm_complete=llm_complete_fn,
                    upgrade_mode=True,
                )

                try:
                    ok = bool(getattr(result, "success", True))
                    err_msg = getattr(result, "error", None)
                    if ok:
                        set_project_status(proj, status="complete", status_error=None)
                    else:
                        set_project_status(proj, status="analysis_failed", status_error=str(err_msg or "analysis failed"))
                except Exception:
                    pass

                JOB_MANAGER._set(
                    job,
                    status="succeeded",
                    progress=1.0,
                    message="done",
                    result={
                        "project_id": project_id,
                        "success": bool(getattr(result, "success", True)),
                        "error": getattr(result, "error", None),
                        "tasks_run": len(getattr(result, "tasks_run", []) or []),
                        "elapsed_seconds": getattr(result, "total_elapsed_seconds", None),
                        "missing_targets": sorted(getattr(result, "missing_targets", set()) or []),
                    },
                )
            except Exception as e:
                try:
                    set_project_status(proj, status="analysis_failed", status_error=f"{type(e).__name__}: {e}")
                except Exception:
                    pass
                if not job.cancel_requested:
                    JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}", result={"project_id": project_id})

        import threading

        threading.Thread(target=runner, daemon=True).start()

        payload = {"job_id": job.id, "project_id": project_id}
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.get("/jobs/{job_id}", openapi_extra={"x-openai-isConsequential": False})
    def job_status(request: Request, job_id: str):
        _rate_limit(request)
        job = JOB_MANAGER.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")

        poll_after_ms = 0
        if job.status == "queued":
            poll_after_ms = 5000
        elif job.status == "running":
            poll_after_ms = 1000

        return JSONResponse(
            {
                "id": job.id,
                "kind": job.kind,
                "created_at": job.created_at,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "result": job.result,
                "poll_after_ms": poll_after_ms,
            }
        )

    @router.get("/results/summary", openapi_extra={"x-openai-isConsequential": False})
    def results_summary(
        request: Request,
        project_id: str,
        top_n: int = 12,
        snippet_chars: int = 600,
        chat_lines: int = 12,
    ):
        _rate_limit(request)
        pid = _validate_project_id(project_id)

        top_n = _clamp_int(top_n, default=12, min_v=1, max_v=30)
        snippet_chars = _clamp_int(snippet_chars, default=600, min_v=100, max_v=2000)
        chat_lines = _clamp_int(chat_lines, default=12, min_v=0, max_v=50)

        proj, proj_data = _load_project(pid)

        meta = {
            "project_id": pid,
            "title": proj_data.get("title") or proj_data.get("video", {}).get("title"),
            "source_url": (proj_data.get("source", {}) or {}).get("source_url") or proj_data.get("source_url"),
            "duration_seconds": (proj_data.get("video", {}) or {}).get("duration_seconds"),
        }

        highlights = (proj_data.get("analysis", {}) or {}).get("highlights", {}) or {}
        candidates = list(highlights.get("candidates") or [])
        top_candidates = []
        for c in candidates[:top_n]:
            if not isinstance(c, dict):
                continue
            top_candidates.append(
                {
                    "rank": c.get("rank"),
                    "score": c.get("score"),
                    "start_s": c.get("start_s"),
                    "end_s": c.get("end_s"),
                    "peak_time_s": c.get("peak_time_s"),
                    "title": c.get("title") or c.get("hook_text") or "",
                    "hook_text": c.get("hook_text"),
                    "reasons": c.get("reasons") or c.get("reason"),
                }
            )

        # Transcript snippet around the top candidate (if available).
        snippet = None
        if top_candidates:
            try:
                from ..analysis_transcript import load_transcript

                tr = load_transcript(proj)
                if tr:
                    start_s = float(top_candidates[0].get("start_s") or 0.0)
                    end_s = float(top_candidates[0].get("end_s") or (start_s + 20.0))
                    win_start = max(0.0, start_s - 4.0)
                    win_end = max(win_start, end_s + 4.0)
                    txt = tr.get_text_in_range(win_start, win_end).strip()
                    if len(txt) > snippet_chars:
                        txt = txt[:snippet_chars].rstrip() + "…"
                    snippet = {"start_s": win_start, "end_s": win_end, "text": txt}
            except Exception:
                snippet = None

        chat_excerpt = None
        if chat_lines > 0:
            try:
                from ..chat.store import ChatStore

                cfg = get_chat_config(proj)
                offset_ms = int(cfg.get("sync_offset_ms", 0) or 0)

                start_s = float(top_candidates[0].get("start_s") or 0.0) if top_candidates else 0.0
                end_s = float(top_candidates[0].get("end_s") or (start_s + 20.0)) if top_candidates else 20.0
                start_ms = int(max(0.0, start_s - 2.0) * 1000)
                end_ms = int(max(0.0, end_s + 2.0) * 1000)

                store = ChatStore(proj.chat_db_path)
                if store.exists:
                    msgs = store.get_messages(start_ms, end_ms, offset_ms=offset_ms, limit=chat_lines)
                    lines = []
                    for m in msgs:
                        t_s = m.t_ms / 1000.0
                        text = (m.text or "").replace("\n", " ").replace("\r", " ").strip()
                        author = (m.author or "").strip()
                        if author:
                            lines.append(f"{t_s:8.1f}s {author}: {text}")
                        else:
                            lines.append(f"{t_s:8.1f}s {text}")
                    chat_excerpt = {"start_s": start_ms / 1000.0, "end_s": end_ms / 1000.0, "lines": lines}
            except Exception:
                chat_excerpt = None

        return JSONResponse(
            {
                "meta": meta,
                "highlights": {"candidates": top_candidates, "total_candidates": len(candidates)},
                "transcript_snippet": snippet,
                "chat_excerpt": chat_excerpt,
            }
        )

    @router.post("/annotations", openapi_extra={"x-openai-isConsequential": False})
    def annotations(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        _rate_limit(request)

        project_id = _validate_project_id(str(body.get("project_id") or ""))
        annotations_obj = body.get("annotations") or {}
        if not isinstance(annotations_obj, dict):
            raise HTTPException(status_code=400, detail="invalid_annotations")

        proj, _proj_data = _load_project(project_id)
        out_path = proj.analysis_dir / "actions_annotations.json"
        payload = {"updated_at": time.time(), "project_id": project_id, "annotations": annotations_obj}
        save_json(out_path, payload)

        # Also record a pointer in project.json (small + safe).
        from ..project import update_project, utc_now_iso

        def _upd(d: Dict[str, Any]) -> None:
            d.setdefault("analysis", {})
            d["analysis"].setdefault("actions", {})
            d["analysis"]["actions"]["annotations_path"] = str(out_path.relative_to(proj.project_dir))
            d["analysis"]["actions"]["updated_at"] = utc_now_iso()

        update_project(proj, _upd)

        return JSONResponse({"ok": True, "project_id": project_id})

    @router.post("/export_batch", openapi_extra={"x-openai-isConsequential": True})
    def export_batch(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        project_id = _validate_project_id(str(body.get("project_id") or ""))
        proj, proj_data = _load_project(project_id)
        if not Path(proj.video_path).exists():
            raise HTTPException(status_code=409, detail="video_not_ready")

        selection_ids = body.get("selection_ids") or []
        top_from_candidates = body.get("top")  # optional

        export_cfg = {**(profile.get("export", {}) or {}), **(body.get("export") or {})}
        cap_cfg = {**(profile.get("captions", {}) or {}), **(body.get("captions") or {})}
        hook_cfg = {**(profile.get("overlay", {}).get("hook_text", {}) or {}), **(body.get("hook_text") or {})}
        pip_cfg = {**(profile.get("layout", {}).get("pip", {}) or {}), **(body.get("pip") or {})}
        with_captions = bool(body.get("with_captions", cap_cfg.get("enabled", False)))

        whisper_cfg = None
        if with_captions:
            from ..transcribe import TranscribeConfig

            whisper_cfg = TranscribeConfig(
                model_size=str(cap_cfg.get("model_size", "small")),
                language=cap_cfg.get("language"),
                device=str(cap_cfg.get("device", "cpu")),
                compute_type=str(cap_cfg.get("compute_type", "int8")),
            )

        # Resolve selections list; optionally create from candidates.
        selections = list((proj_data.get("selections") or []))
        created_ids: list[str] = []

        if selection_ids:
            sel_set = {str(sid) for sid in selection_ids}
            selections = [s for s in selections if str(s.get("id")) in sel_set]
        elif top_from_candidates is not None:
            from ..project import add_selection_from_candidate

            top_n = _clamp_int(top_from_candidates, default=10, min_v=1, max_v=30)
            template = str(export_cfg.get("template", "vertical_blur"))
            candidates = (
                (proj_data.get("analysis", {}) or {}).get("highlights", {}) or {}
            ).get("candidates") or []
            if not candidates:
                raise HTTPException(status_code=404, detail="no_candidates")

            # Create selections (persisted in project.json).
            for cand in candidates[:top_n]:
                if not isinstance(cand, dict):
                    continue
                title = str(cand.get("title") or "")
                sel_id = add_selection_from_candidate(proj, candidate=cand, template=template, title=title)
                created_ids.append(sel_id)

            proj_data = get_project_data(proj)
            selections = list(proj_data.get("selections") or [])
            sel_set = set(created_ids)
            selections = [s for s in selections if str(s.get("id")) in sel_set]

        if not selections:
            raise HTTPException(status_code=404, detail="no_selections")

        director_results = (proj_data.get("analysis", {}) or {}).get("director", {}).get("results") or []

        job = JOB_MANAGER.create("export_batch")
        JOB_MANAGER._set(job, message="queued", result={"project_id": project_id, "created_selection_ids": created_ids})

        @with_prevent_sleep("Exporting batch (Actions)")
        def runner() -> None:
            try:
                total = max(1, len(selections))
                outputs: list[str] = []
                JOB_MANAGER._set(job, status="running", progress=0.0, message=f"exporting 0/{total}")

                for idx, selection in enumerate(selections, start=1):
                    if job.cancel_requested:
                        JOB_MANAGER._set(job, status="cancelled", message="cancelled")
                        return

                    subjob = JOB_MANAGER.start_export(
                        proj=proj,
                        selection=selection,
                        export_dir=proj.exports_dir,
                        with_captions=with_captions,
                        template=str(export_cfg.get("template", selection.get("template") or "vertical_blur")),
                        width=int(export_cfg.get("width", 1080)),
                        height=int(export_cfg.get("height", 1920)),
                        fps=int(export_cfg.get("fps", 30)),
                        crf=int(export_cfg.get("crf", 20)),
                        preset=str(export_cfg.get("preset", "veryfast")),
                        normalize_audio=bool(export_cfg.get("normalize_audio", False)),
                        whisper_cfg=whisper_cfg,
                        hook_cfg=hook_cfg,
                        pip_cfg=pip_cfg,
                        director_results=director_results,
                    )

                    # Wait for subjob completion and mirror progress.
                    while True:
                        time.sleep(0.2)
                        sj = JOB_MANAGER.get(subjob.id)
                        if sj is None:
                            break
                        if sj.status in {"succeeded", "failed", "cancelled"}:
                            break
                        frac = (idx - 1 + sj.progress) / total
                        JOB_MANAGER._set(job, progress=frac, message=f"exporting {idx}/{total}")

                    sj = JOB_MANAGER.get(subjob.id)
                    if sj and sj.status == "failed":
                        raise RuntimeError(sj.message)
                    if sj and sj.status == "succeeded":
                        out = (sj.result or {}).get("output")
                        if out:
                            outputs.append(str(out))

                    JOB_MANAGER._set(job, progress=idx / total, message=f"exporting {idx}/{total}")

                # Write a small manifest for Actions.
                manifest_path = proj.exports_dir / f"actions_export_manifest_{int(time.time())}.json"
                manifest = {
                    "project_id": project_id,
                    "created_selection_ids": created_ids,
                    "outputs": outputs,
                    "export": {
                        "template": str(export_cfg.get("template", "vertical_blur")),
                        "width": int(export_cfg.get("width", 1080)),
                        "height": int(export_cfg.get("height", 1920)),
                        "fps": int(export_cfg.get("fps", 30)),
                        "crf": int(export_cfg.get("crf", 20)),
                        "preset": str(export_cfg.get("preset", "veryfast")),
                        "with_captions": bool(with_captions),
                    },
                    "generated_at": time.time(),
                }
                save_json(manifest_path, manifest)

                JOB_MANAGER._set(
                    job,
                    status="succeeded",
                    progress=1.0,
                    message="done",
                    result={
                        "project_id": project_id,
                        "created_selection_ids": created_ids,
                        "outputs": outputs,
                        "manifest_path": str(manifest_path),
                    },
                )
            except Exception as e:
                if not job.cancel_requested:
                    JOB_MANAGER._set(job, status="failed", message=f"{type(e).__name__}: {e}", result={"project_id": project_id})

        import threading

        threading.Thread(target=runner, daemon=True).start()

        payload = {"job_id": job.id, "project_id": project_id, "created_selection_ids": created_ids}
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.get("/diagnostics", openapi_extra={"x-openai-isConsequential": False})
    def diagnostics(request: Request):
        _rate_limit(request)
        from ..transcription import get_available_backends
        from ..chat.downloader import get_supported_platforms, get_twitch_downloader_info, is_chat_download_available

        backends = get_available_backends()
        return JSONResponse(
            {
                "token_required": bool(os.environ.get("VP_API_TOKEN", "").strip()),
                "transcription": {"backends": backends},
                "chat": {
                    "available": bool(is_chat_download_available()),
                    "platforms": get_supported_platforms(),
                    "twitch_downloader": get_twitch_downloader_info(),
                },
                "paths": {
                    "projects_root": str(default_projects_root()),
                },
                "allowed_domains": sorted(allow_domains),
            }
        )

    return router
