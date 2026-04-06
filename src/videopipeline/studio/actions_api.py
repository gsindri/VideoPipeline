from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import source_inbox as source_inbox_mod
from .. import source_scout as source_scout_mod
from ..ai.helpers import get_llm_complete_fn
from ..analysis import run_analysis
from ..analysis_director import DirectorConfig
from ..analysis_highlights import CONTENT_TYPE_GUIDANCE
from ..ffmpeg import extract_video_frame_jpeg
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
from ..publisher.account_auth import get_publish_account_auth
from ..publisher.accounts import AccountStore
from ..publisher.connectors import get_connector
from ..publisher.jobs import PublishJobStore
from ..publisher.secrets import load_tokens
from .dag_config import (
    apply_llm_mode_to_dag_config,
    build_dag_config,
    dag_config_needs_llm,
    llm_mode_is_strict_external,
    llm_mode_uses_local,
)
from .dag_config import (
    normalize_llm_mode as _normalize_llm_mode_raw,
)
from .dag_config import (
    profile_default_llm_mode as _profile_default_llm_mode_raw,
)
from .dag_config import (
    resolve_llm_mode as _resolve_llm_mode_raw,
)
from .jobs import JOB_MANAGER, with_prevent_sleep
from .publisher_api import is_safe_export_path, scan_project_exports

_PROJECT_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_YOUTUBE_ID_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)")
_TWITCH_VOD_ID_RE = re.compile(r"twitch\.tv/videos/(\d+)")
_CLIP_REVIEW_FRAME_DIR = "clip_review_frames"
_CLIP_REVIEW_MAX_FRAME_CLIPS = 5
_CLIP_REVIEW_MAX_FRAMES_PER_CLIP = 6
_CLIP_REVIEW_FRAME_WIDTH = 320
_CLIP_REVIEW_FRAME_QUALITY = 6


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def _build_project_content_key(
    url: str,
    *,
    fresh_project: bool = False,
    client_request_id: str = "",
) -> str:
    base_content_id = _extract_content_id(url)
    if not fresh_project:
        return base_content_id
    fresh_nonce = client_request_id.strip() or uuid.uuid4().hex
    return f"{base_content_id}::fresh::{fresh_nonce}"


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


def _sanitize_clip_review_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-")
    return cleaned or "clip"


def _resolve_clip_review_video_path(proj: Project) -> Tuple[Optional[Path], Optional[str]]:
    preview_path = proj.preview_video_path
    if preview_path.exists():
        return preview_path, "preview_video"
    if proj.video_path.exists():
        return proj.video_path, "source_video"
    return None, None


def _clip_review_frame_times(
    *,
    start_s: float,
    end_s: float,
    peak_time_s: Optional[float],
    frame_count: int,
) -> list[float]:
    if frame_count <= 0:
        return []
    duration = max(0.05, end_s - start_s)
    if frame_count == 1:
        return [max(start_s, min(end_s, peak_time_s if peak_time_s is not None else start_s + duration * 0.5))]

    times: list[float] = []
    for idx in range(frame_count):
        ratio = 0.12 + (0.76 * idx / max(1, frame_count - 1))
        times.append(start_s + duration * ratio)

    if peak_time_s is not None and start_s <= peak_time_s <= end_s:
        times[len(times) // 2] = peak_time_s

    unique: list[float] = []
    seen = set()
    for value in times:
        clamped = max(start_s, min(end_s, value))
        key = int(round(clamped * 1000.0))
        if key in seen:
            continue
        seen.add(key)
        unique.append(clamped)
    return unique


def _build_clip_review_frame_payload(
    *,
    proj: Project,
    candidate: Dict[str, Any],
    review_id: str,
    frames_per_clip: int,
) -> Dict[str, Any]:
    video_path, source_label = _resolve_clip_review_video_path(proj)
    if video_path is None or not source_label:
        return {
            "frame_source": None,
            "frame_count": 0,
            "frames": [],
        }

    try:
        start_s = float(candidate.get("start_s"))
    except Exception:
        start_s = 0.0
    try:
        end_s = float(candidate.get("end_s"))
    except Exception:
        end_s = start_s
    if end_s <= start_s:
        end_s = start_s + 0.1
    try:
        peak_time_s = float(candidate.get("peak_time_s"))
    except Exception:
        peak_time_s = None

    frame_times = _clip_review_frame_times(
        start_s=start_s,
        end_s=end_s,
        peak_time_s=peak_time_s,
        frame_count=frames_per_clip,
    )
    if not frame_times:
        return {
            "frame_source": source_label,
            "frame_count": 0,
            "frames": [],
        }

    review_token = _sanitize_clip_review_token(review_id)
    frames_dir = proj.analysis_dir / _CLIP_REVIEW_FRAME_DIR / review_token
    labels = ["opening", "setup", "midpoint", "payoff", "ending", "button"]
    frame_items: list[Dict[str, Any]] = []
    for idx, time_s in enumerate(frame_times):
        output_path = frames_dir / f"{review_token}-{idx + 1:02d}.jpg"
        try:
            if not output_path.exists():
                extract_video_frame_jpeg(
                    video_path,
                    output_path=output_path,
                    time_seconds=time_s,
                    width=_CLIP_REVIEW_FRAME_WIDTH,
                    quality=_CLIP_REVIEW_FRAME_QUALITY,
                )
            encoded = base64.b64encode(output_path.read_bytes()).decode("ascii")
        except Exception:
            continue
        frame_items.append(
            {
                "frame_id": f"{review_token}-frame-{idx + 1}",
                "label": labels[idx] if idx < len(labels) else f"frame_{idx + 1}",
                "time_s": round(time_s, 3),
                "file_name": output_path.name,
                "mime_type": "image/jpeg",
                "relative_path": str(output_path.relative_to(proj.project_dir)),
                "content_base64": encoded,
            }
        )

    return {
        "frame_source": source_label,
        "frame_count": len(frame_items),
        "frames": frame_items,
    }


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


def _profile_env_present(*names: str) -> bool:
    for name in names:
        if (os.environ.get(name) or "").strip():
            return True
    return False


def _diagnostics_transcription_backends() -> Dict[str, bool]:
    torch_available = importlib.util.find_spec("torch") is not None
    # Keep diagnostics cheap and side-effect free: importing torch on some
    # Windows ROCm setups can hang inside platform/WMI detection.
    gpu_stack_present = torch_available

    whispercpp_available = importlib.util.find_spec("pywhispercpp") is not None
    faster_whisper_available = importlib.util.find_spec("faster_whisper") is not None
    openai_whisper_available = importlib.util.find_spec("whisper") is not None
    nemo_available = importlib.util.find_spec("nemo") is not None and torch_available
    assemblyai_available = importlib.util.find_spec("assemblyai") is not None and _profile_env_present(
        "ASSEMBLYAI_API_KEY",
        "AAI_API_KEY",
    )

    return {
        "whispercpp": whispercpp_available,
        "whispercpp_gpu": False,
        "faster_whisper": faster_whisper_available,
        "faster_whisper_gpu": faster_whisper_available and gpu_stack_present,
        "openai_whisper": openai_whisper_available,
        "openai_whisper_gpu": openai_whisper_available and gpu_stack_present,
        "nemo_asr": nemo_available,
        "nemo_asr_gpu": nemo_available and gpu_stack_present,
        "assemblyai": assemblyai_available,
        "assemblyai_gpu": False,
    }


def _profile_readiness(profile: Dict[str, Any], *, profile_path: Optional[Path], backends: Dict[str, bool]) -> Dict[str, Any]:
    from ..analysis_audio_events import AudioEventsConfig, check_assemblyai_audio_events_available

    analysis_cfg = (profile.get("analysis", {}) or {}) if isinstance(profile, dict) else {}
    speech_cfg = (analysis_cfg.get("speech", {}) or {}) if isinstance(analysis_cfg, dict) else {}
    audio_events_cfg = (analysis_cfg.get("audio_events", {}) or {}) if isinstance(analysis_cfg, dict) else {}
    diarization_cfg = (analysis_cfg.get("diarization", {}) or {}) if isinstance(analysis_cfg, dict) else {}
    chat_cfg = (analysis_cfg.get("chat", {}) or {}) if isinstance(analysis_cfg, dict) else {}
    highlights_cfg = (analysis_cfg.get("highlights", {}) or {}) if isinstance(analysis_cfg, dict) else {}

    issues: list[str] = []

    speech_backend = str(speech_cfg.get("backend", "auto") or "auto")
    speech_use_gpu = bool(speech_cfg.get("use_gpu", False))
    speech_ready = True
    speech_reason: Optional[str] = None

    if speech_backend == "assemblyai":
        speech_ready = bool(backends.get("assemblyai", False))
        if not speech_ready:
            speech_reason = "AssemblyAI transcription backend unavailable (install SDK and set ASSEMBLYAI_API_KEY)"
    elif speech_backend == "nemo_asr":
        speech_ready = bool(backends.get("nemo_asr", False))
        if speech_ready and speech_use_gpu and not bool(backends.get("nemo_asr_gpu", False)):
            speech_ready = False
            speech_reason = "NeMo ASR requested with GPU but CUDA backend is unavailable"
        elif not speech_ready:
            speech_reason = "NeMo ASR backend unavailable"
    elif speech_backend == "auto":
        speech_ready = any(
            bool(backends.get(name, False))
            for name in ("nemo_asr", "openai_whisper", "faster_whisper", "whispercpp")
        )
        if not speech_ready:
            speech_reason = "No local transcription backend available for auto mode"
    elif speech_backend in backends:
        speech_ready = bool(backends.get(speech_backend, False))
        if not speech_ready:
            speech_reason = f"{speech_backend} backend unavailable"

    if not speech_ready and speech_reason:
        issues.append(f"speech: {speech_reason}")

    events_diag: Dict[str, Any] = {
        "backend": str(audio_events_cfg.get("backend", "auto") or "auto"),
        "strict": bool(audio_events_cfg.get("strict", False)),
        "ready": True,
        "reason": None,
    }
    if events_diag["backend"] == "assemblyai":
        events_cfg = AudioEventsConfig.from_dict(audio_events_cfg)
        ok, reason = check_assemblyai_audio_events_available(events_cfg.assemblyai_api_key)
        events_diag["ready"] = bool(ok)
        events_diag["reason"] = reason
        if not ok:
            issues.append(f"audio_events: {reason or 'AssemblyAI audio events unavailable'}")
    elif events_diag["backend"] == "heuristic":
        events_diag["ready"] = True

    standalone_diarization = bool((diarization_cfg.get("enabled", False)))
    speech_diarization = bool(speech_cfg.get("diarize", False))
    needs_hf_token = bool(standalone_diarization or (speech_diarization and speech_backend != "assemblyai"))
    hf_token_present = _profile_env_present("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    diarization_ready = True
    diarization_reason: Optional[str] = None
    if needs_hf_token and not hf_token_present:
        diarization_ready = False
        diarization_reason = "HF_TOKEN not set for pyannote diarization"
        issues.append(f"diarization: {diarization_reason}")

    return {
        "path": str(profile_path) if profile_path is not None else None,
        "env_selected": bool((os.environ.get("VP_STUDIO_PROFILE") or "").strip()),
        "fail_fast": bool(analysis_cfg.get("fail_fast", False)),
        "chat_llm_strict": bool(chat_cfg.get("llm_strict", False)),
        "highlights_llm_semantic_enabled": bool(highlights_cfg.get("llm_semantic_enabled", False)),
        "speech": {
            "backend": speech_backend,
            "strict": bool(speech_cfg.get("strict", False)),
            "use_gpu": speech_use_gpu,
            "ready": speech_ready,
            "reason": speech_reason,
        },
        "audio_events": events_diag,
        "diarization": {
            "standalone_enabled": standalone_diarization,
            "speech_enabled": speech_diarization,
            "requires_hf_token": needs_hf_token,
            "hf_token_present": hf_token_present,
            "ready": diarization_ready,
            "reason": diarization_reason,
        },
        "readiness": {
            "ok": not issues,
            "issues": issues,
        },
    }


def create_actions_router(
    *,
    profile: Dict[str, Any],
    profile_path: Optional[Path] = None,
    account_store: Optional[AccountStore] = None,
    job_store: Optional[PublishJobStore] = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/actions", tags=["actions"])
    account_store = account_store or AccountStore()
    job_store = job_store or PublishJobStore()

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

    def _normalize_llm_mode(value: Any) -> str:
        try:
            return _normalize_llm_mode_raw(value)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid_llm_mode")

    def _profile_default_llm_mode() -> str:
        try:
            return _profile_default_llm_mode_raw(profile)
        except ValueError:
            raise HTTPException(status_code=500, detail="invalid_profile_default_llm_mode")

    def _resolve_llm_mode(value: Any) -> str:
        try:
            return _resolve_llm_mode_raw(value, profile=profile)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid_llm_mode")

    _LLM_MODE_ENUM = ["local", "external", "external_strict"]
    _EXTERNAL_AI_SOURCES = {"chatgpt_actions", "external_ai", "gondull"}

    def _extract_ai_provenance(
        body: Dict[str, Any],
        *,
        client_request_id: str,
        default_source: str = "chatgpt_actions",
    ) -> Dict[str, Any]:
        raw = body.get("provenance") or {}
        if not isinstance(raw, dict):
            raw = {}

        merged = dict(raw)
        for key in (
            "agent",
            "provider",
            "model",
            "model_family",
            "prompt_id",
            "prompt_version",
            "request_id",
            "run_id",
            "notes",
            "source",
        ):
            if key not in merged and body.get(key) is not None:
                merged[key] = body.get(key)

        out: Dict[str, Any] = {"source": str(merged.get("source") or default_source).strip() or default_source}
        for key in (
            "agent",
            "provider",
            "model",
            "model_family",
            "prompt_id",
            "prompt_version",
            "request_id",
            "run_id",
            "notes",
        ):
            val = merged.get(key)
            if val is None:
                continue
            text = str(val).strip()
            if text:
                out[key] = text
        if client_request_id:
            out["client_request_id"] = client_request_id
        return out

    def _profile_external_ai_requirements() -> Dict[str, bool]:
        analysis_cfg = profile.get("analysis", {}) or {}
        highlights_cfg = analysis_cfg.get("highlights", {}) or {}
        chapters_cfg = analysis_cfg.get("chapters", {}) or {}
        ai_cfg = (profile.get("ai", {}) or {}).get("director", {}) or {}
        director_cfg = analysis_cfg.get("director", {}) or {}
        director_enabled = bool(director_cfg.get("enabled", ai_cfg.get("enabled", True)))
        return {
            "semantic": bool(highlights_cfg.get("llm_semantic_enabled", True) or highlights_cfg.get("llm_filter_enabled", False)),
            "chapters": bool(chapters_cfg.get("enabled", True) and chapters_cfg.get("llm_labeling", True)),
            "director": director_enabled,
        }

    def _external_ai_status(*, proj: Project, proj_data: Dict[str, Any]) -> Dict[str, Any]:
        requirements = _profile_external_ai_requirements()
        analysis_meta = proj_data.get("analysis", {}) or {}
        actions_meta = analysis_meta.get("actions", {}) or {}
        chapters_meta = analysis_meta.get("chapters", {}) or {}
        director_meta = analysis_meta.get("director", {}) or {}
        highlights_meta = analysis_meta.get("highlights", {}) or {}
        candidates = [c for c in (highlights_meta.get("candidates") or []) if isinstance(c, dict)]

        semantic_count = 0
        for cand in candidates:
            ai = cand.get("ai")
            if not isinstance(ai, dict):
                continue
            source = str(ai.get("semantic_source") or "").strip()
            if source in _EXTERNAL_AI_SOURCES:
                semantic_count += 1

        chapters_path = proj.analysis_dir / "chapters.json"
        chapter_count = 0
        labeled_chapter_count = 0
        if chapters_path.exists():
            try:
                chapters_payload = json.loads(chapters_path.read_text(encoding="utf-8"))
                chapters_items = [c for c in (chapters_payload.get("chapters") or []) if isinstance(c, dict)]
                chapter_count = len(chapters_items)
                labeled_chapter_count = sum(
                    1
                    for ch in chapters_items
                    if str(ch.get("labels_source") or "").strip() in _EXTERNAL_AI_SOURCES
                )
            except Exception:
                chapter_count = 0
                labeled_chapter_count = 0

        director_path = proj.analysis_dir / "director.json"
        director_pick_count = 0
        director_source = str(director_meta.get("source") or "").strip()
        director_provenance = director_meta.get("provenance")
        if not isinstance(director_provenance, dict):
            director_provenance = None
        if director_path.exists():
            try:
                director_payload = json.loads(director_path.read_text(encoding="utf-8"))
                director_pick_count = int(director_payload.get("pick_count") or 0)
                director_source = str(
                    director_source
                    or ((director_payload.get("config") or {}).get("source") or director_payload.get("source") or "")
                ).strip()
                payload_provenance = director_payload.get("provenance")
                if isinstance(payload_provenance, dict):
                    director_provenance = payload_provenance
            except Exception:
                director_pick_count = 0

        semantic_done = (not requirements["semantic"]) or (
            bool(actions_meta.get("semantic_applied_at"))
            and (len(candidates) == 0 or semantic_count >= len(candidates))
        )
        chapters_done = (not requirements["chapters"]) or (
            bool(chapters_meta.get("labels_updated_at"))
            and chapter_count > 0
            and labeled_chapter_count >= chapter_count
        )
        director_done = (not requirements["director"]) or (
            director_pick_count > 0
            and (
                director_source in _EXTERNAL_AI_SOURCES
                or str(((director_provenance or {}).get("source")) or "").strip() in _EXTERNAL_AI_SOURCES
            )
        )

        issues: list[str] = []
        if requirements["semantic"] and not semantic_done:
            if candidates:
                issues.append(f"semantic decisions incomplete ({semantic_count}/{len(candidates)})")
            else:
                issues.append("semantic decisions have not been applied")
        if requirements["chapters"] and not chapters_done:
            if chapter_count <= 0:
                issues.append("chapter labels are required but chapters are not available yet")
            else:
                issues.append(f"chapter labels incomplete ({labeled_chapter_count}/{chapter_count})")
        if requirements["director"] and not director_done:
            issues.append("director picks with external provenance are required before export")

        return {
            "required": requirements,
            "completed": {
                "semantic": semantic_done,
                "chapters": chapters_done,
                "director": director_done,
            },
            "details": {
                "semantic": {
                    "applied_at": actions_meta.get("semantic_applied_at"),
                    "candidate_count": len(candidates),
                    "externally_scored_candidates": semantic_count,
                },
                "chapters": {
                    "labels_updated_at": chapters_meta.get("labels_updated_at"),
                    "chapter_count": chapter_count,
                    "externally_labeled_chapters": labeled_chapter_count,
                },
                "director": {
                    "pick_count": director_pick_count,
                    "source": director_source or None,
                    "provenance": director_provenance,
                },
            },
            "strict_export_ready": not issues,
            "issues": issues,
        }

    def _require_external_ai_ready_for_export(*, project_id: str, proj: Project, proj_data: Dict[str, Any], llm_mode: str) -> None:
        if not llm_mode_is_strict_external(llm_mode):
            return
        status = _external_ai_status(proj=proj, proj_data=proj_data)
        if status["strict_export_ready"]:
            return
        raise HTTPException(
            status_code=409,
            detail={
                "code": "external_ai_incomplete",
                "project_id": project_id,
                "issues": status["issues"],
                "status": status,
            },
        )

    def _publish_policy() -> Dict[str, Any]:
        return {
            "default_privacy": "private",
            "allow_without_approval": ["private", "unlisted"],
            "requires_explicit_approval_for": ["public", "scheduled_release"],
        }

    def _build_publish_accounts_payload() -> Dict[str, Any]:
        accounts = []
        ready_total = 0
        for acct in account_store.list():
            auth = get_publish_account_auth(acct)
            if auth.ready:
                ready_total += 1
            item = acct.to_dict()
            item["ready"] = auth.ready
            item["has_tokens"] = auth.has_tokens
            item["auth_state"] = auth.auth_state
            item["auth_error"] = auth.auth_error
            accounts.append(item)
        accounts.sort(key=lambda item: (str(item.get("platform") or ""), str(item.get("label") or "")))
        return {
            "accounts": accounts,
            "summary": {
                "accounts_total": len(accounts),
                "ready_accounts": ready_total,
            },
            "policy": _publish_policy(),
        }

    def _build_publish_exports_payload(*, project_id: str) -> Dict[str, Any]:
        pid = _validate_project_id(project_id)
        proj, proj_data = _load_project(pid)
        exports = scan_project_exports(proj.exports_dir)
        items = []
        for exp in exports:
            item = exp.to_dict()
            metadata = dict(exp.metadata or {})
            item["privacy"] = str(metadata.get("privacy") or _publish_policy()["default_privacy"]).strip().lower() or "private"
            item["publish_at"] = str(metadata.get("publish_at") or "").strip() or None
            items.append(item)
        return {
            "meta": {
                "project_id": pid,
                "title": proj_data.get("title") or (proj_data.get("video", {}) or {}).get("title"),
                "exports_dir": str(proj.exports_dir),
            },
            "exports": items,
            "policy": _publish_policy(),
        }

    def _resolve_publish_metadata(*, export_id: str, base_metadata: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(base_metadata or {})
        if options.get("title_override"):
            metadata["title"] = str(options["title_override"]).strip()
        if options.get("description_override"):
            metadata["description"] = str(options["description_override"]).strip()
        if options.get("privacy") is not None:
            metadata["privacy"] = str(options.get("privacy") or "").strip().lower()
        if options.get("publish_at") is not None:
            publish_at = str(options.get("publish_at") or "").strip()
            metadata["publish_at"] = publish_at or None
        if options.get("hashtags_append"):
            existing = str(metadata.get("description") or "").strip()
            extra = str(options["hashtags_append"]).strip()
            metadata["description"] = f"{existing}\n\n{extra}".strip() if existing else extra
        if not str(metadata.get("title") or "").strip():
            metadata["title"] = export_id
        privacy = str(metadata.get("privacy") or _publish_policy()["default_privacy"]).strip().lower() or "private"
        if privacy not in {"private", "unlisted", "public"}:
            raise HTTPException(status_code=400, detail="invalid_publish_privacy")
        metadata["privacy"] = privacy
        publish_at = str(metadata.get("publish_at") or "").strip()
        if not publish_at:
            metadata.pop("publish_at", None)
        return metadata

    def _require_publish_approval(*, metadata: Dict[str, Any], approved: bool) -> None:
        privacy = str(metadata.get("privacy") or _publish_policy()["default_privacy"]).strip().lower() or "private"
        publish_at = str(metadata.get("publish_at") or "").strip()
        if privacy != "public" and not publish_at:
            return
        if approved:
            return
        issues = []
        if privacy == "public":
            issues.append("privacy=public")
        if publish_at:
            issues.append("publish_at")
        raise HTTPException(
            status_code=409,
            detail={
                "code": "public_release_approval_required",
                "issues": issues,
                "policy": _publish_policy(),
            },
        )

    def _filter_publish_jobs_for_project(*, project_id: str, limit: int) -> Dict[str, Any]:
        pid = _validate_project_id(project_id)
        proj, _proj_data = _load_project(pid)
        exports_dir = proj.exports_dir.resolve()
        jobs = []
        scan_limit = max(limit, 200)
        for job in job_store.list_jobs(limit=scan_limit):
            try:
                file_path = Path(job.file_path).resolve()
            except Exception:
                continue
            try:
                if not file_path.is_relative_to(exports_dir):
                    continue
            except Exception:
                continue
            jobs.append(job.to_dict())
            if len(jobs) >= limit:
                break
        return {
            "project_id": pid,
            "jobs": jobs,
        }

    def _extract_scout_metadata(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = body.get("scout") or {}
        if not isinstance(raw, dict):
            return None

        out: Dict[str, Any] = {}
        for key in (
            "source_id",
            "source_label",
            "source_url",
            "candidate_id",
            "content_key",
            "inbox_id",
            "watchlist_path",
            "selection_mode",
        ):
            val = raw.get(key)
            if val is None:
                continue
            text = str(val).strip()
            if text:
                out[key] = text

        score = raw.get("score")
        if score is not None:
            try:
                out["score"] = float(score)
            except Exception:
                pass

        if raw.get("shadow_mode") is not None:
            out["shadow_mode"] = bool(raw.get("shadow_mode"))

        reasons = raw.get("reasons") or []
        if isinstance(reasons, list):
            cleaned = [str(item).strip() for item in reasons if str(item).strip()]
            if cleaned:
                out["reasons"] = cleaned

        return out or None

    def _persist_project_scout_metadata(*, proj: Project, url: str, body: Dict[str, Any]) -> None:
        scout = _extract_scout_metadata(body)
        if not scout:
            return

        from ..project import update_project, utc_now_iso

        def _upd(data: Dict[str, Any]) -> None:
            data.setdefault("source", {})
            source_block = data.get("source")
            if not isinstance(source_block, dict):
                source_block = {}
                data["source"] = source_block
            source_block["source_url"] = url
            source_block["scout"] = {
                **scout,
                "selected_at": utc_now_iso(),
            }

        update_project(proj, _upd)
        inbox_id = str(scout.get("inbox_id") or "").strip()
        if inbox_id:
            try:
                source_inbox_mod.update_source_inbox_entry(
                    inbox_id,
                    status="selected",
                    project_id=proj.project_dir.name,
                    selection_notes=str(scout.get("selection_mode") or "pipeline_ingest"),
                )
            except Exception:
                pass

    def _build_source_inbox_payload(*, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        limit = max(1, int(limit))
        resolved, entries = source_inbox_mod.list_source_inbox_entries(status=status, limit=limit)
        return {
            "meta": {
                "inbox_path": str(resolved) if resolved is not None else None,
                "status_filter": status or None,
                "limit": limit,
                "entry_count": len(entries),
            },
            "entries": entries,
        }

    def _source_scout_diagnostics() -> Dict[str, Any]:
        twitch_api_ready = source_scout_mod.twitch_api_configured()
        try:
            watchlist_path, watchlist = source_scout_mod.load_source_watchlist()
        except Exception as exc:
            inbox_path, pending_entries = source_inbox_mod.list_source_inbox_entries(status="pending")
            return {
                "configured": False,
                "ready": False,
                "twitch_api_configured": twitch_api_ready,
                "twitch_provider_sources": 0,
                "twitch_helix_ready_sources": 0,
                "watchlist_path": None,
                "shadow_mode": True,
                "enabled_sources": 0,
                "inbox_path": str(inbox_path) if inbox_path is not None else None,
                "inbox_pending": len(pending_entries),
                "issues": [f"{type(exc).__name__}: {exc}"],
            }

        enabled_sources = [
            item for item in (watchlist.get("sources") or []) if bool(item.get("enabled", True))
        ]
        inbox_path, pending_entries = source_inbox_mod.list_source_inbox_entries(status="pending")
        issues: list[str] = []
        fetch_ready_sources = 0
        twitch_provider_sources = 0
        twitch_helix_ready_sources = 0
        for item in enabled_sources:
            provider = str(item.get("provider") or "").strip().lower()
            if provider in {"twitch_helix", "twitch_api"}:
                twitch_provider_sources += 1
            provider_issue = source_scout_mod.source_preflight_issue(item)
            if provider_issue:
                label = str(item.get("label") or item.get("id") or item.get("url") or "source").strip()
                issues.append(f"{label}: {provider_issue}")
            else:
                fetch_ready_sources += 1
                if provider in {"twitch_helix", "twitch_api"}:
                    twitch_helix_ready_sources += 1
        if watchlist_path is None:
            issues.append("source watchlist not found")
        if not fetch_ready_sources and not pending_entries:
            issues.append("no enabled scout sources or pending manual inbox entries")
        return {
            "configured": watchlist_path is not None,
            "ready": bool((watchlist_path is not None and fetch_ready_sources) or pending_entries),
            "twitch_api_configured": twitch_api_ready,
            "twitch_provider_sources": twitch_provider_sources,
            "twitch_helix_ready_sources": twitch_helix_ready_sources,
            "watchlist_path": str(watchlist_path) if watchlist_path is not None else None,
            "shadow_mode": bool(watchlist.get("shadow_mode", True)),
            "enabled_sources": len(enabled_sources),
            "fetch_ready_sources": fetch_ready_sources,
            "inbox_path": str(inbox_path) if inbox_path is not None else None,
            "inbox_pending": len(pending_entries),
            "issues": issues,
        }

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
                            "llm_mode": {
                                "type": "string",
                                "enum": list(_LLM_MODE_ENUM),
                                "description": "LLM mode for analysis: local uses in-app AI, external skips in-app AI, external_strict skips in-app AI and requires external AI completion before export. If omitted, the active profile default is used (external_strict when unspecified).",
                            },
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
                            "llm_mode": {
                                "type": "string",
                                "enum": list(_LLM_MODE_ENUM),
                                "description": "LLM mode for analysis: local uses in-app AI, external skips in-app AI, external_strict skips in-app AI and requires the external AI checkpoint before export. If omitted, the active profile default is used (external_strict when unspecified).",
                            },
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
                            "llm_mode": {
                                "type": "string",
                                "enum": list(_LLM_MODE_ENUM),
                                "description": "LLM mode for unattended runs: local uses in-app AI, external skips in-app AI, external_strict rejects unattended export so external AI must checkpoint first. If omitted, the active profile default is used (external_strict when unspecified).",
                            },
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
                    "JobWaitResponse": {
                        "type": "object",
                        "properties": {
                            "done": {"type": "boolean"},
                            "id": {"type": "string"},
                            "kind": {"type": "string"},
                            "created_at": {"type": "string"},
                            "status": {"type": "string"},
                            "progress": {"type": "number"},
                            "message": {"type": "string"},
                            "result": {"type": "object", "additionalProperties": True},
                            "poll_after_ms": {"type": "integer"},
                        },
                        "required": ["done", "id", "kind", "created_at", "status", "progress", "message", "result", "poll_after_ms"],
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
                    "PublishAccountsResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "PublishExportsResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "PublishQueueRequest": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "account_ids": {"type": "array", "items": {"type": "string"}},
                            "export_ids": {"type": "array", "items": {"type": "string"}},
                            "options": {"type": "object", "additionalProperties": True},
                            "public_release_approved": {"type": "boolean"},
                            "client_request_id": {"type": "string"},
                        },
                        "required": ["project_id", "account_ids", "export_ids"],
                        "additionalProperties": False,
                    },
                    "PublishQueueResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "PublishJobsResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "ScoutCandidatesResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "ScoutInboxResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "ScoutInboxAddRequest": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "title": {"type": "string"},
                            "notes": {"type": "string"},
                            "priority": {"type": "number"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "added_by": {"type": "string"},
                            "source_id": {"type": "string"},
                            "source_label": {"type": "string"},
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                    "ScoutInboxMarkRequest": {
                        "type": "object",
                        "properties": {
                            "inbox_id": {"type": "string"},
                            "status": {"type": "string"},
                            "project_id": {"type": "string"},
                            "selection_notes": {"type": "string"},
                        },
                        "required": ["inbox_id", "status"],
                        "additionalProperties": False,
                    },
                    "AiCandidatesResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "AiBundleResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "AiClipReviewResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "AiVariantsResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "AiChaptersResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "AiApplySemanticRequest": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "provenance": {"type": "object", "additionalProperties": True},
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "candidate_id": {"type": "string"},
                                        "candidate_rank": {"type": "integer"},
                                        "semantic_score": {"type": "number"},
                                        "reason": {"type": "string"},
                                        "best_quote": {"type": "string"},
                                        "keep": {"type": "boolean"},
                                    },
                                    "required": ["candidate_rank", "semantic_score"],
                                    "additionalProperties": True,
                                },
                            },
                            "client_request_id": {"type": "string"},
                        },
                        "required": ["project_id", "items"],
                        "additionalProperties": False,
                    },
                    "AiApplySemanticResponse": {
                        "type": "object",
                        "properties": {
                            "ok": {"type": "boolean"},
                            "project_id": {"type": "string"},
                            "updated_count": {"type": "integer"},
                            "missing_ranks": {"type": "array", "items": {"type": "integer"}},
                            "missing_candidate_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["ok", "project_id"],
                        "additionalProperties": True,
                    },
                    "AiApplyChapterLabelsRequest": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "provenance": {"type": "object", "additionalProperties": True},
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "chapter_id": {"type": "integer"},
                                        "id": {"type": "integer"},
                                        "title": {"type": "string"},
                                        "summary": {"type": "string"},
                                        "keywords": {"type": "array", "items": {"type": "string"}},
                                        "type": {"type": "string"},
                                    },
                                    "additionalProperties": True,
                                },
                            },
                            "client_request_id": {"type": "string"},
                        },
                        "required": ["project_id", "items"],
                        "additionalProperties": False,
                    },
                    "AiApplyChapterLabelsResponse": {
                        "type": "object",
                        "properties": {
                            "ok": {"type": "boolean"},
                            "project_id": {"type": "string"},
                            "updated_count": {"type": "integer"},
                            "missing_ids": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["ok", "project_id"],
                        "additionalProperties": True,
                    },
                    "AiApplyDirectorPicksRequest": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "provenance": {"type": "object", "additionalProperties": True},
                            "picks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "candidate_id": {"type": "string"},
                                        "candidate_rank": {"type": "integer"},
                                        "variant_id": {"type": "string"},
                                        "best_variant_id": {"type": "string"},
                                        "title": {"type": "string"},
                                        "hook": {"type": "string"},
                                        "description": {"type": "string"},
                                        "hashtags": {"type": "array", "items": {"type": "string"}},
                                        "tags": {"type": "array", "items": {"type": "string"}},
                                        "template": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "reason": {"type": "string"},
                                        "reasons": {"type": "array", "items": {"type": "string"}},
                                    },
                                    "required": ["candidate_rank", "variant_id"],
                                    "additionalProperties": True,
                                },
                            },
                            "client_request_id": {"type": "string"},
                        },
                        "required": ["project_id", "picks"],
                        "additionalProperties": False,
                    },
                    "AiApplyDirectorPicksResponse": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "ExportDirectorPicksRequest": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "limit": {"type": "integer"},
                            "llm_mode": {
                                "type": "string",
                                "enum": list(_LLM_MODE_ENUM),
                                "description": "If omitted, the active profile default is used (external_strict when unspecified).",
                            },
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
                    "ExportDirectorPicksResponse": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string"},
                            "project_id": {"type": "string"},
                            "created_selection_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["job_id", "project_id"],
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
                                        "external_llm": {
                                            "summary": "Analyze without local LLM (ChatGPT-in-the-loop)",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "llm_mode": "external",
                                                "client_request_id": "analyze-003",
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
                                        },
                                        "twitch_vod_external_llm": {
                                            "summary": "Ingest + analyze (no local LLM calls)",
                                            "value": {
                                                "url": "https://twitch.tv/videos/1637616602",
                                                "llm_mode": "external",
                                                "client_request_id": "run-001-external",
                                            },
                                        },
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
                "/api/actions/run_full_export_top_unattended": {
                    "post": {
                        "summary": "Run ingest + analyze + export top N as a single workflow (no confirmation)",
                        "operationId": "vp_actions_run_full_export_top_unattended",
                        "x-openai-isConsequential": False,
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
                                                "client_request_id": "run-003",
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
                "/api/actions/jobs/{job_id}/wait": {
                    "get": {
                        "summary": "Wait for a job to finish (bounded)",
                        "operationId": "vp_actions_job_wait",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {
                                "name": "job_id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "timeout_s",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "number"},
                            },
                        ],
                        "responses": {
                            "200": {
                                "description": "Job status (possibly still running)",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/JobWaitResponse"}}
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
                "/api/actions/publish/accounts": {
                    "get": {
                        "summary": "List publishing accounts and readiness",
                        "operationId": "vp_actions_publish_accounts",
                        "x-openai-isConsequential": False,
                        "responses": {
                            "200": {
                                "description": "Publishing accounts",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/PublishAccountsResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/publish/exports": {
                    "get": {
                        "summary": "List exports for a project that are ready to publish",
                        "operationId": "vp_actions_publish_exports",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "project_id", "in": "query", "required": True, "schema": {"type": "string"}},
                        ],
                        "responses": {
                            "200": {
                                "description": "Project exports",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/PublishExportsResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/publish/queue": {
                    "post": {
                        "summary": "Queue publish jobs for selected exports/accounts",
                        "operationId": "vp_actions_publish_queue",
                        "x-openai-isConsequential": True,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PublishQueueRequest"},
                                    "examples": {
                                        "private_batch": {
                                            "summary": "Queue safe private uploads",
                                            "value": {
                                                "project_id": "<project_id from export>",
                                                "account_ids": ["<youtube_account_id>"],
                                                "export_ids": ["clip_001", "clip_002"],
                                                "options": {"privacy": "private"},
                                                "client_request_id": "publish-001",
                                            },
                                        },
                                        "public_release_approved": {
                                            "summary": "Queue a public release with explicit approval",
                                            "value": {
                                                "project_id": "<project_id from export>",
                                                "account_ids": ["<youtube_account_id>"],
                                                "export_ids": ["clip_001"],
                                                "options": {"privacy": "public"},
                                                "public_release_approved": True,
                                                "client_request_id": "publish-002",
                                            },
                                        },
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Publish jobs created",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/PublishQueueResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/publish/jobs": {
                    "get": {
                        "summary": "List publish jobs for a project",
                        "operationId": "vp_actions_publish_jobs",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "project_id", "in": "query", "required": True, "schema": {"type": "string"}},
                            {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer", "default": 50}},
                        ],
                        "responses": {
                            "200": {
                                "description": "Publish jobs",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/PublishJobsResponse"}}
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
                "/api/actions/scout/candidates": {
                    "get": {
                        "summary": "Fetch ranked ingest URL candidates from the source watchlist",
                        "operationId": "vp_actions_scout_candidates",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "per_source", "in": "query", "required": False, "schema": {"type": "integer"}},
                        ],
                        "responses": {
                            "200": {
                                "description": "Ranked URL candidates",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/ScoutCandidatesResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/scout/inbox": {
                    "get": {
                        "summary": "List manual scout inbox entries",
                        "operationId": "vp_actions_scout_inbox_list",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "status", "in": "query", "required": False, "schema": {"type": "string"}},
                            {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}},
                        ],
                        "responses": {
                            "200": {
                                "description": "Manual scout inbox entries",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/ScoutInboxResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/scout/inbox/add": {
                    "post": {
                        "summary": "Add a URL to the manual scout inbox",
                        "operationId": "vp_actions_scout_inbox_add",
                        "x-openai-isConsequential": False,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ScoutInboxAddRequest"},
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Added or deduplicated inbox entry",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/ScoutInboxResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/scout/inbox/mark": {
                    "post": {
                        "summary": "Mark a manual scout inbox entry as selected or dismissed",
                        "operationId": "vp_actions_scout_inbox_mark",
                        "x-openai-isConsequential": False,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ScoutInboxMarkRequest"},
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Updated inbox entry",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/ScoutInboxResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/ai/candidates": {
                    "get": {
                        "summary": "Fetch AI-ready highlight candidates",
                        "operationId": "vp_actions_ai_candidates",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "project_id", "in": "query", "required": True, "schema": {"type": "string"}},
                            {"name": "top_n", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chat_top_n", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {
                                "name": "chat_top",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "integer"},
                                "description": "Alias for chat_top_n.",
                            },
                            {"name": "window_s", "in": "query", "required": False, "schema": {"type": "number"}},
                            {"name": "max_chars", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chat_lines", "in": "query", "required": False, "schema": {"type": "integer"}},
                        ],
                        "responses": {
                            "200": {
                                "description": "Candidates bundle",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AiCandidatesResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/ai/bundle": {
                    "get": {
                        "summary": "Fetch a single external-AI work bundle",
                        "operationId": "vp_actions_ai_bundle",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "project_id", "in": "query", "required": True, "schema": {"type": "string"}},
                            {"name": "top_n", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chat_top_n", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "window_s", "in": "query", "required": False, "schema": {"type": "number"}},
                            {"name": "max_chars", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chat_lines", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "max_variants", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chapter_limit", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chapter_max_chars", "in": "query", "required": False, "schema": {"type": "integer"}},
                        ],
                        "responses": {
                            "200": {
                                "description": "AI work bundle",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AiBundleResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/ai/clip_review": {
                    "get": {
                        "summary": "Fetch bot-ready shortlisted clip review packets",
                        "operationId": "vp_actions_ai_clip_review",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "project_id", "in": "query", "required": True, "schema": {"type": "string"}},
                            {"name": "top_n", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chat_top_n", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "window_s", "in": "query", "required": False, "schema": {"type": "number"}},
                            {"name": "max_chars", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chat_lines", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "max_variants", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chapter_limit", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "chapter_max_chars", "in": "query", "required": False, "schema": {"type": "integer"}},
                        ],
                        "responses": {
                            "200": {
                                "description": "Shortlisted clip review packets",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AiClipReviewResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/ai/variants": {
                    "get": {
                        "summary": "Fetch clip variants for AI selection",
                        "operationId": "vp_actions_ai_variants",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "project_id", "in": "query", "required": True, "schema": {"type": "string"}},
                            {"name": "top_n", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "candidate_ranks", "in": "query", "required": False, "schema": {"type": "string"}},
                            {"name": "candidate_ids", "in": "query", "required": False, "schema": {"type": "string"}},
                            {"name": "max_variants", "in": "query", "required": False, "schema": {"type": "integer"}},
                        ],
                        "responses": {
                            "200": {
                                "description": "Variants bundle",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AiVariantsResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/ai/chapters": {
                    "get": {
                        "summary": "Fetch chapters for AI labeling",
                        "operationId": "vp_actions_ai_chapters",
                        "x-openai-isConsequential": False,
                        "parameters": [
                            {"name": "project_id", "in": "query", "required": True, "schema": {"type": "string"}},
                            {"name": "offset", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}},
                            {"name": "max_chars", "in": "query", "required": False, "schema": {"type": "integer"}},
                        ],
                        "responses": {
                            "200": {
                                "description": "Chapters bundle",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AiChaptersResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/ai/apply_semantic": {
                    "post": {
                        "summary": "Apply semantic scoring decisions",
                        "operationId": "vp_actions_ai_apply_semantic",
                        "x-openai-isConsequential": False,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AiApplySemanticRequest"},
                                    "examples": {
                                        "basic": {
                                            "summary": "Apply semantic scores to a few ranks",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "client_request_id": "ai-semantic-001",
                                                "items": [
                                                    {
                                                        "candidate_rank": 1,
                                                        "semantic_score": 0.93,
                                                        "reason": "Strong setup and payoff; clear standalone moment.",
                                                        "best_quote": "…",
                                                        "keep": True,
                                                    }
                                                ],
                                            },
                                        }
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Applied",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AiApplySemanticResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/ai/apply_chapter_labels": {
                    "post": {
                        "summary": "Apply chapter labels",
                        "operationId": "vp_actions_ai_apply_chapter_labels",
                        "x-openai-isConsequential": False,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AiApplyChapterLabelsRequest"},
                                    "examples": {
                                        "basic": {
                                            "summary": "Label a couple chapters",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "client_request_id": "ai-chapters-001",
                                                "items": [
                                                    {
                                                        "chapter_id": 1,
                                                        "title": "Warmup and early plays",
                                                        "summary": "Introductions and first match momentum.",
                                                        "keywords": ["warmup", "opening"],
                                                        "type": "intro",
                                                    }
                                                ],
                                            },
                                        }
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Applied",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AiApplyChapterLabelsResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/ai/apply_director_picks": {
                    "post": {
                        "summary": "Apply director picks (variant choice + packaging)",
                        "operationId": "vp_actions_ai_apply_director_picks",
                        "x-openai-isConsequential": False,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AiApplyDirectorPicksRequest"},
                                    "examples": {
                                        "basic": {
                                            "summary": "Pick variants and packaging for export",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "client_request_id": "ai-director-001",
                                                "picks": [
                                                    {
                                                        "candidate_rank": 1,
                                                        "variant_id": "medium",
                                                        "title": "INSANE TURNAROUND",
                                                        "hook": "NO WAY",
                                                        "description": "The clutch moment that flipped the game.",
                                                        "hashtags": ["gaming", "clips", "shorts"],
                                                        "template": "vertical_blur",
                                                        "confidence": 0.8,
                                                        "reasons": ["clear payoff", "fast setup"],
                                                    }
                                                ],
                                            },
                                        }
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Applied",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/AiApplyDirectorPicksResponse"}}
                                },
                            }
                        },
                    }
                },
                "/api/actions/export_director_picks": {
                    "post": {
                        "summary": "Export clips from director picks",
                        "operationId": "vp_actions_export_director_picks",
                        "x-openai-isConsequential": True,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ExportDirectorPicksRequest"},
                                    "examples": {
                                        "basic": {
                                            "summary": "Export top picks with defaults",
                                            "value": {
                                                "project_id": "<project_id from ingest>",
                                                "limit": 5,
                                                "client_request_id": "export-director-001",
                                            },
                                        }
                                    },
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Job created",
                                "content": {
                                    "application/json": {"schema": {"$ref": "#/components/schemas/ExportDirectorPicksResponse"}}
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
            thumb = Path(video_result.thumbnail_path) if getattr(video_result, "thumbnail_path", None) else None
            set_project_video(proj, src, preview_path=prev, thumbnail_path=thumb)
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
        llm_mode: str = "external_strict",
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
            llm_mode = _normalize_llm_mode(llm_mode)
            JOB_MANAGER._set(
                job,
                status="running",
                progress=0.0,
                message="Starting analysis...",
                result={"project_id": project_id, "llm_mode": llm_mode},
            )
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

            dag_config = build_dag_config(
                analysis_cfg,
                section_overrides={"speech": speech_cfg},
                include_chat=True,
            )
            dag_config = apply_llm_mode_to_dag_config(dag_config, llm_mode=llm_mode)

            llm_complete_fn = None
            if llm_mode_uses_local(llm_mode):
                try:
                    ai_cfg = profile.get("ai", {}).get("director", {}) or {}
                    llm_needed = dag_config_needs_llm(dag_config)
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
                    "llm_mode": llm_mode,
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

        director_results = _load_export_director_results(proj=proj, proj_data=proj_data)

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
                if sj and sj.status == "cancelled":
                    JOB_MANAGER._set(job, status="cancelled", message="cancelled", result={"active_subjob_id": subjob.id})
                    _set_run(1.0, "Export: cancelled")
                    return
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
        llm_mode: str = "external_strict",
        export_top: Optional[int] = None,
        export_cfg: Optional[Dict[str, Any]] = None,
        cap_cfg: Optional[Dict[str, Any]] = None,
        hook_cfg: Optional[Dict[str, Any]] = None,
        pip_cfg: Optional[Dict[str, Any]] = None,
    ) -> Any:
        llm_mode = _normalize_llm_mode(llm_mode)
        if export_top is not None and llm_mode_is_strict_external(llm_mode):
            raise HTTPException(
                status_code=409,
                detail="external_strict_requires_ai_checkpoint",
            )
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
                "llm_mode": llm_mode,
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
                    llm_mode=llm_mode,
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
        llm_mode = _resolve_llm_mode(body.get("llm_mode"))
        fresh_project = bool(body.get("fresh_project"))

        content_id = _build_project_content_key(
            url,
            fresh_project=fresh_project,
            client_request_id=client_request_id,
        )
        proj = create_project_early(content_id, source_url=url)
        _persist_project_scout_metadata(proj=proj, url=url, body=body)
        project_id = proj.project_dir.name

        run_job = _start_run_job(
            actor_key=actor_key,
            preset="ingest_analyze",
            proj=proj,
            project_id=project_id,
            url=url,
            analyze_overrides=analyze_overrides or {},
            llm_mode=llm_mode,
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
        llm_mode = _resolve_llm_mode(body.get("llm_mode"))
        fresh_project = bool(body.get("fresh_project"))

        top_n = _clamp_int(body.get("top"), default=5, min_v=1, max_v=30)

        export_cfg = body.get("export") or {}
        cap_cfg = body.get("captions") or {}
        hook_cfg = body.get("hook_text") or {}
        pip_cfg = body.get("pip") or {}
        if not isinstance(export_cfg, dict) or not isinstance(cap_cfg, dict) or not isinstance(hook_cfg, dict) or not isinstance(pip_cfg, dict):
            raise HTTPException(status_code=400, detail="invalid_export_config")

        content_id = _build_project_content_key(
            url,
            fresh_project=fresh_project,
            client_request_id=client_request_id,
        )
        proj = create_project_early(content_id, source_url=url)
        _persist_project_scout_metadata(proj=proj, url=url, body=body)
        project_id = proj.project_dir.name

        run_job = _start_run_job(
            actor_key=actor_key,
            preset="full_export_top",
            proj=proj,
            project_id=project_id,
            url=url,
            analyze_overrides=analyze_overrides or {},
            llm_mode=llm_mode,
            export_top=top_n,
            export_cfg=export_cfg,
            cap_cfg=cap_cfg,
            hook_cfg=hook_cfg,
            pip_cfg=pip_cfg,
        )

        payload = {"job_id": run_job.id, "run_id": run_job.id, "project_id": project_id}
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.post("/run_full_export_top_unattended", openapi_extra={"x-openai-isConsequential": False})
    def run_full_export_top_unattended(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        """Identical to run_full_export_top, but marked non-consequential in OpenAPI.

        This allows ChatGPT Actions users to opt into "always allow" for unattended runs.
        """
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        url = _validate_ingest_url(str(body.get("url") or ""), allow_domains=allow_domains)
        analyze_overrides = body.get("analyze_overrides") or body.get("overrides") or {}
        if analyze_overrides and not isinstance(analyze_overrides, dict):
            raise HTTPException(status_code=400, detail="invalid_analyze_overrides")
        llm_mode = _resolve_llm_mode(body.get("llm_mode"))
        fresh_project = bool(body.get("fresh_project"))

        top_n = _clamp_int(body.get("top"), default=5, min_v=1, max_v=30)

        export_cfg = body.get("export") or {}
        cap_cfg = body.get("captions") or {}
        hook_cfg = body.get("hook_text") or {}
        pip_cfg = body.get("pip") or {}
        if not isinstance(export_cfg, dict) or not isinstance(cap_cfg, dict) or not isinstance(hook_cfg, dict) or not isinstance(pip_cfg, dict):
            raise HTTPException(status_code=400, detail="invalid_export_config")

        content_id = _build_project_content_key(
            url,
            fresh_project=fresh_project,
            client_request_id=client_request_id,
        )
        proj = create_project_early(content_id, source_url=url)
        _persist_project_scout_metadata(proj=proj, url=url, body=body)
        project_id = proj.project_dir.name

        run_job = _start_run_job(
            actor_key=actor_key,
            preset="full_export_top",
            proj=proj,
            project_id=project_id,
            url=url,
            analyze_overrides=analyze_overrides or {},
            llm_mode=llm_mode,
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

    @router.get("/scout/candidates", openapi_extra={"x-openai-isConsequential": False})
    def scout_candidates(request: Request, limit: int = 60, per_source: int = 20):
        _rate_limit(request)
        limit = _clamp_int(limit, default=60, min_v=1, max_v=100)
        per_source = _clamp_int(per_source, default=20, min_v=1, max_v=50)
        try:
            report = source_scout_mod.build_source_scout_report(limit=limit, per_source=per_source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        report["diagnostics"] = _source_scout_diagnostics()
        return JSONResponse(report)

    @router.get("/scout/inbox", openapi_extra={"x-openai-isConsequential": False})
    def scout_inbox(request: Request, status: str = "pending", limit: int = 50):
        _rate_limit(request)
        limit = _clamp_int(limit, default=50, min_v=1, max_v=200)
        status_value = str(status or "").strip().lower() or None
        return JSONResponse(_build_source_inbox_payload(status=status_value, limit=limit))

    @router.post("/scout/inbox/add", openapi_extra={"x-openai-isConsequential": False})
    def scout_inbox_add(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        _rate_limit(request)
        url = _validate_ingest_url(str(body.get("url") or ""), allow_domains=allow_domains)
        tags = body.get("tags") or []
        if tags and not isinstance(tags, list):
            raise HTTPException(status_code=400, detail="invalid_tags")
        try:
            inbox_path, entry, created = source_inbox_mod.add_source_inbox_entry(
                url=url,
                title=body.get("title"),
                notes=body.get("notes"),
                priority=body.get("priority"),
                tags=[str(item).strip() for item in tags if str(item).strip()],
                added_by=body.get("added_by"),
                source_id=body.get("source_id"),
                source_label=body.get("source_label"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        source_scout_mod.clear_source_scout_cache()
        return JSONResponse(
            {
                "meta": {"inbox_path": str(inbox_path), "created": bool(created)},
                "entry": entry,
            }
        )

    @router.post("/scout/inbox/mark", openapi_extra={"x-openai-isConsequential": False})
    def scout_inbox_mark(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        _rate_limit(request)
        inbox_id = str(body.get("inbox_id") or "").strip()
        status = str(body.get("status") or "").strip().lower()
        if status not in {"pending", "selected", "dismissed", "processed"}:
            raise HTTPException(status_code=400, detail="invalid_inbox_status")
        try:
            inbox_path, entry = source_inbox_mod.update_source_inbox_entry(
                inbox_id,
                status=status,
                project_id=body.get("project_id"),
                selection_notes=body.get("selection_notes"),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="inbox_entry_not_found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        source_scout_mod.clear_source_scout_cache()
        return JSONResponse({"meta": {"inbox_path": str(inbox_path)}, "entry": entry})

    @router.get("/publish/accounts", openapi_extra={"x-openai-isConsequential": False})
    def publish_accounts(request: Request):
        _rate_limit(request)
        return JSONResponse(_build_publish_accounts_payload())

    @router.get("/publish/exports", openapi_extra={"x-openai-isConsequential": False})
    def publish_exports(request: Request, project_id: str):
        _rate_limit(request)
        return JSONResponse(_build_publish_exports_payload(project_id=project_id))

    @router.post("/publish/queue", openapi_extra={"x-openai-isConsequential": True})
    def publish_queue(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        project_id = _validate_project_id(str(body.get("project_id") or ""))
        options = body.get("options") or {}
        if not isinstance(options, dict):
            raise HTTPException(status_code=400, detail="invalid_publish_options")
        public_release_approved = bool(body.get("public_release_approved", False))

        account_ids = body.get("account_ids")
        if account_ids is None and body.get("account_id"):
            account_ids = [body.get("account_id")]
        if not isinstance(account_ids, list) or not account_ids:
            raise HTTPException(status_code=400, detail="account_ids_required")
        account_ids = [str(aid or "").strip() for aid in account_ids if str(aid or "").strip()]
        if not account_ids:
            raise HTTPException(status_code=400, detail="account_ids_required")

        export_ids = body.get("export_ids")
        if export_ids is None and body.get("export_id"):
            export_ids = [body.get("export_id")]
        if not isinstance(export_ids, list) or not export_ids:
            raise HTTPException(status_code=400, detail="export_ids_required")
        export_ids = [str(eid or "").strip() for eid in export_ids if str(eid or "").strip()]
        if not export_ids:
            raise HTTPException(status_code=400, detail="export_ids_required")

        proj, _proj_data = _load_project(project_id)
        exports_dir = proj.exports_dir
        export_map = {exp.export_id: exp for exp in scan_project_exports(exports_dir)}

        accounts = []
        for aid in account_ids:
            account = account_store.get(aid)
            if not account:
                raise HTTPException(status_code=404, detail=f"account_not_found:{aid}")
            auth = get_publish_account_auth(account)
            if not auth.ready:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "account_needs_reauth" if auth.auth_state == "needs_reauth" else "account_not_ready",
                        "account_id": account.id,
                        "platform": account.platform,
                        "auth_state": auth.auth_state,
                        "auth_error": auth.auth_error,
                    },
                )
            accounts.append(account)

        selected_exports = []
        for eid in export_ids:
            exp = export_map.get(eid)
            if exp is None:
                raise HTTPException(status_code=404, detail=f"export_not_found:{eid}")
            if not is_safe_export_path(exp.mp4_path, exports_dir):
                raise HTTPException(status_code=400, detail=f"invalid_export_path:{eid}")
            selected_exports.append(exp)

        created_jobs = []
        for exp in selected_exports:
            base_metadata = dict(exp.metadata or {})
            metadata = _resolve_publish_metadata(export_id=exp.export_id, base_metadata=base_metadata, options=options)
            _require_publish_approval(metadata=metadata, approved=public_release_approved)

            for account in accounts:
                job_metadata_path = exports_dir / f"{exp.export_id}_{uuid.uuid4().hex[:8]}_publish.json"
                job_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                job = job_store.create_job(
                    job_id=uuid.uuid4().hex,
                    platform=account.platform,
                    account_id=account.id,
                    file_path=str(exp.mp4_path),
                    metadata_path=str(job_metadata_path),
                )
                created_jobs.append(
                    {
                        "job_id": job.id,
                        "export_id": exp.export_id,
                        "account_id": account.id,
                        "platform": account.platform,
                        "privacy": metadata.get("privacy"),
                        "publish_at": metadata.get("publish_at"),
                    }
                )

        payload = {
            "project_id": project_id,
            "jobs": created_jobs,
            "total": len(created_jobs),
            "policy": _publish_policy(),
        }
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.get("/publish/jobs", openapi_extra={"x-openai-isConsequential": False})
    def publish_jobs(request: Request, project_id: str, limit: int = 50):
        _rate_limit(request)
        limit = _clamp_int(limit, default=50, min_v=1, max_v=200)
        return JSONResponse(_filter_publish_jobs_for_project(project_id=project_id, limit=limit))

    @router.post(
        "/publish/jobs/{job_id}/delete-remote",
        openapi_extra={"x-openai-isConsequential": True},
    )
    def publish_delete_remote(
        request: Request,
        job_id: str,
        body: Dict[str, Any] = Body(...),
    ):  # type: ignore[valid-type]
        _rate_limit(request)

        if body.get("confirmed") is not True:
            raise HTTPException(status_code=400, detail="delete_remote_confirmation_required")

        target_job_id = str(job_id or "").strip()
        if not target_job_id:
            raise HTTPException(status_code=400, detail="job_id_required")

        try:
            job = job_store.get_job(target_job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="publish_job_not_found")

        remote_id = str(job.remote_id or "").strip()
        if not remote_id:
            raise HTTPException(status_code=409, detail="publish_job_remote_missing")
        if job.status == "removed":
            raise HTTPException(status_code=409, detail="publish_job_already_removed")

        account = account_store.get(job.account_id)
        if not account:
            raise HTTPException(status_code=404, detail=f"account_not_found:{job.account_id}")

        tokens = load_tokens(job.platform, job.account_id)
        if not tokens:
            raise HTTPException(status_code=409, detail="missing_tokens")

        try:
            connector = get_connector(job.platform, account=account, tokens=tokens)
            connector.delete_remote(remote_id=remote_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        dedup_removed = False
        file_path = Path(job.file_path)
        if file_path.exists() and file_path.is_file():
            dedup_removed = job_store.delete_dedup(
                job.platform,
                job.account_id,
                _file_sha256(file_path),
            )

        deleted_remote_url = job.remote_url
        updated_job = job_store.update_job(
            job.id,
            status="removed",
            progress=1.0,
            remote_url=None,
            last_error="Remote video removed from platform.",
            resume_json=None,
        )
        return JSONResponse(
            {
                "ok": True,
                "job": updated_job.to_dict(),
                "removed_remote_id": remote_id,
                "removed_remote_url": deleted_remote_url,
                "dedup_removed": dedup_removed,
            }
        )

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
        _persist_project_scout_metadata(proj=proj, url=url, body=body)
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
                thumb = Path(video_result.thumbnail_path) if getattr(video_result, "thumbnail_path", None) else None
                set_project_video(proj, src, preview_path=prev, thumbnail_path=thumb)
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
        llm_mode = _resolve_llm_mode(body.get("llm_mode"))

        proj, _proj_data = _load_project(project_id)
        if not Path(proj.video_path).exists():
            raise HTTPException(status_code=409, detail="video_not_ready")

        job = JOB_MANAGER.create("analyze_full")
        JOB_MANAGER._set(job, message="queued", result={"project_id": project_id, "llm_mode": llm_mode})

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

                dag_config = build_dag_config(
                    analysis_cfg,
                    section_overrides={"speech": speech_cfg},
                    include_chat=True,
                )
                dag_config = apply_llm_mode_to_dag_config(dag_config, llm_mode=llm_mode)

                llm_complete_fn = None
                if llm_mode_uses_local(llm_mode):
                    try:
                        ai_cfg = profile.get("ai", {}).get("director", {}) or {}
                        llm_needed = dag_config_needs_llm(dag_config)
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
                        "llm_mode": llm_mode,
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

    @router.get("/jobs/{job_id}/wait", openapi_extra={"x-openai-isConsequential": False})
    def job_wait(request: Request, job_id: str, timeout_s: float = 40.0):
        """Wait (block) for a job to finish, up to a bounded timeout."""
        _rate_limit(request)
        job = JOB_MANAGER.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")

        try:
            timeout = float(timeout_s)
        except Exception:
            timeout = 40.0
        timeout = max(0.0, min(40.0, timeout))

        deadline = time.monotonic() + timeout
        while True:
            job = JOB_MANAGER.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="job_not_found")
            if job.status in {"succeeded", "failed", "cancelled"}:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)

        done = job.status in {"succeeded", "failed", "cancelled"}
        poll_after_ms = 0
        if not done:
            poll_after_ms = 5000 if job.status == "queued" else 1000

        return JSONResponse(
            {
                "done": done,
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

    def _trim_text(s: str, *, max_chars: int) -> Tuple[str, bool]:
        s = (s or "").strip()
        if max_chars <= 0 or len(s) <= max_chars:
            return s, False
        return (s[:max_chars].rstrip() + "…"), True

    def _num_or_default(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
        except Exception:
            return float(default)
        if v != v:  # NaN guard
            return float(default)
        return float(v)

    def _candidate_key(cand: Dict[str, Any], idx: int) -> Tuple[str, Any]:
        cid = str(cand.get("candidate_id") or "").strip().lower()
        if cid:
            return ("id", cid)
        rank = _clamp_int(cand.get("rank"), default=0, min_v=0, max_v=10**9)
        if rank > 0:
            return ("rank", rank)
        return ("idx", idx)

    def _candidate_chat_signal(cand: Dict[str, Any]) -> float:
        raw = cand.get("raw_signals")
        if isinstance(raw, dict) and "chat" in raw:
            return _num_or_default(raw.get("chat"), 0.0)
        breakdown = cand.get("breakdown")
        if isinstance(breakdown, dict) and "chat" in breakdown:
            return _num_or_default(breakdown.get("chat"), 0.0)
        return 0.0

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
            if v == v:
                return v
        except Exception:
            pass
        return float(default)

    def _trim_copy_text(value: Any, *, max_chars: int) -> str:
        s = str(value or "").replace("\n", " ").replace("\r", " ").strip()
        if max_chars <= 0 or len(s) <= max_chars:
            return s
        cut = s[: max_chars - 1]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        return (cut + "…").strip()

    _PACKAGING_DOMAIN_RE = re.compile(
        r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|net|org|gg|tv|io|co|app|dev)\b",
        re.IGNORECASE,
    )
    _PACKAGING_GENERIC_DESCRIPTION_PATTERNS = (
        (re.compile(r"\bthe stream (?:stops being normal|becomes|turns into)\b", re.IGNORECASE), "description_too_generic"),
        (re.compile(r"\binstant\b.{0,48}\binstant\b", re.IGNORECASE), "description_repeated_hype"),
        (re.compile(r"\bclip[- ]friendly\b", re.IGNORECASE), "description_clip_filler"),
        (re.compile(r"\bout of context\b", re.IGNORECASE), "description_context_free_filler"),
        (re.compile(r"\beasy to understand\b", re.IGNORECASE), "description_context_free_filler"),
    )

    def _packaging_quality_issues(*, title: str, hook: str, description: str) -> list[str]:
        issues: list[str] = []
        if _PACKAGING_DOMAIN_RE.search(title or ""):
            issues.append("title_domain_anchor")
        if _PACKAGING_DOMAIN_RE.search(hook or ""):
            issues.append("hook_domain_anchor")
        desc = str(description or "").strip()
        for pattern, code in _PACKAGING_GENERIC_DESCRIPTION_PATTERNS:
            if pattern.search(desc):
                issues.append(code)
        return issues

    def _clean_hashtag(value: Any) -> str:
        tag = str(value or "").strip()
        if not tag:
            return ""
        tag = re.sub(r"[^0-9A-Za-z_]+", "", tag)
        if not tag:
            return ""
        if not tag.startswith("#"):
            tag = "#" + tag
        return tag

    def _normalize_hashtags(raw: Any, *, defaults: Optional[list[str]] = None, min_n: int = 3, max_n: int = 8) -> list[str]:
        if isinstance(raw, str):
            parts = re.split(r"[\s,]+", raw.strip())
        elif isinstance(raw, list):
            parts = raw
        else:
            parts = []

        out: list[str] = []
        seen = set()
        for p in parts:
            t = _clean_hashtag(p)
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            if len(out) >= max_n:
                break

        seed = list(defaults or ["#clips", "#gaming", "#stream"])
        for d in seed:
            if len(out) >= min_n:
                break
            t = _clean_hashtag(d)
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            out.append(t)
            seen.add(key)

        return out[:max_n]

    def _director_config_from_profile() -> DirectorConfig:
        analysis_cfg = profile.get("analysis", {}) or {}
        ai_cfg = (profile.get("ai", {}) or {}).get("director", {}) or {}
        raw: Dict[str, Any] = dict((analysis_cfg.get("director", {}) or {}))
        if "enabled" not in raw:
            raw["enabled"] = bool(ai_cfg.get("enabled", True))
        if "platform" not in raw and ai_cfg.get("platform") is not None:
            raw["platform"] = ai_cfg.get("platform")
        if "fallback_to_rules" not in raw and ai_cfg.get("fallback_to_rules") is not None:
            raw["fallback_to_rules"] = ai_cfg.get("fallback_to_rules")
        return DirectorConfig.from_dict(raw)

    def _director_default_template(cfg: DirectorConfig) -> str:
        allowed = [str(x) for x in list(cfg.allowed_templates or []) if str(x).strip()]
        if cfg.default_template in allowed:
            return str(cfg.default_template)
        if allowed:
            return str(allowed[0])
        return str((profile.get("export", {}) or {}).get("template") or "vertical_streamer_pip")

    def _load_chapter_ranges(proj: Project) -> list[tuple[float, float]]:
        path = proj.analysis_dir / "chapters.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        out: list[tuple[float, float]] = []
        for c in (data.get("chapters") or []):
            if not isinstance(c, dict):
                continue
            a = _safe_float(c.get("start_s"), -1.0)
            b = _safe_float(c.get("end_s"), -1.0)
            if b > a >= 0.0:
                out.append((a, b))
        return out

    def _chapter_index_for_time(chapters: list[tuple[float, float]], t: float) -> Optional[int]:
        for i, (a, b) in enumerate(chapters):
            if a <= t < b:
                return i
        if chapters and t >= chapters[-1][0]:
            return len(chapters) - 1
        return None

    def _get_chat_excerpt(
        *,
        proj: Project,
        start_s: float,
        end_s: float,
        lines: int,
        max_line_chars: int = 180,
    ) -> Optional[Dict[str, Any]]:
        if lines <= 0:
            return None
        try:
            from ..chat.store import ChatStore

            cfg = get_chat_config(proj)
            offset_ms = int(cfg.get("sync_offset_ms", 0) or 0)

            start_ms = int(max(0.0, float(start_s)) * 1000)
            end_ms = int(max(float(start_ms) / 1000.0, float(end_s)) * 1000)

            store = ChatStore(proj.chat_db_path)
            if not store.exists:
                return None

            msgs = store.get_messages(start_ms, end_ms, offset_ms=offset_ms, limit=int(lines))
            out_lines: list[str] = []
            any_trunc = False
            for m in msgs:
                t_s = m.t_ms / 1000.0
                text = (m.text or "").replace("\n", " ").replace("\r", " ").strip()
                author = (m.author or "").strip()
                if max_line_chars > 0 and len(text) > max_line_chars:
                    text = text[:max_line_chars].rstrip() + "…"
                    any_trunc = True
                if author:
                    out_lines.append(f"{t_s:8.1f}s {author}: {text}")
                else:
                    out_lines.append(f"{t_s:8.1f}s {text}")
            return {
                "start_s": start_ms / 1000.0,
                "end_s": end_ms / 1000.0,
                "lines": out_lines,
                "truncated": any_trunc,
            }
        except Exception:
            return None

    def _load_export_director_results(*, proj: Project, proj_data: Dict[str, Any]) -> list[Dict[str, Any]]:
        """Load best-effort AI metadata for exports (hook/title/description/tags).

        Priority:
          1) analysis/director.json (new pipeline director picks)
          2) analysis/ai_director.json (legacy local-LLM director)
          3) project.json legacy fields
        """
        # 1) analysis/director.json (picks[])
        try:
            director_path = proj.analysis_dir / "director.json"
            if director_path.exists():
                data = json.loads(director_path.read_text(encoding="utf-8"))
                picks = data.get("picks") or []
                out: list[Dict[str, Any]] = []
                for p in picks:
                    if not isinstance(p, dict):
                        continue
                    cand_rank = p.get("candidate_rank")
                    if cand_rank is None:
                        continue
                    raw_tags = p.get("tags") or p.get("hashtags") or []
                    if isinstance(raw_tags, str):
                        tags = [t for t in re.split(r"[\\s,]+", raw_tags.strip()) if t]
                    elif isinstance(raw_tags, list):
                        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
                    else:
                        tags = []
                    reasons = p.get("reasons")
                    if isinstance(reasons, list):
                        reason = "; ".join(str(x).strip() for x in reasons if str(x).strip())
                    else:
                        reason = str(reasons or "")
                    out.append(
                        {
                            "candidate_rank": cand_rank,
                            "best_variant_id": p.get("variant_id") or p.get("best_variant_id") or "",
                            "variant_id": p.get("variant_id") or "",
                            "reason": reason,
                            "title": p.get("title") or "",
                            "hook": p.get("hook") or "",
                            "description": p.get("description") or "",
                            "tags": tags,
                            "hashtags": tags,
                            "confidence": p.get("confidence"),
                            "used_fallback": False,
                        }
                    )
                if out:
                    return out
        except Exception:
            pass

        # 2) analysis/ai_director.json (results[])
        try:
            ai_path = proj.analysis_dir / "ai_director.json"
            if ai_path.exists():
                data = json.loads(ai_path.read_text(encoding="utf-8"))
                results = data.get("results") or []
                out: list[Dict[str, Any]] = []
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    raw_tags = r.get("tags") or r.get("hashtags") or []
                    if isinstance(raw_tags, str):
                        tags = [t for t in re.split(r"[\\s,]+", raw_tags.strip()) if t]
                    elif isinstance(raw_tags, list):
                        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
                    else:
                        tags = []
                    rr = dict(r)
                    rr.setdefault("tags", tags)
                    rr.setdefault("hashtags", tags)
                    out.append(rr)
                if out:
                    return out
        except Exception:
            pass

        # 3) project.json legacy fields (if present)
        try:
            legacy = ((proj_data.get("analysis", {}) or {}).get("director", {}) or {}).get("results") or []
            if isinstance(legacy, list):
                return [x for x in legacy if isinstance(x, dict)]
        except Exception:
            pass
        try:
            legacy2 = proj_data.get("director_results") or []
            if isinstance(legacy2, list):
                return [x for x in legacy2 if isinstance(x, dict)]
        except Exception:
            pass

        return []

    def _selection_matches_director_result(
        selection: Dict[str, Any],
        director_results: list[Dict[str, Any]],
        *,
        time_tolerance_s: float = 0.25,
    ) -> bool:
        """True when a selection lines up with a director-reviewed pick."""
        try:
            selection_rank = int(selection.get("candidate_rank") or selection.get("rank") or 0)
        except Exception:
            selection_rank = 0
        selection_variant_id = str(
            selection.get("variant_id")
            or selection.get("best_variant_id")
            or selection.get("chosen_variant_id")
            or ""
        ).strip()
        selection_start = _safe_float(selection.get("start_s"), 0.0)
        selection_end = _safe_float(selection.get("end_s"), selection_start)

        for result in director_results:
            if not isinstance(result, dict):
                continue

            try:
                result_rank = int(result.get("candidate_rank") or result.get("rank") or 0)
            except Exception:
                result_rank = 0
            if selection_rank > 0 and result_rank > 0 and selection_rank != result_rank:
                continue

            result_variant_id = str(
                result.get("variant_id")
                or result.get("best_variant_id")
                or ""
            ).strip()
            if selection_variant_id and result_variant_id and selection_variant_id != result_variant_id:
                continue
            if selection_variant_id and result_variant_id and selection_rank > 0 and result_rank > 0:
                return True

            result_start = _safe_float(result.get("start_s"), 0.0)
            result_end = _safe_float(result.get("end_s"), result_start)
            if (
                abs(selection_start - result_start) <= time_tolerance_s
                and abs(selection_end - result_end) <= time_tolerance_s
            ):
                return True

        return False

    def _require_director_backed_export_batch(
        *,
        project_id: str,
        proj: Project,
        proj_data: Dict[str, Any],
        llm_mode: str,
        top_from_candidates: Any,
        selections: list[Dict[str, Any]],
        director_results: list[Dict[str, Any]],
    ) -> None:
        """Strict external mode must export director-backed picks, not raw tops."""
        if not llm_mode_is_strict_external(llm_mode):
            return

        _require_external_ai_ready_for_export(
            project_id=project_id,
            proj=proj,
            proj_data=proj_data,
            llm_mode=llm_mode,
        )

        if top_from_candidates is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "external_strict_requires_director_picks",
                    "project_id": project_id,
                    "issues": [
                        "external_strict batch export cannot create raw selections from top candidates",
                        "use /api/actions/export_director_picks after director review",
                    ],
                },
            )

        missing_selection_ids = [
            str(selection.get("id") or "")
            for selection in selections
            if not _selection_matches_director_result(selection, director_results)
        ]
        missing_selection_ids = [sid for sid in missing_selection_ids if sid]
        if missing_selection_ids:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "external_strict_requires_director_picks",
                    "project_id": project_id,
                    "issues": [
                        "external_strict batch export requires director-backed selections",
                        "use /api/actions/export_director_picks or export selections created from director picks",
                    ],
                    "selection_ids": missing_selection_ids,
                },
            )

    def _build_ai_candidates_payload(
        *,
        project_id: str,
        top_n: Optional[int] = None,
        chat_top_n: Optional[int] = None,
        chat_top: Optional[int] = None,
        window_s: float = 4.0,
        max_chars: int = 800,
        chat_lines: int = 8,
    ) -> Dict[str, Any]:
        pid = _validate_project_id(project_id)
        top_n_provided = top_n is not None
        # `chat_top` is a compatibility alias for macro prompts.
        chat_top_n_effective = chat_top_n if chat_top_n is not None else chat_top
        chat_top_n_provided = chat_top_n_effective is not None
        top_n = _clamp_int(top_n, default=30, min_v=1, max_v=50) if top_n_provided else 30
        if chat_top_n_provided:
            chat_top_n_i = _clamp_int(chat_top_n_effective, default=15, min_v=0, max_v=30)
        else:
            # Default to hybrid feed even when top_n is provided.
            # Callers can force top-only with chat_top_n=0.
            chat_top_n_i = 15
        window_s = max(0.0, min(30.0, float(window_s)))
        max_chars = _clamp_int(max_chars, default=800, min_v=200, max_v=4000)
        chat_lines = _clamp_int(chat_lines, default=8, min_v=0, max_v=50)

        proj, proj_data = _load_project(pid)

        meta = {
            "project_id": pid,
            "title": proj_data.get("title") or (proj_data.get("video", {}) or {}).get("title"),
            "source_url": (proj_data.get("source", {}) or {}).get("source_url") or proj_data.get("source_url"),
            "duration_seconds": (proj_data.get("video", {}) or {}).get("duration_seconds"),
        }

        highlights = (proj_data.get("analysis", {}) or {}).get("highlights", {}) or {}
        all_candidates = [c for c in (highlights.get("candidates") or []) if isinstance(c, dict)]
        indexed_candidates = list(enumerate(all_candidates))

        selected: list[Tuple[Dict[str, Any], str]] = []
        selected_keys = set()
        for idx, cand in indexed_candidates[:top_n]:
            k = _candidate_key(cand, idx)
            if k in selected_keys:
                continue
            selected_keys.add(k)
            selected.append((cand, "multi_signal"))

        chat_spike_count = 0
        if chat_top_n_i > 0:
            leftovers = [(idx, cand) for idx, cand in indexed_candidates if _candidate_key(cand, idx) not in selected_keys]
            leftovers.sort(
                key=lambda it: (
                    _candidate_chat_signal(it[1]),
                    _num_or_default(it[1].get("score"), 0.0),
                ),
                reverse=True,
            )
            for idx, cand in leftovers[:chat_top_n_i]:
                k = _candidate_key(cand, idx)
                if k in selected_keys:
                    continue
                selected_keys.add(k)
                selected.append((cand, "chat_spike"))
                chat_spike_count += 1

        # Transcript is optional (pipeline may not have produced it yet).
        tr = None
        try:
            from ..analysis_transcript import load_transcript

            tr = load_transcript(proj)
        except Exception:
            tr = None

        out: list[Dict[str, Any]] = []
        for cand, source_bucket in selected:
            start_s = float(cand.get("start_s") or 0.0)
            end_s = float(cand.get("end_s") or (start_s + 0.01))
            excerpt_start = max(0.0, start_s - window_s)
            excerpt_end = max(excerpt_start, end_s + window_s)

            transcript_excerpt = None
            if tr is not None:
                try:
                    txt = (tr.get_text_in_range(excerpt_start, excerpt_end) or "").strip()
                    txt, trunc = _trim_text(txt, max_chars=max_chars)
                    transcript_excerpt = {
                        "start_s": excerpt_start,
                        "end_s": excerpt_end,
                        "text": txt,
                        "truncated": trunc,
                    }
                except Exception:
                    transcript_excerpt = None

            chat_excerpt = None
            if chat_lines > 0:
                # Slightly wider window for chat context
                chat_excerpt = _get_chat_excerpt(
                    proj=proj,
                    start_s=max(0.0, excerpt_start - 2.0),
                    end_s=excerpt_end + 2.0,
                    lines=chat_lines,
                )

            out.append(
                {
                    "rank": cand.get("rank"),
                    "candidate_id": cand.get("candidate_id"),
                    "score": cand.get("score"),
                    "start_s": cand.get("start_s"),
                    "end_s": cand.get("end_s"),
                    "peak_time_s": cand.get("peak_time_s"),
                    "breakdown": cand.get("breakdown"),
                    "hook_text": cand.get("hook_text"),
                    "title": cand.get("title"),
                    "reasons": cand.get("reasons") or cand.get("reason"),
                    "llm_reason": cand.get("llm_reason"),
                    "llm_quote": cand.get("llm_quote"),
                    "selection_source": source_bucket,
                    "chat_signal": _candidate_chat_signal(cand),
                    "transcript_excerpt": transcript_excerpt,
                    "chat_excerpt": chat_excerpt,
                }
            )

        analysis_cfg = profile.get("analysis", {}) or {}
        highlights_cfg = analysis_cfg.get("highlights", {}) or {}
        content_type = str(highlights_cfg.get("content_type", "gaming") or "gaming").strip().lower()
        guide = CONTENT_TYPE_GUIDANCE.get(content_type) or CONTENT_TYPE_GUIDANCE.get("gaming", {})
        director_cfg = _director_config_from_profile()
        director_template_default = _director_default_template(director_cfg)

        return {
            "meta": meta,
            "limits": {
                "top_n": top_n,
                "chat_top_n": chat_top_n_i,
                "window_s": window_s,
                "max_chars": max_chars,
                "chat_lines": chat_lines,
            },
            "strategy": {
                "mode": "hybrid_top_plus_chat" if chat_top_n_i > 0 else "top_only",
                "multi_signal_count": len([1 for _, src in selected if src == "multi_signal"]),
                "chat_spike_count": chat_spike_count,
                "total_selected": len(selected),
                "total_candidates_available": len(all_candidates),
            },
            "ai_guidance": {
                "semantic_scoring": {
                    "content_type": content_type,
                    "description": guide.get("description", ""),
                    "high_value": list(guide.get("high_value", []) or []),
                    "low_value": list(guide.get("low_value", []) or []),
                    "instruction": (
                        "Use transcript/chat context and signal data together; keep high-signal moments high unless "
                        "content is clearly weak."
                    ),
                },
                "director_constraints": {
                    "top_n": int(director_cfg.top_n),
                    "min_gap_s": float(director_cfg.min_gap_s),
                    "max_overlap_ratio": float(director_cfg.max_overlap_ratio),
                    "max_overlap_s": float(director_cfg.max_overlap_s),
                    "title_max_chars": int(director_cfg.title_max_chars),
                    "hook_max_chars": int(director_cfg.hook_max_chars),
                    "description_max_chars": int(director_cfg.description_max_chars),
                    "allowed_templates": list(director_cfg.allowed_templates or []),
                    "default_template": director_template_default,
                    "chapter_diversity_max_per_chapter": 2,
                },
            },
            "candidates": out,
            "total_candidates": len(all_candidates),
        }

    @router.get("/ai/candidates", openapi_extra={"x-openai-isConsequential": False})
    def ai_candidates(
        request: Request,
        project_id: str,
        top_n: Optional[int] = None,
        chat_top_n: Optional[int] = None,
        chat_top: Optional[int] = None,
        window_s: float = 4.0,
        max_chars: int = 800,
        chat_lines: int = 8,
    ):
        """Fetch AI-ready highlight candidates with bounded transcript/chat context.

        Default strategy (when top_n/chat_top_n/chat_top omitted):
          - top 30 by fused multi-signal score
          - plus 15 chat-spike outliers (deduped)
        """
        _rate_limit(request)
        return JSONResponse(
            _build_ai_candidates_payload(
                project_id=project_id,
                top_n=top_n,
                chat_top_n=chat_top_n,
                chat_top=chat_top,
                window_s=window_s,
                max_chars=max_chars,
                chat_lines=chat_lines,
            )
        )

    def _build_ai_variants_payload(
        *,
        project_id: str,
        top_n: int = 12,
        candidate_ranks: str = "",
        candidate_ids: str = "",
        max_variants: int = 8,
    ) -> Dict[str, Any]:
        pid = _validate_project_id(project_id)
        top_n = _clamp_int(top_n, default=12, min_v=1, max_v=30)
        max_variants = _clamp_int(max_variants, default=8, min_v=1, max_v=30)

        ids: Optional[list[str]] = None
        if str(candidate_ids or "").strip():
            parsed_ids: list[str] = []
            for part in str(candidate_ids).split(","):
                part = str(part).strip()
                if part:
                    parsed_ids.append(part)
            seen_ids = set()
            ids = []
            for cid in parsed_ids:
                key = cid.lower()
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                ids.append(cid)
            if not ids:
                ids = None

        ranks: Optional[list[int]] = None
        if ids is None and str(candidate_ranks or "").strip():
            parsed: list[int] = []
            for part in str(candidate_ranks).split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    parsed.append(int(part))
                except Exception:
                    continue
            # Deduplicate while preserving order.
            seen = set()
            ranks = []
            for r in parsed:
                if r in seen:
                    continue
                seen.add(r)
                if r > 0:
                    ranks.append(r)
            if not ranks:
                ranks = None

        proj, proj_data = _load_project(pid)
        from ..clip_variants import load_clip_variants

        candidates = (((proj_data.get("analysis") or {}).get("highlights") or {}).get("candidates") or [])
        rank_by_id: Dict[str, int] = {}
        for c in candidates:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("candidate_id") or "").strip()
            if not cid:
                continue
            try:
                r = int(c.get("rank") or 0)
            except Exception:
                r = 0
            if r > 0:
                rank_by_id[cid] = r

        if ids is not None:
            ranks = []
            for cid in ids:
                r = rank_by_id.get(cid)
                if r:
                    ranks.append(int(r))
            if not ranks:
                ranks = []

        cvs = load_clip_variants(proj) or []
        if ranks is not None:
            cv_by_rank = {int(cv.candidate_rank): cv for cv in cvs}
            cvs = [cv_by_rank[r] for r in ranks if r in cv_by_rank]
        else:
            cvs = sorted(cvs, key=lambda cv: int(getattr(cv, "candidate_rank", 0)))[:top_n]

        out_candidates: list[Dict[str, Any]] = []
        for cv in cvs:
            vars_out: list[Dict[str, Any]] = []
            for v in (cv.variants or [])[:max_variants]:
                vars_out.append(v.to_dict())
            cid = str(getattr(cv, "candidate_id", "") or "")
            if not cid:
                cid = next((k for k, r in rank_by_id.items() if r == int(cv.candidate_rank)), "")
            out_candidates.append(
                {
                    "candidate_rank": int(cv.candidate_rank),
                    "candidate_id": cid,
                    "candidate_peak_time_s": float(cv.candidate_peak_time_s),
                    "variants": vars_out,
                    "truncated": len(cv.variants or []) > len(vars_out),
                }
            )

        return {
            "meta": {
                "project_id": pid,
                "variant_file": "variants.json",
                "candidate_count": len(out_candidates),
            },
            "limits": {
                "top_n": top_n,
                "candidate_ranks": ranks,
                "candidate_ids": ids,
                "max_variants": max_variants,
            },
            "candidates": out_candidates,
        }

    @router.get("/ai/variants", openapi_extra={"x-openai-isConsequential": False})
    def ai_variants(
        request: Request,
        project_id: str,
        top_n: int = 12,
        candidate_ranks: str = "",
        candidate_ids: str = "",
        max_variants: int = 8,
    ):
        """Fetch clip variants (variants.json) for AI selection."""
        _rate_limit(request)
        return JSONResponse(
            _build_ai_variants_payload(
                project_id=project_id,
                top_n=top_n,
                candidate_ranks=candidate_ranks,
                candidate_ids=candidate_ids,
                max_variants=max_variants,
            )
        )

    def _build_ai_chapters_payload(
        *,
        project_id: str,
        offset: int = 0,
        limit: int = 10,
        max_chars: int = 1200,
    ) -> Dict[str, Any]:
        pid = _validate_project_id(project_id)
        offset = _clamp_int(offset, default=0, min_v=0, max_v=10_000)
        limit = _clamp_int(limit, default=10, min_v=1, max_v=50)
        max_chars = _clamp_int(max_chars, default=1200, min_v=200, max_v=8000)

        proj, _proj_data = _load_project(pid)
        chapters_path = proj.analysis_dir / "chapters.json"
        if not chapters_path.exists():
            raise HTTPException(status_code=404, detail="no_chapters")

        data = json.loads(chapters_path.read_text(encoding="utf-8"))
        chapters = [c for c in (data.get("chapters") or []) if isinstance(c, dict)]

        tr = None
        try:
            from ..analysis_transcript import load_transcript

            tr = load_transcript(proj)
        except Exception:
            tr = None

        sliced = chapters[offset : offset + limit]
        out_ch: list[Dict[str, Any]] = []
        for ch in sliced:
            start_s = float(ch.get("start_s") or 0.0)
            end_s = float(ch.get("end_s") or start_s)
            excerpt = None
            if tr is not None:
                try:
                    txt = (tr.get_text_in_range(start_s, end_s) or "").strip()
                    txt, trunc = _trim_text(txt, max_chars=max_chars)
                    excerpt = {"start_s": start_s, "end_s": end_s, "text": txt, "truncated": trunc}
                except Exception:
                    excerpt = None
            out_ch.append(
                {
                    "id": ch.get("id"),
                    "start_s": ch.get("start_s"),
                    "end_s": ch.get("end_s"),
                    "title": ch.get("title"),
                    "summary": ch.get("summary"),
                    "keywords": ch.get("keywords"),
                    "type": ch.get("type"),
                    "transcript_excerpt": excerpt,
                }
            )

        return {
            "meta": {
                "project_id": pid,
                "chapter_count": len(chapters),
                "generated_at": data.get("generated_at"),
            },
            "limits": {"offset": offset, "limit": limit, "max_chars": max_chars},
            "chapters": out_ch,
        }

    @router.get("/ai/chapters", openapi_extra={"x-openai-isConsequential": False})
    def ai_chapters(
        request: Request,
        project_id: str,
        offset: int = 0,
        limit: int = 10,
        max_chars: int = 1200,
    ):
        """Fetch chapters (chapters.json) with bounded transcript context for labeling."""
        _rate_limit(request)
        return JSONResponse(
            _build_ai_chapters_payload(
                project_id=project_id,
                offset=offset,
                limit=limit,
                max_chars=max_chars,
            )
        )

    def _channel_show_formats() -> list[Dict[str, Any]]:
        return [
            {
                "show_format_id": "clip_court",
                "label": "Clip Court",
                "format_id": "commentary_breakdown",
                "lane": "single_clip",
                "pitch": "Verdict-driven commentary that puts one clip on trial and makes the host's judgment the real product.",
                "best_for": [
                    "One standout clip with a clean setup/payoff",
                    "Moments where the host can call it genius, fraud, panic, or pure cinema",
                ],
                "edit_recipe": [
                    "Hook with the accusation or question",
                    "Play the shortest source beat needed for context",
                    "Pause or replay the evidence",
                    "Deliver the host verdict, score, or roast",
                    "Close with a takeaway that stands on its own",
                ],
                "rubric": [
                    "Does the host verdict land within the first few beats?",
                    "Is the clip short enough that commentary still dominates the experience?",
                    "Would the upload still be watchable if the source clip were shortened further?",
                ],
                "packaging_notes": [
                    "Title should name the verdict or offense",
                    "Hook should preview the judgment, not just the event",
                ],
            },
            {
                "show_format_id": "weekly_awards",
                "label": "Weekly Awards",
                "format_id": "ranked_roundup",
                "lane": "multi_clip",
                "pitch": "Recurring awards/ranking show that compares several clips and turns them into one editorial episode.",
                "best_for": [
                    "Two or more shortlisted clips worth comparing",
                    "Recurring categories like choke, karma, betrayal, or pure chaos",
                ],
                "edit_recipe": [
                    "Open with the award category or ranking premise",
                    "Run each shortlisted clip with a short host verdict",
                    "Compare the contenders directly",
                    "Name the winner and why it beat the others",
                    "End with the final scoreboard or host take",
                ],
                "rubric": [
                    "Do the clips share a comparison angle that justifies grouping them together?",
                    "Is each clip short enough to leave room for host commentary between entries?",
                    "Does the winner feel earned rather than arbitrary?",
                ],
                "packaging_notes": [
                    "Title should name the award or ranking category",
                    "Description should explain why these clips belong in the same episode",
                ],
            },
        ]

    def _channel_format_playbook() -> Dict[str, Any]:
        return {
            "channel_positioning": {
                "rule": "Treat streamer clips as raw evidence for an editorial/commentary channel, not as the finished product.",
                "audience_promise": "Every upload should add a clear host viewpoint, verdict, or framing that still makes sense if the source clip is shortened.",
            },
            "launch_stack": ["clip_court", "weekly_awards"],
            "must_have_contribution": [
                "Original commentary, narration, or written editorial framing",
                "A clear thesis, verdict, ranking, or explanation for why the clip matters",
                "Substantive structure beyond the source moment itself",
                "Packaging that matches the actual point of the clip instead of generic clip-dump titling",
            ],
            "not_enough_on_their_own": [
                "Vertical crop only",
                "Captions only",
                "Light meme edits or transitions only",
                "Speed or pitch changes without new commentary",
                "Long uninterrupted source playback",
            ],
            "formats": [
                {
                    "format_id": "commentary_breakdown",
                    "label": "Commentary Breakdown",
                    "best_for": "One strong clip with a clear payoff that benefits from host framing and a verdict.",
                    "structure": [
                        "Setup: tell the viewer what to watch for",
                        "Short source moment",
                        "Pause, replay, or annotate the key beat",
                        "Host explanation, punchline, or criticism",
                        "Final verdict or takeaway",
                    ],
                },
                {
                    "format_id": "ranked_roundup",
                    "label": "Ranked Roundup",
                    "best_for": "Two or more shortlisted clips where comparison is part of the entertainment.",
                    "structure": [
                        "Ranking premise or theme",
                        "Clip A with brief verdict",
                        "Clip B with brief verdict",
                        "Comparison or scorecard",
                        "Final ranking and winner",
                    ],
                },
                {
                    "format_id": "theme_compilation",
                    "label": "Theme Compilation",
                    "best_for": "Several clips tied together by one pattern, running joke, or repeated failure mode.",
                    "structure": [
                        "Theme intro",
                        "Evidence clip",
                        "Host explanation or connective tissue",
                        "Next example",
                        "Closing takeaway",
                    ],
                },
            ],
            "show_formats": _channel_show_formats(),
            "upload_guardrails": [
                "Private or unlisted by default unless public release is explicitly approved",
                "Prefer short, purposeful source excerpts over uninterrupted reposting",
                "Use original voice, on-screen analysis, or host-written framing so the upload is recognizably yours",
            ],
        }

    def _clip_format_recommendations(
        *,
        candidate: Dict[str, Any],
        primary_variant: Optional[Dict[str, Any]],
        shortlist_count: int,
        chapter_context: Dict[str, Any],
        has_export: bool,
    ) -> list[Dict[str, Any]]:
        duration_s = _safe_float(
            (primary_variant or {}).get("duration_s"),
            _safe_float(candidate.get("end_s"), 0.0) - _safe_float(candidate.get("start_s"), 0.0),
        )
        chapter_title = str(((chapter_context.get("current") or {}).get("title") or "")).strip()
        recommendations: list[Dict[str, Any]] = [
            {
                "format_id": "commentary_breakdown",
                "priority": "primary",
                "reason": (
                    "This shortlist item should be framed as a host-led breakdown so the commentary, not the borrowed clip alone, carries the upload."
                ),
            }
        ]
        if shortlist_count >= 2:
            recommendations.append(
                {
                    "format_id": "ranked_roundup",
                    "priority": "secondary",
                    "reason": f"There are {shortlist_count} shortlisted clips available, so comparison/ranking is a viable editorial hook.",
                }
            )
        if shortlist_count >= 3 or (chapter_title and duration_s >= 20.0):
            reason = "This clip looks strong enough to support a recurring theme or pattern with connective host commentary."
            if chapter_title:
                reason = f'The surrounding chapter "{chapter_title}" provides a reusable theme for a multi-clip format.'
            recommendations.append(
                {
                    "format_id": "theme_compilation",
                    "priority": "secondary",
                    "reason": reason,
                }
            )
        if has_export and recommendations:
            recommendations[0]["export_note"] = (
                "An exported MP4 already exists, so verify pacing and whether the host framing still needs to be strengthened before upload."
            )
        return recommendations

    def _clip_show_format_recommendations(
        *,
        candidate: Dict[str, Any],
        primary_variant: Optional[Dict[str, Any]],
        shortlist_count: int,
        chapter_context: Dict[str, Any],
        has_export: bool,
    ) -> list[Dict[str, Any]]:
        duration_s = _safe_float(
            (primary_variant or {}).get("duration_s"),
            _safe_float(candidate.get("end_s"), 0.0) - _safe_float(candidate.get("start_s"), 0.0),
        )
        chapter_title = str(((chapter_context.get("current") or {}).get("title") or "")).strip()
        recommendations: list[Dict[str, Any]] = []
        if shortlist_count >= 2:
            reason = (
                f"There are {shortlist_count} shortlisted clips, so this candidate can compete inside a branded ranking/awards episode."
            )
            if chapter_title:
                reason += f' The current chapter "{chapter_title}" can anchor the category or comparison framing.'
            recommendations.append(
                {
                    "show_format_id": "weekly_awards",
                    "format_id": "ranked_roundup",
                    "priority": "primary",
                    "reason": reason,
                }
            )
            recommendations.append(
                {
                    "show_format_id": "clip_court",
                    "format_id": "commentary_breakdown",
                    "priority": "secondary",
                    "reason": "If this becomes the clear winner, it can still stand alone as a verdict-driven single-clip episode.",
                }
            )
        else:
            reason = "This is a single shortlisted clip, so the cleanest lane is a verdict-driven episode where the host judgment is the product."
            if duration_s >= 25.0:
                reason += " The longer runtime means the host should be especially aggressive about trimming the source beat and adding commentary."
            recommendations.append(
                {
                    "show_format_id": "clip_court",
                    "format_id": "commentary_breakdown",
                    "priority": "primary",
                    "reason": reason,
                }
            )
        if has_export and recommendations:
            recommendations[0]["export_note"] = (
                "An export already exists, so confirm the branded host framing still reads clearly before publish."
            )
        return recommendations

    def _build_ai_clip_review_payload(
        *,
        project_id: str,
        candidates_payload: Dict[str, Any],
        variants_payload: Dict[str, Any],
        chapters_payload: Dict[str, Any],
        frame_clip_limit: int = 0,
        frames_per_clip: int = 0,
        proj: Optional[Project] = None,
        proj_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pid = _validate_project_id(project_id)
        if proj is None or proj_data is None:
            proj, proj_data = _load_project(pid)

        candidate_items = [
            item for item in (candidates_payload.get("candidates") or []) if isinstance(item, dict)
        ]
        frame_clip_limit = _clamp_int(
            frame_clip_limit,
            default=0,
            min_v=0,
            max_v=_CLIP_REVIEW_MAX_FRAME_CLIPS,
        )
        frames_per_clip = _clamp_int(
            frames_per_clip,
            default=0,
            min_v=0,
            max_v=_CLIP_REVIEW_MAX_FRAMES_PER_CLIP,
        )
        variant_rows = [item for item in (variants_payload.get("candidates") or []) if isinstance(item, dict)]
        chapter_rows = [item for item in (chapters_payload.get("chapters") or []) if isinstance(item, dict)]

        def _norm_rank(value: Any) -> Optional[int]:
            try:
                rank = int(value)
            except Exception:
                return None
            return rank if rank > 0 else None

        def _norm_variant_id(value: Any) -> str:
            return str(value or "").strip()

        def _norm_time_key(start_v: Any, end_v: Any) -> Optional[Tuple[int, int]]:
            try:
                start = float(start_v)
                end = float(end_v)
            except Exception:
                return None
            if not (end > start):
                return None
            return (int(round(start * 1000.0)), int(round(end * 1000.0)))

        def _path_key(value: Any) -> Optional[str]:
            raw = str(value or "").strip()
            if not raw:
                return None
            try:
                return str(Path(raw).resolve())
            except Exception:
                return str(Path(raw))

        def _chapter_token(chapter: Dict[str, Any], idx: int) -> Tuple[Any, int, int]:
            return (
                chapter.get("id", idx),
                int(round(_safe_float(chapter.get("start_s"), -1.0) * 1000.0)),
                int(round(_safe_float(chapter.get("end_s"), -1.0) * 1000.0)),
            )

        chapter_excerpt_by_token: Dict[Tuple[Any, int, int], Optional[Dict[str, Any]]] = {}
        for idx, chapter in enumerate(chapter_rows):
            excerpt = chapter.get("transcript_excerpt")
            chapter_excerpt_by_token[_chapter_token(chapter, idx)] = excerpt if isinstance(excerpt, dict) else None

        raw_chapters = chapter_rows
        chapters_path = proj.analysis_dir / "chapters.json"
        if chapters_path.exists():
            try:
                data = json.loads(chapters_path.read_text(encoding="utf-8"))
                loaded = [item for item in (data.get("chapters") or []) if isinstance(item, dict)]
                if loaded:
                    raw_chapters = loaded
            except Exception:
                pass

        chapter_items: list[Dict[str, Any]] = []
        for idx, chapter in enumerate(raw_chapters):
            item = {
                "index": idx,
                "id": chapter.get("id"),
                "start_s": chapter.get("start_s"),
                "end_s": chapter.get("end_s"),
                "title": chapter.get("title"),
                "summary": chapter.get("summary"),
                "keywords": chapter.get("keywords"),
                "type": chapter.get("type"),
                "transcript_excerpt": chapter_excerpt_by_token.get(_chapter_token(chapter, idx)),
            }
            chapter_items.append(item)

        def _chapter_context_for_time(t: float) -> Dict[str, Any]:
            current_idx: Optional[int] = None
            for idx, chapter in enumerate(chapter_items):
                start_s = _safe_float(chapter.get("start_s"), -1.0)
                end_s = _safe_float(chapter.get("end_s"), -1.0)
                if end_s > start_s >= 0.0 and start_s <= t < end_s:
                    current_idx = idx
                    break
            if current_idx is None and chapter_items:
                last = chapter_items[-1]
                if t >= _safe_float(last.get("start_s"), float("inf")):
                    current_idx = len(chapter_items) - 1
            return {
                "chapter_index": current_idx,
                "current": chapter_items[current_idx] if current_idx is not None else None,
                "previous": chapter_items[current_idx - 1] if current_idx is not None and current_idx > 0 else None,
                "next": chapter_items[current_idx + 1] if current_idx is not None and (current_idx + 1) < len(chapter_items) else None,
            }

        variants_by_rank: Dict[int, Dict[str, Any]] = {}
        variants_by_id: Dict[str, Dict[str, Any]] = {}
        for row in variant_rows:
            rank = _norm_rank(row.get("candidate_rank"))
            if rank is not None and rank not in variants_by_rank:
                variants_by_rank[rank] = row
            candidate_id = str(row.get("candidate_id") or "").strip()
            if candidate_id and candidate_id not in variants_by_id:
                variants_by_id[candidate_id] = row

        director_picks: list[Dict[str, Any]] = []
        director_path = proj.analysis_dir / "director.json"
        director_source = None
        director_provenance = None
        if director_path.exists():
            try:
                director_payload = json.loads(director_path.read_text(encoding="utf-8"))
                director_picks = [
                    item for item in (director_payload.get("picks") or []) if isinstance(item, dict)
                ]
                director_source = (
                    str(
                        (director_payload.get("config") or {}).get("source")
                        or director_payload.get("source")
                        or ""
                    ).strip()
                    or None
                )
                payload_provenance = director_payload.get("provenance")
                if isinstance(payload_provenance, dict):
                    director_provenance = payload_provenance
            except Exception:
                director_picks = []

        director_by_rank: Dict[int, Dict[str, Any]] = {}
        director_by_id: Dict[str, Dict[str, Any]] = {}
        for pick in director_picks:
            rank = _norm_rank(pick.get("candidate_rank") or pick.get("rank"))
            if rank is not None and rank not in director_by_rank:
                director_by_rank[rank] = pick
            candidate_id = str(pick.get("candidate_id") or "").strip()
            if candidate_id and candidate_id not in director_by_id:
                director_by_id[candidate_id] = pick

        selections = [item for item in (proj_data.get("selections") or []) if isinstance(item, dict)]
        selection_by_id: Dict[str, Dict[str, Any]] = {}
        selections_by_rank: Dict[int, list[Dict[str, Any]]] = {}
        selections_by_key: Dict[Tuple[int, str], list[Dict[str, Any]]] = {}
        for selection in selections:
            selection_id = str(selection.get("id") or "").strip()
            if selection_id:
                selection_by_id[selection_id] = selection
            rank = _norm_rank(selection.get("candidate_rank") or selection.get("rank"))
            if rank is None:
                continue
            selections_by_rank.setdefault(rank, []).append(selection)
            variant_id = _norm_variant_id(
                selection.get("variant_id")
                or selection.get("best_variant_id")
                or selection.get("chosen_variant_id")
            )
            if variant_id:
                selections_by_key.setdefault((rank, variant_id), []).append(selection)

        project_export_records = [item for item in (proj_data.get("exports") or []) if isinstance(item, dict)]
        export_record_by_output: Dict[str, Dict[str, Any]] = {}
        export_record_by_id: Dict[str, Dict[str, Any]] = {}
        for record in project_export_records:
            output_key = _path_key(record.get("output"))
            if output_key and output_key not in export_record_by_output:
                export_record_by_output[output_key] = record
            output_raw = str(record.get("output") or "").strip()
            if output_raw:
                export_record_by_id.setdefault(Path(output_raw).stem, record)

        exports_by_rank: Dict[int, list[Dict[str, Any]]] = {}
        exports_by_key: Dict[Tuple[int, str], list[Dict[str, Any]]] = {}
        scanned_exports = scan_project_exports(proj.exports_dir)
        for export in scanned_exports:
            record = (
                export_record_by_output.get(_path_key(export.mp4_path))
                or export_record_by_id.get(export.export_id)
            )
            selection_id = str((record or {}).get("selection_id") or "").strip()
            selection = selection_by_id.get(selection_id) if selection_id else None
            metadata_selection = export.metadata.get("selection") if isinstance(export.metadata.get("selection"), dict) else {}
            rank = _norm_rank(
                (selection or {}).get("candidate_rank")
                or metadata_selection.get("candidate_rank")
            )
            variant_id = _norm_variant_id(
                (selection or {}).get("variant_id")
                or metadata_selection.get("variant_id")
            )
            export_item = {
                "export_id": export.export_id,
                "mp4_path": str(export.mp4_path),
                "mp4_filename": export.mp4_path.name,
                "metadata_path": str(export.metadata_path) if export.metadata_path else None,
                "created_at": export.created_at,
                "duration_seconds": export.duration_seconds,
                "file_size_bytes": export.file_size_bytes,
                "title": export.metadata.get("title", ""),
                "description": export.metadata.get("description", ""),
                "template": export.metadata.get("template", ""),
                "privacy": str(export.metadata.get("privacy") or _publish_policy()["default_privacy"]).strip().lower() or "private",
                "publish_at": str(export.metadata.get("publish_at") or "").strip() or None,
                "selection_id": selection_id or None,
                "candidate_rank": rank,
                "variant_id": variant_id or None,
                "selection_status": (record or {}).get("status"),
            }
            if rank is not None:
                exports_by_rank.setdefault(rank, []).append(export_item)
                if variant_id:
                    exports_by_key.setdefault((rank, variant_id), []).append(export_item)

        def _serialize_selection(selection: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": str(selection.get("id") or ""),
                "created_at": selection.get("created_at"),
                "start_s": selection.get("start_s"),
                "end_s": selection.get("end_s"),
                "title": selection.get("title"),
                "notes": selection.get("notes"),
                "template": selection.get("template"),
                "candidate_rank": _norm_rank(selection.get("candidate_rank") or selection.get("rank")),
                "candidate_peak_time_s": selection.get("candidate_peak_time_s"),
                "variant_id": _norm_variant_id(
                    selection.get("variant_id")
                    or selection.get("best_variant_id")
                    or selection.get("chosen_variant_id")
                )
                or None,
                "director_confidence": selection.get("director_confidence"),
            }

        clips: list[Dict[str, Any]] = []
        for clip_idx, candidate in enumerate(candidate_items):
            candidate_rank = _norm_rank(candidate.get("rank"))
            candidate_id = str(candidate.get("candidate_id") or "").strip()
            variant_row = (
                variants_by_id.get(candidate_id)
                or (variants_by_rank.get(candidate_rank) if candidate_rank is not None else None)
                or {}
            )
            variant_options = [
                dict(item) for item in (variant_row.get("variants") or []) if isinstance(item, dict)
            ]

            director_pick = (
                director_by_id.get(candidate_id)
                or (director_by_rank.get(candidate_rank) if candidate_rank is not None else None)
            )
            director_pick_out = None
            chosen_variant_id = ""
            if isinstance(director_pick, dict):
                chosen_variant_id = _norm_variant_id(
                    director_pick.get("variant_id") or director_pick.get("best_variant_id")
                )
                director_pick_out = {
                    "candidate_rank": _norm_rank(director_pick.get("candidate_rank") or director_pick.get("rank")),
                    "candidate_id": director_pick.get("candidate_id"),
                    "variant_id": chosen_variant_id or None,
                    "start_s": director_pick.get("start_s"),
                    "end_s": director_pick.get("end_s"),
                    "duration_s": director_pick.get("duration_s"),
                    "title": director_pick.get("title"),
                    "hook": director_pick.get("hook"),
                    "description": director_pick.get("description"),
                    "hashtags": director_pick.get("hashtags") or director_pick.get("tags") or [],
                    "template": director_pick.get("template"),
                    "confidence": director_pick.get("confidence"),
                    "reasons": director_pick.get("reasons") or [],
                    "chapter_index": director_pick.get("chapter_index"),
                    "provenance": director_pick.get("provenance")
                        if isinstance(director_pick.get("provenance"), dict)
                        else director_provenance,
                    "source": director_pick.get("packaging_source") or director_source,
                }

            peak_time_s = _safe_float(
                candidate.get("peak_time_s"),
                _safe_float(candidate.get("start_s"), 0.0),
            )
            chapter_context = _chapter_context_for_time(peak_time_s)

            matching_selections: list[Dict[str, Any]] = []
            seen_selection_ids = set()
            if candidate_rank is not None and chosen_variant_id:
                for selection in selections_by_key.get((candidate_rank, chosen_variant_id), []):
                    selection_id = str(selection.get("id") or "")
                    if selection_id in seen_selection_ids:
                        continue
                    matching_selections.append(_serialize_selection(selection))
                    seen_selection_ids.add(selection_id)
            if candidate_rank is not None:
                for selection in selections_by_rank.get(candidate_rank, []):
                    selection_id = str(selection.get("id") or "")
                    if selection_id in seen_selection_ids:
                        continue
                    matching_selections.append(_serialize_selection(selection))
                    seen_selection_ids.add(selection_id)

            primary_variant = None
            primary_variant_source = None
            if chosen_variant_id:
                primary_variant = next(
                    (
                        dict(item)
                        for item in variant_options
                        if _norm_variant_id(item.get("variant_id")) == chosen_variant_id
                    ),
                    None,
                )
                if primary_variant is not None:
                    primary_variant_source = "director_pick"
            if primary_variant is None and matching_selections:
                selection_variant_id = str(matching_selections[0].get("variant_id") or "").strip()
                if selection_variant_id:
                    primary_variant = next(
                        (
                            dict(item)
                            for item in variant_options
                            if _norm_variant_id(item.get("variant_id")) == selection_variant_id
                        ),
                        None,
                    )
                    if primary_variant is not None:
                        primary_variant_source = "selection"
            if primary_variant is None and variant_options:
                primary_variant = dict(variant_options[0])
                primary_variant_source = "first_available_variant"

            matching_exports: list[Dict[str, Any]] = []
            seen_export_ids = set()
            if candidate_rank is not None and chosen_variant_id:
                for export_item in exports_by_key.get((candidate_rank, chosen_variant_id), []):
                    export_id = str(export_item.get("export_id") or "")
                    if export_id in seen_export_ids:
                        continue
                    matching_exports.append(dict(export_item))
                    seen_export_ids.add(export_id)
            if candidate_rank is not None:
                for export_item in exports_by_rank.get(candidate_rank, []):
                    export_id = str(export_item.get("export_id") or "")
                    if export_id in seen_export_ids:
                        continue
                    matching_exports.append(dict(export_item))
                    seen_export_ids.add(export_id)

            review_focus = [
                "Does this clip hook quickly and still make sense without outside context?",
                "Is the payoff clear enough to beat the other shortlisted clips?",
            ]
            if variant_options:
                review_focus.append("Pick the strongest variant for setup/payoff balance before export.")
            if matching_exports:
                review_focus.append("Review the exported MP4 for pacing, captions, and upload readiness.")
            else:
                review_focus.append("No export exists yet; decide keep/reject and packaging before export.")

            review_id = candidate_id or (f"candidate-rank-{candidate_rank}" if candidate_rank is not None else "candidate")
            visual_review = (
                _build_clip_review_frame_payload(
                    proj=proj,
                    candidate=candidate,
                    review_id=review_id,
                    frames_per_clip=frames_per_clip,
                )
                if frame_clip_limit > 0 and frames_per_clip > 0 and clip_idx < frame_clip_limit
                else {"frame_source": None, "frame_count": 0, "frames": []}
            )
            clips.append(
                {
                    "review_id": review_id,
                    "candidate": dict(candidate),
                    "chapter_context": chapter_context,
                    "variant_options": variant_options,
                    "primary_variant": primary_variant,
                    "primary_variant_source": primary_variant_source,
                    "director_pick": director_pick_out,
                    "selections": matching_selections,
                    "exports": matching_exports,
                    "format_recommendations": _clip_format_recommendations(
                        candidate=candidate,
                        primary_variant=primary_variant,
                        shortlist_count=len(candidate_items),
                        chapter_context=chapter_context,
                        has_export=bool(matching_exports),
                    ),
                    "show_format_recommendations": _clip_show_format_recommendations(
                        candidate=candidate,
                        primary_variant=primary_variant,
                        shortlist_count=len(candidate_items),
                        chapter_context=chapter_context,
                        has_export=bool(matching_exports),
                    ),
                    "status": {
                        "has_transcript_excerpt": bool(candidate.get("transcript_excerpt")),
                        "has_chat_excerpt": bool(candidate.get("chat_excerpt")),
                        "has_visual_frames": bool(visual_review.get("frames")),
                        "has_variants": bool(variant_options),
                        "has_director_pick": director_pick_out is not None,
                        "has_selection": bool(matching_selections),
                        "has_export": bool(matching_exports),
                    },
                    "visual_review": visual_review,
                    "review_focus": review_focus,
                }
            )

        return {
            "meta": {
                "project_id": pid,
                "clip_count": len(clips),
                "total_candidates": candidates_payload.get("total_candidates"),
                "director_pick_count": len(director_picks),
                "selection_count": len(selections),
                "export_count": len(scanned_exports),
            },
            "workflow": {
                "intended_use": "Use these packets for final LLM judgment on shortlisted clips after broad pipeline selection.",
                "external_ai_status": _external_ai_status(proj=proj, proj_data=proj_data),
                "next_actions": {
                    "semantic": "POST /api/actions/ai/apply_semantic",
                    "chapters": "POST /api/actions/ai/apply_chapter_labels",
                    "director": "POST /api/actions/ai/apply_director_picks",
                    "export": "POST /api/actions/export_director_picks",
                    "publish_accounts": "GET /api/actions/publish/accounts",
                    "publish_queue": "POST /api/actions/publish/queue",
                },
            },
            "limits": {
                "candidates": candidates_payload.get("limits") or {},
                "variants": variants_payload.get("limits") or {},
                "chapters": chapters_payload.get("limits") or {},
                "visual_review": {
                    "frame_clip_limit": frame_clip_limit,
                    "frames_per_clip": frames_per_clip,
                },
            },
            "channel_format_spec": _channel_format_playbook(),
            "review_rubric": {
                "goal": "Choose which shortlisted clips deserve export/upload and finalize the variant plus packaging.",
                "criteria": [
                    "Immediate hook without confusing setup",
                    "Clear standalone payoff",
                    "Strong relative value versus competing shortlisted clips",
                    "Packaging/title/hook that matches the actual clip",
                    "Upload readiness if an exported MP4 already exists",
                ],
                "preferred_outputs": {
                    "semantic": "keep/reject + semantic_score + concise reason",
                    "director": "variant_id + format_id + show_format_id + title + hook + description + hashtags + confidence",
                },
            },
            "clips": clips,
        }

    @router.get("/ai/clip_review", openapi_extra={"x-openai-isConsequential": False})
    def ai_clip_review(
        request: Request,
        project_id: str,
        top_n: Optional[int] = None,
        chat_top_n: Optional[int] = None,
        window_s: float = 4.0,
        max_chars: int = 800,
        chat_lines: int = 8,
        max_variants: int = 8,
        chapter_limit: int = 50,
        chapter_max_chars: int = 1200,
        frame_clip_limit: int = 0,
        frames_per_clip: int = 0,
    ):
        """Fetch joined shortlist review packets for final bot judgment."""
        _rate_limit(request)
        pid = _validate_project_id(project_id)
        proj, proj_data = _load_project(pid)

        candidates_payload = _build_ai_candidates_payload(
            project_id=pid,
            top_n=top_n,
            chat_top_n=chat_top_n,
            window_s=window_s,
            max_chars=max_chars,
            chat_lines=chat_lines,
        )
        candidate_ids = [
            str(candidate.get("candidate_id") or "").strip()
            for candidate in (candidates_payload.get("candidates") or [])
            if isinstance(candidate, dict) and str(candidate.get("candidate_id") or "").strip()
        ]
        variants_payload = _build_ai_variants_payload(
            project_id=pid,
            candidate_ids=",".join(candidate_ids),
            max_variants=max_variants,
        )
        try:
            chapters_payload = _build_ai_chapters_payload(
                project_id=pid,
                offset=0,
                limit=chapter_limit,
                max_chars=chapter_max_chars,
            )
        except HTTPException as exc:
            if exc.status_code != 404 or exc.detail != "no_chapters":
                raise
            chapters_payload = {
                "meta": {"project_id": pid, "chapter_count": 0, "generated_at": None},
                "limits": {"offset": 0, "limit": chapter_limit, "max_chars": chapter_max_chars},
                "chapters": [],
            }

        return JSONResponse(
            _build_ai_clip_review_payload(
                project_id=pid,
                candidates_payload=candidates_payload,
                variants_payload=variants_payload,
                chapters_payload=chapters_payload,
                frame_clip_limit=frame_clip_limit,
                frames_per_clip=frames_per_clip,
                proj=proj,
                proj_data=proj_data,
            )
        )

    @router.get("/ai/bundle", openapi_extra={"x-openai-isConsequential": False})
    def ai_bundle(
        request: Request,
        project_id: str,
        top_n: Optional[int] = None,
        chat_top_n: Optional[int] = None,
        window_s: float = 4.0,
        max_chars: int = 800,
        chat_lines: int = 8,
        max_variants: int = 8,
        chapter_limit: int = 50,
        chapter_max_chars: int = 1200,
        frame_clip_limit: int = 0,
        frames_per_clip: int = 0,
    ):
        """Fetch a single external-AI work bundle for Gondull/Actions workflows."""
        _rate_limit(request)
        pid = _validate_project_id(project_id)
        proj, proj_data = _load_project(pid)

        candidates_payload = _build_ai_candidates_payload(
            project_id=pid,
            top_n=top_n,
            chat_top_n=chat_top_n,
            window_s=window_s,
            max_chars=max_chars,
            chat_lines=chat_lines,
        )
        candidate_ids = [
            str(c.get("candidate_id") or "").strip()
            for c in (candidates_payload.get("candidates") or [])
            if isinstance(c, dict) and str(c.get("candidate_id") or "").strip()
        ]
        variants_payload = _build_ai_variants_payload(
            project_id=pid,
            candidate_ids=",".join(candidate_ids),
            max_variants=max_variants,
        )

        try:
            chapters_payload = _build_ai_chapters_payload(
                project_id=pid,
                offset=0,
                limit=chapter_limit,
                max_chars=chapter_max_chars,
            )
            chapters_available = True
            chapters_reason = None
        except HTTPException as exc:
            if exc.status_code == 404 and exc.detail == "no_chapters":
                chapters_payload = {
                    "meta": {"project_id": pid, "chapter_count": 0, "generated_at": None},
                    "limits": {"offset": 0, "limit": chapter_limit, "max_chars": chapter_max_chars},
                    "chapters": [],
                }
                chapters_available = False
                chapters_reason = "no_chapters"
            else:
                raise

        external_status = _external_ai_status(proj=proj, proj_data=proj_data)
        clip_review_payload = _build_ai_clip_review_payload(
            project_id=pid,
            candidates_payload=candidates_payload,
            variants_payload=variants_payload,
            chapters_payload=chapters_payload,
            frame_clip_limit=frame_clip_limit,
            frames_per_clip=frames_per_clip,
            proj=proj,
            proj_data=proj_data,
        )
        return JSONResponse(
            {
                "meta": candidates_payload.get("meta") or {"project_id": pid},
                "workflow": {
                    "preferred_llm_mode": "external_strict",
                    "supported_llm_modes": list(_LLM_MODE_ENUM),
                    "external_ai_status": external_status,
                    "chapters_available": chapters_available,
                    "chapters_reason": chapters_reason,
                    "next_actions": {
                        "clip_review": "GET /api/actions/ai/clip_review",
                        "semantic": "POST /api/actions/ai/apply_semantic",
                        "chapters": "POST /api/actions/ai/apply_chapter_labels",
                        "director": "POST /api/actions/ai/apply_director_picks",
                        "export": "POST /api/actions/export_director_picks",
                    },
                },
                "guidance": candidates_payload.get("ai_guidance") or {},
                "limits": {
                    "candidates": candidates_payload.get("limits") or {},
                    "variants": variants_payload.get("limits") or {},
                    "chapters": chapters_payload.get("limits") or {},
                },
                "strategy": candidates_payload.get("strategy") or {},
                "candidates": candidates_payload.get("candidates") or [],
                "variants": variants_payload.get("candidates") or [],
                "chapters": chapters_payload.get("chapters") or [],
                "clip_review": clip_review_payload,
            }
        )

    @router.post("/ai/apply_semantic", openapi_extra={"x-openai-isConsequential": False})
    def ai_apply_semantic(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        """Apply semantic scoring/filter decisions to highlight candidates (by candidate_id or rank)."""
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        project_id = _validate_project_id(str(body.get("project_id") or ""))
        provenance = _extract_ai_provenance(body, client_request_id=client_request_id)
        items = body.get("items") or []
        if not isinstance(items, list):
            raise HTTPException(status_code=400, detail="invalid_items")

        updates: list[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            candidate_id = str(it.get("candidate_id") or "").strip()
            try:
                rank = int(it.get("candidate_rank"))
            except Exception:
                rank = 0
            if not candidate_id and rank <= 0:
                continue

            sem = it.get("semantic_score")
            if sem is None:
                continue
            try:
                semantic_score = float(sem)
            except Exception:
                continue
            semantic_score = max(0.0, min(1.0, semantic_score))

            keep = it.get("keep")
            keep_b = True if keep is None else bool(keep)
            updates.append(
                {
                    "candidate_id": candidate_id or None,
                    "candidate_rank": rank if rank > 0 else None,
                    "semantic_score": semantic_score,
                    "reason": str(it.get("reason") or ""),
                    "best_quote": str(it.get("best_quote") or ""),
                    "keep": keep_b,
                }
            )

        if not updates:
            raise HTTPException(status_code=400, detail="no_updates")

        proj, _proj_data = _load_project(project_id)

        from ..project import update_project, utc_now_iso

        now = utc_now_iso()
        semantic_provenance = {**provenance, "recorded_at": now}
        missing_ranks: list[int] = []
        missing_candidate_ids: list[str] = []
        updated_keys: set[str] = set()
        semantic_stats: Dict[str, int] = {
            "candidate_count_before": 0,
            "candidate_count_after": 0,
            "dropped_count": 0,
            "semantic_scored_count": 0,
        }

        updates_by_id: Dict[str, Dict[str, Any]] = {}
        updates_by_rank: Dict[int, Dict[str, Any]] = {}
        for u in updates:
            cid = str(u.get("candidate_id") or "").strip()
            cr = u.get("candidate_rank")
            if cid:
                updates_by_id[cid] = u
            if cr is not None:
                try:
                    r = int(cr)
                except Exception:
                    r = 0
                if r > 0:
                    updates_by_rank[r] = u

        def _apply_to_candidate(c: Dict[str, Any], upd: Dict[str, Any]) -> None:
            ai = c.get("ai")
            if not isinstance(ai, dict):
                ai = {}
                c["ai"] = ai
            ai["semantic_score"] = upd["semantic_score"]
            ai["semantic_reason"] = upd.get("reason") or ""
            ai["semantic_best_quote"] = upd.get("best_quote") or ""
            ai["semantic_keep"] = bool(upd.get("keep", True))
            ai["semantic_source"] = semantic_provenance["source"]
            ai["semantic_updated_at"] = now
            ai["semantic_provenance"] = dict(semantic_provenance)

            # Compatibility fields used by local-LLM semantic scoring.
            c["score_semantic"] = upd["semantic_score"]
            c["llm_reason"] = upd.get("reason") or ""
            c["llm_quote"] = upd.get("best_quote") or ""

        highlights_cfg = (profile.get("analysis", {}) or {}).get("highlights", {}) or {}
        semantic_weight = float(highlights_cfg.get("llm_semantic_weight", 0.3))

        def _materialize_semantic_shortlist(candidates: Any) -> tuple[list[Dict[str, Any]], Dict[str, int]]:
            if not isinstance(candidates, list):
                raise KeyError("analysis.highlights.candidates missing")

            kept_candidates: list[Dict[str, Any]] = []
            semantic_scored_count = 0

            for cand in candidates:
                if not isinstance(cand, dict):
                    continue

                ai = cand.get("ai")
                ai_meta = ai if isinstance(ai, dict) else {}
                if ai_meta.get("semantic_keep") is False:
                    continue

                try:
                    signal_score = float(cand.get("score_signal", cand.get("score", 0.0)) or 0.0)
                except Exception:
                    signal_score = 0.0
                cand["score_signal"] = signal_score

                semantic_raw = ai_meta.get("semantic_score", cand.get("score_semantic"))
                semantic_score: Optional[float]
                if semantic_raw is None:
                    semantic_score = None
                else:
                    try:
                        semantic_score = float(semantic_raw)
                    except Exception:
                        semantic_score = None

                if semantic_score is not None:
                    semantic_score = max(0.0, min(1.0, semantic_score))
                    semantic_z = (semantic_score - 0.5) * 4.0
                    cand["score_semantic"] = semantic_score
                    cand["score"] = (1.0 - semantic_weight) * signal_score + semantic_weight * semantic_z
                    semantic_scored_count += 1

                kept_candidates.append(cand)

            kept_candidates.sort(key=lambda c: float(c.get("score", 0.0) or 0.0), reverse=True)
            for idx, cand in enumerate(kept_candidates, start=1):
                cand["rank"] = idx

            candidate_count_before = len([c for c in candidates if isinstance(c, dict)])
            candidate_count_after = len(kept_candidates)
            return kept_candidates, {
                "candidate_count_before": candidate_count_before,
                "candidate_count_after": candidate_count_after,
                "dropped_count": max(0, candidate_count_before - candidate_count_after),
                "semantic_scored_count": semantic_scored_count,
            }

        def _upd_project(d: Dict[str, Any]) -> None:
            nonlocal missing_ranks
            nonlocal missing_candidate_ids
            nonlocal updated_keys
            nonlocal semantic_stats
            candidates = ((d.get("analysis", {}) or {}).get("highlights", {}) or {}).get("candidates") or []
            if not isinstance(candidates, list):
                raise KeyError("analysis.highlights.candidates missing")

            by_rank = {int(c.get("rank") or 0): c for c in candidates if isinstance(c, dict)}
            by_id: Dict[str, Dict[str, Any]] = {}
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("candidate_id") or "").strip()
                if cid:
                    by_id[cid] = c

            for upd in updates:
                cid = str(upd.get("candidate_id") or "").strip()
                rank = upd.get("candidate_rank")
                cand: Optional[Dict[str, Any]] = None
                if cid:
                    cand = by_id.get(cid)
                    if not cand:
                        missing_candidate_ids.append(cid)
                        continue
                else:
                    try:
                        r = int(rank or 0)
                    except Exception:
                        r = 0
                    if r <= 0:
                        continue
                    cand = by_rank.get(r)
                    if not cand:
                        missing_ranks.append(r)
                        continue

                _apply_to_candidate(cand, upd)
                key = str(cand.get("candidate_id") or cand.get("rank") or "")
                if key:
                    updated_keys.add(key)

            d.setdefault("analysis", {})
            analysis = d["analysis"]
            highlights_meta = analysis.setdefault("highlights", {})
            semantic_candidates, semantic_stats = _materialize_semantic_shortlist(candidates)
            highlights_meta["candidates"] = semantic_candidates
            signals_used = highlights_meta.get("signals_used")
            if not isinstance(signals_used, dict):
                signals_used = {}
                highlights_meta["signals_used"] = signals_used
            signals_used["llm_semantic"] = True
            highlights_meta["llm_semantic_updated_at"] = now
            highlights_meta["semantic_provenance"] = dict(semantic_provenance)
            d["analysis"].setdefault("actions", {})
            d["analysis"]["actions"]["semantic_applied_at"] = now
            d["analysis"]["actions"]["semantic_provenance"] = dict(semantic_provenance)
            d["analysis"]["actions"]["semantic_stats"] = dict(semantic_stats)

        update_project(proj, _upd_project)

        # Best-effort: also update analysis/highlights.json for consistency.
        try:
            highlights_path = proj.analysis_dir / "highlights.json"
            if highlights_path.exists():
                h = json.loads(highlights_path.read_text(encoding="utf-8"))
                hcands = h.get("candidates") or []
                if isinstance(hcands, list):
                    for c in hcands:
                        if not isinstance(c, dict):
                            continue
                        cid = str(c.get("candidate_id") or "").strip()
                        if cid and cid in updates_by_id:
                            _apply_to_candidate(c, updates_by_id[cid])
                            continue
                        r = int(c.get("rank") or 0)
                        if r in updates_by_rank:
                            _apply_to_candidate(c, updates_by_rank[r])
                    semantic_candidates, file_stats = _materialize_semantic_shortlist(hcands)
                    h["candidates"] = semantic_candidates
                    signals_used = h.get("signals_used")
                    if not isinstance(signals_used, dict):
                        signals_used = {}
                        h["signals_used"] = signals_used
                    signals_used["llm_semantic"] = True
                    h["llm_semantic_updated_at"] = now
                    h["semantic_provenance"] = dict(semantic_provenance)
                    h["semantic_stats"] = dict(file_stats)
                    save_json(highlights_path, h)
        except Exception:
            pass

        payload = {
            "ok": True,
            "project_id": project_id,
            "updated_count": len(updated_keys) if updated_keys else 0,
            "candidate_count_before": semantic_stats["candidate_count_before"],
            "candidate_count_after": semantic_stats["candidate_count_after"],
            "dropped_count": semantic_stats["dropped_count"],
            "semantic_scored_count": semantic_stats["semantic_scored_count"],
            "missing_ranks": sorted(set(missing_ranks)),
            "missing_candidate_ids": sorted(set(missing_candidate_ids)),
            "provenance": semantic_provenance,
        }
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.post("/ai/apply_chapter_labels", openapi_extra={"x-openai-isConsequential": False})
    def ai_apply_chapter_labels(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        """Apply chapter titles/summaries/keywords (by chapter id) to chapters.json."""
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        project_id = _validate_project_id(str(body.get("project_id") or ""))
        provenance = _extract_ai_provenance(body, client_request_id=client_request_id)
        items = body.get("items") or []
        if not isinstance(items, list):
            raise HTTPException(status_code=400, detail="invalid_items")

        labels: Dict[int, Dict[str, Any]] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            cid = it.get("chapter_id", it.get("id"))
            try:
                chapter_id = int(cid)
            except Exception:
                continue
            if chapter_id < 0:
                continue
            labels[chapter_id] = {
                "title": it.get("title"),
                "summary": it.get("summary"),
                "keywords": it.get("keywords"),
                "type": it.get("type"),
            }

        if not labels:
            raise HTTPException(status_code=400, detail="no_updates")

        proj, _proj_data = _load_project(project_id)
        chapters_path = proj.analysis_dir / "chapters.json"
        if not chapters_path.exists():
            raise HTTPException(status_code=404, detail="no_chapters")

        from ..project import update_project, utc_now_iso

        now = utc_now_iso()
        label_provenance = {**provenance, "recorded_at": now}
        data = json.loads(chapters_path.read_text(encoding="utf-8"))
        chapters = data.get("chapters") or []
        if not isinstance(chapters, list):
            raise HTTPException(status_code=409, detail="invalid_chapters_file")

        missing_ids: list[int] = []
        updated = 0
        by_id = {int(c.get("id") or 0): c for c in chapters if isinstance(c, dict)}
        for chapter_id, patch in labels.items():
            ch = by_id.get(chapter_id)
            if not ch:
                missing_ids.append(chapter_id)
                continue
            if patch.get("title") is not None:
                ch["title"] = str(patch.get("title") or "")
            if patch.get("summary") is not None:
                ch["summary"] = str(patch.get("summary") or "")
            if patch.get("keywords") is not None:
                kws = patch.get("keywords") or []
                if isinstance(kws, list):
                    ch["keywords"] = [str(x) for x in kws if str(x).strip()]
            if patch.get("type") is not None:
                ch["type"] = str(patch.get("type") or "content")
            ch["labels_source"] = label_provenance["source"]
            ch["labels_updated_at"] = now
            ch["labels_provenance"] = dict(label_provenance)
            updated += 1

        data["labels_updated_at"] = now
        data["labels_provenance"] = dict(label_provenance)
        save_json(chapters_path, data)

        def _upd(d: Dict[str, Any]) -> None:
            d.setdefault("analysis", {})
            d["analysis"].setdefault("chapters", {})
            d["analysis"]["chapters"]["labels_updated_at"] = now
            d["analysis"]["chapters"]["labels_provenance"] = dict(label_provenance)

        update_project(proj, _upd)

        payload = {
            "ok": True,
            "project_id": project_id,
            "updated_count": updated,
            "missing_ids": sorted(set(missing_ids)),
            "provenance": label_provenance,
        }
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.post("/ai/apply_director_picks", openapi_extra={"x-openai-isConsequential": False})
    def ai_apply_director_picks(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        """Apply director picks (candidate_id or rank + variant_id + packaging) into analysis/director.json."""
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        project_id = _validate_project_id(str(body.get("project_id") or ""))
        provenance = _extract_ai_provenance(body, client_request_id=client_request_id)
        picks_in = body.get("picks") or body.get("items") or []
        if not isinstance(picks_in, list):
            raise HTTPException(status_code=400, detail="invalid_picks")

        proj, proj_data = _load_project(project_id)

        from ..clip_variants import load_clip_variants
        from ..project import update_project, utc_now_iso

        variants = load_clip_variants(proj) or []
        cv_by_rank = {int(cv.candidate_rank): cv for cv in variants}
        cv_by_id: Dict[str, Any] = {}
        for cv in variants:
            cid = str(getattr(cv, "candidate_id", "") or "").strip()
            if cid:
                cv_by_id[cid] = cv

        candidates = (((proj_data.get("analysis") or {}).get("highlights") or {}).get("candidates") or [])
        cand_by_rank = {int(c.get("rank") or 0): c for c in candidates if isinstance(c, dict)}
        rank_by_cid: Dict[str, int] = {}
        for c in candidates:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("candidate_id") or "").strip()
            if not cid:
                continue
            try:
                r = int(c.get("rank") or 0)
            except Exception:
                r = 0
            if r > 0:
                rank_by_cid[cid] = r

        now = utc_now_iso()
        packaging_provenance = {**provenance, "recorded_at": now}
        picks: list[Dict[str, Any]] = []
        missing: list[Dict[str, Any]] = []
        director_cfg = _director_config_from_profile()
        allowed_templates = set(str(x) for x in list(director_cfg.allowed_templates or []) if str(x).strip())
        template_default = _director_default_template(director_cfg)
        chapter_ranges = _load_chapter_ranges(proj)
        used_intervals: list[tuple[float, float, float]] = []  # start, end, peak
        used_chapters: Dict[int, int] = {}
        max_picks = max(1, int(director_cfg.top_n))

        def _norm_tags(raw: Any) -> list[str]:
            return _normalize_hashtags(raw, defaults=list(director_cfg.hashtags or []), min_n=3, max_n=8)

        for idx, p in enumerate(picks_in, start=1):
            if not isinstance(p, dict):
                continue
            if len(picks) >= max_picks:
                missing.append({"index": idx, "error": "top_n_limit"})
                continue
            cand_id = str(p.get("candidate_id") or "").strip()
            cand_rank = 0
            if cand_id:
                cand_rank = int(rank_by_cid.get(cand_id) or 0)
            if cand_rank <= 0:
                try:
                    cand_rank = int(p.get("candidate_rank"))
                except Exception:
                    cand_rank = 0
            variant_id = str(p.get("variant_id") or p.get("best_variant_id") or "").strip()
            if cand_rank <= 0 or not variant_id:
                continue

            cv = cv_by_rank.get(cand_rank)
            if cv is None and cand_id:
                cv = cv_by_id.get(cand_id)
                if cv is not None:
                    cand_rank = int(getattr(cv, "candidate_rank", cand_rank) or cand_rank)
            if cv is None:
                missing.append({"candidate_rank": cand_rank, "candidate_id": cand_id or None, "variant_id": variant_id, "error": "candidate_not_found"})
                continue
            v = cv.get_variant(variant_id)
            if v is None:
                missing.append({"candidate_rank": cand_rank, "candidate_id": cand_id or None, "variant_id": variant_id, "error": "variant_not_found"})
                continue

            tags = _norm_tags(p.get("hashtags") or p.get("tags"))
            reason_text = str(p.get("reason") or "").strip()
            reasons_raw = p.get("reasons")
            reasons_out: list[str] = []
            if isinstance(reasons_raw, list):
                reasons_out = [str(x).strip() for x in reasons_raw if str(x).strip()]
            elif isinstance(reasons_raw, str) and reasons_raw.strip():
                reasons_out = [reasons_raw.strip()]
            if not reason_text and reasons_out:
                reason_text = "; ".join(reasons_out)
            if reason_text and not reasons_out:
                reasons_out = [reason_text]

            conf_raw = p.get("confidence")
            try:
                confidence = float(0.5 if conf_raw is None else conf_raw)
            except Exception:
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            cand = cand_by_rank.get(cand_rank) or {}
            start_s = float(getattr(v, "start_s", 0.0))
            end_s = float(getattr(v, "end_s", 0.0))
            if end_s <= start_s:
                missing.append({"candidate_rank": cand_rank, "candidate_id": cand_id or None, "variant_id": variant_id, "error": "invalid_variant_range"})
                continue
            duration_s = max(1e-6, end_s - start_s)
            peak_time = float(cand.get("peak_time_s") or getattr(cv, "candidate_peak_time_s", 0.0) or ((start_s + end_s) / 2.0))
            if not cand_id:
                cand_id = str(cand.get("candidate_id") or getattr(cv, "candidate_id", "") or "").strip()

            too_close = False
            too_close_error = ""
            for (a0, a1, _ap) in used_intervals:
                overlap_s = max(0.0, min(a1, end_s) - max(a0, start_s))
                if overlap_s <= 0.0:
                    continue
                overlap_ratio = overlap_s / max(duration_s, 1e-6)
                if overlap_s >= float(director_cfg.max_overlap_s) or overlap_ratio >= float(director_cfg.max_overlap_ratio):
                    too_close = True
                    too_close_error = "overlap_guard"
                    break
            if not too_close:
                for (_a0, _a1, prev_peak) in used_intervals:
                    if abs(prev_peak - peak_time) < float(director_cfg.min_gap_s):
                        too_close = True
                        too_close_error = "min_gap_guard"
                        break
            if too_close:
                missing.append(
                    {
                        "candidate_rank": cand_rank,
                        "candidate_id": cand_id or None,
                        "variant_id": variant_id,
                        "error": too_close_error or "too_close",
                    }
                )
                continue

            raw_ch_idx = p.get("chapter_index")
            ch_idx: Optional[int] = None
            if raw_ch_idx is not None:
                try:
                    parsed = int(raw_ch_idx)
                    if parsed >= 0:
                        ch_idx = parsed
                except Exception:
                    ch_idx = None
            if ch_idx is None and chapter_ranges:
                ch_idx = _chapter_index_for_time(chapter_ranges, peak_time)
            if ch_idx is not None:
                used_chapters.setdefault(ch_idx, 0)
                if used_chapters[ch_idx] >= 2:
                    missing.append(
                        {
                            "candidate_rank": cand_rank,
                            "candidate_id": cand_id or None,
                            "variant_id": variant_id,
                            "error": "chapter_diversity_cap",
                        }
                    )
                    continue

            title = _trim_copy_text(
                p.get("title") or cand.get("title") or cand.get("hook_text") or "Highlight",
                max_chars=int(director_cfg.title_max_chars),
            )
            hook = _trim_copy_text(
                p.get("hook") or cand.get("hook_text") or title,
                max_chars=int(director_cfg.hook_max_chars),
            )
            description = _trim_copy_text(
                p.get("description") or title,
                max_chars=int(director_cfg.description_max_chars),
            )
            packaging_issues = _packaging_quality_issues(
                title=title,
                hook=hook,
                description=description,
            )
            if packaging_issues:
                missing.append(
                    {
                        "candidate_rank": cand_rank,
                        "candidate_id": cand_id or None,
                        "variant_id": variant_id,
                        "error": "packaging_quality",
                        "issues": packaging_issues,
                    }
                )
                continue
            template = str(p.get("template") or "").strip()
            if not template or (allowed_templates and template not in allowed_templates):
                template = template_default

            pick = {
                "rank": len(picks) + 1,
                "candidate_rank": cand_rank,
                "candidate_id": cand_id or None,
                "peak_time_s": peak_time,
                "variant_id": variant_id,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": float(getattr(v, "duration_s", duration_s) or duration_s),
                "title": title,
                "hook": hook,
                "description": description,
                "hashtags": tags,
                "template": template,
                "packaging_source": packaging_provenance["source"],
                "packaging_error": None,
                "confidence": confidence,
                "reasons": reasons_out,
                "chapter_index": ch_idx,
                "signals": cand.get("meta", {}),
                "provenance": dict(packaging_provenance),
            }
            picks.append(pick)
            used_intervals.append((start_s, end_s, peak_time))
            if ch_idx is not None:
                used_chapters[ch_idx] = used_chapters.get(ch_idx, 0) + 1

        if not picks:
            raise HTTPException(status_code=400, detail="no_valid_picks")

        payload = {
            "created_at": now,
            "provenance": dict(packaging_provenance),
            "config": {
                "source": packaging_provenance["source"],
                "top_n": int(director_cfg.top_n),
                "min_gap_s": float(director_cfg.min_gap_s),
                "max_overlap_ratio": float(director_cfg.max_overlap_ratio),
                "max_overlap_s": float(director_cfg.max_overlap_s),
                "default_template": template_default,
                "allowed_templates": list(director_cfg.allowed_templates or []),
            },
            "pick_count": len(picks),
            "packaging_counts": {packaging_provenance["source"]: len(picks)},
            "picks": picks,
        }
        director_path = proj.analysis_dir / "director.json"
        save_json(director_path, payload)

        def _upd(d: Dict[str, Any]) -> None:
            d.setdefault("analysis", {})
            d["analysis"]["director"] = {
                "created_at": now,
                "pick_count": len(picks),
                "director_json": str(director_path.relative_to(proj.project_dir)),
                "source": packaging_provenance["source"],
                "provenance": dict(packaging_provenance),
            }

            # Best-effort: mirror packaging fields into candidates (like ai/director.py does).
            cands = ((d.get("analysis", {}) or {}).get("highlights", {}) or {}).get("candidates") or []
            if isinstance(cands, list):
                by_rank = {int(c.get("rank") or 0): c for c in cands if isinstance(c, dict)}
                for pk in picks:
                    c = by_rank.get(int(pk.get("candidate_rank") or 0))
                    if not c:
                        continue
                    ai = c.get("ai")
                    if not isinstance(ai, dict):
                        ai = {}
                        c["ai"] = ai
                    ai.update(
                        {
                            "chosen_variant_id": pk.get("variant_id"),
                            "reason": "; ".join(pk.get("reasons") or []) if isinstance(pk.get("reasons"), list) else str(pk.get("reasons") or ""),
                            "title": pk.get("title") or "",
                            "hook": pk.get("hook") or "",
                            "description": pk.get("description") or "",
                            "hashtags": pk.get("hashtags") or [],
                            "tags": pk.get("hashtags") or [],
                            "confidence": pk.get("confidence"),
                            "used_fallback": False,
                            "packaging_source": packaging_provenance["source"],
                            "packaging_provenance": dict(packaging_provenance),
                        }
                    )

        update_project(proj, _upd)

        resp = {
            "ok": True,
            "project_id": project_id,
            "director_path": str(director_path),
            "pick_count": len(picks),
            "missing": missing,
            "provenance": packaging_provenance,
        }
        idem.set(actor_key, client_request_id, resp)
        return JSONResponse(resp)

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

    @router.post("/export_director_picks", openapi_extra={"x-openai-isConsequential": True})
    def export_director_picks(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        """Export clips from analysis/director.json picks (ChatGPT-in-the-loop friendly)."""
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        project_id = _validate_project_id(str(body.get("project_id") or ""))
        limit = _clamp_int(body.get("limit"), default=5, min_v=1, max_v=30)
        llm_mode = _resolve_llm_mode(body.get("llm_mode"))

        proj, proj_data = _load_project(project_id)
        if not Path(proj.video_path).exists():
            raise HTTPException(status_code=409, detail="video_not_ready")

        director_path = proj.analysis_dir / "director.json"
        if not director_path.exists():
            raise HTTPException(status_code=404, detail="no_director_picks")

        director_data = json.loads(director_path.read_text(encoding="utf-8"))
        picks = [p for p in (director_data.get("picks") or []) if isinstance(p, dict)]
        if not picks:
            raise HTTPException(status_code=404, detail="no_picks")
        picks = picks[:limit]

        _require_external_ai_ready_for_export(
            project_id=project_id,
            proj=proj,
            proj_data=proj_data,
            llm_mode=llm_mode,
        )

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

        # Create selections for picks (persisted in project.json), but avoid duplicates.
        import uuid

        from ..project import update_project, utc_now_iso

        created_ids: list[str] = []
        export_ids: list[str] = []
        now = utc_now_iso()

        template_default = str(export_cfg.get("template", "vertical_blur"))

        def _upd(d: Dict[str, Any]) -> None:
            nonlocal created_ids, export_ids
            d.setdefault("selections", [])
            sels = d.get("selections")
            if not isinstance(sels, list):
                sels = []
                d["selections"] = sels

            key_to_id: Dict[Tuple[Any, Any], str] = {}
            time_to_id: Dict[Tuple[int, int], str] = {}

            def _norm_rank(v: Any) -> Optional[int]:
                if v is None:
                    return None
                try:
                    vv = int(v)
                except Exception:
                    return None
                return vv if vv > 0 else None

            def _norm_time_key(start_v: Any, end_v: Any) -> Optional[Tuple[int, int]]:
                try:
                    start = float(start_v)
                    end = float(end_v)
                except Exception:
                    return None
                if not (end > start):
                    return None
                return (int(round(start * 1000.0)), int(round(end * 1000.0)))

            for s in sels:
                if not isinstance(s, dict):
                    continue
                rank = _norm_rank(s.get("candidate_rank") or s.get("rank"))
                variant_id = str(s.get("variant_id") or s.get("best_variant_id") or s.get("chosen_variant_id") or "").strip()
                key = (rank, variant_id)
                sid = s.get("id")
                if sid and key not in key_to_id and key[0] is not None and key[1]:
                    key_to_id[key] = str(sid)
                tk = _norm_time_key(s.get("start_s"), s.get("end_s"))
                if sid and tk is not None and tk not in time_to_id:
                    time_to_id[tk] = str(sid)

            for p in picks:
                rank = _norm_rank(p.get("candidate_rank") or p.get("rank"))
                variant_id = str(p.get("variant_id") or p.get("best_variant_id") or p.get("chosen_variant_id") or "").strip()
                tk = _norm_time_key(p.get("start_s"), p.get("end_s"))
                if tk is None:
                    continue

                sid: Optional[str] = None
                if rank is not None and variant_id:
                    sid = key_to_id.get((rank, variant_id))
                if not sid:
                    sid = time_to_id.get(tk)
                if sid:
                    export_ids.append(sid)
                    continue

                sid = uuid.uuid4().hex
                if rank is not None and variant_id:
                    key_to_id[(rank, variant_id)] = sid
                time_to_id[tk] = sid
                created_ids.append(sid)
                export_ids.append(sid)

                sels.append(
                    {
                        "id": sid,
                        "created_at": now,
                        "start_s": float(p.get("start_s") or 0.0),
                        "end_s": float(p.get("end_s") or 0.0),
                        "title": str(p.get("title") or ""),
                        "notes": str(p.get("hook") or ""),
                        "template": str(p.get("template") or template_default),
                        "candidate_rank": rank,
                        "candidate_peak_time_s": p.get("peak_time_s"),
                        "variant_id": variant_id,
                        "director_confidence": p.get("confidence"),
                    }
                )

        update_project(proj, _upd)

        proj_data = get_project_data(proj)
        sel_by_id = {str(s.get("id")): s for s in (proj_data.get("selections") or []) if isinstance(s, dict)}
        selections = [sel_by_id[sid] for sid in export_ids if sid in sel_by_id]
        if not selections:
            raise HTTPException(status_code=404, detail="no_selections")

        director_results = _load_export_director_results(proj=proj, proj_data=proj_data)

        job = JOB_MANAGER.create("export_director_picks")
        JOB_MANAGER._set(job, message="queued", result={"project_id": project_id, "created_selection_ids": created_ids})

        @with_prevent_sleep("Exporting director picks (Actions)")
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
                        template=str(export_cfg.get("template", selection.get("template") or template_default)),
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
                    if sj and sj.status == "cancelled":
                        JOB_MANAGER._set(job, status="cancelled", message="cancelled", result={"project_id": project_id})
                        return
                    if sj and sj.status == "succeeded":
                        out = (sj.result or {}).get("output")
                        if out:
                            outputs.append(str(out))

                    JOB_MANAGER._set(job, progress=idx / total, message=f"exporting {idx}/{total}")

                manifest_path = proj.exports_dir / f"actions_export_manifest_{int(time.time())}.json"
                manifest = {
                    "project_id": project_id,
                    "created_selection_ids": created_ids,
                    "outputs": outputs,
                    "export": {
                        "template": str(export_cfg.get("template", template_default)),
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

        payload = {
            "job_id": job.id,
            "project_id": project_id,
            "created_selection_ids": created_ids,
            "llm_mode": llm_mode,
        }
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.post("/export_batch", openapi_extra={"x-openai-isConsequential": True})
    def export_batch(request: Request, body: Dict[str, Any] = Body(...)):  # type: ignore[valid-type]
        actor_key = _rate_limit(request)

        client_request_id = str(body.get("client_request_id") or "").strip()
        existing = idem.get(actor_key, client_request_id)
        if existing:
            return JSONResponse(existing)

        project_id = _validate_project_id(str(body.get("project_id") or ""))
        llm_mode = _resolve_llm_mode(body.get("llm_mode"))
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

        director_results = _load_export_director_results(proj=proj, proj_data=proj_data)
        if top_from_candidates is not None and llm_mode_is_strict_external(llm_mode):
            _require_director_backed_export_batch(
                project_id=project_id,
                proj=proj,
                proj_data=proj_data,
                llm_mode=llm_mode,
                top_from_candidates=top_from_candidates,
                selections=[],
                director_results=director_results,
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

        _require_director_backed_export_batch(
            project_id=project_id,
            proj=proj,
            proj_data=proj_data,
            llm_mode=llm_mode,
            top_from_candidates=None,
            selections=selections,
            director_results=director_results,
        )

        job = JOB_MANAGER.create("export_batch")
        JOB_MANAGER._set(
            job,
            message="queued",
            result={
                "project_id": project_id,
                "created_selection_ids": created_ids,
                "llm_mode": llm_mode,
            },
        )

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
                    if sj and sj.status == "cancelled":
                        JOB_MANAGER._set(job, status="cancelled", message="cancelled", result={"project_id": project_id, "created_selection_ids": created_ids})
                        return
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

        payload = {
            "job_id": job.id,
            "project_id": project_id,
            "created_selection_ids": created_ids,
            "llm_mode": llm_mode,
        }
        idem.set(actor_key, client_request_id, payload)
        return JSONResponse(payload)

    @router.get("/diagnostics", openapi_extra={"x-openai-isConsequential": False})
    def diagnostics(request: Request):
        _rate_limit(request)
        from ..chat.downloader import get_supported_platforms, get_twitch_downloader_info, is_chat_download_available

        backends = _diagnostics_transcription_backends()
        publish_accounts = _build_publish_accounts_payload()
        return JSONResponse(
            {
                "token_required": bool(os.environ.get("VP_API_TOKEN", "").strip()),
                "env": {
                    "vp_api_token": bool(os.environ.get("VP_API_TOKEN", "").strip()),
                    "vp_studio_profile": bool(os.environ.get("VP_STUDIO_PROFILE", "").strip()),
                    "assemblyai_api_key": _profile_env_present("ASSEMBLYAI_API_KEY", "AAI_API_KEY"),
                    "hf_token": _profile_env_present("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
                    "twitch_client_id": _profile_env_present("TWITCH_CLIENT_ID", "TWITCH_API_CLIENT_ID"),
                    "twitch_client_secret": _profile_env_present(
                        "TWITCH_CLIENT_SECRET",
                        "TWITCH_API_CLIENT_SECRET",
                    ),
                    "twitch_app_access_token": _profile_env_present(
                        "TWITCH_APP_ACCESS_TOKEN",
                        "TWITCH_ACCESS_TOKEN",
                    ),
                },
                "transcription": {"backends": backends},
                "chat": {
                    "available": bool(is_chat_download_available()),
                    "platforms": get_supported_platforms(),
                    "twitch_downloader": get_twitch_downloader_info(include_version=False),
                },
                "llm": {
                    "supported_modes": list(_LLM_MODE_ENUM),
                    "aliases": {"gondull": "external_strict"},
                    "preferred_mode": "external_strict",
                    "default_mode": _profile_default_llm_mode(),
                    "profile_external_ai_requirements": _profile_external_ai_requirements(),
                },
                "source_scout": _source_scout_diagnostics(),
                "publisher": publish_accounts,
                "profile": _profile_readiness(profile, profile_path=profile_path, backends=backends),
                "paths": {
                    "projects_root": str(default_projects_root()),
                    "profile_path": str(profile_path) if profile_path is not None else None,
                },
                "allowed_domains": sorted(allow_domains),
            }
        )

    return router
