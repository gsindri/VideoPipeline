"""Chat↔video sync estimation.

Estimates a constant sync offset (ms) to align chat activity spikes with
audio-derived "moment intensity" over time.

The offset is stored in project.json under chat.sync_offset_ms and used by:
  - Chat message playback (studio chat panel)
  - Chat mini-timeline visualization
  - Highlights scoring (chat signal alignment)
  - Boundary graph (chat valley alignment)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import numpy as np

from .peaks import moving_average, robust_z
from .project import Project, get_chat_config, load_npz, save_json, update_project


def _as_f64_1d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _get_hop_seconds(npz: Dict[str, Any], *, fallback: float) -> float:
    hop_arr = npz.get("hop_seconds")
    if hop_arr is not None:
        try:
            hop_arr = _as_f64_1d(hop_arr)
            if len(hop_arr) > 0 and float(hop_arr[0]) > 0:
                return float(hop_arr[0])
        except Exception:
            pass
    return float(fallback)


def _get_times(npz: Dict[str, Any], *, hop_s: float, n: int) -> np.ndarray:
    t = npz.get("times")
    if t is not None:
        try:
            tt = _as_f64_1d(t)
            if len(tt) == n and len(tt) > 0:
                return tt
        except Exception:
            pass
    return np.arange(n, dtype=np.float64) * float(hop_s)


def _resample_to(times_target: np.ndarray, values: np.ndarray, *, times_src: np.ndarray) -> np.ndarray:
    if len(values) == 0 or len(times_src) == 0 or len(times_target) == 0:
        return np.zeros(len(times_target), dtype=np.float64)
    # Ensure increasing times for np.interp.
    if len(times_src) >= 2 and np.any(np.diff(times_src) < 0):
        order = np.argsort(times_src)
        times_src = times_src[order]
        values = values[order]
    return np.interp(times_target, times_src, values, left=0.0, right=0.0).astype(np.float64)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = _as_f64_1d(a)
    bb = _as_f64_1d(b)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-12
    if denom <= 0:
        return 0.0
    return float(np.dot(aa, bb) / denom)


@dataclass(frozen=True)
class ChatSyncConfig:
    enabled: bool = True
    max_offset_seconds: float = 20.0
    step_seconds: float = 0.25
    smooth_seconds: float = 1.0
    clip_z: float = 6.0
    min_score: float = 0.06
    min_best_z: float = 2.0
    min_nonzero_fraction: float = 0.01
    allow_override_manual: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChatSyncConfig":
        dd = dict(d or {})
        return cls(
            enabled=bool(dd.get("enabled", True)),
            max_offset_seconds=float(dd.get("max_offset_seconds", 20.0)),
            step_seconds=float(dd.get("step_seconds", 0.25)),
            smooth_seconds=float(dd.get("smooth_seconds", 1.0)),
            clip_z=float(dd.get("clip_z", 6.0)),
            min_score=float(dd.get("min_score", 0.06)),
            min_best_z=float(dd.get("min_best_z", 2.0)),
            min_nonzero_fraction=float(dd.get("min_nonzero_fraction", 0.01)),
            allow_override_manual=bool(dd.get("allow_override_manual", False)),
        )


def _pos_clip(z: np.ndarray, *, clip_z: float) -> np.ndarray:
    x = _as_f64_1d(z)
    x = np.clip(x, 0.0, None)
    if clip_z > 0:
        x = np.clip(x, 0.0, float(clip_z))
    return x


def estimate_chat_sync_offset_ms(
    *,
    audio_times_s: np.ndarray,
    intensity: np.ndarray,
    chat: np.ndarray,
    cfg: ChatSyncConfig,
) -> Dict[str, Any]:
    """Estimate offset_ms that best aligns chat with intensity on audio_times_s.

    Offset convention matches ChatStore + UI:
      video_time = chat_time + offset
    i.e. aligned_chat(t_video) = raw_chat(t_video - offset)
    """
    t = _as_f64_1d(audio_times_s)
    a = _pos_clip(intensity, clip_z=cfg.clip_z)
    b = _pos_clip(chat, clip_z=cfg.clip_z)

    if len(t) == 0 or len(a) != len(t) or len(b) != len(t):
        return {"ok": False, "reason": "bad_inputs"}

    # Quick sanity: skip if signals are essentially empty.
    nz_a = float(np.mean(a > 0.0)) if len(a) > 0 else 0.0
    nz_b = float(np.mean(b > 0.0)) if len(b) > 0 else 0.0
    if nz_a < cfg.min_nonzero_fraction:
        return {"ok": False, "reason": "low_intensity_energy", "nonzero_fraction": nz_a}
    if nz_b < cfg.min_nonzero_fraction:
        return {"ok": False, "reason": "low_chat_energy", "nonzero_fraction": nz_b}

    # Optional extra smoothing (keeps offset scan stable on noisy content).
    if cfg.smooth_seconds > 0:
        hop = float(np.median(np.diff(t))) if len(t) > 1 else 0.5
        win = max(1, int(round(float(cfg.smooth_seconds) / max(hop, 1e-6))))
        if win > 1:
            a = moving_average(a, win, use_cumsum=True)
            b = moving_average(b, win, use_cumsum=True)

    max_off = max(0.0, float(cfg.max_offset_seconds))
    step = max(0.05, float(cfg.step_seconds))
    if step > 2.0:
        step = 2.0

    offsets_s = np.arange(-max_off, max_off + 1e-9, step, dtype=np.float64)
    sims = np.zeros(len(offsets_s), dtype=np.float64)

    # Pre-normalize intensity once (cosine uses norms; keep as-is but stable).
    a_norm = float(np.linalg.norm(a)) + 1e-12
    if a_norm <= 0:
        return {"ok": False, "reason": "zero_intensity_norm"}

    for i, off in enumerate(offsets_s):
        # aligned_chat(t) = raw_chat(t - off)
        shifted = np.interp(t - float(off), t, b, left=0.0, right=0.0).astype(np.float64)
        b_norm = float(np.linalg.norm(shifted)) + 1e-12
        if b_norm <= 0:
            sims[i] = 0.0
            continue
        sims[i] = float(np.dot(a, shifted) / (a_norm * b_norm))

    best_i = int(np.argmax(sims)) if len(sims) > 0 else 0
    best_off_s = float(offsets_s[best_i]) if len(offsets_s) > 0 else 0.0
    best_score = float(sims[best_i]) if len(sims) > 0 else 0.0

    mean = float(np.mean(sims)) if len(sims) > 0 else 0.0
    std = float(np.std(sims)) if len(sims) > 0 else 0.0
    best_z = (best_score - mean) / (std + 1e-9)

    # Second best (excluding the best index).
    if len(sims) >= 2:
        tmp = sims.copy()
        tmp[best_i] = -np.inf
        second = float(np.max(tmp))
    else:
        second = float("-inf")

    # Avoid edge-clinging results (often indicates the true offset is outside the scan
    # OR the signal is too weak and the max is arbitrary).
    at_edge = (
        abs(best_off_s) >= max_off - (step * 0.51)
        if max_off > 0 and len(offsets_s) > 1
        else False
    )

    return {
        "ok": True,
        "offset_ms": int(round(best_off_s * 1000.0)),
        "best_score": best_score,
        "second_best_score": second,
        "best_z": float(best_z),
        "at_edge": bool(at_edge),
        "scan": {
            "max_offset_seconds": max_off,
            "step_seconds": step,
            "count": int(len(offsets_s)),
        },
        "signal_energy": {
            "intensity_nonzero_fraction": nz_a,
            "chat_nonzero_fraction": nz_b,
        },
    }


def compute_chat_sync_analysis(
    proj: Project,
    *,
    cfg: ChatSyncConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute chat sync offset and persist to analysis/chat_sync.json + project.json."""
    created_at = datetime.now(timezone.utc).isoformat()
    out_path = proj.analysis_dir / "chat_sync.json"

    def _report(frac: float) -> None:
        if on_progress:
            try:
                on_progress(frac)
            except TypeError:
                on_progress(frac)

    if not cfg.enabled:
        payload = {
            "created_at": created_at,
            "ok": False,
            "reason": "disabled",
            "config": cfg.__dict__,
        }
        save_json(out_path, payload)
        return payload

    _report(0.05)

    # Require audio features; chat is optional (we still write the artifact).
    if not proj.audio_features_path.exists():
        payload = {
            "created_at": created_at,
            "ok": False,
            "reason": "missing_audio_features",
            "config": cfg.__dict__,
        }
        save_json(out_path, payload)
        return payload

    audio_data = load_npz(proj.audio_features_path)
    audio_scores = audio_data.get("scores")
    if audio_scores is None:
        payload = {
            "created_at": created_at,
            "ok": False,
            "reason": "audio_missing_scores",
            "config": cfg.__dict__,
        }
        save_json(out_path, payload)
        return payload

    audio_scores = _as_f64_1d(audio_scores)
    hop_a = _get_hop_seconds(audio_data, fallback=0.5)
    t = _get_times(audio_data, hop_s=hop_a, n=len(audio_scores))

    _report(0.15)

    chat_used = False
    chat_raw = None
    hop_c = 0.5
    chat_times = None

    if proj.chat_features_path.exists():
        try:
            chat_data = load_npz(proj.chat_features_path)
            chat_raw = chat_data.get("scores_activity")
            if chat_raw is None:
                chat_raw = chat_data.get("scores")
            if chat_raw is not None:
                chat_raw = _as_f64_1d(chat_raw)
                hop_c = _get_hop_seconds(chat_data, fallback=hop_a)
                chat_times = np.arange(len(chat_raw), dtype=np.float64) * float(hop_c)
                chat_used = True
        except Exception:
            chat_used = False

    if not chat_used or chat_raw is None or chat_times is None or len(chat_raw) == 0:
        payload = {
            "created_at": created_at,
            "ok": False,
            "reason": "missing_chat_features",
            "config": cfg.__dict__,
            "audio": {"hop_seconds": hop_a, "frames": int(len(audio_scores))},
        }
        save_json(out_path, payload)
        return payload

    _report(0.25)

    # Build intensity = audio + optional reaction/events.
    intensity = _pos_clip(audio_scores, clip_z=cfg.clip_z)

    signals_used = {
        "audio": True,
        "reaction_audio": False,
        "audio_events": False,
    }

    if proj.reaction_audio_features_path.exists():
        try:
            r = load_npz(proj.reaction_audio_features_path)
            r_raw = r.get("reaction_score")
            if r_raw is None:
                r_raw = r.get("scores")
            if r_raw is not None:
                r_raw = _as_f64_1d(r_raw)
                hop_r = _get_hop_seconds(r, fallback=hop_a)
                tr = _get_times(r, hop_s=hop_r, n=len(r_raw))
                r_on_audio = _resample_to(t, r_raw, times_src=tr)
                intensity = intensity + 0.6 * _pos_clip(r_on_audio, clip_z=cfg.clip_z)
                signals_used["reaction_audio"] = True
        except Exception:
            pass

    if proj.audio_events_features_path.exists():
        try:
            e = load_npz(proj.audio_events_features_path)
            e_raw = e.get("event_combo_z")
            if e_raw is not None:
                e_raw = _as_f64_1d(e_raw)
                hop_e = _get_hop_seconds(e, fallback=hop_a)
                te = _get_times(e, hop_s=hop_e, n=len(e_raw))
                e_on_audio = _resample_to(t, e_raw, times_src=te)
                intensity = intensity + 0.4 * _pos_clip(e_on_audio, clip_z=cfg.clip_z)
                signals_used["audio_events"] = True
        except Exception:
            pass

    # Normalize intensity to be robust across streams.
    intensity = _pos_clip(robust_z(intensity), clip_z=cfg.clip_z)

    # Resample chat onto audio time grid (no offset yet).
    chat_on_audio = _resample_to(t, chat_raw, times_src=chat_times)
    chat_on_audio = _pos_clip(robust_z(chat_on_audio), clip_z=cfg.clip_z)

    _report(0.55)

    est = estimate_chat_sync_offset_ms(
        audio_times_s=t,
        intensity=intensity,
        chat=chat_on_audio,
        cfg=cfg,
    )

    _report(0.75)

    chat_cfg = get_chat_config(proj)
    prev_offset = int(chat_cfg.get("sync_offset_ms", 0) or 0)
    prev_source = str(chat_cfg.get("sync_offset_source", "") or "")

    applied = False
    apply_reason = "not_applied"

    if not est.get("ok"):
        apply_reason = str(est.get("reason") or "estimate_failed")
    else:
        best_score = float(est.get("best_score", 0.0) or 0.0)
        best_z = float(est.get("best_z", 0.0) or 0.0)
        at_edge = bool(est.get("at_edge", False))
        suggested = int(est.get("offset_ms", 0) or 0)

        if at_edge:
            apply_reason = "low_confidence_at_edge"
        elif best_score < cfg.min_score or best_z < cfg.min_best_z:
            apply_reason = "low_confidence"
        elif prev_source == "manual" and not cfg.allow_override_manual:
            apply_reason = "manual_offset_preserved"
        else:
            # Apply suggested offset.
            from .project import set_chat_config

            set_chat_config(
                proj,
                sync_offset_ms=suggested,
                sync_offset_source="auto",
                sync_offset_confidence=float(best_z),
                sync_offset_method="chat_audio_corr_v1",
                sync_offset_updated_at=created_at,
            )
            applied = True
            apply_reason = "applied"

    payload: Dict[str, Any] = {
        "created_at": created_at,
        "ok": bool(est.get("ok", False)),
        "method": "chat_audio_corr_v1",
        "config": cfg.__dict__,
        "estimated": est,
        "previous": {
            "sync_offset_ms": prev_offset,
            "sync_offset_source": prev_source,
        },
        "applied": applied,
        "apply_reason": apply_reason,
        "signals_used": signals_used,
        "audio": {"hop_seconds": hop_a, "frames": int(len(audio_scores))},
        "chat": {"hop_seconds": hop_c, "frames": int(len(chat_raw))},
    }

    save_json(out_path, payload)

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["chat_sync"] = {
            "created_at": created_at,
            "method": payload.get("method"),
            "ok": payload.get("ok", False),
            "applied": applied,
            "apply_reason": apply_reason,
            "estimated_offset_ms": int(est.get("offset_ms", 0) or 0) if est.get("ok") else None,
            "sync_json": str(out_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    _report(1.0)
    return payload


def load_chat_sync(proj: Project) -> Optional[Dict[str, Any]]:
    """Load cached chat sync estimate, if present."""
    path = proj.analysis_dir / "chat_sync.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

