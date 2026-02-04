"""AI Director: pick best clip variants and generate packaging metadata.

This step turns *variants* into a publish-ready plan:
- chooses the best variant per highlight candidate (duration vs. hook vs. context)
- enforces global spacing (avoid near-duplicate moments)
- generates titles/hooks/descriptions/hashtags via an optional LLM
- persists analysis/director.json and updates project.json -> analysis.director

Inputs (required):
  - analysis/variants.json (or legacy analysis/clip_variants.json)

Inputs (optional):
  - analysis/highlights.json (or project.json analysis.highlights.candidates)
  - analysis/chapters.json (for diversity hints)

The LLM interface is intentionally tiny:
  llm_complete(prompt: str) -> str
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .project import Project, get_project_data, save_json, update_project

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _first_nonempty(*vals: Any, default: str = "") -> str:
    for v in vals:
        s = str(v or "").strip()
        if s:
            return s
    return default


def _trim(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    # Trim at a word boundary if possible
    cut = s[: max_chars - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return (cut + "…").strip()


def _clean_hashtag(tag: str) -> str:
    tag = (tag or "").strip()
    if not tag:
        return ""
    # Remove spaces and punctuation-ish separators
    tag = re.sub(r"[^0-9A-Za-z_]+", "", tag)
    if not tag:
        return ""
    if not tag.startswith("#"):
        tag = "#" + tag
    return tag


def _normalize_hashtags(tags: Any, *, min_n: int = 3, max_n: int = 8) -> List[str]:
    out: List[str] = []
    if isinstance(tags, str):
        # allow "#a #b #c" or "a,b,c"
        parts = re.split(r"[\s,]+", tags)
    elif isinstance(tags, list):
        parts = tags
    else:
        parts = []

    seen = set()
    for p in parts:
        t = _clean_hashtag(str(p))
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_n:
            break

    # Pad minimally with generic tags if needed
    defaults = ["#clips", "#gaming", "#stream"]
    for d in defaults:
        if len(out) >= min_n:
            break
        if d.lower() not in seen:
            out.append(d)
            seen.add(d.lower())

    return out[:max_n]


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first top-level JSON object from arbitrary text."""
    if not text:
        return None

    # Common case: model returns pure JSON
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Otherwise, try to find the first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


@dataclass(frozen=True)
class DirectorConfig:
    enabled: bool = True

    # How many clips to pick from the candidate list
    top_n: int = 10

    # Global spacing to avoid near-duplicate moments
    min_gap_s: float = 12.0
    max_overlap_ratio: float = 0.25
    max_overlap_s: float = 4.0

    # Packaging targets
    platform: str = "shorts"  # shorts|tiktok|reels
    language: Optional[str] = None

    # Variant selection preference
    target_duration_s: float = 24.0

    # LLM behavior
    use_llm: bool = True
    fallback_to_rules: bool = True

    # Copy constraints
    title_max_chars: int = 60
    hook_max_chars: int = 80
    description_max_chars: int = 220

    # Templates
    default_template: str = "vertical_streamer_pip"
    allowed_templates: List[str] = field(default_factory=lambda: [
        "vertical_streamer_pip",
        "vertical_blur",
        "original",
    ])

    # Hashtags
    hashtags: List[str] = field(default_factory=lambda: ["#gaming", "#clips", "#stream"])

    # Optional side-effect: write picks into project.json selections
    write_selections: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DirectorConfig":
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", True)),
            top_n=int(d.get("top_n", 10)),
            min_gap_s=float(d.get("min_gap_s", 12.0)),
            max_overlap_ratio=float(d.get("max_overlap_ratio", 0.25)),
            max_overlap_s=float(d.get("max_overlap_s", 4.0)),
            platform=str(d.get("platform", "shorts")),
            language=d.get("language"),
            target_duration_s=float(d.get("target_duration_s", d.get("target_duration", 24.0))),
            use_llm=bool(d.get("use_llm", True)),
            fallback_to_rules=bool(d.get("fallback_to_rules", True)),
            title_max_chars=int(d.get("title_max_chars", 60)),
            hook_max_chars=int(d.get("hook_max_chars", 80)),
            description_max_chars=int(d.get("description_max_chars", 220)),
            default_template=str(d.get("default_template", d.get("template", "vertical_streamer_pip"))),
            allowed_templates=list(d.get("allowed_templates", ["vertical_streamer_pip", "vertical_blur", "original"])),
            hashtags=list(d.get("hashtags", ["#gaming", "#clips", "#stream"])),
            write_selections=bool(d.get("write_selections", False)),
        )


def _load_variants_payload(proj: Project) -> Dict[str, Any]:
    for fname in ("variants.json", "clip_variants.json"):
        p = proj.analysis_dir / fname
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("No variants.json (or legacy clip_variants.json) found")


def _load_candidates_from_project(proj: Project) -> List[Dict[str, Any]]:
    try:
        d = get_project_data(proj)
        cand = d.get("analysis", {}).get("highlights", {}).get("candidates", [])
        if isinstance(cand, list) and cand:
            return cand
    except Exception:
        pass

    # Fallback to highlights.json if project.json isn't populated
    hpath = proj.analysis_dir / "highlights.json"
    if hpath.exists():
        try:
            data = json.loads(hpath.read_text(encoding="utf-8"))
            cand = data.get("candidates", [])
            if isinstance(cand, list):
                return cand
        except Exception:
            pass

    return []


def _chapter_index_for_time(chapters: List[Tuple[float, float]], t: float) -> Optional[int]:
    for i, (a, b) in enumerate(chapters):
        if a <= t < b:
            return i
    if chapters and t >= chapters[-1][0]:
        return len(chapters) - 1
    return None


def _load_chapters(proj: Project) -> List[Tuple[float, float]]:
    p = proj.analysis_dir / "chapters.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ch = []
        for c in data.get("chapters", []):
            ch.append((float(c.get("start_s", 0.0)), float(c.get("end_s", 0.0))))
        return ch
    except Exception:
        return []


def _variant_score(
    *,
    variant: Dict[str, Any],
    candidate: Dict[str, Any],
    cfg: DirectorConfig,
) -> Tuple[float, List[str]]:
    """Score a variant *within a candidate* for packaging suitability.

    Returns: (score, reasons)
    """
    vid = str(variant.get("variant_id", ""))
    dur = _safe_float(variant.get("duration_s", 0.0), 0.0)

    meta = candidate.get("meta", {}) if isinstance(candidate.get("meta"), dict) else {}
    summary = variant.get("scores_summary", {}) if isinstance(variant.get("scores_summary"), dict) else {}

    # Pull a few key signals (z-ish, usually)
    rx = max(_safe_float(meta.get("reaction_audio_z"), 0.0), _safe_float(summary.get("reaction_audio_at_peak"), 0.0))
    turn = max(_safe_float(meta.get("turn_rate_z"), 0.0), _safe_float(summary.get("turn_rate_at_peak"), 0.0))
    chat = max(_safe_float(summary.get("chat_at_peak"), 0.0), _safe_float(summary.get("chat_max"), 0.0))
    overlap = max(_safe_float(meta.get("overlap_z"), 0.0), _safe_float(summary.get("overlap_at_peak"), 0.0))

    reasons: List[str] = []

    # Duration preference (smooth penalty)
    target = float(cfg.target_duration_s)
    if target <= 0:
        target = 24.0
    # normalize: 0 at perfect, grows to ~1 around +-target/2
    dur_pen = abs(dur - target) / max(target * 0.6, 1e-6)
    dur_score = 1.0 - _clamp(dur_pen, 0.0, 1.3)

    score = dur_score

    # Variant-type priors
    if vid in {"short", "punchline_first"}:
        score += 0.15
        reasons.append("fast hook")
    if vid in {"medium"}:
        score += 0.05
        reasons.append("balanced context")
    if vid in {"long"}:
        score -= 0.05
        reasons.append("longer story")
    if vid in {"setup_first"}:
        score += 0.06
        reasons.append("more setup")
    if vid in {"chat_centered"} and chat >= 1.0:
        score += 0.10
        reasons.append("chat arc")
    if vid in {"reaction_arc"} and rx >= 1.0:
        score += 0.14
        reasons.append("voice excitement arc")
    if vid in {"clean_cut"}:
        score += 0.06
        reasons.append("clean cut")

    # Signal-conditioned nudges
    if rx >= 1.5 and vid in {"reaction_arc", "punchline_first", "short"}:
        score += 0.08
        reasons.append("high excitement")
    if turn >= 1.5 and vid in {"medium", "long", "setup_first"}:
        score += 0.07
        reasons.append("rapid back-and-forth")
    if overlap >= 1.5:
        # Overlap is often fun, but harder for captions; avoid super-short hooks.
        if vid in {"punchline_first", "short"}:
            score -= 0.05
            reasons.append("overlap: give breathing room")

    # Clamp for sanity
    score = float(_clamp(score, -1.0, 2.0))
    return score, reasons


def _pick_variant_for_candidate(
    *,
    candidate: Dict[str, Any],
    variants: List[Dict[str, Any]],
    cfg: DirectorConfig,
) -> Tuple[Dict[str, Any], List[str]]:
    best: Optional[Dict[str, Any]] = None
    best_score = -1e9
    best_reasons: List[str] = []

    for v in variants:
        s, reasons = _variant_score(variant=v, candidate=candidate, cfg=cfg)
        if s > best_score:
            best = v
            best_score = s
            best_reasons = reasons

    if best is None:
        # Shouldn't happen, but be defensive
        best = variants[0]
        best_reasons = ["fallback"]

    # Always include the numeric score for debugging
    best_reasons = best_reasons + [f"variant_score={best_score:.2f}"]
    return best, best_reasons


def _choose_template(proj: Project, cfg: DirectorConfig) -> str:
    # If user already set a layout/facecam, PIP templates usually win.
    try:
        d = get_project_data(proj)
        facecam = d.get("layout", {}).get("facecam")
        if isinstance(facecam, dict) and facecam:
            if "vertical_streamer_pip" in cfg.allowed_templates:
                return "vertical_streamer_pip"
    except Exception:
        pass

    # Default fallback
    if cfg.default_template in cfg.allowed_templates:
        return cfg.default_template
    return cfg.allowed_templates[0] if cfg.allowed_templates else cfg.default_template


def _packaging_from_rules(
    *,
    candidate: Dict[str, Any],
    variant: Dict[str, Any],
    cfg: DirectorConfig,
    template: str,
) -> Dict[str, Any]:
    setup = str(variant.get("setup_text", "")).strip()
    payoff = str(variant.get("payoff_text", "")).strip()
    transcript = str(candidate.get("transcript", "")).strip()

    # Title: prefer payoff, else first sentence of transcript
    title_raw = _first_nonempty(payoff, setup, transcript, default="Highlight")
    title = _trim(title_raw.replace("\n", " "), cfg.title_max_chars)

    # Hook: short punchy quote
    hook_raw = _first_nonempty(payoff, transcript, title, default=title)
    hook = _trim(hook_raw.replace("\n", " "), cfg.hook_max_chars)

    # Description: minimal + honest
    desc = f"{title}"
    desc = _trim(desc, cfg.description_max_chars)

    tags = _normalize_hashtags(cfg.hashtags)

    return {
        "title": title,
        "hook": hook,
        "description": desc,
        "hashtags": tags,
        "template": template,
    }


def _packaging_from_llm(
    *,
    llm_complete: Callable[[str], str],
    candidate: Dict[str, Any],
    variant: Dict[str, Any],
    cfg: DirectorConfig,
    template: str,
) -> Optional[Dict[str, Any]]:
    """Ask the LLM for packaging. Returns dict or None if parse failed."""
    setup = str(variant.get("setup_text", "")).strip()
    payoff = str(variant.get("payoff_text", "")).strip()
    transcript = str(candidate.get("transcript", "")).strip()

    meta = candidate.get("meta", {}) if isinstance(candidate.get("meta"), dict) else {}
    # Keep only a small set of hints
    hints = {
        "reaction_audio_z": _safe_float(meta.get("reaction_audio_z"), 0.0),
        "turn_rate_z": _safe_float(meta.get("turn_rate_z"), 0.0),
        "overlap_z": _safe_float(meta.get("overlap_z"), 0.0),
        "audio_events_z": _safe_float(meta.get("audio_events_z"), 0.0),
    }

    prompt = f"""
You are an expert social media clip editor.

Create packaging copy for a {cfg.platform} short.

Return ONLY strict JSON (no markdown, no extra text) with:
{{
  \"title\": string,
  \"hook\": string,
  \"description\": string,
  \"hashtags\": [string, ...],
  \"template\": string
}}

Constraints:
- title <= {cfg.title_max_chars} chars
- hook <= {cfg.hook_max_chars} chars
- description <= {cfg.description_max_chars} chars
- hashtags: 3-8 items, use #prefix, no spaces.
- Do not invent facts or names not present in the transcript.
- Keep it punchy and native to {cfg.platform}.

Clip context:
- duration_s: {float(variant.get('duration_s', 0.0)):.1f}
- variant_id: {variant.get('variant_id')}
- suggested_template: {template}
- setup_text: {setup[:400]}
- payoff_text: {payoff[:400]}
- transcript_snippet: {transcript[:600]}
- signal_hints: {json.dumps(hints)}
""".strip()

    raw = llm_complete(prompt)
    data = _extract_first_json_object(raw)
    if not isinstance(data, dict):
        return None

    title = _trim(str(data.get("title", "")).strip(), cfg.title_max_chars)
    hook = _trim(str(data.get("hook", "")).strip(), cfg.hook_max_chars)
    description = _trim(str(data.get("description", "")).strip(), cfg.description_max_chars)

    hashtags = _normalize_hashtags(data.get("hashtags"), min_n=3, max_n=8)

    tpl = str(data.get("template", template)).strip() or template
    if tpl not in cfg.allowed_templates:
        tpl = template

    if not title:
        return None

    return {
        "title": title,
        "hook": hook,
        "description": description,
        "hashtags": hashtags,
        "template": tpl,
    }


def _confidence(candidate: Dict[str, Any], *, used_llm: bool) -> float:
    """Heuristic confidence 0..1 (used for UI sorting)."""
    base = 0.55 if used_llm else 0.40

    meta = candidate.get("meta", {}) if isinstance(candidate.get("meta"), dict) else {}
    rx = _safe_float(meta.get("reaction_audio_z"), 0.0)
    tr = _safe_float(meta.get("turn_rate_z"), 0.0)
    ae = _safe_float(meta.get("audio_events_z"), 0.0)

    bump = 0.0
    bump += 0.06 if rx >= 1.0 else 0.0
    bump += 0.04 if tr >= 1.0 else 0.0
    bump += 0.05 if ae >= 1.0 else 0.0

    return float(_clamp(base + bump, 0.15, 0.95))


def compute_director(
    proj: Project,
    *,
    cfg: Dict[str, Any],
    llm_complete: Optional[Callable[[str], str]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute AI director picks and packaging.

    Persists:
      - analysis/director.json
      - project.json -> analysis.director

    Returns the director payload.
    """
    dcfg = DirectorConfig.from_dict(cfg or {})
    if not dcfg.enabled:
        return {"created_at": _utc_now_iso(), "enabled": False, "picks": []}

    if dcfg.use_llm and llm_complete is None:
        logger.warning("[director] use_llm=true but no llm_complete provided; packaging will use rules.")

    if on_progress:
        on_progress(0.05)

    variants_payload = _load_variants_payload(proj)
    variant_candidates = variants_payload.get("candidates", [])
    if not isinstance(variant_candidates, list) or not variant_candidates:
        raise ValueError("variants.json has no candidates")

    # Candidate metadata (scores, transcript snippets, etc.)
    candidates = _load_candidates_from_project(proj)
    cand_by_rank: Dict[int, Dict[str, Any]] = {}
    for c in candidates:
        try:
            cand_by_rank[int(c.get("rank"))] = c
        except Exception:
            continue

    # Chapters for diversity hints
    chapters = _load_chapters(proj)

    # Select top picks with spacing
    picks: List[Dict[str, Any]] = []
    used_intervals: List[Tuple[float, float]] = []
    used_chapters: Dict[int, int] = {}
    packaging_counts = {"llm": 0, "rules": 0}

    # Sort variant candidates by underlying highlight score if available
    def _cand_score(vc: Dict[str, Any]) -> float:
        r = int(vc.get("candidate_rank", -1))
        c = cand_by_rank.get(r)
        return _safe_float((c or {}).get("score"), 0.0)

    variant_candidates_sorted = sorted(variant_candidates, key=_cand_score, reverse=True)

    if on_progress:
        on_progress(0.15)

    template_default = _choose_template(proj, dcfg)

    for vc in variant_candidates_sorted:
        if len(picks) >= int(dcfg.top_n):
            break

        rank = int(vc.get("candidate_rank", -1))
        cand = cand_by_rank.get(rank) or {
            "rank": rank,
            "peak_time_s": vc.get("candidate_peak_time_s"),
            "transcript": "",
            "meta": {},
            "score": 0.0,
        }

        variants = vc.get("variants", [])
        if not isinstance(variants, list) or not variants:
            continue

        chosen, reasons = _pick_variant_for_candidate(candidate=cand, variants=variants, cfg=dcfg)

        start_s = _safe_float(chosen.get("start_s"), 0.0)
        end_s = _safe_float(chosen.get("end_s"), start_s)
        if end_s <= start_s:
            continue

        # Global spacing / overlap guard
        peak_t = _safe_float(cand.get("peak_time_s"), _safe_float(vc.get("candidate_peak_time_s"), start_s))
        too_close = False
        for (a0, a1) in used_intervals:
            ov = _overlap_s(a0, a1, start_s, end_s)
            if ov <= 0:
                continue
            frac = ov / max(end_s - start_s, 1e-6)
            if ov >= float(dcfg.max_overlap_s) or frac >= float(dcfg.max_overlap_ratio):
                too_close = True
                break

        if not too_close:
            # Also enforce min gap between peaks
            for (a0, a1) in used_intervals:
                if abs((a0 + a1) / 2.0 - peak_t) < float(dcfg.min_gap_s):
                    too_close = True
                    break

        if too_close:
            continue

        # Diversity: soft cap 2 picks per chapter
        ch_idx = _chapter_index_for_time(chapters, peak_t) if chapters else None
        if ch_idx is not None:
            used_chapters.setdefault(ch_idx, 0)
            if used_chapters[ch_idx] >= 2:
                continue

        # Packaging
        used_llm = False
        pack = None
        packaging_source = "rules"
        packaging_error: Optional[str] = None
        if dcfg.use_llm and llm_complete is not None:
            try:
                pack = _packaging_from_llm(
                    llm_complete=llm_complete,
                    candidate=cand,
                    variant=chosen,
                    cfg=dcfg,
                    template=template_default,
                )
            except Exception as exc:
                packaging_error = f"{type(exc).__name__}: {exc}"
                logger.warning("[director] LLM packaging failed for candidate_rank=%s: %s", rank, packaging_error)
                pack = None
            used_llm = pack is not None
            packaging_source = "llm" if used_llm else "rules"

        if pack is None:
            if not dcfg.fallback_to_rules:
                raise RuntimeError("LLM packaging failed and fallback_to_rules=false")
            pack = _packaging_from_rules(candidate=cand, variant=chosen, cfg=dcfg, template=template_default)
            packaging_source = "rules"

        packaging_counts[packaging_source] = packaging_counts.get(packaging_source, 0) + 1

        # Merge + finalize
        pick = {
            "rank": len(picks) + 1,
            "candidate_rank": rank,
            "peak_time_s": _safe_float(cand.get("peak_time_s"), _safe_float(vc.get("candidate_peak_time_s"), 0.0)),
            "variant_id": str(chosen.get("variant_id", "")),
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "duration_s": round(end_s - start_s, 2),
            "title": pack.get("title", ""),
            "hook": pack.get("hook", ""),
            "description": pack.get("description", ""),
            "hashtags": pack.get("hashtags", []),
            "template": pack.get("template", template_default),
            "packaging_source": packaging_source,
            "packaging_error": packaging_error,
            "confidence": round(_confidence(cand, used_llm=used_llm), 3),
            "reasons": reasons,
            "chapter_index": ch_idx,
            "signals": cand.get("meta", {}),
        }

        picks.append(pick)
        used_intervals.append((start_s, end_s))
        if ch_idx is not None:
            used_chapters[ch_idx] = used_chapters.get(ch_idx, 0) + 1

        if on_progress:
            on_progress(0.15 + 0.75 * (len(picks) / max(1, int(dcfg.top_n))))

    payload = {
        "created_at": _utc_now_iso(),
        "config": {
            "top_n": dcfg.top_n,
            "min_gap_s": dcfg.min_gap_s,
            "max_overlap_ratio": dcfg.max_overlap_ratio,
            "max_overlap_s": dcfg.max_overlap_s,
            "platform": dcfg.platform,
            "language": dcfg.language,
            "target_duration_s": dcfg.target_duration_s,
            "use_llm": dcfg.use_llm and llm_complete is not None,
            "fallback_to_rules": dcfg.fallback_to_rules,
            "default_template": template_default,
            "allowed_templates": dcfg.allowed_templates,
        },
        "pick_count": len(picks),
        "packaging_counts": packaging_counts,
        "picks": picks,
    }

    out_path = proj.analysis_dir / "director.json"
    save_json(out_path, payload)

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["director"] = {
            "created_at": payload["created_at"],
            "pick_count": len(picks),
            "director_json": str(out_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    # Optional: write selections
    if dcfg.write_selections and picks:
        import uuid

        def _upd_sel(d: Dict[str, Any]) -> None:
            d.setdefault("selections", [])
            existing = d.get("selections", [])

            # Avoid duplicates: (candidate_rank, variant_id)
            seen = set()
            for s in existing:
                key = (s.get("candidate_rank"), s.get("variant_id"))
                seen.add(key)

            for p in picks:
                key = (p.get("candidate_rank"), p.get("variant_id"))
                if key in seen:
                    continue
                sel_id = uuid.uuid4().hex
                d["selections"].append(
                    {
                        "id": sel_id,
                        "created_at": payload["created_at"],
                        "start_s": float(p["start_s"]),
                        "end_s": float(p["end_s"]),
                        "title": p.get("title", ""),
                        "notes": p.get("hook", ""),
                        "template": p.get("template", template_default),
                        "candidate_rank": p.get("candidate_rank"),
                        "candidate_peak_time_s": p.get("peak_time_s"),
                        "variant_id": p.get("variant_id"),
                        "director_confidence": p.get("confidence"),
                    }
                )
                seen.add(key)

        update_project(proj, _upd_sel)

    if on_progress:
        on_progress(1.0)

    return payload
