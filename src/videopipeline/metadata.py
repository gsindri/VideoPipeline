from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .subtitles import SubtitleSegment


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _pick_hook_from_segments(segments: Iterable[SubtitleSegment]) -> Optional[str]:
    best = ""
    for seg in segments:
        text = _clean_text(seg.text)
        if len(text.split()) >= 4 and len(text) > len(best):
            best = text
        if len(best) >= 60:
            break
    if not best:
        return None
    return best[:100]


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def derive_hook_text(selection: Dict[str, Any], segments: Optional[Iterable[SubtitleSegment]] = None) -> str:
    if segments:
        hook = _pick_hook_from_segments(segments)
        if hook:
            return hook
    rank = selection.get("candidate_rank")
    if rank:
        return f"Highlight #{rank}"
    start_s = float(selection.get("start_s", 0.0))
    return f"Clip @ {_fmt_time(start_s)}"


def build_metadata(
    *,
    selection: Dict[str, Any],
    output_path: Path,
    template: str,
    with_captions: bool,
    segments: Optional[Iterable[SubtitleSegment]] = None,
    ai_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build export metadata dictionary.
    
    Args:
        selection: Candidate/selection data dict
        output_path: Path to the exported video file
        template: Template name used for export
        with_captions: Whether captions were included
        segments: Optional transcript segments for hook derivation
        ai_metadata: Optional AI director result with title, description, tags, hook
    """
    # Priority: AI metadata > selection fields > derived
    hook = None
    title = None
    caption = None
    hashtags = ["#gaming", "#clips", "#shorts"]
    
    if ai_metadata:
        hook = ai_metadata.get("hook")
        title = ai_metadata.get("title")
        caption = ai_metadata.get("description")
        ai_tags = ai_metadata.get("tags") or []
        if ai_tags:
            hashtags = [f"#{t}" if not t.startswith("#") else t for t in ai_tags]
    
    if not hook:
        hook = derive_hook_text(selection, segments)
    if not title:
        title = selection.get("title") or hook
    if not caption:
        caption = hook
        
    if template.startswith("vertical"):
        if "#vertical" not in hashtags:
            hashtags.append("#vertical")

    payload = {
        "title": title,
        "caption": caption,
        "hashtags": hashtags,
        "template": template,
        "with_captions": with_captions,
        "selection": {
            "id": selection.get("id"),
            "start_s": selection.get("start_s"),
            "end_s": selection.get("end_s"),
            "candidate_rank": selection.get("candidate_rank"),
            "candidate_score": selection.get("candidate_score"),
            "candidate_peak_time_s": selection.get("candidate_peak_time_s"),
        },
        "platform_hints": {
            "shorts_max_seconds": 60,
            "tiktok_max_seconds": 60,
            "safe_zone_top_px": 120,
            "safe_zone_bottom_px": 260,
        },
        "output": str(output_path),
    }
    if segments:
        payload["transcript_snippet"] = _pick_hook_from_segments(segments)
    if ai_metadata:
        payload["ai_generated"] = {
            "title": ai_metadata.get("title"),
            "description": ai_metadata.get("description"),
            "tags": ai_metadata.get("tags"),
            "hook": ai_metadata.get("hook"),
        }
    return payload


def write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
