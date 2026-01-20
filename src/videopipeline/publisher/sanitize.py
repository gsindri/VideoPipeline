from __future__ import annotations

import re
from typing import Any


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _normalize(text: str) -> str:
    text = _CONTROL_RE.sub("", text)
    return " ".join(text.strip().split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _sanitize_hashtags(tags: list[str]) -> list[str]:
    cleaned = []
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"
        cleaned.append(tag)
    return cleaned


def sanitize_metadata(platform: str, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = dict(metadata)
    title = _normalize(str(payload.get("title") or ""))
    caption = _normalize(str(payload.get("caption") or ""))
    description = _normalize(str(payload.get("description") or ""))
    hashtags = _sanitize_hashtags(list(payload.get("hashtags") or []))

    if platform == "youtube":
        payload["title"] = _truncate(title, 100)
        payload["description"] = _truncate(description or caption, 5000)
        payload["hashtags"] = hashtags
    elif platform == "tiktok":
        caption_base = description or caption or title
        joined_tags = " ".join(hashtags)
        full_caption = _normalize(" ".join([caption_base, joined_tags]).strip())
        payload["caption"] = _truncate(full_caption, 2200)
        payload["hashtags"] = hashtags
    else:
        payload["title"] = title
        payload["description"] = description
        payload["caption"] = caption
        payload["hashtags"] = hashtags

    return payload
