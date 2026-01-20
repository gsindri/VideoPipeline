from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


@dataclass
class AccountPreset:
    title_template: str | None = None
    description_template: str | None = None
    default_hashtags: list[str] = field(default_factory=list)
    default_privacy: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AccountPreset":
        return cls(
            title_template=payload.get("title_template"),
            description_template=payload.get("description_template"),
            default_hashtags=list(payload.get("default_hashtags") or []),
            default_privacy=payload.get("default_privacy"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title_template": self.title_template,
            "description_template": self.description_template,
            "default_hashtags": self.default_hashtags,
            "default_privacy": self.default_privacy,
        }


def apply_presets(metadata: dict[str, Any], presets: AccountPreset) -> dict[str, Any]:
    payload = dict(metadata)
    context = _SafeDict(payload)
    context.update(payload.get("selection") or {})

    if presets.title_template:
        payload["title"] = presets.title_template.format_map(context).strip()
    if presets.description_template:
        payload["description"] = presets.description_template.format_map(context).strip()

    hashtags = list(payload.get("hashtags") or [])
    for tag in presets.default_hashtags:
        if tag not in hashtags:
            hashtags.append(tag)
    payload["hashtags"] = hashtags

    if presets.default_privacy and not payload.get("privacy"):
        payload["privacy"] = presets.default_privacy
    return payload
