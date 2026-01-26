from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .state import accounts_path
from ..utils import utc_iso as _utc_iso


@dataclass
class Account:
    id: str
    platform: str
    label: str
    created_at: str = field(default_factory=_utc_iso)
    presets: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "label": self.label,
            "created_at": self.created_at,
            "presets": self.presets,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Account":
        return cls(
            id=str(payload["id"]),
            platform=str(payload["platform"]),
            label=str(payload.get("label", "")),
            created_at=str(payload.get("created_at") or _utc_iso()),
            presets=dict(payload.get("presets") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


class AccountStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or accounts_path()

    def _load_raw(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []

    def _save_raw(self, payload: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list(self) -> list[Account]:
        return [Account.from_dict(item) for item in self._load_raw()]

    def get(self, account_id: str) -> Optional[Account]:
        for account in self.list():
            if account.id == account_id:
                return account
        return None

    def add(
        self,
        *,
        platform: str,
        label: str,
        presets: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Account:
        account = Account(
            id=uuid.uuid4().hex,
            platform=platform,
            label=label,
            presets=presets or {},
            metadata=metadata or {},
        )
        items = self._load_raw()
        items.append(account.to_dict())
        self._save_raw(items)
        return account

    def update(self, account: Account) -> None:
        items = self._load_raw()
        updated = []
        for item in items:
            if item.get("id") == account.id:
                updated.append(account.to_dict())
            else:
                updated.append(item)
        self._save_raw(updated)

    def remove(self, account_id: str) -> bool:
        items = self._load_raw()
        kept = [item for item in items if item.get("id") != account_id]
        if len(kept) == len(items):
            return False
        self._save_raw(kept)
        return True

    def iter_platform(self, platform: str) -> Iterable[Account]:
        return (acct for acct in self.list() if acct.platform == platform)
