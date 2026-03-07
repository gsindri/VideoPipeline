from __future__ import annotations

import json
from typing import Any, Optional

_SERVICE_PREFIX = "VideoPipeline"


def _keyring():
    import keyring

    return keyring


def _key(platform: str, account_id: str) -> str:
    return f"{_SERVICE_PREFIX}:{platform}:{account_id}"


def store_tokens(platform: str, account_id: str, payload: dict[str, Any]) -> None:
    _keyring().set_password(_SERVICE_PREFIX, _key(platform, account_id), json.dumps(payload))


def load_tokens(platform: str, account_id: str) -> Optional[dict[str, Any]]:
    raw = _keyring().get_password(_SERVICE_PREFIX, _key(platform, account_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def delete_tokens(platform: str, account_id: str) -> None:
    _keyring().delete_password(_SERVICE_PREFIX, _key(platform, account_id))
