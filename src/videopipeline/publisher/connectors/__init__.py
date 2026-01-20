from __future__ import annotations

from typing import Any

from ..accounts import Account
from .base import Connector
from .tiktok import TikTokConnector
from .youtube import YouTubeConnector


def get_connector(platform: str, *, account: Account, tokens: dict[str, Any]) -> Connector:
    if platform == "youtube":
        return YouTubeConnector(account=account, tokens=tokens)
    if platform == "tiktok":
        return TikTokConnector(account=account, tokens=tokens)
    raise ValueError(f"unsupported_platform: {platform}")


__all__ = ["Connector", "get_connector", "TikTokConnector", "YouTubeConnector"]
