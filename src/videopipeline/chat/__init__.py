"""Chat replay integration for VideoPipeline.

Supports downloading, storing, and analyzing chat replay data
for both URL-ingested videos and local videos with source URLs.
"""

from .store import ChatStore, ChatMessage, ChatMeta
from .downloader import download_chat, ChatDownloadError
from .normalize import (
    normalize_chat_messages,
    normalize_chat_messages_with_timebase,
    detect_chat_format,
    load_chat_data,
)
from .features import compute_chat_features
from .emote_db import (
    GlobalEmoteDB,
    extract_channel_from_url,
    extract_channel_from_chat_store,
    get_channel_for_project,
    get_global_emote_db,
)

__all__ = [
    "ChatStore",
    "ChatMessage",
    "ChatMeta",
    "download_chat",
    "ChatDownloadError",
    "normalize_chat_messages",
    "normalize_chat_messages_with_timebase",
    "detect_chat_format",
    "load_chat_data",
    "compute_chat_features",
    "GlobalEmoteDB",
    "extract_channel_from_url",
    "extract_channel_from_chat_store",
    "get_channel_for_project",
    "get_global_emote_db",
]
