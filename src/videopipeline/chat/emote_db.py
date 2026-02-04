"""Global emote database for cross-project emote learning persistence.

Stores LLM-learned emotes per channel so they can be reused across videos.
This means the LLM only needs to learn a channel's custom emotes once.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from ..project import Project

from ..utils import utc_iso as _utc_iso

# Default location for global emote database
_DEFAULT_DB_NAME = "learned_emotes.sqlite"


def _get_default_db_path() -> Path:
    """Get the default path for the global emote database."""
    # Store in outputs/ folder (same level as projects/)
    # This keeps it with project data but not inside any specific project
    from ..project import get_outputs_dir
    return get_outputs_dir() / _DEFAULT_DB_NAME


class GlobalEmoteDB:
    """SQLite database for storing learned emotes across projects.
    
    Schema:
        channels: channel_id, channel_name, platform, first_seen, last_updated
        emotes: channel_id, token, source (llm/seed/manual), learned_at, frequency
        
    Usage:
        db = GlobalEmoteDB()
        
        # Load known emotes for a channel
        emotes = db.get_channel_emotes("shroud")
        
        # Save newly learned emotes
        db.save_channel_emotes("shroud", {"KEKW", "OMEGALUL"}, source="llm")
        
        # Merge (add new, keep existing)
        db.merge_channel_emotes("shroud", new_emotes, source="llm")
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _get_default_db_path()
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id TEXT PRIMARY KEY,
                    channel_name TEXT,
                    platform TEXT DEFAULT 'twitch',
                    first_seen TEXT,
                    last_updated TEXT,
                    video_count INTEGER DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS emotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    source TEXT DEFAULT 'llm',
                    learned_at TEXT,
                    frequency INTEGER DEFAULT 1,
                    UNIQUE(channel_id, token),
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_emotes_channel ON emotes(channel_id);
                CREATE INDEX IF NOT EXISTS idx_emotes_token ON emotes(token);
            """)
            conn.commit()
        finally:
            conn.close()
    
    def get_channel_emotes(self, channel_id: str) -> Set[str]:
        """Get all known emotes for a channel."""
        channel_id = channel_id.lower().strip()
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "SELECT token FROM emotes WHERE channel_id = ?",
                (channel_id,)
            )
            return {row[0] for row in cur.fetchall()}
        finally:
            conn.close()
    
    def get_channel_llm_emotes(self, channel_id: str) -> Set[str]:
        """Get only LLM-learned emotes for a channel (excludes seeds)."""
        channel_id = channel_id.lower().strip()
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "SELECT token FROM emotes WHERE channel_id = ? AND source = 'llm'",
                (channel_id,)
            )
            return {row[0] for row in cur.fetchall()}
        finally:
            conn.close()
    
    def get_channel_info(self, channel_id: str) -> Optional[Dict]:
        """Get channel metadata."""
        channel_id = channel_id.lower().strip()
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "SELECT channel_id, channel_name, platform, first_seen, last_updated, video_count "
                "FROM channels WHERE channel_id = ?",
                (channel_id,)
            )
            row = cur.fetchone()
            if row:
                return {
                    "channel_id": row[0],
                    "channel_name": row[1],
                    "platform": row[2],
                    "first_seen": row[3],
                    "last_updated": row[4],
                    "video_count": row[5],
                }
            return None
        finally:
            conn.close()
    
    def save_channel_emotes(
        self,
        channel_id: str,
        emotes: Set[str],
        *,
        source: str = "llm",
        channel_name: Optional[str] = None,
        platform: str = "twitch",
    ) -> int:
        """Save emotes for a channel (replaces existing).
        
        Returns number of emotes saved.
        """
        channel_id = channel_id.lower().strip()
        now = _utc_iso()
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Upsert channel
            conn.execute("""
                INSERT INTO channels (channel_id, channel_name, platform, first_seen, last_updated, video_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(channel_id) DO UPDATE SET
                    channel_name = COALESCE(excluded.channel_name, channel_name),
                    last_updated = excluded.last_updated,
                    video_count = video_count + 1
            """, (channel_id, channel_name or channel_id, platform, now, now))
            
            # Clear existing emotes for this source
            conn.execute(
                "DELETE FROM emotes WHERE channel_id = ? AND source = ?",
                (channel_id, source)
            )
            
            # Insert new emotes
            for token in emotes:
                token = token.lower().strip()
                if token:
                    conn.execute("""
                        INSERT OR REPLACE INTO emotes (channel_id, token, source, learned_at, frequency)
                        VALUES (?, ?, ?, ?, 1)
                    """, (channel_id, token, source, now))
            
            conn.commit()
            return len(emotes)
        finally:
            conn.close()
    
    def merge_channel_emotes(
        self,
        channel_id: str,
        new_emotes: Set[str],
        *,
        source: str = "llm",
        channel_name: Optional[str] = None,
        platform: str = "twitch",
    ) -> Tuple[int, int]:
        """Merge new emotes with existing ones for a channel.
        
        Returns (total_count, new_count) - total emotes and newly added count.
        """
        channel_id = channel_id.lower().strip()
        now = _utc_iso()
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Upsert channel
            conn.execute("""
                INSERT INTO channels (channel_id, channel_name, platform, first_seen, last_updated, video_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(channel_id) DO UPDATE SET
                    channel_name = COALESCE(excluded.channel_name, channel_name),
                    last_updated = excluded.last_updated,
                    video_count = video_count + 1
            """, (channel_id, channel_name or channel_id, platform, now, now))
            
            # Get existing emotes
            cur = conn.execute(
                "SELECT token FROM emotes WHERE channel_id = ?",
                (channel_id,)
            )
            existing = {row[0] for row in cur.fetchall()}
            
            # Insert only new emotes
            new_count = 0
            for token in new_emotes:
                token = token.lower().strip()
                if token and token not in existing:
                    conn.execute("""
                        INSERT INTO emotes (channel_id, token, source, learned_at, frequency)
                        VALUES (?, ?, ?, ?, 1)
                    """, (channel_id, token, source, now))
                    new_count += 1
                elif token in existing:
                    # Increment frequency for existing tokens
                    conn.execute("""
                        UPDATE emotes SET frequency = frequency + 1 
                        WHERE channel_id = ? AND token = ?
                    """, (channel_id, token))
            
            conn.commit()
            
            # Get final count
            cur = conn.execute(
                "SELECT COUNT(*) FROM emotes WHERE channel_id = ?",
                (channel_id,)
            )
            total = cur.fetchone()[0]
            
            return (total, new_count)
        finally:
            conn.close()
    
    def list_channels(self) -> List[Dict]:
        """List all channels with emote counts."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute("""
                SELECT c.channel_id, c.channel_name, c.platform, c.video_count, 
                       c.last_updated, COUNT(e.id) as emote_count
                FROM channels c
                LEFT JOIN emotes e ON c.channel_id = e.channel_id
                GROUP BY c.channel_id
                ORDER BY c.last_updated DESC
            """)
            return [
                {
                    "channel_id": row[0],
                    "channel_name": row[1],
                    "platform": row[2],
                    "video_count": row[3],
                    "last_updated": row[4],
                    "emote_count": row[5],
                }
                for row in cur.fetchall()
            ]
        finally:
            conn.close()
    
    def delete_channel(self, channel_id: str) -> bool:
        """Delete a channel and all its emotes."""
        channel_id = channel_id.lower().strip()
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("DELETE FROM emotes WHERE channel_id = ?", (channel_id,))
            cur = conn.execute("DELETE FROM channels WHERE channel_id = ?", (channel_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            channel_count = conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0]
            emote_count = conn.execute("SELECT COUNT(*) FROM emotes").fetchone()[0]
            llm_emote_count = conn.execute(
                "SELECT COUNT(*) FROM emotes WHERE source = 'llm'"
            ).fetchone()[0]
            return {
                "channel_count": channel_count,
                "emote_count": emote_count,
                "llm_emote_count": llm_emote_count,
                "db_path": str(self.db_path),
            }
        finally:
            conn.close()


def extract_channel_from_url(url: str) -> Optional[Tuple[str, str]]:
    """Extract channel ID and platform from a URL.
    
    Returns (channel_id, platform) or None if not recognized.
    """
    import re
    
    url = url.strip()
    
    # Twitch patterns
    # https://www.twitch.tv/videos/1234567890
    # https://www.twitch.tv/shroud
    # https://www.twitch.tv/shroud/clip/...
    twitch_video = re.search(r"twitch\.tv/videos/(\d+)", url)
    if twitch_video:
        # Video URL - we'll need to get channel from metadata later
        return None
    
    twitch_channel = re.search(r"twitch\.tv/([a-zA-Z0-9_]+)", url)
    if twitch_channel:
        channel = twitch_channel.group(1).lower()
        if channel not in ("videos", "clip", "clips", "directory"):
            return (channel, "twitch")
    
    # YouTube patterns
    # https://www.youtube.com/watch?v=...
    # https://www.youtube.com/@channelname
    # https://www.youtube.com/channel/UC...
    yt_handle = re.search(r"youtube\.com/@([a-zA-Z0-9_-]+)", url)
    if yt_handle:
        return (yt_handle.group(1).lower(), "youtube")
    
    yt_channel = re.search(r"youtube\.com/channel/([a-zA-Z0-9_-]+)", url)
    if yt_channel:
        return (yt_channel.group(1), "youtube")
    
    return None


def extract_channel_from_chat_store(chat_db_path: Path) -> Optional[Tuple[str, str]]:
    """Extract channel ID from chat database metadata.
    
    The Twitch chat downloader often stores streamer info in the chat data.
    
    Returns (channel_id, platform) or None if not found.
    """
    import logging
    log = logging.getLogger("videopipeline.chat.emote_db")
    
    from .store import ChatStore
    
    if not chat_db_path.exists():
        log.debug("[EMOTE] Chat DB path does not exist: %s", chat_db_path)
        return None
    
    try:
        store = ChatStore(chat_db_path)
        
        # Try various metadata keys that might contain channel info
        # ChatMeta uses "channel" as the standard key
        channel = store.get_meta("channel", "")
        channel_id = store.get_meta("channel_id", "")
        channel_name = store.get_meta("channel_name", "")
        streamer = store.get_meta("streamer", "")
        broadcaster = store.get_meta("broadcaster", "")

        # Backfill newer keys for older projects that only stored "channel".
        # This keeps logs and downstream lookups consistent without re-importing chat.
        if channel:
            if not channel_id:
                channel_id = channel
                store.set_meta("channel_id", channel_id)
            if not channel_name:
                channel_name = channel
                store.set_meta("channel_name", channel_name)
        
        log.debug(
            "[EMOTE] Chat store metadata: channel=%r channel_id=%r channel_name=%r streamer=%r broadcaster=%r",
            channel,
            channel_id,
            channel_name,
            streamer,
            broadcaster,
        )
        
        store.close()
        
        # Return first non-empty value
        for candidate in [channel, channel_id, channel_name, streamer, broadcaster]:
            if candidate:
                result = (candidate.lower().strip(), "twitch")
                log.debug("[EMOTE] Extracted channel from chat store: %s", result)
                return result
        
        log.debug("[EMOTE] No channel found in chat store metadata")
        return None
    except Exception as e:
        log.error("[EMOTE] Error extracting channel from chat store: %s", e)
        return None


def get_channel_for_project(proj: Project, source_url: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Get channel ID for a project, trying multiple sources.
    
    Order of precedence:
    1. Source URL (if it has channel info)
    2. Chat database metadata
    3. Project metadata
    
    Args:
        proj: Project instance
        source_url: Optional source URL (if known)
    
    Returns (channel_id, platform) or None
    """
    import logging
    import json
    log = logging.getLogger("videopipeline.chat.emote_db")
    
    log.debug("[EMOTE] get_channel_for_project called with source_url=%s", source_url)
    
    # Try URL first
    if source_url:
        result = extract_channel_from_url(source_url)
        log.debug("[EMOTE] extract_channel_from_url result: %s", result)
        if result:
            log.info("[EMOTE] Channel resolved from URL: %s (%s)", result[0], result[1])
            return result
    
    # Try chat database
    log.debug("[EMOTE] Checking chat_db_path: %s (exists=%s)", proj.chat_db_path, proj.chat_db_path.exists())
    if hasattr(proj, 'chat_db_path') and proj.chat_db_path.exists():
        result = extract_channel_from_chat_store(proj.chat_db_path)
        log.debug("[EMOTE] extract_channel_from_chat_store result: %s", result)
        if result:
            log.info("[EMOTE] Channel resolved from chat store: %s (%s)", result[0], result[1])
            return result
    
    # Try video's info.json (yt-dlp metadata)
    # This works for Twitch VODs where we can't extract channel from URL
    video_path = proj.video_path
    for info_path in [
        video_path.with_suffix(".info.json"),
        video_path.parent / f"{video_path.stem}.info.json",
    ]:
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                # yt-dlp stores Twitch channel as uploader_id (lowercase) or uploader (display)
                # Also check for YouTube channel
                for key in ["uploader_id", "uploader", "channel", "channel_id"]:
                    val = info.get(key, "")
                    if val:
                        channel = str(val).lower().strip()
                        # Detect platform from extractor
                        extractor = info.get("extractor", "").lower()
                        if "twitch" in extractor:
                            platform = "twitch"
                        elif "youtube" in extractor:
                            platform = "youtube"
                        else:
                            platform = "twitch"  # default
                        log.info("[EMOTE] Channel resolved from info.json: %s (%s)", channel, platform)
                        return (channel, platform)
            except Exception as e:
                log.debug("[EMOTE] Failed to read info.json: %s", e)
    
    log.warning("[EMOTE] Could not determine channel for project")
    return None


def get_global_emote_db() -> GlobalEmoteDB:
    """Get the singleton global emote database instance."""
    return GlobalEmoteDB()
