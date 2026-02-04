"""SQLite-based chat storage for efficient time-range queries.

Schema:
  - meta: key-value pairs for chat metadata
  - messages: timestamp-indexed chat messages

This provides fast synced playback queries and supports very large chat logs.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..utils import utc_iso


@dataclass
class ChatMessage:
    """A single normalized chat message."""

    t_ms: int  # Timestamp in milliseconds from video start
    author: str
    author_id: str = ""
    text: str = ""
    badges_json: str = "[]"  # JSON array of badge strings
    emote_count: int = 0
    raw_json: str = "{}"  # Original message data for reference

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_ms": self.t_ms,
            "author": self.author,
            "author_id": self.author_id,
            "text": self.text,
            "badges": json.loads(self.badges_json) if self.badges_json else [],
            "emote_count": self.emote_count,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "ChatMessage":
        return cls(
            t_ms=row[0],
            author=row[1] or "",
            author_id=row[2] or "",
            text=row[3] or "",
            badges_json=row[4] or "[]",
            emote_count=row[5] or 0,
            raw_json=row[6] if len(row) > 6 else "{}",
        )


@dataclass
class ChatMeta:
    """Metadata about the chat replay."""

    source_url: str = ""
    platform: str = ""
    video_id: str = ""
    # Backwards-compatible "channel" field (typically Twitch login, lowercased).
    channel: str = ""

    # Normalized channel identifiers (optional, but preferred when available).
    # For Twitch, these often map to login/display name and may be blank depending on source.
    channel_id: str = ""
    channel_name: str = ""
    message_count: int = 0
    duration_ms: int = 0
    downloaded_at: str = ""
    downloader_version: str = ""
    raw_file: str = ""  # Path to raw downloaded chat file

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "platform": self.platform,
            "video_id": self.video_id,
            "channel": self.channel,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "message_count": self.message_count,
            "duration_ms": self.duration_ms,
            "downloaded_at": self.downloaded_at,
            "downloader_version": self.downloader_version,
            "raw_file": self.raw_file,
        }


class ChatStore:
    """SQLite-backed chat message store.

    Provides efficient time-range queries for synced playback.
    
    Can be used as a context manager:
        with ChatStore(db_path) as store:
            store.initialize()
            store.insert_messages(messages)
        # Automatically closed
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "ChatStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures connection is closed."""
        self.close()

    @property
    def exists(self) -> bool:
        return self.db_path.exists()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def initialize(self) -> None:
        """Create the database schema."""
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                t_ms INTEGER NOT NULL,
                author TEXT,
                author_id TEXT,
                text TEXT,
                badges_json TEXT,
                emote_count INTEGER DEFAULT 0,
                raw_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(t_ms);
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )
        conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata key-value pair."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        """Get a metadata value."""
        conn = self._connect()
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default

    def get_all_meta(self) -> ChatMeta:
        """Get all metadata as a ChatMeta object."""
        return ChatMeta(
            source_url=self.get_meta("source_url"),
            platform=self.get_meta("platform"),
            video_id=self.get_meta("video_id"),
            channel=self.get_meta("channel"),
            channel_id=self.get_meta("channel_id"),
            channel_name=self.get_meta("channel_name"),
            message_count=int(self.get_meta("message_count", "0")),
            duration_ms=int(self.get_meta("duration_ms", "0")),
            downloaded_at=self.get_meta("downloaded_at"),
            downloader_version=self.get_meta("downloader_version"),
            raw_file=self.get_meta("raw_file"),
        )

    def set_all_meta(self, meta: ChatMeta) -> None:
        """Set all metadata from a ChatMeta object."""
        conn = self._connect()
        entries = [
            ("source_url", meta.source_url),
            ("platform", meta.platform),
            ("video_id", meta.video_id),
            ("channel", meta.channel),
            ("channel_id", meta.channel_id),
            ("channel_name", meta.channel_name),
            ("message_count", str(meta.message_count)),
            ("duration_ms", str(meta.duration_ms)),
            ("downloaded_at", meta.downloaded_at),
            ("downloader_version", meta.downloader_version),
            ("raw_file", meta.raw_file),
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            entries,
        )
        conn.commit()

    def clear_messages(self) -> None:
        """Remove all messages (but keep metadata)."""
        conn = self._connect()
        conn.execute("DELETE FROM messages")
        conn.commit()

    def insert_messages(self, messages: List[ChatMessage], batch_size: int = 1000) -> int:
        """Insert messages in batches for efficiency."""
        conn = self._connect()
        count = 0
        batch = []

        for msg in messages:
            batch.append((
                msg.t_ms,
                msg.author,
                msg.author_id,
                msg.text,
                msg.badges_json,
                msg.emote_count,
                msg.raw_json,
            ))
            if len(batch) >= batch_size:
                conn.executemany(
                    """
                    INSERT INTO messages (t_ms, author, author_id, text, badges_json, emote_count, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                count += len(batch)
                batch = []

        if batch:
            conn.executemany(
                """
                INSERT INTO messages (t_ms, author, author_id, text, badges_json, emote_count, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            count += len(batch)

        conn.commit()
        return count

    def get_messages(
        self,
        start_ms: int,
        end_ms: int,
        *,
        offset_ms: int = 0,
        limit: int = 500,
    ) -> List[ChatMessage]:
        """Get messages in a time range, applying sync offset.

        Args:
            start_ms: Start of range in video time (ms)
            end_ms: End of range in video time (ms)
            offset_ms: Chat sync offset to apply (chat_time = video_time + offset)
            limit: Maximum number of messages to return

        Returns:
            List of ChatMessage objects
        """
        conn = self._connect()
        # Apply offset: offset_ms is how much the chat is ahead (+) or behind (-) the video
        # If offset_ms = -2000, chat messages appear 2s earlier in the chat timeline than in video
        # To find chat messages for video time range [start_ms, end_ms], we need:
        # db_time = video_time - offset_ms
        # Example: video_time 3000-5000 with offset -2000 -> db_time 5000-7000
        db_start = start_ms - offset_ms
        db_end = end_ms - offset_ms

        rows = conn.execute(
            """
            SELECT t_ms, author, author_id, text, badges_json, emote_count, raw_json
            FROM messages
            WHERE t_ms >= ? AND t_ms <= ?
            ORDER BY t_ms ASC
            LIMIT ?
            """,
            (db_start, db_end, limit),
        ).fetchall()

        return [ChatMessage.from_row(tuple(row)) for row in rows]

    def get_message_count(self) -> int:
        """Get total number of messages."""
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0] if row else 0

    def get_time_range(self) -> tuple[int, int]:
        """Get (min_t_ms, max_t_ms) of messages."""
        conn = self._connect()
        row = conn.execute(
            "SELECT MIN(t_ms), MAX(t_ms) FROM messages"
        ).fetchone()
        if row and row[0] is not None:
            return (row[0], row[1])
        return (0, 0)

    def get_all_timestamps_ms(self) -> List[int]:
        """Get all message timestamps for feature computation."""
        conn = self._connect()
        rows = conn.execute("SELECT t_ms FROM messages ORDER BY t_ms").fetchall()
        return [row[0] for row in rows]

    def iter_messages(self, batch_size: int = 5000) -> Iterator[ChatMessage]:
        """Iterate over all messages in time order.

        Uses keyset pagination (t_ms, rowid) instead of OFFSET.

        Why: OFFSET becomes progressively slower on very large tables because
        SQLite must scan and discard the skipped rows.
        """
        conn = self._connect()
        last_t_ms = -1
        last_rowid = -1
        while True:
            rows = conn.execute(
                """
                SELECT rowid, t_ms, author, author_id, text, badges_json, emote_count, raw_json
                FROM messages
                WHERE (t_ms > ?) OR (t_ms = ? AND rowid > ?)
                ORDER BY t_ms ASC, rowid ASC
                LIMIT ?
                """,
                (last_t_ms, last_t_ms, last_rowid, batch_size),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                last_rowid = row[0]
                last_t_ms = row[1]
                # Build ChatMessage from row[1:] (skip rowid)
                yield ChatMessage.from_row(tuple(row[1:]))
