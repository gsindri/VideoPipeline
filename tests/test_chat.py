"""Tests for chat replay integration.

Tests:
- Chat store SQLite operations
- Chat message normalization
- Chat feature computation
- Time range queries with offset
- Highlight integration with chat
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from videopipeline.chat.store import ChatStore, ChatMessage, ChatMeta
from videopipeline.chat.normalize import (
    normalize_chat_messages,
    detect_chat_format,
    ChatFormat,
    load_and_normalize,
)
from videopipeline.chat.features import compute_chat_features


# ---------------------------------------------------------------------------
# ChatStore Tests
# ---------------------------------------------------------------------------


class TestChatStore:
    """Tests for SQLite chat storage."""

    def test_create_and_initialize(self, tmp_path: Path):
        """Test creating and initializing a chat store."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        assert db_path.exists()
        assert store.get_message_count() == 0
        store.close()

    def test_insert_and_retrieve_messages(self, tmp_path: Path):
        """Test inserting and retrieving messages."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        messages = [
            ChatMessage(t_ms=0, author="user1", text="Hello"),
            ChatMessage(t_ms=1000, author="user2", text="Hi there"),
            ChatMessage(t_ms=2000, author="user1", text="How are you?"),
            ChatMessage(t_ms=5000, author="user3", text="Great!"),
        ]

        count = store.insert_messages(messages)
        assert count == 4
        assert store.get_message_count() == 4
        store.close()

    def test_time_range_query(self, tmp_path: Path):
        """Test querying messages by time range."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        messages = [
            ChatMessage(t_ms=1000, author="a", text="m1"),
            ChatMessage(t_ms=2000, author="b", text="m2"),
            ChatMessage(t_ms=3000, author="c", text="m3"),
            ChatMessage(t_ms=4000, author="d", text="m4"),
            ChatMessage(t_ms=5000, author="e", text="m5"),
        ]
        store.insert_messages(messages)

        # Query middle range
        result = store.get_messages(2000, 4000)
        assert len(result) == 3
        assert result[0].author == "b"
        assert result[2].author == "d"

        store.close()

    def test_offset_application(self, tmp_path: Path):
        """Test that sync offset is correctly applied."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        messages = [
            ChatMessage(t_ms=5000, author="a", text="m1"),
            ChatMessage(t_ms=6000, author="b", text="m2"),
            ChatMessage(t_ms=7000, author="c", text="m3"),
        ]
        store.insert_messages(messages)

        # Without offset: query 5000-7000 should return all 3
        result = store.get_messages(5000, 7000, offset_ms=0)
        assert len(result) == 3

        # With negative offset (-2000): chat appears earlier in video
        # To find messages at video time 3000-5000, we need db time 5000-7000
        result = store.get_messages(3000, 5000, offset_ms=-2000)
        assert len(result) == 3
        assert result[0].t_ms == 5000

        # With positive offset (+2000): chat appears later in video
        # To find messages at video time 7000-9000, we need db time 5000-7000
        result = store.get_messages(7000, 9000, offset_ms=2000)
        assert len(result) == 3

        store.close()

    def test_metadata_storage(self, tmp_path: Path):
        """Test metadata storage and retrieval."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        meta = ChatMeta(
            source_url="https://twitch.tv/videos/123456",
            platform="twitch",
            video_id="123456",
            channel="test_channel",
            message_count=1000,
            duration_ms=3600000,
            downloaded_at="2026-01-22T12:00:00Z",
            downloader_version="1.0.0",
            raw_file="/path/to/chat.json",
        )
        store.set_all_meta(meta)

        retrieved = store.get_all_meta()
        assert retrieved.source_url == meta.source_url
        assert retrieved.platform == meta.platform
        assert retrieved.message_count == meta.message_count
        assert retrieved.duration_ms == meta.duration_ms

        store.close()

    def test_get_all_timestamps(self, tmp_path: Path):
        """Test getting all timestamps for feature computation."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        messages = [
            ChatMessage(t_ms=100, author="a", text="m1"),
            ChatMessage(t_ms=200, author="b", text="m2"),
            ChatMessage(t_ms=300, author="c", text="m3"),
        ]
        store.insert_messages(messages)

        timestamps = store.get_all_timestamps_ms()
        assert timestamps == [100, 200, 300]

        store.close()


# ---------------------------------------------------------------------------
# Normalization Tests
# ---------------------------------------------------------------------------


class TestChatNormalization:
    """Tests for chat message normalization."""

    def test_detect_chat_replay_downloader_format(self):
        """Test detecting chat-replay-downloader format."""
        data = [
            {"time_in_seconds": 0.0, "message": "Hello", "author": {"name": "user1"}},
            {"time_in_seconds": 1.5, "message": "Hi", "author": {"name": "user2"}},
        ]
        fmt = detect_chat_format(data)
        assert fmt == ChatFormat.CHAT_REPLAY_DOWNLOADER

    def test_detect_twitch_vod_format(self):
        """Test detecting Twitch VOD format."""
        data = [
            {
                "content_offset_seconds": 0.0,
                "commenter": {"display_name": "user1"},
                "message": {"body": "Hello"},
            }
        ]
        fmt = detect_chat_format(data)
        assert fmt == ChatFormat.TWITCH_VOD

    def test_detect_generic_format(self):
        """Test detecting generic JSON format."""
        data = [{"timestamp": 1000, "text": "Hello", "username": "user1"}]
        fmt = detect_chat_format(data)
        assert fmt == ChatFormat.GENERIC_JSON

    def test_normalize_chat_replay_downloader(self):
        """Test normalizing chat-replay-downloader output."""
        data = [
            {
                "time_in_seconds": 0.0,
                "message": "Hello world",
                "author": {"name": "streamer", "id": "12345"},
            },
            {
                "time_in_seconds": 2.5,
                "message": "POG",
                "author": {"name": "viewer1", "id": "67890"},
            },
        ]

        messages = normalize_chat_messages(data)

        assert len(messages) == 2
        assert messages[0].t_ms == 0
        assert messages[0].author == "streamer"
        assert messages[0].text == "Hello world"
        assert messages[1].t_ms == 2500
        assert messages[1].author == "viewer1"

    def test_normalize_millisecond_timestamps(self):
        """Test handling millisecond timestamps."""
        data = [
            {"timestamp_ms": 0, "text": "First"},
            {"timestamp_ms": 1500, "text": "Second"},
            {"timestamp_ms": 3000, "text": "Third"},
        ]

        messages = normalize_chat_messages(data)

        assert len(messages) == 3
        assert messages[0].t_ms == 0
        assert messages[1].t_ms == 1500
        assert messages[2].t_ms == 3000

    def test_normalize_handles_missing_fields(self):
        """Test that normalization handles missing optional fields."""
        data = [{"time_in_seconds": 1.0, "message": "Test"}]

        messages = normalize_chat_messages(data)

        assert len(messages) == 1
        assert messages[0].author == ""
        assert messages[0].author_id == ""

    def test_normalize_sorts_by_timestamp(self):
        """Test that messages are sorted by timestamp."""
        data = [
            {"timestamp": 3.0, "text": "Third"},
            {"timestamp": 1.0, "text": "First"},
            {"timestamp": 2.0, "text": "Second"},
        ]

        messages = normalize_chat_messages(data)

        assert len(messages) == 3
        assert messages[0].t_ms == 1000
        assert messages[1].t_ms == 2000
        assert messages[2].t_ms == 3000

    def test_normalize_handles_nested_messages(self):
        """Test handling nested messages structure."""
        data = {
            "metadata": {"video_id": "123"},
            "messages": [
                {"timestamp": 0.0, "text": "Hello"},
                {"timestamp": 1.0, "text": "World"},
            ],
        }

        messages = normalize_chat_messages(data)

        assert len(messages) == 2


# ---------------------------------------------------------------------------
# Feature Computation Tests
# ---------------------------------------------------------------------------


class TestChatFeatures:
    """Tests for chat feature extraction."""

    def test_compute_basic_features(self, tmp_path: Path):
        """Test computing basic chat features."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        # Create a burst of messages at t=5s
        messages = []
        for i in range(10):
            messages.append(ChatMessage(t_ms=5000 + i * 50, author=f"user{i}", text="poggers"))

        # Add some messages elsewhere
        messages.append(ChatMessage(t_ms=1000, author="user0", text="hello"))
        messages.append(ChatMessage(t_ms=10000, author="user0", text="end"))

        store.insert_messages(messages)

        features = compute_chat_features(store, duration_s=15.0, hop_s=1.0, smooth_s=1.0)

        # Should have 16 bins (0-15 seconds)
        assert len(features["counts"]) == 16

        # The burst should be at bin 5
        assert features["counts"][5] == 10  # 10 messages in that second
        assert features["counts"][1] == 1  # 1 message

        # Scores should be higher around the burst
        assert features["scores"][5] > features["scores"][1]

        store.close()

    def test_known_burst_creates_spike(self, tmp_path: Path):
        """Test that a known burst creates a spike in scores."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        # Create a baseline of ~2 messages per second
        messages = []
        for i in range(30):
            t = i * 1000
            messages.append(ChatMessage(t_ms=t, author=f"user{i%5}", text="normal"))
            messages.append(ChatMessage(t_ms=t + 100, author=f"user{(i+1)%5}", text="chat"))

        # Create a huge burst at t=15s (50 messages in 1 second)
        for i in range(50):
            messages.append(ChatMessage(t_ms=15000 + i * 20, author=f"burst{i}", text="POGGERS"))

        store.insert_messages(messages)

        features = compute_chat_features(store, duration_s=30.0, hop_s=0.5, smooth_s=2.0)

        # Find the peak
        peak_idx = int(np.argmax(features["scores"]))
        peak_time_s = peak_idx * 0.5

        # Peak should be around t=15s (within smoothing window)
        assert 13.0 <= peak_time_s <= 17.0

        store.close()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestChatIntegration:
    """Integration tests for chat with highlights."""

    def test_load_and_normalize_file(self, tmp_path: Path):
        """Test loading and normalizing a chat file."""
        chat_file = tmp_path / "chat.json"
        data = [
            {"time_in_seconds": 0.0, "message": "Hello", "author": {"name": "user1"}},
            {"time_in_seconds": 5.0, "message": "World", "author": {"name": "user2"}},
        ]
        chat_file.write_text(json.dumps(data))

        messages, fmt = load_and_normalize(chat_file)

        assert len(messages) == 2
        assert fmt == ChatFormat.CHAT_REPLAY_DOWNLOADER

    def test_message_to_dict(self):
        """Test ChatMessage serialization."""
        msg = ChatMessage(
            t_ms=1500,
            author="streamer",
            author_id="123",
            text="Hello viewers!",
            badges_json='["broadcaster", "subscriber"]',
            emote_count=2,
        )

        d = msg.to_dict()

        assert d["t_ms"] == 1500
        assert d["author"] == "streamer"
        assert d["badges"] == ["broadcaster", "subscriber"]
        assert d["emote_count"] == 2


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_chat_store(self, tmp_path: Path):
        """Test operations on empty chat store."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        assert store.get_message_count() == 0
        assert store.get_messages(0, 10000) == []
        assert store.get_all_timestamps_ms() == []

        store.close()

    def test_normalize_empty_data(self):
        """Test normalizing empty data."""
        assert normalize_chat_messages([]) == []
        assert normalize_chat_messages({}) == []
        assert normalize_chat_messages({"messages": []}) == []

    def test_normalize_invalid_timestamps(self):
        """Test handling invalid timestamps."""
        data = [
            {"time_in_seconds": "invalid", "text": "bad1"},
            {"time_in_seconds": None, "text": "bad2"},
            {"timestamp": 1.0, "text": "good"},
        ]

        messages = normalize_chat_messages(data)

        # Only the valid message should be included
        assert len(messages) == 1
        assert messages[0].text == "good"

    def test_negative_timestamps_clamped(self):
        """Test that negative timestamps are clamped to 0."""
        data = [{"timestamp": -5.0, "text": "before start"}]

        messages = normalize_chat_messages(data)

        assert len(messages) == 1
        assert messages[0].t_ms == 0

    def test_large_batch_insert(self, tmp_path: Path):
        """Test inserting large batches of messages."""
        db_path = tmp_path / "chat.sqlite"
        store = ChatStore(db_path)
        store.initialize()

        # Create 10,000 messages
        messages = [
            ChatMessage(t_ms=i * 100, author=f"user{i%100}", text=f"message {i}")
            for i in range(10000)
        ]

        count = store.insert_messages(messages, batch_size=1000)

        assert count == 10000
        assert store.get_message_count() == 10000

        store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
