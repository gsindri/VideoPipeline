"""Tests for the Studio publisher API."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import pytest

from videopipeline.studio.publisher_api import (
    ExportInfo,
    scan_project_exports,
    is_safe_export_path,
)
from videopipeline.publisher.accounts import AccountStore, Account
from videopipeline.publisher.jobs import PublishJobStore


class TestExportScanning:
    """Tests for export discovery."""

    def test_scan_empty_directory(self, tmp_path: Path):
        """Empty exports directory returns empty list."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()
        result = scan_project_exports(exports_dir)
        assert result == []

    def test_scan_nonexistent_directory(self, tmp_path: Path):
        """Nonexistent directory returns empty list."""
        exports_dir = tmp_path / "nonexistent"
        result = scan_project_exports(exports_dir)
        assert result == []

    def test_scan_finds_mp4_files(self, tmp_path: Path):
        """Finds mp4 files in exports directory."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()

        # Create test mp4 files
        (exports_dir / "clip1.mp4").write_bytes(b"fake video data")
        (exports_dir / "clip2.mp4").write_bytes(b"fake video data 2")

        result = scan_project_exports(exports_dir)
        assert len(result) == 2

        export_ids = {e.export_id for e in result}
        assert export_ids == {"clip1", "clip2"}

    def test_scan_with_metadata_json(self, tmp_path: Path):
        """Finds and parses matching metadata.json files."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()

        (exports_dir / "clip1.mp4").write_bytes(b"fake video")
        metadata = {
            "title": "Test Clip",
            "description": "A test description",
            "duration_seconds": 30.5,
            "template": "vertical_blur",
        }
        (exports_dir / "clip1.json").write_text(json.dumps(metadata), encoding="utf-8")

        result = scan_project_exports(exports_dir)
        assert len(result) == 1

        export = result[0]
        assert export.export_id == "clip1"
        assert export.metadata["title"] == "Test Clip"
        assert export.metadata["description"] == "A test description"
        assert export.duration_seconds == 30.5

    def test_scan_with_underscore_metadata(self, tmp_path: Path):
        """Finds _metadata.json pattern."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()

        (exports_dir / "clip1.mp4").write_bytes(b"fake video")
        metadata = {"title": "Alt Pattern"}
        (exports_dir / "clip1_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        result = scan_project_exports(exports_dir)
        assert len(result) == 1
        assert result[0].metadata["title"] == "Alt Pattern"

    def test_scan_handles_invalid_json(self, tmp_path: Path):
        """Gracefully handles invalid JSON metadata."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()

        (exports_dir / "clip1.mp4").write_bytes(b"fake video")
        (exports_dir / "clip1.json").write_text("not valid json {{{", encoding="utf-8")

        result = scan_project_exports(exports_dir)
        assert len(result) == 1
        assert result[0].metadata == {}


class TestPathSafety:
    """Tests for path restriction logic."""

    def test_safe_path_inside_exports(self, tmp_path: Path):
        """Files inside exports directory are safe."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()
        file_path = exports_dir / "clip.mp4"
        file_path.touch()

        assert is_safe_export_path(file_path, exports_dir) is True

    def test_unsafe_path_outside_exports(self, tmp_path: Path):
        """Files outside exports directory are unsafe."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        file_path = other_dir / "clip.mp4"
        file_path.touch()

        assert is_safe_export_path(file_path, exports_dir) is False

    def test_unsafe_path_parent_traversal(self, tmp_path: Path):
        """Parent traversal attempts are unsafe."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()
        # Try to escape via ..
        file_path = exports_dir / ".." / "secret.mp4"

        assert is_safe_export_path(file_path, exports_dir) is False

    def test_unsafe_absolute_path(self, tmp_path: Path):
        """Absolute paths outside exports are unsafe."""
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()

        if Path("C:/").exists():
            # Windows
            file_path = Path("C:/Windows/System32/something.mp4")
        else:
            # Unix
            file_path = Path("/etc/passwd")

        assert is_safe_export_path(file_path, exports_dir) is False


class TestQueueAPI:
    """Tests for queue API logic."""

    def test_job_store_creates_job(self, tmp_path: Path):
        """Job store creates jobs correctly."""
        db_path = tmp_path / "publisher.sqlite"
        store = PublishJobStore(path=db_path)

        job = store.create_job(
            job_id="test123",
            platform="youtube",
            account_id="acc123",
            file_path="/path/to/video.mp4",
            metadata_path="/path/to/metadata.json",
        )

        assert job.id == "test123"
        assert job.platform == "youtube"
        assert job.account_id == "acc123"
        assert job.status == "queued"
        assert job.progress == 0.0

    def test_job_store_retry(self, tmp_path: Path):
        """Retry resets job status."""
        db_path = tmp_path / "publisher.sqlite"
        store = PublishJobStore(path=db_path)

        job = store.create_job(
            job_id="test456",
            platform="youtube",
            account_id="acc123",
            file_path="/path/to/video.mp4",
            metadata_path="/path/to/metadata.json",
        )

        # Simulate failure
        store.update_job(job.id, status="failed", last_error="Some error")

        # Retry
        retried = store.retry(job.id)
        assert retried.status == "queued"
        assert retried.last_error is None
        assert retried.progress == 0.0

    def test_job_store_cancel(self, tmp_path: Path):
        """Cancel sets job status to canceled."""
        db_path = tmp_path / "publisher.sqlite"
        store = PublishJobStore(path=db_path)

        job = store.create_job(
            job_id="test789",
            platform="youtube",
            account_id="acc123",
            file_path="/path/to/video.mp4",
            metadata_path="/path/to/metadata.json",
        )

        canceled = store.cancel(job.id)
        assert canceled.status == "canceled"


class TestAccountStore:
    """Tests for account store."""

    def test_account_store_add_and_list(self, tmp_path: Path):
        """Can add and list accounts."""
        accounts_path = tmp_path / "accounts.json"
        store = AccountStore(path=accounts_path)

        account = store.add(
            platform="youtube",
            label="Test Channel",
            metadata={"channel_id": "UC123"},
        )

        assert account.platform == "youtube"
        assert account.label == "Test Channel"

        accounts = store.list()
        assert len(accounts) == 1
        assert accounts[0].id == account.id

    def test_account_store_get(self, tmp_path: Path):
        """Can get account by ID."""
        accounts_path = tmp_path / "accounts.json"
        store = AccountStore(path=accounts_path)

        account = store.add(platform="tiktok", label="My TikTok")

        retrieved = store.get(account.id)
        assert retrieved is not None
        assert retrieved.label == "My TikTok"

        missing = store.get("nonexistent")
        assert missing is None


class TestExportInfoToDict:
    """Tests for ExportInfo serialization."""

    def test_to_dict(self, tmp_path: Path):
        """ExportInfo converts to dict correctly."""
        mp4_path = tmp_path / "clip.mp4"
        metadata_path = tmp_path / "clip.json"

        info = ExportInfo(
            export_id="clip",
            mp4_path=mp4_path,
            metadata_path=metadata_path,
            metadata={"title": "Test", "description": "Desc", "template": "vertical_blur"},
            duration_seconds=45.0,
            created_at="2024-01-01T00:00:00Z",
            file_size_bytes=1024000,
        )

        d = info.to_dict()
        assert d["export_id"] == "clip"
        assert d["mp4_filename"] == "clip.mp4"
        assert d["title"] == "Test"
        assert d["description"] == "Desc"
        assert d["template"] == "vertical_blur"
        assert d["duration_seconds"] == 45.0
        assert d["file_size_bytes"] == 1024000
