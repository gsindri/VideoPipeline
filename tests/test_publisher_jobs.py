import json
import sys
import threading
import types
from pathlib import Path

import pytest


def _install_google_stubs() -> None:
    if "google.auth.transport.requests" not in sys.modules:
        google_mod = sys.modules.setdefault("google", types.ModuleType("google"))
        auth_mod = sys.modules.setdefault("google.auth", types.ModuleType("google.auth"))
        transport_mod = sys.modules.setdefault("google.auth.transport", types.ModuleType("google.auth.transport"))
        requests_mod = types.ModuleType("google.auth.transport.requests")
        exceptions_mod = types.ModuleType("google.auth.exceptions")

        class Request:
            pass

        class AuthorizedSession:
            def __init__(self, *args, **kwargs):
                pass

        class RefreshError(Exception):
            pass

        requests_mod.Request = Request
        requests_mod.AuthorizedSession = AuthorizedSession
        exceptions_mod.RefreshError = RefreshError
        sys.modules["google.auth.transport.requests"] = requests_mod
        sys.modules["google.auth.exceptions"] = exceptions_mod
        setattr(transport_mod, "requests", requests_mod)
        setattr(auth_mod, "exceptions", exceptions_mod)
        setattr(auth_mod, "transport", transport_mod)
        setattr(google_mod, "auth", auth_mod)

    if "google.oauth2.credentials" not in sys.modules:
        google_mod = sys.modules.setdefault("google", types.ModuleType("google"))
        oauth2_mod = sys.modules.setdefault("google.oauth2", types.ModuleType("google.oauth2"))
        credentials_mod = types.ModuleType("google.oauth2.credentials")

        class Credentials:
            def __init__(self, *args, **kwargs):
                self.token = kwargs.get("token")
                self.refresh_token = kwargs.get("refresh_token")
                self.expired = False

            def refresh(self, request):
                self.expired = False

        credentials_mod.Credentials = Credentials
        sys.modules["google.oauth2.credentials"] = credentials_mod
        setattr(oauth2_mod, "credentials", credentials_mod)
        setattr(google_mod, "oauth2", oauth2_mod)


_install_google_stubs()

from videopipeline.publisher.accounts import Account, AccountStore
from videopipeline.publisher.connectors.tiktok import TikTokConnector
from videopipeline.publisher.connectors.youtube import YouTubeConnector
from videopipeline.publisher.jobs import PublishJobStore
from videopipeline.publisher.queue import PublishWorker


def test_job_store_create_update_and_dedup(tmp_path: Path):
    store = PublishJobStore(path=tmp_path / "publisher.sqlite")
    job = store.create_job(
        job_id="job123",
        platform="youtube",
        account_id="acct",
        file_path="/tmp/video.mp4",
        metadata_path="/tmp/meta.json",
    )
    assert job.status == "queued"

    updated = store.update_job(job.id, status="running", progress=0.5)
    assert updated.status == "running"
    assert updated.progress == 0.5

    store.mark_dedup("youtube", "acct", "hash123", "remote123", "https://youtu.be/remote123")
    dedup = store.lookup_dedup("youtube", "acct", "hash123")
    assert dedup is not None
    assert dedup["remote_id"] == "remote123"
    assert store.delete_dedup("youtube", "acct", "hash123") is True
    assert store.lookup_dedup("youtube", "acct", "hash123") is None


def test_job_store_backoff_skips_recent_retries(tmp_path: Path):
    store = PublishJobStore(path=tmp_path / "publisher.sqlite")
    store.create_job(
        job_id="job123",
        platform="youtube",
        account_id="acct",
        file_path="/tmp/video.mp4",
        metadata_path="/tmp/meta.json",
    )
    store.update_job("job123", attempts=1)
    claimed = store.claim_next(backoff_fn=lambda attempts: 999)
    assert claimed is None


def test_job_store_cancel_rejects_completed_job(tmp_path: Path):
    store = PublishJobStore(path=tmp_path / "publisher.sqlite")
    store.create_job(
        job_id="job123",
        platform="youtube",
        account_id="acct",
        file_path="/tmp/video.mp4",
        metadata_path="/tmp/meta.json",
    )
    store.update_job("job123", status="succeeded", progress=1.0)

    with pytest.raises(ValueError):
        store.cancel("job123")


def test_publish_worker_preserves_canceled_job(tmp_path: Path, monkeypatch):
    store = PublishJobStore(path=tmp_path / "publisher.sqlite")
    accounts = AccountStore(path=tmp_path / "accounts.json")
    account = accounts.add(platform="youtube", label="YT")

    video_path = tmp_path / "clip.mp4"
    metadata_path = tmp_path / "clip.json"
    video_path.write_bytes(b"video")
    metadata_path.write_text(json.dumps({"title": "Clip"}), encoding="utf-8")

    store.create_job(
        job_id="job123",
        platform="youtube",
        account_id=account.id,
        file_path=str(video_path),
        metadata_path=str(metadata_path),
    )

    started = threading.Event()
    finish = threading.Event()

    class DummyConnector:
        def validate_media(self, file_path, metadata):
            return None

        def publish(self, *, file_path, metadata, resume_state, on_progress, on_resume):
            started.set()
            on_resume({"session": "abc"})
            while not finish.wait(0.01):
                pass
            on_progress(0.5)
            raise AssertionError("publish should not continue after cancellation")

    monkeypatch.setattr("videopipeline.publisher.queue.load_tokens", lambda platform, account_id: {"token": "x"})
    monkeypatch.setattr("videopipeline.publisher.queue.get_connector", lambda platform, account, tokens: DummyConnector())
    monkeypatch.setattr("videopipeline.publisher.queue.logs_dir", lambda: tmp_path)

    worker = PublishWorker(job_store=store, account_store=accounts)
    thread = threading.Thread(target=worker.run_once, kwargs={"job_id": "job123"}, daemon=True)
    thread.start()

    assert started.wait(timeout=2.0)
    canceled = store.cancel("job123")
    assert canceled.status == "canceled"
    finish.set()
    thread.join(timeout=2.0)

    final = store.get_job("job123")
    assert final.status == "canceled"
    assert final.remote_id is None


def test_connectors_inherit_shared_publish_flow():
    youtube = YouTubeConnector(
        account=Account(id="acct-yt", platform="youtube", label="YT"),
        tokens={},
    )
    tiktok = TikTokConnector(
        account=Account(id="acct-tt", platform="tiktok", label="TT"),
        tokens={},
    )

    assert callable(getattr(youtube, "publish", None))
    assert callable(getattr(tiktok, "publish", None))


def test_youtube_connector_delete_remote_accepts_success_and_not_found():
    class DummyResponse:
        def __init__(self, status_code: int, text: str = "") -> None:
            self.status_code = status_code
            self.text = text

    class DummySession:
        def __init__(self) -> None:
            self.calls = []

        def delete(self, url, params=None):
            self.calls.append((url, params))
            return DummyResponse(204)

    connector = YouTubeConnector(
        account=Account(id="acct-yt", platform="youtube", label="YT"),
        tokens={},
    )
    session = DummySession()
    connector._session = lambda: session  # type: ignore[method-assign]

    connector.delete_remote(remote_id="video123")

    assert session.calls == [
        ("https://www.googleapis.com/youtube/v3/videos", {"id": "video123"})
    ]
