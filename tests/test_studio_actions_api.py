import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _disable_publish_worker(monkeypatch):
    # Studio starts a background publisher worker; it's irrelevant for these tests.
    from videopipeline.publisher.queue import PublishWorker

    monkeypatch.setattr(PublishWorker, "start", lambda self: None)


@pytest.fixture(autouse=True)
def _reset_job_manager():
    from videopipeline.studio.jobs import JOB_MANAGER

    with JOB_MANAGER._lock:  # type: ignore[attr-defined]
        JOB_MANAGER._jobs.clear()  # type: ignore[attr-defined]
    yield
    with JOB_MANAGER._lock:  # type: ignore[attr-defined]
        JOB_MANAGER._jobs.clear()  # type: ignore[attr-defined]


def _make_client(tmp_path: Path, monkeypatch, *, token: str | None = None) -> TestClient:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VP_API_TOKEN", raising=False)

    if token is not None:
        monkeypatch.setenv("VP_API_TOKEN", token)

    from videopipeline.studio.app import create_app

    app = create_app(video_path=None, profile_path=None)
    return TestClient(app)


def test_auth_disabled_allows_api(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token=None)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_auth_enabled_requires_header_or_query(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")

    r = client.get("/api/health")
    assert r.status_code == 401

    r = client.get("/api/health", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200

    r = client.get("/api/health?token=secret")
    assert r.status_code == 200

    r = client.get("/api/health?token=wrong")
    assert r.status_code == 401


def test_actions_ingest_url_blocks_ssrf(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.post("/api/actions/ingest_url", headers=hdr, json={"url": "http://127.0.0.1/"})
    assert r.status_code == 400
    assert r.json().get("detail") == "invalid_url_host"

    r = client.post("/api/actions/ingest_url", headers=hdr, json={"url": "https://example.com/video"})
    assert r.status_code == 400
    assert r.json().get("detail") == "url_domain_not_allowed"


def test_actions_ingest_url_www_normalization(tmp_path, monkeypatch):
    """URLs with www. prefix, trailing dots, or mixed case should match the allowlist."""
    from videopipeline.studio.actions_api import _validate_ingest_url

    allow = {"twitch.tv", "youtube.com"}
    # www. prefix should be stripped and matched
    assert _validate_ingest_url("https://www.twitch.tv/videos/123", allow_domains=allow)
    # trailing dot
    assert _validate_ingest_url("https://twitch.tv./videos/123", allow_domains=allow)
    # mixed case
    assert _validate_ingest_url("https://WWW.TWITCH.TV/videos/123", allow_domains=allow)
    # subdomain still works
    assert _validate_ingest_url("https://clips.twitch.tv/SomeClip", allow_domains=allow)


def test_actions_allow_domains_env(tmp_path, monkeypatch):
    """VP_ACTIONS_ALLOW_DOMAINS env var extends the built-in allowlist."""
    monkeypatch.setenv("VP_ACTIONS_ALLOW_DOMAINS", "example.com, custom.io")
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    # example.com is now allowed (was blocked before)
    r = client.post("/api/actions/ingest_url", headers=hdr, json={"url": "https://example.com/video"})
    # Should not be 400/url_domain_not_allowed (may fail for other reasons in test, but domain is accepted)
    assert r.json().get("detail") != "url_domain_not_allowed"

    # diagnostics should list it
    r = client.get("/api/actions/diagnostics", headers=hdr)
    assert r.status_code == 200
    domains = r.json().get("allowed_domains", [])
    assert "example.com" in domains
    assert "custom.io" in domains
    # defaults are still present
    assert "twitch.tv" in domains


def test_actions_rate_limiting(tmp_path, monkeypatch):
    monkeypatch.setenv("VP_ACTIONS_RL_PER_ACTOR_CAPACITY", "2")
    monkeypatch.setenv("VP_ACTIONS_RL_PER_ACTOR_REFILL_PER_S", "0")
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    assert client.get("/api/actions/health", headers=hdr).status_code == 200
    assert client.get("/api/actions/health", headers=hdr).status_code == 200
    r = client.get("/api/actions/health", headers=hdr)
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_actions_ingest_url_idempotency(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    # Patch out the actual downloader so we don't hit network / yt-dlp in tests.
    import videopipeline.studio.actions_api as actions_api
    from videopipeline.ingest.models import IngestResult

    dummy_src = tmp_path / "dummy.mp4"
    dummy_src.write_bytes(b"")

    def fake_download_url(url: str, *args, **kwargs):
        on_progress = kwargs.get("on_progress")
        if on_progress:
            on_progress(1.0, "done")
        return IngestResult(video_path=dummy_src, url=url, title="dummy")

    monkeypatch.setattr(actions_api, "download_url", fake_download_url)

    # Chat download/import is best-effort; make it fail fast.
    import videopipeline.chat.downloader as chat_downloader

    monkeypatch.setattr(chat_downloader, "download_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no chat")))

    body = {
        "url": "https://www.twitch.tv/videos/123",
        "client_request_id": "req-1",
        "options": {"create_preview": False},
    }

    r1 = client.post("/api/actions/ingest_url", headers=hdr, json=body)
    assert r1.status_code == 200
    payload1 = r1.json()

    r2 = client.post("/api/actions/ingest_url", headers=hdr, json=body)
    assert r2.status_code == 200
    payload2 = r2.json()

    assert payload1 == payload2
    assert "job_id" in payload1
    assert "project_id" in payload1

    expected_pid = hashlib.sha256("twitch_123".encode("utf-8")).hexdigest()
    assert payload1["project_id"] == expected_pid

    project_json = tmp_path / "outputs" / "projects" / expected_pid / "project.json"
    assert project_json.exists()


def test_actions_results_summary_clamps(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_123".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    proj_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for i in range(50):
        candidates.append(
            {
                "rank": i + 1,
                "score": 1.0 - (i * 0.001),
                "start_s": float(i * 10),
                "end_s": float(i * 10 + 20),
                "peak_time_s": float(i * 10 + 10),
                "title": f"cand {i+1}",
            }
        )

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 3600.0},
        "analysis": {"highlights": {"candidates": candidates}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    r = client.get(
        f"/api/actions/results/summary?project_id={pid}&top_n=999&snippet_chars=999999&chat_lines=999",
        headers=hdr,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["project_id"] == pid
    assert data["highlights"]["total_candidates"] == 50
    assert len(data["highlights"]["candidates"]) == 30


def test_actions_openapi_freeform_object_schemas_have_properties(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/openapi.json", headers=hdr)
    assert r.status_code == 200
    spec = r.json()
    schemas = ((spec.get("components") or {}).get("schemas") or {})

    results_summary = schemas.get("ResultsSummaryResponse")
    diagnostics = schemas.get("DiagnosticsResponse")

    assert isinstance(results_summary, dict)
    assert isinstance(diagnostics, dict)
    assert results_summary.get("type") == "object"
    assert diagnostics.get("type") == "object"
    assert results_summary.get("properties") == {}
    assert diagnostics.get("properties") == {}
    assert results_summary.get("additionalProperties") is True
    assert diagnostics.get("additionalProperties") is True


def test_actions_run_ingest_analyze_returns_run_id_and_project_id(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    # Patch out the actual downloader + analysis so the background thread is fast.
    import videopipeline.studio.actions_api as actions_api
    from videopipeline.ingest.models import IngestResult

    dummy_src = tmp_path / "dummy.mp4"
    dummy_src.write_bytes(b"")

    def fake_download_url(url: str, *args, **kwargs):
        on_progress = kwargs.get("on_progress")
        if on_progress:
            on_progress(1.0, "done")
        return IngestResult(video_path=dummy_src, url=url, title="dummy")

    class DummyAnalysisResult:
        success = True
        error = None
        tasks_run = []
        total_elapsed_seconds = 0.0
        missing_targets = set()

    monkeypatch.setattr(actions_api, "download_url", fake_download_url)
    monkeypatch.setattr(actions_api, "run_analysis", lambda *a, **k: DummyAnalysisResult())

    # Chat download/import is best-effort; make it fail fast.
    import videopipeline.chat.downloader as chat_downloader

    monkeypatch.setattr(chat_downloader, "download_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no chat")))

    r = client.post(
        "/api/actions/run_ingest_analyze",
        headers=hdr,
        json={"url": "https://www.twitch.tv/videos/123", "client_request_id": "run-req-1"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "run_id" in data
    assert "job_id" in data
    assert "project_id" in data
    assert data["run_id"] == data["job_id"]


def test_actions_openapi_run_full_export_top_is_consequential(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/openapi.json", headers=hdr)
    assert r.status_code == 200
    spec = r.json()
    paths = spec.get("paths") or {}
    op = (((paths.get("/api/actions/run_full_export_top") or {}).get("post")) or {})
    assert op.get("x-openai-isConsequential") is True


def test_actions_runs_last_returns_after_creating_run(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    # Make the background run fast.
    import videopipeline.studio.actions_api as actions_api
    from videopipeline.ingest.models import IngestResult

    dummy_src = tmp_path / "dummy.mp4"
    dummy_src.write_bytes(b"")

    monkeypatch.setattr(actions_api, "download_url", lambda url, *a, **k: IngestResult(video_path=dummy_src, url=url, title="dummy"))

    class DummyAnalysisResult:
        success = True
        error = None
        tasks_run = []
        total_elapsed_seconds = 0.0
        missing_targets = set()

    monkeypatch.setattr(actions_api, "run_analysis", lambda *a, **k: DummyAnalysisResult())

    r = client.post(
        "/api/actions/run_ingest_analyze",
        headers=hdr,
        json={"url": "https://www.twitch.tv/videos/123", "client_request_id": "run-req-2"},
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    r2 = client.get("/api/actions/runs/last", headers=hdr)
    assert r2.status_code == 200
    last = r2.json()
    assert last["run_id"] == run_id
    assert last["job_id"] == run_id


def test_actions_cancel_sets_cancel_requested(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    from videopipeline.studio.jobs import JOB_MANAGER

    job = JOB_MANAGER.create("dummy")
    r = client.post(f"/api/actions/jobs/{job.id}/cancel", headers=hdr, json={})
    assert r.status_code == 200

    j2 = JOB_MANAGER.get(job.id)
    assert j2 is not None
    assert j2.cancel_requested is True
