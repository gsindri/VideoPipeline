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


def test_actions_openapi_run_full_export_top_unattended_is_not_consequential(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/openapi.json", headers=hdr)
    assert r.status_code == 200
    spec = r.json()
    paths = spec.get("paths") or {}
    op = (((paths.get("/api/actions/run_full_export_top_unattended") or {}).get("post")) or {})
    assert op.get("x-openai-isConsequential") is False


def test_actions_openapi_includes_job_wait_endpoint(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/openapi.json", headers=hdr)
    assert r.status_code == 200
    spec = r.json()
    paths = spec.get("paths") or {}
    assert "/api/actions/jobs/{job_id}/wait" in paths


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


def test_actions_job_wait_returns_done_true_for_completed_job(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    from videopipeline.studio.jobs import JOB_MANAGER

    job = JOB_MANAGER.create("dummy")
    JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done")

    r = client.get(f"/api/actions/jobs/{job.id}/wait?timeout_s=0.1", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is True
    assert data["status"] == "succeeded"


def test_actions_analyze_full_external_llm_skips_local_llm(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import time
    import videopipeline.studio.actions_api as actions_api
    from videopipeline.studio.jobs import JOB_MANAGER

    pid = hashlib.sha256("twitch_999".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    video_path = proj_dir / "video" / "video.mp4"
    video_path.write_bytes(b"")

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(video_path), "duration_seconds": 60.0},
        "analysis": {"highlights": {"candidates": []}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    called = {"n": 0}

    def fake_get_llm_complete_fn(*a, **k):
        called["n"] += 1
        return lambda prompt: "ok"

    class DummyAnalysisResult:
        success = True
        error = None
        tasks_run = []
        total_elapsed_seconds = 0.0
        missing_targets = set()

    monkeypatch.setattr(actions_api, "get_llm_complete_fn", fake_get_llm_complete_fn)
    monkeypatch.setattr(actions_api, "run_analysis", lambda *a, **k: DummyAnalysisResult())

    r = client.post(
        "/api/actions/analyze_full",
        headers=hdr,
        json={"project_id": pid, "llm_mode": "external", "client_request_id": "analyze-ext-1"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # Wait for background job to finish
    for _ in range(200):
        job = JOB_MANAGER.get(job_id)
        if job and job.status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    job = JOB_MANAGER.get(job_id)
    assert job is not None
    assert job.status == "succeeded"
    assert called["n"] == 0


def test_actions_ai_candidates_returns_transcript_excerpt(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_234".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "video" / "video.mp4").write_bytes(b"")

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 60.0},
        "analysis": {
            "highlights": {
                "candidates": [
                    {"rank": 1, "candidate_id": "cid1", "score": 1.0, "start_s": 10.0, "end_s": 20.0, "peak_time_s": 15.0, "title": "cand 1"},
                ]
            }
        },
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    transcript_json = {
        "transcript": {
            "segments": [
                {"start": 9.0, "end": 12.0, "text": "a" * 150},
                {"start": 12.0, "end": 18.0, "text": "b" * 150},
            ],
            "duration_seconds": 60.0,
        }
    }
    (proj_dir / "analysis" / "transcript_full.json").write_text(json.dumps(transcript_json), encoding="utf-8")

    r = client.get(
        f"/api/actions/ai/candidates?project_id={pid}&top_n=1&window_s=1&max_chars=200&chat_lines=0",
        headers=hdr,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["project_id"] == pid
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["candidate_id"] == "cid1"
    excerpt = data["candidates"][0]["transcript_excerpt"]
    assert excerpt is not None
    assert isinstance(excerpt["text"], str)
    assert excerpt["truncated"] is True


def test_actions_ai_candidates_defaults_to_hybrid_top_plus_chat(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_234_hybrid".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "video" / "video.mp4").write_bytes(b"")

    candidates = []
    for rank in range(1, 51):
        # Keep global score descending by rank.
        score = 1000.0 - float(rank)
        # Make late-ranked items chat-heavy so they surface in chat bucket.
        chat_signal = float(rank - 30) if rank > 30 else 0.0
        candidates.append(
            {
                "rank": rank,
                "candidate_id": f"cid{rank}",
                "score": score,
                "start_s": float(rank),
                "end_s": float(rank + 6),
                "peak_time_s": float(rank + 2),
                "breakdown": {"chat": chat_signal},
                "raw_signals": {"chat": chat_signal},
            }
        )

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 120.0},
        "analysis": {"highlights": {"candidates": candidates}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    # Omit top_n/chat_top_n so endpoint applies the new hybrid defaults.
    r = client.get(f"/api/actions/ai/candidates?project_id={pid}&chat_lines=0", headers=hdr)
    assert r.status_code == 200
    data = r.json()

    assert data["limits"]["top_n"] == 30
    assert data["limits"]["chat_top_n"] == 15
    assert data["strategy"]["mode"] == "hybrid_top_plus_chat"
    assert len(data["candidates"]) == 45

    chat_spike = [c for c in data["candidates"] if c.get("selection_source") == "chat_spike"]
    assert len(chat_spike) == 15
    assert any(int(c.get("rank") or 0) > 30 for c in chat_spike)


def test_actions_ai_candidates_explicit_top_n_defaults_to_hybrid_and_supports_top_only_override(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_234_legacy".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "video" / "video.mp4").write_bytes(b"")

    candidates = []
    for rank in range(1, 7):
        candidates.append(
            {
                "rank": rank,
                "candidate_id": f"cid{rank}",
                "score": float(100 - rank),
                "start_s": float(rank),
                "end_s": float(rank + 5),
                "peak_time_s": float(rank + 2),
                "raw_signals": {"chat": float(100 if rank == 6 else 0)},
            }
        )

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 60.0},
        "analysis": {"highlights": {"candidates": candidates}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    # Explicit top_n now defaults to hybrid behavior (+chat bucket).
    r = client.get(
        f"/api/actions/ai/candidates?project_id={pid}&top_n=3&chat_lines=0",
        headers=hdr,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["limits"]["top_n"] == 3
    assert data["limits"]["chat_top_n"] == 15
    assert data["strategy"]["mode"] == "hybrid_top_plus_chat"
    assert len(data["candidates"]) == 6
    assert any(c.get("selection_source") == "chat_spike" for c in data["candidates"])

    # Explicit top-only override.
    r2 = client.get(
        f"/api/actions/ai/candidates?project_id={pid}&top_n=3&chat_top_n=0&chat_lines=0",
        headers=hdr,
    )
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["limits"]["top_n"] == 3
    assert data2["limits"]["chat_top_n"] == 0
    assert data2["strategy"]["mode"] == "top_only"
    assert len(data2["candidates"]) == 3
    assert all(c.get("selection_source") == "multi_signal" for c in data2["candidates"])

    # Alias compatibility for macro-style `chat_top`.
    r3 = client.get(
        f"/api/actions/ai/candidates?project_id={pid}&top_n=3&chat_top=1&chat_lines=0",
        headers=hdr,
    )
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3["limits"]["chat_top_n"] == 1
    assert data3["strategy"]["mode"] == "hybrid_top_plus_chat"
    assert len(data3["candidates"]) == 4


def test_actions_ai_variants_filters_and_limits(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_235".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 60.0},
        "analysis": {"highlights": {"candidates": [{"rank": 1, "candidate_id": "cid1"}]}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    variants_json = {
        "created_at": "now",
        "candidates": [
            {
                "candidate_rank": 1,
                "candidate_peak_time_s": 12.0,
                "variants": [
                    {"variant_id": "medium", "start_s": 10.0, "end_s": 30.0, "duration_s": 20.0},
                    {"variant_id": "long", "start_s": 8.0, "end_s": 40.0, "duration_s": 32.0},
                ],
            }
        ],
    }
    (proj_dir / "analysis" / "variants.json").write_text(json.dumps(variants_json), encoding="utf-8")

    r = client.get(
        f"/api/actions/ai/variants?project_id={pid}&candidate_ranks=1&max_variants=1",
        headers=hdr,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["project_id"] == pid
    assert len(data["candidates"]) == 1
    assert len(data["candidates"][0]["variants"]) == 1
    assert data["candidates"][0]["truncated"] is True
    assert data["candidates"][0]["candidate_id"] == "cid1"


def test_actions_ai_variants_supports_candidate_ids_param(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_235b".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 60.0},
        "analysis": {"highlights": {"candidates": [{"rank": 1, "candidate_id": "cid1"}]}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    variants_json = {
        "created_at": "now",
        "candidates": [
            {
                "candidate_rank": 1,
                "candidate_peak_time_s": 12.0,
                "variants": [
                    {"variant_id": "medium", "start_s": 10.0, "end_s": 30.0, "duration_s": 20.0},
                ],
            }
        ],
    }
    (proj_dir / "analysis" / "variants.json").write_text(json.dumps(variants_json), encoding="utf-8")

    r = client.get(
        f"/api/actions/ai/variants?project_id={pid}&candidate_ids=cid1&max_variants=8",
        headers=hdr,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["project_id"] == pid
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["candidate_rank"] == 1
    assert data["candidates"][0]["candidate_id"] == "cid1"


def test_actions_ai_apply_semantic_updates_project(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_236".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    proj_dir.mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 60.0},
        "analysis": {
            "highlights": {
                "candidates": [
                    {"rank": 1, "score": 1.0, "start_s": 10.0, "end_s": 20.0},
                    {"rank": 2, "candidate_id": "c2", "score": 0.9, "start_s": 30.0, "end_s": 40.0},
                ]
            }
        },
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    r = client.post(
        "/api/actions/ai/apply_semantic",
        headers=hdr,
        json={
            "project_id": pid,
            "client_request_id": "apply-sem-1",
            "items": [{"candidate_id": "c2", "semantic_score": 0.8, "reason": "Good moment", "best_quote": "wow", "keep": True}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["updated_count"] == 1

    updated = json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))
    c2 = updated["analysis"]["highlights"]["candidates"][1]
    assert c2["rank"] == 2
    assert c2["score_semantic"] == 0.8
    assert c2["llm_reason"] == "Good moment"
    assert c2["llm_quote"] == "wow"
    assert c2["ai"]["semantic_score"] == 0.8


def test_actions_ai_apply_director_picks_writes_director_json(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_237".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 60.0},
        "analysis": {
            "highlights": {
                "candidates": [
                    {"rank": 1, "candidate_id": "cid1", "score": 1.0, "start_s": 10.0, "end_s": 20.0, "peak_time_s": 15.0},
                ]
            }
        },
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    variants_json = {
        "created_at": "now",
        "candidates": [
            {
                "candidate_rank": 1,
                "candidate_peak_time_s": 15.0,
                "variants": [
                    {"variant_id": "medium", "start_s": 10.0, "end_s": 30.0, "duration_s": 20.0},
                ],
            }
        ],
    }
    (proj_dir / "analysis" / "variants.json").write_text(json.dumps(variants_json), encoding="utf-8")

    r = client.post(
        "/api/actions/ai/apply_director_picks",
        headers=hdr,
        json={
            "project_id": pid,
            "client_request_id": "apply-dir-1",
            "picks": [{"candidate_id": "cid1", "variant_id": "medium", "title": "t", "hook": "h", "description": "d", "hashtags": ["clips"]}],
        },
    )
    assert r.status_code == 200
    director_path = proj_dir / "analysis" / "director.json"
    assert director_path.exists()
    director = json.loads(director_path.read_text(encoding="utf-8"))
    assert director["pick_count"] == 1
    assert director["picks"][0]["candidate_rank"] == 1
    assert director["picks"][0]["variant_id"] == "medium"


def test_actions_openapi_export_director_picks_is_consequential(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/openapi.json", headers=hdr)
    assert r.status_code == 200
    spec = r.json()
    paths = spec.get("paths") or {}
    op = (((paths.get("/api/actions/export_director_picks") or {}).get("post")) or {})
    assert op.get("x-openai-isConsequential") is True


def test_actions_export_director_picks_creates_job(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import time
    from videopipeline.studio.jobs import JOB_MANAGER

    pid = hashlib.sha256("twitch_238".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "exports").mkdir(parents=True, exist_ok=True)
    video_path = proj_dir / "video" / "video.mp4"
    video_path.write_bytes(b"")

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(video_path), "duration_seconds": 60.0},
        "analysis": {"highlights": {"candidates": []}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    director_json = {
        "created_at": "now",
        "pick_count": 1,
        "picks": [
            {"rank": 1, "candidate_rank": 1, "variant_id": "medium", "start_s": 10.0, "end_s": 30.0, "duration_s": 20.0, "title": "t", "hook": "h", "confidence": 0.7},
        ],
    }
    (proj_dir / "analysis" / "director.json").write_text(json.dumps(director_json), encoding="utf-8")

    # Avoid running ffmpeg in tests.
    def fake_start_export(*, proj, selection, export_dir, **kwargs):
        job = JOB_MANAGER.create("export")
        JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result={"output": str(export_dir / "dummy.mp4")})
        return job

    monkeypatch.setattr(JOB_MANAGER, "start_export", fake_start_export)

    r = client.post(
        "/api/actions/export_director_picks",
        headers=hdr,
        json={"project_id": pid, "limit": 1, "client_request_id": "export-dir-1"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    for _ in range(200):
        job = JOB_MANAGER.get(job_id)
        if job and job.status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    job = JOB_MANAGER.get(job_id)
    assert job is not None
    assert job.status == "succeeded"


def test_actions_export_director_picks_accepts_legacy_pick_shape(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import time
    from videopipeline.studio.jobs import JOB_MANAGER

    pid = hashlib.sha256("twitch_239".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "exports").mkdir(parents=True, exist_ok=True)
    video_path = proj_dir / "video" / "video.mp4"
    video_path.write_bytes(b"")

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(video_path), "duration_seconds": 60.0},
        "analysis": {"highlights": {"candidates": []}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    # Legacy-ish pick: uses rank + best_variant_id instead of candidate_rank + variant_id.
    director_json = {
        "created_at": "now",
        "pick_count": 1,
        "picks": [
            {
                "rank": 2,
                "best_variant_id": "medium",
                "start_s": 12.0,
                "end_s": 28.0,
                "duration_s": 16.0,
                "title": "t",
                "hook": "h",
                "confidence": 0.8,
            },
        ],
    }
    (proj_dir / "analysis" / "director.json").write_text(json.dumps(director_json), encoding="utf-8")

    # Avoid running ffmpeg in tests.
    def fake_start_export(*, proj, selection, export_dir, **kwargs):
        job = JOB_MANAGER.create("export")
        JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result={"output": str(export_dir / "dummy.mp4")})
        return job

    monkeypatch.setattr(JOB_MANAGER, "start_export", fake_start_export)

    r = client.post(
        "/api/actions/export_director_picks",
        headers=hdr,
        json={"project_id": pid, "limit": 1, "client_request_id": "export-dir-legacy-1"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    for _ in range(200):
        job = JOB_MANAGER.get(job_id)
        if job and job.status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    job = JOB_MANAGER.get(job_id)
    assert job is not None
    assert job.status == "succeeded"
