import hashlib
import json
import sys
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _install_google_stubs() -> None:
    """Tests don't need real google-auth just to import publisher code."""
    if "google.auth.transport.requests" not in sys.modules:
        google_mod = sys.modules.setdefault("google", types.ModuleType("google"))
        auth_mod = sys.modules.setdefault("google.auth", types.ModuleType("google.auth"))
        transport_mod = sys.modules.setdefault("google.auth.transport", types.ModuleType("google.auth.transport"))
        requests_mod = types.ModuleType("google.auth.transport.requests")
        exceptions_mod = types.ModuleType("google.auth.exceptions")

        class Request:  # noqa: D401 - tiny stub for imports
            """Stub request transport."""

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
                self.token_uri = kwargs.get("token_uri")
                self.client_id = kwargs.get("client_id")
                self.client_secret = kwargs.get("client_secret")
                self.scopes = kwargs.get("scopes")

            def refresh(self, request):
                self.expired = False

        credentials_mod.Credentials = Credentials
        sys.modules["google.oauth2.credentials"] = credentials_mod
        setattr(oauth2_mod, "credentials", credentials_mod)
        setattr(google_mod, "oauth2", oauth2_mod)


_install_google_stubs()


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


def _make_client(tmp_path: Path, monkeypatch, *, token: str | None = None, profile_path: Path | None = None) -> TestClient:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VP_API_TOKEN", raising=False)

    if token is not None:
        monkeypatch.setenv("VP_API_TOKEN", token)

    projects_root = tmp_path / "outputs" / "projects"
    publisher_state_root = tmp_path / "publisher_state"
    accounts_json = publisher_state_root / "accounts.json"
    publisher_db = publisher_state_root / "publisher.sqlite"

    import videopipeline.project as project_mod
    import videopipeline.publisher.accounts as publisher_accounts_mod
    import videopipeline.publisher.jobs as publisher_jobs_mod
    import videopipeline.source_scout as source_scout_mod
    import videopipeline.studio.actions_api as actions_api
    from videopipeline.studio.app import create_app

    monkeypatch.setattr(project_mod, "default_projects_root", lambda: projects_root)
    monkeypatch.setattr(source_scout_mod, "default_projects_root", lambda: projects_root)
    monkeypatch.setattr(actions_api, "default_projects_root", lambda: projects_root)
    monkeypatch.setattr(publisher_accounts_mod, "accounts_path", lambda: accounts_json)
    monkeypatch.setattr(publisher_jobs_mod, "publisher_db_path", lambda: publisher_db)

    app = create_app(video_path=None, profile_path=profile_path)
    return TestClient(app)


def _wait_for_terminal_job_state(job_id: str, *, timeout_s: float = 2.0):
    from videopipeline.studio.jobs import JOB_MANAGER

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = JOB_MANAGER.get(job_id)
        if job and job.status in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    return JOB_MANAGER.get(job_id)


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


def test_actions_diagnostics_reports_profile_readiness(tmp_path, monkeypatch):
    profile_path = tmp_path / "gaming_nvidia.yaml"
    profile_path.write_text(
        """
studio:
  default_llm_mode: external_strict
analysis:
  fail_fast: true
  speech:
    backend: assemblyai
    strict: true
    diarize: true
    use_gpu: true
  diarization:
    enabled: false
  audio_events:
    backend: assemblyai
    strict: true
""".strip(),
        encoding="utf-8",
    )

    import videopipeline.analysis_audio_events as audio_events_mod
    import videopipeline.studio.actions_api as actions_api_mod

    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("AAI_API_KEY", raising=False)
    monkeypatch.setattr(
        actions_api_mod,
        "_diagnostics_transcription_backends",
        lambda: {
            "whispercpp": False,
            "whispercpp_gpu": False,
            "faster_whisper": False,
            "faster_whisper_gpu": False,
            "openai_whisper": True,
            "openai_whisper_gpu": True,
            "nemo_asr": False,
            "nemo_asr_gpu": False,
            "assemblyai": False,
            "assemblyai_gpu": False,
        },
    )
    monkeypatch.setattr(
        audio_events_mod,
        "check_assemblyai_audio_events_available",
        lambda explicit_key=None: (False, "ASSEMBLYAI_API_KEY not set"),
    )

    client = _make_client(tmp_path, monkeypatch, token="secret", profile_path=profile_path)
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/diagnostics", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["paths"]["profile_path"] == str(profile_path)
    assert data["profile"]["path"] == str(profile_path)
    assert data["profile"]["fail_fast"] is True
    assert data["profile"]["speech"]["backend"] == "assemblyai"
    assert data["profile"]["speech"]["ready"] is False
    assert "AssemblyAI" in data["profile"]["speech"]["reason"]
    assert data["profile"]["audio_events"]["backend"] == "assemblyai"
    assert data["profile"]["audio_events"]["ready"] is False
    assert data["profile"]["diarization"]["speech_enabled"] is True
    assert data["profile"]["diarization"]["requires_hf_token"] is False
    assert data["profile"]["readiness"]["ok"] is False
    assert data["llm"]["supported_modes"] == ["local", "external", "external_strict"]
    assert data["llm"]["aliases"]["gondull"] == "external_strict"
    assert data["llm"]["preferred_mode"] == "external_strict"
    assert data["llm"]["default_mode"] == "external_strict"
    assert data["llm"]["profile_external_ai_requirements"] == {
        "semantic": True,
        "chapters": True,
        "director": True,
    }
    assert data["publisher"]["summary"]["accounts_total"] == 0
    assert data["publisher"]["summary"]["ready_accounts"] == 0
    assert data["publisher"]["policy"]["default_privacy"] == "private"
    assert any("speech:" in issue for issue in data["profile"]["readiness"]["issues"])
    assert any("audio_events:" in issue for issue in data["profile"]["readiness"]["issues"])


def test_actions_diagnostics_reports_source_scout_readiness(tmp_path, monkeypatch):
    watchlist_dir = tmp_path / "sources"
    watchlist_dir.mkdir(parents=True, exist_ok=True)
    watchlist_path = watchlist_dir / "watchlist.yaml"
    watchlist_path.write_text(
        """
shadow_mode: true
sources:
  - id: rebbi-twitch
    label: Rebbi Twitch
    url: https://www.twitch.tv/rebbi/videos
    platform: twitch
    enabled: true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("VP_SOURCE_WATCHLIST", str(watchlist_path))

    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/diagnostics", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["source_scout"]["configured"] is True
    assert data["source_scout"]["ready"] is True
    assert data["source_scout"]["twitch_api_configured"] is False
    assert data["source_scout"]["twitch_provider_sources"] == 0
    assert data["source_scout"]["twitch_helix_ready_sources"] == 0
    assert data["source_scout"]["watchlist_path"] == str(watchlist_path)
    assert data["source_scout"]["shadow_mode"] is True
    assert data["source_scout"]["enabled_sources"] == 1
    assert data["source_scout"]["inbox_pending"] == 0


def test_actions_diagnostics_reports_twitch_provider_blockers(tmp_path, monkeypatch):
    watchlist_dir = tmp_path / "sources"
    watchlist_dir.mkdir(parents=True, exist_ok=True)
    watchlist_path = watchlist_dir / "watchlist.yaml"
    watchlist_path.write_text(
        """
shadow_mode: true
sources:
  - id: ludwig-twitch
    label: Ludwig Twitch
    provider: twitch_helix
    platform: twitch
    channel_login: ludwig
    enabled: true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("VP_SOURCE_WATCHLIST", str(watchlist_path))
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_API_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TWITCH_API_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TWITCH_APP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TWITCH_ACCESS_TOKEN", raising=False)

    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/diagnostics", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["source_scout"]["configured"] is True
    assert data["source_scout"]["ready"] is False
    assert data["source_scout"]["twitch_api_configured"] is False
    assert data["source_scout"]["twitch_provider_sources"] == 1
    assert data["source_scout"]["twitch_helix_ready_sources"] == 0
    assert data["source_scout"]["enabled_sources"] == 1
    assert data["source_scout"]["fetch_ready_sources"] == 0
    assert any("TWITCH_CLIENT_ID" in issue for issue in data["source_scout"]["issues"])


def test_actions_diagnostics_reports_twitch_provider_ready_when_credentials_present(tmp_path, monkeypatch):
    watchlist_dir = tmp_path / "sources"
    watchlist_dir.mkdir(parents=True, exist_ok=True)
    watchlist_path = watchlist_dir / "watchlist.yaml"
    watchlist_path.write_text(
        """
shadow_mode: true
sources:
  - id: ludwig-twitch
    label: Ludwig Twitch
    provider: twitch_helix
    platform: twitch
    channel_login: ludwig
    enabled: true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("VP_SOURCE_WATCHLIST", str(watchlist_path))
    monkeypatch.setenv("TWITCH_CLIENT_ID", "client-id")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "client-secret")
    monkeypatch.delenv("TWITCH_APP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TWITCH_ACCESS_TOKEN", raising=False)

    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/diagnostics", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["env"]["twitch_client_id"] is True
    assert data["env"]["twitch_client_secret"] is True
    assert data["env"]["twitch_app_access_token"] is False
    assert data["source_scout"]["twitch_api_configured"] is True
    assert data["source_scout"]["twitch_provider_sources"] == 1
    assert data["source_scout"]["twitch_helix_ready_sources"] == 1
    assert data["source_scout"]["ready"] is True


def test_actions_scout_inbox_add_and_list(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    add = client.post(
        "/api/actions/scout/inbox/add",
        headers=hdr,
        json={
            "url": "https://www.twitch.tv/videos/456",
            "title": "Rebbi clutch VOD",
            "notes": "Good candidate from app",
            "priority": 6,
            "tags": ["rebbi", "clutch"],
            "added_by": "gondull-platform",
        },
    )
    assert add.status_code == 200
    payload = add.json()
    assert payload["meta"]["created"] is True
    assert payload["entry"]["status"] == "pending"
    assert payload["entry"]["url"] == "https://www.twitch.tv/videos/456"

    listing = client.get("/api/actions/scout/inbox?status=pending", headers=hdr)
    assert listing.status_code == 200
    data = listing.json()
    assert data["meta"]["entry_count"] == 1
    assert data["entries"][0]["title"] == "Rebbi clutch VOD"
    assert data["entries"][0]["added_by"] == "gondull-platform"


def test_actions_scout_candidates_ranks_urls_from_watchlist(tmp_path, monkeypatch):
    watchlist_dir = tmp_path / "sources"
    watchlist_dir.mkdir(parents=True, exist_ok=True)
    watchlist_path = watchlist_dir / "watchlist.yaml"
    watchlist_path.write_text(
        """
shadow_mode: true
sources:
  - id: rebbi-twitch
    label: Rebbi Twitch
    url: https://www.twitch.tv/rebbi/videos
    platform: twitch
    enabled: true
    priority: 4
    recent_hours: 72
    max_candidates: 3
    min_duration_s: 1800
    max_duration_s: 14400
    title_include_any: [challenge, insane]
    title_exclude_any: [rerun]
    profile:
      category: opportunity
      clip_density_rating: 4
      style_fit_rating: 5
      saturation_rating: 2
      rights_risk_rating: 3
      rated_by: sindri
      rated_at: 2026-03-08
      notes: Strong fit with moderate competition.
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("VP_SOURCE_WATCHLIST", str(watchlist_path))

    existing_url = "https://www.twitch.tv/videos/111"
    existing_pid = hashlib.sha256("twitch_111".encode("utf-8")).hexdigest()
    existing_dir = tmp_path / "outputs" / "projects" / existing_pid
    existing_dir.mkdir(parents=True, exist_ok=True)
    (existing_dir / "project.json").write_text(
        json.dumps(
            {
                "project_id": existing_pid,
                "created_at": "now",
                "source": {
                    "source_url": existing_url,
                    "scout": {"source_id": "rebbi-twitch"},
                },
                "analysis": {
                    "highlights": {"candidates": [{"rank": 1}]},
                    "director": {"pick_count": 1},
                },
                "exports": [{"status": "succeeded"}],
                "layout": {},
                "selections": [],
            }
        ),
        encoding="utf-8",
    )

    import videopipeline.source_scout as source_scout_mod

    now_iso = "2026-03-08T01:00:00+00:00"

    def fake_fetch(source, *, limit):
        assert source["id"] == "rebbi-twitch"
        assert limit == 3
        return [
            {
                "url": existing_url,
                "title": "insane challenge vod",
                "duration_seconds": 7200.0,
                "published_at": now_iso,
                "channel_name": "Rebbi",
                "fetch_mode": "yt_dlp",
            },
            {
                "url": "https://www.twitch.tv/videos/222",
                "title": "insane challenge run",
                "duration_seconds": 6800.0,
                "published_at": now_iso,
                "channel_name": "Rebbi",
                "fetch_mode": "yt_dlp",
            },
            {
                "url": "https://www.twitch.tv/videos/333",
                "title": "rerun archive",
                "duration_seconds": 6800.0,
                "published_at": now_iso,
                "channel_name": "Rebbi",
                "fetch_mode": "yt_dlp",
            },
        ]

    monkeypatch.setattr(source_scout_mod, "fetch_source_entries", fake_fetch)

    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/scout/candidates?limit=5&per_source=3", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["watchlist_path"] == str(watchlist_path)
    assert data["meta"]["shadow_mode"] is True
    assert data["meta"]["candidate_count"] == 1
    assert data["diagnostics"]["ready"] is True
    assert data["recommended"]["url"] == "https://www.twitch.tv/videos/222"
    assert data["recommended"]["source_id"] == "rebbi-twitch"
    assert data["recommended"]["source_fetch_mode"] == "yt_dlp"
    assert data["recommended"]["history"]["source_project_count"] == 1
    assert data["recommended"]["source_profile"]["category"] == "opportunity"
    assert data["recommended"]["source_profile"]["judgments"]["clip_density_rating"] == 4
    assert data["sources"][0]["fetch_mode"] == "yt_dlp"
    assert data["sources"][0]["profile"]["recommendation"]["band"] in {"medium", "high"}
    assert data["skipped"]["already_processed"] == 1
    assert data["skipped"]["title_excluded"] == 1


def test_actions_scout_candidates_prefers_manual_inbox(tmp_path, monkeypatch):
    watchlist_dir = tmp_path / "sources"
    watchlist_dir.mkdir(parents=True, exist_ok=True)
    watchlist_path = watchlist_dir / "watchlist.yaml"
    watchlist_path.write_text(
        """
shadow_mode: true
sources:
  - id: rebbi-twitch
    label: Rebbi Twitch
    url: https://www.twitch.tv/rebbi/videos
    platform: twitch
    enabled: true
    priority: 3
    recent_hours: 72
    max_candidates: 1
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("VP_SOURCE_WATCHLIST", str(watchlist_path))

    import videopipeline.source_scout as source_scout_mod

    monkeypatch.setattr(
        source_scout_mod,
        "fetch_source_entries",
        lambda source, *, limit: [
            {
                "url": "https://www.twitch.tv/videos/999",
                "title": "watchlist candidate",
                "duration_seconds": 5400.0,
                "published_at": "2026-03-08T01:00:00+00:00",
            }
        ],
    )

    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    add = client.post(
        "/api/actions/scout/inbox/add",
        headers=hdr,
        json={
            "url": "https://www.twitch.tv/videos/456",
            "title": "Manual priority pick",
            "priority": 7,
            "added_by": "app",
            "notes": "Should outrank watchlist",
        },
    )
    assert add.status_code == 200
    inbox_id = add.json()["entry"]["inbox_id"]

    r = client.get("/api/actions/scout/candidates?limit=5&per_source=1", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["candidate_count"] == 2
    assert data["recommended"]["url"] == "https://www.twitch.tv/videos/456"
    assert data["recommended"]["selection_mode"] == "manual_inbox"
    assert data["recommended"]["inbox_id"] == inbox_id
    assert data["strategy"]["manual_inbox"]["pending_count"] == 1


def test_actions_scout_candidates_uses_wider_default_window(tmp_path, monkeypatch):
    import videopipeline.source_scout as source_scout_mod

    captured = {}

    def fake_build_source_scout_report(*, limit, per_source, **kwargs):
        captured["limit"] = limit
        captured["per_source"] = per_source
        return {
            "meta": {
                "generated_at": "2026-03-15T00:00:00+00:00",
                "watchlist_path": None,
                "shadow_mode": True,
                "enabled_source_count": 0,
                "candidate_count": 0,
                "chat_probe": {"status": "not_enough_candidates"},
            },
            "strategy": {
                "mode": "shadow",
                "per_source": per_source,
                "limit": limit,
                "ranking_factors": [],
                "chat_probe": {},
                "source_profile_model": {},
                "dedupe": {},
                "manual_inbox": {"inbox_path": None, "pending_count": 0},
            },
            "sources": [],
            "skipped": {},
            "recommended": None,
            "candidates": [],
        }

    monkeypatch.setattr(source_scout_mod, "build_source_scout_report", fake_build_source_scout_report)

    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/scout/candidates", headers=hdr)
    assert r.status_code == 200
    assert captured == {"limit": 60, "per_source": 20}


def test_actions_run_ingest_analyze_persists_scout_metadata(tmp_path, monkeypatch):
    import videopipeline.chat.downloader as chat_downloader
    import videopipeline.studio.actions_api as actions_api_mod
    from videopipeline.ingest.models import IngestResult

    dummy_src = tmp_path / "dummy.mp4"
    dummy_src.write_bytes(b"")

    def fake_download_url(url: str, *args, **kwargs):
        return IngestResult(video_path=dummy_src, url=url, title="dummy")

    monkeypatch.setattr(actions_api_mod, "download_url", fake_download_url)
    monkeypatch.setattr(
        chat_downloader,
        "download_chat",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no chat")),
    )

    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    url = "https://www.twitch.tv/videos/12345"
    add = client.post(
        "/api/actions/scout/inbox/add",
        headers=hdr,
        json={
            "url": url,
            "title": "Inbox candidate",
            "added_by": "app",
        },
    )
    assert add.status_code == 200
    inbox_id = add.json()["entry"]["inbox_id"]

    r = client.post(
        "/api/actions/ingest_url",
        headers=hdr,
        json={
            "url": url,
            "scout": {
                "inbox_id": inbox_id,
                "source_id": "rebbi-twitch",
                "source_label": "Rebbi Twitch",
                "candidate_id": "cand-222",
                "score": 0.84,
                "shadow_mode": True,
                "reasons": ["source priority 4", "duration_score=1.00"],
            },
        },
    )
    assert r.status_code == 200
    pid = r.json()["project_id"]
    project_json = json.loads(
        (tmp_path / "outputs" / "projects" / pid / "project.json").read_text(encoding="utf-8")
    )
    assert project_json["source"]["source_url"] == url
    assert project_json["source"]["scout"]["source_id"] == "rebbi-twitch"
    assert project_json["source"]["scout"]["candidate_id"] == "cand-222"
    assert project_json["source"]["scout"]["inbox_id"] == inbox_id
    assert project_json["source"]["scout"]["score"] == 0.84
    assert project_json["source"]["scout"]["shadow_mode"] is True
    assert project_json["source"]["scout"]["reasons"] == [
        "source priority 4",
        "duration_score=1.00",
    ]
    assert project_json["source"]["scout"]["selected_at"]

    inbox_listing = client.get("/api/actions/scout/inbox?status=selected", headers=hdr)
    assert inbox_listing.status_code == 200
    assert inbox_listing.json()["entries"][0]["inbox_id"] == inbox_id
    assert inbox_listing.json()["entries"][0]["project_id"] == pid


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


def test_actions_run_ingest_analyze_fresh_project_creates_new_project_for_same_url(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

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

    import videopipeline.chat.downloader as chat_downloader

    monkeypatch.setattr(
        chat_downloader,
        "download_chat",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no chat")),
    )

    first = client.post(
        "/api/actions/run_ingest_analyze",
        headers=hdr,
        json={"url": "https://www.twitch.tv/videos/123", "client_request_id": "run-req-base"},
    )
    assert first.status_code == 200
    first_project_id = first.json()["project_id"]

    second = client.post(
        "/api/actions/run_ingest_analyze",
        headers=hdr,
        json={
            "url": "https://www.twitch.tv/videos/123",
            "client_request_id": "run-req-fresh-1",
            "fresh_project": True,
        },
    )
    assert second.status_code == 200
    second_project_id = second.json()["project_id"]

    third = client.post(
        "/api/actions/run_ingest_analyze",
        headers=hdr,
        json={
            "url": "https://www.twitch.tv/videos/123",
            "client_request_id": "run-req-fresh-2",
            "fresh_project": True,
        },
    )
    assert third.status_code == 200
    third_project_id = third.json()["project_id"]

    assert second_project_id != first_project_id
    assert third_project_id != first_project_id
    assert third_project_id != second_project_id


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


def test_actions_openapi_publish_queue_is_consequential(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.get("/api/actions/openapi.json", headers=hdr)
    assert r.status_code == 200
    spec = r.json()
    paths = spec.get("paths") or {}
    op = (((paths.get("/api/actions/publish/queue") or {}).get("post")) or {})
    assert op.get("x-openai-isConsequential") is True
    client.close()


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


@pytest.mark.parametrize("llm_mode", ["external", "external_strict"])
def test_actions_analyze_full_external_llm_skips_local_llm(tmp_path, monkeypatch, llm_mode):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import videopipeline.studio.actions_api as actions_api

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
        json={"project_id": pid, "llm_mode": llm_mode, "client_request_id": f"analyze-{llm_mode}-1"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    job = _wait_for_terminal_job_state(job_id)
    assert job is not None
    assert job.status == "succeeded"
    assert called["n"] == 0


def test_actions_publish_accounts_reports_readiness(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import videopipeline.studio.actions_api as actions_api
    from videopipeline.publisher.account_auth import PublishAccountAuthStatus
    from videopipeline.publisher.accounts import AccountStore

    store = AccountStore()
    ready = store.add(platform="youtube", label="Ready Channel")
    missing = store.add(platform="youtube", label="Missing Tokens")

    monkeypatch.setattr(
        actions_api,
        "get_publish_account_auth",
        lambda account: (
            PublishAccountAuthStatus(ready=True, has_tokens=True, auth_state="ready")
            if account.id == ready.id
            else PublishAccountAuthStatus(
                ready=False,
                has_tokens=False,
                auth_state="missing_tokens",
                auth_error="No stored credentials are available for this account.",
            )
        ),
    )

    r = client.get("/api/actions/publish/accounts", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    accounts = {item["id"]: item for item in data["accounts"]}
    assert accounts[ready.id]["ready"] is True
    assert accounts[missing.id]["ready"] is False
    assert data["summary"]["accounts_total"] == 2
    assert data["summary"]["ready_accounts"] == 1
    assert data["policy"]["default_privacy"] == "private"
    client.close()


def test_actions_publish_accounts_reports_reconnect_required(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import videopipeline.studio.actions_api as actions_api
    from videopipeline.publisher.account_auth import PublishAccountAuthStatus
    from videopipeline.publisher.accounts import AccountStore

    account = AccountStore().add(platform="youtube", label="Rebbi")
    monkeypatch.setattr(
        actions_api,
        "get_publish_account_auth",
        lambda _account: PublishAccountAuthStatus(
            ready=False,
            has_tokens=True,
            auth_state="needs_reauth",
            auth_error="Google refresh token was rejected (invalid_grant). Reconnect the YouTube account.",
        ),
    )

    r = client.get("/api/actions/publish/accounts", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["accounts"][0]["id"] == account.id
    assert data["accounts"][0]["ready"] is False
    assert data["accounts"][0]["auth_state"] == "needs_reauth"
    assert "Reconnect the YouTube account" in data["accounts"][0]["auth_error"]
    assert data["summary"]["ready_accounts"] == 0
    client.close()


def test_actions_publish_exports_lists_project_exports(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_publish_exports".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "exports").mkdir(parents=True, exist_ok=True)
    video_path = proj_dir / "video" / "video.mp4"
    video_path.write_bytes(b"")
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "project_id": pid,
                "created_at": "now",
                "video": {"path": str(video_path), "duration_seconds": 90.0},
                "analysis": {"highlights": {"candidates": []}},
                "layout": {},
                "selections": [],
                "exports": [],
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "exports" / "clip1.mp4").write_bytes(b"fake video")
    (proj_dir / "exports" / "clip1.json").write_text(
        json.dumps({"title": "Clip 1", "description": "desc", "privacy": "unlisted"}),
        encoding="utf-8",
    )

    r = client.get(f"/api/actions/publish/exports?project_id={pid}", headers=hdr)
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["project_id"] == pid
    assert len(data["exports"]) == 1
    assert data["exports"][0]["export_id"] == "clip1"
    assert data["exports"][0]["privacy"] == "unlisted"
    client.close()


@pytest.mark.parametrize(
    ("options", "expected_issue"),
    [
        ({"privacy": "public"}, "privacy=public"),
        ({"publish_at": "2026-03-10T12:00:00Z"}, "publish_at"),
    ],
)
def test_actions_publish_queue_requires_public_release_approval(tmp_path, monkeypatch, options, expected_issue):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import videopipeline.studio.actions_api as actions_api
    from videopipeline.publisher.account_auth import PublishAccountAuthStatus
    from videopipeline.publisher.accounts import AccountStore

    store = AccountStore()
    account = store.add(platform="youtube", label="Main Channel")
    monkeypatch.setattr(
        actions_api,
        "get_publish_account_auth",
        lambda _account: PublishAccountAuthStatus(ready=True, has_tokens=True, auth_state="ready"),
    )

    pid = hashlib.sha256("twitch_publish_approval".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "exports").mkdir(parents=True, exist_ok=True)
    video_path = proj_dir / "video" / "video.mp4"
    video_path.write_bytes(b"")
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "project_id": pid,
                "created_at": "now",
                "video": {"path": str(video_path), "duration_seconds": 90.0},
                "analysis": {"highlights": {"candidates": []}},
                "layout": {},
                "selections": [],
                "exports": [],
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "exports" / "clip1.mp4").write_bytes(b"fake video")
    (proj_dir / "exports" / "clip1.json").write_text(json.dumps({"title": "Clip 1"}), encoding="utf-8")

    r = client.post(
        "/api/actions/publish/queue",
        headers=hdr,
        json={
            "project_id": pid,
            "account_ids": [account.id],
            "export_ids": ["clip1"],
            "options": options,
            "client_request_id": f"publish-approval-{expected_issue}",
        },
    )
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "public_release_approval_required"
    assert expected_issue in detail["issues"]
    client.close()


def test_actions_publish_queue_creates_jobs_and_lists_them(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import videopipeline.studio.actions_api as actions_api
    from videopipeline.publisher.account_auth import PublishAccountAuthStatus
    from videopipeline.publisher.accounts import AccountStore
    from videopipeline.publisher.jobs import PublishJobStore

    store = AccountStore()
    account = store.add(platform="youtube", label="Main Channel")
    monkeypatch.setattr(
        actions_api,
        "get_publish_account_auth",
        lambda _account: PublishAccountAuthStatus(ready=True, has_tokens=True, auth_state="ready"),
    )

    pid = hashlib.sha256("twitch_publish_queue".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "exports").mkdir(parents=True, exist_ok=True)
    video_path = proj_dir / "video" / "video.mp4"
    video_path.write_bytes(b"")
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "project_id": pid,
                "created_at": "now",
                "video": {"path": str(video_path), "duration_seconds": 90.0},
                "analysis": {"highlights": {"candidates": []}},
                "layout": {},
                "selections": [],
                "exports": [],
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "exports" / "clip1.mp4").write_bytes(b"fake video")
    (proj_dir / "exports" / "clip1.json").write_text(
        json.dumps({"title": "Clip 1", "description": "desc"}),
        encoding="utf-8",
    )

    r = client.post(
        "/api/actions/publish/queue",
        headers=hdr,
        json={
            "project_id": pid,
            "account_ids": [account.id],
            "export_ids": ["clip1"],
            "options": {"privacy": "unlisted"},
            "client_request_id": "publish-queue-1",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["project_id"] == pid
    assert data["total"] == 1
    assert data["jobs"][0]["privacy"] == "unlisted"

    job_id = data["jobs"][0]["job_id"]
    job = PublishJobStore().get_job(job_id)
    assert job.status == "queued"
    assert job.account_id == account.id

    jobs_r = client.get(f"/api/actions/publish/jobs?project_id={pid}&limit=10", headers=hdr)
    assert jobs_r.status_code == 200
    jobs_data = jobs_r.json()
    assert jobs_data["project_id"] == pid
    assert any(item["id"] == job_id for item in jobs_data["jobs"])
    client.close()


def test_actions_publish_delete_remote_requires_confirmation(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    from videopipeline.publisher.accounts import AccountStore
    from videopipeline.publisher.jobs import PublishJobStore

    account = AccountStore().add(platform="youtube", label="Main Channel")
    video_path = tmp_path / "outputs" / "projects" / "project-a" / "exports" / "clip1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake video")
    metadata_path = video_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"title": "Clip 1"}), encoding="utf-8")

    job = PublishJobStore().create_job(
        job_id="job-delete-1",
        platform="youtube",
        account_id=account.id,
        file_path=str(video_path),
        metadata_path=str(metadata_path),
    )
    PublishJobStore().update_job(
        job.id,
        status="succeeded",
        progress=1.0,
        remote_id="video123",
        remote_url="https://youtu.be/video123",
    )

    r = client.post(
        f"/api/actions/publish/jobs/{job.id}/delete-remote",
        headers=hdr,
        json={"confirmed": False},
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "delete_remote_confirmation_required"
    client.close()


def test_actions_publish_delete_remote_marks_job_removed_and_clears_dedup(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import videopipeline.studio.actions_api as actions_api
    from videopipeline.publisher.accounts import AccountStore
    from videopipeline.publisher.jobs import PublishJobStore

    account = AccountStore().add(platform="youtube", label="Main Channel")
    video_path = tmp_path / "outputs" / "projects" / "project-b" / "exports" / "clip1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake video")
    metadata_path = video_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"title": "Clip 1"}), encoding="utf-8")

    store = PublishJobStore()
    job = store.create_job(
        job_id="job-delete-2",
        platform="youtube",
        account_id=account.id,
        file_path=str(video_path),
        metadata_path=str(metadata_path),
    )
    updated = store.update_job(
        job.id,
        status="succeeded",
        progress=1.0,
        remote_id="video456",
        remote_url="https://youtu.be/video456",
    )
    sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
    store.mark_dedup("youtube", account.id, sha, "video456", "https://youtu.be/video456")

    deleted = {}

    class DummyConnector:
        def delete_remote(self, *, remote_id: str) -> None:
            deleted["remote_id"] = remote_id

    monkeypatch.setattr(actions_api, "load_tokens", lambda platform, account_id: {"token": "x"})
    monkeypatch.setattr(
        actions_api,
        "get_connector",
        lambda platform, account, tokens: DummyConnector(),
    )

    r = client.post(
        f"/api/actions/publish/jobs/{updated.id}/delete-remote",
        headers=hdr,
        json={"confirmed": True},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["ok"] is True
    assert payload["removed_remote_id"] == "video456"
    assert payload["removed_remote_url"] == "https://youtu.be/video456"
    assert payload["dedup_removed"] is True
    assert deleted == {"remote_id": "video456"}

    final_job = store.get_job(updated.id)
    assert final_job.status == "removed"
    assert final_job.remote_id == "video456"
    assert final_job.remote_url is None
    assert final_job.last_error == "Remote video removed from platform."
    assert store.lookup_dedup("youtube", account.id, sha) is None
    client.close()


def test_actions_publish_queue_rejects_accounts_that_need_reauth(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    import videopipeline.studio.actions_api as actions_api
    from videopipeline.publisher.account_auth import PublishAccountAuthStatus
    from videopipeline.publisher.accounts import AccountStore

    account = AccountStore().add(platform="youtube", label="Main Channel")
    monkeypatch.setattr(
        actions_api,
        "get_publish_account_auth",
        lambda _account: PublishAccountAuthStatus(
            ready=False,
            has_tokens=True,
            auth_state="needs_reauth",
            auth_error="Google refresh token was rejected (invalid_grant). Reconnect the YouTube account.",
        ),
    )

    pid = hashlib.sha256("twitch_publish_reauth".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "exports").mkdir(parents=True, exist_ok=True)
    video_path = proj_dir / "video" / "video.mp4"
    video_path.write_bytes(b"")
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "project_id": pid,
                "created_at": "now",
                "video": {"path": str(video_path), "duration_seconds": 90.0},
                "analysis": {"highlights": {"candidates": []}},
                "layout": {},
                "selections": [],
                "exports": [],
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "exports" / "clip1.mp4").write_bytes(b"fake video")
    (proj_dir / "exports" / "clip1.json").write_text(json.dumps({"title": "Clip 1"}), encoding="utf-8")

    r = client.post(
        "/api/actions/publish/queue",
        headers=hdr,
        json={
            "project_id": pid,
            "account_ids": [account.id],
            "export_ids": ["clip1"],
            "client_request_id": "publish-needs-reauth",
        },
    )
    assert r.status_code == 409
    assert r.json()["detail"] == {
        "code": "account_needs_reauth",
        "account_id": account.id,
        "platform": "youtube",
        "auth_state": "needs_reauth",
        "auth_error": "Google refresh token was rejected (invalid_grant). Reconnect the YouTube account.",
    }
    client.close()


def test_actions_ai_bundle_reports_external_status(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_bundle".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "video" / "video.mp4").write_bytes(b"")

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 90.0},
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
    (proj_dir / "analysis" / "variants.json").write_text(
        json.dumps(
            {
                "created_at": "now",
                "candidates": [
                    {
                        "candidate_rank": 1,
                        "candidate_id": "cid1",
                        "candidate_peak_time_s": 15.0,
                        "variants": [
                            {"variant_id": "medium", "start_s": 10.0, "end_s": 28.0, "duration_s": 18.0},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "analysis" / "chapters.json").write_text(
        json.dumps(
            {
                "generated_at": "now",
                "chapter_count": 1,
                "chapters": [{"id": 0, "start_s": 0.0, "end_s": 90.0, "title": "Chapter 1", "summary": "", "keywords": []}],
            }
        ),
        encoding="utf-8",
    )

    r = client.get(
        f"/api/actions/ai/bundle?project_id={pid}&top_n=1&chat_top_n=0&chapter_limit=10",
        headers=hdr,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["workflow"]["preferred_llm_mode"] == "external_strict"
    assert data["workflow"]["external_ai_status"]["required"]["semantic"] is True
    assert data["workflow"]["external_ai_status"]["completed"]["semantic"] is False
    assert data["workflow"]["external_ai_status"]["completed"]["director"] is False
    assert data["workflow"]["chapters_available"] is True
    assert data["workflow"]["next_actions"]["clip_review"] == "GET /api/actions/ai/clip_review"
    assert len(data["candidates"]) == 1
    assert len(data["variants"]) == 1
    assert len(data["chapters"]) == 1
    assert data["clip_review"]["meta"]["clip_count"] == 1
    assert data["clip_review"]["channel_format_spec"]["channel_positioning"]["rule"].startswith(
        "Treat streamer clips as raw evidence"
    )
    assert data["clip_review"]["channel_format_spec"]["launch_stack"] == ["clip_court", "weekly_awards"]
    assert data["clip_review"]["channel_format_spec"]["show_formats"][0]["show_format_id"] == "clip_court"
    assert (
        data["clip_review"]["review_rubric"]["preferred_outputs"]["director"]
        == "variant_id + optional start_s/end_s override + format_id + show_format_id + title + hook + description + hashtags + confidence"
    )
    assert data["clip_review"]["clips"][0]["candidate"]["candidate_id"] == "cid1"
    assert data["clip_review"]["clips"][0]["format_recommendations"][0]["format_id"] == "commentary_breakdown"
    assert data["clip_review"]["clips"][0]["show_format_recommendations"][0]["show_format_id"] == "clip_court"


def test_actions_ai_clip_review_joins_director_exports_and_chapters(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_clip_review".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "exports").mkdir(parents=True, exist_ok=True)
    (proj_dir / "video" / "video.mp4").write_bytes(b"")

    export_path = proj_dir / "exports" / "clip1.mp4"
    export_path.write_bytes(b"fake video bytes")
    (proj_dir / "exports" / "clip1.metadata.json").write_text(
        json.dumps(
            {
                "title": "Rebbi Test Clip",
                "description": "Packaged export metadata",
                "template": "vertical_blur",
                "duration_seconds": 19.5,
                "privacy": "unlisted",
            }
        ),
        encoding="utf-8",
    )

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 120.0},
        "analysis": {
            "highlights": {
                "candidates": [
                    {
                        "rank": 1,
                        "candidate_id": "cid1",
                        "score": 1.0,
                        "start_s": 40.0,
                        "end_s": 58.0,
                        "peak_time_s": 50.0,
                        "title": "Boss fight clutch",
                        "hook_text": "no way",
                    },
                ]
            }
        },
        "layout": {},
        "selections": [
            {
                "id": "sel1",
                "created_at": "now",
                "start_s": 40.0,
                "end_s": 59.5,
                "title": "Boss fight clutch",
                "template": "vertical_blur",
                "candidate_rank": 1,
                "candidate_peak_time_s": 50.0,
                "variant_id": "medium",
                "director_confidence": 0.91,
            }
        ],
        "exports": [
            {
                "created_at": "now",
                "selection_id": "sel1",
                "output": str(export_path),
                "template": "vertical_blur",
                "with_captions": False,
                "status": "succeeded",
            }
        ],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")
    (proj_dir / "analysis" / "variants.json").write_text(
        json.dumps(
            {
                "created_at": "now",
                "candidates": [
                    {
                        "candidate_rank": 1,
                        "candidate_id": "cid1",
                        "candidate_peak_time_s": 50.0,
                        "variants": [
                            {
                                "variant_id": "medium",
                                "start_s": 40.0,
                                "end_s": 59.5,
                                "duration_s": 19.5,
                                "description": "Best cut",
                            },
                            {
                                "variant_id": "long",
                                "start_s": 35.0,
                                "end_s": 65.0,
                                "duration_s": 30.0,
                                "description": "Longer setup",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "analysis" / "chapters.json").write_text(
        json.dumps(
            {
                "generated_at": "now",
                "chapters": [
                    {"id": 0, "start_s": 0.0, "end_s": 30.0, "title": "Setup", "summary": "Warmup", "keywords": ["warmup"]},
                    {"id": 1, "start_s": 30.0, "end_s": 70.0, "title": "Boss fight", "summary": "Main action", "keywords": ["boss"]},
                    {"id": 2, "start_s": 70.0, "end_s": 120.0, "title": "Aftermath", "summary": "Wrap-up", "keywords": ["ending"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "analysis" / "director.json").write_text(
        json.dumps(
            {
                "created_at": "now",
                "provenance": {"source": "gondull", "model": "gpt-5.3"},
                "config": {"source": "gondull"},
                "pick_count": 1,
                "picks": [
                    {
                        "rank": 1,
                        "candidate_rank": 1,
                        "candidate_id": "cid1",
                        "variant_id": "medium",
                        "start_s": 40.0,
                        "end_s": 59.5,
                        "duration_s": 19.5,
                        "title": "REBBI WINS",
                        "hook": "NO WAY",
                        "description": "Clutch finish",
                        "hashtags": ["gaming", "clips", "shorts"],
                        "template": "vertical_blur",
                        "confidence": 0.91,
                        "reasons": ["clean payoff"],
                        "chapter_index": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    r = client.get(
        f"/api/actions/ai/clip_review?project_id={pid}&top_n=1&chat_top_n=0&chapter_limit=10",
        headers=hdr,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["clip_count"] == 1
    assert data["meta"]["export_count"] == 1
    assert data["workflow"]["external_ai_status"]["completed"]["director"] is True
    clip = data["clips"][0]
    assert clip["review_id"] == "cid1"
    assert clip["candidate"]["candidate_id"] == "cid1"
    assert clip["primary_variant"]["variant_id"] == "medium"
    assert clip["primary_variant_source"] == "director_pick"
    assert clip["director_pick"]["variant_id"] == "medium"
    assert clip["director_pick"]["title"] == "REBBI WINS"
    assert clip["chapter_context"]["current"]["title"] == "Boss fight"
    assert clip["chapter_context"]["previous"]["title"] == "Setup"
    assert clip["chapter_context"]["next"]["title"] == "Aftermath"
    assert clip["selections"][0]["id"] == "sel1"
    assert clip["selections"][0]["variant_id"] == "medium"
    assert clip["exports"][0]["export_id"] == "clip1"
    assert clip["exports"][0]["title"] == "Rebbi Test Clip"
    assert clip["exports"][0]["privacy"] == "unlisted"
    assert clip["exports"][0]["selection_id"] == "sel1"
    assert clip["status"]["has_export"] is True
    assert data["channel_format_spec"]["formats"][0]["format_id"] == "commentary_breakdown"
    assert data["channel_format_spec"]["show_formats"][1]["show_format_id"] == "weekly_awards"
    assert clip["format_recommendations"][0]["format_id"] == "commentary_breakdown"
    assert "export_note" in clip["format_recommendations"][0]
    assert clip["show_format_recommendations"][0]["show_format_id"] == "clip_court"
    assert "export_note" in clip["show_format_recommendations"][0]


def test_actions_ai_clip_review_recommends_weekly_awards_for_multi_clip_shortlist(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_clip_review_multi".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (proj_dir / "video" / "video.mp4").write_bytes(b"")

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 180.0},
        "analysis": {
            "highlights": {
                "candidates": [
                    {
                        "rank": 1,
                        "candidate_id": "cid1",
                        "score": 1.0,
                        "start_s": 15.0,
                        "end_s": 34.0,
                        "peak_time_s": 23.0,
                        "title": "Clip one",
                    },
                    {
                        "rank": 2,
                        "candidate_id": "cid2",
                        "score": 0.9,
                        "start_s": 60.0,
                        "end_s": 84.0,
                        "peak_time_s": 72.0,
                        "title": "Clip two",
                    },
                ]
            }
        },
        "layout": {},
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")
    (proj_dir / "analysis" / "variants.json").write_text(
        json.dumps(
            {
                "created_at": "now",
                "candidates": [
                    {
                        "candidate_rank": 1,
                        "candidate_id": "cid1",
                        "candidate_peak_time_s": 23.0,
                        "variants": [
                            {"variant_id": "tight", "start_s": 15.0, "end_s": 34.0, "duration_s": 19.0}
                        ],
                    },
                    {
                        "candidate_rank": 2,
                        "candidate_id": "cid2",
                        "candidate_peak_time_s": 72.0,
                        "variants": [
                            {"variant_id": "tight", "start_s": 60.0, "end_s": 84.0, "duration_s": 24.0}
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "analysis" / "chapters.json").write_text(
        json.dumps(
            {
                "generated_at": "now",
                "chapters": [
                    {"id": 0, "start_s": 0.0, "end_s": 90.0, "title": "Weekly chaos", "summary": "", "keywords": []},
                    {"id": 1, "start_s": 90.0, "end_s": 180.0, "title": "Aftershow", "summary": "", "keywords": []},
                ],
            }
        ),
        encoding="utf-8",
    )

    r = client.get(
        f"/api/actions/ai/clip_review?project_id={pid}&top_n=2&chat_top_n=0&chapter_limit=10",
        headers=hdr,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["clip_count"] == 2
    first = data["clips"][0]
    second = data["clips"][1]
    assert first["show_format_recommendations"][0]["show_format_id"] == "weekly_awards"
    assert first["show_format_recommendations"][0]["format_id"] == "ranked_roundup"
    assert second["show_format_recommendations"][0]["show_format_id"] == "weekly_awards"
    assert second["show_format_recommendations"][1]["show_format_id"] == "clip_court"


def test_actions_ai_clip_review_includes_visual_frames_when_requested(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_clip_review_visual".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "video").mkdir(parents=True, exist_ok=True)
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)
    video_path = proj_dir / "video" / "video.mp4"
    video_path.write_bytes(b"fake mp4 bytes")

    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "project_id": pid,
                "created_at": "now",
                "video": {"path": str(video_path), "duration_seconds": 90.0},
                "analysis": {
                    "highlights": {
                        "candidates": [
                            {
                                "rank": 1,
                                "candidate_id": "cid-visual",
                                "score": 0.99,
                                "start_s": 12.0,
                                "end_s": 27.0,
                                "peak_time_s": 21.0,
                                "title": "Visual beat",
                            }
                        ]
                    }
                },
                "layout": {},
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "analysis" / "variants.json").write_text(
        json.dumps(
            {
                "created_at": "now",
                "candidates": [
                    {
                        "candidate_rank": 1,
                        "candidate_id": "cid-visual",
                        "candidate_peak_time_s": 21.0,
                        "variants": [
                            {"variant_id": "tight", "start_s": 12.0, "end_s": 27.0, "duration_s": 15.0}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "analysis" / "chapters.json").write_text(
        json.dumps(
            {
                "generated_at": "now",
                "chapters": [
                    {"id": 0, "start_s": 0.0, "end_s": 90.0, "title": "Main segment", "summary": "", "keywords": []}
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_extract_video_frame_jpeg(video_path, *, output_path, time_seconds, width=None, quality=4):
        del video_path, time_seconds, width, quality
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\xff\xd8\xff\xdbfake-jpeg")
        return output_path

    monkeypatch.setattr(
        "videopipeline.studio.actions_api.extract_video_frame_jpeg",
        _fake_extract_video_frame_jpeg,
    )

    r = client.get(
        (
            f"/api/actions/ai/clip_review?project_id={pid}&top_n=1&chat_top_n=0"
            "&chapter_limit=10&frame_clip_limit=1&frames_per_clip=3"
        ),
        headers=hdr,
    )
    assert r.status_code == 200
    data = r.json()
    clip = data["clips"][0]
    assert clip["status"]["has_visual_frames"] is True
    assert clip["visual_review"]["frame_source"] == "source_video"
    assert clip["visual_review"]["frame_count"] == 3
    assert len(clip["visual_review"]["frames"]) == 3
    first_frame = clip["visual_review"]["frames"][0]
    assert first_frame["mime_type"] == "image/jpeg"
    assert first_frame["file_name"].endswith(".jpg")
    assert first_frame["relative_path"].replace("\\", "/").startswith(
        "analysis/clip_review_frames/cid-visual/"
    )
    assert first_frame["content_base64"] == " /9j/22Zha2UtanBlZw==".strip()
    assert data["limits"]["visual_review"]["frame_clip_limit"] == 1
    assert data["limits"]["visual_review"]["frames_per_clip"] == 3


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
    profile_path = tmp_path / "semantic_weight.yaml"
    profile_path.write_text(
        """
analysis:
  highlights:
    llm_semantic_weight: 0.5
""".strip(),
        encoding="utf-8",
    )
    client = _make_client(tmp_path, monkeypatch, token="secret", profile_path=profile_path)
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_236".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)

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
    (proj_dir / "analysis" / "highlights.json").write_text(
        json.dumps(
            {
                "created_at": "now",
                "signals_used": {"speech": True},
                "candidates": [
                    {"rank": 1, "candidate_id": "c1", "score": 1.0, "start_s": 10.0, "end_s": 20.0},
                    {"rank": 2, "candidate_id": "c2", "score": 0.9, "start_s": 30.0, "end_s": 40.0},
                ],
            }
        ),
        encoding="utf-8",
    )

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
    assert data["candidate_count_before"] == 2
    assert data["candidate_count_after"] == 2
    assert data["dropped_count"] == 0
    assert data["semantic_scored_count"] == 1

    updated = json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))
    candidates = updated["analysis"]["highlights"]["candidates"]
    c2 = candidates[0]
    assert c2["candidate_id"] == "c2"
    assert c2["rank"] == 1
    assert c2["score_signal"] == 0.9
    assert c2["score_semantic"] == 0.8
    assert c2["score"] == pytest.approx(1.05)
    assert c2["llm_reason"] == "Good moment"
    assert c2["llm_quote"] == "wow"
    assert c2["ai"]["semantic_score"] == 0.8
    assert updated["analysis"]["highlights"]["signals_used"]["llm_semantic"] is True
    assert updated["analysis"]["actions"]["semantic_stats"]["semantic_scored_count"] == 1

    highlights = json.loads((proj_dir / "analysis" / "highlights.json").read_text(encoding="utf-8"))
    assert highlights["candidates"][0]["candidate_id"] == "c2"
    assert highlights["signals_used"]["llm_semantic"] is True


def test_actions_ai_apply_semantic_filters_rejected_candidates(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_sem_filter".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    proj_dir.mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 60.0},
        "analysis": {
            "highlights": {
                "candidates": [
                    {"rank": 1, "candidate_id": "c1", "score": 1.2, "start_s": 10.0, "end_s": 20.0},
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
            "client_request_id": "apply-sem-filter-1",
            "items": [{"candidate_id": "c1", "semantic_score": 0.1, "reason": "no payoff", "keep": False}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["candidate_count_before"] == 2
    assert data["candidate_count_after"] == 1
    assert data["dropped_count"] == 1

    updated = json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))
    candidates = updated["analysis"]["highlights"]["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == "c2"
    assert candidates[0]["rank"] == 1
    assert updated["analysis"]["actions"]["semantic_stats"]["dropped_count"] == 1


def test_actions_ai_apply_semantic_persists_provenance(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_sem_prov".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    proj_dir.mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 60.0},
        "analysis": {"highlights": {"candidates": [{"rank": 1, "candidate_id": "c1", "score": 1.0, "start_s": 10.0, "end_s": 20.0}]}},
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
            "client_request_id": "apply-sem-prov-1",
            "provenance": {
                "agent": "gondull",
                "provider": "openai",
                "model": "gpt-5.3-xhigh",
                "prompt_version": "semantic-v1",
            },
            "items": [{"candidate_id": "c1", "semantic_score": 0.91, "reason": "clear payoff"}],
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["provenance"]["agent"] == "gondull"
    assert payload["provenance"]["model"] == "gpt-5.3-xhigh"

    updated = json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))
    cand = updated["analysis"]["highlights"]["candidates"][0]
    assert cand["ai"]["semantic_source"] == "chatgpt_actions"
    assert cand["ai"]["semantic_provenance"]["agent"] == "gondull"
    assert updated["analysis"]["actions"]["semantic_provenance"]["prompt_version"] == "semantic-v1"


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


def test_actions_ai_apply_director_picks_persists_camera_plan(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_237_camera_plan".encode("utf-8")).hexdigest()
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
            "client_request_id": "apply-dir-camera-plan-1",
            "picks": [
                {
                    "candidate_id": "cid1",
                    "variant_id": "medium",
                    "title": "t",
                    "hook": "h",
                    "description": "d",
                    "camera_plan": [
                        {"at_s": 0.0, "focus_x": 0.45, "focus_y": 0.35, "zoom": 1.0},
                        {"at_s": 3.0, "focus_x": 0.66, "focus_y": 0.52, "zoom": 1.8},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 200
    director = json.loads((proj_dir / "analysis" / "director.json").read_text(encoding="utf-8"))
    assert director["picks"][0]["camera_plan"] == [
        {"at_s": 0.0, "focus_x": 0.45, "focus_y": 0.35, "zoom": 1.0},
        {"at_s": 3.0, "focus_x": 0.66, "focus_y": 0.52, "zoom": 1.8},
    ]


def test_actions_ai_apply_director_picks_allows_exact_timing_overrides(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_237_custom_cut".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 120.0},
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
            "client_request_id": "apply-dir-custom-cut-1",
            "picks": [
                {
                    "candidate_id": "cid1",
                    "variant_id": "medium",
                    "start_s": 12.0,
                    "end_s": 36.0,
                    "title": "t",
                    "hook": "h",
                    "description": "d",
                }
            ],
        },
    )
    assert r.status_code == 200
    director = json.loads((proj_dir / "analysis" / "director.json").read_text(encoding="utf-8"))
    assert director["picks"][0]["variant_id"] == "medium"
    assert director["picks"][0]["start_s"] == 12.0
    assert director["picks"][0]["end_s"] == 36.0
    assert director["picks"][0]["duration_s"] == 24.0
    assert director["picks"][0]["timing_source"] == "custom_override"


def test_actions_ai_apply_director_picks_enforces_overlap_and_normalizes_packaging(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_237_overlap".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 120.0},
        "analysis": {
            "highlights": {
                "candidates": [
                    {"rank": 1, "candidate_id": "cid1", "score": 1.0, "start_s": 10.0, "end_s": 20.0, "peak_time_s": 15.0},
                    {"rank": 2, "candidate_id": "cid2", "score": 0.9, "start_s": 12.0, "end_s": 22.0, "peak_time_s": 17.0},
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
            },
            {
                "candidate_rank": 2,
                "candidate_peak_time_s": 17.0,
                "variants": [
                    {"variant_id": "medium", "start_s": 12.0, "end_s": 32.0, "duration_s": 20.0},
                ],
            },
        ],
    }
    (proj_dir / "analysis" / "variants.json").write_text(json.dumps(variants_json), encoding="utf-8")

    r = client.post(
        "/api/actions/ai/apply_director_picks",
        headers=hdr,
        json={
            "project_id": pid,
            "client_request_id": "apply-dir-overlap-1",
            "picks": [
                {
                    "candidate_id": "cid1",
                    "variant_id": "medium",
                    "title": "A" * 120,
                    "hook": "H" * 140,
                    "description": "D" * 420,
                    "hashtags": ["clips"],
                    "template": "bad_template_name",
                },
                {
                    "candidate_id": "cid2",
                    "variant_id": "medium",
                    "title": "second",
                },
            ],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["pick_count"] == 1
    assert any(m.get("error") == "overlap_guard" for m in data.get("missing", []))

    director = json.loads((proj_dir / "analysis" / "director.json").read_text(encoding="utf-8"))
    assert director["pick_count"] == 1
    pick = director["picks"][0]
    assert pick["candidate_id"] == "cid1"
    assert len(pick["title"]) <= 60
    assert len(pick["hook"]) <= 80
    assert len(pick["description"]) <= 220
    assert len(pick["hashtags"]) >= 3
    assert all(str(t).startswith("#") for t in pick["hashtags"])
    assert pick["template"] in {
        "speaker_broll",
        "reaction_stack",
        "proof_overlay",
        "single_subject_punch",
        "vertical_streamer_pip",
        "vertical_streamer_split",
        "vertical_blur",
        "vertical_crop_center",
        "original",
    }


def test_actions_ai_apply_director_picks_rejects_generic_or_domain_anchored_packaging(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_237_packaging_guard".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 120.0},
        "analysis": {
            "highlights": {
                "candidates": [
                    {"rank": 1, "candidate_id": "cid1", "score": 1.0, "start_s": 10.0, "end_s": 20.0, "peak_time_s": 15.0},
                    {"rank": 2, "candidate_id": "cid2", "score": 0.9, "start_s": 40.0, "end_s": 50.0, "peak_time_s": 45.0},
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
            },
            {
                "candidate_rank": 2,
                "candidate_peak_time_s": 45.0,
                "variants": [
                    {"variant_id": "medium", "start_s": 40.0, "end_s": 60.0, "duration_s": 20.0},
                ],
            },
        ],
    }
    (proj_dir / "analysis" / "variants.json").write_text(json.dumps(variants_json), encoding="utf-8")

    r = client.post(
        "/api/actions/ai/apply_director_picks",
        headers=hdr,
        json={
            "project_id": pid,
            "client_request_id": "apply-dir-packaging-1",
            "picks": [
                {
                    "candidate_id": "cid1",
                    "variant_id": "medium",
                    "title": "She Found DolphinGirlGames.com and Locked In",
                    "hook": "Wait for it",
                    "description": "The stream stops being normal the second DolphinGirlGames appears. Instant obsession, instant comedy, instant clip.",
                    "hashtags": ["clips"],
                },
                {
                    "candidate_id": "cid2",
                    "variant_id": "medium",
                    "title": "Chat Called the Doxx Risk First",
                    "hook": "Oh no",
                    "description": "She clicks anyway, chat panics instantly, and the self-own lands right after.",
                    "hashtags": ["clips"],
                },
            ],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["pick_count"] == 1
    assert any(
        m.get("error") == "packaging_quality"
        and "title_domain_anchor" in (m.get("issues") or [])
        and "description_too_generic" in (m.get("issues") or [])
        for m in data.get("missing", [])
    )

    director = json.loads((proj_dir / "analysis" / "director.json").read_text(encoding="utf-8"))
    assert director["pick_count"] == 1
    pick = director["picks"][0]
    assert pick["candidate_id"] == "cid2"
    assert pick["title"] == "Chat Called the Doxx Risk First"


def test_actions_ai_apply_chapter_labels_accepts_chapter_zero(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_240_ch0".encode("utf-8")).hexdigest()
    proj_dir = tmp_path / "outputs" / "projects" / pid
    (proj_dir / "analysis").mkdir(parents=True, exist_ok=True)

    project_json = {
        "project_id": pid,
        "created_at": "now",
        "video": {"path": str(proj_dir / "video" / "video.mp4"), "duration_seconds": 60.0},
        "analysis": {"highlights": {"candidates": []}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")
    chapters_json = {
        "generated_at": "now",
        "chapter_count": 1,
        "chapters": [{"id": 0, "start_s": 0.0, "end_s": 60.0, "title": "old", "summary": "", "keywords": []}],
    }
    (proj_dir / "analysis" / "chapters.json").write_text(json.dumps(chapters_json), encoding="utf-8")

    r = client.post(
        "/api/actions/ai/apply_chapter_labels",
        headers=hdr,
        json={
            "project_id": pid,
            "client_request_id": "apply-ch0-1",
            "items": [{"chapter_id": 0, "title": "new title"}],
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["updated_count"] == 1

    updated = json.loads((proj_dir / "analysis" / "chapters.json").read_text(encoding="utf-8"))
    assert updated["chapters"][0]["id"] == 0
    assert updated["chapters"][0]["title"] == "new title"


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
        json={"project_id": pid, "limit": 1, "llm_mode": "local", "client_request_id": "export-dir-1"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    job = _wait_for_terminal_job_state(job_id)
    assert job is not None
    assert job.status == "succeeded"


def test_actions_export_director_picks_passes_camera_plan_to_export(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    from videopipeline.studio.jobs import JOB_MANAGER

    pid = hashlib.sha256("twitch_238_camera_plan".encode("utf-8")).hexdigest()
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
            {
                "rank": 1,
                "candidate_rank": 1,
                "variant_id": "medium",
                "start_s": 10.0,
                "end_s": 30.0,
                "duration_s": 20.0,
                "title": "t",
                "hook": "h",
                "confidence": 0.7,
                "camera_plan": [
                    {"at_s": 0.0, "focus_x": 0.4, "focus_y": 0.33, "zoom": 1.0},
                    {"at_s": 2.0, "focus_x": 0.58, "focus_y": 0.48, "zoom": 1.6},
                ],
            },
        ],
    }
    (proj_dir / "analysis" / "director.json").write_text(json.dumps(director_json), encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_start_export(*, proj, selection, export_dir, **kwargs):
        captured["selection"] = selection
        job = JOB_MANAGER.create("export")
        JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result={"output": str(export_dir / "dummy.mp4")})
        return job

    monkeypatch.setattr(JOB_MANAGER, "start_export", fake_start_export)

    r = client.post(
        "/api/actions/export_director_picks",
        headers=hdr,
        json={"project_id": pid, "limit": 1, "llm_mode": "local", "client_request_id": "export-dir-camera-plan-1"},
    )
    assert r.status_code == 200
    job = _wait_for_terminal_job_state(r.json()["job_id"])
    assert job is not None
    assert job.status == "succeeded"
    assert captured["selection"]["camera_plan"] == [
        {"at_s": 0.0, "focus_x": 0.4, "focus_y": 0.33, "zoom": 1.0},
        {"at_s": 2.0, "focus_x": 0.58, "focus_y": 0.48, "zoom": 1.6},
    ]


def test_actions_export_batch_external_strict_rejects_raw_top_candidates(tmp_path, monkeypatch):
    profile_path = tmp_path / "strict_export_batch.yaml"
    profile_path.write_text(
        """
analysis:
  highlights:
    llm_semantic_enabled: false
    llm_filter_enabled: false
  chapters:
    enabled: false
    llm_labeling: false
ai:
  director:
    enabled: true
""".strip(),
        encoding="utf-8",
    )
    client = _make_client(tmp_path, monkeypatch, token="secret", profile_path=profile_path)
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_export_batch_strict_gate".encode("utf-8")).hexdigest()
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
    (proj_dir / "analysis" / "director.json").write_text(
        json.dumps(
            {
                "created_at": "now",
                "pick_count": 1,
                "provenance": {"source": "chatgpt_actions", "agent": "gondull"},
                "config": {"source": "chatgpt_actions"},
                "picks": [
                    {
                        "rank": 1,
                        "candidate_rank": 1,
                        "variant_id": "medium",
                        "start_s": 10.0,
                        "end_s": 30.0,
                        "duration_s": 20.0,
                        "title": "t",
                        "hook": "h",
                        "description": "d",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    r = client.post(
        "/api/actions/export_batch",
        headers=hdr,
        json={"project_id": pid, "top": 1, "llm_mode": "external_strict", "client_request_id": "export-batch-strict-1"},
    )
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "external_strict_requires_director_picks"
    assert any("raw selections" in issue for issue in detail["issues"])


def test_actions_export_batch_top_candidates_uses_resolved_export_defaults(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    from videopipeline.studio.jobs import JOB_MANAGER
    from videopipeline.subtitles import DEFAULT_CAPTION_THEME

    pid = hashlib.sha256("twitch_export_batch_defaults".encode("utf-8")).hexdigest()
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
        "analysis": {
            "highlights": {
                "candidates": [
                    {"rank": 1, "candidate_id": "cid1", "score": 1.0, "start_s": 10.0, "end_s": 20.0, "title": "clip title"},
                ]
            }
        },
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_start_export(*, proj, selection, export_dir, **kwargs):
        captured["selection"] = selection
        captured["kwargs"] = kwargs
        job = JOB_MANAGER.create("export")
        JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result={"output": str(export_dir / "dummy.mp4")})
        return job

    monkeypatch.setattr(JOB_MANAGER, "start_export", fake_start_export)

    r = client.post(
        "/api/actions/export_batch",
        headers=hdr,
        json={
            "project_id": pid,
            "top": 1,
            "llm_mode": "local",
            "client_request_id": "export-batch-defaults-1",
            "export": {"layout_preset": "speaker_broll"},
        },
    )
    assert r.status_code == 200

    job = _wait_for_terminal_job_state(r.json()["job_id"])
    assert job is not None
    assert job.status == "succeeded"
    assert captured["selection"]["title"] == "clip title"
    assert captured["kwargs"]["template"] == "speaker_broll"
    assert captured["kwargs"]["caption_theme"] == DEFAULT_CAPTION_THEME


@pytest.mark.parametrize("path", ["/api/actions/run_full_export_top", "/api/actions/run_full_export_top_unattended"])
def test_actions_run_full_export_top_external_strict_requires_checkpoint(tmp_path, monkeypatch, path):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    r = client.post(
        path,
        headers=hdr,
        json={
            "url": "https://www.twitch.tv/videos/123456789",
            "top": 1,
            "llm_mode": "external_strict",
            "client_request_id": f"strict-checkpoint-{path.rsplit('/', 1)[-1]}",
        },
    )
    assert r.status_code == 409
    assert r.json()["detail"] == "external_strict_requires_ai_checkpoint"


@pytest.mark.parametrize("path", ["/api/actions/run_full_export_top", "/api/actions/run_full_export_top_unattended"])
def test_actions_run_full_export_top_uses_profile_default_llm_mode(tmp_path, monkeypatch, path):
    profile_path = tmp_path / "strict_default.yaml"
    profile_path.write_text(
        """
studio:
  default_llm_mode: external_strict
""".strip(),
        encoding="utf-8",
    )
    client = _make_client(tmp_path, monkeypatch, token="secret", profile_path=profile_path)
    hdr = {"Authorization": "Bearer secret"}

    r = client.post(
        path,
        headers=hdr,
        json={
            "url": "https://www.twitch.tv/videos/123456789",
            "top": 1,
            "client_request_id": f"profile-default-{path.rsplit('/', 1)[-1]}",
        },
    )
    assert r.status_code == 409
    assert r.json()["detail"] == "external_strict_requires_ai_checkpoint"


def test_actions_export_director_picks_external_strict_requires_external_ai(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

    pid = hashlib.sha256("twitch_export_strict_gate".encode("utf-8")).hexdigest()
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
        "analysis": {"highlights": {"candidates": [{"rank": 1, "candidate_id": "cid1", "score": 1.0, "start_s": 10.0, "end_s": 20.0}]}},
        "layout": {},
        "selections": [],
        "exports": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(project_json), encoding="utf-8")
    (proj_dir / "analysis" / "director.json").write_text(
        json.dumps(
            {
                "created_at": "now",
                "pick_count": 1,
                "picks": [{"rank": 1, "candidate_rank": 1, "variant_id": "medium", "start_s": 10.0, "end_s": 30.0}],
            }
        ),
        encoding="utf-8",
    )

    r = client.post(
        "/api/actions/export_director_picks",
        headers=hdr,
        json={"project_id": pid, "limit": 1, "llm_mode": "external_strict", "client_request_id": "export-dir-strict-1"},
    )
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "external_ai_incomplete"
    assert any("semantic" in issue for issue in detail["issues"])


def test_actions_export_director_picks_external_strict_succeeds_with_external_provenance(tmp_path, monkeypatch):
    profile_path = tmp_path / "strict_external.yaml"
    profile_path.write_text(
        """
analysis:
  highlights:
    llm_semantic_enabled: false
    llm_filter_enabled: false
  chapters:
    enabled: false
    llm_labeling: false
ai:
  director:
    enabled: true
""".strip(),
        encoding="utf-8",
    )
    client = _make_client(tmp_path, monkeypatch, token="secret", profile_path=profile_path)
    hdr = {"Authorization": "Bearer secret"}

    from videopipeline.studio.jobs import JOB_MANAGER

    pid = hashlib.sha256("twitch_export_strict_ok".encode("utf-8")).hexdigest()
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
    (proj_dir / "analysis" / "variants.json").write_text(
        json.dumps(
            {
                "created_at": "now",
                "candidates": [
                    {
                        "candidate_rank": 1,
                        "candidate_id": "cid1",
                        "candidate_peak_time_s": 15.0,
                        "variants": [{"variant_id": "medium", "start_s": 10.0, "end_s": 30.0, "duration_s": 20.0}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    apply_r = client.post(
        "/api/actions/ai/apply_director_picks",
        headers=hdr,
        json={
            "project_id": pid,
            "client_request_id": "apply-dir-prov-1",
            "provenance": {"agent": "gondull", "provider": "openai", "model": "gpt-5.3-xhigh"},
            "picks": [{"candidate_id": "cid1", "variant_id": "medium", "title": "t", "hook": "h", "description": "d"}],
        },
    )
    assert apply_r.status_code == 200
    director = json.loads((proj_dir / "analysis" / "director.json").read_text(encoding="utf-8"))
    assert director["provenance"]["agent"] == "gondull"
    assert director["config"]["source"] == "chatgpt_actions"

    def fake_start_export(*, proj, selection, export_dir, **kwargs):
        job = JOB_MANAGER.create("export")
        JOB_MANAGER._set(job, status="succeeded", progress=1.0, message="done", result={"output": str(export_dir / "dummy.mp4")})
        return job

    monkeypatch.setattr(JOB_MANAGER, "start_export", fake_start_export)

    export_r = client.post(
        "/api/actions/export_director_picks",
        headers=hdr,
        json={"project_id": pid, "limit": 1, "llm_mode": "external_strict", "client_request_id": "export-dir-prov-1"},
    )
    assert export_r.status_code == 200
    job = _wait_for_terminal_job_state(export_r.json()["job_id"])
    assert job is not None
    assert job.status == "succeeded"


def test_actions_export_director_picks_accepts_legacy_pick_shape(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, token="secret")
    hdr = {"Authorization": "Bearer secret"}

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
        json={"project_id": pid, "limit": 1, "llm_mode": "local", "client_request_id": "export-dir-legacy-1"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    job = _wait_for_terminal_job_state(job_id)
    assert job is not None
    assert job.status == "succeeded"
