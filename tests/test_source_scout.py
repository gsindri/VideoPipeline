from __future__ import annotations

from pathlib import Path

import videopipeline.source_scout as source_scout_mod


class _FakeResponse:
    def __init__(self, *, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.ok = 200 <= status_code < 300
        self.text = str(payload)

    def json(self):
        return self._payload


def test_source_preflight_issue_requires_twitch_credentials(monkeypatch):
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_API_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TWITCH_API_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TWITCH_APP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TWITCH_ACCESS_TOKEN", raising=False)

    issue = source_scout_mod.source_preflight_issue(
        {
            "provider": "twitch_helix",
            "channel_login": "ludwig",
        }
    )

    assert issue is not None
    assert "TWITCH_CLIENT_ID" in issue


def test_resolve_watchlist_path_prefers_local_watchlist(tmp_path, monkeypatch):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    watchlist_path = sources_dir / "watchlist.local.yaml"
    watchlist_path.write_text("shadow_mode: true\nsources: []\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    resolved = source_scout_mod.resolve_watchlist_path()

    assert resolved == watchlist_path


def test_source_preflight_issue_allows_twitch_url_fallback_without_credentials(monkeypatch):
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_API_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TWITCH_API_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TWITCH_APP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TWITCH_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(source_scout_mod, "yt_dlp_available", lambda: True)

    issue = source_scout_mod.source_preflight_issue(
        {
            "provider": "twitch_helix",
            "url": "https://www.twitch.tv/ludwig/videos",
        }
    )

    assert issue is None


def test_fetch_source_entries_twitch_helix_by_login(monkeypatch):
    monkeypatch.setenv("TWITCH_CLIENT_ID", "client-id")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "client-secret")
    monkeypatch.delenv("TWITCH_APP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TWITCH_ACCESS_TOKEN", raising=False)
    source_scout_mod._TWITCH_TOKEN_CACHE["access_token"] = ""
    source_scout_mod._TWITCH_TOKEN_CACHE["expires_at"] = 0.0
    source_scout_mod._TWITCH_USER_CACHE.clear()

    def fake_post(url, data=None, timeout=None):
        assert "oauth2/token" in url
        assert data["grant_type"] == "client_credentials"
        return _FakeResponse(
            status_code=200,
            payload={
                "access_token": "app-token",
                "expires_in": 3600,
                "token_type": "bearer",
            },
        )

    def fake_get(url, headers=None, params=None, timeout=None):
        assert headers["Client-ID"] == "client-id"
        assert headers["Authorization"] == "Bearer app-token"
        if url.endswith("/users"):
            assert params == {"login": "ludwig"}
            return _FakeResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "user-123",
                            "login": "ludwig",
                            "display_name": "Ludwig",
                        }
                    ]
                },
            )
        if url.endswith("/videos"):
            assert params["user_id"] == "user-123"
            assert params["type"] == "archive"
            return _FakeResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "id": "987654321",
                            "url": "https://www.twitch.tv/videos/987654321",
                            "title": "Ludwig stream title",
                            "duration": "2h19m52s",
                            "published_at": "2026-03-08T01:02:03Z",
                            "view_count": 12345,
                            "user_name": "Ludwig",
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(source_scout_mod.requests, "post", fake_post)
    monkeypatch.setattr(source_scout_mod.requests, "get", fake_get)

    entries = source_scout_mod.fetch_source_entries(
        {
            "provider": "twitch_helix",
            "platform": "twitch",
            "channel_login": "ludwig",
        },
        limit=3,
    )

    assert len(entries) == 1
    assert entries[0]["url"] == "https://www.twitch.tv/videos/987654321"
    assert entries[0]["channel_id"] == "user-123"
    assert entries[0]["channel_name"] == "Ludwig"
    assert entries[0]["duration_seconds"] == 8392.0
    assert entries[0]["platform"] == "twitch"
    assert entries[0]["fetch_mode"] == "twitch_helix"


def test_fetch_source_entries_twitch_without_api_uses_yt_dlp_fallback(monkeypatch):
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_API_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TWITCH_API_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TWITCH_APP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TWITCH_ACCESS_TOKEN", raising=False)

    class _FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            assert url == "https://www.twitch.tv/ludwig/videos"
            assert download is False
            return {
                "entries": [
                    {
                        "id": "987654321",
                        "ie_key": "TwitchVod",
                        "url": "987654321",
                        "title": "Fallback twitch VOD",
                        "duration": 1234,
                        "timestamp": 1762502400,
                        "channel": "Ludwig",
                        "channel_id": "user-123",
                    }
                ]
            }

    monkeypatch.setattr(source_scout_mod, "_load_yt_dlp", lambda: _FakeYoutubeDL)

    entries = source_scout_mod.fetch_source_entries(
        {
            "provider": "twitch_helix",
            "platform": "twitch",
            "url": "https://www.twitch.tv/ludwig/videos",
        },
        limit=2,
    )

    assert len(entries) == 1
    assert entries[0]["url"] == "https://www.twitch.tv/videos/987654321"
    assert entries[0]["platform"] == "twitch"
    assert entries[0]["channel_name"] == "Ludwig"
    assert entries[0]["fetch_mode"] == "yt_dlp_fallback"


def test_build_source_profile_separates_metrics_and_judgments():
    profile = source_scout_mod.build_source_profile(
        {
            "id": "ludwig-twitch",
            "priority": 5,
            "profile": {
                "category": "anchor",
                "clip_density_rating": 4,
                "style_fit_rating": 5,
                "saturation_rating": 4,
                "rights_risk_rating": 3,
                "rated_by": "sindri",
                "rated_at": "2026-03-08",
                "notes": "High reach and strong fit, but already saturated.",
            },
        },
        {
            "source_stats": {
                "ludwig-twitch": {
                    "project_count": 4,
                    "projects_with_candidates": 4,
                    "projects_with_director_picks": 3,
                    "projects_with_exports": 2,
                }
            }
        },
    )

    assert profile["category"] == "anchor"
    assert profile["metrics"]["project_count"] == 4
    assert profile["metrics"]["candidate_hit_rate"] == 1.0
    assert profile["metrics"]["director_pick_rate"] == 0.75
    assert profile["metrics"]["export_success_rate"] == 0.5
    assert profile["judgments"]["clip_density_rating"] == 4
    assert profile["judgments"]["style_fit_rating"] == 5
    assert profile["judgments"]["saturation_rating"] == 4
    assert profile["judgments"]["rights_risk_rating"] == 3
    assert profile["judgments"]["rated_by"] == "sindri"
    assert profile["judgments"]["notes"] == "High reach and strong fit, but already saturated."
    assert profile["recommendation"]["band"] in {"medium", "high"}
    assert any("clip density rated 4/5" in reason for reason in profile["recommendation"]["reasons"])


def test_build_source_scout_report_reranks_shortlist_with_chat_probe(tmp_path, monkeypatch):
    watchlist_path = tmp_path / "watchlist.local.yaml"
    watchlist_path.write_text(
        """
shadow_mode: true
scout_probe:
  enabled: true
  shortlist: 3
  min_candidates: 2
  rerank_weight: 0.24
sources:
  - id: ludwig
    label: Ludwig Twitch VODs
    platform: twitch
    provider: twitch_helix
    enabled: true
    priority: 2
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        source_scout_mod,
        "build_project_history",
        lambda projects_root=None: {
            "processed_urls": set(),
            "processed_content_keys": set(),
            "source_stats": {},
        },
    )
    monkeypatch.setattr(
        source_scout_mod,
        "list_source_inbox_entries",
        lambda status="pending": (None, []),
    )

    def fake_fetch_entries(source, limit):
        assert source["id"] == "ludwig"
        return [
            {
                "url": "https://www.twitch.tv/videos/111",
                "title": "Candidate one",
                "duration_seconds": 7200,
                "published_at": "2026-03-15T14:00:00Z",
                "platform": "twitch",
                "channel_name": "Ludwig",
                "video_id": "111",
            },
            {
                "url": "https://www.twitch.tv/videos/222",
                "title": "Candidate two",
                "duration_seconds": 7200,
                "published_at": "2026-03-15T03:00:00Z",
                "platform": "twitch",
                "channel_name": "Ludwig",
                "video_id": "222",
            },
            {
                "url": "https://www.twitch.tv/videos/333",
                "title": "Candidate three",
                "duration_seconds": 7200,
                "published_at": "2026-03-14T22:00:00Z",
                "platform": "twitch",
                "channel_name": "Ludwig",
                "video_id": "333",
            },
        ]

    probed_urls = []

    def fake_probe_candidates(selected, *, probe_config, now_ts):
        probed_urls.extend(item["url"] for item in selected)
        return {
            "https://www.twitch.tv/videos/111": {
                "status": "ok",
                "score": 0.10,
                "peak_count": 1,
                "laugh_peak_count": 0,
            },
            "https://www.twitch.tv/videos/222": {
                "status": "ok",
                "score": 0.95,
                "peak_count": 5,
                "laugh_peak_count": 2,
            },
            "https://www.twitch.tv/videos/333": {
                "status": "ok",
                "score": 0.40,
                "peak_count": 2,
                "laugh_peak_count": 0,
            },
        }

    report = source_scout_mod.build_source_scout_report(
        watchlist_path=watchlist_path,
        per_source=3,
        limit=5,
        now_ts=1742061600.0,  # 2026-03-15T15:00:00Z
        fetch_entries_fn=fake_fetch_entries,
        probe_candidates_fn=fake_probe_candidates,
    )

    ranked_urls = [item["url"] for item in report["candidates"]]
    assert probed_urls == ranked_urls[:3]
    assert ranked_urls[0] == "https://www.twitch.tv/videos/222"
    assert report["meta"]["chat_probe"]["status"] == "applied"
    assert report["meta"]["chat_probe"]["used_count"] == 3
    assert "chat excitement probe across shortlisted VODs" in report["strategy"]["ranking_factors"]
    assert report["candidates"][0]["chat_probe"]["score"] == 0.95
    assert any(
        "chat probe compared multiple VODs" in reason
        for reason in report["candidates"][0]["reasons"]
    )


def test_build_source_scout_report_skips_chat_probe_without_multiple_candidates(tmp_path, monkeypatch):
    watchlist_path = tmp_path / "watchlist.local.yaml"
    watchlist_path.write_text(
        """
shadow_mode: true
scout_probe:
  enabled: true
  shortlist: 3
  min_candidates: 2
sources:
  - id: ludwig
    label: Ludwig Twitch VODs
    platform: twitch
    provider: twitch_helix
    enabled: true
    priority: 2
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        source_scout_mod,
        "build_project_history",
        lambda projects_root=None: {
            "processed_urls": set(),
            "processed_content_keys": set(),
            "source_stats": {},
        },
    )
    monkeypatch.setattr(
        source_scout_mod,
        "list_source_inbox_entries",
        lambda status="pending": (None, []),
    )

    probe_calls = {"count": 0}

    def fake_probe_candidates(selected, *, probe_config, now_ts):
        probe_calls["count"] += 1
        return {}

    report = source_scout_mod.build_source_scout_report(
        watchlist_path=watchlist_path,
        per_source=3,
        limit=5,
        now_ts=1742061600.0,
        fetch_entries_fn=lambda source, limit: [
            {
                "url": "https://www.twitch.tv/videos/111",
                "title": "Only candidate",
                "duration_seconds": 7200,
                "published_at": "2026-03-15T14:00:00Z",
                "platform": "twitch",
                "channel_name": "Ludwig",
                "video_id": "111",
            }
        ],
        probe_candidates_fn=fake_probe_candidates,
    )

    assert probe_calls["count"] == 0
    assert report["meta"]["chat_probe"]["status"] == "not_enough_candidates"
    assert report["candidates"][0]["url"] == "https://www.twitch.tv/videos/111"


def test_build_source_scout_report_dedupes_probe_shortlist_keys(tmp_path, monkeypatch):
    watchlist_path = tmp_path / "watchlist.local.yaml"
    watchlist_path.write_text(
        """
shadow_mode: true
scout_probe:
  enabled: true
  shortlist: 4
  min_candidates: 2
sources:
  - id: ludwig
    label: Ludwig Twitch VODs
    platform: twitch
    provider: twitch_helix
    enabled: true
    priority: 2
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        source_scout_mod,
        "build_project_history",
        lambda projects_root=None: {
            "processed_urls": set(),
            "processed_content_keys": set(),
            "source_stats": {},
        },
    )
    monkeypatch.setattr(
        source_scout_mod,
        "list_source_inbox_entries",
        lambda status="pending": (None, []),
    )

    probed_urls = []

    def fake_probe_candidates(selected, *, probe_config, now_ts):
        probed_urls.extend(item["url"] for item in selected)
        return {
            item["url"]: {"status": "ok", "score": 0.7, "peak_count": 2, "laugh_peak_count": 1}
            for item in selected
        }

    report = source_scout_mod.build_source_scout_report(
        watchlist_path=watchlist_path,
        per_source=4,
        limit=5,
        now_ts=1742061600.0,
        fetch_entries_fn=lambda source, limit: [
            {
                "url": "https://www.twitch.tv/videos/111",
                "title": "Candidate one",
                "duration_seconds": 7200,
                "published_at": "2026-03-15T14:00:00Z",
                "platform": "twitch",
                "channel_name": "Ludwig",
                "video_id": "111",
            },
            {
                "url": "https://www.twitch.tv/videos/111",
                "title": "Duplicate candidate one",
                "duration_seconds": 7100,
                "published_at": "2026-03-15T13:00:00Z",
                "platform": "twitch",
                "channel_name": "Ludwig",
                "video_id": "111",
            },
            {
                "url": "https://www.twitch.tv/videos/222",
                "title": "Candidate two",
                "duration_seconds": 7200,
                "published_at": "2026-03-15T12:00:00Z",
                "platform": "twitch",
                "channel_name": "Ludwig",
                "video_id": "222",
            },
        ],
        probe_candidates_fn=fake_probe_candidates,
    )

    assert probed_urls == [
        "https://www.twitch.tv/videos/111",
        "https://www.twitch.tv/videos/222",
    ]
    assert report["meta"]["chat_probe"]["shortlist_count"] == 2
    assert report["meta"]["chat_probe"]["used_count"] == 2


def test_probe_single_candidate_chat_replaces_existing_probe_files(tmp_path, monkeypatch):
    import videopipeline.chat.downloader as chat_downloader_mod

    cache_root = tmp_path / "probe"
    raw_path = cache_root / "chat.json"
    db_path = cache_root / "chat.sqlite"
    summary_path = cache_root / "summary.json"
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("old raw", encoding="utf-8")
    db_path.write_text("old db", encoding="utf-8")

    monkeypatch.setattr(
        source_scout_mod,
        "_probe_cache_paths",
        lambda url, *, content_key=None: {
            "root": cache_root,
            "summary": summary_path,
            "raw": raw_path,
            "db": db_path,
        },
    )
    monkeypatch.setattr(source_scout_mod, "_load_cached_probe_summary", lambda *a, **k: None)
    monkeypatch.setattr(chat_downloader_mod, "is_chat_download_available", lambda: True)

    download_calls = {"count": 0}
    compute_calls = {"count": 0}

    def fake_download_chat(url, output_path, **kwargs):
        download_calls["count"] += 1
        assert Path(output_path) != raw_path
        assert output_path.name.startswith("chat-")
        Path(output_path).write_text("new raw", encoding="utf-8")
        return object()

    monkeypatch.setattr(chat_downloader_mod, "download_chat", fake_download_chat)

    def fake_compute_chat_probe_summary(**kwargs):
        compute_calls["count"] += 1
        assert kwargs["raw_path"] != raw_path
        assert kwargs["db_path"] != db_path
        Path(kwargs["db_path"]).write_text("new db", encoding="utf-8")
        return {
            "status": "ok",
            "method": "chat_excitement_probe",
            "score": 0.8,
        }

    monkeypatch.setattr(source_scout_mod, "_compute_chat_probe_summary", fake_compute_chat_probe_summary)

    summary = source_scout_mod._probe_single_candidate_chat(
        {
            "url": "https://www.twitch.tv/videos/111",
            "content_key": "twitch_111",
            "platform": "twitch",
        },
        probe_config={"ttl_hours": 18.0},
        now_ts=1742061600.0,
    )

    assert download_calls["count"] == 1
    assert compute_calls["count"] == 1
    assert raw_path.read_text(encoding="utf-8") == "new raw"
    assert db_path.read_text(encoding="utf-8") == "new db"
    assert summary_path.exists()
    assert summary["status"] == "ok"
