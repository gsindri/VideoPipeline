from __future__ import annotations

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
