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
