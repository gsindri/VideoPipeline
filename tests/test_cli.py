# ruff: noqa: E402

import argparse
import sys
import types

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

from videopipeline import cli
from videopipeline.publisher.accounts import Account
from videopipeline.publisher.youtube_oauth import YOUTUBE_FORCE_SSL_SCOPE, YOUTUBE_UPLOAD_SCOPE


class DummyAnalysisResult:
    def __init__(self, *, success: bool = True, error: str | None = None):
        self.success = success
        self.error = error


def test_compute_highlights_candidates_payload_uses_dag_runner(monkeypatch):
    proj = object()
    captured = {}

    def fake_run_analysis(project, *, targets=None, config=None, **kwargs):
        captured["project"] = project
        captured["targets"] = targets
        captured["config"] = config
        return DummyAnalysisResult()

    monkeypatch.setattr(cli, "run_analysis", fake_run_analysis)
    monkeypatch.setattr(
        cli,
        "get_project_data",
        lambda project: {
            "analysis": {
                "highlights": {
                    "candidates": [{"rank": 1, "candidate_id": "cid1"}],
                    "signals_used": {"speech": True},
                }
            }
        },
    )

    payload = cli._compute_highlights_candidates_payload(
        proj,
        {
            "audio_events": {"enabled": False},
            "highlights": {"llm_semantic_enabled": True, "llm_filter_enabled": True},
        },
    )

    assert payload["candidates"][0]["candidate_id"] == "cid1"
    assert captured["project"] is proj
    assert captured["targets"] == {"highlights_candidates"}
    assert captured["config"]["include_chat"] is True
    assert captured["config"]["include_audio_events"] is False
    assert captured["config"]["highlights"]["llm_semantic_enabled"] is False
    assert captured["config"]["highlights"]["llm_filter_enabled"] is False


def test_compute_highlights_candidates_payload_raises_when_no_candidates(monkeypatch):
    monkeypatch.setattr(cli, "run_analysis", lambda *args, **kwargs: DummyAnalysisResult())
    monkeypatch.setattr(cli, "get_project_data", lambda project: {"analysis": {"highlights": {"candidates": []}}})

    with pytest.raises(cli.UserFacingError, match="No highlight candidates produced by DAG analysis."):
        cli._compute_highlights_candidates_payload(object(), {})


def test_cmd_accounts_add_youtube_uses_delete_capable_default_scopes(monkeypatch):
    captured: dict[str, object] = {}

    class DummyFlow:
        @classmethod
        def from_client_secrets_file(cls, path, scopes):
            captured["path"] = path
            captured["scopes"] = list(scopes)
            return cls()

        def run_local_server(self, port):
            return types.SimpleNamespace(
                token="access-token",
                refresh_token="refresh-token",
                token_uri="https://oauth2.googleapis.com/token",
                client_id="client-id",
                client_secret="client-secret",
                scopes=list(captured["scopes"]),
            )

    google_auth_oauthlib = sys.modules.setdefault(
        "google_auth_oauthlib", types.ModuleType("google_auth_oauthlib")
    )
    flow_mod = types.ModuleType("google_auth_oauthlib.flow")
    flow_mod.InstalledAppFlow = DummyFlow
    sys.modules["google_auth_oauthlib.flow"] = flow_mod
    setattr(google_auth_oauthlib, "flow", flow_mod)

    account = Account(id="acct-yt", platform="youtube", label="YT")

    class DummyStore:
        def add(self, *, platform: str, label: str):
            captured["added"] = (platform, label)
            return account

    stored: dict[str, object] = {}
    monkeypatch.setattr(cli, "AccountStore", lambda: DummyStore())
    monkeypatch.setattr(
        cli,
        "store_tokens",
        lambda platform, account_id, payload: stored.update(
            {"platform": platform, "account_id": account_id, "payload": payload}
        ),
    )
    monkeypatch.setattr(cli, "accounts_path", lambda: "accounts.json")

    cli.cmd_accounts_add_youtube(
        argparse.Namespace(
            client_secrets="client-secret.json",
            label="Rebbi",
            scopes=None,
            redirect_port=8080,
        )
    )

    assert captured["scopes"] == [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_FORCE_SSL_SCOPE]
    assert captured["added"] == ("youtube", "Rebbi")
    assert stored["platform"] == "youtube"
    assert stored["account_id"] == "acct-yt"
    assert stored["payload"]["scopes"] == [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_FORCE_SSL_SCOPE]


def test_cmd_accounts_reconnect_youtube_upgrades_legacy_upload_only_scope(monkeypatch):
    captured: dict[str, object] = {}

    class DummyFlow:
        @classmethod
        def from_client_config(cls, config, scopes):
            captured["config"] = config
            captured["scopes"] = list(scopes)
            return cls()

        def run_local_server(self, **kwargs):
            captured["run_local_server"] = kwargs
            return types.SimpleNamespace(
                token="new-access-token",
                refresh_token="new-refresh-token",
                token_uri="https://oauth2.googleapis.com/token",
                client_id="client-id",
                client_secret="client-secret",
                scopes=list(captured["scopes"]),
            )

    google_auth_oauthlib = sys.modules.setdefault(
        "google_auth_oauthlib", types.ModuleType("google_auth_oauthlib")
    )
    flow_mod = types.ModuleType("google_auth_oauthlib.flow")
    flow_mod.InstalledAppFlow = DummyFlow
    sys.modules["google_auth_oauthlib.flow"] = flow_mod
    setattr(google_auth_oauthlib, "flow", flow_mod)

    account = Account(id="acct-legacy", platform="youtube", label="Legacy")

    class DummyStore:
        def get(self, account_id: str):
            return account if account_id == account.id else None

    stored: dict[str, object] = {}
    monkeypatch.setattr(cli, "AccountStore", lambda: DummyStore())
    monkeypatch.setattr(
        cli,
        "load_tokens",
        lambda platform, account_id: {
            "client_id": "client-id",
            "client_secret": "client-secret",
            "refresh_token": "old-refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "scopes": [YOUTUBE_UPLOAD_SCOPE],
        },
    )
    monkeypatch.setattr(
        cli,
        "store_tokens",
        lambda platform, account_id, payload: stored.update(
            {"platform": platform, "account_id": account_id, "payload": payload}
        ),
    )

    cli.cmd_accounts_reconnect_youtube(
        argparse.Namespace(
            account_id=account.id,
            redirect_port=8765,
            retry_job_id=None,
            scopes=None,
            no_browser=True,
        )
    )

    assert captured["scopes"] == [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_FORCE_SSL_SCOPE]
    assert stored["platform"] == "youtube"
    assert stored["account_id"] == account.id
    assert stored["payload"]["scopes"] == [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_FORCE_SSL_SCOPE]
