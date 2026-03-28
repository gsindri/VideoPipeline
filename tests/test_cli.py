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
