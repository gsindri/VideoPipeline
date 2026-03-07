import logging
import json

from videopipeline import launcher


def test_port_from_env_valid(monkeypatch):
    monkeypatch.setenv("VP_STUDIO_PORT", "57820")
    assert launcher._port_from_env() == 57820


def test_profile_from_env_valid(monkeypatch):
    monkeypatch.setenv("VP_STUDIO_PROFILE", "profiles/gaming_nvidia.yaml")
    assert launcher._profile_from_env() == launcher.Path("profiles/gaming_nvidia.yaml")


def test_profile_from_env_empty(monkeypatch):
    monkeypatch.delenv("VP_STUDIO_PROFILE", raising=False)
    assert launcher._profile_from_env() is None


def test_port_from_env_invalid(monkeypatch, caplog):
    monkeypatch.setenv("VP_STUDIO_PORT", "not-a-port")
    launcher.logger.addHandler(caplog.handler)
    old_propagate = launcher.logger.propagate
    launcher.logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            assert launcher._port_from_env() is None
    finally:
        launcher.logger.propagate = old_propagate
        launcher.logger.removeHandler(caplog.handler)
    assert "VP_STUDIO_PORT" in caplog.text


def test_resolve_port_prefers_explicit(monkeypatch):
    monkeypatch.setenv("VP_STUDIO_PORT", "57820")
    monkeypatch.setattr(launcher, "_pick_free_port", lambda: 49000)
    assert launcher._resolve_port(8765) == 8765


def test_resolve_port_prefers_env_when_no_explicit(monkeypatch):
    monkeypatch.setenv("VP_STUDIO_PORT", "57820")
    monkeypatch.setattr(launcher, "_pick_free_port", lambda: 49000)
    assert launcher._resolve_port(None) == 57820


def test_resolve_port_falls_back_to_free_port(monkeypatch):
    monkeypatch.delenv("VP_STUDIO_PORT", raising=False)
    monkeypatch.setattr(launcher, "_pick_free_port", lambda: 49000)
    assert launcher._resolve_port(None) == 49000


def test_reuse_existing_profile_match(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime.json"
    profile = (tmp_path / "gaming_nvidia.yaml").resolve()
    runtime.write_text(
        json.dumps({"host": "127.0.0.1", "port": 57820, "profile": str(profile)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "_runtime_file", lambda: runtime)
    monkeypatch.setattr(launcher, "_http_ok", lambda *args, **kwargs: True)

    assert launcher._reuse_existing_if_running(requested_profile=profile) == "http://127.0.0.1:57820"


def test_reuse_existing_profile_mismatch(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime.json"
    running_profile = (tmp_path / "gaming.yaml").resolve()
    requested_profile = (tmp_path / "gaming_nvidia.yaml").resolve()
    runtime.write_text(
        json.dumps({"host": "127.0.0.1", "port": 57820, "profile": str(running_profile)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "_runtime_file", lambda: runtime)
    monkeypatch.setattr(launcher, "_http_ok", lambda *args, **kwargs: True)

    assert launcher._reuse_existing_if_running(requested_profile=requested_profile) is None


def test_reuse_existing_profile_unknown_runtime_profile(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime.json"
    requested_profile = (tmp_path / "gaming_nvidia.yaml").resolve()
    runtime.write_text(
        json.dumps({"host": "127.0.0.1", "port": 57820}),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "_runtime_file", lambda: runtime)
    monkeypatch.setattr(launcher, "_http_ok", lambda *args, **kwargs: True)

    assert launcher._reuse_existing_if_running(requested_profile=requested_profile) is None
