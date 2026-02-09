import logging

from videopipeline import launcher


def test_port_from_env_valid(monkeypatch):
    monkeypatch.setenv("VP_STUDIO_PORT", "57820")
    assert launcher._port_from_env() == 57820


def test_port_from_env_invalid(monkeypatch, caplog):
    monkeypatch.setenv("VP_STUDIO_PORT", "not-a-port")
    with caplog.at_level(logging.WARNING):
        assert launcher._port_from_env() is None
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
