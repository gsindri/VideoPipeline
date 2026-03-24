import json
import logging

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


def test_parse_env_assignment_supports_export_and_quotes():
    assert launcher._parse_env_assignment('export TWITCH_CLIENT_ID="abc123"') == ("TWITCH_CLIENT_ID", "abc123")
    assert launcher._parse_env_assignment("TWITCH_CLIENT_SECRET='shh'") == ("TWITCH_CLIENT_SECRET", "shh")
    assert launcher._parse_env_assignment("# comment") is None
    assert launcher._parse_env_assignment("not-an-assignment") is None


def test_load_local_env_file_sets_missing_values_only(monkeypatch, tmp_path):
    env_file = tmp_path / "studio.env"
    env_file.write_text(
        'TWITCH_CLIENT_ID=client-id\nTWITCH_CLIENT_SECRET="secret-1"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("VP_STUDIO_ENV_FILE", str(env_file))
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "already-set")

    loaded = launcher._load_local_env_file()

    assert loaded == env_file
    assert launcher.os.environ["TWITCH_CLIENT_ID"] == "client-id"
    assert launcher.os.environ["TWITCH_CLIENT_SECRET"] == "already-set"


def test_try_acquire_startup_lock_skips_recent_existing_lock(monkeypatch, tmp_path):
    lock_path = tmp_path / "startup.lock"
    lock_path.write_text(json.dumps({"pid": 111, "started_at": 100.0}), encoding="utf-8")

    monkeypatch.setattr(launcher, "_startup_lock_file", lambda: lock_path)

    assert launcher._try_acquire_startup_lock(now_ts=120.0) is None
    assert lock_path.exists()


def test_try_acquire_startup_lock_reclaims_stale_lock(monkeypatch, tmp_path):
    lock_path = tmp_path / "startup.lock"
    lock_path.write_text(json.dumps({"pid": 111, "started_at": 0.0}), encoding="utf-8")

    monkeypatch.setattr(launcher, "_startup_lock_file", lambda: lock_path)

    acquired = launcher._try_acquire_startup_lock(now_ts=launcher.STARTUP_LOCK_STALE_S + 5.0)

    assert acquired == lock_path
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == launcher.os.getpid()


def test_await_running_instance_or_startup_slot_reuses_pending_start(monkeypatch):
    sleep_calls = {"count": 0}

    def fake_reuse_existing_if_running(**kwargs):
        if sleep_calls["count"] < 2:
            return None
        return "http://127.0.0.1:57820"

    monkeypatch.setattr(launcher, "_reuse_existing_if_running", fake_reuse_existing_if_running)
    monkeypatch.setattr(launcher, "_try_acquire_startup_lock", lambda now_ts=None: None)
    monkeypatch.setattr(launcher.time, "time", lambda: float(sleep_calls["count"]))

    def fake_sleep(seconds):
        sleep_calls["count"] += 1

    monkeypatch.setattr(launcher.time, "sleep", fake_sleep)

    lock_path, existing = launcher._await_running_instance_or_startup_slot(
        requested_bind_host="127.0.0.1",
        max_wait_s=5.0,
        poll_s=0.1,
    )

    assert lock_path is None
    assert existing == "http://127.0.0.1:57820"
    assert sleep_calls["count"] == 2
