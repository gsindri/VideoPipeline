"""
VideoPipeline Launcher

A Windows-friendly launcher that:
- Starts the Studio server on a free port
- Reuses an existing running instance if one is already up
- Opens the browser to the Home screen
- Writes logs somewhere sane
- Works both from source and when frozen into an .exe
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Suppress console windows from GPU libraries on Windows
# This must be done BEFORE importing torch/tensorflow/etc
if sys.platform == "win32":
    # Environment variables to suppress console spam from various GPU libraries
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow
    os.environ.setdefault("ORT_DISABLE_ALL_LOGS", "1")  # ONNX Runtime
    os.environ.setdefault("HIP_LAUNCH_BLOCKING", "0")    # ROCm/HIP

    # Monkey-patch subprocess.Popen to always use CREATE_NO_WINDOW
    # This catches any subprocess spawned by child libraries (torch ROCm, etc.)
    import subprocess
    _original_popen = subprocess.Popen

    class _SilentPopen(_original_popen):
        def __init__(self, *args, **kwargs):
            if sys.platform == "win32" and "creationflags" not in kwargs:
                # CREATE_NO_WINDOW = 0x08000000
                kwargs["creationflags"] = kwargs.get("creationflags", 0) | 0x08000000
            super().__init__(*args, **kwargs)

    subprocess.Popen = _SilentPopen

import requests
import uvicorn

from videopipeline.logging_config import setup_logging
from videopipeline.profile import resolve_default_profile_path

APP_NAME = "VideoPipeline"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765  # used only if --port specified; otherwise we auto-pick
STUDIO_PORT_ENV = "VP_STUDIO_PORT"
STUDIO_PROFILE_ENV = "VP_STUDIO_PROFILE"
STUDIO_ENV_FILE_ENV = "VP_STUDIO_ENV_FILE"
STARTUP_LOCK_STALE_S = 90.0
STARTUP_LOCK_WAIT_S = 30.0
STARTUP_LOCK_POLL_S = 0.25

logger = logging.getLogger(__name__)


def _is_frozen() -> bool:
    """Check if running as a frozen executable (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def _fix_frozen_streams() -> None:
    """Fix stdout/stderr when running as frozen exe without console.

    When a PyInstaller app runs with --noconsole (or --windowed), stdout/stderr
    are None, which causes uvicorn's logging to crash when calling .isatty().

    This redirects them to a log file or a null stream.
    """
    if not _is_frozen():
        return

    # Check if streams are broken (None or missing isatty)
    def _needs_fix(stream):
        if stream is None:
            return True
        try:
            stream.isatty()
            return False
        except (AttributeError, OSError):
            return True

    if _needs_fix(sys.stdout) or _needs_fix(sys.stderr):
        # Redirect to log file
        log_path = _log_file()
        try:
            log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
            if _needs_fix(sys.stdout):
                sys.stdout = log_handle
            if _needs_fix(sys.stderr):
                sys.stderr = log_handle
        except Exception:
            # Fall back to devnull if we can't open the log file
            devnull = open(os.devnull, "w", encoding="utf-8")
            if _needs_fix(sys.stdout):
                sys.stdout = devnull
            if _needs_fix(sys.stderr):
                sys.stderr = devnull


def _exe_dir() -> Path:
    """Get the directory containing the executable or source."""
    if _is_frozen():
        # When frozen, sys.executable is the .exe path
        return Path(sys.executable).resolve().parent
    # When running from source, use repo root (two levels up from this file)
    return Path(__file__).resolve().parents[2]


def _default_profile_path() -> Optional[Path]:
    """Get the default profile path for launcher/runtime use."""
    exe = _exe_dir()
    return resolve_default_profile_path(search_roots=[exe, exe.parent])


def _profile_from_env() -> Optional[Path]:
    """Return an explicit Studio profile from env, if configured."""
    raw = (os.getenv(STUDIO_PROFILE_ENV) or "").strip()
    if not raw:
        return None
    return Path(raw)


def _resolve_profile_path(profile_path: Optional[Path]) -> Optional[Path]:
    """Resolve profile path to an absolute path for stable comparisons/runtime metadata."""
    if profile_path is None:
        return None
    p = Path(profile_path).expanduser()
    try:
        return p.resolve()
    except Exception:
        return p.absolute()


def _normalize_profile_path(profile_path: Optional[Path]) -> Optional[str]:
    """Normalize profile path for case-insensitive matching on Windows."""
    resolved = _resolve_profile_path(profile_path)
    if resolved is None:
        return None
    out = str(resolved)
    if os.name == "nt":
        out = os.path.normcase(out)
    return out


def _state_dir() -> Path:
    """Get the directory for launcher runtime state (APPDATA on Windows)."""
    if os.name == "nt":
        base = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        p = Path(base) / APP_NAME
    else:
        p = Path.home() / f".{APP_NAME.lower()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _workspace_dir() -> Path:
    """Get the workspace directory for outputs/projects when running as exe."""
    if os.name == "nt":
        # LOCALAPPDATA is better for big files than roaming APPDATA
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home())
        p = Path(base) / APP_NAME / "Workspace"
    else:
        p = _state_dir() / "workspace"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _local_env_candidates() -> List[Path]:
    paths: List[Path] = []
    explicit = (os.getenv(STUDIO_ENV_FILE_ENV) or "").strip()
    if explicit:
        paths.append(Path(explicit).expanduser())
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home() / "AppData" / "Local")
        paths.append(Path(base) / APP_NAME / "studio.env")
    else:
        paths.append(Path.home() / f".{APP_NAME.lower()}" / "studio.env")
    paths.append(_exe_dir() / ".env.local")
    seen = set()
    out: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _parse_env_assignment(raw_line: str) -> Optional[tuple[str, str]]:
    line = str(raw_line or "").strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def _load_local_env_file() -> Optional[Path]:
    for path in _local_env_candidates():
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_env_assignment(raw_line)
                if parsed is None:
                    continue
                key, value = parsed
                os.environ.setdefault(key, value)
            return path
        except Exception as exc:
            logger.warning("Failed to load Studio env file %s: %s", path, exc)
    return None


def _runtime_file() -> Path:
    """Get the path to the runtime state file."""
    return _state_dir() / "runtime.json"


def _startup_lock_file() -> Path:
    """Get the path to the transient startup lock file."""
    return _state_dir() / "startup.lock"


def _log_file() -> Path:
    """Get the path to the launcher log file."""
    logs = _state_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / "launcher.log"


def _pick_free_port() -> int:
    """Find a free port by letting the OS pick one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((DEFAULT_HOST, 0))
        return int(s.getsockname()[1])


def _port_from_env() -> Optional[int]:
    """Return the fixed Studio port from env, if valid."""
    raw = (os.getenv(STUDIO_PORT_ENV) or "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r (expected integer 1..65535)", STUDIO_PORT_ENV, raw)
        return None
    if not (1 <= port <= 65535):
        logger.warning("Ignoring invalid %s=%r (expected range 1..65535)", STUDIO_PORT_ENV, raw)
        return None
    return port


def _resolve_port(explicit_port: Optional[int]) -> int:
    """Resolve the Studio port with precedence: CLI --port > env > random free port."""
    if explicit_port is not None:
        return explicit_port
    env_port = _port_from_env()
    if env_port is not None:
        return env_port
    return _pick_free_port()


def _http_ok(url: str, timeout: float = 0.25) -> bool:
    """Check if an HTTP endpoint returns 200 OK."""
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def _load_startup_lock(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _try_acquire_startup_lock(now_ts: Optional[float] = None) -> Optional[Path]:
    lock_path = _startup_lock_file()
    attempt_ts = float(time.time() if now_ts is None else now_ts)

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            payload = _load_startup_lock(lock_path)
            started_at = None
            if isinstance(payload, dict):
                try:
                    started_at = float(payload.get("started_at"))
                except Exception:
                    started_at = None
            if started_at is None or (attempt_ts - started_at) >= STARTUP_LOCK_STALE_S:
                _safe_unlink(lock_path)
                continue
            return None

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started_at": attempt_ts}, handle)
        return lock_path


def _release_startup_lock(lock_path: Optional[Path]) -> None:
    if lock_path is None:
        return
    _safe_unlink(lock_path)


def _await_running_instance_or_startup_slot(
    *,
    requested_profile: Optional[Path] = None,
    requested_bind_host: Optional[str] = None,
    max_wait_s: float = STARTUP_LOCK_WAIT_S,
    poll_s: float = STARTUP_LOCK_POLL_S,
) -> tuple[Optional[Path], Optional[str]]:
    deadline = time.time() + max(0.0, float(max_wait_s))

    while True:
        existing = _reuse_existing_if_running(
            requested_profile=requested_profile,
            requested_bind_host=requested_bind_host,
        )
        if existing:
            return None, existing

        lock_path = _try_acquire_startup_lock()
        if lock_path is not None:
            return lock_path, None

        if time.time() >= deadline:
            return None, None
        time.sleep(max(0.05, float(poll_s)))


def _release_startup_lock_when_ready(lock_path: Optional[Path], url: str) -> None:
    if lock_path is None:
        return

    def _watch_startup() -> None:
        deadline = time.time() + STARTUP_LOCK_WAIT_S
        while time.time() < deadline:
            if _http_ok(url + "/api/health", timeout=0.2):
                break
            time.sleep(0.1)
        _release_startup_lock(lock_path)

    threading.Thread(target=_watch_startup, daemon=True).start()


def _reuse_existing_if_running(
    *,
    requested_profile: Optional[Path] = None,
    requested_bind_host: Optional[str] = None,
) -> Optional[str]:
    """Check if a matching Studio instance is already running and return its URL if so."""
    rf = _runtime_file()
    if not rf.exists():
        return None
    try:
        data = json.loads(rf.read_text(encoding="utf-8"))
        requested_norm = _normalize_profile_path(requested_profile)
        running_profile_raw = data.get("profile")
        running_norm = _normalize_profile_path(Path(running_profile_raw)) if running_profile_raw else None
        if requested_norm is not None and running_norm != requested_norm:
            return None
        running_bind_host = str(data.get("bind_host") or data.get("host") or DEFAULT_HOST)
        if requested_bind_host and running_bind_host != requested_bind_host:
            return None
        host = data.get("host", DEFAULT_HOST)
        port = int(data.get("port"))
        url = f"http://{host}:{port}"
        if _http_ok(url + "/api/health"):
            return url
    except Exception:
        return None
    return None


def _write_runtime(
    host: str,
    port: int,
    *,
    bind_host: Optional[str] = None,
    profile_path: Optional[Path] = None,
) -> None:
    """Write the runtime state file with server info."""
    payload = {
        "host": host,
        "bind_host": bind_host or host,
        "port": port,
        "pid": os.getpid(),
        "started_at": time.time(),
    }
    normalized_profile = _normalize_profile_path(profile_path)
    if normalized_profile is not None:
        payload["profile"] = normalized_profile
    _runtime_file().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_runtime() -> None:
    """Remove the runtime state file."""
    try:
        _runtime_file().unlink(missing_ok=True)
    except TypeError:
        # For older Python without missing_ok
        try:
            if _runtime_file().exists():
                _runtime_file().unlink()
        except Exception:
            pass


def _windows_ffmpeg_path_candidates() -> list[Path]:
    """Return best-effort Windows ffmpeg directories for PATH hydration."""
    if os.name != "nt":
        return []

    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        normalized = str(path).strip().rstrip('\\/')
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(Path(normalized))

    raw = os.environ.get("FFMPEG_SHARED_BIN", "")
    if raw:
        for entry in raw.split(os.pathsep):
            entry = entry.strip().strip('"')
            if entry:
                _add(Path(entry))

    _add(Path(r"C:\Tools\ffmpeg-shared\bin"))

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        winget_root = (
            Path(local_app_data)
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        )
        if winget_root.exists():
            versioned_bins = sorted(winget_root.glob("**/bin"), reverse=True)
            for path in versioned_bins:
                _add(path)

    return candidates


def _maybe_add_ffmpeg_to_path() -> None:
    """
    Add bundled or well-known ffmpeg locations to PATH if present.

    This keeps yt-dlp merge/download flows working even when the process was
    launched without the user's normal Windows PATH hydrated.
    """
    base = _exe_dir()
    candidates = [base, base / "ffmpeg"]
    if os.name == "nt":
        candidates.extend(_windows_ffmpeg_path_candidates())
    for c in candidates:
        ffmpeg_exe = c / "ffmpeg.exe" if os.name == "nt" else c / "ffmpeg"
        ffprobe_exe = c / "ffprobe.exe" if os.name == "nt" else c / "ffprobe"
        if ffmpeg_exe.exists() and ffprobe_exe.exists():
            current = os.environ.get("PATH", "")
            os.environ["PATH"] = str(c) + (os.pathsep + current if current else "")
            return


def _open_browser_when_ready(url: str, max_wait_s: float = 8.0) -> None:
    """Wait for the server to be ready, then open the browser."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        if _http_ok(url + "/api/health", timeout=0.2):
            break
        time.sleep(0.1)
    try:
        webbrowser.open(url)
    except Exception as exc:
        logger.warning("Failed to open browser to %s: %s", url, exc)


@dataclass
class LaunchArgs:
    """Parsed launcher arguments."""
    host: str = DEFAULT_HOST
    runtime_host: Optional[str] = None
    port: Optional[int] = None
    profile: Optional[Path] = None
    no_browser: bool = False
    reuse: bool = True


def main(argv: Optional[List[str]] = None) -> None:
    """Main launcher entry point."""
    # Fix broken stdout/stderr when running as frozen exe without console
    _fix_frozen_streams()

    loaded_env_file = _load_local_env_file()
    level = getattr(logging, os.getenv("VP_LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_path = Path(os.getenv("VP_LOG_FILE", "")).expanduser() if os.getenv("VP_LOG_FILE") else _log_file()
    setup_logging(level=level, log_file=log_path)
    if loaded_env_file is not None:
        logger.info("Loaded Studio env file: %s", loaded_env_file)

    argv = argv if argv is not None else sys.argv[1:]

    # Minimal arg parsing (keeps it stable inside PyInstaller)
    args = LaunchArgs()
    it = iter(argv)
    for token in it:
        if token == "--host":
            args.host = next(it)
        elif token == "--runtime-host":
            args.runtime_host = next(it)
        elif token == "--port":
            args.port = int(next(it))
        elif token == "--profile":
            args.profile = Path(next(it))
        elif token == "--no-browser":
            args.no_browser = True
        elif token == "--no-reuse":
            args.reuse = False
        # Ignore unknown tokens to avoid breaking old shortcuts

    # Resolve profile early so --profile / VP_STUDIO_PROFILE are respected even when reuse is enabled.
    profile_path = _resolve_profile_path(
        args.profile if args.profile is not None else (_profile_from_env() or _default_profile_path())
    )
    if profile_path is not None and not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    startup_lock: Optional[Path] = None

    # If already running, reuse (perfect UX) and avoid duplicate launches
    # while a sibling launcher is still bringing the runtime up.
    if args.reuse:
        existing = _reuse_existing_if_running(
            requested_profile=profile_path,
            requested_bind_host=args.host,
        )
        if existing:
            if not args.no_browser:
                try:
                    webbrowser.open(existing)
                except Exception as exc:
                    logger.warning("Failed to open browser to %s: %s", existing, exc)
            return
        startup_lock, existing = _await_running_instance_or_startup_slot(
            requested_profile=profile_path,
            requested_bind_host=args.host,
        )
        if existing:
            if not args.no_browser:
                try:
                    webbrowser.open(existing)
                except Exception as exc:
                    logger.warning("Failed to open browser to %s: %s", existing, exc)
            return
        if startup_lock is None:
            logger.info("VideoPipeline startup is already in progress; skipping duplicate launch.")
            return
    else:
        startup_lock = _try_acquire_startup_lock()

    # When frozen, run from a stable workspace folder so outputs don't end up in System32
    if _is_frozen():
        os.chdir(_workspace_dir())

    _maybe_add_ffmpeg_to_path()

    port = _resolve_port(args.port)
    runtime_host = args.runtime_host or args.host
    url = f"http://{runtime_host}:{port}"

    # Create app in HOME mode with either explicit --profile or the resolved
    # default orchestration profile for this runtime.
    from videopipeline.studio.app import create_app

    app = create_app(video_path=None, profile_path=profile_path)

    _write_runtime(runtime_host, port, bind_host=args.host, profile_path=profile_path)
    _release_startup_lock_when_ready(startup_lock, url)

    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    try:
        # Uvicorn server
        config = uvicorn.Config(
            app,
            host=args.host,
            port=port,
            log_level="info",
            access_log=False,
        )
        server = uvicorn.Server(config)
        server.run()
    finally:
        _release_startup_lock(startup_lock)
        _clear_runtime()


if __name__ == "__main__":
    # Needed for PyInstaller + multiprocessing quirks on Windows
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass

    main()
