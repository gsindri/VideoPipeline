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
from videopipeline.studio.app import create_app

APP_NAME = "VideoPipeline"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765  # used only if --port specified; otherwise we auto-pick
STUDIO_PORT_ENV = "VP_STUDIO_PORT"

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
    """Get the default profile path (gaming.yaml in profiles dir)."""
    exe = _exe_dir()
    candidates = [
        exe / "profiles" / "gaming.yaml",
        exe.parent / "profiles" / "gaming.yaml",  # Source layout
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


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


def _runtime_file() -> Path:
    """Get the path to the runtime state file."""
    return _state_dir() / "runtime.json"


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


def _reuse_existing_if_running() -> Optional[str]:
    """Check if a Studio instance is already running and return its URL if so."""
    rf = _runtime_file()
    if not rf.exists():
        return None
    try:
        data = json.loads(rf.read_text(encoding="utf-8"))
        host = data.get("host", DEFAULT_HOST)
        port = int(data.get("port"))
        url = f"http://{host}:{port}"
        if _http_ok(url + "/api/health"):
            return url
    except Exception:
        return None
    return None


def _write_runtime(host: str, port: int) -> None:
    """Write the runtime state file with server info."""
    payload = {
        "host": host,
        "port": port,
        "pid": os.getpid(),
        "started_at": time.time(),
    }
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


def _maybe_add_ffmpeg_to_path() -> None:
    """
    Add bundled ffmpeg to PATH if present.

    If ffmpeg.exe/ffprobe.exe are placed next to the .exe (or in ./ffmpeg/),
    add that directory to PATH at runtime so ffmpeg lookup works.
    """
    base = _exe_dir()
    candidates = [base, base / "ffmpeg"]
    for c in candidates:
        ffmpeg_exe = c / "ffmpeg.exe" if os.name == "nt" else c / "ffmpeg"
        ffprobe_exe = c / "ffprobe.exe" if os.name == "nt" else c / "ffprobe"
        if ffmpeg_exe.exists() and ffprobe_exe.exists():
            os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
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
    port: Optional[int] = None
    no_browser: bool = False
    reuse: bool = True


def main(argv: Optional[List[str]] = None) -> None:
    """Main launcher entry point."""
    # Fix broken stdout/stderr when running as frozen exe without console
    _fix_frozen_streams()

    level = getattr(logging, os.getenv("VP_LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_path = Path(os.getenv("VP_LOG_FILE", "")).expanduser() if os.getenv("VP_LOG_FILE") else _log_file()
    setup_logging(level=level, log_file=log_path)

    argv = argv if argv is not None else sys.argv[1:]

    # Minimal arg parsing (keeps it stable inside PyInstaller)
    args = LaunchArgs()
    it = iter(argv)
    for token in it:
        if token == "--host":
            args.host = next(it)
        elif token == "--port":
            args.port = int(next(it))
        elif token == "--no-browser":
            args.no_browser = True
        elif token == "--no-reuse":
            args.reuse = False
        # Ignore unknown tokens to avoid breaking old shortcuts

    # If already running, reuse (perfect UX)
    if args.reuse:
        existing = _reuse_existing_if_running()
        if existing:
            if not args.no_browser:
                try:
                    webbrowser.open(existing)
                except Exception as exc:
                    logger.warning("Failed to open browser to %s: %s", existing, exc)
            return

    # When frozen, run from a stable workspace folder so outputs don't end up in System32
    if _is_frozen():
        os.chdir(_workspace_dir())

    _maybe_add_ffmpeg_to_path()

    port = _resolve_port(args.port)
    url = f"http://{args.host}:{port}"

    # Create app in HOME mode with default profile (gaming.yaml)
    profile_path = _default_profile_path()
    app = create_app(video_path=None, profile_path=profile_path)

    _write_runtime(args.host, port)

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
        _clear_runtime()


if __name__ == "__main__":
    # Needed for PyInstaller + multiprocessing quirks on Windows
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass

    main()
