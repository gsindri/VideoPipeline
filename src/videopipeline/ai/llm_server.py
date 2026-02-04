"""Auto-managed llama.cpp server for on-demand LLM inference.

Starts the server when needed and optionally stops it after idle timeout.
This saves VRAM when not in use (important for gaming machines).

Auto-stop works via a watchdog process that monitors a heartbeat file.
When idle timeout is exceeded, the watchdog kills the llama-server process.
"""
from __future__ import annotations

import atexit
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Optional, Callable, Union
import os
import json

# Type alias for path-like arguments (string or Path)
PathLike = Union[str, Path]


def _as_path(p: Optional[PathLike]) -> Optional[Path]:
    """Coerce a path-like value to a Path object.
    
    This handles YAML config values which are strings, not Path objects.
    """
    if p is None:
        return None
    if isinstance(p, Path):
        return p
    return Path(p)


# Heartbeat file location (in temp dir)
def _get_heartbeat_path() -> Path:
    """Get path to heartbeat file."""
    import tempfile
    return Path(tempfile.gettempdir()) / "llama_server_heartbeat.json"


def _write_heartbeat() -> None:
    """Write current timestamp to heartbeat file."""
    try:
        data = {"last_used": time.time(), "pid": os.getpid()}
        _get_heartbeat_path().write_text(json.dumps(data))
    except Exception:
        pass


def _read_heartbeat() -> Optional[float]:
    """Read last used timestamp from heartbeat file."""
    try:
        path = _get_heartbeat_path()
        if path.exists():
            data = json.loads(path.read_text())
            return float(data.get("last_used", 0))
    except Exception:
        pass
    return None


# Watchdog script that runs as a detached process
_WATCHDOG_SCRIPT = '''
import time
import json
import os
import sys
import subprocess

heartbeat_path = sys.argv[1]
idle_timeout = float(sys.argv[2])
check_interval = 30  # Check every 30 seconds

def read_heartbeat():
    try:
        if os.path.exists(heartbeat_path):
            with open(heartbeat_path) as f:
                data = json.load(f)
                return float(data.get("last_used", 0))
    except:
        pass
    return 0

def is_server_running():
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:11435/health", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except:
        return False

def kill_server():
    # Kill llama-server process on Windows
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], 
                      capture_output=True, timeout=10)
    except:
        pass

# Wait a bit before first check (let server warm up)
time.sleep(60)

while True:
    if not is_server_running():
        # Server not running, exit watchdog
        break
    
    last_used = read_heartbeat()
    idle_time = time.time() - last_used if last_used > 0 else float("inf")
    
    if idle_time > idle_timeout:
        # Idle too long, kill server
        kill_server()
        # Clean up heartbeat file
        try:
            os.remove(heartbeat_path)
        except:
            pass
        break
    
    time.sleep(check_interval)
'''


@dataclass
class LlamaServerConfig:
    """Configuration for llama.cpp server."""
    server_path: Path = Path("C:/llama.cpp/llama-server.exe")
    model_path: Path = Path("C:/llama.cpp/models/qwen2.5-7b-instruct-q4_k_m.gguf")
    port: int = 11435
    n_gpu_layers: int = 99  # -1 or 99 = offload all to GPU
    context_size: int = 4096
    startup_timeout_s: float = 120.0  # 2 minutes - large models (7B+) need time to load
    health_check_interval_s: float = 1.0
    auto_stop_after_idle_s: Optional[float] = None  # None = don't auto-stop


class LlamaServerError(Exception):
    """Raised when server operations fail."""
    pass


class LlamaServerManager:
    """Manages a llama.cpp server process with auto-start capability.
    
    Usage:
        manager = LlamaServerManager(config)
        manager.ensure_running()  # Starts if needed, waits for ready
        # ... use the server ...
        manager.stop()  # Optional: stop when done
    
    The manager is thread-safe and can be shared across the application.
    """
    
    _instance: Optional["LlamaServerManager"] = None
    _instance_lock = Lock()
    
    def __init__(self, cfg: LlamaServerConfig):
        self.cfg = cfg
        self._process: Optional[subprocess.Popen] = None
        self._lock = Lock()
        self._last_use_time: float = 0
        self._idle_monitor_thread: Optional[Thread] = None
        self._stopping = False
        self._on_status: Optional[Callable[[str], None]] = None
        self._we_started_it = False  # Track if we started the server
        
        # Only register cleanup if we might auto-stop
        # For detached servers, we don't want to kill them on exit
        # atexit.register(self._cleanup)
    
    @classmethod
    def get_instance(cls, cfg: Optional[LlamaServerConfig] = None) -> "LlamaServerManager":
        """Get or create the singleton server manager."""
        with cls._instance_lock:
            if cls._instance is None:
                if cfg is None:
                    cfg = LlamaServerConfig()
                cls._instance = cls(cfg)
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None
    
    @property
    def endpoint(self) -> str:
        """Get the server endpoint URL."""
        return f"http://127.0.0.1:{self.cfg.port}"
    
    def set_status_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set a callback for status messages."""
        self._on_status = callback
    
    def _status(self, msg: str) -> None:
        """Emit a status message."""
        if self._on_status:
            try:
                self._on_status(msg)
            except Exception:
                pass
    
    def is_running(self) -> bool:
        """Check if the server process is running and responding."""
        # First check if we have a process
        if self._process is not None:
            if self._process.poll() is not None:
                # Process has exited
                self._process = None
        
        # Check if server responds
        return self._health_check()
    
    def _health_check(self) -> bool:
        """Check if server is responding to health checks."""
        try:
            url = f"{self.endpoint}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            # Try v1/models as fallback
            try:
                url = f"{self.endpoint}/v1/models"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    return resp.status == 200
            except Exception:
                return False
    
    def ensure_running(self, on_status: Optional[Callable[[str], None]] = None) -> bool:
        """Ensure the server is running, starting it if necessary.
        
        Args:
            on_status: Optional callback for status messages
            
        Returns:
            True if server is running, False if failed to start
        """
        if on_status:
            self._on_status = on_status
        
        with self._lock:
            self._last_use_time = time.time()
            
            # Already running?
            if self.is_running():
                self._status("LLM server already running")
                return True
            
            # Check if another instance is running (not started by us)
            if self._health_check():
                self._status("LLM server already running (external)")
                return True
            
            # Need to start
            return self._start_server()
    
    def _start_server(self) -> bool:
        """Start the llama.cpp server process."""
        # Validate paths
        if not self.cfg.server_path.exists():
            self._status(f"llama-server not found: {self.cfg.server_path}")
            return False
        
        if not self.cfg.model_path.exists():
            self._status(f"Model not found: {self.cfg.model_path}")
            return False
        
        self._status(f"Starting LLM server ({self.cfg.model_path.name})...")
        
        # Build command
        cmd = [
            str(self.cfg.server_path),
            "-m", str(self.cfg.model_path),
            "--port", str(self.cfg.port),
            "-ngl", str(self.cfg.n_gpu_layers),
            "-c", str(self.cfg.context_size),
        ]
        
        try:
            # Start process detached so it survives parent exit
            # On Windows, use CREATE_NO_WINDOW to hide the console
            startupinfo = None
            creationflags = 0
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                # CREATE_NO_WINDOW (0x08000000) - don't create a console window
                # Note: DETACHED_PROCESS can cause window flashing on some systems
                creationflags = 0x08000000
            
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                startupinfo=startupinfo,
                creationflags=creationflags,
                cwd=str(self.cfg.server_path.parent),
                # Don't close file handles on exec - helps with detached process
                close_fds=False if sys.platform == "win32" else True,
            )
        except Exception as e:
            self._status(f"Failed to start server: {e}")
            return False
        
        # Wait for server to be ready
        start_time = time.time()
        check_interval = self.cfg.health_check_interval_s
        
        while time.time() - start_time < self.cfg.startup_timeout_s:
            if self._process.poll() is not None:
                # Process exited prematurely
                self._status("Server process exited unexpectedly")
                self._process = None
                return False
            
            if self._health_check():
                elapsed = time.time() - start_time
                self._status(f"LLM server ready ({elapsed:.1f}s)")
                self._we_started_it = True
                
                # Write initial heartbeat and start watchdog if configured
                _write_heartbeat()
                if self.cfg.auto_stop_after_idle_s is not None:
                    self._start_idle_watchdog()
                
                return True
            
            time.sleep(check_interval)
        
        # Timeout
        self._status(f"Server startup timeout ({self.cfg.startup_timeout_s}s)")
        self.stop()
        return False
    
    def _start_idle_watchdog(self) -> None:
        """Start a detached watchdog process to stop server after idle timeout.
        
        The watchdog runs as a separate process so it survives when Studio exits.
        It monitors the heartbeat file and kills the server if idle too long.
        """
        if self.cfg.auto_stop_after_idle_s is None:
            return
        
        heartbeat_path = str(_get_heartbeat_path())
        idle_timeout = self.cfg.auto_stop_after_idle_s
        
        self._status(f"Starting idle watchdog ({idle_timeout:.0f}s timeout)")
        
        try:
            # Write watchdog script to temp file
            import tempfile
            script_path = Path(tempfile.gettempdir()) / "llama_watchdog.py"
            script_path.write_text(_WATCHDOG_SCRIPT)
            
            # Start watchdog as detached process
            # Use pythonw.exe on Windows to avoid console window
            python_exe = sys.executable
            if sys.platform == "win32":
                pythonw = Path(sys.executable).parent / "pythonw.exe"
                if pythonw.exists():
                    python_exe = str(pythonw)
            
            cmd = [python_exe, str(script_path), heartbeat_path, str(idle_timeout)]
            
            startupinfo = None
            creationflags = 0
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                # CREATE_NO_WINDOW only - DETACHED_PROCESS can flash
                creationflags = 0x08000000
            
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                startupinfo=startupinfo,
                creationflags=creationflags,
                close_fds=False if sys.platform == "win32" else True,
            )
        except Exception as e:
            self._status(f"Failed to start watchdog: {e}")
    
    def _start_idle_monitor(self) -> None:
        """Start a background thread to stop server after idle timeout.
        
        NOTE: This only works while the Python process is running.
        For detached servers, use _start_idle_watchdog() instead.
        """
        if self._idle_monitor_thread is not None and self._idle_monitor_thread.is_alive():
            return
        
        self._stopping = False
        self._idle_monitor_thread = Thread(target=self._idle_monitor_loop, daemon=True)
        self._idle_monitor_thread.start()
    
    def _idle_monitor_loop(self) -> None:
        """Monitor idle time and stop server if exceeded."""
        while not self._stopping:
            time.sleep(10)  # Check every 10 seconds
            
            if self._stopping:
                break
            
            idle_time = time.time() - self._last_use_time
            if idle_time > self.cfg.auto_stop_after_idle_s:
                self._status(f"Stopping LLM server (idle {idle_time:.0f}s)")
                self.stop()
                break
    
    def mark_used(self) -> None:
        """Mark the server as recently used (resets idle timer)."""
        self._last_use_time = time.time()
        # Write heartbeat for watchdog process
        _write_heartbeat()
    
    def stop(self) -> None:
        """Stop the server if we started it."""
        with self._lock:
            self._stopping = True
            
            if self._process is not None:
                try:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=2.0)
                except Exception:
                    pass
                finally:
                    self._process = None
    
    def _cleanup(self) -> None:
        """Cleanup on exit."""
        self.stop()


def get_server_manager(
    server_path: Optional[PathLike] = None,
    model_path: Optional[PathLike] = None,
    port: int = 11435,
    auto_stop_after_idle_s: Optional[float] = None,
    startup_timeout_s: float = 120.0,
) -> LlamaServerManager:
    """Get or create a server manager with the given config.
    
    Args:
        server_path: Path to llama-server.exe (string or Path)
        model_path: Path to GGUF model (string or Path)
        port: Server port
        auto_stop_after_idle_s: Stop server after this many idle seconds (None = don't auto-stop)
        startup_timeout_s: Maximum time to wait for server to start (default 120s for large models)
    
    Returns:
        LlamaServerManager instance (singleton)
    """
    # Coerce string paths to Path objects (YAML config values are strings)
    server_path_p = _as_path(server_path)
    model_path_p = _as_path(model_path)
    
    cfg = LlamaServerConfig(
        server_path=server_path_p or Path("C:/llama.cpp/llama-server.exe"),
        model_path=model_path_p or Path("C:/llama.cpp/models/qwen2.5-7b-instruct-q4_k_m.gguf"),
        port=port,
        auto_stop_after_idle_s=auto_stop_after_idle_s,
        startup_timeout_s=startup_timeout_s,
    )
    return LlamaServerManager.get_instance(cfg)


def ensure_llm_server(
    server_path: Optional[PathLike] = None,
    model_path: Optional[PathLike] = None,
    port: int = 11435,
    auto_stop_after_idle_s: Optional[float] = 600.0,  # Default: 10 minutes
    startup_timeout_s: float = 120.0,  # Default: 2 minutes (large models need time)
    on_status: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Ensure LLM server is running and return the endpoint.
    
    This is the main entry point for auto-starting the server.
    
    Args:
        server_path: Path to llama-server.exe (string or Path)
        model_path: Path to GGUF model (string or Path)
        port: Server port
        auto_stop_after_idle_s: Stop server after idle (default 600s = 10min)
        startup_timeout_s: Maximum time to wait for server startup (default 120s)
        on_status: Callback for status messages
    
    Returns:
        Server endpoint URL if running, None if failed
    """
    manager = get_server_manager(server_path, model_path, port, auto_stop_after_idle_s, startup_timeout_s)
    if manager.ensure_running(on_status=on_status):
        manager.mark_used()  # Reset idle timer
        return manager.endpoint
    return None
