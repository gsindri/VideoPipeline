"""Chat downloader integration.

Supports downloading chat from:
  - Twitch VODs (via TwitchDownloaderCLI)
  - YouTube live streams / premieres (via chat-downloader)

Uses TwitchDownloaderCLI for Twitch, chat-downloader for YouTube,
with fallback to yt-dlp for basic chat extraction.

Includes retry logic and fallback mechanisms for robustness.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .normalize import load_chat_data
from ..utils import subprocess_flags as _subprocess_flags, utc_iso as _utc_iso


# Minimum recommended version of TwitchDownloaderCLI
TWITCH_CLI_MIN_VERSION = "1.54.0"
TWITCH_CLI_DOWNLOAD_URL = "https://github.com/lay295/TwitchDownloader/releases"


def _get_tools_dir() -> Path:
    """Get the tools directory path."""
    # Check relative to this file's package (for development)
    pkg_dir = Path(__file__).parent.parent.parent.parent  # chat -> videopipeline -> src -> VideoPipeline
    tools_dir = pkg_dir / "tools"
    if tools_dir.exists():
        return tools_dir
    
    # Check PyInstaller bundled data location
    if getattr(sys, 'frozen', False):
        # PyInstaller extracts bundled data to _MEIPASS
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            tools_dir = Path(meipass) / "tools"
            if tools_dir.exists():
                return tools_dir
        
        # Also check relative to exe (COLLECT puts files alongside exe)
        exe_dir = Path(sys.executable).parent
        tools_dir = exe_dir / "tools"
        if tools_dir.exists():
            return tools_dir
    
    return Path("tools")


def _find_twitch_downloader_cli() -> Optional[Path]:
    """Find TwitchDownloaderCLI executable."""
    # Check tools directory
    tools_dir = _get_tools_dir()
    cli_path = tools_dir / "TwitchDownloaderCLI.exe"
    if cli_path.exists():
        return cli_path
    
    # Check PATH
    which_result = shutil.which("TwitchDownloaderCLI")
    if which_result:
        return Path(which_result)
    
    # Also try without .exe extension
    which_result = shutil.which("TwitchDownloaderCLI.exe")
    if which_result:
        return Path(which_result)
    
    return None


def _get_twitch_downloader_version(cli_path: Path) -> Tuple[Optional[str], bool]:
    """Get TwitchDownloaderCLI version and check if update needed.
    
    Returns:
        Tuple of (version_string, needs_update)
    """
    import logging
    log = logging.getLogger("videopipeline.chat")
    
    try:
        result = subprocess.run(
            [str(cli_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            **_subprocess_flags(),
        )
        
        # TwitchDownloaderCLI outputs version on stderr, format:
        # "TwitchDownloaderCLI \n1.56.2+hash"
        output = (result.stdout + result.stderr).strip()
        
        # Look for version pattern (X.Y.Z possibly with +hash suffix)
        version_match = re.search(r"(\d+\.\d+(?:\.\d+)?)", output)
        if version_match:
            version = version_match.group(1)
            
            # Compare versions
            try:
                current_parts = [int(x) for x in version.split(".")[:3]]
                min_parts = [int(x) for x in TWITCH_CLI_MIN_VERSION.split(".")[:3]]
                
                # Pad to same length
                while len(current_parts) < 3:
                    current_parts.append(0)
                while len(min_parts) < 3:
                    min_parts.append(0)
                
                needs_update = current_parts < min_parts
                
                if needs_update:
                    log.warning(
                        f"TwitchDownloaderCLI {version} is outdated. "
                        f"Recommended: {TWITCH_CLI_MIN_VERSION}+. "
                        f"Download from: {TWITCH_CLI_DOWNLOAD_URL}"
                    )
                
                return version, needs_update
            except ValueError:
                return version, False
                
    except Exception as e:
        log.debug(f"Could not get TwitchDownloaderCLI version: {e}")
    
    return None, False


class ChatDownloadError(Exception):
    """Raised when chat download fails."""

    pass


@dataclass
class ChatDownloadResult:
    """Result of chat download."""

    output_path: Path
    platform: str
    video_id: str
    message_count: int
    duration_ms: int
    downloader: str  # "chat_replay_downloader" or "yt-dlp"
    downloader_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "platform": self.platform,
            "video_id": self.video_id,
            "message_count": self.message_count,
            "duration_ms": self.duration_ms,
            "downloader": self.downloader,
            "downloader_version": self.downloader_version,
        }


def _detect_platform(url: str) -> tuple[str, str]:
    """Detect platform and video ID from URL.

    Returns:
        Tuple of (platform, video_id)
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # Twitch VOD
    if "twitch.tv" in domain:
        # https://www.twitch.tv/videos/1234567890
        match = re.search(r"/videos/(\d+)", parsed.path)
        if match:
            return "twitch", match.group(1)
        # https://www.twitch.tv/channel/video/1234567890
        match = re.search(r"/video/(\d+)", parsed.path)
        if match:
            return "twitch", match.group(1)

    # YouTube
    if "youtube.com" in domain or "youtu.be" in domain:
        # youtube.com/watch?v=VIDEO_ID
        if "youtube.com" in domain:
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return "youtube", qs["v"][0]
        # youtu.be/VIDEO_ID
        if "youtu.be" in domain:
            video_id = parsed.path.strip("/").split("/")[0]
            if video_id:
                return "youtube", video_id

    return "unknown", ""


def _find_chat_downloader() -> Optional[str]:
    """Find chat-downloader executable."""
    # Try Python module
    try:
        import chat_downloader

        return "chat_downloader"
    except ImportError:
        pass

    # Try CLI tool
    if shutil.which("chat_downloader"):
        return "chat_downloader"

    # Try alternate names
    for name in ["chat-downloader"]:
        if shutil.which(name):
            return name

    return None


def _find_yt_dlp() -> Optional[str]:
    """Find yt-dlp executable."""
    if shutil.which("yt-dlp"):
        return "yt-dlp"
    return None


def _download_with_chat_downloader(
    url: str,
    output_path: Path,
    *,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> ChatDownloadResult:
    """Download chat using chat-downloader."""
    platform, video_id = _detect_platform(url)

    if on_progress:
        on_progress(0.0, "Starting chat download...")

    # Try Python library first
    try:
        from chat_downloader import ChatDownloader
        import chat_downloader as crd_module

        downloader = ChatDownloader()

        if on_progress:
            on_progress(0.1, "Downloading chat messages...")

        # Get chat messages
        chat = downloader.get_chat(url, output=str(output_path))

        # Count messages
        messages = list(chat) if hasattr(chat, "__iter__") else []
        message_count = len(messages)

        # Get duration from last message
        duration_ms = 0
        if messages:
            last_msg = messages[-1]
            if "time_in_seconds" in last_msg:
                duration_ms = int(last_msg["time_in_seconds"] * 1000)

        if on_progress:
            on_progress(1.0, f"Downloaded {message_count} messages")

        return ChatDownloadResult(
            output_path=output_path,
            platform=platform,
            video_id=video_id,
            message_count=message_count,
            duration_ms=duration_ms,
            downloader="chat_downloader",
            downloader_version=getattr(crd_module, "__version__", "unknown"),
        )

    except ImportError:
        pass
    except Exception as e:
        # Re-raise any other errors (network, API, etc.) as ChatDownloadError
        raise ChatDownloadError(f"Chat download failed: {e}")

    # Fall back to CLI
    cmd_name = _find_chat_downloader()
    if not cmd_name:
        raise ChatDownloadError(
            "chat-downloader not found. Install with: pip install chat-downloader"
        )

    if on_progress:
        on_progress(0.1, "Running chat-downloader...")

    cmd = [cmd_name, url, "-o", str(output_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            **_subprocess_flags(),
        )

        if result.returncode != 0:
            raise ChatDownloadError(f"chat-downloader failed: {result.stderr}")

    except FileNotFoundError:
        raise ChatDownloadError(f"Command not found: {cmd_name}")

    # Parse output file to get stats
    if not output_path.exists():
        raise ChatDownloadError("Chat download produced no output file")

    data = load_chat_data(output_path)
    messages = data if isinstance(data, list) else data.get("messages", data.get("chat", []))
    message_count = len(messages)

    duration_ms = 0
    if messages:
        last_msg = messages[-1] if isinstance(messages[-1], dict) else {}
        if "time_in_seconds" in last_msg:
            duration_ms = int(last_msg["time_in_seconds"] * 1000)
        elif "offset" in last_msg:
            duration_ms = int(float(last_msg["offset"]) * 1000)

    if on_progress:
        on_progress(1.0, f"Downloaded {message_count} messages")

    return ChatDownloadResult(
        output_path=output_path,
        platform=platform,
        video_id=video_id,
        message_count=message_count,
        duration_ms=duration_ms,
        downloader="chat_downloader_cli",
        downloader_version="unknown",
    )


def _download_with_ytdlp(
    url: str,
    output_path: Path,
    *,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> ChatDownloadResult:
    """Download chat using yt-dlp (limited support)."""
    platform, video_id = _detect_platform(url)

    if platform == "twitch":
        raise ChatDownloadError(
            "yt-dlp does not support Twitch chat. Place TwitchDownloaderCLI.exe in the tools/ folder."
        )

    yt_dlp = _find_yt_dlp()
    if not yt_dlp:
        raise ChatDownloadError("yt-dlp not found")

    if on_progress:
        on_progress(0.1, "Downloading with yt-dlp...")

    # yt-dlp can extract live chat for YouTube
    # Use --write-subs to get live chat as a subtitle-like format
    temp_dir = output_path.parent
    temp_base = output_path.stem

    cmd = [
        yt_dlp,
        "--skip-download",
        "--write-subs",
        "--sub-langs",
        "live_chat",
        "-o",
        str(temp_dir / f"{temp_base}.%(ext)s"),
        url,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            **_subprocess_flags(),
        )

        if result.returncode != 0:
            raise ChatDownloadError(f"yt-dlp failed: {result.stderr}")

    except FileNotFoundError:
        raise ChatDownloadError("yt-dlp not found")

    # Find the output file
    chat_file = None
    for ext in [".live_chat.json", ".json"]:
        candidate = temp_dir / f"{temp_base}{ext}"
        if candidate.exists():
            chat_file = candidate
            break

    if not chat_file or not chat_file.exists():
        raise ChatDownloadError("yt-dlp did not produce a chat file. This video may not have live chat.")

    # Move to target location
    if chat_file != output_path:
        shutil.move(str(chat_file), str(output_path))

    # Parse to get stats
    data = json.loads(output_path.read_text(encoding="utf-8"))
    messages = data if isinstance(data, list) else data.get("messages", data.get("chat", []))
    message_count = len(messages)

    if on_progress:
        on_progress(1.0, f"Downloaded {message_count} messages")

    return ChatDownloadResult(
        output_path=output_path,
        platform=platform,
        video_id=video_id,
        message_count=message_count,
        duration_ms=0,
        downloader="yt-dlp",
        downloader_version="unknown",
    )


class ChatDownloadCancelled(Exception):
    """Raised when chat download is cancelled by user."""
    pass


def _download_with_twitch_downloader_cli(
    url: str,
    output_path: Path,
    video_id: str,
    *,
    on_progress: Optional[Callable[[float, str], None]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
    timeout: int = 1800,  # 30 minutes for large VODs
) -> ChatDownloadResult:
    """Download Twitch chat using TwitchDownloaderCLI."""
    import logging
    import queue
    import threading
    import time
    
    log = logging.getLogger("videopipeline.chat")
    
    cli_path = _find_twitch_downloader_cli()
    if not cli_path:
        raise ChatDownloadError(
            "TwitchDownloaderCLI not found. Place TwitchDownloaderCLI.exe in the tools/ folder."
        )
    
    if on_progress:
        on_progress(0.0, "Starting chat download...")
    
    # TwitchDownloaderCLI chatdownload -u VIDEO_ID -o output.json
    cmd = [
        str(cli_path),
        "chatdownload",
        "--id", video_id,
        "-o", str(output_path),
        "-E",  # Embed emote data
    ]
    
    log.info(f"[CHAT CLI] Running: {' '.join(cmd)}")
    
    # Avoid interactive overwrite prompts (and stale/empty files) by removing any
    # existing output file before running TwitchDownloaderCLI.
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
    except Exception:
        # Best effort — if we can't remove it, the CLI may refuse to overwrite and fail.
        pass
    
    try:
        # Use Popen with merged stderr->stdout to avoid index-shift bugs
        # Also use unbuffered binary mode to handle \r progress updates
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge streams - avoids the index shift bug
            stdin=subprocess.DEVNULL,  # Prevents hanging on accidental prompts
            bufsize=0,  # Unbuffered (we're reading bytes)
            **_subprocess_flags(),
        )
        
        assert process.stdout is not None
        
        # Queue for passing lines from reader thread to main thread
        line_queue: queue.Queue[str] = queue.Queue()
        all_output_lines: list[str] = []  # For error reporting
        
        def reader():
            """Read bytes and split on both \\n and \\r to handle CLI progress updates."""
            buf = b""
            while True:
                try:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    # Split on BOTH \r and \n - many CLIs use \r for progress updates
                    parts = re.split(br"[\r\n]+", buf)
                    buf = parts.pop()  # Keep remainder (partial line)
                    for p in parts:
                        if not p:
                            continue
                        decoded = p.decode("utf-8", "replace")
                        line_queue.put(decoded)
                        all_output_lines.append(decoded)
                except Exception:
                    break
            # Flush any remaining buffer
            if buf:
                decoded = buf.decode("utf-8", "replace")
                line_queue.put(decoded)
                all_output_lines.append(decoded)
        
        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        
        start_time = time.time()
        last_status = "Starting..."
        last_output_time = start_time
        
        def handle_line(line: str) -> None:
            nonlocal last_status, last_output_time
            last_output_time = time.time()
            
            log.debug(f"[CHAT CLI] {line}")
            
            if not on_progress:
                return
            
            # Try to parse percentage (supports "12%" and "12.3%")
            pct_match = re.search(r"(\d+(?:\.\d+)?)%", line)
            if pct_match:
                pct = float(pct_match.group(1))
                comments_match = re.search(r"\((\d+)\s*comments?\)", line, re.IGNORECASE)
                if comments_match:
                    last_status = f"{pct:.0f}% ({comments_match.group(1)} messages)"
                else:
                    last_status = f"{pct:.0f}%"
                on_progress(min(pct / 100.0, 1.0) * 0.9, last_status)
                return
            
            # Other status heuristics
            if "fetching" in line.lower() or "getting" in line.lower():
                last_status = "Fetching video info..."
                on_progress(0.02, last_status)
            elif "comment" in line.lower():
                count_match = re.search(r"(\d+)\s*comments?", line, re.IGNORECASE)
                if count_match:
                    last_status = f"Downloaded {count_match.group(1)} messages"
                    on_progress(0.85, last_status)
            elif "emote" in line.lower():
                last_status = "Processing emotes..."
                on_progress(0.92, last_status)
            elif "writing" in line.lower() or "saving" in line.lower():
                last_status = "Writing to file..."
                on_progress(0.95, last_status)
        
        # Poll process and update progress
        while process.poll() is None:
            # Check for cancellation
            if check_cancel and check_cancel():
                log.info("[CHAT CLI] Cancellation requested, terminating process...")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    log.info("[CHAT CLI] Process didn't terminate, killing...")
                    process.kill()
                raise ChatDownloadCancelled("Chat download cancelled by user")
            
            elapsed = time.time() - start_time
            if elapsed > timeout:
                process.kill()
                raise ChatDownloadError(f"TwitchDownloaderCLI timed out after {timeout} seconds")
            
            # Drain all available lines from the queue
            drained_any = False
            try:
                while True:
                    line = line_queue.get_nowait()
                    line = line.strip()
                    if not line:
                        continue
                    drained_any = True
                    handle_line(line)
            except queue.Empty:
                pass
            
            # If we haven't seen any output for a bit and status is still "Starting...", 
            # show a timer-based status so UI doesn't look frozen
            if on_progress and (time.time() - last_output_time) > 2.0 and last_status == "Starting...":
                # Make this slightly increasing so UI doesn't look frozen
                fallback = min(0.15, elapsed / max(timeout, 1) * 0.15)
                on_progress(fallback, f"Downloading... ({int(elapsed)}s)")
            
            time.sleep(0.2)
        
        # Wait for reader thread to finish
        reader_thread.join(timeout=2)
        
        # Drain any remaining lines after process finished
        try:
            while True:
                line = line_queue.get_nowait()
                line = line.strip()
                if line:
                    handle_line(line)
        except queue.Empty:
            pass
        
        log.info(f"[CHAT CLI] Process finished with return code: {process.returncode}")
        log.info(f"[CHAT CLI] Total output lines: {len(all_output_lines)}")
        if all_output_lines:
            log.info(f"[CHAT CLI] Last few lines: {all_output_lines[-5:]}")
        
        if process.returncode != 0:
            error_msg = "\n".join(all_output_lines) or "Unknown error"
            
            # Decode common .NET/CLR error codes for better error messages
            # 0xE0434352 (-532462766 signed, 3762504530 unsigned) = .NET CLR exception
            if process.returncode == -532462766 or process.returncode == 3762504530:
                # .NET unhandled exception - usually means API error or missing runtime
                hint = (
                    "TwitchDownloaderCLI crashed (.NET exception). Possible causes:\n"
                    "  1. Twitch API error (try again later)\n"
                    "  2. VOD chat is unavailable (deleted, sub-only, or disabled)\n"
                    "  3. Missing .NET 6.0+ runtime\n"
                    "  4. Rate limited by Twitch\n"
                )
                raise ChatDownloadError(f"{hint}CLI output: {error_msg[:400]}")
            else:
                raise ChatDownloadError(f"TwitchDownloaderCLI failed (code {process.returncode}): {error_msg[:500]}")
            
    except subprocess.TimeoutExpired:
        process.kill()
        raise ChatDownloadError(f"TwitchDownloaderCLI timed out after {timeout} seconds")
    except FileNotFoundError:
        raise ChatDownloadError(f"TwitchDownloaderCLI not found at: {cli_path}")
    
    if not output_path.exists():
        raise ChatDownloadError("TwitchDownloaderCLI produced no output file")
    
    # Parse output to get message count
    data = json.loads(output_path.read_text(encoding="utf-8"))
    comments = data.get("comments", [])
    message_count = len(comments)
    
    duration_ms = 0
    if comments:
        last_comment = comments[-1]
        duration_ms = int(last_comment.get("content_offset_seconds", 0) * 1000)
    
    if on_progress:
        on_progress(1.0, f"Downloaded {message_count} messages")
    
    return ChatDownloadResult(
        output_path=output_path,
        platform="twitch",
        video_id=video_id,
        message_count=message_count,
        duration_ms=duration_ms,
        downloader="TwitchDownloaderCLI",
        downloader_version="1.x",
    )


def _download_twitch_with_chat_downloader_fallback(
    url: str,
    output_path: Path,
    video_id: str,
    *,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> ChatDownloadResult:
    """Try to download Twitch chat using chat-downloader as fallback.
    
    chat-downloader supports Twitch but may have different rate limits
    and capabilities than TwitchDownloaderCLI.
    """
    import logging
    log = logging.getLogger("videopipeline.chat")
    
    # Check if chat-downloader is available
    if not _find_chat_downloader():
        try:
            import chat_downloader
        except ImportError:
            raise ChatDownloadError(
                "chat-downloader fallback not available. Install with: pip install chat-downloader"
            )
    
    if on_progress:
        on_progress(0.0, "Trying chat-downloader fallback...")
    
    log.info(f"[CHAT] Attempting chat-downloader fallback for Twitch VOD {video_id}")
    
    try:
        from chat_downloader import ChatDownloader
        import chat_downloader as crd_module
        
        downloader = ChatDownloader()
        
        if on_progress:
            on_progress(0.1, "chat-downloader: Fetching messages...")
        
        # chat-downloader accepts Twitch URLs
        chat = downloader.get_chat(url, output=str(output_path))
        
        # Count messages - need to iterate
        messages = list(chat) if hasattr(chat, "__iter__") else []
        message_count = len(messages)
        
        duration_ms = 0
        if messages:
            last_msg = messages[-1]
            if "time_in_seconds" in last_msg:
                duration_ms = int(last_msg["time_in_seconds"] * 1000)
        
        if on_progress:
            on_progress(1.0, f"Downloaded {message_count} messages (chat-downloader)")
        
        log.info(f"[CHAT] chat-downloader fallback succeeded: {message_count} messages")
        
        return ChatDownloadResult(
            output_path=output_path,
            platform="twitch",
            video_id=video_id,
            message_count=message_count,
            duration_ms=duration_ms,
            downloader="chat_downloader_fallback",
            downloader_version=getattr(crd_module, "__version__", "unknown"),
        )
        
    except Exception as e:
        log.warning(f"[CHAT] chat-downloader fallback also failed: {e}")
        raise ChatDownloadError(f"chat-downloader fallback failed: {e}")


def _download_twitch_with_retry(
    url: str,
    output_path: Path,
    video_id: str,
    *,
    on_progress: Optional[Callable[[float, str], None]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
    max_retries: int = 2,
    retry_delay_base: float = 5.0,
) -> ChatDownloadResult:
    """Download Twitch chat with retry logic and fallback.
    
    Attempts:
    1. TwitchDownloaderCLI (primary, with retries)
    2. chat-downloader (fallback)
    
    Args:
        url: Twitch VOD URL
        output_path: Output path for chat JSON
        video_id: Twitch video ID
        on_progress: Progress callback
        check_cancel: Callback returning True if download should be cancelled
        max_retries: Maximum retry attempts for TwitchDownloaderCLI
        retry_delay_base: Base delay between retries (exponential backoff)
    """
    import logging
    log = logging.getLogger("videopipeline.chat")
    
    cli = _find_twitch_downloader_cli()
    last_error = None
    
    if cli:
        # Check and log version
        version, needs_update = _get_twitch_downloader_version(cli)
        if version:
            log.info(f"[CHAT] TwitchDownloaderCLI version: {version}")
        
        # Try TwitchDownloaderCLI with retries
        for attempt in range(max_retries + 1):
            try:
                # Check for cancellation before each attempt
                if check_cancel and check_cancel():
                    raise ChatDownloadCancelled("Chat download cancelled by user")
                
                if attempt > 0:
                    delay = retry_delay_base * (2 ** (attempt - 1))  # Exponential backoff
                    if on_progress:
                        on_progress(0.0, f"Retry {attempt}/{max_retries} in {delay:.0f}s...")
                    log.info(f"[CHAT] Retry attempt {attempt}/{max_retries} after {delay}s delay")
                    _time.sleep(delay)
                
                return _download_with_twitch_downloader_cli(
                    url, output_path, video_id, 
                    on_progress=on_progress,
                    check_cancel=check_cancel,
                )
                
            except ChatDownloadCancelled:
                raise  # Don't retry on cancellation
            except ChatDownloadError as e:
                last_error = e
                error_str = str(e)
                
                # Check if this is a .NET crash (retryable)
                is_dotnet_crash = ".NET exception" in error_str or "3762504530" in error_str
                
                # Check if this is clearly a non-retryable error
                is_permanent = any(x in error_str.lower() for x in [
                    "not found",
                    "invalid id", 
                    "video unavailable",
                    "private video",
                ])
                
                if is_permanent:
                    log.warning(f"[CHAT] Permanent error, not retrying: {e}")
                    break
                
                if attempt < max_retries:
                    log.warning(f"[CHAT] TwitchDownloaderCLI failed (attempt {attempt + 1}): {e}")
                else:
                    log.warning(f"[CHAT] TwitchDownloaderCLI failed after {max_retries + 1} attempts")
    else:
        log.warning("[CHAT] TwitchDownloaderCLI not found, trying fallback")
    
    # Try chat-downloader fallback
    try:
        return _download_twitch_with_chat_downloader_fallback(
            url, output_path, video_id, on_progress=on_progress
        )
    except ChatDownloadError as fallback_error:
        # Both methods failed - provide comprehensive error message
        primary_msg = str(last_error) if last_error else "TwitchDownloaderCLI not available"
        fallback_msg = str(fallback_error)
        
        raise ChatDownloadError(
            f"All chat download methods failed.\n"
            f"Primary (TwitchDownloaderCLI): {primary_msg[:200]}\n"
            f"Fallback (chat-downloader): {fallback_msg[:200]}\n\n"
            f"Suggestions:\n"
            f"  1. Update TwitchDownloaderCLI from {TWITCH_CLI_DOWNLOAD_URL}\n"
            f"  2. Check if the VOD has chat available (not deleted/sub-only)\n"
            f"  3. Try again later (Twitch API may be temporarily unavailable)"
        )


def download_chat(
    url: str,
    output_path: Path,
    *,
    on_progress: Optional[Callable[[float, str], None]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
) -> ChatDownloadResult:
    """Download chat replay from URL.

    Uses TwitchDownloaderCLI for Twitch (with retry + fallback to chat-downloader),
    chat-downloader for YouTube, with fallback to yt-dlp.

    Args:
        url: Video URL (Twitch VOD, YouTube, etc.)
        output_path: Path to save chat JSON
        on_progress: Optional progress callback (frac, message)
        check_cancel: Optional callback that returns True if download should be cancelled

    Returns:
        ChatDownloadResult with download info

    Raises:
        ChatDownloadError: If download fails
        ChatDownloadCancelled: If download is cancelled by user
    """
    platform, video_id = _detect_platform(url)

    if not platform or platform == "unknown":
        raise ChatDownloadError(f"Could not detect platform from URL: {url}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use TwitchDownloaderCLI for Twitch (with retry and fallback)
    if platform == "twitch":
        return _download_twitch_with_retry(url, output_path, video_id, on_progress=on_progress, check_cancel=check_cancel)

    # Try chat-downloader for other platforms (YouTube, etc.)
    crd = _find_chat_downloader()
    if crd:
        try:
            return _download_with_chat_downloader(url, output_path, on_progress=on_progress)
        except ChatDownloadError:
            # Re-raise chat download errors (these are meaningful)
            raise
        except Exception as e:
            # Log but continue to fallback
            if on_progress:
                on_progress(0.0, f"chat-downloader failed: {e}, trying yt-dlp...")

    # Try yt-dlp as fallback
    try:
        return _download_with_ytdlp(url, output_path, on_progress=on_progress)
    except ChatDownloadError:
        raise
    except Exception as e:
        raise ChatDownloadError(f"Failed to download chat: {e}")


def is_chat_download_available() -> bool:
    """Check if any chat download tool is available."""
    return (
        _find_twitch_downloader_cli() is not None
        or _find_chat_downloader() is not None
        or _find_yt_dlp() is not None
    )


def get_supported_platforms() -> list[str]:
    """Get list of supported platforms."""
    platforms = []
    if _find_twitch_downloader_cli():
        platforms.append("twitch")
    if _find_chat_downloader():
        platforms.extend(["youtube", "facebook", "instagram"])
    elif _find_yt_dlp():
        platforms.append("youtube")
    return list(set(platforms))  # Remove duplicates


def find_twitch_downloader_cli() -> Optional[Path]:
    """Public wrapper for finding TwitchDownloaderCLI."""
    return _find_twitch_downloader_cli()


def get_twitch_downloader_info() -> Dict[str, Any]:
    """Get information about TwitchDownloaderCLI installation.
    
    Returns:
        Dict with keys: available, path, version, needs_update, download_url
    """
    cli_path = _find_twitch_downloader_cli()
    
    if not cli_path:
        return {
            "available": False,
            "path": None,
            "version": None,
            "needs_update": False,
            "download_url": TWITCH_CLI_DOWNLOAD_URL,
            "min_version": TWITCH_CLI_MIN_VERSION,
        }
    
    version, needs_update = _get_twitch_downloader_version(cli_path)
    
    return {
        "available": True,
        "path": str(cli_path),
        "version": version,
        "needs_update": needs_update,
        "download_url": TWITCH_CLI_DOWNLOAD_URL,
        "min_version": TWITCH_CLI_MIN_VERSION,
    }


def import_chat_to_project(proj: "Project", chat_json_path: Path) -> bool:
    """Import a downloaded chat JSON file into a project's chat database.
    
    Args:
        proj: Project instance
        chat_json_path: Path to downloaded chat JSON (TwitchDownloader format)
        
    Returns:
        True if import succeeded
    """
    from .store import ChatStore, ChatMeta
    from .normalize import normalize_chat_messages
    
    if not chat_json_path.exists():
        raise ChatDownloadError(f"Chat file not found: {chat_json_path}")
    
    # Load chat JSON
    try:
        raw_data = json.loads(chat_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ChatDownloadError(f"Failed to parse chat JSON: {e}")
    
    # Normalize messages
    messages = normalize_chat_messages(raw_data)
    if not messages:
        raise ChatDownloadError("No messages found in chat file")
    
    # Compute duration (ChatMessage uses t_ms, not offset_ms)
    duration_ms = max(m.t_ms for m in messages) if messages else 0
    
    # Extract channel/streamer identifiers from TwitchDownloader JSON.
    # TwitchDownloaderCLI typically includes a "streamer" object and a "video" object.
    channel = ""
    channel_id = ""
    channel_name = ""
    video_obj = raw_data.get("video", {}) or {}
    streamer_obj = raw_data.get("streamer", {}) or {}

    # Prefer login/user_login as the stable identifier (lowercased).
    for candidate in [
        streamer_obj.get("login", ""),
        video_obj.get("user_login", ""),
        streamer_obj.get("name", ""),  # some formats use name as login
        video_obj.get("channel", ""),
        video_obj.get("user_name", ""),
    ]:
        if candidate:
            channel = str(candidate).lower().strip()
            break

    # Prefer display_name/user_name for human-facing name (preserve case where possible).
    for candidate in [
        streamer_obj.get("display_name", ""),
        video_obj.get("user_name", ""),
        streamer_obj.get("name", ""),
        video_obj.get("channel", ""),
    ]:
        if candidate:
            channel_name = str(candidate).strip()
            break

    # Numeric or opaque IDs when available; fall back to channel identifier for consistency.
    for candidate in [
        streamer_obj.get("id", ""),
        streamer_obj.get("user_id", ""),
        video_obj.get("user_id", ""),
        video_obj.get("channel_id", ""),
    ]:
        if candidate:
            channel_id = str(candidate).strip()
            break
    if not channel_id:
        channel_id = channel
    if not channel_name:
        channel_name = channel
    
    # Initialize store and save
    store = ChatStore(proj.chat_db_path)
    store.initialize()
    store.insert_messages(messages)
    store.set_all_meta(ChatMeta(
        video_id=raw_data.get("video", {}).get("id", "unknown"),
        duration_ms=duration_ms,
        message_count=len(messages),
        platform="twitch",
        downloaded_at=_utc_iso(),
        channel=channel,  # Back-compat (used by older code / lookups)
        channel_id=channel_id or channel,
        channel_name=channel_name or channel,
    ))
    store.close()
    
    # Update project config
    from ..project import update_project
    
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("chat", {})
        d["chat"]["enabled"] = True
        d["chat"]["source"] = "twitch_downloader"
        d["chat"]["imported_at"] = _utc_iso()
        d["chat"]["message_count"] = len(messages)
        d["chat"]["duration_ms"] = duration_ms
    
    update_project(proj, _upd)
    
    return True


# Re-export for convenience
__all__ = [
    "ChatDownloadError",
    "ChatDownloadResult",
    "download_chat",
    "find_twitch_downloader_cli",
    "get_twitch_downloader_info",
    "import_chat_to_project",
    "is_chat_download_available",
    "get_supported_platforms",
    "TWITCH_CLI_DOWNLOAD_URL",
    "TWITCH_CLI_MIN_VERSION",
]
