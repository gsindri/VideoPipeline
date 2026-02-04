"""Shared utility functions for VideoPipeline.

This module provides common utilities used across multiple modules:
- subprocess_flags(): Windows-specific flags to hide console windows
- utc_iso(): UTC timestamp in ISO format
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


def subprocess_flags() -> Dict[str, Any]:
    """Return subprocess flags to hide console window on Windows.
    
    Usage:
        result = subprocess.run(cmd, **subprocess_flags())
    
    Returns:
        Dict with 'creationflags' on Windows, empty dict otherwise.
    """
    if sys.platform == "win32":
        # CREATE_NO_WINDOW = 0x08000000
        return {"creationflags": 0x08000000}
    return {}


def utc_iso() -> str:
    """Return current UTC time in ISO 8601 format.
    
    Returns:
        ISO formatted timestamp string like '2024-01-15T10:30:00+00:00'
    """
    return datetime.now(timezone.utc).isoformat()


# Aliases for backward compatibility
_subprocess_flags = subprocess_flags
_utc_iso = utc_iso


# ============================================================================
# Sleep Prevention (Windows)
# ============================================================================

class PreventSleep:
    """Context manager to prevent Windows from sleeping during long operations.
    
    Uses SetThreadExecutionState to tell Windows the system is busy.
    On non-Windows platforms, this is a no-op.
    
    Usage:
        with PreventSleep("Exporting video"):
            # ... long operation ...
            pass
    """
    
    # Windows constants for SetThreadExecutionState
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002  # Also keeps display on
    
    def __init__(self, reason: str = "VideoPipeline job running", keep_display_on: bool = False):
        self.reason = reason
        self.keep_display_on = keep_display_on
        self._kernel32 = None
    
    def __enter__(self):
        if sys.platform != "win32":
            return self
        
        try:
            import ctypes
            self._kernel32 = ctypes.windll.kernel32
            
            # Set flags: continuous + system required (+ display if requested)
            flags = self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
            if self.keep_display_on:
                flags |= self.ES_DISPLAY_REQUIRED
            
            self._kernel32.SetThreadExecutionState(flags)
        except Exception as exc:
            # Best-effort: do not fail the job for sleep prevention issues.
            logger.debug("PreventSleep failed to enable (%s): %s", self.reason, exc)
        
        return self
    
    def __exit__(self, *args):
        if self._kernel32 is None:
            return
        
        try:
            # Clear the flags (allow sleep again)
            self._kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        except Exception as exc:
            logger.debug("PreventSleep failed to disable (%s): %s", self.reason, exc)
