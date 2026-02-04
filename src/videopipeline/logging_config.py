"""Centralized logging configuration for VideoPipeline.

Usage:
    from videopipeline.logging_config import setup_logging
    setup_logging()  # Call once at startup
    
    # Then in any module:
    import logging
    log = logging.getLogger("videopipeline.mymodule")
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional


_CONFIGURED = False


def _parse_module_levels(spec: str) -> Dict[str, int]:
    """Parse per-module logger levels from an env-var style string.

    Format:
        VP_LOG_MODULE_LEVELS="videopipeline.chat=DEBUG,videopipeline.studio=INFO"

    Notes:
      - Names not starting with "videopipeline" are auto-prefixed.
      - Separators: comma/semicolon. Assignment: "=" or ":".
      - Invalid entries are ignored.
    """
    out: Dict[str, int] = {}
    if not spec:
        return out
    for part in re.split(r"[;,]+", spec):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, level_str = part.split("=", 1)
        elif ":" in part:
            name, level_str = part.split(":", 1)
        else:
            continue
        name = name.strip()
        level_str = level_str.strip().upper()
        if not name or not level_str:
            continue
        if not name.startswith("videopipeline"):
            name = f"videopipeline.{name}"
        level = getattr(logging, level_str, None)
        if isinstance(level, int):
            out[name] = level
    return out


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure logging for the videopipeline package.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        format_string: Custom format string (default uses a standard format)
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Configure the root videopipeline logger
    logger = logging.getLogger("videopipeline")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    # Keep handler permissive so per-module overrides can enable DEBUG without
    # globally switching everything to DEBUG.
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    
    # Don't propagate to root logger
    logger.propagate = False

    # Optional per-module overrides.
    # Example:
    #   VP_LOG_MODULE_LEVELS="videopipeline.chat=DEBUG,videopipeline.studio=DEBUG"
    module_levels = _parse_module_levels(os.getenv("VP_LOG_MODULE_LEVELS", ""))
    for name, lvl in module_levels.items():
        logging.getLogger(name).setLevel(lvl)
    
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.
    
    If name doesn't start with 'videopipeline', it will be prefixed.
    
    Args:
        name: Module name (e.g., 'studio.app' or 'videopipeline.studio.app')
    
    Returns:
        Configured logger instance
    """
    if not name.startswith("videopipeline"):
        name = f"videopipeline.{name}"
    return logging.getLogger(name)
