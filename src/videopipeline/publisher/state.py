from __future__ import annotations

import os
from pathlib import Path


def state_dir() -> Path:
    if os.name == "nt":
        base = os.getenv("APPDATA")
        if base:
            return Path(base) / "VideoPipeline"
        return Path.home() / "AppData" / "Roaming" / "VideoPipeline"
    return Path.home() / ".videopipeline"


def ensure_state_dir() -> Path:
    path = state_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def accounts_path() -> Path:
    return ensure_state_dir() / "accounts.json"


def publisher_db_path() -> Path:
    return ensure_state_dir() / "publisher.sqlite"


def logs_dir() -> Path:
    path = ensure_state_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
