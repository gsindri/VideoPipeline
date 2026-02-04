"""Home screen utilities for Studio."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..project import default_projects_root, load_json
from ..utils import subprocess_flags as _subprocess_flags


def windows_open_video_dialog() -> Optional[str]:
    """Open a native Windows file dialog for selecting a video file.

    Returns the selected file path, or None if cancelled.
    Uses PowerShell with System.Windows.Forms.OpenFileDialog.
    """
    if sys.platform != "win32":
        return None

    ps = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = "Select a video"
$dialog.Filter = "Video Files|*.mp4;*.mov;*.mkv;*.webm;*.avi|All Files|*.*"
$dialog.Multiselect = $false
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
  Write-Output $dialog.FileName
}
"""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-STA", "-Command", ps],
            text=True,
            stderr=subprocess.STDOUT,
            **_subprocess_flags(),
        ).strip()
        return out or None
    except Exception:
        return None


def list_recent_projects(
    projects_root: Optional[Path] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Scan projects directory and return recent projects sorted by update time.

    Returns a list of project summaries with:
      - project_id
      - video_path
      - video_name
      - duration_seconds
      - updated_at (mtime of project.json)
      - selections_count
      - exports_count
      - favorite
    """
    projects_root = projects_root or default_projects_root()
    if not projects_root.exists():
        return []

    projects: List[Dict[str, Any]] = []

    for pdir in projects_root.iterdir():
        if not pdir.is_dir():
            continue
        pjson = pdir / "project.json"
        if not pjson.exists():
            continue

        try:
            data = load_json(pjson)
            video_info = data.get("video", {})
            video_path = video_info.get("path", "")

            # Get mtime for sorting
            mtime = pjson.stat().st_mtime

            projects.append({
                "project_id": data.get("project_id", pdir.name),
                "project_dir": str(pdir),
                "video_path": video_path,
                "video_name": Path(video_path).name if video_path else "",
                "duration_seconds": video_info.get("duration_seconds", 0),
                "created_at": data.get("created_at", ""),
                "updated_at": mtime,
                "selections_count": len(data.get("selections", [])),
                "exports_count": len(data.get("exports", [])),
                "favorite": data.get("favorite", False),
            })
        except Exception:
            continue

    # Sort by updated_at descending (most recent first)
    projects.sort(key=lambda p: p["updated_at"], reverse=True)

    return projects[:limit]


def check_video_exists(video_path: str) -> bool:
    """Check if a video file exists at the given path."""
    if not video_path:
        return False
    return Path(video_path).exists()
