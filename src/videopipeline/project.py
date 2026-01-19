from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .ffmpeg import ffprobe_duration_seconds


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint_file(
    path: Path,
    *,
    head_bytes: int = 2 * 1024 * 1024,
    tail_bytes: int = 2 * 1024 * 1024,
) -> str:
    """Compute a fast, content-informed fingerprint for large media.

    Reads:
      - file size
      - first `head_bytes`
      - last `tail_bytes` (if present)

    This avoids hashing the entire file while remaining stable across renames.
    """
    path = Path(path)
    st = path.stat()
    size = st.st_size

    h = hashlib.sha256()
    h.update(str(size).encode("utf-8"))

    with path.open("rb") as f:
        head = f.read(head_bytes)
        h.update(head)

        if size > tail_bytes:
            try:
                f.seek(max(0, size - tail_bytes))
                tail = f.read(tail_bytes)
                h.update(tail)
            except OSError:
                # Some file-like sources may not seek well; ignore tail.
                pass

    return h.hexdigest()


def default_projects_root() -> Path:
    return Path("outputs") / "projects"


def project_dir_for_video(video_path: Path, projects_root: Optional[Path] = None) -> Path:
    projects_root = projects_root or default_projects_root()
    pid = fingerprint_file(video_path)
    return projects_root / pid


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


@dataclass
class Project:
    project_dir: Path
    video_path: Path

    @property
    def project_json_path(self) -> Path:
        return self.project_dir / "project.json"

    @property
    def analysis_dir(self) -> Path:
        return self.project_dir / "analysis"

    @property
    def audio_features_path(self) -> Path:
        return self.analysis_dir / "audio_features.npz"

    @property
    def motion_features_path(self) -> Path:
        return self.analysis_dir / "motion_features.npz"

    @property
    def chat_features_path(self) -> Path:
        return self.analysis_dir / "chat_features.npz"

    @property
    def highlights_features_path(self) -> Path:
        return self.analysis_dir / "highlights_features.npz"

    @property
    def scenes_path(self) -> Path:
        return self.analysis_dir / "scenes.json"

    @property
    def chat_raw_path(self) -> Path:
        return self.analysis_dir / "chat_raw.json"

    @property
    def exports_dir(self) -> Path:
        return self.project_dir / "exports"


def create_or_load_project(video_path: Path, projects_root: Optional[Path] = None) -> Project:
    video_path = Path(video_path).expanduser().resolve()
    pdir = project_dir_for_video(video_path, projects_root)
    pdir.mkdir(parents=True, exist_ok=True)

    proj = Project(project_dir=pdir, video_path=video_path)

    if not proj.project_json_path.exists():
        duration_s = ffprobe_duration_seconds(video_path)
        st = video_path.stat()
        initial = {
            "project_id": pdir.name,
            "created_at": utc_now_iso(),
            "video": {
                "path": str(video_path),
                "size_bytes": st.st_size,
                "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
                "duration_seconds": duration_s,
            },
            "analysis": {},
            "layout": {},
            "selections": [],
            "exports": [],
        }
        save_json(proj.project_json_path, initial)

    return proj


def update_project(proj: Project, updater) -> Dict[str, Any]:
    data = load_json(proj.project_json_path)
    updater(data)
    save_json(proj.project_json_path, data)
    return data


def get_project_data(proj: Project) -> Dict[str, Any]:
    return load_json(proj.project_json_path)


def add_selection(
    proj: Project,
    *,
    start_s: float,
    end_s: float,
    title: str = "",
    notes: str = "",
    template: str = "vertical_blur",
) -> Dict[str, Any]:
    import uuid

    sel_id = uuid.uuid4().hex

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("selections", [])
        d["selections"].append(
            {
                "id": sel_id,
                "created_at": utc_now_iso(),
                "start_s": float(start_s),
                "end_s": float(end_s),
                "title": title,
                "notes": notes,
                "template": template,
            }
        )

    return update_project(proj, _upd)


def add_selection_from_candidate(
    proj: Project,
    *,
    candidate: Dict[str, Any],
    template: str,
    title: str = "",
) -> str:
    import uuid

    sel_id = uuid.uuid4().hex

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("selections", [])
        d["selections"].append(
            {
                "id": sel_id,
                "created_at": utc_now_iso(),
                "start_s": float(candidate.get("start_s")),
                "end_s": float(candidate.get("end_s")),
                "title": title,
                "notes": "",
                "template": template,
                "candidate_rank": candidate.get("rank"),
                "candidate_score": candidate.get("score"),
                "candidate_peak_time_s": candidate.get("peak_time_s"),
            }
        )

    update_project(proj, _upd)
    return sel_id


def set_layout_facecam(proj: Project, *, rect: Dict[str, float]) -> Dict[str, Any]:
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("layout", {})
        d["layout"]["facecam"] = rect

    return update_project(proj, _upd)


def update_selection(proj: Project, selection_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    def _upd(d: Dict[str, Any]) -> None:
        sels = d.get("selections", [])
        for s in sels:
            if s.get("id") == selection_id:
                s.update(patch)
                s["updated_at"] = utc_now_iso()
                return
        raise KeyError(f"Selection not found: {selection_id}")

    return update_project(proj, _upd)


def remove_selection(proj: Project, selection_id: str) -> Dict[str, Any]:
    def _upd(d: Dict[str, Any]) -> None:
        before = d.get("selections", [])
        after = [s for s in before if s.get("id") != selection_id]
        d["selections"] = after

    return update_project(proj, _upd)


def record_export(
    proj: Project,
    *,
    selection_id: str,
    output_path: Path,
    template: str,
    with_captions: bool,
    status: str,
) -> Dict[str, Any]:
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("exports", [])
        d["exports"].append(
            {
                "created_at": utc_now_iso(),
                "selection_id": selection_id,
                "output": str(output_path),
                "template": template,
                "with_captions": bool(with_captions),
                "status": status,
            }
        )

    return update_project(proj, _upd)
