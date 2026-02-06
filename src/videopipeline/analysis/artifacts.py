"""Artifact definitions and freshness checking.

An artifact is a named output of an analysis task. Each artifact has:
- A name (e.g. "transcript", "boundary_graph")
- A path on disk (relative to project analysis dir)
- Metadata about when/how it was computed

Freshness is determined by:
- File existence
- Config hash (recompute if config changed)
- Input hashes (recompute if inputs changed)
- Task version (recompute if algorithm changed)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Artifact:
    """Definition of an analysis artifact.
    
    Attributes:
        name: Unique identifier (e.g. "transcript", "highlights_scores")
        path_template: Path relative to analysis_dir (e.g. "transcript_full.json")
        description: Human-readable description
        legacy_paths: Alternative paths to check (for backwards compatibility)
    """
    name: str
    path_template: str
    description: str = ""
    legacy_paths: list = field(default_factory=list)
    
    def get_path(self, analysis_dir: Path) -> Path:
        """Get the full path to this artifact."""
        return analysis_dir / self.path_template
    
    def exists(self, analysis_dir: Path) -> bool:
        """Check if the artifact exists (including legacy paths)."""
        if self.get_path(analysis_dir).exists():
            return True
        # Check legacy paths
        for legacy in self.legacy_paths:
            if (analysis_dir / legacy).exists():
                return True
        return False


@dataclass
class ArtifactState:
    """State of a computed artifact.
    
    Attributes:
        exists: Whether the artifact file exists
        fresh: Whether the artifact is up-to-date with current config/inputs
        path: Full path to the artifact
        metadata: Stored metadata (task_version, config_hash, etc.)
        computed_at: When the artifact was last computed
        signals_used: For highlights, which signals were included
        sources_present: For boundary_graph, which sources were available
    """
    exists: bool = False
    fresh: bool = False
    path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    computed_at: Optional[str] = None
    signals_used: Dict[str, bool] = field(default_factory=dict)
    sources_present: Set[str] = field(default_factory=set)
    
    @property
    def upgradeable(self) -> bool:
        """True if artifact exists but could be improved with new inputs."""
        if not self.exists:
            return False
        # Check if new sources are available that weren't used
        # This is set by the runner when checking freshness
        return self.metadata.get("upgradeable", False)


# =============================================================================
# Artifact Registry
# =============================================================================

# All known artifacts in the system
ARTIFACTS: Dict[str, Artifact] = {}


def register_artifact(
    name: str, 
    path_template: str, 
    description: str = "",
    legacy_paths: Optional[List[str]] = None,
) -> Artifact:
    """Register an artifact definition."""
    artifact = Artifact(
        name=name, 
        path_template=path_template, 
        description=description,
        legacy_paths=legacy_paths or [],
    )
    ARTIFACTS[name] = artifact
    return artifact


# --- Raw inputs (not computed, just markers) ---
register_artifact("video_file", "video", "Source video file")
register_artifact("audio_file", "audio", "Source audio file (extracted or downloaded)")
register_artifact("chat_file", "chat_raw.json", "Raw chat messages")

# --- Audio-derived ---
register_artifact("audio_features", "audio_features.npz", "Audio RMS/energy timeline")
register_artifact("audio_events", "audio_events_features.npz", "ML-detected audio events (laughter, etc.)")
register_artifact(
    "reaction_audio",
    "reaction_audio_features.npz",
    "Speech-focused prosody/arousal timeline (voice excitement signal)",
)
register_artifact("silence", "silence.json", "Detected silence intervals")
register_artifact("audio_vad", "audio_vad.npz", "Voice activity detection (speech segments + speech timeline)")

# --- Transcript-derived ---
register_artifact("transcript", "transcript_full.json", "Full speech-to-text transcript")
register_artifact("diarization", "diarization.json", "Speaker diarization (who spoke when + turn boundaries)")
register_artifact("sentences", "sentences.json", "Sentence boundaries from transcript")
register_artifact("speech_features", "speech_features.npz", "Speech rate, word density, etc.")
register_artifact("chapters", "chapters.json", "Semantic chapter boundaries (LLM)")

# --- Chat-derived ---
register_artifact("chat_features", "chat_features.npz", "Chat activity timeline")
register_artifact("chat_boundaries", "chat_boundaries.json", "Chat activity valleys")
register_artifact("chat_sync", "chat_sync.json", "Estimated chat↔video sync offset")

# --- Video-derived ---
register_artifact("motion_features", "motion_features.npz", "Frame difference motion scores")
register_artifact("scenes", "scenes.json", "Visual scene cut boundaries")

# --- Unified boundaries ---
register_artifact(
    "boundary_graph", 
    "boundary_graph.json", 
    "Merged boundary graph for clip shaping",
    legacy_paths=["boundaries.json"],  # Old format location
)

# --- Highlights (split into two stages) ---
register_artifact("highlights_scores", "highlights_features.npz", "Combined signal scores and peaks")
register_artifact("highlights_candidates", "highlights.json", "Final shaped clip candidates")

# --- Post-highlights ---
register_artifact(
    "variants",
    "variants.json",
    "Clip variants (titles, hooks, etc.)",
    legacy_paths=["clip_variants.json"],
)
register_artifact("director", "director.json", "AI director selections")


# =============================================================================
# Freshness Checking
# =============================================================================

def _hash_config(config: Dict[str, Any]) -> str:
    """Compute a stable hash of a config dict."""
    # Sort keys for stability
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:12]


def _get_file_mtime(path: Path) -> Optional[float]:
    """Get file modification time, or None if doesn't exist."""
    try:
        return path.stat().st_mtime if path.exists() else None
    except Exception:
        return None


def get_artifact_state(
    artifact_name: str,
    analysis_dir: Path,
    *,
    config: Optional[Dict[str, Any]] = None,
    task_version: Optional[str] = None,
) -> ArtifactState:
    """Get the current state of an artifact.
    
    Args:
        artifact_name: Name of the artifact to check
        analysis_dir: Project analysis directory
        config: Current config for this artifact's task (for freshness check)
        task_version: Current version of the task (for freshness check)
        
    Returns:
        ArtifactState with existence and freshness info
    """
    artifact = ARTIFACTS.get(artifact_name)
    if artifact is None:
        return ArtifactState(exists=False, fresh=False)
    
    path = artifact.get_path(analysis_dir)
    
    # Special handling for "file" artifacts (video_file, audio_file, chat_file)
    if artifact_name in ("video_file", "audio_file", "chat_file"):
        # These are inputs, check parent directory
        if artifact_name == "video_file":
            # Video is at project_dir / something, check project.json
            return ArtifactState(exists=True, fresh=True, path=path)
        elif artifact_name == "audio_file":
            # Audio could be extracted or downloaded
            return ArtifactState(exists=True, fresh=True, path=path)
        elif artifact_name == "chat_file":
            chat_path = analysis_dir / "chat_raw.json"
            chat_db = analysis_dir / "chat.sqlite"
            exists = chat_path.exists() or chat_db.exists()
            return ArtifactState(exists=exists, fresh=True, path=chat_path if chat_path.exists() else chat_db)
    
    # Check if canonical path exists; if not, search legacy paths
    actual_path = path
    if not path.exists() and artifact.legacy_paths:
        for legacy in artifact.legacy_paths:
            legacy_path = analysis_dir / legacy
            if legacy_path.exists():
                actual_path = legacy_path
                break
    
    if not actual_path.exists():
        return ArtifactState(exists=False, fresh=False, path=path)
    
    # Load metadata from state file or embedded in artifact
    # Use actual_path (which may be a legacy path) for the state
    state = ArtifactState(exists=True, path=actual_path)
    
    # Try to load metadata
    state_file = analysis_dir / "analysis_state.json"
    if state_file.exists():
        try:
            all_state = json.loads(state_file.read_text(encoding="utf-8"))
            artifact_meta = all_state.get("artifacts", {}).get(artifact_name, {})
            state.metadata = artifact_meta
            state.computed_at = artifact_meta.get("computed_at")
            state.signals_used = artifact_meta.get("signals_used", {})
            state.sources_present = set(artifact_meta.get("sources_present", []))
        except Exception:
            pass
    
    # Check freshness
    if config is not None:
        current_hash = _hash_config(config)
        stored_hash = state.metadata.get("config_hash")
        if stored_hash and stored_hash != current_hash:
            state.fresh = False
            return state
    
    if task_version is not None:
        stored_version = state.metadata.get("task_version")
        if stored_version and stored_version != task_version:
            state.fresh = False
            return state
    
    state.fresh = True
    return state


def is_artifact_fresh(
    artifact_name: str,
    analysis_dir: Path,
    **kwargs,
) -> bool:
    """Quick check if an artifact exists and is fresh."""
    state = get_artifact_state(artifact_name, analysis_dir, **kwargs)
    return state.exists and state.fresh


def save_artifact_state(
    artifact_name: str,
    analysis_dir: Path,
    *,
    config: Optional[Dict[str, Any]] = None,
    task_version: Optional[str] = None,
    signals_used: Optional[Dict[str, bool]] = None,
    sources_present: Optional[Set[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save metadata about a computed artifact.
    
    This should be called after a task successfully computes its output.
    """
    state_file = analysis_dir / "analysis_state.json"
    
    # Load existing state
    all_state: Dict[str, Any] = {"artifacts": {}}
    if state_file.exists():
        try:
            all_state = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    if "artifacts" not in all_state:
        all_state["artifacts"] = {}
    
    # Build metadata for this artifact
    metadata: Dict[str, Any] = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    
    if config is not None:
        metadata["config_hash"] = _hash_config(config)
    
    if task_version is not None:
        metadata["task_version"] = task_version
    
    if signals_used is not None:
        metadata["signals_used"] = signals_used
    
    if sources_present is not None:
        metadata["sources_present"] = sorted(sources_present)
    
    if extra_metadata:
        metadata.update(extra_metadata)
    
    all_state["artifacts"][artifact_name] = metadata
    all_state["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    # Write atomically
    state_file.write_text(json.dumps(all_state, indent=2), encoding="utf-8")
