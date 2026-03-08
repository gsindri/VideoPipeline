"""Analysis DAG system for VideoPipeline.

This package provides a task-based analysis pipeline that:
- Declares explicit dependencies between analysis steps
- Supports incremental computation (skips fresh artifacts)
- Enables progressive refinement (early audio+chat, later video+scenes)
- Guarantees boundary graph before final clip shaping

Key modules:
- artifacts: Artifact definitions and freshness checking
- tasks: Task registry with all analysis tasks
- runner: DAG executor with dependency resolution
- bundles: Pre-defined target bundles (pre_download, full)
"""

from .artifacts import (
    ARTIFACTS,
    Artifact,
    ArtifactState,
    get_artifact_state,
    is_artifact_fresh,
    save_artifact_state,
)
from .bundles import (
    BUNDLE_EXPORT,
    BUNDLE_FULL,
    BUNDLE_PRE_DOWNLOAD,
    BUNDLE_PRE_DOWNLOAD_NO_CHAT,
    BUNDLE_TRANSCRIPT,
    get_bundle,
    list_bundles,
)
from .runner import AnalysisResult, AnalysisRunner, TaskResult, run_analysis, task_lock
from .tasks import Task, task_registry

__all__ = [
    # Artifacts
    "Artifact",
    "ArtifactState",
    "ARTIFACTS",
    "get_artifact_state",
    "save_artifact_state",
    "is_artifact_fresh",
    # Tasks
    "Task",
    "task_registry",
    # Runner
    "run_analysis",
    "AnalysisRunner",
    "AnalysisResult",
    "TaskResult",
    "task_lock",
    # Bundles
    "BUNDLE_PRE_DOWNLOAD",
    "BUNDLE_PRE_DOWNLOAD_NO_CHAT",
    "BUNDLE_FULL",
    "BUNDLE_EXPORT",
    "BUNDLE_TRANSCRIPT",
    "get_bundle",
    "list_bundles",
]
