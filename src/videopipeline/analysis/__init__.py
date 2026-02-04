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
    Artifact,
    ArtifactState,
    ARTIFACTS,
    get_artifact_state,
    save_artifact_state,
    is_artifact_fresh,
)
from .tasks import Task, task_registry
from .runner import run_analysis, AnalysisRunner, AnalysisResult, TaskResult, task_lock
from .bundles import (
    BUNDLE_PRE_DOWNLOAD,
    BUNDLE_PRE_DOWNLOAD_NO_CHAT,
    BUNDLE_FULL,
    BUNDLE_EXPORT,
    BUNDLE_TRANSCRIPT,
    get_bundle,
    list_bundles,
)

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
