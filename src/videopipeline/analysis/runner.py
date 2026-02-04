"""DAG runner for analysis tasks.

The runner:
- Resolves dependencies recursively
- Skips tasks whose outputs already exist and are fresh
- Supports progressive refinement (upgrade triggers)
- Reports progress across all tasks
- Can run in "partial" mode (pre-download) or "full" mode
- Uses file-based locking to prevent race conditions
- Runs independent tasks in PARALLEL for faster analysis
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set

from ..project import Project
from .artifacts import (
    get_artifact_state,
    save_artifact_state,
    is_artifact_fresh,
    ARTIFACTS,
)
from .tasks import task_registry, Task

logger = logging.getLogger(__name__)


# ============================================================================
# Task Locking - Prevent race conditions when multiple processes run analysis
# ============================================================================

@contextmanager
def task_lock(locks_dir: Path, task_name: str, timeout: float = 300.0) -> Generator[bool, None, None]:
    """File-based lock for a task to prevent concurrent execution.
    
    Args:
        locks_dir: Directory for lock files
        task_name: Name of the task to lock
        timeout: Max seconds to wait for lock (default 5 minutes)
        
    Yields:
        True if lock acquired, False if timed out
    """
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_file = locks_dir / f"{task_name}.lock"
    
    start = time.time()
    acquired = False
    
    while time.time() - start < timeout:
        try:
            # Try to create lock file exclusively
            # On Windows, 'x' mode fails if file exists
            with open(lock_file, 'x') as f:
                f.write(f"{time.time()}\n")
            acquired = True
            break
        except FileExistsError:
            # Lock exists - check if it's stale (older than timeout)
            try:
                lock_time = float(lock_file.read_text().strip())
                if time.time() - lock_time > timeout:
                    # Stale lock, remove it
                    lock_file.unlink(missing_ok=True)
                    continue
            except (ValueError, OSError):
                # Corrupted lock file, remove it
                lock_file.unlink(missing_ok=True)
                continue
            
            # Wait and retry
            time.sleep(0.5)
    
    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock_file.unlink(missing_ok=True)
            except OSError:
                pass


@dataclass
class TaskResult:
    """Result of running a single task."""
    task_name: str
    status: str  # "completed", "skipped", "failed", "disabled"
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    artifacts_produced: Set[str] = field(default_factory=set)
    signals_used: Dict[str, bool] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result of running the full analysis pipeline."""
    targets: Set[str]
    tasks_run: List[TaskResult]
    total_elapsed_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    missing_targets: Set[str] = field(default_factory=set)


class AnalysisRunner:
    """Executes analysis tasks in dependency order.
    
    The runner supports two main modes:
    
    1. Target-based: Specify artifact targets, runner computes dependencies
       runner.run(targets={"highlights_candidates"})
       
    2. Bundle-based: Use pre-defined bundles
       runner.run(bundle="pre_download")  # Audio+chat based early analysis
       runner.run(bundle="full")          # Full analysis with video
    
    The runner also supports progressive refinement:
    - If a task has upgrade_triggers and those become available, it re-runs
    - This allows early "good enough" results that improve later
    """
    
    def __init__(
        self,
        proj: Project,
        config: Dict[str, Any],
        *,
        on_progress: Optional[Callable[[float, str], None]] = None,
        llm_complete: Optional[Callable[[str], str]] = None,
        upgrade_mode: bool = False,
    ):
        """Initialize the runner.
        
        Args:
            proj: Project to analyze
            config: Merged config dict with sub-configs for each task
            on_progress: Callback(fraction, message) for progress reporting
            llm_complete: Optional LLM completion function for semantic tasks
            upgrade_mode: If True, re-run tasks that have upgrade triggers satisfied
        """
        self.proj = proj
        self.config = config
        self.on_progress = on_progress
        self.llm_complete = llm_complete
        self.upgrade_mode = upgrade_mode
        
        # Inject LLM function into config for tasks that need it
        if llm_complete:
            self.config["_llm_complete"] = llm_complete
        
        # Track what we've computed this run (thread-safe via lock)
        import threading
        self._computed: Set[str] = set()
        self._computed_lock = threading.Lock()
        self._results: List[TaskResult] = []
        
        # Logger for debug output
        self._log = logger
    
    def _artifact_exists(self, artifact: str) -> bool:
        """Check if an artifact exists (file on disk)."""
        artifact_def = ARTIFACTS.get(artifact)
        if artifact_def is None:
            logger.debug(f"[DAG] Artifact '{artifact}' not found in registry")
            return False
        
        # Use the artifact's exists() method which checks legacy paths too
        exists = artifact_def.exists(self.proj.analysis_dir)
        path = artifact_def.get_path(self.proj.analysis_dir)
        logger.debug(f"[DAG] Artifact '{artifact}' -> {path.name}: exists={exists}")
        return exists

    def _get_gpu_concurrency(self) -> int:
        """Return max number of concurrent GPU-heavy tasks.

        This is a pragmatic scheduling knob to avoid GPU contention (e.g. running
        Whisper transcription and diarization concurrently on the same device).

        Config:
          - analysis.dag.gpu_concurrency (int)
          - VP_DAG_GPU_CONCURRENCY (env var override)

        A value <= 0 disables the limit.
        """
        # Env override first (easy to A/B test without editing profiles).
        env = os.environ.get("VP_DAG_GPU_CONCURRENCY", "").strip()
        if env != "":
            try:
                v = int(env)
                return v
            except Exception:
                logger.warning("[DAG] Invalid VP_DAG_GPU_CONCURRENCY=%r (expected int)", env)

        try:
            v = int((self.config.get("dag", {}) or {}).get("gpu_concurrency", 1))
            return v
        except Exception:
            return 1

    def _task_uses_gpu_heavy(self, task: Task) -> bool:
        """Heuristic: True if this task is expected to use the primary GPU heavily.

        We keep this conservative (only tasks known to be GPU-intensive) to avoid
        serializing lightweight GPU use unnecessarily.
        """
        name = task.name

        def _is_cpu_device(device: Any) -> bool:
            d = str(device or "").strip().lower()
            return d == "cpu"

        if name == "transcript":
            speech_cfg = self.config.get("speech", {}) or {}
            if not bool(speech_cfg.get("use_gpu", True)):
                return False
            return not _is_cpu_device(speech_cfg.get("device", "cuda"))

        if name == "diarization":
            # Mirrors DiarizationConfig.from_dict behavior (device=cpu forces use_gpu=False).
            speech_cfg = self.config.get("speech", {}) or {}
            diar_cfg = self.config.get("diarization", {}) or {}
            merged: Dict[str, Any] = {**speech_cfg, **diar_cfg}
            device = merged.get("device", "")
            if _is_cpu_device(device):
                return False
            return bool(merged.get("use_gpu", True))

        return False
    
    def _should_run_task(self, task: Task) -> tuple[bool, str]:
        """Determine if a task should run.
        
        Returns:
            (should_run, reason)
        """
        # Check if task is enabled
        if not task.is_enabled(self.config):
            return False, "disabled by config"
        
        # Check if all outputs already exist
        all_outputs_exist = all(
            self._artifact_exists(a) or a in self._computed
            for a in task.produces
        )
        
        if all_outputs_exist:
            # Check upgrade triggers in upgrade mode
            if self.upgrade_mode and task.upgrade_triggers:
                # Only consider triggers that actually exist NOW (not just in self._computed)
                # This prevents "inventing" missing triggers
                available_triggers = {
                    t for t in task.upgrade_triggers
                    if self._artifact_exists(t)
                }
                logger.debug(
                    f"[Upgrade check] {task.name}: triggers={task.upgrade_triggers}, "
                    f"available={available_triggers}"
                )
                if available_triggers:
                    # Check if task was run without these triggers before
                    state = get_artifact_state(
                        list(task.produces)[0],
                        self.proj.analysis_dir,
                    )
                    sources = state.sources_present or set()
                    new_sources = available_triggers - sources
                    logger.debug(
                        f"[Upgrade check] {task.name}: sources_present={sources}, "
                        f"new_sources={new_sources}"
                    )
                    if new_sources:
                        return True, f"upgrade: new sources {new_sources}"
            
            return False, "outputs exist"
        
        return True, "outputs missing"
    
    def _can_run_task(
        self, 
        task: Task, 
        will_be_produced: Optional[Set[str]] = None,
    ) -> tuple[bool, str]:
        """Check if task's dependencies are satisfied.
        
        Args:
            task: Task to check
            will_be_produced: Set of artifacts that will be produced by other 
                tasks in this DAG run. Used during planning phase.
        
        Returns:
            (can_run, reason)
        """
        if will_be_produced is None:
            will_be_produced = set()
            
        missing = []
        for req in task.requires:
            exists = (
                self._artifact_exists(req) 
                or req in self._computed 
                or req in will_be_produced
            )
            if not exists:
                missing.append(req)
        
        if missing:
            if task.can_run_partial:
                # Check if at least minimum requirements are met
                # For now, require all non-optional inputs
                required = task.requires - task.optional_inputs
                missing_required = [r for r in required if r in missing]
                if missing_required:
                    return False, f"missing required: {missing_required}"
                return True, f"partial: missing optional {missing}"
            return False, f"missing: {missing}"
        
        return True, "dependencies satisfied"
    
    def _group_tasks_by_level(
        self, 
        tasks_to_run: List[tuple[Task, str]]
    ) -> List[List[tuple[Task, str]]]:
        """Group tasks into levels for parallel execution.
        
        Tasks in the same level have no dependencies on each other and can
        run in parallel. Level N tasks depend only on outputs from levels < N.
        
        Args:
            tasks_to_run: List of (task, reason) tuples in dependency order
            
        Returns:
            List of levels, where each level is a list of (task, reason) tuples
        """
        if not tasks_to_run:
            return []
        
        # Build a set of all artifacts that will be produced by tasks in this run
        will_be_produced: Set[str] = set()
        for task, _ in tasks_to_run:
            will_be_produced.update(task.produces)
        
        # Track which artifacts are "available" (already exist AND won't be re-produced)
        # If a task is going to re-run (e.g., for upgrade), its outputs are NOT available
        # until that task completes
        available: Set[str] = set(self._computed)
        for artifact in ARTIFACTS:
            if self._artifact_exists(artifact):
                # Only mark as available if it won't be re-produced by a task in this run
                if artifact not in will_be_produced:
                    available.add(artifact)
        
        levels: List[List[tuple[Task, str]]] = []
        remaining = list(tasks_to_run)
        
        while remaining:
            # Find all tasks whose dependencies are satisfied
            current_level: List[tuple[Task, str]] = []
            still_remaining: List[tuple[Task, str]] = []
            
            for task, reason in remaining:
                # Check if all required dependencies are available
                required_deps = task.requires - task.optional_inputs
                deps_satisfied = all(dep in available for dep in required_deps)
                
                # ALSO check: if any optional_input is being produced in this run,
                # wait for it to be available before running this task.
                # This ensures we get the best quality results when both tasks are scheduled.
                optional_deps_being_produced = task.optional_inputs & will_be_produced
                optional_deps_ready = all(dep in available for dep in optional_deps_being_produced)
                
                if deps_satisfied and optional_deps_ready:
                    current_level.append((task, reason))
                else:
                    still_remaining.append((task, reason))
            
            if not current_level:
                # No progress possible - shouldn't happen if dependencies are correct
                # Fall back to running remaining tasks sequentially
                logger.warning(f"[DAG] Could not resolve parallel levels for: {[t.name for t, _ in still_remaining]}")
                for task, reason in still_remaining:
                    levels.append([(task, reason)])
                break
            
            levels.append(current_level)
            
            # Mark outputs from this level as available for next level
            for task, _ in current_level:
                available.update(task.produces)
            
            remaining = still_remaining
        
        return levels
    
    def _run_task(
        self,
        task: Task,
        task_progress_base: float,
        task_progress_weight: float,
    ) -> TaskResult:
        """Execute a single task.
        
        Args:
            task: Task to run
            task_progress_base: Base progress (0-1) for this task
            task_progress_weight: Weight of this task in total progress
            
        Returns:
            TaskResult
        """
        import time
        
        result = TaskResult(task_name=task.name, status="pending")
        
        # Create progress wrapper that accepts optional message
        def task_progress(frac: float, msg: Optional[str] = None) -> None:
            if self.on_progress:
                overall = task_progress_base + frac * task_progress_weight
                try:
                    p = float(frac)
                except Exception:
                    p = 0.0
                p = max(0.0, min(1.0, p))
                pct = int(p * 100)
                if msg:
                    self.on_progress(overall, f"{task.name}: {pct}% {msg}")
                else:
                    self.on_progress(overall, f"{task.name}: {pct}%")
        
        # Use file-based lock to prevent race conditions
        locks_dir = self.proj.analysis_dir / ".locks"
        
        with task_lock(locks_dir, task.name) as acquired:
            if not acquired:
                logger.warning(f"[DAG] Could not acquire lock for '{task.name}', skipping")
                result.status = "skipped"
                result.error = "lock timeout"
                return result
            
            # Re-check whether the task still needs to run now that we hold the lock.
            # This matters for upgrade-mode runs where outputs exist but we still want to
            # rerun to incorporate newly-available optional inputs.
            should_run_now, reason_now = self._should_run_task(task)
            if not should_run_now:
                logger.info(
                    f"[DAG] Task '{task.name}' no longer needs to run ({reason_now}), skipping"
                )
                result.status = "skipped"
                # Only mark outputs computed if they actually exist.
                with self._computed_lock:
                    for a in task.produces:
                        if self._artifact_exists(a) or a in self._computed:
                            self._computed.add(a)
                return result
            
            try:
                start = time.time()
                
                # Run the task
                task.run(self.proj, self.config, task_progress)
                
                result.elapsed_seconds = time.time() - start
                # Verify the produced artifact files actually exist before marking as completed.
                # Tasks can "return early" (e.g., no chat data) and produce nothing.
                # Only mark as computed if the output file exists.
                actually_produced: Set[str] = set()
                for artifact in task.produces:
                    if self._artifact_exists(artifact):
                        actually_produced.add(artifact)
                
                if actually_produced:
                    result.status = "completed"
                    result.artifacts_produced = actually_produced
                    
                    # Mark only actually produced artifacts as computed (thread-safe)
                    with self._computed_lock:
                        self._computed.update(actually_produced)
                    
                    # Save state for each produced artifact
                    for artifact in actually_produced:
                        # Determine which sources were present for this run.
                        # IMPORTANT: Only use _artifact_exists (file existence), not self._computed.
                        # This prevents marking sources as "present" when they didn't produce a file.
                        sources_present: Set[str] = set()
                        for opt in task.optional_inputs:
                            if self._artifact_exists(opt):
                                sources_present.add(opt)
                        
                        save_artifact_state(
                            artifact,
                            self.proj.analysis_dir,
                            config=self.config.get(task.name, {}),
                            task_version=task.version,
                            sources_present=sources_present,
                        )
                    
                else:
                    # Task ran but didn't produce any output files (e.g., no chat data)
                    result.status = "skipped"
                    result.error = "no outputs produced"
                
            except Exception as e:
                result.status = "failed"
                result.error = str(e)
        
        return result
    
    def _resolve_dependencies(self, targets: Set[str]) -> List[Task]:
        """Resolve task dependencies and return execution order.
        
        Args:
            targets: Set of artifact names to produce
            
        Returns:
            List of tasks in dependency order (topological sort)
        """
        # Build set of all tasks needed
        needed_tasks: Set[str] = set()
        
        def add_task_and_deps(artifact: str) -> None:
            task = task_registry.get_by_artifact(artifact)
            if task is None:
                return  # Unknown artifact, treat as external input
            
            if task.name in needed_tasks:
                return
            
            # Add dependencies first
            for dep in task.requires:
                add_task_and_deps(dep)
            
            # Add optional dependencies if they exist OR are in targets
            # This ensures boundary_graph gets computed when requested
            for opt in task.optional_inputs:
                if self._artifact_exists(opt) or opt in targets:
                    add_task_and_deps(opt)
            
            needed_tasks.add(task.name)
        
        for target in targets:
            add_task_and_deps(target)
        
        # Topological sort (simple: just follow dependency order)
        # This works because we added deps before the task itself
        ordered: List[Task] = []
        added: Set[str] = set()
        
        def add_in_order(task_name: str) -> None:
            if task_name in added:
                return
            
            task = task_registry.get_by_name(task_name)
            if task is None:
                return
            
            # Add dependencies first
            for dep in task.requires:
                dep_task = task_registry.get_by_artifact(dep)
                if dep_task:
                    add_in_order(dep_task.name)
            
            ordered.append(task)
            added.add(task_name)
        
        for task_name in needed_tasks:
            add_in_order(task_name)
        
        return ordered
    
    def run(
        self,
        *,
        targets: Optional[Iterable[str]] = None,
        bundle: Optional[str] = None,
    ) -> AnalysisResult:
        """Run analysis to produce the requested targets.
        
        Args:
            targets: Artifact names to produce (e.g. {"highlights_candidates"})
            bundle: Named bundle to use (e.g. "pre_download", "full")
            
        Returns:
            AnalysisResult with status and timing info
        """
        import time
        from .bundles import get_bundle
        
        start = time.time()
        explicit_targets = targets is not None and bundle is None
        
        # Resolve targets
        if bundle:
            target_set = get_bundle(bundle)
        elif targets:
            target_set = set(targets)
        else:
            raise ValueError("Must specify either targets or bundle")
        
        result = AnalysisResult(targets=target_set, tasks_run=[])
        tasks: List[Task] = []
        planned_results: Dict[str, TaskResult] = {}
        blocked_results: Dict[str, TaskResult] = {}
        executed_results: Dict[str, TaskResult] = {}
        
        try:
            # Resolve dependencies
            tasks = self._resolve_dependencies(target_set)
            
            logger.info(f"[DAG] Resolved {len(tasks)} tasks for targets: {target_set}")
            for task in tasks:
                logger.debug(f"[DAG]   Task '{task.name}' (produces: {task.produces})")
            
            if not tasks:
                logger.warning("[DAG] No tasks to run (all outputs exist)")
                result.missing_targets = {t for t in target_set if not self._artifact_exists(t)}
                if result.missing_targets:
                    logger.warning(f"[DAG] Missing targets after run: {sorted(result.missing_targets)}")
                    if explicit_targets:
                        result.success = False
                        result.error = f"Missing targets: {sorted(result.missing_targets)}"
                result.total_elapsed_seconds = time.time() - start
                return result
            
            # Three-pass filtering:
            # Pass 1: Find all tasks that should run (outputs missing or need upgrade)
            tasks_that_should_run: List[tuple[Task, str]] = []
            will_be_produced: Set[str] = set()
            
            for task in tasks:
                should_run, reason = self._should_run_task(task)
                if should_run:
                    tasks_that_should_run.append((task, reason))
                    will_be_produced.update(task.produces)
                else:
                    status = "disabled" if reason == "disabled by config" else "skipped"
                    planned_results[task.name] = TaskResult(task_name=task.name, status=status, error=reason)
                    logger.debug(f"[DAG] Skipping '{task.name}': {reason}")
            
            # Pass 2: Re-check tasks that were skipped - they might need upgrade
            # now that we know what will_be_produced contains.
            # IMPORTANT: If an upgrade trigger is being PRODUCED in this run (recomputed),
            # we must re-run the downstream task even if the trigger was "present" before.
            # This fixes the "boundary_graph upgraded but highlights_candidates didn't rerun" bug.
            for task in tasks:
                # Skip if already marked to run
                if any(t.name == task.name for t, _ in tasks_that_should_run):
                    continue
                
                # Skip if task has no upgrade triggers
                if not task.upgrade_triggers:
                    continue
                    
                # Check upgrade triggers against will_be_produced
                trigger_will_be_produced = task.upgrade_triggers & will_be_produced
                if trigger_will_be_produced:
                    # If a trigger is being re-produced in this run, we should re-run
                    # this task to get the updated data, regardless of previous sources_present
                    reason = f"upgrade: triggers being produced {trigger_will_be_produced}"
                    tasks_that_should_run.append((task, reason))
                    will_be_produced.update(task.produces)
                    planned_results.pop(task.name, None)
                    logger.info(f"[DAG] Adding '{task.name}' for upgrade: {trigger_will_be_produced}")
                    continue
                
                # Also check: task output exists, but was created WITHOUT a trigger that now exists
                # This catches the case where highlights_candidates ran before boundary_graph existed
                all_outputs_exist = all(self._artifact_exists(a) for a in task.produces)
                if all_outputs_exist:
                    available_triggers = {
                        t for t in task.upgrade_triggers
                        if self._artifact_exists(t)
                    }
                    if available_triggers:
                        state = get_artifact_state(
                            list(task.produces)[0],
                            self.proj.analysis_dir,
                        )
                        sources = state.sources_present or set()
                        new_sources = available_triggers - sources
                        if new_sources:
                            reason = f"upgrade: new sources available {new_sources}"
                            tasks_that_should_run.append((task, reason))
                            will_be_produced.update(task.produces)
                            planned_results.pop(task.name, None)
                            logger.info(f"[DAG] Adding '{task.name}' for upgrade (new sources): {new_sources}")
            
            # Pass 3: Filter to tasks that can run (considering what will be produced)
            tasks_to_run: List[tuple[Task, str]] = []
            for task, reason in tasks_that_should_run:
                can_run, dep_reason = self._can_run_task(task, will_be_produced)
                if can_run:
                    tasks_to_run.append((task, reason))
                    logger.info(f"[DAG] Will run '{task.name}': {reason}")
                else:
                    logger.warning(f"[DAG] Cannot run '{task.name}': {dep_reason}")
                    blocked_results[task.name] = TaskResult(task_name=task.name, status="skipped", error=dep_reason)
            
            if not tasks_to_run:
                logger.warning(f"[DAG] All {len(tasks)} tasks skipped (outputs exist or can't run)")
                all_results: Dict[str, TaskResult] = {}
                all_results.update(planned_results)
                all_results.update(blocked_results)
                all_results.update(executed_results)

                result.tasks_run = []
                for task in tasks:
                    tr = all_results.get(task.name)
                    if tr is None:
                        tr = TaskResult(task_name=task.name, status="skipped", error="not scheduled")
                    result.tasks_run.append(tr)

                result.missing_targets = {t for t in target_set if not self._artifact_exists(t)}
                if result.missing_targets:
                    logger.warning(f"[DAG] Missing targets after run: {sorted(result.missing_targets)}")
                    if explicit_targets:
                        result.success = False
                        if not result.error:
                            result.error = f"Missing targets: {sorted(result.missing_targets)}"
                result.total_elapsed_seconds = time.time() - start
                return result
            
            # Group tasks into parallel levels based on dependencies
            # Level 0: tasks with no unsatisfied dependencies
            # Level 1: tasks that depend only on level 0 outputs
            # etc.
            levels = self._group_tasks_by_level(tasks_to_run)
            
            logger.info(f"[DAG] Grouped {len(tasks_to_run)} tasks into {len(levels)} parallel levels:")
            for level_idx, level_tasks in enumerate(levels):
                task_names = [t.name for t, _ in level_tasks]
                logger.info(f"[DAG]   Level {level_idx}: {task_names}")
            
            # Calculate progress: each level gets equal weight
            completed_tasks = 0
            total_tasks = len(tasks_to_run)
            
            # Run each level - tasks within a level run in parallel
            for level_idx, level_tasks in enumerate(levels):
                level_start_progress = completed_tasks / total_tasks if total_tasks > 0 else 0
                
                if len(level_tasks) == 1:
                    # Single task - run directly (no threading overhead)
                    task, reason = level_tasks[0]
                    logger.info(f"[DAG] Running task '{task.name}' ({reason})")
                    
                    if self.on_progress:
                        self.on_progress(level_start_progress, f"Running {task.name}...")
                    
                    task_result = self._run_task(
                        task,
                        task_progress_base=level_start_progress,
                        task_progress_weight=1.0 / total_tasks,
                    )
                    executed_results[task.name] = task_result
                    completed_tasks += 1

                    if task_result.status == "completed":
                        produced = sorted(task_result.artifacts_produced) if task_result.artifacts_produced else []
                        if produced:
                            logger.info(
                                f"[DAG] Task '{task.name}' completed in {task_result.elapsed_seconds:.1f}s, "
                                f"produced: {produced}"
                            )
                        else:
                            logger.info(f"[DAG] Task '{task.name}' completed in {task_result.elapsed_seconds:.1f}s")
                    elif task_result.status == "skipped" and task_result.error == "no outputs produced":
                        logger.info(
                            f"[DAG] Task '{task.name}' ran but produced no output files "
                            f"(this is OK for optional data)"
                        )
                    elif task_result.status == "failed":
                        # Check if this is a cancellation
                        if task_result.error and "cancel" in task_result.error.lower():
                            logger.info(f"[DAG] Task '{task.name}' cancelled")
                            result.success = False
                            result.error = "Cancelled by user"
                        else:
                            logger.error(f"[DAG] Task '{task.name}' failed: {task_result.error}")
                            result.success = False
                            result.error = f"Task '{task.name}' failed: {task_result.error}"
                        break
                else:
                    # Multiple tasks - run in parallel
                    parallel_names = [t.name for t, _ in level_tasks]
                    logger.info(f"[DAG] Running {len(level_tasks)} tasks in parallel: {parallel_names}")
                    
                    if self.on_progress:
                        self.on_progress(level_start_progress, f"Running {', '.join(parallel_names)} in parallel...")
                    
                    # Use ThreadPoolExecutor for parallel execution
                    # Limit to 4 workers to avoid overwhelming the system
                    max_workers = min(4, len(level_tasks))
                    cancelled = False
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_task: Dict[concurrent.futures.Future, tuple[Task, str]] = {}
                        future_started_at: Dict[concurrent.futures.Future, float] = {}
                        pending: Set[concurrent.futures.Future] = set()

                        gpu_limit = self._get_gpu_concurrency()
                        gpu_limit = max(0, int(gpu_limit))

                        def _is_gpu_heavy(t: Task) -> bool:
                            return self._task_uses_gpu_heavy(t)

                        gpu_heavy_names = [t.name for t, _ in level_tasks if _is_gpu_heavy(t)]
                        if gpu_limit > 0 and len(gpu_heavy_names) > gpu_limit:
                            logger.info(
                                "[DAG] Level %d: limiting to %d concurrent GPU-heavy task(s): %s",
                                level_idx,
                                gpu_limit,
                                gpu_heavy_names,
                            )

                        cpu_queue = deque([(t, r) for t, r in level_tasks if not _is_gpu_heavy(t)])
                        gpu_queue = deque([(t, r) for t, r in level_tasks if _is_gpu_heavy(t)])
                        gpu_inflight = 0

                        def _submit_task(t: Task, r: str) -> None:
                            nonlocal gpu_inflight
                            weight = 1.0 / total_tasks
                            logger.info("[DAG] Starting task '%s' (%s)", t.name, r)
                            fut = executor.submit(
                                self._run_task,
                                t,
                                task_progress_base=level_start_progress,
                                task_progress_weight=weight,
                            )
                            future_to_task[fut] = (t, r)
                            future_started_at[fut] = time.time()
                            pending.add(fut)
                            if _is_gpu_heavy(t):
                                gpu_inflight += 1

                        def _fill_slots() -> None:
                            while len(pending) < max_workers:
                                if gpu_queue and (gpu_limit <= 0 or gpu_inflight < gpu_limit):
                                    t, r = gpu_queue.popleft()
                                    _submit_task(t, r)
                                    continue
                                if cpu_queue:
                                    t, r = cpu_queue.popleft()
                                    _submit_task(t, r)
                                    continue
                                break

                        _fill_slots()
                        
                        heartbeat_s = 30.0
                        try:
                            env = os.environ.get("VP_DAG_HEARTBEAT_SECONDS", "").strip()
                            if env != "":
                                heartbeat_s = float(env)
                        except Exception:
                            heartbeat_s = 30.0
                        heartbeat_s = max(0.0, heartbeat_s)

                        while pending:
                            # Emit a periodic heartbeat so long-running tasks are visible in logs.
                            timeout = heartbeat_s if heartbeat_s > 0 else None
                            done, not_done = concurrent.futures.wait(
                                pending,
                                timeout=timeout,
                                return_when=concurrent.futures.FIRST_COMPLETED,
                            )

                            if not done and not_done and heartbeat_s > 0:
                                now = time.time()
                                running: List[tuple[float, str]] = []
                                for fut in not_done:
                                    t, _ = future_to_task.get(fut, (None, None))
                                    name = getattr(t, "name", "unknown")
                                    started = future_started_at.get(fut, now)
                                    running.append((now - started, name))
                                running.sort(reverse=True)
                                detail = ", ".join(f"{name} ({elapsed:.0f}s)" for elapsed, name in running)
                                logger.info(
                                    f"[DAG] Level {level_idx}: still running {len(not_done)} task(s): {detail}"
                                )
                                pending = not_done
                                continue

                            pending = not_done
                            for future in done:
                                task, reason = future_to_task.pop(future)
                                if _is_gpu_heavy(task) and gpu_inflight > 0:
                                    gpu_inflight -= 1
                                try:
                                    task_result = future.result()
                                    executed_results[task.name] = task_result
                                    completed_tasks += 1
                                    
                                    if task_result.status == "completed":
                                        produced = sorted(task_result.artifacts_produced) if task_result.artifacts_produced else []
                                        if produced:
                                            logger.info(
                                                f"[DAG] Task '{task.name}' completed in {task_result.elapsed_seconds:.1f}s, "
                                                f"produced: {produced}"
                                            )
                                        else:
                                            logger.info(f"[DAG] Task '{task.name}' completed in {task_result.elapsed_seconds:.1f}s")
                                    elif task_result.status == "skipped" and task_result.error == "no outputs produced":
                                        logger.info(
                                            f"[DAG] Task '{task.name}' ran but produced no output files "
                                            f"(this is OK for optional data)"
                                        )
                                    elif task_result.status == "failed":
                                        # Check if this is a cancellation
                                        if task_result.error and "cancel" in task_result.error.lower():
                                            logger.info(f"[DAG] Task '{task.name}' cancelled")
                                            cancelled = True
                                        else:
                                            logger.error(f"[DAG] Task '{task.name}' failed: {task_result.error}")
                                            # Don't stop other parallel tasks, but mark overall as failed
                                            result.success = False
                                            if not result.error:
                                                result.error = f"Task '{task.name}' failed: {task_result.error}"
                                except Exception as e:
                                    error_str = str(e)
                                    # Check if this is a cancellation
                                    if "cancel" in error_str.lower():
                                        logger.info(f"[DAG] Task '{task.name}' cancelled")
                                        cancelled = True
                                    else:
                                        logger.error(f"[DAG] Task '{task.name}' raised exception: {e}")
                                        # Create a failed result
                                        task_result = TaskResult(task_name=task.name, status="failed", error=error_str)
                                        executed_results[task.name] = task_result
                                        completed_tasks += 1
                                        result.success = False
                                        if not result.error:
                                            result.error = f"Task '{task.name}' failed: {e}"

                            # Submit more tasks if we have capacity (and GPU slots available).
                            _fill_slots()
                    
                    # If cancelled, stop immediately
                    if cancelled:
                        logger.info(f"[DAG] Stopping due to cancellation")
                        result.success = False
                        result.error = "Cancelled by user"
                        break
                    
                    # If any task in this level failed, stop processing further levels
                    if not result.success:
                        logger.warning(f"[DAG] Stopping after level {level_idx} due to task failure")
                        break
            
            if self.on_progress:
                self.on_progress(1.0, "Analysis complete")
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"[DAG] Analysis failed: {e}")
        
        all_results: Dict[str, TaskResult] = {}
        all_results.update(planned_results)
        all_results.update(blocked_results)
        all_results.update(executed_results)

        result.tasks_run = []
        for task in tasks:
            tr = all_results.get(task.name)
            if tr is None:
                tr = TaskResult(task_name=task.name, status="skipped", error="not scheduled")
            result.tasks_run.append(tr)

        result.missing_targets = {t for t in target_set if not self._artifact_exists(t)}
        if result.missing_targets:
            logger.warning(f"[DAG] Missing targets after run: {sorted(result.missing_targets)}")
            if explicit_targets and result.success:
                result.success = False
                if not result.error:
                    result.error = f"Missing targets: {sorted(result.missing_targets)}"

        result.total_elapsed_seconds = time.time() - start
        return result


def run_analysis(
    proj: Project,
    *,
    targets: Optional[Iterable[str]] = None,
    bundle: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[float, str], None]] = None,
    llm_complete: Optional[Callable[[str], str]] = None,
    upgrade_mode: bool = False,
) -> AnalysisResult:
    """Convenience function to run analysis.
    
    Args:
        proj: Project to analyze
        targets: Artifact names to produce
        bundle: Named bundle ("pre_download" or "full")
        config: Config dict (defaults to project's profile)
        on_progress: Progress callback
        llm_complete: LLM function for semantic tasks
        upgrade_mode: Re-run tasks when upgrade triggers are satisfied
        
    Returns:
        AnalysisResult
    """
    if config is None:
        from ..project import get_project_data
        proj_data = get_project_data(proj)
        profile = proj_data.get("profile", {})
        config = profile.get("analysis", {})
    
    runner = AnalysisRunner(
        proj,
        config,
        on_progress=on_progress,
        llm_complete=llm_complete,
        upgrade_mode=upgrade_mode,
    )
    
    return runner.run(targets=targets, bundle=bundle)
