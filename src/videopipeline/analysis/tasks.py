"""Task definitions for the VideoPipeline analysis DAG.

Each task declares:
- requires: Artifacts that must exist before this task runs
- produces: Artifacts this task generates
- run: Function that computes the output(s)
- version: Bump when algorithm changes (triggers recompute)
- can_run_partial: True if task can run with some inputs missing
- upgrade_triggers: Artifacts that, when they appear, should trigger re-run

Tasks are registered at import time. The runner executes them in dependency order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

from ..project import Project

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A unit of work in the analysis DAG.
    
    Attributes:
        name: Human-readable name
        requires: Artifacts that must exist (or be optional via can_run_partial)
        produces: Artifacts this task generates
        run: Function(project, config, on_progress) -> None
        version: Algorithm version (bump to force recompute)
        can_run_partial: If True, task runs even if some inputs missing
        optional_inputs: Inputs that are nice-to-have but not required
        upgrade_triggers: If these artifacts appear later, re-run this task
        enabled_check: Optional function to check if task should run based on config
    """
    name: str
    requires: Set[str]
    produces: Set[str]
    run: Callable[[Project, Dict[str, Any], Optional[Callable[[float], None]]], None]
    version: str = "1.0"
    can_run_partial: bool = False
    optional_inputs: Set[str] = field(default_factory=set)
    upgrade_triggers: Set[str] = field(default_factory=set)
    enabled_check: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        """Check if this task should run based on config."""
        if self.enabled_check is None:
            return True
        return self.enabled_check(config)


class TaskRegistry:
    """Registry of tasks keyed by produced artifact."""
    
    def __init__(self) -> None:
        self._by_artifact: Dict[str, Task] = {}
        self._by_name: Dict[str, Task] = {}
        self._all_tasks: list[Task] = []
    
    def register(self, task: Task) -> None:
        """Register a task."""
        self._by_name[task.name] = task
        self._all_tasks.append(task)
        for artifact in task.produces:
            self._by_artifact[artifact] = task
    
    def get_by_artifact(self, artifact: str) -> Optional[Task]:
        """Get the task that produces an artifact."""
        return self._by_artifact.get(artifact)
    
    def get_by_name(self, name: str) -> Optional[Task]:
        """Get a task by name."""
        return self._by_name.get(name)
    
    def all_tasks(self) -> list[Task]:
        """Get all registered tasks."""
        return list(self._all_tasks)


# Global registry
task_registry = TaskRegistry()


# =============================================================================
# Dependency availability checks
# =============================================================================

def _is_torch_available() -> bool:
    """Check if torch is installed."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _torch_missing_reason() -> str:
    """Return reason string when torch is missing."""
    return "torch not installed (pip install torch)"


# =============================================================================
# Task Implementations
# =============================================================================

def _run_audio_features(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute audio RMS/energy features."""
    if proj.audio_features_path.exists():
        return
    
    from ..analysis_audio import compute_audio_analysis
    
    audio_cfg = cfg.get("audio", {})
    compute_audio_analysis(
        proj,
        sample_rate=int(audio_cfg.get("sample_rate", 16000)),
        hop_s=float(audio_cfg.get("hop_seconds", 0.5)),
        smooth_s=float(audio_cfg.get("smooth_seconds", 3.0)),
        top=int(audio_cfg.get("top", 12)),
        min_gap_s=float(audio_cfg.get("min_gap_seconds", 20.0)),
        pre_s=float(audio_cfg.get("pre_seconds", 8.0)),
        post_s=float(audio_cfg.get("post_seconds", 22.0)),
        skip_start_s=float(audio_cfg.get("skip_start_seconds", 10.0)),
        on_progress=on_progress,
    )


def _run_audio_events(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute ML-based audio event detection (laughter, applause, etc.)."""
    if proj.audio_events_features_path.exists():
        return
    
    audio_events_cfg = cfg.get("audio_events", {})
    if not audio_events_cfg.get("enabled", True):
        return
    
    from ..analysis_audio_events import compute_audio_events_analysis, AudioEventsConfig
    
    compute_audio_events_analysis(
        proj,
        cfg=AudioEventsConfig.from_dict(audio_events_cfg),
        on_progress=on_progress,
    )


def _run_silence(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Detect silence intervals in audio."""
    silence_path = proj.analysis_dir / "silence.json"
    if silence_path.exists():
        return
    
    from ..analysis_silence import compute_silence_analysis, SilenceConfig
    
    silence_cfg = cfg.get("silence", {})
    compute_silence_analysis(
        proj,
        cfg=SilenceConfig(
            noise_db=float(silence_cfg.get("noise_db", -35.0)),
            min_duration=float(silence_cfg.get("min_duration", 0.3)),
        ),
        on_progress=on_progress,
    )


def _run_audio_vad(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Detect speech segments (VAD) in audio using Silero VAD.

    This is typically a *better* boundary signal than FFmpeg silencedetect for
    gaming/streaming content, because there may be constant game audio.
    """
    # Check torch availability first - skip gracefully if not installed
    if not _is_torch_available():
        logger.info("[audio_vad] torch not installed; skipping VAD (install with: pip install torch)")
        return

    vad_path = proj.analysis_dir / "audio_vad.npz"
    if vad_path.exists():
        return

    from ..analysis_vad import compute_audio_vad_analysis, VadConfig

    vad_cfg = cfg.get("vad", {}) or {}
    compute_audio_vad_analysis(
        proj,
        cfg=VadConfig(
            enabled=bool(vad_cfg.get("enabled", True)),
            sample_rate=int(vad_cfg.get("sample_rate", 16000)),
            hop_seconds=float(vad_cfg.get("hop_seconds", 0.5)),
            threshold=float(vad_cfg.get("threshold", 0.5)),
            min_silence_duration_ms=int(vad_cfg.get("min_silence_duration_ms", 250)),
            speech_pad_ms=int(vad_cfg.get("speech_pad_ms", 60)),
            min_speech_duration_ms=int(vad_cfg.get("min_speech_duration_ms", 250)),
            use_onnx=bool(vad_cfg.get("use_onnx", False)),
            opset_version=int(vad_cfg.get("opset_version", 16)),
            torch_threads=int(vad_cfg.get("torch_threads", 1)),
            device=str(vad_cfg.get("device", "cpu")),
        ),
        on_progress=on_progress,
    )


def _run_reaction_audio(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute a speech-focused "voice excitement" timeline.

    Output: analysis/reaction_audio_features.npz
    """
    # Project has a helper, but fall back to analysis_dir to keep this task portable.
    out_path = getattr(proj, "reaction_audio_features_path", proj.analysis_dir / "reaction_audio_features.npz")
    if out_path.exists():
        return

    reaction_cfg = cfg.get("reaction_audio", {})
    if not reaction_cfg.get("enabled", True):
        return

    from ..analysis_reaction_audio import compute_reaction_audio_features, ReactionAudioConfig

    # Default hop to the main analysis hop for easy alignment.
    audio_cfg = cfg.get("audio", {})
    merged = dict(reaction_cfg)
    merged.setdefault("hop_seconds", audio_cfg.get("hop_seconds", 0.5))

    compute_reaction_audio_features(
        proj,
        cfg=ReactionAudioConfig.from_dict(merged),
        on_progress=on_progress,
    )


def _run_diarization(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute speaker diarization as a standalone analysis artifact."""
    diar_path = proj.analysis_dir / "diarization.json"
    speech_cfg = cfg.get("speech", {})
    if not speech_cfg.get("diarize", False):
        return

    # If diarization already exists, still try to merge speaker labels into the transcript
    # (older projects may have transcript_full.json without speakers).
    if diar_path.exists():
        try:
            from ..analysis_transcript import merge_diarization_json_into_transcript

            merge_diarization_json_into_transcript(proj)
        except Exception:
            pass
        return

    from ..analysis_diarization import compute_diarization_analysis, DiarizationConfig

    diar_cfg = cfg.get("diarization", {})
    # If there's no dedicated diarization section, fall back to speech.* keys.
    merged_cfg: Dict[str, Any] = {
        **speech_cfg,
        **diar_cfg,
    }

    compute_diarization_analysis(
        proj,
        cfg=DiarizationConfig.from_dict(merged_cfg),
        on_progress=on_progress,
    )

    # Best-effort: merge speaker labels into transcript_full.json if it exists.
    try:
        from ..analysis_transcript import merge_diarization_json_into_transcript

        merge_diarization_json_into_transcript(proj)
    except Exception:
        pass


def _run_transcript(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute full video transcript."""
    transcript_path = proj.analysis_dir / "transcript_full.json"
    
    from ..analysis_transcript import compute_transcript_analysis, TranscriptConfig
    
    speech_cfg = cfg.get("speech", {})

    # If transcript already exists, still try to merge diarization speaker labels (if enabled).
    if transcript_path.exists():
        if speech_cfg.get("diarize", False):
            try:
                from ..analysis_transcript import merge_diarization_json_into_transcript

                merge_diarization_json_into_transcript(proj)
            except Exception:
                pass
        return

    compute_transcript_analysis(
        proj,
        cfg=TranscriptConfig(
            backend=str(speech_cfg.get("backend", "auto")),
            model_size=str(speech_cfg.get("model_size", "small")),
            language=speech_cfg.get("language"),
            device=str(speech_cfg.get("device", "cuda")),
            compute_type=str(speech_cfg.get("compute_type", "float16")),
            vad_filter=bool(speech_cfg.get("vad_filter", True)),
            word_timestamps=bool(speech_cfg.get("word_timestamps", True)),
            use_gpu=bool(speech_cfg.get("use_gpu", True)),
            verbose=bool(speech_cfg.get("verbose", False)),
        ),
        on_progress=on_progress,
    )

    # Best-effort: merge speaker labels once both transcript and diarization exist.
    if speech_cfg.get("diarize", False):
        try:
            from ..analysis_transcript import merge_diarization_json_into_transcript

            merge_diarization_json_into_transcript(proj)
        except Exception:
            pass


def _run_sentences(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Extract sentence boundaries from transcript."""
    sentences_path = proj.analysis_dir / "sentences.json"
    if sentences_path.exists():
        return
    
    from ..analysis_sentences import compute_sentences_analysis, SentenceConfig
    
    sentences_cfg = cfg.get("sentences", {})
    compute_sentences_analysis(
        proj,
        cfg=SentenceConfig(
            max_sentence_words=int(sentences_cfg.get("max_sentence_words", 30)),
            sentence_end_chars=sentences_cfg.get("sentence_end_chars", ".!?"),
        ),
        on_progress=on_progress,
    )


def _run_speech_features(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute speech features (word density, speech rate, etc.)."""
    if proj.speech_features_path.exists():
        return
    
    from ..analysis_speech_features import compute_speech_features, SpeechFeatureConfig
    
    speech_cfg = cfg.get("speech_features", cfg.get("speech", {}))
    compute_speech_features(
        proj,
        cfg=SpeechFeatureConfig(
            hop_seconds=float(speech_cfg.get("hop_seconds", 0.5)),
        ),
        on_progress=on_progress,
    )


def _run_chapters(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute semantic chapters using embeddings + LLM labeling."""
    chapters_path = proj.analysis_dir / "chapters.json"
    if chapters_path.exists():
        return
    
    chapters_cfg = cfg.get("chapters", {})
    if not chapters_cfg.get("enabled", True):
        return
    
    from ..analysis_chapters import compute_chapters_analysis, ChapterConfig
    
    # Get LLM function if available
    llm_complete = cfg.get("_llm_complete")  # Injected by runner
    
    compute_chapters_analysis(
        proj,
        cfg=ChapterConfig(
            min_chapter_len_s=float(chapters_cfg.get("min_chapter_len_s", 60.0)),
            max_chapter_len_s=float(chapters_cfg.get("max_chapter_len_s", 900.0)),
            embedding_model=str(chapters_cfg.get("embedding_model", "all-mpnet-base-v2")),
            changepoint_penalty=float(chapters_cfg.get("changepoint_penalty", 10.0)),
            llm_labeling=bool(chapters_cfg.get("llm_labeling", True)) and llm_complete is not None,
        ),
        llm_complete=llm_complete,
        on_progress=on_progress,
    )


def _run_chat_features(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute chat activity features."""
    if proj.chat_features_path.exists():
        return
    
    # Check if chat data exists
    chat_db = proj.analysis_dir / "chat.sqlite"
    chat_json = proj.analysis_dir / "chat_raw.json"
    
    if not chat_db.exists() and not chat_json.exists():
        logger.info("[chat_features] No chat data found; skipping.")
        return  # No chat data
    
    highlights_cfg = cfg.get("highlights", {})
    audio_cfg = cfg.get("audio", {})
    hop_s = float(audio_cfg.get("hop_seconds", 0.5))
    
    if chat_db.exists():
        from ..chat.features import compute_and_save_chat_features
        from ..chat.emote_db import get_channel_for_project

        channel_info = get_channel_for_project(proj)
        channel_id = channel_info[0] if channel_info else None
        platform = channel_info[1] if channel_info else "twitch"
        compute_and_save_chat_features(
            proj,
            hop_s=hop_s,
            smooth_s=float(highlights_cfg.get("chat_smooth_seconds", 3.0)),
            on_progress=on_progress,
            llm_complete=cfg.get("_llm_complete"),
            channel_id=channel_id,
            platform=platform,
        )
    else:
        from ..analysis_chat import compute_chat_analysis
        compute_chat_analysis(
            proj,
            chat_path=chat_json,
            hop_s=hop_s,
            smooth_s=float(highlights_cfg.get("chat_smooth_seconds", 3.0)),
            on_progress=on_progress,
        )


def _run_chat_boundaries(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute chat activity valleys (good boundary points)."""
    boundaries_path = proj.analysis_dir / "chat_boundaries.json"
    if boundaries_path.exists():
        return
    
    if not proj.chat_features_path.exists():
        return  # Need chat features first
    
    from ..analysis_chat_boundaries import compute_chat_boundaries_analysis, ChatBoundaryConfig
    
    chat_cfg = cfg.get("chat_boundaries", {})
    compute_chat_boundaries_analysis(
        proj,
        cfg=ChatBoundaryConfig(
            valley_threshold_z=float(chat_cfg.get("valley_threshold_z", -0.5)),
            burst_threshold_z=float(chat_cfg.get("burst_threshold_z", 1.5)),
            min_valley_gap_s=float(chat_cfg.get("min_valley_gap_s", 5.0)),
            min_burst_gap_s=float(chat_cfg.get("min_burst_gap_s", 3.0)),
        ),
        on_progress=on_progress,
    )


def _run_motion_features(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute motion features from video frames."""
    if proj.motion_features_path.exists():
        return
    
    from ..analysis_motion import compute_motion_analysis
    
    motion_cfg = cfg.get("motion", {})
    compute_motion_analysis(
        proj,
        sample_fps=float(motion_cfg.get("sample_fps", 3.0)),
        scale_width=int(motion_cfg.get("scale_width", 160)),
        smooth_s=float(motion_cfg.get("smooth_seconds", 2.5)),
        on_progress=on_progress,
    )


def _run_scenes(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Detect visual scene cuts."""
    if proj.scenes_path.exists():
        return
    
    scenes_cfg = cfg.get("scenes", {})
    if not scenes_cfg.get("enabled", True):
        return
    
    from ..analysis_scenes import compute_scene_analysis
    
    compute_scene_analysis(
        proj,
        threshold_z=float(scenes_cfg.get("threshold_z", 3.5)),
        min_scene_len_seconds=float(scenes_cfg.get("min_scene_len_seconds", 1.2)),
        snap_window_seconds=float(scenes_cfg.get("snap_window_seconds", 1.0)),
        on_progress=on_progress,
    )


def _run_boundary_graph(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute unified boundary graph from all available sources.
    
    This task is special: it can run with partial inputs and should be
    re-run when new boundary sources become available.
    """
    from ..analysis_boundaries import compute_boundary_graph, BoundaryConfig, save_boundary_graph
    
    # Load scene cuts if available
    scene_cuts = None
    if proj.scenes_path.exists():
        import json
        try:
            scenes_data = json.loads(proj.scenes_path.read_text(encoding="utf-8"))
            # Support both new (cuts_seconds) and legacy (cuts) keys
            scene_cuts = scenes_data.get("cuts_seconds") or scenes_data.get("cuts") or []
        except Exception as exc:
            logger.warning("[boundary_graph] Failed to read/parse scenes.json (%s): %s", proj.scenes_path, exc)
    
    boundaries_cfg = cfg.get("boundaries", {})
    boundary_cfg = BoundaryConfig(
        prefer_vad=bool(boundaries_cfg.get("prefer_vad", True)),
        vad_boundary_score=float(boundaries_cfg.get("vad_boundary_score", 1.0)),
        prefer_silence=bool(boundaries_cfg.get("prefer_silence", True)),
        prefer_sentences=bool(boundaries_cfg.get("prefer_sentences", True)),
        prefer_scene_cuts=bool(boundaries_cfg.get("prefer_scene_cuts", True)),
        prefer_chat_valleys=bool(boundaries_cfg.get("prefer_chat_valleys", True)),
        prefer_chapters=bool(boundaries_cfg.get("prefer_chapters", True)),
        prefer_turn_boundaries=bool(boundaries_cfg.get("prefer_turn_boundaries", True)),
        snap_tolerance_s=float(boundaries_cfg.get("snap_tolerance_s", 1.0)),
        chapter_boundary_score=float(boundaries_cfg.get("chapter_boundary_score", 2.0)),
        turn_boundary_score=float(boundaries_cfg.get("turn_boundary_score", 1.3)),
    )
    
    graph = compute_boundary_graph(proj, boundary_cfg, scene_cuts=scene_cuts)
    save_boundary_graph(proj, graph)
    
    if on_progress:
        on_progress(1.0)


def _run_highlights_scores(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Compute combined highlight scores and peak indices.
    
    This is the SCORING phase - it fuses signals and finds peaks.
    Clip shaping is done separately in highlights_candidates.
    """
    # NOTE: We keep this call forwards/backwards compatible.
    # If your compute_highlights_scores implementation doesn't yet support
    # reaction_audio / diarization-derived signals, this will still run.
    from ..analysis_highlights import compute_highlights_scores

    import inspect

    kwargs: Dict[str, Any] = {
        "audio_cfg": cfg.get("audio", {}),
        "motion_cfg": cfg.get("motion", {}),
        "highlights_cfg": cfg.get("highlights", {}),
        "audio_events_cfg": cfg.get("audio_events"),
        "include_chat": cfg.get("include_chat", True),
        "include_audio_events": cfg.get("include_audio_events", True),
        "on_progress": on_progress,
    }

    sig = inspect.signature(compute_highlights_scores)
    # Optional new knobs (only passed if your function accepts them)
    if "reaction_audio_cfg" in sig.parameters:
        kwargs["reaction_audio_cfg"] = cfg.get("reaction_audio", {})
    if "include_reaction_audio" in sig.parameters:
        kwargs["include_reaction_audio"] = cfg.get("include_reaction_audio", True)
    if "diarization_cfg" in sig.parameters:
        # If there's no dedicated diarization section, fall back to speech keys.
        kwargs["diarization_cfg"] = {**cfg.get("speech", {}), **cfg.get("diarization", {})}
    if "include_diarization" in sig.parameters:
        kwargs["include_diarization"] = cfg.get("include_diarization", True)

    compute_highlights_scores(proj, **kwargs)


def _run_highlights_candidates(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Shape peaks into clip candidates using boundary graph.
    
    This is the SHAPING phase - it uses the boundary graph to find
    optimal clip boundaries. This is fast and can be re-run when
    the boundary graph improves.
    """
    from ..analysis_highlights import compute_highlights_candidates
    
    compute_highlights_candidates(
        proj,
        highlights_cfg=cfg.get("highlights", {}),
        scenes_cfg=cfg.get("scenes", {}),
        llm_complete=cfg.get("_llm_complete"),
        on_progress=on_progress,
    )


def _run_variants(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Generate clip variants (titles, hooks, etc.)."""
    from ..clip_variants import compute_variants
    
    compute_variants(
        proj,
        cfg=cfg.get("variants", {}),
        llm_complete=cfg.get("_llm_complete"),
        on_progress=on_progress,
    )


def _run_director(proj: Project, cfg: Dict[str, Any], on_progress: Optional[Callable[[float], None]]) -> None:
    """Run the AI Director to pick best variants + generate packaging metadata.

    Output: analysis/director.json

    This is an optional step (enable via analysis.director.enabled).
    """
    director_cfg = cfg.get("director", {})
    if not director_cfg.get("enabled", False):
        return

    director_path = proj.analysis_dir / "director.json"
    if director_path.exists():
        return

    from ..analysis_director import compute_director

    compute_director(
        proj,
        cfg=director_cfg,
        llm_complete=cfg.get("_llm_complete"),
        on_progress=on_progress,
    )


# =============================================================================
# Task Registration
# =============================================================================

# --- Audio-derived tasks ---
task_registry.register(Task(
    name="audio_features",
    requires=set(),  # Needs video/audio file (implicit)
    produces={"audio_features"},
    run=_run_audio_features,
    version="1.0",
))

task_registry.register(Task(
    name="audio_events",
    requires=set(),  # Needs audio (implicit)
    produces={"audio_events"},
    run=_run_audio_events,
    version="1.0",
    enabled_check=lambda cfg: cfg.get("audio_events", {}).get("enabled", True),
))

task_registry.register(Task(
    name="silence",
    requires=set(),  # Needs audio (implicit)
    produces={"silence"},
    run=_run_silence,
    version="1.0",
))

# --- Transcript-derived tasks ---
task_registry.register(Task(
    name="transcript",
    requires=set(),  # Needs audio (implicit)
    produces={"transcript"},
    run=_run_transcript,
    version="1.0",
))

task_registry.register(Task(
    name="sentences",
    requires={"transcript"},
    produces={"sentences"},
    run=_run_sentences,
    version="1.0",
))

task_registry.register(Task(
    name="speech_features",
    requires={"transcript"},
    produces={"speech_features"},
    run=_run_speech_features,
    version="1.0",
))

task_registry.register(Task(
    name="chapters",
    requires={"transcript", "sentences"},
    produces={"chapters"},
    run=_run_chapters,
    version="1.0",
    enabled_check=lambda cfg: cfg.get("chapters", {}).get("enabled", True),
))

task_registry.register(Task(
    name="audio_vad",
    requires=set(),  # Needs audio (implicit)
    produces={"audio_vad"},
    run=_run_audio_vad,
    version="1.0",
    enabled_check=lambda cfg: cfg.get("vad", {}).get("enabled", True),
))

task_registry.register(Task(
    name="diarization",
    requires=set(),  # Needs audio (implicit)
    produces={"diarization"},
    run=_run_diarization,
    version="1.0",
    # Disabled by default unless speech.diarize = true, also requires torch
    enabled_check=lambda cfg: bool(cfg.get("speech", {}).get("diarize", False)),
))

task_registry.register(Task(
    name="reaction_audio",
    # We *prefer* VAD (speech_fraction gating), but can fall back if missing.
    # This pattern makes the runner wait for audio_vad when both are scheduled,
    # while still allowing a partial run if audio_vad is disabled.
    requires={"audio_vad"},
    optional_inputs={"audio_vad", "diarization"},
    produces={"reaction_audio"},
    run=_run_reaction_audio,
    version="1.0",
    can_run_partial=True,
    enabled_check=lambda cfg: cfg.get("reaction_audio", {}).get("enabled", True),
))

# --- Chat-derived tasks ---
task_registry.register(Task(
    name="chat_features",
    requires=set(),  # Needs chat file (implicit, may not exist)
    produces={"chat_features"},
    run=_run_chat_features,
    version="1.0",
    can_run_partial=True,  # OK if no chat data
))

task_registry.register(Task(
    name="chat_boundaries",
    requires={"chat_features"},
    produces={"chat_boundaries"},
    run=_run_chat_boundaries,
    version="1.0",
    can_run_partial=True,
))

# --- Video-derived tasks ---
task_registry.register(Task(
    name="motion_features",
    requires=set(),  # Needs video file (implicit)
    produces={"motion_features"},
    run=_run_motion_features,
    version="1.0",
))

task_registry.register(Task(
    name="scenes",
    requires={"motion_features"},
    produces={"scenes"},
    run=_run_scenes,
    version="1.0",
    enabled_check=lambda cfg: cfg.get("scenes", {}).get("enabled", True),
))

# --- Boundary graph (merges all boundary sources) ---
task_registry.register(Task(
    name="boundary_graph",
    # Requires transcript-derived sentence boundaries; other sources are optional.
    requires={"sentences"},
    produces={"boundary_graph"},
    run=_run_boundary_graph,
    version="1.0",
    can_run_partial=True,  # Can run without scenes/chapters/audio_vad (if torch missing)
    optional_inputs={"silence", "diarization", "scenes", "chapters", "chat_boundaries", "audio_vad"},
    upgrade_triggers={"silence", "diarization", "scenes", "chapters", "chat_boundaries", "audio_vad"},  # Re-run when these appear
))

# --- Highlights (split into scoring + shaping) ---
task_registry.register(Task(
    name="highlights_scores",
    requires={"audio_features"},  # Minimum required
    produces={"highlights_scores"},
    run=_run_highlights_scores,
    version="1.0",
    can_run_partial=True,
    optional_inputs={"motion_features", "chat_features", "audio_events", "speech_features", "audio_vad", "reaction_audio", "diarization"},
    upgrade_triggers={"motion_features", "chat_features", "audio_events", "speech_features", "audio_vad", "reaction_audio", "diarization"},  # Re-score when richer signals become available
))

task_registry.register(Task(
    name="highlights_candidates",
    requires={"highlights_scores"},  # Need peaks
    produces={"highlights_candidates"},
    run=_run_highlights_candidates,
    version="1.0",
    can_run_partial=True,  # Can run without boundary graph (rough cuts)
    optional_inputs={"boundary_graph", "scenes", "motion_features", "chat_features", "audio_events", "speech_features", "audio_vad", "reaction_audio", "diarization"},
    upgrade_triggers={"boundary_graph", "scenes", "motion_features", "chat_features", "audio_events", "speech_features", "audio_vad", "reaction_audio", "diarization"},  # Re-shape when boundaries OR scoring signals improve
))

# --- Post-highlights ---
task_registry.register(Task(
    name="variants",
    requires={"highlights_candidates", "transcript"},
    optional_inputs={
        "boundary_graph",
        "sentences",
        "chat_boundaries",
        "chapters",
        "audio_vad",
        "reaction_audio",
        "diarization",
    },
    produces={"variants"},
    run=_run_variants,
    version="1.1",
    upgrade_triggers={
        "boundary_graph",
        "sentences",
        "chat_boundaries",
        "chapters",
        "audio_vad",
        "reaction_audio",
        "diarization",
    },
    can_run_partial=True,
))

task_registry.register(Task(
    name="director",
    # Requires computed variants; director can re-run when variants get upgraded.
    requires={"variants"},
    optional_inputs={
        # Optional enrichments that influence copy/selection if available.
        "highlights_candidates",
        "chapters",
        "diarization",
        "reaction_audio",
        "audio_events",
        "audio_vad",
    },
    produces={"director"},
    run=_run_director,
    version="1.0",
    upgrade_triggers={"variants"},
    enabled_check=lambda cfg: cfg.get("director", {}).get("enabled", False),
))
