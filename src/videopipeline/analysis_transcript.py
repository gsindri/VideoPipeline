"""Project-level transcription with pluggable backends.

Supports whisper.cpp (pywhispercpp) and faster-whisper backends.
Runs Whisper once per video and caches the full transcript for reuse.
"""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from .ffmpeg import _require_cmd, ffprobe_duration_seconds
from .project import Project, save_json, update_project
from .utils import subprocess_flags as _subprocess_flags

logger = logging.getLogger(__name__)


# Re-export for backward compatibility
class WhisperNotInstalledError(RuntimeError):
    pass


BackendType = Literal["whispercpp", "faster_whisper", "auto"]


@dataclass(frozen=True)
class TranscriptConfig:
    """Configuration for full-video transcription.
    
    Attributes:
        backend: Transcription backend ("whispercpp", "faster_whisper", "auto")
        model_size: Model size (tiny, base, small, medium, large)
                    Also supports quantized: small.en-q8_0, tiny-q5_1
        language: Language code or None for auto-detect
        device: Device for faster-whisper ("cpu" or "cuda")
        compute_type: Compute type for faster-whisper
        sample_rate: Audio sample rate
        vad_filter: Use voice activity detection (major speedup!)
        word_timestamps: Generate word-level timestamps
        use_gpu: Request GPU acceleration (if available)
        threads: CPU threads (0 = use all cores)
        n_processors: Parallel decoders for long files (whisper.cpp)
        strict: If True, don't fall back to other backends (for A/B testing)
        diarize: Enable speaker diarization (identify who is speaking)
        diarize_min_speakers: Min expected speakers (None = auto-detect)
        diarize_max_speakers: Max expected speakers (None = auto-detect)
        hf_token: Hugging Face token for pyannote models
    """
    backend: BackendType = "auto"
    model_size: str = "small"
    language: Optional[str] = None
    device: str = "cuda"  # "cpu" or "cuda"
    compute_type: str = "float16"  # "int8" on CPU; "float16" on GPU
    sample_rate: int = 16000
    vad_filter: bool = True
    word_timestamps: bool = True
    use_gpu: bool = True
    threads: int = 0  # 0 = use all cores
    n_processors: int = 1  # Parallel decoders (whisper.cpp)
    strict: bool = False  # If True, don't fall back to other backends
    verbose: bool = False  # If True, print transcript to console as it's transcribed
    diarize: bool = False  # Enable speaker diarization
    diarize_min_speakers: Optional[int] = None
    diarize_max_speakers: Optional[int] = None
    hf_token: Optional[str] = None


@dataclass
class TranscriptWord:
    """A single word with timing."""
    word: str
    start: float
    end: float
    probability: float = 1.0
    speaker: Optional[str] = None  # Speaker label if diarization was used

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "probability": self.probability,
        }
        if self.speaker is not None:
            d["speaker"] = self.speaker
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TranscriptWord":
        return cls(
            word=str(d.get("word", "")),
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            probability=float(d.get("probability", 1.0)),
            speaker=d.get("speaker"),
        )


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    start: float
    end: float
    text: str
    words: Optional[List[TranscriptWord]] = None
    speaker: Optional[str] = None  # Speaker label if diarization was used

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.words:
            d["words"] = [w.to_dict() for w in self.words]
        if self.speaker is not None:
            d["speaker"] = self.speaker
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TranscriptSegment":
        words = None
        if "words" in d and d["words"]:
            words = [TranscriptWord.from_dict(w) for w in d["words"]]
        return cls(
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            text=str(d.get("text", "")),
            words=words,
            speaker=d.get("speaker"),
        )


@dataclass
class FullTranscript:
    """Complete transcript of a video."""
    segments: List[TranscriptSegment]
    language: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FullTranscript":
        return cls(
            segments=[TranscriptSegment.from_dict(s) for s in d.get("segments", [])],
            language=d.get("language"),
            duration_seconds=float(d.get("duration_seconds", 0.0)),
        )

    def get_text_in_range(self, start_s: float, end_s: float) -> str:
        """Get concatenated text for segments overlapping the time range."""
        texts = []
        for seg in self.segments:
            if seg.end > start_s and seg.start < end_s:
                texts.append(seg.text)
        return " ".join(texts)

    def get_segments_in_range(self, start_s: float, end_s: float) -> List[TranscriptSegment]:
        """Get segments that overlap with the time range."""
        return [s for s in self.segments if s.end > start_s and s.start < end_s]

    def get_words_in_range(self, start_s: float, end_s: float) -> List[TranscriptWord]:
        """Get all words that fall within the time range."""
        words = []
        for seg in self.segments:
            if seg.words:
                for w in seg.words:
                    if w.end > start_s and w.start < end_s:
                        words.append(w)
        return words


def _extract_full_audio_wav(video_path: Path, sr: int, wav_path: Path) -> None:
    """Extract the full audio track as WAV for transcription."""
    _require_cmd("ffmpeg")
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "wav",
        str(wav_path),
    ]
    subprocess.check_call(cmd, **_subprocess_flags())


def _use_new_transcription_engine(
    proj: Project,
    cfg: TranscriptConfig,
    wav_path: Path,
    on_progress: Optional[Callable[[float], None]] = None,
) -> tuple[List[TranscriptSegment], Optional[str], str, bool, Optional[List[str]], bool]:
    """Use the new pluggable transcription engine.
    
    Returns:
        Tuple of (segments, detected_language, backend_used, gpu_used, speakers, diarization_used)
    """
    from .transcription import TranscriberConfig, get_transcriber
    
    # Convert our config to the new format
    transcriber_cfg = TranscriberConfig(
        backend=cfg.backend,
        model=cfg.model_size,
        language=cfg.language,
        use_gpu=cfg.use_gpu or cfg.device == "cuda",
        threads=cfg.threads,
        n_processors=cfg.n_processors,
        vad_filter=cfg.vad_filter,
        word_timestamps=cfg.word_timestamps,
        sample_rate=cfg.sample_rate,
        compute_type=cfg.compute_type,
        strict=cfg.strict,
        verbose=cfg.verbose,
        diarize=cfg.diarize,
        diarize_min_speakers=cfg.diarize_min_speakers,
        diarize_max_speakers=cfg.diarize_max_speakers,
        hf_token=cfg.hf_token,
    )
    
    # Get transcriber with fallback
    transcriber = get_transcriber(transcriber_cfg)
    
    logger.info(f"Using transcription backend: {transcriber.backend_name}")
    
    # Determine progress allocation based on whether diarization is enabled
    diarization_enabled = cfg.diarize
    if diarization_enabled:
        # Transcription: 20%-70%, Diarization: 70%-90%
        transcribe_progress_start = 0.2
        transcribe_progress_end = 0.7
        diarize_progress_start = 0.7
        diarize_progress_end = 0.9
    else:
        # Transcription: 20%-90%
        transcribe_progress_start = 0.2
        transcribe_progress_end = 0.9
    
    # Wrap progress callback for transcription
    def progress_wrapper(p: float) -> None:
        if on_progress:
            mapped = transcribe_progress_start + (transcribe_progress_end - transcribe_progress_start) * p
            on_progress(mapped)
    
    # Transcribe
    result = transcriber.transcribe(wav_path, on_progress=progress_wrapper)
    
    # Unload transcription model to free GPU memory before diarization
    transcriber.unload_model()
    
    # Run diarization if enabled
    speakers: Optional[List[str]] = None
    diarization_used = False
    
    if diarization_enabled:
        try:
            from .transcription.diarization import (
                is_diarization_available,
                diarize_audio,
                merge_diarization_with_transcript,
            )
            from .transcription import TranscriptResult as TResult
            
            if is_diarization_available():
                logger.info("Running speaker diarization...")
                
                def diarize_progress(p: float) -> None:
                    if on_progress:
                        mapped = diarize_progress_start + (diarize_progress_end - diarize_progress_start) * p
                        on_progress(mapped)
                
                # Run diarization
                diarization = diarize_audio(
                    wav_path,
                    hf_token=cfg.hf_token,
                    use_gpu=cfg.use_gpu,
                    min_speakers=cfg.diarize_min_speakers,
                    max_speakers=cfg.diarize_max_speakers,
                    exclusive=True,
                    on_progress=diarize_progress,
                )
                
                # Merge with transcript
                result = merge_diarization_with_transcript(result, diarization)
                speakers = diarization.speakers
                diarization_used = True
                logger.info(f"Diarization complete: found {len(speakers)} speakers")
            else:
                logger.warning("Diarization requested but pyannote-audio not installed. Skipping.")
        except ImportError as e:
            logger.warning(f"Diarization not available: {e}")
        except Exception as e:
            logger.warning(f"Diarization failed: {e}. Continuing without speaker labels.")
    
    # Convert to our segment format
    segments: List[TranscriptSegment] = []
    for seg in result.segments:
        words = None
        if seg.words:
            words = [
                TranscriptWord(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                    speaker=getattr(w, 'speaker', None),
                )
                for w in seg.words
            ]
        
        segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=words,
            speaker=getattr(seg, 'speaker', None),
        ))
    
    return segments, result.language, result.backend_used, result.gpu_used, speakers, diarization_used


def _config_matches(saved_config: Dict[str, Any], cfg: TranscriptConfig) -> bool:
    """Check if a saved transcript config matches the current config.
    
    Returns True if the cached transcript can be reused.
    """
    # Core parameters that affect transcription output
    critical_keys = [
        "backend",
        "model_size",
        "language",
        "word_timestamps",
    ]
    
    for key in critical_keys:
        saved_value = saved_config.get(key)
        current_value = getattr(cfg, key, None)
        
        # Handle "auto" backend: accept any backend
        if key == "backend" and (saved_value == "auto" or current_value == "auto"):
            continue
        
        # Handle None language (auto-detect): always accept
        if key == "language" and (saved_value is None or current_value is None):
            continue
        
        if saved_value != current_value:
            return False
    
    return True


def compute_transcript_analysis(
    proj: Project,
    *,
    cfg: TranscriptConfig,
    on_progress: Optional[Callable[[float], None]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Transcribe the full video using the configured backend.

    Supports whisper.cpp (pywhispercpp) and faster-whisper backends.
    Backend selection:
      - "auto": tries whispercpp first, falls back to faster-whisper
      - "whispercpp": uses pywhispercpp (fast CPU, GPU with Vulkan build)
      - "faster_whisper": uses faster-whisper (CUDA GPU support)

    Args:
        proj: Project to transcribe
        cfg: Transcription configuration
        on_progress: Optional progress callback
        force: If True, ignore cache and re-transcribe

    Persists:
      - analysis/transcript_full.json (segments with timestamps)
      - project.json -> analysis.transcript section
    
    Returns cached transcript if:
      - transcript_full.json exists
      - Config (backend, model, language, word_timestamps) matches
      - force=False
    """
    video_path = Path(proj.audio_source)  # Use audio_source for fallback during early analysis
    transcript_path = proj.analysis_dir / "transcript_full.json"
    
    # Check for cached transcript (major speedup on re-runs)
    if not force and transcript_path.exists():
        try:
            cached_data = json.loads(transcript_path.read_text(encoding="utf-8"))
            saved_config = cached_data.get("config", {})
            
            if _config_matches(saved_config, cfg):
                logger.info(
                    f"Using cached transcript ({cached_data.get('segment_count', 0)} segments, "
                    f"backend={cached_data.get('backend_used', 'unknown')})"
                )
                if on_progress:
                    on_progress(1.0)
                return cached_data
            else:
                logger.info(
                    f"Transcript cache exists but config differs, re-transcribing "
                    f"(saved={saved_config.get('model_size')}, current={cfg.model_size})"
                )
        except Exception as e:
            logger.warning(f"Failed to load cached transcript: {e}")
    
    # Track computation time
    import time as _time
    start_time = _time.time()
    
    # Get duration from ffprobe (may be 0 if it fails)
    ffprobe_duration = ffprobe_duration_seconds(video_path)

    if on_progress:
        on_progress(0.05)

    # Extract audio to temporary file (shared by all backends)
    with tempfile.TemporaryDirectory(prefix="vp_transcript_") as td:
        td_path = Path(td)
        wav_path = td_path / "audio.wav"
        _extract_full_audio_wav(video_path, cfg.sample_rate, wav_path)

        if on_progress:
            on_progress(0.2)

        # Use the new transcription engine
        try:
            segments, detected_language, backend_used, gpu_used, speakers, diarization_used = _use_new_transcription_engine(
                proj, cfg, wav_path, on_progress
            )
        except Exception as e:
            # Log the error but re-raise
            logger.error(f"Transcription failed: {e}")
            raise

    # Compute final duration: prefer ffprobe, fallback to max segment end
    duration_from_segments = max((seg.end for seg in segments), default=0.0)
    duration_s = ffprobe_duration if ffprobe_duration > 0 else duration_from_segments

    transcript = FullTranscript(
        segments=segments,
        language=detected_language or cfg.language,
        duration_seconds=duration_s,
    )

    # Calculate elapsed time
    elapsed_seconds = _time.time() - start_time

    # Save transcript JSON
    transcript_path = proj.analysis_dir / "transcript_full.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "config": {
            "backend": cfg.backend,
            "model_size": cfg.model_size,
            "language": cfg.language,
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "sample_rate": cfg.sample_rate,
            "vad_filter": cfg.vad_filter,
            "word_timestamps": cfg.word_timestamps,
            "use_gpu": cfg.use_gpu,
            "threads": cfg.threads,
            "n_processors": cfg.n_processors,
            "strict": cfg.strict,
            "diarize": cfg.diarize,
        },
        "backend_used": backend_used,
        "gpu_used": gpu_used,
        "diarization_used": diarization_used,
        "speakers": speakers,
        "detected_language": transcript.language,
        "duration_seconds": transcript.duration_seconds,
        "segment_count": len(transcript.segments),
        "transcript": transcript.to_dict(),
    }

    save_json(transcript_path, payload)

    # Update project.json
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["transcript"] = {
            "created_at": payload["created_at"],
            "elapsed_seconds": elapsed_seconds,
            "config": payload["config"],
            "backend_used": backend_used,
            "gpu_used": gpu_used,
            "diarization_used": diarization_used,
            "speakers": speakers,
            "detected_language": transcript.language,
            "duration_seconds": transcript.duration_seconds,
            "segment_count": len(transcript.segments),
            "transcript_path": str(transcript_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload


def load_transcript(proj: Project) -> Optional[FullTranscript]:
    """Load the cached transcript for a project, if available."""
    transcript_path = proj.analysis_dir / "transcript_full.json"
    if not transcript_path.exists():
        return None

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    return FullTranscript.from_dict(data.get("transcript", {}))


def compute_transcript_analysis_from_audio(
    audio_path: Path,
    *,
    cfg: TranscriptConfig,
    on_progress: Optional[Callable[[float], None]] = None,
    duration_hint: float = 0.0,
) -> Dict[str, Any]:
    """Transcribe directly from an audio file (no video extraction needed).

    This is used for early transcription during download - the audio is downloaded
    first as a smaller file, transcription starts while video download continues.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        cfg: Transcription configuration
        on_progress: Optional progress callback
        duration_hint: Approximate duration in seconds (for progress calculation)

    Returns:
        Dict with transcript data (same format as compute_transcript_analysis)
        Can be saved to a project later using save_transcript_to_project()
    """
    from .transcription import TranscriberConfig, get_transcriber
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Track computation time (for pipeline status timers)
    import time as _time
    start_time = _time.time()

    if on_progress:
        on_progress(0.05)

    # Convert audio to WAV format if needed (whisper expects 16kHz mono WAV)
    needs_conversion = audio_path.suffix.lower() not in [".wav"]
    
    if needs_conversion:
        with tempfile.TemporaryDirectory(prefix="vp_transcript_") as td:
            td_path = Path(td)
            wav_path = td_path / "audio.wav"
            
            # Convert to WAV
            _require_cmd("ffmpeg")
            import subprocess
            cmd = [
                "ffmpeg",
                "-y",
                "-v", "error",
                "-i", str(audio_path),
                "-vn",
                "-ac", "1",
                "-ar", str(cfg.sample_rate),
                "-f", "wav",
                str(wav_path),
            ]
            subprocess.check_call(cmd, **_subprocess_flags())
            
            if on_progress:
                on_progress(0.15)
            
            segments, detected_language, backend_used, gpu_used, speakers, diarization_used = _transcribe_audio_file(
                wav_path, cfg, on_progress
            )
    else:
        # Already a WAV file, use directly
        if on_progress:
            on_progress(0.15)
        
        segments, detected_language, backend_used, gpu_used, speakers, diarization_used = _transcribe_audio_file(
            audio_path, cfg, on_progress
        )

    # Compute duration from segments
    duration_from_segments = max((seg.end for seg in segments), default=0.0)
    duration_s = duration_hint if duration_hint > 0 else duration_from_segments

    transcript = FullTranscript(
        segments=segments,
        language=detected_language or cfg.language,
        duration_seconds=duration_s,
    )

    # Calculate elapsed time
    elapsed_seconds = _time.time() - start_time

    # Build payload (not saved to disk yet - that happens when project is created)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "config": {
            "backend": cfg.backend,
            "model_size": cfg.model_size,
            "language": cfg.language,
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "sample_rate": cfg.sample_rate,
            "vad_filter": cfg.vad_filter,
            "word_timestamps": cfg.word_timestamps,
            "use_gpu": cfg.use_gpu,
            "threads": cfg.threads,
            "n_processors": cfg.n_processors,
            "strict": cfg.strict,
            "diarize": cfg.diarize,
        },
        "backend_used": backend_used,
        "gpu_used": gpu_used,
        "diarization_used": diarization_used,
        "speakers": speakers,
        "detected_language": transcript.language,
        "duration_seconds": transcript.duration_seconds,
        "segment_count": len(transcript.segments),
        "transcript": transcript.to_dict(),
    }

    if on_progress:
        on_progress(1.0)

    return payload


def _transcribe_audio_file(
    wav_path: Path,
    cfg: TranscriptConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> tuple[List[TranscriptSegment], Optional[str], str, bool, Optional[List[str]], bool]:
    """Internal: Transcribe a WAV file using the configured backend.
    
    Returns:
        Tuple of (segments, detected_language, backend_used, gpu_used, speakers, diarization_used)
    """
    from .transcription import TranscriberConfig, get_transcriber
    
    # Convert our config to the new format
    transcriber_cfg = TranscriberConfig(
        backend=cfg.backend,
        model=cfg.model_size,
        language=cfg.language,
        use_gpu=cfg.use_gpu or cfg.device == "cuda",
        threads=cfg.threads,
        n_processors=cfg.n_processors,
        vad_filter=cfg.vad_filter,
        word_timestamps=cfg.word_timestamps,
        sample_rate=cfg.sample_rate,
        compute_type=cfg.compute_type,
        strict=cfg.strict,
        verbose=cfg.verbose,
        diarize=cfg.diarize,
        diarize_min_speakers=cfg.diarize_min_speakers,
        diarize_max_speakers=cfg.diarize_max_speakers,
        hf_token=cfg.hf_token,
    )
    
    # Get transcriber with fallback
    transcriber = get_transcriber(transcriber_cfg)
    
    logger.info(f"Using transcription backend: {transcriber.backend_name}")
    
    # Wrap progress callback
    def progress_wrapper(p: float) -> None:
        if on_progress:
            # Map 0-1 to 0.15-0.95 (conversion was 0-0.15, final is 0.95-1.0)
            on_progress(0.15 + 0.8 * p)
    
    # Transcribe
    result = transcriber.transcribe(wav_path, on_progress=progress_wrapper)
    
    # Unload model to free GPU memory
    transcriber.unload_model()
    
    # Run diarization if requested and available
    speakers: Optional[List[str]] = None
    diarization_used = False
    
    if cfg.diarize:
        from .transcription import is_diarization_available, diarize_audio, merge_diarization_with_transcript
        
        if is_diarization_available():
            logger.info("Running speaker diarization...")
            
            try:
                diarization_result = diarize_audio(
                    wav_path,
                    hf_token=cfg.hf_token,
                    min_speakers=cfg.diarize_min_speakers,
                    max_speakers=cfg.diarize_max_speakers,
                    exclusive=True,
                )
                
                # Merge diarization with transcript
                result = merge_diarization_with_transcript(result, diarization_result)
                speakers = diarization_result.speakers
                diarization_used = True
                
                logger.info(f"Diarization complete: identified {len(speakers)} speakers")
            except Exception as e:
                logger.warning(f"Diarization failed: {e}. Continuing without speaker labels.")
        else:
            logger.warning("Diarization requested but pyannote-audio not installed. "
                          "Install with: pip install pyannote-audio")
    
    # Convert to our segment format
    segments: List[TranscriptSegment] = []
    for seg in result.segments:
        words = None
        if seg.words:
            words = [
                TranscriptWord(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                    speaker=getattr(w, 'speaker', None),
                )
                for w in seg.words
            ]
        
        segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=words,
            speaker=getattr(seg, 'speaker', None),
        ))
    
    return segments, result.language, result.backend_used, result.gpu_used, speakers, diarization_used


def save_transcript_to_project(
    proj: Project,
    transcript_data: Dict[str, Any],
) -> None:
    """Save pre-computed transcript data to a project.
    
    This is used when transcription was done early (during download)
    and now needs to be saved to the project directory.
    
    Args:
        proj: Project to save transcript to
        transcript_data: Transcript data from compute_transcript_analysis_from_audio()
    """
    # Save transcript JSON
    transcript_path = proj.analysis_dir / "transcript_full.json"
    save_json(transcript_path, transcript_data)

    # Update project.json
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["transcript"] = {
            "created_at": transcript_data["created_at"],
            "elapsed_seconds": transcript_data.get("elapsed_seconds"),
            "config": transcript_data["config"],
            "backend_used": transcript_data.get("backend_used", "unknown"),
            "gpu_used": transcript_data.get("gpu_used", False),
            "diarization_used": bool(transcript_data.get("diarization_used", False)),
            "speakers": transcript_data.get("speakers"),
            "detected_language": transcript_data.get("detected_language"),
            "duration_seconds": transcript_data.get("duration_seconds", 0.0),
            "segment_count": transcript_data.get("segment_count", 0),
            "transcript_path": str(transcript_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)
    logger.info(f"Saved early transcript to project: {transcript_data.get('segment_count', 0)} segments")


def merge_diarization_json_into_transcript(proj: Project) -> bool:
    """Merge `analysis/diarization.json` speaker labels into `analysis/transcript_full.json`.

    This is a lightweight post-processing step that assigns a `speaker` label to:
      - transcript segments (dominant diarization speaker over the segment window)
      - transcript words (speaker over the word window; falls back to segment speaker)

    Returns:
        True if a merge was applied, False otherwise.
    """
    diar_path = proj.analysis_dir / "diarization.json"
    transcript_path = proj.analysis_dir / "transcript_full.json"
    if not diar_path.exists() or not transcript_path.exists():
        return False

    try:
        diar_data = json.loads(diar_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[transcript] Failed to read diarization.json for merge: %s", e)
        return False

    try:
        transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except Exception as e:
        # Transcript may still be in the process of being written by another task.
        logger.info("[transcript] Transcript not ready for diarization merge (%s)", e)
        return False

    diar_segments_raw = diar_data.get("speaker_segments") or []
    if not isinstance(diar_segments_raw, list) or not diar_segments_raw:
        return False

    diar_segments: list[tuple[float, float, str]] = []
    for s in diar_segments_raw:
        if not isinstance(s, dict):
            continue
        try:
            spk = str(s.get("speaker", ""))
            st = float(s.get("start_s", 0.0))
            en = float(s.get("end_s", st))
        except Exception:
            continue
        if not spk or en <= st:
            continue
        diar_segments.append((st, en, spk))

    if not diar_segments:
        return False

    diar_segments.sort(key=lambda t: (t[0], t[1]))

    t_obj = transcript_data.get("transcript") or {}
    if not isinstance(t_obj, dict):
        return False

    t_segments = t_obj.get("segments") or []
    if not isinstance(t_segments, list) or not t_segments:
        return False

    def dominant_speaker_in_range(start_s: float, end_s: float, idx: int) -> tuple[Optional[str], int]:
        n = len(diar_segments)
        i = idx
        while i < n and diar_segments[i][1] <= start_s:
            i += 1

        speaker_times: Dict[str, float] = {}
        j = i
        while j < n and diar_segments[j][0] < end_s:
            ds_st, ds_en, ds_spk = diar_segments[j]
            ov_st = max(ds_st, start_s)
            ov_en = min(ds_en, end_s)
            if ov_st < ov_en:
                speaker_times[ds_spk] = speaker_times.get(ds_spk, 0.0) + (ov_en - ov_st)
            j += 1

        if not speaker_times:
            return None, i
        spk = max(speaker_times.items(), key=lambda kv: kv[1])[0]
        return spk, i

    applied = False
    seg_idx = 0
    word_idx = 0

    for seg in t_segments:
        if not isinstance(seg, dict):
            continue
        try:
            st = float(seg.get("start", 0.0))
            en = float(seg.get("end", st))
        except Exception:
            continue
        if en <= st:
            continue

        seg_speaker, seg_idx = dominant_speaker_in_range(st, en, seg_idx)
        if seg_speaker is not None:
            if seg.get("speaker") != seg_speaker:
                seg["speaker"] = seg_speaker
                applied = True
        else:
            # If transcript had a speaker label already, preserve it.
            seg_speaker = seg.get("speaker") if isinstance(seg.get("speaker"), str) else None

        words = seg.get("words")
        if isinstance(words, list) and words:
            for w in words:
                if not isinstance(w, dict):
                    continue
                try:
                    w_st = float(w.get("start", 0.0))
                    w_en = float(w.get("end", w_st))
                except Exception:
                    continue
                if w_en <= w_st:
                    continue

                w_spk, word_idx = dominant_speaker_in_range(w_st, w_en, word_idx)
                if w_spk is None:
                    w_spk = seg_speaker
                if w_spk is not None and w.get("speaker") != w_spk:
                    w["speaker"] = w_spk
                    applied = True

    if not applied:
        return False

    # Update transcript header metadata.
    speakers = diar_data.get("speakers")
    if not isinstance(speakers, list):
        speakers = sorted({spk for _, _, spk in diar_segments})
    else:
        speakers = [str(s) for s in speakers if str(s)]

    transcript_data["diarization_used"] = True
    transcript_data["speakers"] = speakers
    transcript_data["diarization_merged_from"] = {
        "generated_at": diar_data.get("generated_at"),
        "speaker_count": diar_data.get("speaker_count"),
        "diarization_path": str(diar_path.relative_to(proj.project_dir)),
    }

    save_json(transcript_path, transcript_data)

    # Keep project.json in sync (UI reads from this summary).
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        t = d["analysis"].get("transcript") or {}
        if not isinstance(t, dict):
            t = {}
        t["diarization_used"] = True
        t["speakers"] = speakers
        t["diarization_merged_from"] = transcript_data.get("diarization_merged_from")
        d["analysis"]["transcript"] = t

    update_project(proj, _upd)

    return True
