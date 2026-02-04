"""faster-whisper backend.

This is the fallback backend using CTranslate2-based faster-whisper.
Supports CUDA GPUs but not AMD/Vulkan.

Optimizations:
- threads: Controls CPU threads (0 = use all cores)
- vad_filter: Built-in silero VAD for skipping silence (major speedup)
- compute_type: int8 for fast CPU, float16 for GPU
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, List, Optional

from .base import (
    BaseTranscriber,
    BackendNotAvailableError,
    TranscriberConfig,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)
from ..utils import subprocess_flags

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if faster-whisper is installed."""
    try:
        from faster_whisper import WhisperModel
        return True
    except ImportError:
        return False


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available for faster-whisper."""
    if not is_available():
        return False
    
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using faster-whisper (CTranslate2)."""
    
    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self._gpu_used = False
        
        if not is_available():
            raise BackendNotAvailableError(
                "faster-whisper is not installed. Install with: pip install faster-whisper"
            )
    
    @property
    def backend_name(self) -> str:
        return "faster_whisper"
    
    @property
    def gpu_available(self) -> bool:
        return check_gpu_available()
    
    def _get_device_and_compute_type(self) -> tuple[str, str]:
        """Determine device and compute type based on config and availability."""
        if self.config.use_gpu and self.gpu_available:
            self._gpu_used = True
            return "cuda", "float16"
        else:
            self._gpu_used = False
            return "cpu", self.config.compute_type
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds, supporting multiple formats."""
        # Try ffprobe first (handles all formats)
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path)
                ],
                capture_output=True,
                text=True,
                timeout=10,
                **subprocess_flags(),
            )
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                logger.debug(f"Audio duration (ffprobe): {duration:.1f}s")
                return duration
        except Exception as e:
            logger.debug(f"ffprobe failed: {e}")
        
        # Fallback: try wave module for WAV files
        if audio_path.suffix.lower() == ".wav":
            try:
                import wave
                with wave.open(str(audio_path), 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
                    logger.debug(f"Audio duration (wave): {duration:.1f}s")
                    return duration
            except Exception:
                pass
        
        # Final fallback: estimate from file size
        # Typical audio: ~128kbps = 16KB/s
        try:
            size_bytes = audio_path.stat().st_size
            estimated = size_bytes / 16000  # Rough estimate
            logger.debug(f"Audio duration (estimated): {estimated:.1f}s")
            return max(estimated, 60.0)  # At least 1 minute
        except Exception:
            return 300.0  # Default 5 minutes

    def _load_model(self) -> Any:
        """Load the faster-whisper model."""
        from faster_whisper import WhisperModel
        
        device, compute_type = self._get_device_and_compute_type()
        
        model_name = self.config.model
        if self.config.model_path:
            model_name = self.config.model_path
        
        # Thread count: 0 means use all cores
        cpu_threads = self.config.threads
        if cpu_threads <= 0:
            cpu_threads = os.cpu_count() or 4
        
        logger.info(
            f"Loading faster-whisper model: {model_name} on {device} "
            f"({compute_type}, {cpu_threads} threads)"
        )
        
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        
        return model
    
    def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> TranscriptResult:
        """Transcribe audio using faster-whisper."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if on_progress:
            on_progress(0.05)
        
        # Get audio duration for accurate progress tracking
        # Use ffprobe or fallback to estimate - faster-whisper handles all formats internally
        audio_duration_s = self._get_audio_duration(audio_path)
        
        model = self.ensure_model_loaded()
        
        if on_progress:
            on_progress(0.1)
        
        # Build transcription options
        transcribe_opts = {
            "language": self.config.language,
            "vad_filter": self.config.vad_filter,
            "word_timestamps": self.config.word_timestamps,
        }
        
        # Log transcription settings
        lang_str = self.config.language or "auto-detect"
        vad_str = "enabled" if self.config.vad_filter else "disabled"
        logger.info(
            f"Transcribing: {audio_path} "
            f"(lang={lang_str}, VAD={vad_str}, word_timestamps={self.config.word_timestamps})"
        )
        
        # Run transcription
        segments_iter, info = model.transcribe(str(audio_path), **transcribe_opts)
        
        # Process segments
        segments: List[TranscriptSegment] = []
        max_end_time = 0.0
        
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            
            max_end_time = max(max_end_time, float(seg.end))
            
            # Word-level timestamps if requested and available
            words = None
            if self.config.word_timestamps and hasattr(seg, "words") and seg.words:
                words = [
                    TranscriptWord(
                        word=w.word.strip(),
                        start=float(w.start),
                        end=float(w.end),
                        probability=float(getattr(w, "probability", 1.0)),
                    )
                    for w in seg.words
                    if w.word.strip()
                ]
            
            segments.append(TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                words=words,
            ))
            
            # Update progress based on actual audio duration
            if on_progress and audio_duration_s > 0:
                # Map 0.1-0.9 for transcription progress
                internal_progress = min(0.95, seg.end / audio_duration_s)
                prog = 0.1 + 0.8 * internal_progress
                on_progress(prog)
        
        if on_progress:
            on_progress(1.0)
        
        return TranscriptResult(
            segments=segments,
            language=info.language if hasattr(info, "language") else self.config.language,
            duration_seconds=max_end_time,
            backend_used=self.backend_name,
            gpu_used=self._gpu_used,
        )
