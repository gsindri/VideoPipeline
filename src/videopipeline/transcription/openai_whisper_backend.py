"""OpenAI Whisper backend using PyTorch.

This backend uses the original openai-whisper implementation.
Supports AMD GPUs via ROCm PyTorch on Windows, as well as CUDA GPUs.

Key benefits:
- Works with AMD GPUs using AMD's PyTorch on Windows Edition (ROCm)
- Works with NVIDIA GPUs via standard CUDA PyTorch
- Uses torch.cuda API which ROCm maps transparently

Optimizations:
- fp16: Uses half precision on GPU for speed/memory efficiency
- Supports all standard Whisper model sizes
"""

from __future__ import annotations

import io
import logging
import os
import re
import subprocess
import sys
import threading
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

# Regex to match whisper's verbose output: [00:00.000 --> 00:05.000] text
_TIMESTAMP_PATTERN = re.compile(r'\[(\d+):(\d+)\.(\d+)\s*-->')


class TqdmProgressCallback:
    """Wraps tqdm to capture progress and forward to a callback.
    
    Whisper uses tqdm internally when verbose=False. We monkey-patch tqdm
    to intercept progress updates and call our callback.
    """
    
    def __init__(
        self,
        on_progress: Optional[Callable[[float], None]] = None,
        base_progress: float = 0.1,
        max_progress: float = 0.9,
    ):
        self.on_progress = on_progress
        self.base_progress = base_progress
        self.max_progress = max_progress
        self._original_tqdm = None
        self._total = 0
        self._current = 0
    
    def _calculate_progress(self) -> float:
        """Calculate progress value (base_progress to max_progress range)."""
        if self._total <= 0:
            return self.base_progress
        ratio = min(self._current / self._total, 1.0)
        return self.base_progress + ratio * (self.max_progress - self.base_progress)
    
    def _create_wrapper(self):
        """Create a tqdm wrapper class that reports progress."""
        callback_self = self
        original_tqdm = self._original_tqdm
        
        class TqdmWrapper(original_tqdm):
            def __init__(self, *args, **kwargs):
                # Disable the visual bar but keep tracking
                kwargs['disable'] = True
                super().__init__(*args, **kwargs)
                callback_self._total = self.total or 0
                callback_self._current = 0
            
            def update(self, n=1):
                result = super().update(n)
                callback_self._current = self.n
                if callback_self.on_progress:
                    progress = callback_self._calculate_progress()
                    callback_self.on_progress(progress)
                return result
        
        return TqdmWrapper
    
    def __enter__(self):
        import tqdm as tqdm_module
        self._original_tqdm = tqdm_module.tqdm
        tqdm_module.tqdm = self._create_wrapper()
        return self
    
    def __exit__(self, *args):
        import tqdm as tqdm_module
        if self._original_tqdm:
            tqdm_module.tqdm = self._original_tqdm
            self._original_tqdm = None


class ProgressCapture:
    """Captures stdout to extract whisper progress from timestamps.
    
    Whisper outputs lines like: [00:30.000 --> 00:35.000] Some text
    We parse the start timestamp and calculate progress from total duration.
    
    Note: This is a fallback for when verbose=True. For better progress tracking,
    use TqdmProgressCallback with verbose=False.
    """
    
    def __init__(
        self,
        total_duration: float,
        on_progress: Optional[Callable[[float], None]] = None,
        base_progress: float = 0.1,  # Progress value when starting (after model load)
        max_progress: float = 0.9,   # Max progress before post-processing
    ):
        self.total_duration = total_duration
        self.on_progress = on_progress
        self.base_progress = base_progress
        self.max_progress = max_progress
        self._original_stdout = None
        self._original_stderr = None
        self._last_progress = base_progress
    
    def _parse_timestamp(self, line: str) -> Optional[float]:
        """Extract timestamp in seconds from whisper output line."""
        match = _TIMESTAMP_PATTERN.search(line)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            # Milliseconds might be 3 digits (000) or truncated
            ms_str = match.group(3)
            milliseconds = int(ms_str) * (10 ** (3 - len(ms_str))) if ms_str else 0
            return minutes * 60 + seconds + milliseconds / 1000
        return None
    
    def _calculate_progress(self, current_time: float) -> float:
        """Calculate progress value (base_progress to max_progress range)."""
        if self.total_duration <= 0:
            return self.base_progress
        ratio = min(current_time / self.total_duration, 1.0)
        return self.base_progress + ratio * (self.max_progress - self.base_progress)
    
    def write(self, text: str) -> int:
        """Intercept writes, parse for progress, pass through to original stdout."""
        # Always write to original stdout so user sees progress
        if self._original_stdout:
            self._original_stdout.write(text)
            self._original_stdout.flush()
        
        # Parse timestamp and update progress
        timestamp = self._parse_timestamp(text)
        if timestamp is not None and self.on_progress:
            progress = self._calculate_progress(timestamp)
            # Only update if progress increased (avoid jitter)
            if progress > self._last_progress:
                self._last_progress = progress
                self.on_progress(progress)
        
        return len(text)
    
    def flush(self):
        if self._original_stdout:
            self._original_stdout.flush()
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self
        return self
    
    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        self._original_stdout = None


def is_available() -> bool:
    """Check if openai-whisper is installed."""
    try:
        import whisper
        return True
    except ImportError:
        return False


def _patch_whisper_triton_import():
    """Patch whisper's triton_ops import to gracefully handle missing triton.
    
    The openai-whisper package (v20250625+) tries to import triton kernels for
    CUDA acceleration. If triton isn't installed, this causes an ImportError.
    
    The whisper code catches RuntimeError but not ImportError, so we need to
    patch the timing module to handle this case gracefully.
    """
    try:
        import sys

        # If whisper isn't installed, there's nothing to patch.
        try:
            import whisper  # noqa: F401
        except ImportError:
            return
        
        # Create a stub triton_ops module if the real one fails to import
        if 'whisper.triton_ops' not in sys.modules:
            try:
                # Try importing the real module first
                import whisper.triton_ops
            except (ImportError, RuntimeError) as e:
                # Create a dummy module that raises RuntimeError on kernel launch.
                #
                # Newer openai-whisper versions use Triton kernels that are launched
                # via subscript syntax: `kernel[grid](...)`.
                # If we provide a plain Python function, whisper will crash with:
                #   TypeError: 'function' object is not subscriptable
                # Instead we provide an object that supports both patterns:
                #   - kernel[grid](...)  -> __getitem__ then __call__
                #   - kernel(...)        -> __call__
                # In both cases we raise RuntimeError so whisper's own fallback path
                # (CPU implementations) can handle it.
                import types
                stub_module = types.ModuleType('whisper.triton_ops')
                
                class _TritonKernelStub:
                    def __getitem__(self, _grid):  # noqa: ANN001
                        return self

                    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
                        raise RuntimeError("triton not available")

                stub_module.median_filter_cuda = _TritonKernelStub()
                stub_module.dtw_kernel = _TritonKernelStub()
                
                sys.modules['whisper.triton_ops'] = stub_module
                logger.debug(f"Patched whisper.triton_ops stub (triton not available: {e})")
    except Exception as e:
        logger.debug(f"Could not patch whisper triton import: {e}")


# Apply the patch when this module loads
_patch_whisper_triton_import()


def check_gpu_available() -> bool:
    """Check if GPU is available for openai-whisper (CUDA or ROCm)."""
    if not is_available():
        return False
    
    try:
        import torch
        # torch.cuda.is_available() returns True for both CUDA and ROCm
        return torch.cuda.is_available()
    except Exception:
        return False


def get_gpu_name() -> Optional[str]:
    """Get the name of the GPU if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


class OpenAIWhisperTranscriber(BaseTranscriber):
    """Transcriber using openai-whisper (PyTorch).
    
    This backend supports:
    - AMD GPUs via ROCm PyTorch on Windows
    - NVIDIA GPUs via standard CUDA PyTorch
    - CPU fallback
    """
    
    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self._gpu_used = False
        self._device: Optional[str] = None
        
        if not is_available():
            raise BackendNotAvailableError(
                "openai-whisper is not installed. Install with: pip install openai-whisper"
            )
    
    @property
    def backend_name(self) -> str:
        return "openai_whisper"
    
    @property
    def gpu_available(self) -> bool:
        return check_gpu_available()
    
    def _get_device(self) -> str:
        """Determine device based on config and availability."""
        if self.config.use_gpu and self.gpu_available:
            self._gpu_used = True
            gpu_name = get_gpu_name()
            if gpu_name:
                logger.info(f"Using GPU: {gpu_name}")
            return "cuda"
        else:
            self._gpu_used = False
            return "cpu"
    
    def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        if self._model is not None:
            logger.info("Unloading openai-whisper model to free GPU memory")
            del self._model
            self._model = None
            self._device = None
            
            # Force CUDA to release memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.debug("GPU memory cache cleared")
            except Exception as e:
                logger.debug(f"Could not clear GPU cache: {e}")
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
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
                return float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"ffprobe failed: {e}")
        
        # Fallback estimate
        try:
            size_bytes = audio_path.stat().st_size
            return max(size_bytes / 16000, 60.0)
        except Exception:
            return 300.0

    def _load_model(self) -> Any:
        """Load the openai-whisper model."""
        import whisper
        
        self._device = self._get_device()
        model_name = self.config.model
        
        # Map common model names
        # openai-whisper uses: tiny, base, small, medium, large, large-v2, large-v3
        # Also supports .en variants: tiny.en, base.en, small.en, medium.en
        
        logger.info(
            f"Loading openai-whisper model: {model_name} on {self._device}"
        )
        
        model = whisper.load_model(model_name, device=self._device)
        
        return model
    
    def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> TranscriptResult:
        """Transcribe audio using openai-whisper."""
        import warnings
        
        # Suppress harmless Triton/CUDA warnings from Whisper's timing module
        # These occur when Triton kernels can't be launched (missing full CUDA toolkit)
        # and Whisper falls back to slower CPU implementations for median/DTW operations
        warnings.filterwarnings(
            "ignore",
            message=".*Failed to launch Triton kernels.*",
            category=UserWarning,
            module="whisper.timing"
        )
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if on_progress:
            on_progress(0.05)
        
        audio_duration_s = self._get_audio_duration(audio_path)
        
        model = self.ensure_model_loaded()
        
        if on_progress:
            on_progress(0.1)
        
        # Build transcription options with GPU optimizations
        transcribe_opts: dict[str, Any] = {
            "word_timestamps": self.config.word_timestamps,
            "fp16": self._device == "cuda",  # Use fp16 on GPU for speed/memory
            "verbose": self.config.verbose,  # Print transcript to console if enabled
            # Speed optimizations:
            "condition_on_previous_text": False,  # Faster, slight quality tradeoff
            "compression_ratio_threshold": 2.4,  # Skip bad segments
            "logprob_threshold": -1.0,  # Quality threshold
            "no_speech_threshold": 0.6,  # Skip silence faster
            "temperature": 0,  # Greedy decoding - faster than beam search fallbacks
        }
        
        # Beam search: use smaller beam for speed
        # With temperature=0 and no fallback, beam_size=1 is essentially greedy
        if self._device == "cuda":
            transcribe_opts["beam_size"] = 1  # Greedy - fastest
            transcribe_opts["best_of"] = 1  # No sampling
        else:
            transcribe_opts["beam_size"] = 1  # Also fast on CPU
            transcribe_opts["best_of"] = 1
        
        if self.config.language:
            transcribe_opts["language"] = self.config.language
        
        # Log transcription settings
        lang_str = self.config.language or "auto-detect"
        logger.info(
            f"Transcribing: {audio_path} ({audio_duration_s:.1f}s) "
            f"(lang={lang_str}, device={self._device}, word_timestamps={self.config.word_timestamps})"
        )
        
        # Run transcription with progress capture
        # When verbose=True, Whisper outputs timestamps to stdout - we capture those
        # When verbose=False, Whisper uses tqdm internally - we intercept that
        if self.config.verbose:
            # Use ProgressCapture to parse timestamps from verbose output
            with ProgressCapture(
                total_duration=audio_duration_s,
                on_progress=on_progress,
                base_progress=0.1,
                max_progress=0.9,
            ):
                result = model.transcribe(str(audio_path), **transcribe_opts)
        else:
            # Use TqdmProgressCallback to capture tqdm progress
            with TqdmProgressCallback(
                on_progress=on_progress,
                base_progress=0.1,
                max_progress=0.9,
            ):
                result = model.transcribe(str(audio_path), **transcribe_opts)
        
        if on_progress:
            on_progress(0.9)
        
        # Process segments
        segments: List[TranscriptSegment] = []
        max_end_time = 0.0
        
        for seg in result.get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            max_end_time = max(max_end_time, end)
            
            # Word-level timestamps if available
            words = None
            if self.config.word_timestamps and "words" in seg and seg["words"]:
                words = [
                    TranscriptWord(
                        word=w.get("word", "").strip(),
                        start=float(w.get("start", 0.0)),
                        end=float(w.get("end", 0.0)),
                        probability=float(w.get("probability", 1.0)),
                    )
                    for w in seg["words"]
                    if w.get("word", "").strip()
                ]
            
            segments.append(TranscriptSegment(
                start=start,
                end=end,
                text=text,
                words=words,
            ))
        
        if on_progress:
            on_progress(1.0)
        
        return TranscriptResult(
            segments=segments,
            language=result.get("language", self.config.language),
            duration_seconds=max_end_time,
            backend_used=self.backend_name,
            gpu_used=self._gpu_used,
        )
