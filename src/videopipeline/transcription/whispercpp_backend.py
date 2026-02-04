"""whisper.cpp backend using pywhispercpp.

This is the preferred backend - faster than faster-whisper on CPU,
and supports GPU via Vulkan when built with GGML_VULKAN=1.

Optimizations:
- threads: 0 means "use all CPU cores" (not pywhispercpp's default of 4)
- n_processors: Split audio into chunks for parallel processing
- vad_filter: Uses silero-vad to skip silence (major speedup)
- Supports quantized models (e.g., small.en-q8_0, tiny-q5_1)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from .base import (
    BaseTranscriber,
    BackendNotAvailableError,
    TranscriberConfig,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if pywhispercpp is installed."""
    try:
        from pywhispercpp.model import Model
        return True
    except ImportError:
        return False


def check_gpu_available() -> bool:
    """Check if GPU acceleration is available in pywhispercpp.
    
    Note: Pre-built wheels are CPU-only. GPU requires building from source
    with GGML_VULKAN=1 (or GGML_CUDA=1 for NVIDIA).
    """
    if not is_available():
        return False
    
    # We can't easily detect GPU support without loading a model.
    # The whisper.cpp logs will show "no GPU found" if unavailable.
    # For now, return False and let the backend handle fallback.
    return False


class WhisperCppTranscriber(BaseTranscriber):
    """Transcriber using whisper.cpp via pywhispercpp."""
    
    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self._gpu_used = False
        
        if not is_available():
            raise BackendNotAvailableError(
                "pywhispercpp is not installed. Install with: pip install pywhispercpp"
            )
    
    @property
    def backend_name(self) -> str:
        return "whispercpp"
    
    @property
    def gpu_available(self) -> bool:
        return self._gpu_used
    
    def _get_model_name(self) -> str:
        """Get the model name for pywhispercpp.
        
        pywhispercpp uses names like 'tiny.en', 'base', 'small', etc.
        Also supports quantized models: 'small.en-q8_0', 'tiny-q5_1', etc.
        
        Quantized models provide significant CPU speedup:
        - q8_0: Good balance of speed and quality
        - q5_1: Faster, slight quality loss
        - q5_0: Fastest, more quality loss
        """
        model = self.config.model.lower()
        
        # Standard models
        standard_models = {
            "tiny.en", "tiny",
            "base.en", "base",
            "small.en", "small",
            "medium.en", "medium",
            "large", "large-v1", "large-v2", "large-v3",
        }
        
        # Quantized suffixes supported by pywhispercpp
        quantized_suffixes = ["-q8_0", "-q5_1", "-q5_0", "-q4_1", "-q4_0"]
        
        # Check if it's already a valid model name
        if model in standard_models:
            return model
        
        # Check if it's a quantized model (e.g., "small.en-q8_0")
        for suffix in quantized_suffixes:
            if model.endswith(suffix):
                base_model = model[:-len(suffix)]
                if base_model in standard_models:
                    return model  # Return as-is, pywhispercpp handles it
        
        # Fall back to returning as-is (let pywhispercpp handle errors)
        return model
    
    def _run_vad(self, audio_path: Path) -> Optional[List[Tuple[float, float]]]:
        """Run VAD to detect speech segments.
        
        Returns list of (start, end) tuples in seconds, or None if VAD unavailable.
        Uses silero-vad which is fast and accurate.
        """
        try:
            import torch
            import numpy as np
            import wave
            
            # Load audio
            with wave.open(str(audio_path), 'rb') as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Load silero-vad model (cached after first load)
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            (get_speech_timestamps, _, _, _, _) = utils
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                torch.from_numpy(audio),
                model,
                sampling_rate=sr,
                threshold=0.3,  # Lower threshold = more sensitive
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
                speech_pad_ms=30,
            )
            
            if not speech_timestamps:
                logger.warning("VAD found no speech in audio")
                return None
            
            # Convert to seconds
            segments = [
                (ts['start'] / sr, ts['end'] / sr)
                for ts in speech_timestamps
            ]
            
            # Merge nearby segments (within 0.5s)
            merged = []
            for start, end in segments:
                if merged and start - merged[-1][1] < 0.5:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((start, end))
            
            total_speech = sum(end - start for start, end in merged)
            total_duration = len(audio) / sr
            logger.info(
                f"VAD: {len(merged)} speech segments, "
                f"{total_speech:.1f}s speech / {total_duration:.1f}s total "
                f"({100*total_speech/total_duration:.0f}%)"
            )
            
            return merged
            
        except ImportError:
            logger.debug("silero-vad not available (need torch), skipping VAD pre-filtering")
            return None
        except Exception as e:
            logger.warning(f"VAD failed, transcribing full audio: {e}")
            return None
    
    def _extract_speech_segments(
        self,
        audio_path: Path,
        segments: List[Tuple[float, float]],
        output_path: Path,
    ) -> List[Tuple[float, float]]:
        """Extract speech segments to a new audio file.
        
        Returns the mapping from new file positions to original timestamps.
        """
        import wave
        import numpy as np
        
        with wave.open(str(audio_path), 'rb') as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
        
        # Extract segments with small padding
        pad_samples = int(0.1 * sr)  # 100ms padding
        extracted = []
        segment_map = []  # Maps new positions back to original times
        current_pos = 0.0
        
        for orig_start, orig_end in segments:
            start_sample = max(0, int(orig_start * sr) - pad_samples)
            end_sample = min(len(audio), int(orig_end * sr) + pad_samples)
            
            segment_audio = audio[start_sample:end_sample]
            extracted.append(segment_audio)
            
            # Record mapping: new_start -> original_start
            new_duration = len(segment_audio) / sr
            segment_map.append((current_pos, start_sample / sr, orig_start, orig_end))
            current_pos += new_duration
        
        # Write concatenated audio
        concatenated = np.concatenate(extracted)
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sr)
            wf.writeframes(concatenated.tobytes())
        
        return segment_map
    
    def _remap_timestamps(
        self,
        segments: List["TranscriptSegment"],
        segment_map: List[Tuple[float, float, float, float]],
    ) -> List["TranscriptSegment"]:
        """Remap segment timestamps from VAD-filtered audio back to original times."""
        if not segment_map:
            return segments
        
        remapped = []
        for seg in segments:
            # Find which VAD segment this transcription segment falls into
            for new_start, audio_start, orig_start, orig_end in segment_map:
                new_end = new_start + (orig_end - orig_start + 0.2)  # Account for padding
                
                if seg.start >= new_start and seg.start < new_end:
                    # Remap timestamps
                    offset = audio_start - new_start
                    new_seg = TranscriptSegment(
                        start=seg.start + offset,
                        end=seg.end + offset,
                        text=seg.text,
                        words=None,
                    )
                    
                    # Remap word timestamps too
                    if seg.words:
                        new_seg.words = [
                            TranscriptWord(
                                word=w.word,
                                start=w.start + offset,
                                end=w.end + offset,
                                probability=w.probability,
                            )
                            for w in seg.words
                        ]
                    
                    remapped.append(new_seg)
                    break
            else:
                # Segment didn't map, keep original (shouldn't happen normally)
                remapped.append(seg)
        
        return remapped
    
    def _load_model(self) -> Any:
        """Load the pywhispercpp model."""
        from pywhispercpp.model import Model
        
        model_name = self._get_model_name()
        
        # If a specific model path is provided, use it
        if self.config.model_path:
            model_path = self.config.model_path
        else:
            model_path = model_name
        
        logger.info(f"Loading whisper.cpp model: {model_path}")
        
        # Model constructor only takes model name/path and optionally models_dir
        # Other params like n_threads go in transcribe() call
        model = Model(model_path)
        
        return model
    
    def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> TranscriptResult:
        """Transcribe audio using whisper.cpp.
        
        If vad_filter is enabled, uses silero-vad to pre-filter silence,
        which can dramatically speed up transcription (2-5x on typical streams).
        """
        import wave
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if on_progress:
            on_progress(0.02)
        
        # Get audio duration for progress estimation
        audio_duration_s = 0.0
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                audio_duration_s = frames / float(rate)
            logger.debug(f"Audio duration: {audio_duration_s:.1f}s")
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            audio_duration_s = 0.0
        
        # VAD pre-filtering: extract only speech segments
        # This can speed up transcription 2-5x on streams with silence/downtime
        vad_segment_map = None
        transcribe_path = audio_path
        temp_dir = None
        
        if self.config.vad_filter:
            if on_progress:
                on_progress(0.05)
            
            vad_segments = self._run_vad(audio_path)
            
            if vad_segments:
                # Create temporary file with only speech segments
                temp_dir = tempfile.mkdtemp(prefix="vp_vad_")
                vad_audio_path = Path(temp_dir) / "vad_audio.wav"
                
                try:
                    vad_segment_map = self._extract_speech_segments(
                        audio_path, vad_segments, vad_audio_path
                    )
                    transcribe_path = vad_audio_path
                    
                    # Update expected duration for progress tracking
                    with wave.open(str(vad_audio_path), 'rb') as wf:
                        audio_duration_s = wf.getnframes() / float(wf.getframerate())
                    
                    logger.info(f"VAD-filtered audio: {audio_duration_s:.1f}s to transcribe")
                except Exception as e:
                    logger.warning(f"VAD extraction failed, using full audio: {e}")
                    vad_segment_map = None
        
        model = self.ensure_model_loaded()
        
        if on_progress:
            on_progress(0.1)
        
        # Build transcription parameters
        # See pywhispercpp get_params_schema() for all available params
        transcribe_kwargs: dict = {
            "print_progress": False,
            "print_realtime": False,
        }
        
        # Thread count:
        # If user sets 0, use all logical cores for speed.
        # pywhispercpp default is min(4, cores), which underutilizes modern CPUs.
        n_threads = self.config.threads
        if n_threads <= 0:
            n_threads = os.cpu_count() or 4
        transcribe_kwargs["n_threads"] = int(n_threads)
        logger.debug(f"Using {n_threads} threads for whisper.cpp")
        
        # Multi-processor support for long files
        # n_processors splits audio and runs multiple decoder instances
        n_processors = getattr(self.config, 'n_processors', 1)
        if n_processors > 1:
            transcribe_kwargs["n_processors"] = n_processors
            logger.debug(f"Using {n_processors} processors for parallel decoding")
        
        if self.config.language:
            transcribe_kwargs["language"] = self.config.language
        
        # Token timestamps enable word-level timing
        if self.config.word_timestamps:
            transcribe_kwargs["token_timestamps"] = True
        
        # Progress callback using segment timestamps
        # We estimate progress based on how far through the audio we've transcribed
        last_progress_time = [0.0]  # Use list to allow mutation in closure
        
        def segment_callback(seg):
            if on_progress and audio_duration_s > 0 and hasattr(seg, "t1"):
                # t1 is end time in centiseconds (100ths of a second)
                current_time_s = seg.t1 / 100.0
                # Only update if we've made progress (avoid going backwards)
                if current_time_s > last_progress_time[0]:
                    last_progress_time[0] = current_time_s
                    # Map progress: 0.1 (model loaded) to 0.9 (transcription done)
                    # So internal progress 0-1 maps to 0.1-0.9
                    internal_progress = min(0.95, current_time_s / audio_duration_s)
                    overall_progress = 0.1 + 0.8 * internal_progress
                    on_progress(overall_progress)
        
        if on_progress and audio_duration_s > 0:
            transcribe_kwargs["new_segment_callback"] = segment_callback
        
        # Run transcription
        logger.info(f"Transcribing: {transcribe_path}")
        raw_segments = model.transcribe(str(transcribe_path), **transcribe_kwargs)
        
        # Cleanup VAD temp file
        if temp_dir:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        
        if on_progress:
            on_progress(0.9)
        
        # Convert to our segment format
        segments: List[TranscriptSegment] = []
        detected_language = None
        max_end_time = 0.0
        
        for seg in raw_segments:
            text = str(seg.text).strip() if hasattr(seg, "text") else ""
            if not text:
                continue
            
            # pywhispercpp segment times are in seconds
            start = float(seg.t0) / 100.0 if hasattr(seg, "t0") else 0.0
            end = float(seg.t1) / 100.0 if hasattr(seg, "t1") else 0.0
            
            max_end_time = max(max_end_time, end)
            
            # Word-level timestamps if available
            words = None
            if self.config.word_timestamps and hasattr(seg, "tokens"):
                words = []
                for token in seg.tokens:
                    if hasattr(token, "text") and token.text.strip():
                        word_text = token.text.strip()
                        word_start = float(token.t0) / 100.0 if hasattr(token, "t0") else start
                        word_end = float(token.t1) / 100.0 if hasattr(token, "t1") else end
                        prob = float(token.p) if hasattr(token, "p") else 1.0
                        
                        words.append(TranscriptWord(
                            word=word_text,
                            start=word_start,
                            end=word_end,
                            probability=prob,
                        ))
            
            segments.append(TranscriptSegment(
                start=start,
                end=end,
                text=text,
                words=words if words else None,
            ))
        
        # Remap timestamps back to original audio if we used VAD filtering
        if vad_segment_map:
            segments = self._remap_timestamps(segments, vad_segment_map)
            # Recalculate max_end_time from remapped segments
            max_end_time = max((seg.end for seg in segments), default=0.0)
        
        if on_progress:
            on_progress(1.0)
        
        return TranscriptResult(
            segments=segments,
            language=detected_language or self.config.language,
            duration_seconds=max_end_time,
            backend_used=self.backend_name,
            gpu_used=self._gpu_used,
        )
