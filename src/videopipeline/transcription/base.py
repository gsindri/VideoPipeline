"""Base types and protocol for transcription backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, runtime_checkable


class TranscriberError(RuntimeError):
    """Base error for transcription failures."""
    pass


class BackendNotAvailableError(TranscriberError):
    """Raised when a requested backend is not installed or available."""
    pass


@dataclass
class TranscriptWord:
    """A single word with timing information."""
    word: str
    start: float
    end: float
    probability: float = 1.0
    speaker: Optional[str] = None  # Speaker label (e.g., "SPEAKER_00", "SPEAKER_01")

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
    """A segment of transcribed speech with optional word-level timing."""
    start: float
    end: float
    text: str
    words: Optional[List[TranscriptWord]] = None
    speaker: Optional[str] = None  # Speaker label (e.g., "SPEAKER_00", "SPEAKER_01")

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
class TranscriptResult:
    """Result of a transcription operation."""
    segments: List[TranscriptSegment]
    language: Optional[str] = None
    duration_seconds: float = 0.0
    backend_used: str = ""
    gpu_used: bool = False
    speakers: Optional[List[str]] = None  # List of speaker labels found (e.g., ["SPEAKER_00", "SPEAKER_01"])
    diarization_used: bool = False  # Whether speaker diarization was applied

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "backend_used": self.backend_used,
            "gpu_used": self.gpu_used,
            "diarization_used": self.diarization_used,
        }
        if self.speakers:
            d["speakers"] = self.speakers
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TranscriptResult":
        return cls(
            segments=[TranscriptSegment.from_dict(s) for s in d.get("segments", [])],
            language=d.get("language"),
            duration_seconds=float(d.get("duration_seconds", 0.0)),
            backend_used=str(d.get("backend_used", "")),
            gpu_used=bool(d.get("gpu_used", False)),
            speakers=d.get("speakers"),
            diarization_used=bool(d.get("diarization_used", False)),
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


BackendType = Literal["whispercpp", "faster_whisper", "openai_whisper", "auto"]


@dataclass
class TranscriberConfig:
    """Configuration for transcription engine.
    
    Attributes:
        backend: Which engine to use ("openai_whisper", "faster_whisper", "whispercpp", "auto")
        model: Model size ("tiny", "base", "small", "medium", "large")
               Also supports quantized models: "small.en-q8_0", "tiny-q5_1"
        language: Language code (None for auto-detect)
        use_gpu: Whether to attempt GPU acceleration (AMD ROCm or NVIDIA CUDA)
        threads: Number of CPU threads (0 = use all cores)
        n_processors: Number of parallel decoders for long files (whisper.cpp only)
        vad_filter: Use voice activity detection to filter silence
        word_timestamps: Generate word-level timestamps
        sample_rate: Audio sample rate (16000 for Whisper)
        compute_type: Compute type for faster-whisper ("int8", "float16", etc.)
        strict: If True, don't fall back to other backends (for A/B testing)
        verbose: If True, print transcript segments to console as they are transcribed
        diarize: If True, run speaker diarization to identify who is speaking
        diarize_min_speakers: Minimum number of speakers to detect (None for auto)
        diarize_max_speakers: Maximum number of speakers to detect (None for auto)
        hf_token: Hugging Face token for pyannote models (can also use HF_TOKEN env var)
    """
    backend: BackendType = "openai_whisper"
    model: str = "small"
    language: Optional[str] = None
    use_gpu: bool = True
    threads: int = 0  # 0 = use all cores
    n_processors: int = 1  # Parallel decoders (whisper.cpp), >1 for long files
    vad_filter: bool = True
    word_timestamps: bool = True
    sample_rate: int = 16000
    compute_type: str = "int8"  # For faster-whisper: int8 (CPU), float16 (GPU)
    strict: bool = False  # If True, don't fall back to other backends
    verbose: bool = False  # If True, print transcript to console during transcription
    
    # Speaker diarization options
    diarize: bool = False  # Enable speaker diarization
    diarize_min_speakers: Optional[int] = None  # Min speakers (None = auto-detect)
    diarize_max_speakers: Optional[int] = None  # Max speakers (None = auto-detect)
    hf_token: Optional[str] = None  # Hugging Face token for pyannote
    
    # Model path override (if using local models)
    model_path: Optional[str] = None

    @classmethod
    def from_profile(cls, speech_cfg: Dict[str, Any]) -> "TranscriberConfig":
        """Create config from profile speech settings."""
        return cls(
            backend=speech_cfg.get("backend", "auto"),
            model=speech_cfg.get("model_size", speech_cfg.get("model", "small")),
            language=speech_cfg.get("language"),
            use_gpu=speech_cfg.get("use_gpu", speech_cfg.get("device", "cpu") != "cpu"),
            threads=speech_cfg.get("threads", 0),
            n_processors=speech_cfg.get("n_processors", 1),
            vad_filter=speech_cfg.get("vad_filter", True),
            word_timestamps=speech_cfg.get("word_timestamps", True),
            sample_rate=speech_cfg.get("sample_rate", 16000),
            compute_type=speech_cfg.get("compute_type", "int8"),
            model_path=speech_cfg.get("model_path"),
            strict=speech_cfg.get("strict", False),
            verbose=speech_cfg.get("verbose", False),
            diarize=speech_cfg.get("diarize", False),
            diarize_min_speakers=speech_cfg.get("diarize_min_speakers"),
            diarize_max_speakers=speech_cfg.get("diarize_max_speakers"),
            hf_token=speech_cfg.get("hf_token"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "use_gpu": self.use_gpu,
            "threads": self.threads,
            "n_processors": self.n_processors,
            "vad_filter": self.vad_filter,
            "word_timestamps": self.word_timestamps,
            "sample_rate": self.sample_rate,
            "compute_type": self.compute_type,
            "model_path": self.model_path,
            "strict": self.strict,
            "verbose": self.verbose,
            "diarize": self.diarize,
            "diarize_min_speakers": self.diarize_min_speakers,
            "diarize_max_speakers": self.diarize_max_speakers,
            "hf_token": self.hf_token,
        }


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for transcription backends."""
    
    @property
    def backend_name(self) -> str:
        """Name of this backend (e.g., 'whispercpp', 'faster_whisper')."""
        ...
    
    @property
    def gpu_available(self) -> bool:
        """Whether GPU acceleration is available for this backend."""
        ...
    
    def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> TranscriptResult:
        """Transcribe an audio file.
        
        Args:
            audio_path: Path to WAV file (16kHz mono recommended)
            on_progress: Optional callback for progress updates (0.0 to 1.0)
            
        Returns:
            TranscriptResult with segments, language, and metadata
        """
        ...


class BaseTranscriber(ABC):
    """Abstract base class for transcription backends."""
    
    def __init__(self, config: TranscriberConfig):
        self.config = config
        self._model = None
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of this backend."""
        pass
    
    @property
    @abstractmethod
    def gpu_available(self) -> bool:
        """Whether GPU is available."""
        pass
    
    @abstractmethod
    def _load_model(self) -> Any:
        """Load the transcription model."""
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> TranscriptResult:
        """Transcribe an audio file."""
        pass
    
    def ensure_model_loaded(self) -> Any:
        """Ensure the model is loaded, loading it if necessary."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def unload_model(self) -> None:
        """Unload the model to free memory (especially GPU VRAM)."""
        if self._model is not None:
            del self._model
            self._model = None
