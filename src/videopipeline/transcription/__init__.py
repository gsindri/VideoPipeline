"""Transcription engine abstraction layer.

Provides a pluggable backend for speech-to-text transcription:
- openai-whisper (PyTorch - supports AMD GPUs via ROCm, NVIDIA via CUDA)
- faster-whisper (CTranslate2 - NVIDIA CUDA only, fastest on CUDA)
- NVIDIA NeMo ASR (PyTorch - optimized for NVIDIA CUDA)
- whisper.cpp via pywhispercpp (fast CPU, Vulkan GPU with custom build)
- AssemblyAI (cloud API transcription + optional speaker labels)

Optional speaker diarization via pyannote-audio:
- Identifies who is speaking when
- Merges speaker labels with transcription

Usage:
    from videopipeline.transcription import get_transcriber, TranscriberConfig

    config = TranscriberConfig(backend="openai_whisper", model="small", use_gpu=True)
    transcriber = get_transcriber(config)
    segments = transcriber.transcribe(audio_path)

    # With speaker diarization:
    config = TranscriberConfig(backend="openai_whisper", diarize=True)
    transcriber = get_transcriber(config)
    result = transcriber.transcribe(audio_path)
    # result.segments[0].speaker == "SPEAKER_00"
"""

from .base import (
    BackendNotAvailableError,
    Transcriber,
    TranscriberConfig,
    TranscriberError,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)
from .factory import get_available_backends, get_transcriber

# Diarization is optional - import conditionally
try:
    from . import diarization as _diarization

    DiarizationResult = _diarization.DiarizationResult
    DiarizationSegment = _diarization.DiarizationSegment
    diarize_audio = _diarization.diarize_audio
    is_diarization_available = _diarization.is_diarization_available
    merge_diarization_with_transcript = _diarization.merge_diarization_with_transcript
    _HAS_DIARIZATION = True
except ImportError:
    _HAS_DIARIZATION = False

    def is_diarization_available() -> bool:
        return False

__all__ = [
    "TranscriberConfig",
    "TranscriptSegment",
    "TranscriptWord",
    "TranscriptResult",
    "Transcriber",
    "TranscriberError",
    "BackendNotAvailableError",
    "get_transcriber",
    "get_available_backends",
    "is_diarization_available",
]

if _HAS_DIARIZATION:
    __all__.extend([
        "diarize_audio",
        "merge_diarization_with_transcript",
        "DiarizationResult",
        "DiarizationSegment",
    ])
