"""Factory for creating transcriber instances.

Handles backend selection, fallback, and runtime detection.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Type

from .base import (
    BackendNotAvailableError,
    BaseTranscriber,
    Transcriber,
    TranscriberConfig,
)

logger = logging.getLogger(__name__)


# Lazy imports to avoid loading unused backends
def _get_whispercpp_class() -> Optional[Type[BaseTranscriber]]:
    try:
        from .whispercpp_backend import WhisperCppTranscriber, is_available
        if is_available():
            return WhisperCppTranscriber
    except ImportError:
        pass
    return None


def _get_faster_whisper_class() -> Optional[Type[BaseTranscriber]]:
    try:
        from .faster_whisper_backend import FasterWhisperTranscriber, is_available
        if is_available():
            return FasterWhisperTranscriber
    except ImportError:
        pass
    return None


def _get_openai_whisper_class() -> Optional[Type[BaseTranscriber]]:
    try:
        from .openai_whisper_backend import OpenAIWhisperTranscriber, is_available
        if is_available():
            return OpenAIWhisperTranscriber
    except ImportError:
        pass
    return None


def _get_nemo_asr_class() -> Optional[Type[BaseTranscriber]]:
    try:
        from .nemo_asr_backend import NemoASRTranscriber, is_available

        if is_available():
            return NemoASRTranscriber
    except ImportError:
        pass
    return None


def _get_assemblyai_class() -> Optional[Type[BaseTranscriber]]:
    try:
        from .assemblyai_backend import AssemblyAITranscriber, is_available

        if is_available():
            return AssemblyAITranscriber
    except ImportError:
        pass
    return None


def get_available_backends() -> Dict[str, bool]:
    """Get a dict of available backends and their status.

    Returns:
        Dict mapping backend name to availability (True/False)
    """
    backends = {}

    # Check whisper.cpp
    try:
        from .whispercpp_backend import check_gpu_available, is_available
        backends["whispercpp"] = is_available()
        backends["whispercpp_gpu"] = check_gpu_available()
    except ImportError:
        backends["whispercpp"] = False
        backends["whispercpp_gpu"] = False

    # Check faster-whisper
    try:
        from .faster_whisper_backend import check_gpu_available, is_available
        backends["faster_whisper"] = is_available()
        backends["faster_whisper_gpu"] = check_gpu_available()
    except ImportError:
        backends["faster_whisper"] = False
        backends["faster_whisper_gpu"] = False

    # Check openai-whisper
    try:
        from .openai_whisper_backend import check_gpu_available, is_available
        backends["openai_whisper"] = is_available()
        backends["openai_whisper_gpu"] = check_gpu_available()
    except ImportError:
        backends["openai_whisper"] = False
        backends["openai_whisper_gpu"] = False

    # Check NVIDIA NeMo ASR
    try:
        from .nemo_asr_backend import check_gpu_available, is_available

        backends["nemo_asr"] = is_available()
        backends["nemo_asr_gpu"] = check_gpu_available()
    except ImportError:
        backends["nemo_asr"] = False
        backends["nemo_asr_gpu"] = False

    # Check AssemblyAI cloud backend
    try:
        from .assemblyai_backend import check_gpu_available, is_available

        backends["assemblyai"] = is_available()
        backends["assemblyai_gpu"] = check_gpu_available()
    except ImportError:
        backends["assemblyai"] = False
        backends["assemblyai_gpu"] = False

    return backends


def get_transcriber(config: TranscriberConfig, *, strict: Optional[bool] = None) -> Transcriber:
    """Create a transcriber instance based on config.

    Backend selection logic:
    1. If backend="auto": pick best installed local backend for current hardware
    2. If backend specified and strict=False: try that backend, fall back if unavailable
    3. If backend specified and strict=True: use ONLY that backend, error if unavailable
    4. If use_gpu=True but GPU unavailable: log warning, continue on CPU

    Args:
        config: Transcription configuration
        strict: If True, don't fall back to other backends (useful for A/B testing).
                If None, uses config.strict. Defaults to False.

    Returns:
        Transcriber instance

    Raises:
        BackendNotAvailableError: If requested backend is unavailable (in strict mode)
                                  or if no backend is available
    """
    backend = config.backend

    # Determine strict mode: parameter overrides config
    if strict is None:
        strict = getattr(config, 'strict', False)

    # Get available backend classes
    whispercpp_cls = _get_whispercpp_class()
    faster_whisper_cls = _get_faster_whisper_class()
    openai_whisper_cls = _get_openai_whisper_class()
    nemo_asr_cls = _get_nemo_asr_class()
    assemblyai_cls = _get_assemblyai_class()

    # Log available backends
    available = get_available_backends()
    logger.debug(f"Available transcription backends: {available}")
    logger.debug(f"Strict mode: {strict}")

    # Auto selection logic:
    # - If use_gpu=True: prefer NeMo on CUDA, then openai_whisper, then faster_whisper, then whispercpp
    # - If use_gpu=False: prefer faster_whisper, then openai_whisper, then whispercpp
    # NOTE: assemblyai is never auto-selected to avoid unexpected API usage/cost.
    if backend == "auto":
        if config.use_gpu:
            if nemo_asr_cls:
                logger.info("Using NeMo ASR backend (auto-selected for GPU)")
                backend = "nemo_asr"
            # GPU path: openai_whisper supports AMD (ROCm) + NVIDIA (CUDA)
            elif openai_whisper_cls:
                logger.info("Using openai-whisper backend (auto-selected for GPU, supports AMD/NVIDIA)")
                backend = "openai_whisper"
            elif faster_whisper_cls:
                logger.info("Using faster-whisper backend (auto-selected for GPU, NVIDIA only)")
                backend = "faster_whisper"
            elif whispercpp_cls:
                logger.info("Using whisper.cpp backend (auto-selected, needs Vulkan build for GPU)")
                backend = "whispercpp"
            else:
                raise BackendNotAvailableError(
                    "No transcription backend available. "
                    "Install NeMo, openai-whisper, faster-whisper, or pywhispercpp: "
                    "pip install nemo_toolkit[asr]"
                )
        else:
            # CPU path: faster_whisper is fastest
            if faster_whisper_cls:
                logger.info("Using faster-whisper backend (auto-selected for CPU)")
                backend = "faster_whisper"
            elif openai_whisper_cls:
                logger.info("Using openai-whisper backend (auto-selected, others unavailable)")
                backend = "openai_whisper"
            elif whispercpp_cls:
                logger.info("Using whisper.cpp backend (auto-selected, faster-whisper unavailable)")
                backend = "whispercpp"
            else:
                raise BackendNotAvailableError(
                    "No transcription backend available. "
                    "Install faster-whisper, openai-whisper, or pywhispercpp: "
                    "pip install faster-whisper"
                )

    # Create the requested backend (with optional fallback)
    if backend == "whispercpp":
        if whispercpp_cls:
            try:
                transcriber = whispercpp_cls(config)

                # Check GPU availability if requested
                if config.use_gpu and not transcriber.gpu_available:
                    logger.warning(
                        "GPU requested but not available for whisper.cpp. "
                        "Running on CPU. To enable GPU, build pywhispercpp with GGML_VULKAN=1"
                    )

                return transcriber
            except Exception as e:
                logger.warning(f"Failed to initialize whisper.cpp: {e}")
                if strict:
                    raise BackendNotAvailableError(
                        f"whisper.cpp backend failed to initialize (strict mode): {e}"
                    )
                if faster_whisper_cls:
                    logger.info("Falling back to faster-whisper")
                    return faster_whisper_cls(config)
                raise
        elif strict:
            raise BackendNotAvailableError(
                "whisper.cpp backend requested but pywhispercpp is not installed (strict mode). "
                "Install with: pip install pywhispercpp"
            )
        elif faster_whisper_cls:
            logger.warning("whisper.cpp not available, falling back to faster-whisper")
            return faster_whisper_cls(config)
        else:
            raise BackendNotAvailableError(
                "whisper.cpp backend requested but pywhispercpp is not installed"
            )

    elif backend == "faster_whisper":
        if faster_whisper_cls:
            try:
                transcriber = faster_whisper_cls(config)

                # Check GPU availability if requested
                if config.use_gpu and not transcriber.gpu_available:
                    logger.warning(
                        "GPU requested but CUDA not available for faster-whisper. "
                        "Running on CPU."
                    )

                return transcriber
            except Exception as e:
                logger.warning(f"Failed to initialize faster-whisper: {e}")
                if strict:
                    raise BackendNotAvailableError(
                        f"faster-whisper backend failed to initialize (strict mode): {e}"
                    )
                if whispercpp_cls:
                    logger.info("Falling back to whisper.cpp")
                    return whispercpp_cls(config)
                raise
        elif strict:
            raise BackendNotAvailableError(
                "faster-whisper backend requested but it is not installed (strict mode). "
                "Install with: pip install faster-whisper"
            )
        elif whispercpp_cls:
            logger.warning("faster-whisper not available, falling back to whisper.cpp")
            return whispercpp_cls(config)
        else:
            raise BackendNotAvailableError(
                "faster-whisper backend requested but it is not installed"
            )

    elif backend == "nemo_asr":
        if nemo_asr_cls:
            try:
                transcriber = nemo_asr_cls(config)

                if config.use_gpu and not transcriber.gpu_available:
                    logger.warning(
                        "GPU requested but CUDA not available for NeMo ASR. "
                        "Running on CPU."
                    )

                return transcriber
            except Exception as e:
                logger.warning(f"Failed to initialize NeMo ASR: {e}")
                if strict:
                    raise BackendNotAvailableError(
                        f"NeMo ASR backend failed to initialize (strict mode): {e}"
                    )
                if openai_whisper_cls:
                    logger.info("Falling back to openai-whisper")
                    return openai_whisper_cls(config)
                if faster_whisper_cls:
                    logger.info("Falling back to faster-whisper")
                    return faster_whisper_cls(config)
                if whispercpp_cls:
                    logger.info("Falling back to whisper.cpp")
                    return whispercpp_cls(config)
                raise
        elif strict:
            raise BackendNotAvailableError(
                "NeMo ASR backend requested but it is not installed (strict mode). "
                "Install with: pip install nemo_toolkit[asr]"
            )
        elif openai_whisper_cls:
            logger.warning("NeMo ASR not available, falling back to openai-whisper")
            return openai_whisper_cls(config)
        elif faster_whisper_cls:
            logger.warning("NeMo ASR not available, falling back to faster-whisper")
            return faster_whisper_cls(config)
        elif whispercpp_cls:
            logger.warning("NeMo ASR not available, falling back to whisper.cpp")
            return whispercpp_cls(config)
        else:
            raise BackendNotAvailableError(
                "NeMo ASR backend requested but it is not installed"
            )

    elif backend == "openai_whisper":
        if openai_whisper_cls:
            try:
                transcriber = openai_whisper_cls(config)

                # Check GPU availability if requested
                if config.use_gpu and not transcriber.gpu_available:
                    logger.warning(
                        "GPU requested but not available for openai-whisper. "
                        "Running on CPU. For AMD GPU support, install AMD's PyTorch on Windows."
                    )

                return transcriber
            except Exception as e:
                logger.warning(f"Failed to initialize openai-whisper: {e}")
                if strict:
                    raise BackendNotAvailableError(
                        f"openai-whisper backend failed to initialize (strict mode): {e}"
                    )
                # Fall back to NeMo, then faster-whisper, then whispercpp
                if nemo_asr_cls:
                    logger.info("Falling back to NeMo ASR")
                    return nemo_asr_cls(config)
                if faster_whisper_cls:
                    logger.info("Falling back to faster-whisper")
                    return faster_whisper_cls(config)
                if whispercpp_cls:
                    logger.info("Falling back to whisper.cpp")
                    return whispercpp_cls(config)
                raise
        elif strict:
            raise BackendNotAvailableError(
                "openai-whisper backend requested but it is not installed (strict mode). "
                "Install with: pip install openai-whisper"
            )
        elif nemo_asr_cls:
            logger.warning("openai-whisper not available, falling back to NeMo ASR")
            return nemo_asr_cls(config)
        elif faster_whisper_cls:
            logger.warning("openai-whisper not available, falling back to faster-whisper")
            return faster_whisper_cls(config)
        elif whispercpp_cls:
            logger.warning("openai-whisper not available, falling back to whisper.cpp")
            return whispercpp_cls(config)
        else:
            raise BackendNotAvailableError(
                "openai-whisper backend requested but it is not installed"
            )

    elif backend == "assemblyai":
        if assemblyai_cls:
            try:
                return assemblyai_cls(config)
            except Exception as e:
                logger.warning(f"Failed to initialize AssemblyAI backend: {e}")
                if strict:
                    raise BackendNotAvailableError(
                        f"AssemblyAI backend failed to initialize (strict mode): {e}"
                    )
                if nemo_asr_cls:
                    logger.info("Falling back to NeMo ASR")
                    return nemo_asr_cls(config)
                if openai_whisper_cls:
                    logger.info("Falling back to openai-whisper")
                    return openai_whisper_cls(config)
                if faster_whisper_cls:
                    logger.info("Falling back to faster-whisper")
                    return faster_whisper_cls(config)
                if whispercpp_cls:
                    logger.info("Falling back to whisper.cpp")
                    return whispercpp_cls(config)
                raise
        elif strict:
            raise BackendNotAvailableError(
                "AssemblyAI backend requested but it is not available (strict mode). "
                "Install with: pip install assemblyai and set ASSEMBLYAI_API_KEY."
            )
        elif nemo_asr_cls:
            logger.warning("AssemblyAI not available, falling back to NeMo ASR")
            return nemo_asr_cls(config)
        elif openai_whisper_cls:
            logger.warning("AssemblyAI not available, falling back to openai-whisper")
            return openai_whisper_cls(config)
        elif faster_whisper_cls:
            logger.warning("AssemblyAI not available, falling back to faster-whisper")
            return faster_whisper_cls(config)
        elif whispercpp_cls:
            logger.warning("AssemblyAI not available, falling back to whisper.cpp")
            return whispercpp_cls(config)
        else:
            raise BackendNotAvailableError(
                "AssemblyAI backend requested but it is not available. "
                "Install with: pip install assemblyai and set ASSEMBLYAI_API_KEY."
            )

    else:
        raise ValueError(f"Unknown backend: {backend}")
