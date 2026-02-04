"""Factory for creating transcriber instances.

Handles backend selection, fallback, and runtime detection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from .base import (
    BackendNotAvailableError,
    BaseTranscriber,
    Transcriber,
    TranscriberConfig,
    TranscriberError,
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


def get_available_backends() -> Dict[str, bool]:
    """Get a dict of available backends and their status.
    
    Returns:
        Dict mapping backend name to availability (True/False)
    """
    backends = {}
    
    # Check whisper.cpp
    try:
        from .whispercpp_backend import is_available, check_gpu_available
        backends["whispercpp"] = is_available()
        backends["whispercpp_gpu"] = check_gpu_available()
    except ImportError:
        backends["whispercpp"] = False
        backends["whispercpp_gpu"] = False
    
    # Check faster-whisper
    try:
        from .faster_whisper_backend import is_available, check_gpu_available
        backends["faster_whisper"] = is_available()
        backends["faster_whisper_gpu"] = check_gpu_available()
    except ImportError:
        backends["faster_whisper"] = False
        backends["faster_whisper_gpu"] = False
    
    # Check openai-whisper
    try:
        from .openai_whisper_backend import is_available, check_gpu_available
        backends["openai_whisper"] = is_available()
        backends["openai_whisper_gpu"] = check_gpu_available()
    except ImportError:
        backends["openai_whisper"] = False
        backends["openai_whisper_gpu"] = False
    
    return backends


def get_transcriber(config: TranscriberConfig, *, strict: Optional[bool] = None) -> Transcriber:
    """Create a transcriber instance based on config.
    
    Backend selection logic:
    1. If backend="auto": try faster_whisper first, then whispercpp
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
    
    # Log available backends
    available = get_available_backends()
    logger.debug(f"Available transcription backends: {available}")
    logger.debug(f"Strict mode: {strict}")
    
    # Auto selection logic:
    # - If use_gpu=True: prefer openai_whisper (supports AMD + NVIDIA), then faster_whisper (NVIDIA only)
    # - If use_gpu=False: prefer faster_whisper (fastest on CPU), then whispercpp, then openai_whisper
    if backend == "auto":
        if config.use_gpu:
            # GPU path: openai_whisper supports AMD (ROCm) + NVIDIA (CUDA)
            if openai_whisper_cls:
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
                    "Install openai-whisper, faster-whisper, or pywhispercpp: "
                    "pip install openai-whisper"
                )
        else:
            # CPU path: faster_whisper is fastest
            if faster_whisper_cls:
                logger.info("Using faster-whisper backend (auto-selected for CPU)")
                backend = "faster_whisper"
            elif whispercpp_cls:
                logger.info("Using whisper.cpp backend (auto-selected, faster-whisper unavailable)")
                backend = "whispercpp"
            elif openai_whisper_cls:
                logger.info("Using openai-whisper backend (auto-selected, others unavailable)")
                backend = "openai_whisper"
            else:
                raise BackendNotAvailableError(
                    "No transcription backend available. "
                    "Install faster-whisper, pywhispercpp, or openai-whisper: "
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
                # Fall back to faster-whisper, then whispercpp
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
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
