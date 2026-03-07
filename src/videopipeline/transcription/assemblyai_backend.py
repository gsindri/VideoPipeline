"""AssemblyAI cloud transcription backend.

This backend sends audio to AssemblyAI's API and retrieves transcription results.
It is optional and only active when explicitly selected.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from .base import (
    BackendNotAvailableError,
    BaseTranscriber,
    TranscriberConfig,
    TranscriberError,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)

logger = logging.getLogger(__name__)

_DEFAULT_SPEECH_MODELS = ("universal-3-pro", "universal-2")
_WHISPER_MODEL_ALIASES = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
}


def _resolve_api_key(explicit_key: Optional[str]) -> Optional[str]:
    key = (explicit_key or "").strip()
    if key:
        return key
    for env_name in ("ASSEMBLYAI_API_KEY", "AAI_API_KEY"):
        val = (os.getenv(env_name) or "").strip()
        if val:
            return val
    return None


def is_available() -> bool:
    """Return True when AssemblyAI SDK is importable and API key is configured."""
    try:
        import assemblyai as _aai  # noqa: F401
    except Exception:
        return False
    return bool(_resolve_api_key(None))


def check_gpu_available() -> bool:
    """Cloud backend does not use local GPU."""
    return False


def _ms_to_s(v: Any) -> float:
    try:
        return max(0.0, float(v) / 1000.0)
    except Exception:
        return 0.0


def _safe_float(v: Any, default: float = 1.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


class AssemblyAITranscriber(BaseTranscriber):
    """Transcriber using AssemblyAI's managed API."""

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)

        try:
            import assemblyai as _aai  # noqa: F401
        except Exception as exc:
            raise BackendNotAvailableError(
                "AssemblyAI backend requested but assemblyai SDK is not installed. "
                "Install with: pip install assemblyai"
            ) from exc

        self._api_key = _resolve_api_key(config.assemblyai_api_key)
        if not self._api_key:
            raise BackendNotAvailableError(
                "AssemblyAI backend requested but no API key is configured. "
                "Set ASSEMBLYAI_API_KEY."
            )

    @property
    def backend_name(self) -> str:
        return "assemblyai"

    @property
    def gpu_available(self) -> bool:
        return False

    def _load_model(self) -> Any:
        import assemblyai as aai

        aai.settings.api_key = self._api_key
        return aai.Transcriber()

    def _resolve_speech_models(self) -> List[str]:
        if self.config.assemblyai_speech_models:
            models = [str(m).strip() for m in self.config.assemblyai_speech_models if str(m).strip()]
            if models:
                return models

        raw_model = str(self.config.model or "").strip()
        if raw_model:
            model_tokens = [p.strip() for p in raw_model.split(",") if p.strip()]
            if model_tokens and not all(m.lower() in _WHISPER_MODEL_ALIASES for m in model_tokens):
                return model_tokens

        return list(_DEFAULT_SPEECH_MODELS)

    def _build_sdk_config(self, aai_mod: Any) -> Any:
        kwargs: dict[str, Any] = {
            "speech_models": self._resolve_speech_models(),
            "speaker_labels": bool(self.config.diarize),
        }

        if self.config.language:
            kwargs["language_code"] = self.config.language
        else:
            kwargs["language_detection"] = True

        if self.config.diarize:
            min_sp = self.config.diarize_min_speakers
            max_sp = self.config.diarize_max_speakers
            if min_sp is not None and max_sp is not None and int(min_sp) == int(max_sp):
                kwargs["speakers_expected"] = int(min_sp)
            elif min_sp is not None or max_sp is not None:
                kwargs["speaker_options"] = aai_mod.SpeakerOptions(
                    min_speakers_expected=int(min_sp) if min_sp is not None else None,
                    max_speakers_expected=int(max_sp) if max_sp is not None else None,
                )

        return aai_mod.TranscriptionConfig(**kwargs)

    def _poll_until_complete(
        self,
        *,
        aai_mod: Any,
        transcript: Any,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> Any:
        poll_interval = max(0.5, float(self.config.assemblyai_poll_interval_s))
        timeout_s = max(poll_interval, float(self.config.assemblyai_timeout_s))
        started = time.time()
        progress = 0.12

        while transcript.status in (aai_mod.TranscriptStatus.queued, aai_mod.TranscriptStatus.processing):
            if time.time() - started > timeout_s:
                raise TimeoutError(
                    f"AssemblyAI transcription timed out after {timeout_s:.1f}s (id={transcript.id})"
                )

            time.sleep(poll_interval)
            transcript = aai_mod.Transcript.get_by_id(transcript.id)

            if on_progress:
                progress = min(0.92, progress + 0.03)
                on_progress(progress)

        return transcript

    @staticmethod
    def _collect_speakers(segments: Sequence[TranscriptSegment]) -> Optional[List[str]]:
        seen: List[str] = []
        for seg in segments:
            sp = (seg.speaker or "").strip()
            if sp and sp not in seen:
                seen.append(sp)
        return seen or None

    def _convert_utterances(self, utterances: Sequence[Any]) -> List[TranscriptSegment]:
        out: List[TranscriptSegment] = []
        for utt in utterances:
            text = str(getattr(utt, "text", "") or "").strip()
            if not text:
                continue

            start = _ms_to_s(getattr(utt, "start", 0))
            end = _ms_to_s(getattr(utt, "end", 0))
            speaker = getattr(utt, "speaker", None)

            words: Optional[List[TranscriptWord]] = None
            raw_words = list(getattr(utt, "words", []) or [])
            if self.config.word_timestamps and raw_words:
                words = []
                for w in raw_words:
                    w_text = str(getattr(w, "text", "") or "").strip()
                    if not w_text:
                        continue

                    w_start = _ms_to_s(getattr(w, "start", 0))
                    w_end = _ms_to_s(getattr(w, "end", 0))
                    if w_end < w_start:
                        continue

                    words.append(
                        TranscriptWord(
                            word=w_text,
                            start=w_start,
                            end=w_end,
                            probability=_safe_float(getattr(w, "confidence", 1.0), 1.0),
                            speaker=getattr(w, "speaker", speaker),
                        )
                    )

            out.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    words=words,
                    speaker=speaker,
                )
            )

        return out

    def _convert_words_fallback(self, transcript: Any) -> List[TranscriptSegment]:
        text = str(getattr(transcript, "text", "") or "").strip()
        if not text:
            return []

        words: Optional[List[TranscriptWord]] = None
        raw_words = list(getattr(transcript, "words", []) or [])
        if self.config.word_timestamps and raw_words:
            words = []
            for w in raw_words:
                w_text = str(getattr(w, "text", "") or "").strip()
                if not w_text:
                    continue
                w_start = _ms_to_s(getattr(w, "start", 0))
                w_end = _ms_to_s(getattr(w, "end", 0))
                if w_end < w_start:
                    continue
                words.append(
                    TranscriptWord(
                        word=w_text,
                        start=w_start,
                        end=w_end,
                        probability=_safe_float(getattr(w, "confidence", 1.0), 1.0),
                        speaker=getattr(w, "speaker", None),
                    )
                )

        end = _ms_to_s(getattr(transcript, "audio_duration", 0))
        if words:
            end = max(end, max((w.end for w in words), default=end))

        return [TranscriptSegment(start=0.0, end=end, text=text, words=words)]

    def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> TranscriptResult:
        import assemblyai as aai

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if on_progress:
            on_progress(0.05)

        transcriber = self.ensure_model_loaded()
        sdk_cfg = self._build_sdk_config(aai)

        logger.info(
            "Submitting AssemblyAI transcript (models=%s, diarize=%s)",
            self._resolve_speech_models(),
            bool(self.config.diarize),
        )
        transcript = transcriber.submit(str(audio_path), config=sdk_cfg)

        if on_progress:
            on_progress(0.12)

        transcript = self._poll_until_complete(aai_mod=aai, transcript=transcript, on_progress=on_progress)

        if transcript.status == aai.TranscriptStatus.error:
            raise TranscriberError(f"AssemblyAI transcription failed: {getattr(transcript, 'error', 'unknown error')}")

        logger.info(
            "AssemblyAI transcript completed (id=%s, speech_model_used=%s)",
            getattr(transcript, "id", ""),
            getattr(transcript, "speech_model_used", ""),
        )

        utterances = list(getattr(transcript, "utterances", []) or [])
        if utterances:
            segments = self._convert_utterances(utterances)
        else:
            segments = self._convert_words_fallback(transcript)

        duration_s = _ms_to_s(getattr(transcript, "audio_duration", 0))
        duration_s = max(duration_s, max((s.end for s in segments), default=duration_s))
        speakers = self._collect_speakers(segments)
        diarization_used = bool(self.config.diarize and speakers)

        if on_progress:
            on_progress(1.0)

        return TranscriptResult(
            segments=segments,
            language=getattr(transcript, "language_code", None) or self.config.language,
            duration_seconds=duration_s,
            backend_used=self.backend_name,
            gpu_used=False,
            speakers=speakers,
            diarization_used=diarization_used,
        )
