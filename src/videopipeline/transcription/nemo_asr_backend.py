"""NVIDIA NeMo ASR backend.

This backend uses NVIDIA NeMo for speech-to-text and is aimed at CUDA systems.
It integrates as an optional backend and falls back gracefully when NeMo is not
installed.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

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

# Common Whisper-style values users may already have in profiles.
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

# Reasonable NeMo defaults for this pipeline.
_DEFAULT_EN_MODEL = "stt_en_conformer_ctc_small"
_DEFAULT_MULTI_MODEL = "stt_multilingual_fastconformer_hybrid_large_pc"


def is_available() -> bool:
    """Check if NeMo ASR dependencies are available."""
    _ensure_editdistance_fallback()
    try:
        import nemo.collections.asr as nemo_asr  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available for NeMo."""
    if not is_available():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_gpu_name() -> Optional[str]:
    """Get GPU name for logging, if available."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return None


def _ensure_editdistance_fallback() -> None:
    """Provide a pure-Python editdistance shim when wheels are unavailable.

    NeMo imports `editdistance` for WER metrics even in inference-only paths.
    On Python 3.13 on Windows, `editdistance` currently has no binary wheel.
    We install a small compatibility module backed by rapidfuzz.
    """
    if "editdistance" in sys.modules:
        return
    try:
        import editdistance  # noqa: F401

        return
    except Exception:
        pass

    try:
        from rapidfuzz.distance import Levenshtein
    except Exception:
        return

    shim = types.ModuleType("editdistance")

    def _eval(a: Any, b: Any) -> int:
        return int(Levenshtein.distance(str(a), str(b)))

    shim.eval = _eval  # type: ignore[attr-defined]
    shim.distance = _eval  # type: ignore[attr-defined]
    sys.modules["editdistance"] = shim


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _clean_word_text(s: str) -> str:
    return str(s or "").strip()


def _join_words(words: Sequence[TranscriptWord]) -> str:
    """Join tokenized words while avoiding spaces before punctuation."""
    text = ""
    for w in words:
        token = _clean_word_text(w.word)
        if not token:
            continue
        if not text:
            text = token
            continue
        if token[0] in ",.!?:;%)]}":
            text += token
        else:
            text += f" {token}"
    return text.strip()


def _looks_like_path_model(model_ref: str) -> bool:
    p = Path(model_ref)
    if p.exists():
        return True
    return model_ref.lower().endswith(".nemo")


def _extract_text(item: Any) -> str:
    """Extract transcript text from common NeMo output shapes."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for k in ("text", "pred_text", "transcript"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    txt = getattr(item, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    return ""


def _parse_word_item(entry: Any) -> Optional[TranscriptWord]:
    """Parse one word timestamp entry from dict/tuple/object formats."""
    word = ""
    start = None
    end = None
    prob = 1.0

    if isinstance(entry, dict):
        word = str(entry.get("word") or entry.get("token") or entry.get("text") or "").strip()
        start = _safe_float(
            entry.get("start")
            or entry.get("start_s")
            or entry.get("start_time")
            or entry.get("begin")
        )
        end = _safe_float(
            entry.get("end")
            or entry.get("end_s")
            or entry.get("end_time")
            or entry.get("stop")
        )
        p = _safe_float(entry.get("probability"))
        if p is not None:
            prob = p
    elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
        word = str(entry[0]).strip()
        start = _safe_float(entry[1])
        end = _safe_float(entry[2])
        if len(entry) >= 4:
            p = _safe_float(entry[3])
            if p is not None:
                prob = p
    else:
        word = str(getattr(entry, "word", "") or getattr(entry, "text", "")).strip()
        start = _safe_float(
            getattr(entry, "start", None)
            or getattr(entry, "start_s", None)
            or getattr(entry, "start_time", None)
        )
        end = _safe_float(
            getattr(entry, "end", None)
            or getattr(entry, "end_s", None)
            or getattr(entry, "end_time", None)
        )
        p = _safe_float(getattr(entry, "probability", None))
        if p is not None:
            prob = p

    if not word or start is None or end is None or end <= start:
        return None

    return TranscriptWord(word=word, start=float(start), end=float(end), probability=float(prob))


def _iter_word_candidates(item: Any) -> Iterable[Any]:
    if isinstance(item, dict):
        for key in ("words", "word_timestamps", "timestamps", "timestamp"):
            v = item.get(key)
            if isinstance(v, list):
                for x in v:
                    yield x
            elif isinstance(v, dict):
                for nested_key in ("word", "words"):
                    nv = v.get(nested_key)
                    if isinstance(nv, list):
                        for x in nv:
                            yield x
    else:
        for key in ("words", "word_timestamps", "timestamps", "timestamp"):
            v = getattr(item, key, None)
            if isinstance(v, list):
                for x in v:
                    yield x
            elif isinstance(v, dict):
                for nested_key in ("word", "words"):
                    nv = v.get(nested_key)
                    if isinstance(nv, list):
                        for x in nv:
                            yield x


def _extract_words(item: Any) -> List[TranscriptWord]:
    words: List[TranscriptWord] = []
    for raw in _iter_word_candidates(item):
        parsed = _parse_word_item(raw)
        if parsed is not None:
            words.append(parsed)
    words.sort(key=lambda w: (w.start, w.end))
    return words


def _parse_segment_item(entry: Any) -> Optional[TranscriptSegment]:
    text = ""
    start = None
    end = None

    if isinstance(entry, dict):
        text = str(entry.get("text") or entry.get("word") or "").strip()
        start = _safe_float(entry.get("start") or entry.get("start_s") or entry.get("start_time"))
        end = _safe_float(entry.get("end") or entry.get("end_s") or entry.get("end_time"))
    else:
        text = str(getattr(entry, "text", "") or "").strip()
        start = _safe_float(
            getattr(entry, "start", None)
            or getattr(entry, "start_s", None)
            or getattr(entry, "start_time", None)
        )
        end = _safe_float(
            getattr(entry, "end", None)
            or getattr(entry, "end_s", None)
            or getattr(entry, "end_time", None)
        )

    if not text or start is None or end is None or end <= start:
        return None
    return TranscriptSegment(start=float(start), end=float(end), text=text)


def _extract_explicit_segments(item: Any) -> List[TranscriptSegment]:
    """Extract explicit segment arrays if the model provides them."""
    candidates: list[Any] = []
    if isinstance(item, dict):
        for key in ("segments", "chunks", "segment_timestamps"):
            v = item.get(key)
            if isinstance(v, list):
                candidates.extend(v)
    else:
        for key in ("segments", "chunks", "segment_timestamps"):
            v = getattr(item, key, None)
            if isinstance(v, list):
                candidates.extend(v)

    out: List[TranscriptSegment] = []
    for c in candidates:
        parsed = _parse_segment_item(c)
        if parsed is not None:
            out.append(parsed)
    out.sort(key=lambda s: (s.start, s.end))
    return out


def _segments_from_words(words: Sequence[TranscriptWord]) -> List[TranscriptSegment]:
    """Build segment-level view from timestamped words."""
    if not words:
        return []

    out: List[TranscriptSegment] = []
    cur: List[TranscriptWord] = [words[0]]

    for w in words[1:]:
        prev = cur[-1]
        should_break = (w.start - prev.end) > 0.85 or prev.word.strip().endswith((".", "!", "?"))
        if should_break:
            text = _join_words(cur)
            if text:
                out.append(
                    TranscriptSegment(
                        start=float(cur[0].start),
                        end=float(cur[-1].end),
                        text=text,
                        words=list(cur),
                    )
                )
            cur = [w]
        else:
            cur.append(w)

    if cur:
        text = _join_words(cur)
        if text:
            out.append(
                TranscriptSegment(
                    start=float(cur[0].start),
                    end=float(cur[-1].end),
                    text=text,
                    words=list(cur),
                )
            )
    return out


class NemoASRTranscriber(BaseTranscriber):
    """Transcriber using NVIDIA NeMo ASR."""

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self._gpu_used = False
        self._device = "cpu"
        self._model_ref = ""

        if not is_available():
            raise BackendNotAvailableError(
                "NeMo ASR is not installed. Install with: pip install nemo_toolkit[asr]"
            )

    @property
    def backend_name(self) -> str:
        return "nemo_asr"

    @property
    def gpu_available(self) -> bool:
        return check_gpu_available()

    def _resolve_model_ref(self) -> str:
        """Resolve configured model to a usable NeMo model reference."""
        if self.config.model_path:
            return str(self.config.model_path)

        raw = str(self.config.model or "").strip()
        if not raw:
            raw = "small.en"

        # If caller already passed a NeMo style model name/path, keep as-is.
        if "/" in raw or raw.startswith("stt_") or _looks_like_path_model(raw):
            return raw

        # Map common whisper profile values to a practical NeMo default.
        if raw.lower() in _WHISPER_MODEL_ALIASES:
            lang = (self.config.language or "").lower()
            if not lang or lang.startswith("en"):
                return _DEFAULT_EN_MODEL
            return _DEFAULT_MULTI_MODEL

        return raw

    def _get_audio_duration(self, audio_path: Path) -> float:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                **subprocess_flags(),
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass

        try:
            size_bytes = audio_path.stat().st_size
            return max(size_bytes / 16000, 60.0)
        except Exception:
            return 300.0

    def _load_model(self) -> Any:
        import nemo.collections.asr as nemo_asr
        import torch

        self._model_ref = self._resolve_model_ref()
        model_ref = self._model_ref

        if self.config.use_gpu and torch.cuda.is_available():
            self._device = "cuda"
            self._gpu_used = True
        else:
            self._device = "cpu"
            self._gpu_used = False

        logger.info("Loading NeMo ASR model: %s on %s", model_ref, self._device)
        if self._gpu_used:
            gpu_name = get_gpu_name()
            if gpu_name:
                logger.info("NeMo GPU: %s", gpu_name)

        if _looks_like_path_model(model_ref):
            model = nemo_asr.models.ASRModel.restore_from(restore_path=model_ref)
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)

        try:
            model = model.to(self._device)
        except Exception as exc:
            if self._device == "cuda":
                logger.warning("Failed to move NeMo model to CUDA (%s). Falling back to CPU.", exc)
                self._device = "cpu"
                self._gpu_used = False
                model = model.to("cpu")

        try:
            model.eval()
        except Exception:
            pass

        return model

    @staticmethod
    def _is_windows_file_lock_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "winerror 32" in msg or "being used by another process" in msg

    def _run_transcribe(self, model: Any, audio_path: Path, *, with_hyp: bool) -> Any:
        """Call NeMo transcribe with compatibility fallbacks across versions."""
        import inspect

        path_str = str(audio_path)

        transcribe_fn = getattr(model, "transcribe")
        try:
            sig = inspect.signature(transcribe_fn)
            params = set(sig.parameters.keys())
        except Exception:
            params = set()

        kwargs: Dict[str, Any] = {}
        if "audio" in params:
            kwargs["audio"] = [path_str]
        elif "paths2audio_files" in params:
            kwargs["paths2audio_files"] = [path_str]
        else:
            # Unknown signature; try positional fallback.
            return transcribe_fn([path_str])  # type: ignore[misc]

        if "batch_size" in params:
            kwargs["batch_size"] = 1
        if "num_workers" in params:
            kwargs["num_workers"] = 0
        if "return_hypotheses" in params:
            kwargs["return_hypotheses"] = with_hyp
        if "timestamps" in params and with_hyp:
            kwargs["timestamps"] = True
        if "channel_selector" in params:
            # Robust on stereo/5.1 files; training models generally expect mono.
            kwargs["channel_selector"] = 0

        try:
            return transcribe_fn(**kwargs)
        except TypeError:
            # Last resort for versions with incompatible keyword naming.
            return transcribe_fn([path_str])  # type: ignore[misc]

    def _extract_primary_item(self, raw: Any) -> Any:
        if raw is None:
            return ""
        if isinstance(raw, tuple):
            if len(raw) > 0:
                return raw[0]
            return ""
        return raw

    def _extract_language(self, item: Any) -> Optional[str]:
        if isinstance(item, dict):
            for k in ("language", "lang", "language_code"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        for k in ("language", "lang", "language_code"):
            v = getattr(item, k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return self.config.language

    def _normalize_output(self, raw: Any) -> Any:
        item = self._extract_primary_item(raw)
        if isinstance(item, (list, tuple)):
            if len(item) == 0:
                return ""
            return item[0]
        return item

    def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> TranscriptResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if on_progress:
            on_progress(0.05)

        duration_s = self._get_audio_duration(audio_path)
        model = self.ensure_model_loaded()

        if on_progress:
            on_progress(0.15)

        request_word_timestamps = bool(self.config.word_timestamps)
        raw = None
        last_exc: Optional[Exception] = None

        attempts: List[bool] = [request_word_timestamps]
        if request_word_timestamps:
            # NeMo on Windows can occasionally fail while cleaning up temporary
            # manifest files for timestamped transcribe paths. Retry once
            # without timestamp hypotheses so transcription can still complete.
            attempts.append(False)

        for idx, with_hyp in enumerate(attempts):
            try:
                raw = self._run_transcribe(model, audio_path, with_hyp=with_hyp)
                break
            except Exception as exc:
                last_exc = exc
                should_retry = (
                    idx < len(attempts) - 1
                    and self._is_windows_file_lock_error(exc)
                )
                if should_retry:
                    logger.warning(
                        "NeMo timestamped transcription failed with a Windows file lock (%s); "
                        "retrying without word timestamps.",
                        exc,
                    )
                    time.sleep(0.25)
                    continue
                raise

        if raw is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("NeMo transcribe returned no result.")

        item = self._normalize_output(raw)

        explicit_segments = _extract_explicit_segments(item)
        words = _extract_words(item)

        if explicit_segments:
            segments = explicit_segments
        elif words:
            segments = _segments_from_words(words)
        else:
            text = _extract_text(item)
            if not text:
                text = _extract_text(raw)
            if not text and isinstance(item, str):
                text = item.strip()
            if not text:
                raise RuntimeError("NeMo returned no transcript text.")
            seg_end = float(duration_s if duration_s > 0 else 0.0)
            segments = [TranscriptSegment(start=0.0, end=seg_end, text=text)]

        # Attach word-level timestamps only when we could parse them and no explicit words already set.
        if words and segments and not any(s.words for s in segments):
            if len(segments) == 1:
                segments[0].words = words

        # Ensure monotonic ordering and non-empty text.
        cleaned: List[TranscriptSegment] = []
        for s in sorted(segments, key=lambda x: (x.start, x.end)):
            text = str(s.text or "").strip()
            if not text:
                continue
            start = float(s.start)
            end = float(s.end)
            if end < start:
                continue
            cleaned.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    words=s.words,
                )
            )

        if not cleaned:
            raise RuntimeError("NeMo produced no usable transcript segments.")

        if on_progress:
            on_progress(1.0)

        return TranscriptResult(
            segments=cleaned,
            language=self._extract_language(item),
            duration_seconds=max((s.end for s in cleaned), default=duration_s),
            backend_used=self.backend_name,
            gpu_used=self._gpu_used,
        )
