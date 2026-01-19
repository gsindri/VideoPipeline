from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .ffmpeg import _require_cmd
from .subtitles import SubtitleSegment


class WhisperNotInstalledError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscribeConfig:
    model_size: str = "small"
    language: Optional[str] = None
    device: str = "cpu"  # "cpu" or "cuda"
    compute_type: str = "int8"  # "int8" on CPU is fast; "float16" on GPU
    sample_rate: int = 16000


def _load_whisper_model(cfg: TranscribeConfig):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise WhisperNotInstalledError(
            "faster-whisper is not installed. Install optional deps with: pip install -e '.[whisper]'"
        ) from e

    return WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)


def _extract_audio_wav(video_path: Path, start_s: float, end_s: float, sr: int, wav_path: Path) -> None:
    _require_cmd("ffmpeg")
    duration = max(0.01, float(end_s - start_s))
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(wav_path),
    ]
    import subprocess

    subprocess.check_call(cmd)


def transcribe_segment(
    video_path: Path,
    *,
    start_s: float,
    end_s: float,
    cfg: TranscribeConfig,
) -> List[SubtitleSegment]:
    """Transcribe a video segment into subtitle segments (times relative to segment start)."""
    video_path = Path(video_path)

    model = _load_whisper_model(cfg)

    with tempfile.TemporaryDirectory(prefix="vp_whisper_") as td:
        td_path = Path(td)
        wav_path = td_path / "audio.wav"
        _extract_audio_wav(video_path, start_s, end_s, cfg.sample_rate, wav_path)

        segments, info = model.transcribe(
            str(wav_path),
            language=cfg.language,
            vad_filter=True,
            word_timestamps=False,
        )

        out: List[SubtitleSegment] = []
        for seg in segments:
            # seg.start/end are relative to extracted audio (0..duration)
            text = (seg.text or "").strip()
            if not text:
                continue
            out.append(SubtitleSegment(start_s=float(seg.start), end_s=float(seg.end), text=text))

    return out


def save_transcript_json(path: Path, segments: List[SubtitleSegment], cfg: TranscribeConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "config": {
            "model_size": cfg.model_size,
            "language": cfg.language,
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "sample_rate": cfg.sample_rate,
        },
        "segments": [
            {"start_s": s.start_s, "end_s": s.end_s, "text": s.text} for s in segments
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_transcript_json(path: Path) -> List[SubtitleSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segs = []
    for s in payload.get("segments", []):
        segs.append(SubtitleSegment(start_s=float(s["start_s"]), end_s=float(s["end_s"]), text=str(s["text"])))
    return segs
