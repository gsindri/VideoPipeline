from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .ffmpeg import AudioStreamParams, stream_audio_blocks_f32le


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16000
    hop_seconds: float = 0.5
    eps: float = 1e-12  # avoids log(0)


def audio_rms_db_timeline(video_path: Path, cfg: AudioFeatureConfig) -> List[float]:
    """
    Returns a loudness timeline in dB (one value per hop_seconds).
    Uses constant memory; suitable for multi-hour videos.
    """
    hop_samples = int(cfg.sample_rate * cfg.hop_seconds)
    if hop_samples <= 0:
        raise ValueError("hop_seconds too small; resulted in non-positive hop_samples.")

    params = AudioStreamParams(sample_rate=cfg.sample_rate, channels=1)
    timeline_db: List[float] = []

    for block in stream_audio_blocks_f32le(video_path, params=params, block_samples=hop_samples):
        # RMS on float64 for numerical stability
        block64 = block.astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(block64 * block64)))
        db = float(20.0 * np.log10(rms + cfg.eps))
        timeline_db.append(db)

    return timeline_db
