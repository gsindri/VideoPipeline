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
    eps: float = 1e-12


def audio_rms_db_timeline(video_path: Path, cfg: AudioFeatureConfig) -> List[float]:
    """
    Compute a loudness timeline (dB) at fixed hop size.

    - One value per `hop_seconds`
    - Constant memory, works for multi-hour videos
    """
    hop_samples = int(cfg.sample_rate * cfg.hop_seconds)
    if hop_samples <= 0:
        raise ValueError("hop_seconds too small; resulted in non-positive hop_samples")

    params = AudioStreamParams(sample_rate=cfg.sample_rate, channels=1)
    timeline_db: List[float] = []

    for block in stream_audio_blocks_f32le(video_path, params=params, block_samples=hop_samples):
        block64 = block.astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(block64 * block64)))
        db = float(20.0 * np.log10(rms + cfg.eps))
        timeline_db.append(db)

    return timeline_db
