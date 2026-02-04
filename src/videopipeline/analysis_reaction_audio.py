"""Reaction audio feature extraction.

Computes acoustic features that correlate with emotional reactions:
- RMS loudness
- Zero-crossing rate (ZCR) - higher for laughter/high-pitch bursts
- Spectral centroid - higher when voice gets sharper/brighter
- Spectral flux - higher when audio changes rapidly (bursts)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .ffmpeg import AudioStreamParams, ffprobe_duration_seconds, stream_audio_blocks_f32le
from .peaks import moving_average, robust_z
from .project import Project, save_npz, update_project


@dataclass
class ReactionAudioConfig:
    """Configuration for reaction audio feature extraction."""
    sample_rate: int = 16000
    hop_seconds: float = 0.5
    smooth_seconds: float = 1.5  # Light smoothing for reaction signals

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReactionAudioConfig":
        d = d or {}
        allowed = {"enabled", "sample_rate", "hop_seconds", "smooth_seconds"}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"Unknown reaction_audio config keys: {sorted(unknown)}")

        cfg = cls(
            sample_rate=int(d.get("sample_rate", 16000)),
            hop_seconds=float(d.get("hop_seconds", 0.5)),
            smooth_seconds=float(d.get("smooth_seconds", 1.5)),
        )

        if cfg.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0 (got {cfg.sample_rate})")
        if cfg.hop_seconds <= 0:
            raise ValueError(f"hop_seconds must be > 0 (got {cfg.hop_seconds})")
        if cfg.smooth_seconds <= 0:
            raise ValueError(f"smooth_seconds must be > 0 (got {cfg.smooth_seconds})")

        return cfg


def _compute_rms(samples: np.ndarray) -> float:
    """Compute RMS of audio samples."""
    samples64 = samples.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(samples64 * samples64) + 1e-12))


def _compute_zcr(samples: np.ndarray) -> float:
    """Compute zero-crossing rate.
    
    Higher ZCR often indicates:
    - High-pitched voices
    - Laughter
    - Screams/yells
    """
    if len(samples) < 2:
        return 0.0
    
    signs = np.sign(samples)
    # Count sign changes
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (len(samples) - 1))


def _compute_spectral_centroid(samples: np.ndarray, sample_rate: int) -> float:
    """Compute spectral centroid (brightness/sharpness of sound).
    
    Higher centroid indicates:
    - Brighter, sharper sounds
    - Higher pitched vocalizations
    - Excited speech
    """
    if len(samples) < 64:
        return 0.0
    
    # Use FFT
    fft_size = min(2048, len(samples))
    windowed = samples[:fft_size] * np.hanning(fft_size)
    spectrum = np.abs(np.fft.rfft(windowed))
    
    if np.sum(spectrum) < 1e-12:
        return 0.0
    
    freqs = np.fft.rfftfreq(fft_size, d=1.0/sample_rate)
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)
    
    return float(centroid)


def _compute_spectral_flux(prev_spectrum: np.ndarray, curr_spectrum: np.ndarray) -> float:
    """Compute spectral flux (rate of spectral change).
    
    Higher flux indicates:
    - Rapid changes in audio
    - Bursts of sound
    - Sudden vocalizations
    """
    if len(prev_spectrum) == 0 or len(curr_spectrum) == 0:
        return 0.0
    
    # Normalize spectra
    prev_norm = prev_spectrum / (np.sum(prev_spectrum) + 1e-12)
    curr_norm = curr_spectrum / (np.sum(curr_spectrum) + 1e-12)
    
    # Half-wave rectified difference (only increases)
    diff = curr_norm - prev_norm
    flux = float(np.sum(np.maximum(diff, 0)))
    
    return flux


def _get_spectrum(samples: np.ndarray) -> np.ndarray:
    """Get magnitude spectrum of samples."""
    if len(samples) < 64:
        return np.array([])
    
    fft_size = min(2048, len(samples))
    windowed = samples[:fft_size] * np.hanning(fft_size)
    return np.abs(np.fft.rfft(windowed))


def compute_reaction_audio_features(
    proj: Project,
    *,
    cfg: ReactionAudioConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute reaction-focused audio features.

    Generates timelines:
      - rms: RMS loudness
      - zcr: Zero-crossing rate
      - spectral_centroid: Spectral brightness
      - spectral_flux: Rate of spectral change
      - reaction_score: Normalized composite reaction score

    Persists:
      - analysis/reaction_audio_features.npz
      - project.json -> analysis.reaction_audio section
    """
    video_path = Path(proj.audio_source)  # Use audio_source for fallback during early analysis
    duration_s = ffprobe_duration_seconds(video_path)

    hop_samples = int(cfg.sample_rate * cfg.hop_seconds)
    if hop_samples <= 0:
        raise ValueError("hop_seconds too small")

    params = AudioStreamParams(sample_rate=cfg.sample_rate, channels=1)

    total_frames = max(1, int(duration_s / cfg.hop_seconds))

    # Feature arrays
    rms_values = []
    zcr_values = []
    centroid_values = []
    flux_values = []

    prev_spectrum = np.array([])
    processed = 0
    
    # Helper for progress reporting with optional message
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)

    _report(0.05, "Extracting audio blocks")

    for block in stream_audio_blocks_f32le(video_path, params=params, block_samples=hop_samples):
        # RMS
        rms = _compute_rms(block)
        rms_values.append(rms)

        # ZCR
        zcr = _compute_zcr(block)
        zcr_values.append(zcr)

        # Spectral centroid
        centroid = _compute_spectral_centroid(block, cfg.sample_rate)
        centroid_values.append(centroid)

        # Spectral flux
        curr_spectrum = _get_spectrum(block)
        flux = _compute_spectral_flux(prev_spectrum, curr_spectrum)
        flux_values.append(flux)
        prev_spectrum = curr_spectrum

        processed += 1
        if processed % 20 == 0:
            pct = 100 * processed / total_frames
            _report(min(0.85, 0.05 + 0.8 * (processed / total_frames)), f"Analyzing audio: {processed}/{total_frames} ({pct:.0f}%)")

    _report(0.85, "Applying smoothing and normalization")

    # Convert to arrays
    rms = np.array(rms_values, dtype=np.float64)
    zcr = np.array(zcr_values, dtype=np.float64)
    spectral_centroid = np.array(centroid_values, dtype=np.float64)
    spectral_flux = np.array(flux_values, dtype=np.float64)
    
    # Create explicit times array for consistent resampling
    times = np.arange(len(rms)) * cfg.hop_seconds

    # Apply smoothing
    smooth_frames = max(1, int(round(cfg.smooth_seconds / cfg.hop_seconds)))
    rms_smooth = moving_average(rms, smooth_frames)
    zcr_smooth = moving_average(zcr, smooth_frames)
    centroid_smooth = moving_average(spectral_centroid, smooth_frames)
    flux_smooth = moving_average(spectral_flux, smooth_frames)

    # Normalize with robust z-score
    rms_z = robust_z(rms_smooth)
    zcr_z = robust_z(zcr_smooth)
    centroid_z = robust_z(centroid_smooth)
    flux_z = robust_z(flux_smooth)

    _report(0.9, "Computing composite reaction score")

    # Composite reaction score
    # Weight towards features that indicate vocal reactions
    reaction_score = (
        0.35 * np.clip(rms_z, 0, None)  # Loudness
        + 0.25 * np.clip(zcr_z, 0, None)  # High-frequency content
        + 0.20 * np.clip(centroid_z, 0, None)  # Brightness
        + 0.20 * np.clip(flux_z, 0, None)  # Rapid changes
    )

    # Re-normalize the composite
    reaction_score = robust_z(reaction_score) if np.any(reaction_score != 0) else reaction_score

    # Save features
    reaction_features_path = proj.analysis_dir / "reaction_audio_features.npz"
    save_npz(
        reaction_features_path,
        rms=rms,
        zcr=zcr,
        spectral_centroid=spectral_centroid,
        spectral_flux=spectral_flux,
        rms_smooth=rms_smooth,
        zcr_smooth=zcr_smooth,
        centroid_smooth=centroid_smooth,
        flux_smooth=flux_smooth,
        rms_z=rms_z,
        zcr_z=zcr_z,
        centroid_z=centroid_z,
        flux_z=flux_z,
        reaction_score=reaction_score,
        times=times,
        hop_seconds=np.array([cfg.hop_seconds], dtype=np.float64),
    )

    # Update project
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "sample_rate": cfg.sample_rate,
            "hop_seconds": cfg.hop_seconds,
            "smooth_seconds": cfg.smooth_seconds,
        },
        "duration_seconds": duration_s,
        "hop_count": len(rms),
        "features_npz": str(reaction_features_path.relative_to(proj.project_dir)),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["reaction_audio"] = payload

    update_project(proj, _upd)

    _report(1.0, "Done")

    return payload


def load_reaction_audio_features(proj: Project) -> Optional[Dict[str, np.ndarray]]:
    """Load reaction audio features if available."""
    reaction_features_path = proj.analysis_dir / "reaction_audio_features.npz"
    if not reaction_features_path.exists():
        return None

    data = np.load(reaction_features_path)
    return {k: data[k] for k in data.files}
