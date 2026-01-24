from __future__ import annotations

from typing import List

import numpy as np


def moving_average(x: np.ndarray, window: int, *, use_cumsum: bool = False) -> np.ndarray:
    """Moving average with optional cumsum optimization.
    
    Args:
        x: Input array.
        window: Window size for averaging.
        use_cumsum: If True, use cumsum-based O(n) algorithm (faster for large windows).
                    If False, use np.convolve (better for small windows).
    
    Returns:
        Smoothed array with same length as input.
    """
    if window <= 1:
        return x
    window = int(window)
    
    if use_cumsum:
        # Centered moving average using an O(n) cumsum approach.
        #
        # Note: A common pitfall is implementing a *trailing* average, which shifts
        # the signal relative to the convolution-based "same" output. This
        # implementation matches the centered behavior of
        # np.convolve(..., mode="same") (up to small floating point differences).

        # Pad on both sides so the output stays aligned/centered.
        # Use zero-padding to match np.convolve(..., mode="same").
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        xpad = np.pad(x, (pad_left, pad_right), mode="constant", constant_values=0.0)

        # Prefix with a 0 so we can use the standard cumsum window trick.
        csum = np.cumsum(xpad, dtype=np.float64)
        csum = np.concatenate(([0.0], csum))

        out = (csum[window:] - csum[:-window]) / float(window)
        # out length is len(x)
        return out
    else:
        # np.convolve approach - good for small windows
        kernel = np.ones(window, dtype=np.float64) / float(window)
        return np.convolve(x, kernel, mode="same")


def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median and MAD."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-9:
        std = np.std(x)
        scale = std if std > 1e-9 else 1.0
    else:
        scale = 1.4826 * mad
    return (x - med) / scale


def is_local_max(x: np.ndarray, i: int) -> bool:
    left = x[i - 1] if i - 1 >= 0 else -np.inf
    right = x[i + 1] if i + 1 < len(x) else -np.inf
    return x[i] >= left and x[i] >= right


def pick_top_peaks(
    scores: np.ndarray,
    *,
    top_k: int,
    min_gap_frames: int,
    min_score: float = 0.0,
) -> List[int]:
    """Greedy peak selection by score with minimum spacing.
    
    Args:
        scores: Array of scores to find peaks in.
        top_k: Maximum number of peaks to return.
        min_gap_frames: Minimum spacing between peaks in frames.
        min_score: Minimum score threshold. Peaks below this are ignored.
    
    Returns:
        List of peak indices, sorted by score descending.
    """
    if top_k <= 0:
        return []

    order = np.argsort(scores)[::-1]
    chosen: List[int] = []

    for idx in order:
        idx = int(idx)
        if not np.isfinite(scores[idx]):
            continue
        if scores[idx] <= min_score:
            break
        if not is_local_max(scores, idx):
            continue
        if any(abs(idx - c) < min_gap_frames for c in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= top_k:
            break

    return chosen
