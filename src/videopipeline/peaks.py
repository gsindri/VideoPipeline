from __future__ import annotations

from typing import List

import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def robust_z(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score using median and MAD.
    Falls back to std if MAD is tiny.
    """
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
) -> List[int]:
    """
    Greedy peak selection by score with minimum spacing.
    Returns indices in descending score order.
    """
    if top_k <= 0:
        return []

    order = np.argsort(scores)[::-1]
    chosen: List[int] = []

    for idx in order:
        idx = int(idx)
        if scores[idx] <= 0:
            break
        if not is_local_max(scores, idx):
            continue
        if any(abs(idx - c) < min_gap_frames for c in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= top_k:
            break

    return chosen
