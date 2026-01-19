from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def default_profile() -> Dict[str, Any]:
    return {
        "analysis": {
            "audio": {
                "sample_rate": 16000,
                "hop_seconds": 0.5,
                "smooth_seconds": 3.0,
                "top": 12,
                "min_gap_seconds": 20.0,
                "pre_seconds": 8.0,
                "post_seconds": 22.0,
                "skip_start_seconds": 10.0,
            },
            "motion": {
                "sample_fps": 3.0,
                "scale_width": 160,
                "smooth_seconds": 2.5,
            },
            "scenes": {
                "enabled": True,
                "threshold_z": 3.5,
                "min_scene_len_seconds": 1.2,
                "snap_window_seconds": 1.0,
            },
            "highlights": {
                "top": 20,
                "min_gap_seconds": 15.0,
                "skip_start_seconds": 10.0,
                "weights": {
                    "audio": 0.55,
                    "motion": 0.45,
                    "chat": 0.0,
                },
                "clip": {
                    "min_seconds": 12,
                    "max_seconds": 60,
                    "min_pre_seconds": 2,
                    "max_pre_seconds": 12,
                    "min_post_seconds": 4,
                    "max_post_seconds": 28,
                },
            },
        },
        "export": {
            "template": "vertical_blur",
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "crf": 20,
            "preset": "veryfast",
            "normalize_audio": False,
        },
        "captions": {
            "enabled": False,
            "model_size": "small",
            "language": None,
            "device": "cpu",
            "compute_type": "int8",
        },
    }


def load_profile(profile_path: Optional[Path]) -> Dict[str, Any]:
    if profile_path is None:
        return default_profile()

    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Profile YAML must be a mapping")
    return data
