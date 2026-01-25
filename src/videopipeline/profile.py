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
                    "audio": 0.45,
                    "motion": 0.35,
                    "chat": 0.0,
                    "audio_events": 0.20,
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
            "speech": {
                "enabled": True,
                "backend": "openai_whisper",  # "openai_whisper", "faster_whisper", "whispercpp", or "auto"
                "model_size": "small",  # tiny, base, small, medium, large
                "language": None,  # None for auto-detect
                "device": "cuda",  # cpu or cuda
                "compute_type": "float16",  # int8 (CPU), float16 (GPU)
                "use_gpu": True,  # Request GPU if available (AMD ROCm or NVIDIA CUDA)
                "threads": 0,  # CPU threads (0 = auto, whisper.cpp only)
                "vad_filter": True,
                "word_timestamps": True,
                "hop_seconds": 0.5,
            },
            "reaction_audio": {
                "enabled": True,
                "sample_rate": 16000,
                "hop_seconds": 0.5,
                "smooth_seconds": 1.5,
            },
            "audio_events": {
                "enabled": True,
                "hop_seconds": 0.5,
                "smooth_seconds": 2.0,
                "sample_rate": 16000,
                "events": {
                    "laughter": 1.0,
                    "cheering": 0.7,
                    "applause": 0.5,
                    "screaming": 0.8,
                    "shouting": 0.6,
                },
            },
            "enrich": {
                "enabled": True,
                # Note: weights removed - score fusion is handled by highlights analysis
                "hook": {
                    "max_chars": 60,
                    "window_seconds": 4.0,
                    "phrases": [
                        "no way",
                        "oh my god",
                        "let's go",
                        "lets go",
                        "bro",
                        "what",
                        "holy",
                        "wtf",
                        "omg",
                        "insane",
                        "clutch",
                        "no shot",
                        "bruh",
                        "dude",
                        "sheesh",
                    ],
                },
                "quote": {
                    "max_chars": 120,
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
        "overlay": {
            "hook_text": {
                "enabled": False,
                "duration_seconds": 2.0,
                "text": None,
                "font": "auto",
                "fontsize": 64,
                "y": 120,
            }
        },
        "layout": {
            "pip": {
                "position": "top_left",
                "margin": 40,
                "width_fraction": 0.28,
                "border_px": 6,
            }
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
