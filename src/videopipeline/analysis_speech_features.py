"""Speech feature extraction from transcript.

Converts transcript to hop-aligned timelines for speech analysis.
"""
from __future__ import annotations

import re
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from .analysis_transcript import FullTranscript, load_transcript
from .peaks import robust_z
from .project import Project, save_npz, update_project


# Default reaction phrases for gaming/streaming content
DEFAULT_REACTION_PHRASES: List[str] = [
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
    "yo",
    "oh no",
    "wait",
    "huh",
    "wow",
    "nice",
    "crazy",
    "dead",
    "gg",
    "pog",
    "hype",
]


@dataclass
class SpeechFeatureConfig:
    """Configuration for speech feature extraction."""
    hop_seconds: float = 0.5
    reaction_phrases: List[str] = field(default_factory=lambda: DEFAULT_REACTION_PHRASES.copy())
    # Weights for lexical excitement components
    exclamation_weight: float = 1.0
    question_weight: float = 0.5
    uppercase_weight: float = 0.8
    repeated_chars_weight: float = 0.6
    phrase_weight: float = 1.5


def _count_exclamations_questions(text: str) -> tuple[int, int]:
    """Count exclamation and question marks."""
    return text.count("!"), text.count("?")


def _uppercase_ratio(text: str) -> float:
    """Calculate ratio of uppercase letters (ignoring non-letters)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def _count_repeated_chars(text: str) -> int:
    """Count instances of 3+ repeated characters (e.g., 'nooooo', 'whaaaat')."""
    pattern = r"(.)\1{2,}"
    matches = re.findall(pattern, text.lower())
    return len(matches)


def _count_reaction_phrases(text: str, phrases: List[str]) -> int:
    """Count occurrences of reaction phrases in text."""
    text_lower = text.lower()
    count = 0
    for phrase in phrases:
        # Use word boundary matching for single words
        if " " not in phrase:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            count += len(re.findall(pattern, text_lower))
        else:
            # For multi-word phrases, simple substring match
            count += text_lower.count(phrase.lower())
    return count


def compute_lexical_excitement(
    text: str,
    phrases: List[str],
    cfg: SpeechFeatureConfig,
) -> float:
    """Compute lexical excitement score for a text segment."""
    if not text.strip():
        return 0.0

    excl, quest = _count_exclamations_questions(text)
    upper_ratio = _uppercase_ratio(text)
    repeated = _count_repeated_chars(text)
    phrase_count = _count_reaction_phrases(text, phrases)

    # Combine features with weights
    score = (
        cfg.exclamation_weight * excl
        + cfg.question_weight * quest
        + cfg.uppercase_weight * upper_ratio * 2.0  # Scale ratio contribution
        + cfg.repeated_chars_weight * repeated
        + cfg.phrase_weight * phrase_count
    )

    return score


def compute_speech_features(
    proj: Project,
    *,
    cfg: SpeechFeatureConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute hop-aligned speech features from transcript.

    Generates timelines:
      - speech_presence: 1 if speech covers this hop, else 0
      - words_per_second: speech rate in the hop window
      - lexical_excitement: excitement score from text features
      - speech_score: normalized composite speech score

    Persists:
      - analysis/speech_features.npz
      - project.json -> analysis.speech section
    """
    start_time = _time.time()
    
    transcript = load_transcript(proj)
    if transcript is None:
        raise ValueError("No transcript found. Run transcript analysis first.")

    duration_s = transcript.duration_seconds
    if duration_s <= 0:
        raise ValueError("Invalid transcript duration")

    hop_s = cfg.hop_seconds
    n_hops = int(np.ceil(duration_s / hop_s))
    
    # Helper for progress reporting with optional message
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)

    _report(0.1, "Initializing feature arrays")

    # Initialize arrays
    speech_presence = np.zeros(n_hops, dtype=np.float64)
    words_per_second = np.zeros(n_hops, dtype=np.float64)
    lexical_excitement = np.zeros(n_hops, dtype=np.float64)
    
    # Check if we have word timestamps available
    has_word_timestamps = any(
        seg.words and len(seg.words) > 0 
        for seg in transcript.segments
    )
    
    if has_word_timestamps:
        # PATH A: Word timestamps available - bin words into hops once (O(n_words))
        # This is much faster and more accurate than per-hop scanning
        hop_word_counts = np.zeros(n_hops, dtype=np.int32)
        hop_tokens: List[List[str]] = [[] for _ in range(n_hops)]
        
        for seg in transcript.segments:
            if seg.words:
                for word in seg.words:
                    # Use word midpoint to assign to hop
                    word_mid = (word.start + word.end) / 2.0
                    hop_idx = int(word_mid / hop_s)
                    if 0 <= hop_idx < n_hops:
                        hop_word_counts[hop_idx] += 1
                        hop_tokens[hop_idx].append(word.word)
                        speech_presence[hop_idx] = 1.0
        
        # Compute features from binned data
        words_per_second = hop_word_counts.astype(np.float64) / hop_s
        
        for i in range(n_hops):
            if hop_tokens[i]:
                text = " ".join(hop_tokens[i])
                lexical_excitement[i] = compute_lexical_excitement(
                    text, cfg.reaction_phrases, cfg
                )
            
            if on_progress and i % 500 == 0:
                _report(0.1 + 0.7 * (i / n_hops), f"Processing hop {i}/{n_hops}")
    else:
        # PATH B: No word timestamps - distribute segment-level stats into overlapping hops
        # This avoids the "whole segment text repeated per hop" inflation bug
        eps = 1e-9
        
        for seg in transcript.segments:
            seg_duration = max(seg.end - seg.start, eps)
            seg_text = seg.text.strip()
            if not seg_text:
                continue
            
            # Compute segment-level metrics once
            seg_word_count = len(seg_text.split())
            seg_wps = seg_word_count / seg_duration
            seg_excitement = compute_lexical_excitement(seg_text, cfg.reaction_phrases, cfg)
            
            # Distribute to all overlapping hops
            start_hop = int(seg.start / hop_s)
            end_hop = int(np.ceil(seg.end / hop_s))
            
            for i in range(max(0, start_hop), min(n_hops, end_hop)):
                speech_presence[i] = 1.0
                words_per_second[i] = seg_wps
                lexical_excitement[i] = seg_excitement

        _report(0.8, "Segment processing complete")

    _report(0.8, "Normalizing with z-scores")

    # Normalize each feature with robust z-score
    speech_presence_z = robust_z(speech_presence) if np.any(speech_presence > 0) else speech_presence
    words_per_second_z = robust_z(words_per_second) if np.any(words_per_second > 0) else words_per_second
    lexical_excitement_z = robust_z(lexical_excitement) if np.any(lexical_excitement > 0) else lexical_excitement

    # Composite speech score (emphasize lexical excitement and speech rate)
    speech_score = (
        0.2 * np.clip(speech_presence_z, 0, None)  # Presence as binary bonus
        + 0.4 * np.clip(words_per_second_z, 0, None)  # Speech rate
        + 0.4 * np.clip(lexical_excitement_z, 0, None)  # Excitement
    )

    # Re-normalize the composite
    speech_score = robust_z(speech_score) if np.any(speech_score != 0) else speech_score

    _report(0.9, "Saving speech_features.npz")

    # Save features
    speech_features_path = proj.analysis_dir / "speech_features.npz"
    save_npz(
        speech_features_path,
        speech_presence=speech_presence,
        words_per_second=words_per_second,
        lexical_excitement=lexical_excitement,
        speech_presence_z=speech_presence_z,
        words_per_second_z=words_per_second_z,
        lexical_excitement_z=lexical_excitement_z,
        speech_score=speech_score,
        hop_seconds=np.array([hop_s], dtype=np.float64),
    )

    # Update project
    elapsed_seconds = _time.time() - start_time
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "config": {
            "hop_seconds": hop_s,
            "reaction_phrases": cfg.reaction_phrases,
        },
        "duration_seconds": duration_s,
        "hop_count": n_hops,
        "features_npz": str(speech_features_path.relative_to(proj.project_dir)),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["speech"] = payload

    update_project(proj, _upd)

    _report(1.0, "Done")

    return payload


def load_speech_features(proj: Project) -> Optional[Dict[str, np.ndarray]]:
    """Load speech features if available."""
    speech_features_path = proj.analysis_dir / "speech_features.npz"
    if not speech_features_path.exists():
        return None

    data = np.load(speech_features_path)
    return {k: data[k] for k in data.files}
