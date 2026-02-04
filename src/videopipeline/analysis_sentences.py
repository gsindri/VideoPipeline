"""Sentence boundary extraction from transcript.

Converts transcript segments into timestamped sentences for clip shaping.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .analysis_transcript import FullTranscript, TranscriptSegment, TranscriptWord, load_transcript
from .project import Project, save_json, update_project


@dataclass(frozen=True)
class SentenceConfig:
    """Configuration for sentence boundary extraction."""
    max_sentence_words: int = 30  # Force sentence break after this many words
    sentence_end_chars: str = ".!?"  # Characters that end a sentence


@dataclass
class Sentence:
    """A sentence with timing information."""
    t0: float  # Start time
    t1: float  # End time
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"t0": self.t0, "t1": self.t1, "text": self.text}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sentence":
        return cls(
            t0=float(d["t0"]),
            t1=float(d["t1"]),
            text=str(d.get("text", "")),
        )

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def _is_sentence_end(text: str, end_chars: str) -> bool:
    """Check if text ends with a sentence-ending character."""
    text = text.strip()
    if not text:
        return False
    return text[-1] in end_chars


def _cleanup_punctuation_spacing(text: str) -> str:
    """Remove spaces before punctuation marks for cleaner display.
    
    Converts "hello , world !" to "hello, world!"
    """
    # Remove space before common punctuation
    text = re.sub(r"\s+([,.!?;:'\"])", r"\1", text)
    # Ensure single space after punctuation (except end of string)
    text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)
    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_sentences_from_words(
    words: List[TranscriptWord],
    cfg: SentenceConfig,
) -> List[Sentence]:
    """Extract sentences from word-level timestamps."""
    if not words:
        return []

    sentences: List[Sentence] = []
    current_words: List[TranscriptWord] = []

    for word in words:
        current_words.append(word)

        # Check if we should end the sentence
        should_end = (
            _is_sentence_end(word.word, cfg.sentence_end_chars)
            or len(current_words) >= cfg.max_sentence_words
        )

        if should_end and current_words:
            text = " ".join(w.word.strip() for w in current_words)
            text = _cleanup_punctuation_spacing(text)
            if text:
                sentences.append(Sentence(
                    t0=current_words[0].start,
                    t1=current_words[-1].end,
                    text=text,
                ))
            current_words = []

    # Handle remaining words
    if current_words:
        text = " ".join(w.word.strip() for w in current_words)
        text = _cleanup_punctuation_spacing(text)
        if text:
            sentences.append(Sentence(
                t0=current_words[0].start,
                t1=current_words[-1].end,
                text=text,
            ))

    return sentences


def _extract_sentences_from_segments(
    segments: List[TranscriptSegment],
    cfg: SentenceConfig,
) -> List[Sentence]:
    """Extract sentences from segment-level data (fallback when no word timestamps)."""
    if not segments:
        return []

    sentences: List[Sentence] = []

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        # Split segment text by sentence-ending punctuation
        # Keep the delimiter at the end of each sentence
        pattern = f"([{re.escape(cfg.sentence_end_chars)}])"
        parts = re.split(pattern, text)

        # Reconstruct sentences (pair text with delimiter)
        segment_sentences: List[str] = []
        current = ""
        for part in parts:
            if part in cfg.sentence_end_chars:
                current += part
                if current.strip():
                    segment_sentences.append(current.strip())
                current = ""
            else:
                current = part

        if current.strip():
            segment_sentences.append(current.strip())

        if not segment_sentences:
            segment_sentences = [text]

        # Distribute timing proportionally across sentences
        total_chars = sum(len(s) for s in segment_sentences)
        if total_chars <= 0:
            total_chars = 1

        segment_duration = segment.end - segment.start
        current_time = segment.start

        for sent_text in segment_sentences:
            if not sent_text:
                continue
            # Proportional duration based on character count
            sent_duration = (len(sent_text) / total_chars) * segment_duration
            sentences.append(Sentence(
                t0=current_time,
                t1=current_time + sent_duration,
                text=sent_text,
            ))
            current_time += sent_duration

    return sentences


def extract_sentences(
    transcript: FullTranscript,
    cfg: SentenceConfig,
) -> List[Sentence]:
    """Extract sentences from transcript, handling mixed word/segment timestamps.
    
    For segments with word timestamps: use word-based extraction (more accurate).
    For segments without word timestamps: use segment-based extraction (fallback).
    Results are merged and sorted by start time.
    """
    # Collect words and segments separately based on availability
    all_words: List[TranscriptWord] = []
    segments_without_words: List[TranscriptSegment] = []

    for segment in transcript.segments:
        if segment.words and len(segment.words) > 0:
            all_words.extend(segment.words)
        else:
            segments_without_words.append(segment)

    sentences: List[Sentence] = []
    
    # Extract from word timestamps where available
    if all_words:
        sentences.extend(_extract_sentences_from_words(all_words, cfg))
    
    # Also extract from segments that don't have word timestamps
    # This handles mixed transcripts where some segments have words and some don't
    if segments_without_words:
        sentences.extend(_extract_sentences_from_segments(segments_without_words, cfg))
    
    # Sort by start time to interleave properly
    sentences.sort(key=lambda s: s.t0)
    
    return sentences


def compute_sentences_analysis(
    proj: Project,
    *,
    cfg: SentenceConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Extract sentences from transcript and save to project.
    
    Persists:
      - analysis/sentences.json
      - project.json -> analysis.sentences section
    """
    if on_progress:
        on_progress(0.1)

    transcript = load_transcript(proj)
    if transcript is None:
        raise ValueError("No transcript found. Run transcript analysis first.")

    if on_progress:
        on_progress(0.3)

    sentences = extract_sentences(transcript, cfg)

    if on_progress:
        on_progress(0.8)

    # Build payload
    sentences_path = proj.analysis_dir / "sentences.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_sentence_words": cfg.max_sentence_words,
            "sentence_end_chars": cfg.sentence_end_chars,
        },
        "sentences": [s.to_dict() for s in sentences],
        "count": len(sentences),
    }

    # Save sentences.json
    save_json(sentences_path, payload)

    # Update project.json
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["sentences"] = {
            "created_at": payload["created_at"],
            "config": payload["config"],
            "count": len(sentences),
            "sentences_json": str(sentences_path.relative_to(proj.project_dir)),
        }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload


def load_sentences(proj: Project) -> Optional[List[Sentence]]:
    """Load cached sentences if available."""
    sentences_path = proj.analysis_dir / "sentences.json"
    if not sentences_path.exists():
        return None

    data = json.loads(sentences_path.read_text(encoding="utf-8"))
    return [Sentence.from_dict(s) for s in data.get("sentences", [])]


def get_sentence_boundaries(sentences: List[Sentence]) -> Dict[str, List[float]]:
    """Extract boundary timestamps from sentences.
    
    Returns:
        Dict with:
          - sentence_starts: Times where sentences begin (good start points)
          - sentence_ends: Times where sentences end (good end points)
    """
    return {
        "sentence_starts": [s.t0 for s in sentences],
        "sentence_ends": [s.t1 for s in sentences],
    }


def get_sentence_at_time(sentences: List[Sentence], time_s: float) -> Optional[Sentence]:
    """Get the sentence that contains a given timestamp."""
    for sent in sentences:
        if sent.t0 <= time_s <= sent.t1:
            return sent
    return None


def get_sentences_in_range(
    sentences: List[Sentence],
    start_s: float,
    end_s: float,
) -> List[Sentence]:
    """Get all sentences that overlap with a time range."""
    return [s for s in sentences if s.t1 > start_s and s.t0 < end_s]
