"""Candidate enrichment with hook and quote text extraction.

Takes existing highlight candidates and enriches them with:
- hook_text: short punchy overlay text for the clip
- quote_text: best sentence for title/caption

Note: Score fusion is handled by analysis_highlights.py. This module only
adds text-based metadata without modifying scores.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .analysis_speech_features import (
    DEFAULT_REACTION_PHRASES,
    compute_lexical_excitement,
    SpeechFeatureConfig,
)
from .analysis_transcript import FullTranscript, load_transcript, TranscriptSegment
from .project import Project, get_project_data, update_project


# Payoff words that indicate a good quote/moment
PAYOFF_WORDS: List[str] = [
    "win", "won", "dead", "died", "clutch", "insane", "crazy", "nice",
    "yes", "no", "what", "how", "finally", "got", "kill", "killed",
    "best", "worst", "gg", "ez", "lets go", "let's go", "omg", "pog",
]


@dataclass
class EnrichConfig:
    """Configuration for candidate enrichment."""
    enabled: bool = True
    # Hook extraction settings
    hook_max_chars: int = 60
    hook_window_seconds: float = 4.0
    # Quote extraction settings
    quote_max_chars: int = 120
    # Reaction phrases for hook detection
    reaction_phrases: List[str] = field(default_factory=lambda: DEFAULT_REACTION_PHRASES.copy())


# Legacy alias for backward compatibility
RerankConfig = EnrichConfig


def _clean_text(text: str) -> str:
    """Clean text for display (remove extra whitespace, etc.)."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_punchy(text: str, phrases: List[str]) -> bool:
    """Check if text is punchy/hook-worthy."""
    text_lower = text.lower()
    
    # Has exclamation or question
    if "!" in text or "?" in text:
        return True
    
    # Contains reaction phrase
    for phrase in phrases:
        if phrase.lower() in text_lower:
            return True
    
    # Short and snappy
    if len(text) <= 40:
        return True
    
    return False


def _sentence_excitement(text: str, phrases: List[str]) -> float:
    """Score a sentence for excitement/hook potential."""
    cfg = SpeechFeatureConfig(reaction_phrases=phrases)
    return compute_lexical_excitement(text, phrases, cfg)


def _has_payoff_word(text: str) -> bool:
    """Check if text contains a payoff word."""
    text_lower = text.lower()
    for word in PAYOFF_WORDS:
        if word in text_lower:
            return True
    return False


def extract_hook_text(
    transcript: FullTranscript,
    start_s: float,
    end_s: float,
    cfg: EnrichConfig,
) -> Optional[str]:
    """Extract a punchy hook text from the transcript within a time range.
    
    Looks for short, exciting phrases near the peak moment.
    """
    # Get segments overlapping the candidate window
    segments: List[TranscriptSegment] = []
    for seg in transcript.segments:
        if seg.end >= start_s and seg.start <= end_s:
            segments.append(seg)
    
    if not segments:
        return None
    
    # Collect word-level candidates if available
    word_candidates: List[Tuple[float, float, str]] = []
    for seg in segments:
        if seg.words:
            for word in seg.words:
                if word.end >= start_s and word.start <= end_s:
                    word_candidates.append((word.start, word.end, word.word))
    
    # Build candidate phrases from consecutive words
    phrase_candidates: List[Tuple[float, str]] = []
    
    if word_candidates:
        # Try phrases of 2-5 words
        for phrase_len in range(2, 6):
            for i in range(len(word_candidates) - phrase_len + 1):
                words = word_candidates[i:i + phrase_len]
                phrase = " ".join(w[2] for w in words)
                phrase = _clean_text(phrase)
                
                if len(phrase) <= cfg.hook_max_chars and _is_punchy(phrase, cfg.reaction_phrases):
                    # Score by position (prefer middle of clip) and excitement
                    mid_time = (words[0][0] + words[-1][1]) / 2
                    clip_mid = (start_s + end_s) / 2
                    position_score = 1.0 - abs(mid_time - clip_mid) / ((end_s - start_s) / 2 + 0.001)
                    excitement_score = _sentence_excitement(phrase, cfg.reaction_phrases)
                    total_score = 0.4 * position_score + 0.6 * excitement_score
                    phrase_candidates.append((total_score, phrase))
    
    # Also try segment-level text
    for seg in segments:
        text = _clean_text(seg.text)
        if text and len(text) <= cfg.hook_max_chars and _is_punchy(text, cfg.reaction_phrases):
            excitement_score = _sentence_excitement(text, cfg.reaction_phrases)
            phrase_candidates.append((excitement_score, text))
    
    if not phrase_candidates:
        return None
    
    # Return best scoring phrase
    phrase_candidates.sort(key=lambda x: x[0], reverse=True)
    return phrase_candidates[0][1]


def extract_quote_text(
    transcript: FullTranscript,
    start_s: float,
    end_s: float,
    cfg: EnrichConfig,
) -> Optional[str]:
    """Extract the best quotable sentence from a time range.
    
    Prefers complete sentences with payoff words and excitement.
    """
    # Get segments overlapping the candidate window
    segments: List[TranscriptSegment] = []
    for seg in transcript.segments:
        if seg.end >= start_s and seg.start <= end_s:
            segments.append(seg)
    
    if not segments:
        return None
    
    # Collect full text and split into sentences
    full_text = " ".join(seg.text for seg in segments)
    
    # Simple sentence splitting on . ! ?
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    # Score each sentence
    scored_sentences: List[Tuple[float, str]] = []
    for sentence in sentences:
        sentence = _clean_text(sentence)
        if not sentence or len(sentence) > cfg.quote_max_chars:
            continue
        
        score = _sentence_excitement(sentence, cfg.reaction_phrases)
        
        # Bonus for payoff words
        if _has_payoff_word(sentence):
            score += 2.0
        
        # Penalty for very short sentences (less meaningful)
        if len(sentence) < 10:
            score *= 0.5
        
        scored_sentences.append((score, sentence))
    
    if not scored_sentences:
        # Fallback to first segment text
        first_text = _clean_text(segments[0].text)
        if first_text and len(first_text) <= cfg.quote_max_chars:
            return first_text
        return None
    
    # Sort by score descending
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    return scored_sentences[0][1]


def enrich_candidates(
    proj: Project,
    *,
    cfg: EnrichConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Enrich highlight candidates with hook and quote text.

    For each candidate:
    - Extract hook_text (short punchy overlay text)
    - Extract quote_text (best sentence for title/caption)

    Note: Scores are NOT modified - score fusion is handled by highlights analysis.

    Persists:
      - project.json -> analysis.highlights.candidates (updated with text fields)
      - project.json -> analysis.enrich section
    """
    if not cfg.enabled:
        return {"enabled": False, "message": "Enrichment disabled"}

    proj_data = get_project_data(proj)
    highlights = proj_data.get("analysis", {}).get("highlights", {})
    candidates = highlights.get("candidates", [])

    if not candidates:
        return {"enabled": True, "error": "No candidates to enrich"}

    if on_progress:
        on_progress(0.1)

    # Load transcript
    transcript = load_transcript(proj)
    has_transcript = transcript is not None

    if on_progress:
        on_progress(0.2)

    # Process each candidate
    enriched_candidates: List[Dict[str, Any]] = []

    for i, cand in enumerate(candidates):
        start_s = float(cand.get("start_s", 0))
        end_s = float(cand.get("end_s", 0))

        # Extract hook and quote text
        hook_text = None
        quote_text = None
        if has_transcript and transcript is not None:
            hook_text = extract_hook_text(transcript, start_s, end_s, cfg)
            quote_text = extract_quote_text(transcript, start_s, end_s, cfg)

        # Create enriched candidate (preserve all existing fields including score)
        enriched = {**cand}

        if hook_text:
            enriched["hook_text"] = hook_text
        if quote_text:
            enriched["quote_text"] = quote_text

        enriched_candidates.append(enriched)

        if on_progress and (i + 1) % 5 == 0:
            on_progress(0.2 + 0.7 * ((i + 1) / len(candidates)))

    if on_progress:
        on_progress(0.9)

    # Update project with enriched candidates
    enrich_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "hook_max_chars": cfg.hook_max_chars,
            "hook_window_seconds": cfg.hook_window_seconds,
            "quote_max_chars": cfg.quote_max_chars,
        },
        "has_transcript": has_transcript,
        "candidate_count": len(enriched_candidates),
    }

    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"].setdefault("highlights", {})
        d["analysis"]["highlights"]["candidates"] = enriched_candidates
        d["analysis"]["highlights"]["enriched_at"] = enrich_payload["created_at"]
        d["analysis"]["enrich"] = enrich_payload

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return {
        "enabled": True,
        "candidates": enriched_candidates,
        **enrich_payload,
    }


# Legacy alias for backward compatibility
compute_reranked_candidates = enrich_candidates


def get_candidate_hook(candidate: Dict[str, Any]) -> Optional[str]:
    """Get the hook text for a candidate if available."""
    return candidate.get("hook_text")


def get_candidate_quote(candidate: Dict[str, Any]) -> Optional[str]:
    """Get the quote text for a candidate if available."""
    return candidate.get("quote_text")
