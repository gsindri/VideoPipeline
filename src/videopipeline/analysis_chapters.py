"""Semantic chapter detection using embeddings + changepoint detection.

Segments transcripts into semantic chapters based on topic shifts detected
via sentence embeddings and changepoint algorithms. Optionally uses LLM
to generate chapter titles and summaries.

This is distinct from analysis_scenes.py which detects visual shot cuts.
"""
from __future__ import annotations

# Workaround for PyTorch 2.9+ where torch.distributed.is_initialized was removed/moved
# Must be applied BEFORE importing sentence-transformers or related packages
try:
    import torch.distributed as _dist
    if not hasattr(_dist, 'is_initialized'):
        _dist.is_initialized = lambda: False
except ImportError:
    pass  # torch not installed

import json
import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .analysis_sentences import Sentence, compute_sentences_analysis, SentenceConfig
from .analysis_transcript import FullTranscript, load_transcript
from .project import Project, save_json, update_project

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class ChapterConfig:
    """Configuration for semantic chapter detection.
    
    Attributes:
        min_chapter_len_s: Minimum chapter length in seconds (default 60s)
        max_chapter_len_s: Maximum chapter length in seconds (default 900s = 15 min)
        embedding_model: Sentence-transformers model for embeddings
                        "all-mpnet-base-v2" (quality) or "all-MiniLM-L6-v2" (speed)
        changepoint_method: Ruptures method ("pelt", "binseg", "window")
        changepoint_penalty: Penalty for PELT/Binseg (higher = fewer chapters)
        snap_to_silence_window_s: Snap boundaries to silence within ±N seconds
        llm_labeling: Whether to use LLM for chapter titles/summaries
        llm_endpoint: LLM server endpoint (uses same pattern as AI Director)
        llm_model_name: Model name for LLM
        llm_timeout_s: Timeout for LLM requests
        max_chars_per_chapter: Max transcript chars to send to LLM per chapter
    """
    min_chapter_len_s: float = 60.0
    max_chapter_len_s: float = 900.0
    embedding_model: str = "all-mpnet-base-v2"
    changepoint_method: str = "pelt"
    changepoint_penalty: float = 10.0
    snap_to_silence_window_s: float = 10.0
    llm_labeling: bool = True
    llm_endpoint: str = "http://127.0.0.1:11435"
    llm_model_name: str = "local-gguf-vulkan"
    llm_timeout_s: float = 30.0
    max_chars_per_chapter: int = 6000


@dataclass
class Chapter:
    """A semantic chapter with timing and optional metadata."""
    id: int
    start_s: float
    end_s: float
    title: str = ""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    chapter_type: str = "content"  # gameplay, story, intro, break, outro, other

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "title": self.title,
            "summary": self.summary,
            "keywords": self.keywords,
            "type": self.chapter_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chapter":
        return cls(
            id=int(d.get("id", 0)),
            start_s=float(d.get("start_s", 0.0)),
            end_s=float(d.get("end_s", 0.0)),
            title=str(d.get("title", "")),
            summary=str(d.get("summary", "")),
            keywords=list(d.get("keywords", [])),
            chapter_type=str(d.get("type", "content")),
        )


# ============================================================================
# Embedding Generation
# ============================================================================

def _load_sentence_transformer(model_name: str):
    """Load a sentence-transformers model (lazy import for optional dependency)."""
    try:
        # Import directly to avoid sentence_transformers/__init__.py side-effects (cross-encoder, datasets, etc.).
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        logger.info(f"Loading sentence transformer model: {model_name}")
        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for semantic chapters. "
            "Install with: pip install sentence-transformers"
        )


def _compute_embeddings(
    texts: List[str],
    model_name: str,
    on_progress: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    """Compute sentence embeddings for a list of texts."""
    if not texts:
        return np.array([])
    
    model = _load_sentence_transformer(model_name)
    
    # Batch encode with progress
    batch_size = 32
    embeddings_list = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings_list.append(batch_embeddings)
        
        if on_progress:
            progress = min((i + batch_size) / len(texts), 1.0)
            on_progress(progress)
    
    return np.vstack(embeddings_list)


# ============================================================================
# Changepoint Detection
# ============================================================================

def _detect_changepoints_ruptures(
    embeddings: np.ndarray,
    method: str = "pelt",
    penalty: float = 10.0,
) -> List[int]:
    """Detect topic changepoints using ruptures library."""
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError(
            "ruptures is required for semantic chapter detection. "
            "Install with: pip install ruptures"
        )
    
    n_samples = len(embeddings)
    if n_samples < 3:
        return []
    
    # Use RBF cost function (good for embedding vectors)
    if method == "pelt":
        algo = rpt.Pelt(model="rbf", min_size=2).fit(embeddings)
        # PELT returns [breakpoint1, breakpoint2, ..., n_samples]
        changepoints = algo.predict(pen=penalty)
    elif method == "binseg":
        algo = rpt.Binseg(model="rbf", min_size=2).fit(embeddings)
        # Estimate number of breakpoints based on penalty
        n_bkps = max(1, int(n_samples / penalty))
        changepoints = algo.predict(n_bkps=n_bkps)
    elif method == "window":
        algo = rpt.Window(model="rbf", width=5, min_size=2).fit(embeddings)
        changepoints = algo.predict(pen=penalty)
    else:
        raise ValueError(f"Unknown changepoint method: {method}")
    
    # Remove the final point (which is always n_samples)
    if changepoints and changepoints[-1] == n_samples:
        changepoints = changepoints[:-1]
    
    return changepoints


def _detect_changepoints_cosine(
    embeddings: np.ndarray,
    threshold: float = 0.3,
    window_size: int = 1,
) -> List[int]:
    """Simple cosine similarity-based changepoint detection (fallback)."""
    from numpy.linalg import norm
    
    if len(embeddings) < 2:
        return []
    
    changepoints = []
    
    for i in range(window_size, len(embeddings) - window_size):
        # Compare windows before and after
        before = embeddings[i - window_size:i].mean(axis=0)
        after = embeddings[i:i + window_size].mean(axis=0)
        
        # Cosine similarity
        sim = np.dot(before, after) / (norm(before) * norm(after) + 1e-9)
        
        if sim < (1.0 - threshold):
            changepoints.append(i)
    
    return changepoints


# ============================================================================
# Boundary Snapping
# ============================================================================

def _load_silence_intervals(proj: Project) -> List[Tuple[float, float]]:
    """Load silence intervals from analysis/silence.json."""
    silence_path = proj.analysis_dir / "silence.json"
    if not silence_path.exists():
        return []
    
    try:
        data = json.loads(silence_path.read_text(encoding="utf-8"))
        intervals = data.get("silences", [])
        return [(s["start"], s["end"]) for s in intervals]
    except Exception as e:
        logger.warning(f"Could not load silence intervals: {e}")
        return []


def _snap_to_silence(
    boundary_time: float,
    silence_intervals: List[Tuple[float, float]],
    window_s: float,
) -> float:
    """Snap a boundary time to the nearest silence end within window."""
    if not silence_intervals:
        return boundary_time
    
    best_time = boundary_time
    best_distance = float("inf")
    
    for start, end in silence_intervals:
        # Check if silence end is within window
        if abs(end - boundary_time) <= window_s:
            distance = abs(end - boundary_time)
            if distance < best_distance:
                best_distance = distance
                best_time = end
    
    return best_time


# ============================================================================
# LLM Labeling
# ============================================================================

def _label_chapter_with_llm(
    transcript_text: str,
    cfg: ChapterConfig,
) -> Dict[str, Any]:
    """Use LLM to generate title, summary, and keywords for a chapter."""
    from .ai.llm_client import LLMClient, LLMClientConfig
    
    # Truncate transcript if too long
    if len(transcript_text) > cfg.max_chars_per_chapter:
        transcript_text = transcript_text[:cfg.max_chars_per_chapter] + "..."
    
    prompt = f"""Analyze this video transcript segment and provide a JSON response with:
- "title": A short, engaging title (3-8 words)
- "summary": A 1-2 sentence summary of what happens
- "keywords": 3-5 relevant keywords/topics
- "type": One of: "intro", "gameplay", "story", "commentary", "break", "outro", "other"

Transcript:
{transcript_text}

Respond ONLY with valid JSON, no other text."""

    try:
        client = LLMClient(
            LLMClientConfig(
                endpoint=cfg.llm_endpoint,
                timeout_s=cfg.llm_timeout_s,
            )
        )
        
        result = client.complete(
            prompt=prompt,
            max_tokens=256,
            temperature=0.3,
            json_mode=True,
        )
        
        # Result is already a parsed dict from client.complete()
        return {
            "title": str(result.get("title", "")),
            "summary": str(result.get("summary", "")),
            "keywords": list(result.get("keywords", [])),
            "type": str(result.get("type", "content")),
        }
    except Exception as e:
        logger.warning(f"LLM labeling failed: {e}")
        return {}


def _label_chapter_fallback(transcript_text: str) -> Dict[str, Any]:
    """Fallback labeling using simple heuristics (no LLM)."""
    # Extract first sentence as title
    sentences = transcript_text.split(".")
    title = sentences[0].strip()[:50] if sentences else "Chapter"
    
    # Simple keyword extraction (most common words > 4 chars)
    words = transcript_text.lower().split()
    word_freq: Dict[str, int] = {}
    for w in words:
        w = "".join(c for c in w if c.isalnum())
        if len(w) > 4:
            word_freq[w] = word_freq.get(w, 0) + 1
    
    keywords = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:5]
    
    return {
        "title": title if len(title) > 3 else "Chapter",
        "summary": "",
        "keywords": keywords,
        "type": "content",
    }


# ============================================================================
# Main Analysis Function
# ============================================================================

def compute_chapters_analysis(
    proj: Project,
    *,
    cfg: ChapterConfig = ChapterConfig(),
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute semantic chapter boundaries for a project.
    
    Pipeline:
    1. Load transcript (sentences preferred, segments as fallback)
    2. Compute embeddings for each sentence/segment
    3. Detect topic changepoints using ruptures
    4. Snap boundaries to silence intervals (if available)
    5. Enforce min/max chapter length constraints
    6. Optionally label chapters with LLM
    7. Save chapters.json and update project.json
    
    Args:
        proj: Project instance
        cfg: Chapter configuration
        on_progress: Optional progress callback (0.0 to 1.0)
    
    Returns:
        Dict with chapter analysis results
    """
    start_time = _time.time()
    
    if on_progress:
        on_progress(0.0)
    
    # Step 1: Load transcript
    transcript = load_transcript(proj)
    if transcript is None or not transcript.segments:
        raise ValueError("Transcript required for semantic chapters. Run transcription first.")
    
    # Try to load sentences (better granularity)
    sentences_path = proj.analysis_dir / "sentences.json"
    units: List[Tuple[float, float, str]] = []  # (start, end, text)
    
    if sentences_path.exists():
        try:
            data = json.loads(sentences_path.read_text(encoding="utf-8"))
            for s in data.get("sentences", []):
                units.append((s["t0"], s["t1"], s.get("text", "")))
            logger.info(f"Loaded {len(units)} sentences for chapter detection")
        except Exception as e:
            logger.warning(f"Could not load sentences: {e}")
    
    # Fallback to transcript segments
    if not units:
        for seg in transcript.segments:
            units.append((seg.start, seg.end, seg.text))
        logger.info(f"Using {len(units)} transcript segments for chapter detection")
    
    if len(units) < 3:
        raise ValueError("Not enough text units for chapter detection (need at least 3)")
    
    if on_progress:
        on_progress(0.1)
    
    # Step 2: Compute embeddings
    texts = [u[2] for u in units]
    
    def embed_progress(frac: float) -> None:
        if on_progress:
            on_progress(0.1 + 0.4 * frac)
    
    try:
        embeddings = _compute_embeddings(texts, cfg.embedding_model, on_progress=embed_progress)
    except ImportError as e:
        logger.warning(f"Embedding failed, using fallback: {e}")
        # Fallback: simple TF-IDF-like approach
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        embeddings = vectorizer.fit_transform(texts).toarray()
    
    if on_progress:
        on_progress(0.5)
    
    # Step 3: Detect changepoints
    try:
        changepoints = _detect_changepoints_ruptures(
            embeddings,
            method=cfg.changepoint_method,
            penalty=cfg.changepoint_penalty,
        )
    except ImportError:
        logger.warning("ruptures not available, using cosine similarity fallback")
        changepoints = _detect_changepoints_cosine(embeddings, threshold=0.3)
    
    logger.info(f"Detected {len(changepoints)} raw changepoints")
    
    if on_progress:
        on_progress(0.6)
    
    # Step 4: Convert changepoint indices to times and snap to silence
    silence_intervals = _load_silence_intervals(proj)
    
    boundary_times: List[float] = [0.0]  # Always start at 0
    for cp_idx in changepoints:
        if cp_idx < len(units):
            raw_time = units[cp_idx][0]  # Start time of unit at changepoint
            snapped_time = _snap_to_silence(
                raw_time,
                silence_intervals,
                cfg.snap_to_silence_window_s,
            )
            boundary_times.append(snapped_time)
    
    # Add end time
    video_end = units[-1][1] if units else 0.0
    boundary_times.append(video_end)
    
    # Sort and dedupe
    boundary_times = sorted(set(boundary_times))
    
    if on_progress:
        on_progress(0.65)
    
    # Step 5: Enforce min/max chapter length constraints
    final_boundaries: List[float] = [0.0]
    
    for i in range(1, len(boundary_times)):
        gap = boundary_times[i] - final_boundaries[-1]
        
        # If gap is too small, skip this boundary
        if gap < cfg.min_chapter_len_s:
            continue
        
        # If gap is too large, insert intermediate boundaries
        while gap > cfg.max_chapter_len_s:
            # Find a good split point (prefer silence)
            mid_target = final_boundaries[-1] + cfg.max_chapter_len_s * 0.8
            snapped_mid = _snap_to_silence(
                mid_target,
                silence_intervals,
                cfg.snap_to_silence_window_s * 2,
            )
            # Ensure we're making progress
            if snapped_mid <= final_boundaries[-1] + cfg.min_chapter_len_s:
                snapped_mid = final_boundaries[-1] + cfg.max_chapter_len_s * 0.8
            final_boundaries.append(snapped_mid)
            gap = boundary_times[i] - final_boundaries[-1]
        
        if gap >= cfg.min_chapter_len_s:
            final_boundaries.append(boundary_times[i])
    
    # Ensure we end at video end
    if final_boundaries[-1] < video_end - cfg.min_chapter_len_s:
        final_boundaries.append(video_end)
    elif final_boundaries[-1] != video_end:
        final_boundaries[-1] = video_end
    
    if on_progress:
        on_progress(0.7)
    
    # Step 6: Build chapter objects
    chapters: List[Chapter] = []
    
    for i in range(len(final_boundaries) - 1):
        start_s = final_boundaries[i]
        end_s = final_boundaries[i + 1]
        
        # Collect transcript text for this chapter
        chapter_text = ""
        for u_start, u_end, u_text in units:
            if u_start >= start_s and u_end <= end_s:
                chapter_text += " " + u_text
            elif u_start < end_s and u_end > start_s:
                # Partial overlap
                chapter_text += " " + u_text
        
        chapter_text = chapter_text.strip()
        
        chapters.append(Chapter(
            id=i,
            start_s=start_s,
            end_s=end_s,
            title=f"Chapter {i + 1}",  # Default title
        ))
    
    if on_progress:
        on_progress(0.75)
    
    # Step 7: LLM labeling (optional)
    if cfg.llm_labeling and chapters:
        logger.info(f"Running LLM labeling for {len(chapters)} chapters...")
        
        consecutive_failures = 0
        max_consecutive_failures = 2  # Bail after 2 consecutive LLM failures
        llm_disabled = False
        
        for i, chapter in enumerate(chapters):
            # Collect text for this chapter
            chapter_text = ""
            for u_start, u_end, u_text in units:
                if u_start >= chapter.start_s and u_start < chapter.end_s:
                    chapter_text += " " + u_text
            chapter_text = chapter_text.strip()
            
            if chapter_text:
                labels = None
                
                # Only try LLM if we haven't hit the failure threshold
                if not llm_disabled:
                    labels = _label_chapter_with_llm(chapter_text, cfg)
                    
                    if labels:
                        consecutive_failures = 0  # Reset on success
                        chapter.title = labels.get("title", chapter.title)
                        chapter.summary = labels.get("summary", "")
                        chapter.keywords = labels.get("keywords", [])
                        chapter.chapter_type = labels.get("type", "content")
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.warning(
                                f"LLM failed {consecutive_failures} times consecutively. "
                                f"Disabling LLM labeling for remaining {len(chapters) - i - 1} chapters. "
                                f"(Is the LLM server running?)"
                            )
                            llm_disabled = True
                
                # Use fallback if LLM failed or was disabled
                if not labels:
                    labels = _label_chapter_fallback(chapter_text)
                    chapter.title = labels.get("title", chapter.title)
                    chapter.keywords = labels.get("keywords", [])
            
            if on_progress:
                progress = 0.75 + 0.2 * ((i + 1) / len(chapters))
                on_progress(progress)
    
    if on_progress:
        on_progress(0.95)
    
    # Step 8: Build and save results
    elapsed_seconds = _time.time() - start_time
    generated_at = datetime.now(timezone.utc).isoformat()
    
    payload = {
        "method": "embedding_changepoint_v1",
        "config": {
            "min_chapter_len_s": cfg.min_chapter_len_s,
            "max_chapter_len_s": cfg.max_chapter_len_s,
            "embedding_model": cfg.embedding_model,
            "changepoint_method": cfg.changepoint_method,
            "changepoint_penalty": cfg.changepoint_penalty,
            "snap_to_silence_window_s": cfg.snap_to_silence_window_s,
            "llm_labeling": cfg.llm_labeling,
        },
        "generated_at": generated_at,
        "elapsed_seconds": elapsed_seconds,
        "chapter_count": len(chapters),
        "chapters": [c.to_dict() for c in chapters],
    }
    
    # Save chapters.json
    save_json(proj.chapters_path, payload)
    
    # Update project.json
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["chapters"] = {
            "generated_at": generated_at,
            "elapsed_seconds": elapsed_seconds,
            "chapter_count": len(chapters),
            "chapters_json": str(proj.chapters_path.relative_to(proj.project_dir)),
            "config": payload["config"],
        }
    
    update_project(proj, _upd)
    
    if on_progress:
        on_progress(1.0)
    
    logger.info(f"Chapter analysis complete: {len(chapters)} chapters in {elapsed_seconds:.1f}s")
    
    return payload


# ============================================================================
# Loading Functions
# ============================================================================

def load_chapters(proj: Project) -> Optional[List[Chapter]]:
    """Load chapters from analysis/chapters.json."""
    if not proj.chapters_path.exists():
        return None
    
    try:
        data = json.loads(proj.chapters_path.read_text(encoding="utf-8"))
        return [Chapter.from_dict(c) for c in data.get("chapters", [])]
    except Exception as e:
        logger.warning(f"Could not load chapters: {e}")
        return None


def get_chapter_at_time(chapters: List[Chapter], time_s: float) -> Optional[Chapter]:
    """Find the chapter containing a specific time."""
    for chapter in chapters:
        if chapter.start_s <= time_s < chapter.end_s:
            return chapter
    return None


def get_chapter_boundaries(chapters: List[Chapter]) -> List[float]:
    """Get all chapter boundary times (starts only, not ends)."""
    return [c.start_s for c in chapters]
