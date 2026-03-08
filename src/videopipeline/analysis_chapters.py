"""Semantic chapter detection using embeddings + changepoint detection.

Segments transcripts into semantic chapters based on topic shifts detected
via sentence embeddings and changepoint algorithms. Optionally uses LLM
to generate chapter titles and summaries.

This is distinct from analysis_scenes.py which detects visual shot cuts.
"""
from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .analysis_transcript import load_transcript
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
        changepoint_method: Segmentation method ("auto", "pelt", "binseg", "window")
        changepoint_penalty: Initial penalty for PELT/Binseg (higher = fewer chapters)
        changepoint_auto_tune: Auto-tune changepoint strength to avoid 0/over-fragmented splits
        min_changepoints: Minimum raw changepoints to target before fallback
        target_chapter_len_s: Preferred chapter length for adaptive tuning
        target_chapters_min: Optional hard lower bound for chapter count
        target_chapters_max: Optional hard upper bound for chapter count
        target_chapters_ideal: Optional preferred chapter count (overrides auto estimate)
        snap_to_silence_window_s: Snap boundaries to silence within ±N seconds
        llm_labeling: Whether to use LLM for chapter titles/summaries
        llm_endpoint: LLM server endpoint (uses same pattern as AI Director)
        llm_model_name: Model name for LLM
        llm_api_key: Optional API key (falls back to OPENAI_API_KEY when omitted)
        llm_timeout_s: Timeout for LLM requests
        max_chars_per_chapter: Max transcript chars to send to LLM per chapter
    """
    min_chapter_len_s: float = 60.0
    max_chapter_len_s: float = 900.0
    embedding_model: str = "all-mpnet-base-v2"
    changepoint_method: str = "auto"
    changepoint_penalty: float = 10.0
    changepoint_auto_tune: bool = True
    min_changepoints: int = 1
    target_chapter_len_s: float = 600.0
    target_chapters_min: Optional[int] = None
    target_chapters_max: Optional[int] = None
    target_chapters_ideal: Optional[int] = None
    snap_to_silence_window_s: float = 10.0
    llm_labeling: bool = True
    llm_endpoint: str = "http://127.0.0.1:11435"
    llm_model_name: str = "local-gguf-vulkan"
    llm_api_key: Optional[str] = None
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
        try:
            import torch.distributed as _dist

            if not hasattr(_dist, "is_initialized"):
                _dist.is_initialized = lambda: False
        except ImportError:
            pass

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


def _detect_changepoints_binseg_fixed(
    embeddings: np.ndarray,
    n_bkps: int,
) -> List[int]:
    """Detect changepoints with a fixed BinSeg breakpoint count."""
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError(
            "ruptures is required for semantic chapter detection. "
            "Install with: pip install ruptures"
        )

    n_samples = len(embeddings)
    if n_samples < 3 or n_bkps <= 0:
        return []

    capped = min(max(1, int(n_bkps)), max(1, n_samples - 2))
    algo = rpt.Binseg(model="rbf", min_size=2).fit(embeddings)
    changepoints = algo.predict(n_bkps=capped)
    if changepoints and changepoints[-1] == n_samples:
        changepoints = changepoints[:-1]
    return changepoints


def _estimate_target_chapter_range(
    duration_s: float,
    cfg: ChapterConfig,
) -> Tuple[int, int, int]:
    """Estimate target chapter count range for adaptive changepoint tuning."""
    safe_min_len = max(1.0, float(cfg.min_chapter_len_s))
    safe_max_len = max(safe_min_len, float(cfg.max_chapter_len_s))
    safe_duration = max(safe_max_len, float(duration_s))

    # Hard bounds implied by chapter length constraints.
    min_by_len = max(1, int(np.ceil(safe_duration / safe_max_len)))
    max_by_len = max(min_by_len, int(np.floor(safe_duration / safe_min_len)))

    preferred_len = min(max(float(cfg.target_chapter_len_s), safe_min_len), safe_max_len)
    auto_ideal = max(1, int(round(safe_duration / preferred_len)))
    auto_ideal = max(min_by_len, min(max_by_len, auto_ideal))

    spread = max(1, int(round(auto_ideal * 0.5)))
    auto_min = max(min_by_len, auto_ideal - spread)
    auto_max = min(max_by_len, auto_ideal + spread)

    min_target = int(cfg.target_chapters_min) if cfg.target_chapters_min is not None else auto_min
    max_target = int(cfg.target_chapters_max) if cfg.target_chapters_max is not None else auto_max
    min_target = max(min_by_len, min_target)
    max_target = min(max_by_len, max(max_target, min_target))

    if cfg.target_chapters_ideal is not None:
        ideal = int(cfg.target_chapters_ideal)
    else:
        ideal = auto_ideal
    ideal = max(min_target, min(max_target, ideal))

    return min_target, max_target, ideal


def _generate_penalty_candidates(initial_penalty: float) -> List[float]:
    """Generate robust penalty candidates around an initial value."""
    base = max(1e-3, float(initial_penalty))
    multipliers = [
        64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0,
        0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125,
    ]
    fixed = [
        3.0, 2.5, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0,
        0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01,
    ]

    values: List[float] = [base * m for m in multipliers] + fixed + [base]
    dedup: List[float] = []
    seen = set()
    for v in values:
        vv = max(1e-3, min(1e4, float(v)))
        key = round(vv, 6)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(vv)
    return dedup


def _detect_changepoints_pelt_adaptive(
    embeddings: np.ndarray,
    *,
    initial_penalty: float,
    target_min: int,
    target_max: int,
    target_ideal: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Auto-tune PELT penalty to avoid zero or unstable changepoint counts."""
    penalties = _generate_penalty_candidates(initial_penalty)
    tested: List[Tuple[float, int, List[int]]] = []
    for pen in penalties:
        cps = _detect_changepoints_ruptures(embeddings, method="pelt", penalty=pen)
        tested.append((pen, len(cps), cps))

    def _penalty_distance(pen: float) -> float:
        return abs(np.log(max(pen, 1e-6)) - np.log(max(initial_penalty, 1e-6)))

    in_range_nonzero = [
        t for t in tested
        if target_min <= t[1] <= target_max and (t[1] > 0 or target_ideal == 0)
    ]

    if in_range_nonzero:
        selected = min(
            in_range_nonzero,
            key=lambda t: (abs(t[1] - target_ideal), _penalty_distance(t[0])),
        )
        reason = "in_target_range"
    else:
        nonzero = [t for t in tested if t[1] > 0]
        if nonzero:
            selected = min(
                nonzero,
                key=lambda t: (
                    abs(t[1] - target_ideal),
                    0 if t[1] >= target_min else 1,
                    _penalty_distance(t[0]),
                ),
            )
            reason = "closest_nonzero"
        else:
            selected = min(tested, key=lambda t: _penalty_distance(t[0]))
            reason = "all_zero"

    selected_pen, selected_count, selected_cps = selected
    diagnostics = {
        "requested_method": "pelt",
        "selected_method": "pelt",
        "selected_penalty": selected_pen,
        "selected_count": selected_count,
        "selection_reason": reason,
        "target_count_min": target_min,
        "target_count_max": target_max,
        "target_count_ideal": target_ideal,
        "tested_penalties": [
            {"penalty": round(p, 6), "count": c}
            for p, c, _ in tested
        ],
    }
    return selected_cps, diagnostics


def _detect_changepoints_cosine_adaptive(
    embeddings: np.ndarray,
    *,
    target_min: int,
    target_max: int,
    target_ideal: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Adaptive cosine fallback when ruptures is unavailable."""
    thresholds = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    tested: List[Tuple[float, int, List[int]]] = []
    for th in thresholds:
        cps = _detect_changepoints_cosine(embeddings, threshold=th, window_size=1)
        tested.append((th, len(cps), cps))

    in_range_nonzero = [
        t for t in tested
        if target_min <= t[1] <= target_max and (t[1] > 0 or target_ideal == 0)
    ]
    if in_range_nonzero:
        selected = min(in_range_nonzero, key=lambda t: abs(t[1] - target_ideal))
        reason = "in_target_range"
    else:
        nonzero = [t for t in tested if t[1] > 0]
        if nonzero:
            selected = min(nonzero, key=lambda t: abs(t[1] - target_ideal))
            reason = "closest_nonzero"
        else:
            selected = tested[-1]
            reason = "all_zero"

    threshold, count, cps = selected
    diagnostics = {
        "requested_method": "cosine_fallback",
        "selected_method": "cosine",
        "selected_threshold": threshold,
        "selected_count": count,
        "selection_reason": reason,
        "target_count_min": target_min,
        "target_count_max": target_max,
        "target_count_ideal": target_ideal,
        "tested_thresholds": [
            {"threshold": th, "count": c}
            for th, c, _ in tested
        ],
    }
    return cps, diagnostics


def _prune_changepoints_by_time_gap(
    changepoints: List[int],
    units: List[Tuple[float, float, str]],
    min_gap_s: float,
) -> List[int]:
    """Drop changepoints that are too close together in time."""
    if not changepoints:
        return []

    cleaned = sorted(set(int(i) for i in changepoints if 0 < int(i) < len(units)))
    if not cleaned:
        return []

    min_gap = max(1.0, float(min_gap_s))
    pruned: List[int] = []
    last_t = -1e9
    for idx in cleaned:
        t = float(units[idx][0])
        if t - last_t >= min_gap:
            pruned.append(idx)
            last_t = t
    return pruned


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
    *,
    llm_complete: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Use LLM to generate title, summary, and keywords for a chapter."""

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
        result: Any
        if llm_complete is not None:
            raw = llm_complete(prompt)
            if isinstance(raw, dict):
                result = raw
            else:
                raw_text = str(raw or "").strip()
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    if lines and lines[0].strip().startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    raw_text = "\n".join(lines).strip()
                    if raw_text.lower().startswith("json"):
                        raw_text = raw_text[4:].lstrip()
                result = json.loads(raw_text)
        else:
            from .ai.llm_client import LLMClient, LLMClientConfig

            client = LLMClient(
                LLMClientConfig(
                    endpoint=cfg.llm_endpoint,
                    model_name=cfg.llm_model_name,
                    api_key=cfg.llm_api_key,
                    timeout_s=cfg.llm_timeout_s,
                )
            )

            result = client.complete(
                prompt=prompt,
                max_tokens=256,
                temperature=0.3,
                json_mode=True,
            )

        if not isinstance(result, dict):
            raise ValueError("LLM chapter labeling returned non-dict payload")

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
    llm_complete: Optional[Callable[[str], str]] = None,
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
        llm_complete: Optional completion callable supplied by DAG runner
            (uses active Studio AI backend, e.g. OpenAI API or local server)
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
    video_end = units[-1][1] if units else 0.0
    target_ch_min, target_ch_max, target_ch_ideal = _estimate_target_chapter_range(video_end, cfg)
    target_cp_min = max(0, target_ch_min - 1)
    target_cp_max = max(0, target_ch_max - 1)
    target_cp_ideal = max(0, target_ch_ideal - 1)

    changepoint_meta: Dict[str, Any] = {
        "target_chapters_min": target_ch_min,
        "target_chapters_max": target_ch_max,
        "target_chapters_ideal": target_ch_ideal,
        "target_raw_changepoints_min": target_cp_min,
        "target_raw_changepoints_max": target_cp_max,
        "target_raw_changepoints_ideal": target_cp_ideal,
    }

    method = str(cfg.changepoint_method or "auto").strip().lower()
    changepoints: List[int] = []
    try:
        if method in {"auto", "pelt"}:
            if method == "auto" or cfg.changepoint_auto_tune:
                changepoints, tune_meta = _detect_changepoints_pelt_adaptive(
                    embeddings,
                    initial_penalty=cfg.changepoint_penalty,
                    target_min=target_cp_min,
                    target_max=target_cp_max,
                    target_ideal=target_cp_ideal,
                )
                changepoint_meta.update(tune_meta)
            else:
                changepoints = _detect_changepoints_ruptures(
                    embeddings,
                    method="pelt",
                    penalty=cfg.changepoint_penalty,
                )
                changepoint_meta.update({
                    "requested_method": "pelt",
                    "selected_method": "pelt",
                    "selected_penalty": cfg.changepoint_penalty,
                })
        elif method == "binseg":
            if cfg.changepoint_auto_tune:
                n_bkps = max(cfg.min_changepoints, target_cp_ideal, 1)
                changepoints = _detect_changepoints_binseg_fixed(embeddings, n_bkps=n_bkps)
                changepoint_meta.update({
                    "requested_method": "binseg",
                    "selected_method": "binseg",
                    "selected_n_bkps": n_bkps,
                })
            else:
                changepoints = _detect_changepoints_ruptures(
                    embeddings,
                    method="binseg",
                    penalty=cfg.changepoint_penalty,
                )
                changepoint_meta.update({
                    "requested_method": "binseg",
                    "selected_method": "binseg",
                    "selected_penalty": cfg.changepoint_penalty,
                })
        elif method == "window":
            changepoints = _detect_changepoints_ruptures(
                embeddings,
                method="window",
                penalty=cfg.changepoint_penalty,
            )
            changepoint_meta.update({
                "requested_method": "window",
                "selected_method": "window",
                "selected_penalty": cfg.changepoint_penalty,
            })
        else:
            raise ValueError(
                f"Unknown changepoint method '{cfg.changepoint_method}'. "
                "Expected one of: auto, pelt, binseg, window"
            )
    except ImportError:
        logger.warning("ruptures not available, using adaptive cosine fallback")
        changepoints, cosine_meta = _detect_changepoints_cosine_adaptive(
            embeddings,
            target_min=target_cp_min,
            target_max=target_cp_max,
            target_ideal=target_cp_ideal,
        )
        changepoint_meta.update(cosine_meta)

    if len(changepoints) < max(0, int(cfg.min_changepoints)):
        fallback_n = max(int(cfg.min_changepoints), target_cp_ideal, 1)
        try:
            binseg_cps = _detect_changepoints_binseg_fixed(embeddings, n_bkps=fallback_n)
        except ImportError:
            binseg_cps = []
        if len(binseg_cps) > len(changepoints):
            changepoints = binseg_cps
            changepoint_meta["fallback_method"] = "binseg_fixed"
            changepoint_meta["fallback_n_bkps"] = fallback_n
            changepoint_meta["selected_method"] = "binseg"

    # Prevent over-fragmented raw boundaries from destabilizing chapter lengths.
    pre_prune_count = len(changepoints)
    changepoints = _prune_changepoints_by_time_gap(
        changepoints,
        units,
        min_gap_s=max(5.0, cfg.min_chapter_len_s * 0.35),
    )
    changepoint_meta["raw_count_before_gap_prune"] = pre_prune_count
    changepoint_meta["raw_count_after_gap_prune"] = len(changepoints)

    logger.info(
        "[chapters] Detected %d raw changepoints (method=%s, requested=%s, target=%d-%d, ideal=%d)",
        len(changepoints),
        changepoint_meta.get("selected_method", "unknown"),
        method,
        target_cp_min,
        target_cp_max,
        target_cp_ideal,
    )
    if len(changepoints) == 0:
        logger.warning(
            "[chapters] No semantic changepoints detected; chapter boundaries will rely on max length fallback"
        )

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
                    labels = _label_chapter_with_llm(chapter_text, cfg, llm_complete=llm_complete)

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
            "changepoint_auto_tune": cfg.changepoint_auto_tune,
            "min_changepoints": cfg.min_changepoints,
            "target_chapter_len_s": cfg.target_chapter_len_s,
            "target_chapters_min": cfg.target_chapters_min,
            "target_chapters_max": cfg.target_chapters_max,
            "target_chapters_ideal": cfg.target_chapters_ideal,
            "snap_to_silence_window_s": cfg.snap_to_silence_window_s,
            "llm_labeling": cfg.llm_labeling,
        },
        "changepoint_diagnostics": changepoint_meta,
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
