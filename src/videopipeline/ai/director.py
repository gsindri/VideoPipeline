"""AI Director for clip variant selection and metadata generation.

Uses a local LLM to intelligently select the best clip variant and
generate title, hook, description, and hashtags for each candidate.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..clip_variants import CandidateVariants, ClipVariant, load_clip_variants
from ..project import Project, get_project_data, save_json, update_project
from .llm_client import (
    LLMClient,
    LLMClientConfig,
    LLMClientError,
    LLMServerUnavailableError,
    create_llm_client,
)


@dataclass(frozen=True)
class DirectorConfig:
    """Configuration for AI Director."""
    enabled: bool = True
    engine: str = "llama_cpp_server"
    endpoint: str = "http://127.0.0.1:11435"
    model_name: str = "local-gguf-vulkan"
    timeout_s: float = 30.0
    max_tokens: int = 256
    temperature: float = 0.2
    platform: str = "shorts"  # shorts, tiktok, youtube
    # Fallback settings
    fallback_to_rules: bool = True
    # Auto-start server settings
    auto_start: bool = False
    server_path: str = "C:/llama.cpp/llama-server.exe"
    model_path: str = "C:/llama.cpp/models/qwen2.5-7b-instruct-q4_k_m.gguf"
    startup_timeout_s: float = 120.0
    auto_stop_idle_s: float = 600.0


@dataclass
class DirectorResult:
    """Result from AI Director for a single candidate."""
    candidate_rank: int
    best_variant_id: str
    reason: str
    title: str
    hook: str
    description: str
    hashtags: List[str]
    confidence: float
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_rank": self.candidate_rank,
            "best_variant_id": self.best_variant_id,
            "reason": self.reason,
            "title": self.title,
            "hook": self.hook,
            "description": self.description,
            "hashtags": self.hashtags,
            "confidence": round(self.confidence, 2),
            "used_fallback": self.used_fallback,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DirectorResult":
        return cls(
            candidate_rank=int(d.get("candidate_rank", 0)),
            best_variant_id=str(d.get("best_variant_id", "medium")),
            reason=str(d.get("reason", "")),
            title=str(d.get("title", "")),
            hook=str(d.get("hook", "")),
            description=str(d.get("description", "")),
            hashtags=list(d.get("hashtags", [])),
            confidence=float(d.get("confidence", 0.5)),
            used_fallback=bool(d.get("used_fallback", False)),
        )


# System prompt for the director
DIRECTOR_SYSTEM_PROMPT = """You are a video clip editor AI. Your job is to select the best clip variant and create engaging metadata for social media shorts.

Rules:
1. Choose the variant that best captures the highlight while keeping viewer attention
2. For shorts/TikTok: prefer shorter variants (16-30s) unless setup is crucial
3. For YouTube: can use longer variants if story is compelling
4. Title should be attention-grabbing, use caps for emphasis
5. Hook is shown at start of video (max 2-3 words)
6. Description should be brief and engaging
7. Include relevant hashtags for discoverability

Output ONLY valid JSON matching the schema exactly. No explanation text."""


def _build_director_prompt(
    candidate: CandidateVariants,
    platform: str,
    chat_summary: Optional[str] = None,
    score_info: Optional[Dict[str, float]] = None,
) -> str:
    """Build the prompt for the director LLM."""
    variants_info = []
    for v in candidate.variants:
        variants_info.append({
            "id": v.variant_id,
            "duration_s": round(v.duration_s, 1),
            "description": v.description,
            "setup_text": v.setup_text[:100] if v.setup_text else "",
            "payoff_text": v.payoff_text[:100] if v.payoff_text else "",
        })

    prompt_data = {
        "task": "Select best variant and generate metadata",
        "platform": platform,
        "candidate": {
            "rank": candidate.candidate_rank,
            "peak_time_s": round(candidate.candidate_peak_time_s, 1),
        },
        "variants": variants_info,
    }

    if chat_summary:
        prompt_data["chat_context"] = chat_summary

    if score_info:
        prompt_data["scores"] = score_info

    prompt = f"""Select the best clip variant and generate metadata for this highlight.

Input:
{json.dumps(prompt_data, indent=2)}

Output JSON schema:
{{
  "best_variant_id": "short|medium|long|setup_first|punchline_first|chat_centered",
  "reason": "Brief explanation for choice",
  "title": "Catchy title (max 60 chars)",
  "hook": "2-3 word hook for overlay",
  "description": "Brief engaging description",
  "hashtags": ["tag1", "tag2", "tag3"],
  "confidence": 0.0-1.0
}}

Respond with ONLY the JSON object:"""

    return prompt


def _generate_fallback_metadata(
    candidate: CandidateVariants,
    platform: str,
) -> DirectorResult:
    """Generate rule-based fallback metadata when LLM is unavailable."""
    # Select variant based on platform
    if platform in ("shorts", "tiktok"):
        preferred_ids = ["short", "punchline_first", "medium"]
    else:
        preferred_ids = ["medium", "long", "setup_first"]

    best_variant: Optional[ClipVariant] = None
    for vid in preferred_ids:
        best_variant = candidate.get_variant(vid)
        if best_variant:
            break

    if not best_variant and candidate.variants:
        best_variant = candidate.variants[0]

    if not best_variant:
        # Ultimate fallback
        return DirectorResult(
            candidate_rank=candidate.candidate_rank,
            best_variant_id="medium",
            reason="Default selection",
            title=f"Highlight #{candidate.candidate_rank}",
            hook="WATCH THIS",
            description="Check out this highlight!",
            hashtags=["gaming", "highlight", "shorts"],
            confidence=0.3,
            used_fallback=True,
        )

    # Generate title from payoff text
    title = "INSANE moment"
    if best_variant.payoff_text:
        # Use first few words of payoff
        words = best_variant.payoff_text.split()[:6]
        title = " ".join(words).upper()
        if len(title) > 50:
            title = title[:47] + "..."

    # Generate hook from common reaction words
    hook = "WATCH THIS"
    payoff_lower = (best_variant.payoff_text or "").lower()
    if any(w in payoff_lower for w in ["no way", "insane", "crazy"]):
        hook = "NO WAY"
    elif any(w in payoff_lower for w in ["let's go", "lets go", "yes"]):
        hook = "LET'S GO"
    elif any(w in payoff_lower for w in ["what", "wait", "huh"]):
        hook = "WAIT..."

    return DirectorResult(
        candidate_rank=candidate.candidate_rank,
        best_variant_id=best_variant.variant_id,
        reason=f"Rule-based: {best_variant.description}",
        title=title,
        hook=hook,
        description=f"Check out this {best_variant.variant_id} clip!",
        hashtags=["gaming", "streamer", "shorts"],
        confidence=0.4,
        used_fallback=True,
    )


def _parse_director_response(
    response: Dict[str, Any],
    candidate_rank: int,
) -> DirectorResult:
    """Parse and validate LLM response."""
    # Extract and validate fields with defaults
    best_variant_id = str(response.get("best_variant_id", "medium"))
    if best_variant_id not in ("short", "medium", "long", "setup_first", "punchline_first", "chat_centered"):
        best_variant_id = "medium"

    title = str(response.get("title", ""))[:60]
    hook = str(response.get("hook", "WATCH THIS"))[:20]
    description = str(response.get("description", ""))[:200]

    hashtags = response.get("hashtags", [])
    if isinstance(hashtags, list):
        hashtags = [str(h).lstrip("#") for h in hashtags[:10]]
    else:
        hashtags = ["gaming", "shorts"]

    confidence = float(response.get("confidence", 0.7))
    confidence = max(0.0, min(1.0, confidence))

    return DirectorResult(
        candidate_rank=candidate_rank,
        best_variant_id=best_variant_id,
        reason=str(response.get("reason", "")),
        title=title,
        hook=hook,
        description=description,
        hashtags=hashtags,
        confidence=confidence,
        used_fallback=False,
    )


class AIDirector:
    """AI Director for clip selection and metadata generation."""

    def __init__(
        self,
        cfg: DirectorConfig,
        cache_dir: Optional[Path] = None,
    ):
        self.cfg = cfg
        self._client: Optional[LLMClient] = None

        if cfg.enabled and cfg.engine == "llama_cpp_server":
            self._client = create_llm_client(
                endpoint=cfg.endpoint,
                cache_dir=cache_dir,
                timeout_s=cfg.timeout_s,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                model_name=cfg.model_name,
            )

    def is_available(self) -> bool:
        """Check if the AI director is available."""
        if not self.cfg.enabled or not self._client:
            return False
        return self._client.is_available()

    def process_candidate(
        self,
        candidate: CandidateVariants,
        *,
        chat_summary: Optional[str] = None,
        score_info: Optional[Dict[str, float]] = None,
    ) -> DirectorResult:
        """Process a single candidate and generate metadata.
        
        Args:
            candidate: Candidate with variants
            chat_summary: Optional summary of chat activity
            score_info: Optional score breakdown
            
        Returns:
            DirectorResult with variant selection and metadata
        """
        # If LLM not available, use fallback
        if not self._client or not self.is_available():
            if self.cfg.fallback_to_rules:
                return _generate_fallback_metadata(candidate, self.cfg.platform)
            raise LLMServerUnavailableError("LLM server not available and fallback disabled")

        # Build prompt
        prompt = _build_director_prompt(
            candidate,
            self.cfg.platform,
            chat_summary=chat_summary,
            score_info=score_info,
        )

        # Call LLM with fallback
        fallback = _generate_fallback_metadata(candidate, self.cfg.platform)

        try:
            response = self._client.complete(
                prompt,
                system_prompt=DIRECTOR_SYSTEM_PROMPT,
                json_mode=True,
            )
            return _parse_director_response(response, candidate.candidate_rank)
        except LLMClientError:
            if self.cfg.fallback_to_rules:
                return fallback
            raise


def compute_director_analysis(
    proj: Project,
    *,
    cfg: DirectorConfig,
    top_n: int = 25,
    on_progress: Optional[Callable[[float], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run AI Director on all clip variants.
    
    Persists:
      - analysis/ai_director.json
      - Updates candidates in project.json with chosen_variant and metadata
    """
    import logging
    log = logging.getLogger("videopipeline.ai")
    
    # Track computation time (for pipeline status timers)
    import time as _time
    start_time = _time.time()

    if on_progress:
        on_progress(0.05)

    # Auto-start LLM server if configured
    actual_endpoint = cfg.endpoint
    if cfg.auto_start and cfg.engine == "llama_cpp_server":
        try:
            from .llm_server import ensure_llm_server
            from pathlib import Path
            
            log.info(f"[director] Auto-starting LLM server (timeout={cfg.startup_timeout_s}s)...")
            if on_status:
                on_status("Starting LLM server...")
            
            started_endpoint = ensure_llm_server(
                server_path=Path(cfg.server_path),
                model_path=Path(cfg.model_path),
                port=int(cfg.endpoint.split(":")[-1]),
                auto_stop_after_idle_s=cfg.auto_stop_idle_s,
                startup_timeout_s=cfg.startup_timeout_s,
                on_status=on_status,
            )
            if started_endpoint:
                actual_endpoint = started_endpoint
                log.info(f"[director] LLM server ready at {actual_endpoint}")
            else:
                log.warning(f"[director] LLM server failed to start within {cfg.startup_timeout_s}s")
        except Exception as e:
            log.warning(f"[director] Auto-start failed: {e}")

    if on_progress:
        on_progress(0.1)

    # Load clip variants
    variants_list = load_clip_variants(proj)
    if not variants_list:
        raise ValueError("No clip variants found. Run clip variant analysis first.")

    # Create director with potentially updated endpoint
    cache_dir = proj.analysis_dir
    # Create a new config with the actual endpoint (may have been updated by auto-start)
    effective_cfg = DirectorConfig(
        enabled=cfg.enabled,
        engine=cfg.engine,
        endpoint=actual_endpoint,
        model_name=cfg.model_name,
        timeout_s=cfg.timeout_s,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        platform=cfg.platform,
        fallback_to_rules=cfg.fallback_to_rules,
        auto_start=False,  # Already handled
        server_path=cfg.server_path,
        model_path=cfg.model_path,
        startup_timeout_s=cfg.startup_timeout_s,
        auto_stop_idle_s=cfg.auto_stop_idle_s,
    )
    director = AIDirector(effective_cfg, cache_dir=cache_dir)

    # Check availability
    llm_available = director.is_available()

    if on_progress:
        on_progress(0.2)

    # Process candidates
    results: List[DirectorResult] = []
    candidates_to_process = variants_list[:top_n]
    total = len(candidates_to_process)

    proj_data = get_project_data(proj)
    orig_candidates = proj_data.get("analysis", {}).get("highlights", {}).get("candidates", [])

    # Build score lookup
    score_lookup: Dict[int, Dict[str, float]] = {}
    for c in orig_candidates:
        rank = c.get("rank", 0)
        score_lookup[rank] = c.get("breakdown", {})

    for i, cv in enumerate(candidates_to_process):
        score_info = score_lookup.get(cv.candidate_rank)

        result = director.process_candidate(
            cv,
            score_info=score_info,
        )
        results.append(result)

        if on_progress:
            on_progress(0.2 + 0.7 * ((i + 1) / total))

    # Calculate elapsed time
    elapsed_seconds = _time.time() - start_time
    generated_at = datetime.now(timezone.utc).isoformat()

    # Build payload
    director_path = proj.analysis_dir / "ai_director.json"
    payload = {
        "created_at": generated_at,
        "generated_at": generated_at,
        "elapsed_seconds": elapsed_seconds,
        "config": {
            "enabled": cfg.enabled,
            "engine": cfg.engine,
            "endpoint": cfg.endpoint,
            "model_name": cfg.model_name,
            "platform": cfg.platform,
            "temperature": cfg.temperature,
        },
        "llm_available": llm_available,
        "candidate_count": len(results),
        "fallback_count": sum(1 for r in results if r.used_fallback),
        "results": [r.to_dict() for r in results],
    }

    # Save ai_director.json
    save_json(director_path, payload)

    # Update candidates in project.json with AI metadata
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})

        # Update ai_director section
        d["analysis"]["ai_director"] = {
            "created_at": payload["created_at"],
            "generated_at": payload.get("generated_at", payload["created_at"]),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "config": payload["config"],
            "llm_available": llm_available,
            "candidate_count": len(results),
            "director_json": str(director_path.relative_to(proj.project_dir)),
        }

        # Update each candidate with AI metadata
        candidates = d.get("analysis", {}).get("highlights", {}).get("candidates", [])
        result_by_rank = {r.candidate_rank: r for r in results}

        for c in candidates:
            rank = c.get("rank", 0)
            if rank in result_by_rank:
                r = result_by_rank[rank]
                c["ai"] = {
                    "chosen_variant_id": r.best_variant_id,
                    "reason": r.reason,
                    "title": r.title,
                    "hook": r.hook,
                    "description": r.description,
                    "hashtags": r.hashtags,
                    "confidence": r.confidence,
                    "used_fallback": r.used_fallback,
                }

    update_project(proj, _upd)

    if on_progress:
        on_progress(1.0)

    return payload


def load_director_results(proj: Project) -> Optional[List[DirectorResult]]:
    """Load cached director results if available."""
    director_path = proj.analysis_dir / "ai_director.json"
    if not director_path.exists():
        return None

    data = json.loads(director_path.read_text(encoding="utf-8"))
    return [DirectorResult.from_dict(r) for r in data.get("results", [])]


def get_director_result_for_candidate(
    proj: Project,
    candidate_rank: int,
) -> Optional[DirectorResult]:
    """Get director result for a specific candidate."""
    results = load_director_results(proj)
    if not results:
        return None

    for r in results:
        if r.candidate_rank == candidate_rank:
            return r

    return None
