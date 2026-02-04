"""Adaptive concurrency tuner for yt-dlp downloads.

Stores per-domain tuning state in a JSON file and adjusts fragment
concurrency based on throttling detection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..utils import utc_iso as _utc_iso


@dataclass
class DomainTuning:
    """Tuning state for a specific domain."""
    
    N: int = 16           # Current concurrency
    min_N: int = 2        # Minimum concurrency
    max_N: int = 32       # Maximum concurrency
    last_result: str = "unknown"  # "ok", "throttled", "failed", "unknown"
    last_updated: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "N": self.N,
            "min_N": self.min_N,
            "max_N": self.max_N,
            "last_result": self.last_result,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainTuning":
        return cls(
            N=data.get("N", 16),
            min_N=data.get("min_N", 2),
            max_N=data.get("max_N", 32),
            last_result=data.get("last_result", "unknown"),
            last_updated=data.get("last_updated", ""),
        )


def tuning_file_path() -> Path:
    """Get path to the yt-dlp tuning state file."""
    if os.name == "nt":
        base = os.getenv("APPDATA")
        if base:
            return Path(base) / "VideoPipeline" / "ytdlp_tuning.json"
        return Path.home() / "AppData" / "Roaming" / "VideoPipeline" / "ytdlp_tuning.json"
    return Path.home() / ".config" / "videopipeline" / "ytdlp_tuning.json"


def load_all_tuning() -> dict[str, DomainTuning]:
    """Load all domain tuning from state file."""
    path = tuning_file_path()
    if not path.exists():
        return {}
    
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {domain: DomainTuning.from_dict(d) for domain, d in data.items()}
    except Exception:
        return {}


def save_all_tuning(tuning: dict[str, DomainTuning]) -> None:
    """Save all domain tuning to state file."""
    path = tuning_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {domain: t.to_dict() for domain, t in tuning.items()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_domain_tuning(domain: str) -> DomainTuning:
    """Get tuning for a specific domain.
    
    Returns default tuning if domain not found.
    """
    tuning = load_all_tuning()
    
    if domain in tuning:
        return tuning[domain]
    
    # Apply domain-specific defaults
    domain_lower = domain.lower()
    if "twitch" in domain_lower:
        return DomainTuning(N=16, min_N=2, max_N=32)
    
    # Generic default: lower concurrency
    return DomainTuning(N=4, min_N=1, max_N=16)


def update_domain_tuning(domain: str, N: int, result: str) -> None:
    """Update tuning for a domain after a download attempt.
    
    Args:
        domain: The domain (e.g., "www.twitch.tv")
        N: The concurrency that was used
        result: One of "ok", "throttled", "failed"
    """
    tuning = load_all_tuning()
    
    if domain not in tuning:
        tuning[domain] = get_domain_tuning(domain)
    
    tuning[domain].N = N
    tuning[domain].last_result = result
    tuning[domain].last_updated = _utc_iso()
    
    save_all_tuning(tuning)


def calculate_backoff_n(current_n: int, min_n: int = 2) -> int:
    """Calculate backed-off concurrency after throttling.
    
    Halves the concurrency, respecting minimum.
    """
    return max(min_n, current_n // 2)


def looks_like_throttle(error: Exception) -> bool:
    """Check if an error looks like throttling/rate limiting.
    
    Args:
        error: The exception that occurred
    
    Returns:
        True if this looks like a throttling error
    """
    error_str = str(error).lower()
    
    throttle_indicators = [
        "429",
        "403",
        "too many requests",
        "rate limit",
        "throttl",
        "temporary failure",
        "fragment",
        "unable to download",
        "giving up",
        "retries",
        "connection reset",
        "timed out",
    ]
    
    return any(indicator in error_str for indicator in throttle_indicators)


def extract_domain(url: str) -> str:
    """Extract domain from URL for tuning lookup."""
    from urllib.parse import urlparse
    
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        return netloc if netloc else "unknown"
    except Exception:
        return "unknown"
