"""Target bundles for analysis.

Bundles define sets of artifacts to produce for common use cases.
"""

from __future__ import annotations

from typing import Dict, Set


# Pre-download bundle: Quick analysis using audio and chat
# Run this while video is still downloading
BUNDLE_PRE_DOWNLOAD: Set[str] = {
    "audio_features",
    "audio_events",
    "reaction_audio",  # Prosody/arousal voice excitement signal
    "audio_vad",  # VAD for speech boundaries (robust even with game audio)
    "silence",
    "diarization",  # Optional speaker diarization (turn-taking, overlaps)
    "transcript",
    "sentences",
    "speech_features",
    "chat_features",
    "chat_boundaries",
    "chat_sync",
    "boundary_graph",  # Better early clip start/end snapping (can run partial)
    # Early highlights using available signals
    "highlights_scores",
    "highlights_candidates",
}

# Pre-download bundle without chat-derived tasks. Used when chat is not available yet
# (e.g., chat download still in progress) to avoid redundant "no chat" work.
BUNDLE_PRE_DOWNLOAD_NO_CHAT: Set[str] = BUNDLE_PRE_DOWNLOAD - {
    "chat_features",
    "chat_boundaries",
    "chat_sync",
}

# Full bundle: Complete analysis including video-based features
# Run this after video is fully downloaded
BUNDLE_FULL: Set[str] = BUNDLE_PRE_DOWNLOAD | {
    "motion_features",
    "scenes",
    "chapters",
    "boundary_graph",
    # Re-run candidates with full boundary graph
    "highlights_candidates",
    "variants",
    "director",  # AI packaging + best-variant selection
}

# Export bundle: Just enough for clip export (if analysis already exists)
BUNDLE_EXPORT: Set[str] = {
    "highlights_candidates",
    "variants",
    "director",  # AI packaging + best-variant selection
}

# Transcript bundle: Just transcript and sentences
BUNDLE_TRANSCRIPT: Set[str] = {
    "transcript",
    "sentences",
}


_BUNDLES: Dict[str, Set[str]] = {
    "pre_download": BUNDLE_PRE_DOWNLOAD,
    "pre_download_no_chat": BUNDLE_PRE_DOWNLOAD_NO_CHAT,
    "full": BUNDLE_FULL,
    "export": BUNDLE_EXPORT,
    "transcript": BUNDLE_TRANSCRIPT,
}


def get_bundle(name: str) -> Set[str]:
    """Get a bundle by name.
    
    Args:
        name: Bundle name ("pre_download", "full", "export", "transcript")
        
    Returns:
        Set of artifact names
        
    Raises:
        ValueError: If bundle not found
    """
    if name not in _BUNDLES:
        raise ValueError(f"Unknown bundle '{name}'. Available: {list(_BUNDLES.keys())}")
    return _BUNDLES[name].copy()


def list_bundles() -> Dict[str, Set[str]]:
    """List all available bundles.
    
    Returns:
        Dict mapping bundle name to artifact set
    """
    return {k: v.copy() for k, v in _BUNDLES.items()}
