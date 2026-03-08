from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def normalize_llm_mode(value: Any) -> str:
    """Normalize llm_mode and validate allowed values."""
    mode = str(value or "local").strip().lower()
    if mode == "gondull":
        mode = "external_strict"
    if mode not in {"local", "external", "external_strict"}:
        raise ValueError("invalid_llm_mode")
    return mode


def llm_mode_uses_local(value: Any) -> bool:
    """Return True when the selected mode is allowed to invoke in-app LLMs."""
    return normalize_llm_mode(value) == "local"


def llm_mode_is_external(value: Any) -> bool:
    """Return True when AI decisions are expected to happen outside Studio."""
    return not llm_mode_uses_local(value)


def llm_mode_is_strict_external(value: Any) -> bool:
    """Return True when external AI completion is required before export."""
    return normalize_llm_mode(value) == "external_strict"


def build_dag_config(
    analysis_cfg: Optional[Mapping[str, Any]],
    *,
    section_overrides: Optional[Mapping[str, Any]] = None,
    include_chat: bool = True,
    include_audio_events: Optional[bool] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a normalized DAG config from profile analysis config + overrides."""
    cfg: Dict[str, Any] = dict(analysis_cfg or {})

    # Copy nested dict sections so call sites can mutate safely.
    for key, value in list(cfg.items()):
        if isinstance(value, dict):
            cfg[key] = dict(value)

    if section_overrides:
        for key, value in section_overrides.items():
            if isinstance(value, dict):
                cfg[key] = dict(value)
            else:
                cfg[key] = value

    if include_audio_events is None:
        include_audio_events = bool((cfg.get("audio_events", {}) or {}).get("enabled", True))

    cfg["include_chat"] = bool(include_chat)
    cfg["include_audio_events"] = bool(include_audio_events)

    if extra:
        for key, value in extra.items():
            cfg[key] = value

    return cfg


def apply_llm_mode_to_dag_config(
    dag_config: Mapping[str, Any],
    *,
    llm_mode: str,
) -> Dict[str, Any]:
    """Force config changes needed to ensure no local LLM calls occur."""
    mode = normalize_llm_mode(llm_mode)
    cfg: Dict[str, Any] = dict(dag_config or {})

    # Copy nested dict sections so changes stay local.
    for key, value in list(cfg.items()):
        if isinstance(value, dict):
            cfg[key] = dict(value)

    if mode == "local":
        return cfg

    highlights_cfg = dict(cfg.get("highlights", {}) or {})
    highlights_cfg["llm_semantic_enabled"] = False
    highlights_cfg["llm_filter_enabled"] = False
    cfg["highlights"] = highlights_cfg

    chapters_cfg = dict(cfg.get("chapters", {}) or {})
    chapters_cfg["llm_labeling"] = False
    cfg["chapters"] = chapters_cfg

    director_cfg = dict(cfg.get("director", {}) or {})
    director_cfg["enabled"] = False
    cfg["director"] = director_cfg

    # In external mode we intentionally avoid local LLM calls for chat-emote
    # learning. Keep chat feature extraction enabled, but disable strict failure
    # when llm_complete is absent.
    chat_cfg = dict(cfg.get("chat", {}) or {})
    chat_cfg["llm_strict"] = False
    cfg["chat"] = chat_cfg

    return cfg


def dag_config_needs_llm(dag_config: Mapping[str, Any]) -> bool:
    """Return True when DAG config includes any in-app LLM-consuming step."""
    cfg: Dict[str, Any] = dict(dag_config or {})
    highlights_cfg = dict(cfg.get("highlights", {}) or {})
    chapters_cfg = dict(cfg.get("chapters", {}) or {})
    director_cfg = dict(cfg.get("director", {}) or {})
    chat_cfg = dict(cfg.get("chat", {}) or {})

    needs_highlights = bool(highlights_cfg.get("llm_semantic_enabled", True)) or bool(
        highlights_cfg.get("llm_filter_enabled", False)
    )
    needs_chapters = bool(chapters_cfg.get("llm_labeling", True))
    needs_director = bool(director_cfg.get("enabled", False)) and bool(director_cfg.get("use_llm", True))
    needs_chat = bool(chat_cfg.get("llm_strict", False))

    return bool(needs_highlights or needs_chapters or needs_director or needs_chat)
