"""Helper functions for LLM client setup.

Provides a unified way to get an LLM completion function with auto-start support.
This reduces code duplication across the codebase.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..project import Project

log = logging.getLogger("videopipeline.ai")


def get_llm_complete_fn(
    ai_cfg: Dict[str, Any],
    cache_dir: Optional[Path] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> Optional[Callable[[str], str]]:
    """Get an LLM completion function with auto-start support.
    
    This is the unified way to set up LLM access across the codebase.
    It handles:
    - Checking if AI is enabled
    - Auto-starting the LLM server if configured
    - Creating the client and checking availability
    - Returning a simple completion function or None
    
    Args:
        ai_cfg: The AI director config dict (from profile.get("ai", {}).get("director", {}))
        cache_dir: Optional directory for response caching (e.g., proj.analysis_dir)
        on_status: Optional callback for status messages during server startup
    
    Returns:
        A function that takes a prompt string and returns JSON response string,
        or None if LLM is not available.
    
    Example:
        ai_cfg = ctx.profile.get("ai", {}).get("director", {})
        llm_complete = get_llm_complete_fn(ai_cfg, proj.analysis_dir)
        if llm_complete:
            response = llm_complete("Classify these tokens...")
    """
    if not ai_cfg.get("enabled", True):
        log.debug("[llm] AI disabled in config")
        return None
    
    try:
        from .llm_server import ensure_llm_server
        from .llm_client import create_llm_client
        
        endpoint = str(ai_cfg.get("endpoint", "http://127.0.0.1:11435"))
        
        # Auto-start server if configured
        if ai_cfg.get("auto_start", False):
            server_path = Path(ai_cfg.get("server_path", "C:/llama.cpp/llama-server.exe"))
            model_path = Path(ai_cfg.get("model_path", "C:/llama.cpp/models/qwen2.5-7b-instruct-q4_k_m.gguf"))
            auto_stop_s = ai_cfg.get("auto_stop_idle_s", 600.0)
            startup_timeout_s = float(ai_cfg.get("startup_timeout_s", 120.0))
            
            log.info(f"[llm] Auto-starting server (timeout={startup_timeout_s}s)...")
            if on_status:
                on_status("Starting LLM server...")
            
            started_endpoint = ensure_llm_server(
                server_path=server_path,
                model_path=model_path,
                port=int(endpoint.split(":")[-1]),
                auto_stop_after_idle_s=auto_stop_s,
                startup_timeout_s=startup_timeout_s,
                on_status=on_status,
            )
            if started_endpoint:
                endpoint = started_endpoint
                log.info(f"[llm] Server started at {endpoint}")
            else:
                log.warning(f"[llm] Server failed to start within {startup_timeout_s}s")
        
        # Create client
        # Default max_tokens is 2048 to handle highlight scoring (15 candidates × ~100 tokens)
        llm_client = create_llm_client(
            endpoint=endpoint,
            cache_dir=cache_dir,
            timeout_s=float(ai_cfg.get("timeout_s", 60)),
            max_tokens=int(ai_cfg.get("max_tokens", 2048)),
            temperature=float(ai_cfg.get("temperature", 0.2)),
        )
        
        if llm_client.is_available():
            log.info(f"[llm] Client available at {endpoint}")
            
            def complete_fn(prompt: str) -> str:
                resp = llm_client.complete(prompt, json_mode=True)
                # Convert response back to JSON string (handles dict, list, or str)
                if isinstance(resp, (dict, list)):
                    return json.dumps(resp)
                return str(resp)
            
            return complete_fn
        else:
            log.warning(f"[llm] Client NOT available at {endpoint}")
            return None
            
    except Exception as e:
        log.warning(f"[llm] Setup failed: {e}")
        return None
