"""HTTP client for local LLM server (llama.cpp).

Provides a simple interface to call a local llama.cpp server over HTTP
with JSON output enforcement and response caching.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

# Import heartbeat for idle timer reset
try:
    from .llm_server import _write_heartbeat
except ImportError:
    _write_heartbeat = None  # type: ignore


@dataclass(frozen=True)
class LLMClientConfig:
    """Configuration for LLM client."""
    endpoint: str = "http://127.0.0.1:11435"
    timeout_s: float = 60.0
    max_tokens: int = 512
    temperature: float = 0.2
    model_name: str = "local-gguf"
    # Cache settings
    cache_enabled: bool = True
    cache_db_path: Optional[Path] = None


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMServerUnavailableError(LLMClientError):
    """Raised when the LLM server is not reachable."""
    pass


class LLMResponseError(LLMClientError):
    """Raised when the LLM response is invalid."""
    pass


def _compute_cache_key(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Compute a cache key for a prompt."""
    key_data = json.dumps({
        "prompt": prompt,
        "model": model_name,
        "temp": temperature,
        "max_tokens": max_tokens,
    }, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


class LLMResponseCache:
    """SQLite-based cache for LLM responses."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the cache database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt_hash TEXT,
                    model_name TEXT,
                    response_json TEXT,
                    created_at TEXT,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_cache_created
                ON llm_cache(created_at)
            """)
            conn.commit()
        finally:
            conn.close()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response by key."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "SELECT response_json FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
            if row:
                # Update hit count
                conn.execute(
                    "UPDATE llm_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (cache_key,),
                )
                conn.commit()
                return json.loads(row[0])
            return None
        finally:
            conn.close()

    def set(
        self,
        cache_key: str,
        prompt_hash: str,
        model_name: str,
        response: Dict[str, Any],
    ) -> None:
        """Store response in cache."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (cache_key, prompt_hash, model_name, response_json, created_at, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (
                    cache_key,
                    prompt_hash,
                    model_name,
                    json.dumps(response),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_old(self, max_age_days: int = 30) -> int:
        """Remove entries older than max_age_days."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "DELETE FROM llm_cache WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


def _extract_json_from_response(text: str) -> Any:
    """Extract JSON from LLM response text.
    
    Handles common cases like markdown code blocks, extra text, etc.
    Supports both JSON objects and arrays.
    """
    text = text.strip()

    # Try direct parse first (handles both objects and arrays)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code block
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON array anywhere in text (for list responses)
    bracket_match = re.search(r"\[[\s\S]*\]", text)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object anywhere in text
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise LLMResponseError(f"Could not extract JSON from response: {text[:200]}...")


class LLMClient:
    """HTTP client for local llama.cpp server."""

    def __init__(self, cfg: LLMClientConfig):
        self.cfg = cfg
        self._cache: Optional[LLMResponseCache] = None

        if cfg.cache_enabled and cfg.cache_db_path:
            self._cache = LLMResponseCache(cfg.cache_db_path)

    def is_available(self) -> bool:
        """Check if the LLM server is available."""
        try:
            # Try to hit the health endpoint
            health_url = f"{self.cfg.endpoint}/health"
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except Exception:
            # Also try v1/models for OpenAI-compatible endpoint
            try:
                models_url = f"{self.cfg.endpoint}/v1/models"
                req = urllib.request.Request(models_url, method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    return resp.status == 200
            except Exception:
                return False

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = True,
    ) -> Any:
        """Send a completion request to the LLM server.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Temperature override
            max_tokens: Max tokens override
            json_mode: If True, parse and return JSON; if False, return raw text
            
        Returns:
            Parsed JSON response dict (json_mode=True) or raw string (json_mode=False)
            
        Raises:
            LLMServerUnavailableError: If server is not reachable
            LLMResponseError: If response is invalid
        """
        temp = temperature if temperature is not None else self.cfg.temperature
        tokens = max_tokens if max_tokens is not None else self.cfg.max_tokens

        # Include json_mode and system_prompt in cache key to avoid cross-contamination
        cache_key = _compute_cache_key(
            prompt + f"__json={json_mode}__sys={system_prompt or ''}",
            self.cfg.model_name, temp, tokens
        )
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                # Handle text responses cached with __text__ wrapper
                if not json_mode and isinstance(cached, dict) and "__text__" in cached:
                    return cached["__text__"]
                return cached

        # Build request body (OpenAI-compatible format)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": False,
        }

        # Add JSON mode hint if supported
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        # Send request
        url = f"{self.cfg.endpoint}/v1/chat/completions"
        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise LLMServerUnavailableError(f"LLM server unavailable: {e}") from e
        except TimeoutError as e:
            raise LLMServerUnavailableError(f"LLM server timeout: {e}") from e

        # Extract content from response
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMResponseError(f"Invalid response structure: {e}") from e

        # Update heartbeat to reset idle timer (keeps server alive)
        if _write_heartbeat:
            _write_heartbeat()

        # If caller wants raw text, return it directly
        if not json_mode:
            if self._cache:
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
                # Wrap text in dict for cache storage
                self._cache.set(cache_key, prompt_hash, self.cfg.model_name, {"__text__": content})
            return content

        # JSON mode: parse JSON from content
        result = _extract_json_from_response(content)

        # Cache the result
        if self._cache:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            self._cache.set(cache_key, prompt_hash, self.cfg.model_name, result)

        return result

    def complete_with_fallback(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        fallback: Dict[str, Any],
        **kwargs,
    ) -> tuple[Dict[str, Any], bool]:
        """Complete with fallback on error.
        
        Returns:
            Tuple of (response_dict, used_fallback)
        """
        try:
            result = self.complete(prompt, system_prompt=system_prompt, **kwargs)
            return result, False
        except (LLMServerUnavailableError, LLMResponseError):
            return fallback, True


def create_llm_client(
    endpoint: str = "http://127.0.0.1:11435",
    *,
    cache_dir: Optional[Path] = None,
    **kwargs,
) -> LLMClient:
    """Create an LLM client with sensible defaults.
    
    Args:
        endpoint: LLM server endpoint
        cache_dir: Directory for cache database
        **kwargs: Additional config options
        
    Returns:
        Configured LLMClient instance
    """
    cache_path = None
    if cache_dir:
        cache_path = cache_dir / "llm_cache.sqlite"

    cfg = LLMClientConfig(
        endpoint=endpoint,
        cache_db_path=cache_path,
        **kwargs,
    )
    return LLMClient(cfg)
