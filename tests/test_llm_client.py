"""Tests for LLM client module."""

import json
import hashlib
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from videopipeline.ai.llm_client import (
    LLMClientConfig,
    LLMResponseCache,
    _compute_cache_key,
)


class TestLLMClientConfig:
    """Tests for LLMClientConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        cfg = LLMClientConfig()
        assert cfg.endpoint == "http://127.0.0.1:11435"
        assert cfg.timeout_s == 60.0
        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.2
        assert cfg.cache_enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        cfg = LLMClientConfig(
            endpoint="http://custom:8080",
            timeout_s=30.0,
            cache_enabled=False,
        )
        assert cfg.endpoint == "http://custom:8080"
        assert cfg.timeout_s == 30.0
        assert cfg.cache_enabled is False


class TestLLMResponseCache:
    """Tests for LLMResponseCache SQLite caching."""

    def test_cache_get_miss(self, tmp_path):
        """Test cache miss returns None."""
        cache = LLMResponseCache(tmp_path / "cache.db")
        
        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_set_and_get(self, tmp_path):
        """Test setting and getting cached values."""
        cache = LLMResponseCache(tmp_path / "cache.db")
        
        test_data = {"response": "Hello, world!", "tokens": 5}
        cache.set("test_key", "prompt_hash", "test-model", test_data)
        
        result = cache.get("test_key")
        assert result == test_data

    def test_cache_overwrite(self, tmp_path):
        """Test overwriting cached values."""
        cache = LLMResponseCache(tmp_path / "cache.db")
        
        cache.set("key", "hash1", "model", {"v": 1})
        cache.set("key", "hash2", "model", {"v": 2})
        
        result = cache.get("key")
        assert result == {"v": 2}

    def test_cache_persistence(self, tmp_path):
        """Test that cache persists across instances."""
        cache_path = tmp_path / "cache.db"
        
        # Write with first instance
        cache1 = LLMResponseCache(cache_path)
        cache1.set("persistent_key", "hash", "model", {"data": "value"})
        del cache1
        
        # Read with second instance
        cache2 = LLMResponseCache(cache_path)
        result = cache2.get("persistent_key")
        assert result == {"data": "value"}

    def test_cache_key_hashing(self, tmp_path):
        """Test that cache properly handles long keys."""
        cache = LLMResponseCache(tmp_path / "cache.db")
        
        # Long key
        long_key = "A" * 1000
        cache.set(long_key, "hash", "model", {"result": "test"})
        
        result = cache.get(long_key)
        assert result == {"result": "test"}

    def test_cache_complex_json(self, tmp_path):
        """Test caching complex JSON structures."""
        cache = LLMResponseCache(tmp_path / "cache.db")
        
        complex_data = {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}],
            "unicode": "こんにちは",
            "number": 3.14159,
            "boolean": True,
            "null": None,
        }
        cache.set("complex", "hash", "model", complex_data)
        
        result = cache.get("complex")
        assert result == complex_data


class TestComputeCacheKey:
    """Tests for cache key computation."""

    def test_key_deterministic(self):
        """Test that cache keys are deterministic."""
        prompt = "Test prompt"
        model = "test-model"
        temp = 0.7
        max_tokens = 100
        
        # Generate key multiple times
        keys = []
        for _ in range(5):
            keys.append(_compute_cache_key(prompt, model, temp, max_tokens))
        
        # All keys should be identical
        assert len(set(keys)) == 1

    def test_key_unique_for_different_inputs(self):
        """Test that different inputs produce different keys."""
        key1 = _compute_cache_key("prompt", "model", 0.7, 100)
        key2 = _compute_cache_key("prompt", "model", 0.8, 100)  # Different temp
        
        assert key1 != key2

    def test_key_unique_for_different_prompts(self):
        """Test that different prompts produce different keys."""
        key1 = _compute_cache_key("prompt1", "model", 0.7, 100)
        key2 = _compute_cache_key("prompt2", "model", 0.7, 100)
        
        assert key1 != key2
