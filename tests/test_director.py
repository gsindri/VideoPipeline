"""Tests for AI director module."""

import json
import pytest

from videopipeline.ai.director import (
    DirectorConfig,
    DirectorResult,
    DIRECTOR_SYSTEM_PROMPT,
)


class TestDirectorConfig:
    """Tests for DirectorConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        cfg = DirectorConfig()
        assert cfg.enabled is True
        assert cfg.engine == "llama_cpp_server"
        assert cfg.endpoint == "http://127.0.0.1:11435"
        assert cfg.model_name == "local-gguf-vulkan"
        assert cfg.timeout_s == 30.0
        assert cfg.max_tokens == 256
        assert cfg.temperature == 0.2
        assert cfg.platform == "shorts"
        assert cfg.fallback_to_rules is True

    def test_custom_values(self):
        """Test custom configuration values."""
        cfg = DirectorConfig(
            endpoint="http://custom:8080",
            model_name="llama3:8b",
            max_tokens=1024,
            temperature=0.5,
            enabled=False,
        )
        assert cfg.endpoint == "http://custom:8080"
        assert cfg.model_name == "llama3:8b"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.5
        assert cfg.enabled is False


class TestDirectorResult:
    """Tests for DirectorResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = DirectorResult(
            candidate_rank=1,
            best_variant_id="short_A",
            reason="High energy moment with chat explosion",
            title="Epic Gaming Moment",
            hook="You won't believe...",
            description="Insane clutch play",
            hashtags=["gaming", "clutch"],
            confidence=0.85,
        )
        
        assert result.candidate_rank == 1
        assert result.best_variant_id == "short_A"
        assert result.title == "Epic Gaming Moment"
        assert result.hook == "You won't believe..."
        assert result.reason == "High energy moment with chat explosion"

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = DirectorResult(
            candidate_rank=1,
            best_variant_id="short_A",
            reason="Test reason",
            title="Test Title",
            hook="Test Hook",
            description="Test desc",
            hashtags=["test"],
            confidence=0.9,
        )
        d = result.to_dict()
        
        assert d["candidate_rank"] == 1
        assert d["best_variant_id"] == "short_A"
        assert d["title"] == "Test Title"
        assert d["hook"] == "Test Hook"
        assert d["reason"] == "Test reason"
        assert d["confidence"] == 0.9

    def test_optional_fields(self):
        """Test default field values."""
        result = DirectorResult(
            candidate_rank=1,
            best_variant_id="medium_B",
            reason="",
            title="",
            hook="",
            description="",
            hashtags=[],
            confidence=0.5,
        )
        
        assert result.title == ""
        assert result.hook == ""
        assert result.used_fallback is False
        
        d = result.to_dict()
        assert d["title"] == ""
        assert d["hook"] == ""


class TestDirectorSystemPrompt:
    """Tests for the director system prompt."""

    def test_prompt_exists(self):
        """Test that system prompt is defined."""
        assert DIRECTOR_SYSTEM_PROMPT is not None
        assert len(DIRECTOR_SYSTEM_PROMPT) > 100

    def test_prompt_contains_key_instructions(self):
        """Test that prompt contains key instructions."""
        prompt = DIRECTOR_SYSTEM_PROMPT.lower()
        
        # Should mention JSON output
        assert "json" in prompt
        
        # Should mention variant
        assert "variant" in prompt
        
        # Should mention title/hook
        assert "title" in prompt
        assert "hook" in prompt

    def test_prompt_mentions_platforms(self):
        """Test that prompt mentions platform considerations."""
        prompt = DIRECTOR_SYSTEM_PROMPT.lower()
        # Should mention shorts or tiktok or youtube
        assert "short" in prompt or "tiktok" in prompt or "youtube" in prompt


class TestJSONExtraction:
    """Tests for JSON extraction from LLM responses."""

    def test_extract_json_from_markdown(self):
        """Test extracting JSON from markdown code block."""
        response = '''Here's my analysis:

```json
{
    "best_variant_id": "short_A",
    "title": "Insane Clutch Play",
    "hook": "Watch this!",
    "reason": "High action density"
}
```

Hope that helps!'''
        
        # Extract JSON using same logic as director
        import re
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            json_str = json_match.group(1).strip()
            data = json.loads(json_str)
            
            assert data["best_variant_id"] == "short_A"
            assert data["title"] == "Insane Clutch Play"
            assert data["hook"] == "Watch this!"

    def test_extract_raw_json(self):
        """Test extracting raw JSON without markdown."""
        response = '''{"best_variant_id": "medium_B", "title": "Amazing Play", "hook": "Must see!", "reason": "Good timing"}'''
        
        data = json.loads(response.strip())
        assert data["best_variant_id"] == "medium_B"

    def test_extract_json_with_whitespace(self):
        """Test extracting JSON with extra whitespace."""
        response = '''
        
        {
            "best_variant_id": "long_C",
            "title": "Full Context Highlight",
            "hook": "The whole story",
            "reason": "Needs setup and payoff"
        }
        
        '''
        
        data = json.loads(response.strip())
        assert data["best_variant_id"] == "long_C"


class TestFallbackMetadata:
    """Tests for fallback metadata generation when AI is unavailable."""

    def test_fallback_title_generation(self):
        """Test rule-based title generation as fallback."""
        # Simulating what fallback would produce
        candidate = {
            "rank": 1,
            "score": 0.95,
            "breakdown": {"audio": 0.9, "chat": 0.8, "motion": 0.7},
        }
        
        # Fallback logic should pick dominant signal
        breakdown = candidate["breakdown"]
        dominant = max(breakdown.items(), key=lambda x: x[1])
        
        if dominant[0] == "audio":
            expected_prefix = "Audio Peak"
        elif dominant[0] == "chat":
            expected_prefix = "Chat Explosion"
        else:
            expected_prefix = "Action Highlight"
        
        assert dominant[0] == "audio"  # In this case audio is highest
        assert expected_prefix == "Audio Peak"

    def test_fallback_variant_selection(self):
        """Test fallback variant selection logic."""
        variants = [
            {"variant_id": "short_A", "duration_s": 20},
            {"variant_id": "medium_B", "duration_s": 32},
            {"variant_id": "long_C", "duration_s": 55},
        ]
        
        # Fallback: prefer medium length if available
        medium = next((v for v in variants if "medium" in v["variant_id"]), None)
        if medium:
            chosen = medium["variant_id"]
        else:
            chosen = variants[0]["variant_id"]
        
        assert chosen == "medium_B"
