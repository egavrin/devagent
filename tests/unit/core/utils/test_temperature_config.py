"""Tests for centralized temperature configuration."""

import os
from unittest.mock import patch

import pytest

from ai_dev_agent.core.utils.config import Settings, load_settings
from ai_dev_agent.core.utils.constants import (
    LLM_DEFAULT_TEMPERATURE,
    MODELS_WITHOUT_TEMPERATURE_SUPPORT,
)
from ai_dev_agent.providers.llm.base import supports_temperature


class TestLLMDefaultTemperature:
    """Test the LLM_DEFAULT_TEMPERATURE constant."""

    def test_default_temperature_is_zero(self):
        """Verify the default temperature is 0.0 for maximum reproducibility."""
        assert LLM_DEFAULT_TEMPERATURE == 0.0

    def test_default_temperature_type(self):
        """Verify the default temperature is a float."""
        assert isinstance(LLM_DEFAULT_TEMPERATURE, float)


class TestModelsWithoutTemperatureSupport:
    """Test the MODELS_WITHOUT_TEMPERATURE_SUPPORT constant."""

    def test_is_frozenset(self):
        """Verify it's an immutable frozenset."""
        assert isinstance(MODELS_WITHOUT_TEMPERATURE_SUPPORT, frozenset)

    def test_contains_openai_reasoning_models(self):
        """Verify OpenAI reasoning models are included."""
        assert "o1" in MODELS_WITHOUT_TEMPERATURE_SUPPORT
        assert "o1-mini" in MODELS_WITHOUT_TEMPERATURE_SUPPORT
        assert "o1-preview" in MODELS_WITHOUT_TEMPERATURE_SUPPORT
        assert "o3" in MODELS_WITHOUT_TEMPERATURE_SUPPORT
        assert "o3-mini" in MODELS_WITHOUT_TEMPERATURE_SUPPORT

    def test_contains_deepseek_reasoning_models(self):
        """Verify DeepSeek reasoning models are included."""
        assert "deepseek-r1" in MODELS_WITHOUT_TEMPERATURE_SUPPORT
        assert "deepseek-reasoner" in MODELS_WITHOUT_TEMPERATURE_SUPPORT


class TestSupportsTemperature:
    """Test the supports_temperature helper function."""

    def test_standard_models_support_temperature(self):
        """Standard chat models should support temperature."""
        assert supports_temperature("gpt-4") is True
        assert supports_temperature("gpt-4-turbo") is True
        assert supports_temperature("gpt-3.5-turbo") is True
        assert supports_temperature("claude-3-opus") is True
        assert supports_temperature("claude-3-sonnet") is True
        assert supports_temperature("deepseek-chat") is True
        assert supports_temperature("deepseek-coder") is True

    def test_openai_reasoning_models_dont_support_temperature(self):
        """OpenAI o1/o3 reasoning models don't support temperature."""
        assert supports_temperature("o1") is False
        assert supports_temperature("o1-mini") is False
        assert supports_temperature("o1-preview") is False
        assert supports_temperature("o3") is False
        assert supports_temperature("o3-mini") is False

    def test_deepseek_reasoning_models_dont_support_temperature(self):
        """DeepSeek reasoning models don't support temperature."""
        assert supports_temperature("deepseek-r1") is False
        assert supports_temperature("deepseek-reasoner") is False

    def test_case_insensitive_matching(self):
        """Model name matching should be case-insensitive."""
        assert supports_temperature("O1") is False
        assert supports_temperature("O1-Mini") is False
        assert supports_temperature("DEEPSEEK-R1") is False

    def test_model_name_with_prefix(self):
        """Model names with provider prefixes should work."""
        assert supports_temperature("openai/o1") is False
        assert supports_temperature("openai/o1-mini") is False
        assert supports_temperature("openrouter/deepseek-r1") is False
        assert supports_temperature("openrouter/gpt-4") is True


class TestSettingsTemperature:
    """Test temperature configuration in Settings."""

    def test_settings_default_temperature(self):
        """Settings should have temperature field defaulting to 0.0."""
        settings = Settings()
        assert hasattr(settings, "temperature")
        assert settings.temperature == 0.0

    def test_settings_temperature_from_env(self):
        """Temperature should be configurable via environment variable."""
        with patch.dict(os.environ, {"DEVAGENT_TEMPERATURE": "0.5"}):
            settings = load_settings()
            assert settings.temperature == 0.5

    def test_settings_temperature_from_env_zero(self):
        """Temperature=0 from env should work correctly."""
        with patch.dict(os.environ, {"DEVAGENT_TEMPERATURE": "0.0"}):
            settings = load_settings()
            assert settings.temperature == 0.0

    def test_settings_temperature_from_env_one(self):
        """Temperature=1.0 from env should work correctly."""
        with patch.dict(os.environ, {"DEVAGENT_TEMPERATURE": "1.0"}):
            settings = load_settings()
            assert settings.temperature == 1.0
