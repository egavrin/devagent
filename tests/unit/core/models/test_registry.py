"""Tests for model registry."""

import tempfile
from pathlib import Path

import pytest

from ai_dev_agent.core.models.registry import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_RESPONSE_HEADROOM,
    MODEL_REGISTRY,
    PROVIDER_REGISTRY,
    ModelSpec,
    ProviderConfig,
    get_effective_context,
    get_model_spec,
    get_provider_config,
    list_models,
    list_providers,
    load_model_from_file,
    load_models_directory,
    load_models_from_config,
    load_models_from_file,
    load_providers_from_config,
    register_model,
    register_provider,
)


class TestModelSpec:
    """Tests for ModelSpec dataclass."""

    def test_effective_context_calculation(self):
        """Test that effective_context properly subtracts headroom."""
        spec = ModelSpec(context_window=200_000, response_headroom=8_000, tokenizer="claude")
        assert spec.effective_context == 192_000

    def test_default_capabilities(self):
        """Test that default capabilities are True."""
        spec = ModelSpec(context_window=100_000, response_headroom=4_000, tokenizer="test")
        assert spec.supports_tools is True
        assert spec.supports_parallel_tools is True
        assert spec.supports_temperature is True

    def test_custom_capabilities(self):
        """Test setting custom capabilities."""
        spec = ModelSpec(
            context_window=64_000,
            response_headroom=8_000,
            tokenizer="deepseek",
            supports_parallel_tools=False,
            supports_temperature=False,
        )
        assert spec.supports_tools is True
        assert spec.supports_parallel_tools is False
        assert spec.supports_temperature is False

    def test_frozen_dataclass(self):
        """Test that ModelSpec is immutable."""
        spec = ModelSpec(context_window=100_000, response_headroom=4_000, tokenizer="test")
        with pytest.raises(AttributeError):
            spec.context_window = 200_000


class TestModelRegistry:
    """Tests for the model registry."""

    def test_registry_not_empty(self):
        """Test that registry is populated with models."""
        assert len(MODEL_REGISTRY) > 30  # We have 40+ models

    def test_deepseek_models_present(self):
        """Test DeepSeek models are registered."""
        assert "deepseek-chat" in MODEL_REGISTRY
        assert "deepseek-coder" in MODEL_REGISTRY
        assert "deepseek-reasoner" in MODEL_REGISTRY

    def test_openrouter_openai_models_present(self):
        """Test OpenRouter OpenAI models are registered."""
        assert "openai/gpt-4o" in MODEL_REGISTRY
        assert "openai/gpt-4" in MODEL_REGISTRY
        assert "openai/o1" in MODEL_REGISTRY

    def test_openrouter_claude_models_present(self):
        """Test OpenRouter Claude models are registered."""
        assert "anthropic/claude-3.5-sonnet" in MODEL_REGISTRY
        assert "anthropic/claude-3-opus" in MODEL_REGISTRY

    def test_direct_api_models_present(self):
        """Test direct API models (without provider prefix)."""
        assert "gpt-4o" in MODEL_REGISTRY
        assert "claude-3-5-sonnet-20241022" in MODEL_REGISTRY

    def test_deepseek_no_parallel_tools(self):
        """Test that DeepSeek models don't support parallel tools."""
        assert MODEL_REGISTRY["deepseek-chat"].supports_parallel_tools is False
        assert MODEL_REGISTRY["deepseek-coder"].supports_parallel_tools is False

    def test_reasoning_models_no_temperature(self):
        """Test that reasoning models don't support temperature."""
        assert MODEL_REGISTRY["openai/o1"].supports_temperature is False
        assert MODEL_REGISTRY["deepseek-reasoner"].supports_temperature is False


class TestGetModelSpec:
    """Tests for get_model_spec function."""

    def test_exact_match(self):
        """Test exact model name match."""
        spec = get_model_spec("gpt-4o")
        assert spec.context_window == 128_000

    def test_exact_match_with_prefix(self):
        """Test exact match with provider prefix."""
        spec = get_model_spec("anthropic/claude-3.5-sonnet")
        assert spec.context_window == 200_000

    def test_fuzzy_match_dated_version(self):
        """Test fuzzy matching for dated model versions."""
        # gpt-4o-2024-11-20 should match gpt-4o or the exact entry
        spec = get_model_spec("openai/gpt-4o-2024-05-13")
        assert spec.context_window == 128_000  # Should match gpt-4o family

    def test_unknown_model_non_strict(self):
        """Test unknown model returns default in non-strict mode."""
        spec = get_model_spec("unknown-model-xyz", strict=False)
        assert spec.context_window == DEFAULT_CONTEXT_WINDOW
        assert spec.response_headroom == DEFAULT_RESPONSE_HEADROOM
        assert spec.tokenizer == "heuristic"

    def test_unknown_model_strict(self):
        """Test unknown model raises error in strict mode."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_spec("unknown-model-xyz", strict=True)

    def test_claude_large_context(self):
        """Test Claude models have 200K context."""
        spec = get_model_spec("anthropic/claude-3.5-sonnet")
        assert spec.context_window == 200_000
        assert spec.effective_context == 200_000 - 8_192

    def test_gpt4_small_context(self):
        """Test original GPT-4 has 8K context."""
        spec = get_model_spec("gpt-4")
        assert spec.context_window == 8_192

    def test_gemini_million_context(self):
        """Test Gemini models have 1M context."""
        spec = get_model_spec("google/gemini-pro-1.5")
        assert spec.context_window == 1_000_000


class TestGetEffectiveContext:
    """Tests for get_effective_context function."""

    def test_effective_context(self):
        """Test effective context calculation."""
        effective = get_effective_context("anthropic/claude-3.5-sonnet")
        assert effective == 200_000 - 8_192

    def test_effective_context_strict(self):
        """Test strict mode raises for unknown models."""
        with pytest.raises(ValueError):
            get_effective_context("unknown-model", strict=True)


class TestRegisterModel:
    """Tests for register_model function."""

    def test_register_custom_model(self):
        """Test registering a custom model."""
        custom_spec = ModelSpec(
            context_window=500_000,
            response_headroom=10_000,
            tokenizer="custom",
        )
        register_model("my-custom-model", custom_spec)

        spec = get_model_spec("my-custom-model")
        assert spec.context_window == 500_000
        assert spec.tokenizer == "custom"

        # Cleanup
        del MODEL_REGISTRY["my-custom-model"]


class TestListModels:
    """Tests for list_models function."""

    def test_list_all_models(self):
        """Test listing all models."""
        models = list_models()
        assert len(models) > 30
        assert models == sorted(models)  # Should be sorted

    def test_list_openai_models(self):
        """Test filtering by OpenAI provider."""
        models = list_models(provider="openai/")
        assert all(m.startswith("openai/") for m in models)
        assert "openai/gpt-4o" in models

    def test_list_anthropic_models(self):
        """Test filtering by Anthropic provider."""
        models = list_models(provider="anthropic/")
        assert all(m.startswith("anthropic/") for m in models)
        assert "anthropic/claude-3.5-sonnet" in models

    def test_list_deepseek_via_openrouter(self):
        """Test listing DeepSeek models via OpenRouter."""
        models = list_models(provider="deepseek/")
        assert "deepseek/deepseek-chat" in models


class TestModelCapabilities:
    """Tests for model capability tracking."""

    def test_o1_models_high_headroom(self):
        """Test that o1 models have high response headroom for reasoning."""
        spec = get_model_spec("openai/o1")
        assert spec.response_headroom == 100_000  # 100K for extended reasoning

    def test_gpt4o_moderate_headroom(self):
        """Test GPT-4o has moderate headroom."""
        spec = get_model_spec("gpt-4o")
        assert spec.response_headroom == 16_000

    def test_tokenizer_assignment(self):
        """Test correct tokenizer assignment."""
        assert get_model_spec("gpt-4o").tokenizer == "o200k_base"
        assert get_model_spec("gpt-4").tokenizer == "cl100k_base"
        assert get_model_spec("anthropic/claude-3.5-sonnet").tokenizer == "claude"
        assert get_model_spec("deepseek-chat").tokenizer == "deepseek"


class TestLoadModelsFromConfig:
    """Tests for file-based model configuration loading."""

    def test_load_from_toml_file(self):
        """Test loading model specs from a TOML config file."""
        toml_content = """
[models.my-test-model]
context_window = 256000
response_headroom = 8000
tokenizer = "claude"
supports_tools = true
supports_parallel_tools = false
supports_temperature = true

[models."internal/custom-llm"]
context_window = 64000
response_headroom = 4000
tokenizer = "llama"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            models = load_models_from_config(config_path)

            assert len(models) == 2
            assert "my-test-model" in models
            assert "internal/custom-llm" in models

            spec = models["my-test-model"]
            assert spec.context_window == 256000
            assert spec.response_headroom == 8000
            assert spec.tokenizer == "claude"
            assert spec.supports_tools is True
            assert spec.supports_parallel_tools is False
            assert spec.supports_temperature is True

            spec2 = models["internal/custom-llm"]
            assert spec2.context_window == 64000
            assert spec2.tokenizer == "llama"
        finally:
            config_path.unlink()

    def test_load_partial_spec_uses_defaults(self):
        """Test that missing fields use default values."""
        toml_content = """
[models.partial-model]
context_window = 50000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            models = load_models_from_config(config_path)

            spec = models["partial-model"]
            assert spec.context_window == 50000
            assert spec.response_headroom == DEFAULT_RESPONSE_HEADROOM
            assert spec.tokenizer == "heuristic"
            assert spec.supports_tools is True
            assert spec.supports_parallel_tools is True
            assert spec.supports_temperature is True
        finally:
            config_path.unlink()

    def test_load_nonexistent_file_returns_empty(self):
        """Test that loading from nonexistent file returns empty dict."""
        models = load_models_from_config(Path("/nonexistent/path/config.toml"))
        assert models == {}

    def test_load_file_without_models_section(self):
        """Test that loading from file without [models] section returns empty dict."""
        toml_content = """
provider = "deepseek"
model = "deepseek-chat"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            models = load_models_from_config(config_path)
            assert models == {}
        finally:
            config_path.unlink()

    def test_load_invalid_toml_returns_empty(self):
        """Test that loading invalid TOML returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("this is not valid toml {{{")
            config_path = Path(f.name)

        try:
            models = load_models_from_config(config_path)
            assert models == {}
        finally:
            config_path.unlink()

    def test_load_skips_invalid_model_specs(self):
        """Test that invalid model specs are skipped but valid ones load."""
        toml_content = """
[models.valid-model]
context_window = 100000
response_headroom = 4000
tokenizer = "test"

[models.invalid-model]
# Not a dict, just a string
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            models = load_models_from_config(config_path)
            # Valid model should be loaded
            assert "valid-model" in models
            assert models["valid-model"].context_window == 100000
        finally:
            config_path.unlink()

    def test_config_overrides_registry(self):
        """Test that config file models can be added to the registry."""
        toml_content = """
[models.config-test-model]
context_window = 999999
response_headroom = 1000
tokenizer = "custom"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            models = load_models_from_config(config_path)
            # Manually register it
            MODEL_REGISTRY.update(models)

            # Now it should be findable via get_model_spec
            spec = get_model_spec("config-test-model")
            assert spec.context_window == 999999
            assert spec.tokenizer == "custom"

            # Cleanup
            del MODEL_REGISTRY["config-test-model"]
        finally:
            config_path.unlink()

    def test_load_with_all_capabilities_false(self):
        """Test loading model with all capabilities set to false."""
        toml_content = """
[models.limited-model]
context_window = 32000
response_headroom = 2000
tokenizer = "test"
supports_tools = false
supports_parallel_tools = false
supports_temperature = false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            models = load_models_from_config(config_path)
            spec = models["limited-model"]
            assert spec.supports_tools is False
            assert spec.supports_parallel_tools is False
            assert spec.supports_temperature is False
        finally:
            config_path.unlink()


class TestLoadModelFromFile:
    """Tests for loading individual model files."""

    def test_load_single_model_file(self):
        """Test loading a single model from a .toml file."""
        toml_content = """
name = "my-file-model"
context_window = 300000
response_headroom = 10000
tokenizer = "custom"
supports_tools = true
supports_parallel_tools = false
supports_temperature = true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            file_path = Path(f.name)

        try:
            result = load_model_from_file(file_path)
            assert result is not None
            name, spec = result
            assert name == "my-file-model"
            assert spec.context_window == 300000
            assert spec.response_headroom == 10000
            assert spec.tokenizer == "custom"
            assert spec.supports_parallel_tools is False
        finally:
            file_path.unlink()

    def test_load_model_name_from_filename(self):
        """Test that model name defaults to filename if not specified."""
        toml_content = """
context_window = 128000
response_headroom = 4000
tokenizer = "llama"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False, prefix="test-model-"
        ) as f:
            f.write(toml_content)
            file_path = Path(f.name)

        try:
            result = load_model_from_file(file_path)
            assert result is not None
            name, spec = result
            # Name should be the filename stem (without .toml)
            assert name == file_path.stem
            assert spec.context_window == 128000
        finally:
            file_path.unlink()

    def test_load_model_file_nonexistent(self):
        """Test that loading nonexistent file returns None."""
        result = load_model_from_file(Path("/nonexistent/model.toml"))
        assert result is None

    def test_load_model_file_non_toml(self):
        """Test that loading non-.toml file returns None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not a toml file")
            file_path = Path(f.name)

        try:
            result = load_model_from_file(file_path)
            assert result is None
        finally:
            file_path.unlink()

    def test_load_model_file_invalid_toml(self):
        """Test that invalid TOML returns None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("this is not { valid toml")
            file_path = Path(f.name)

        try:
            result = load_model_from_file(file_path)
            assert result is None
        finally:
            file_path.unlink()


class TestLoadModelsDirectory:
    """Tests for loading models from a directory."""

    def test_load_multiple_models_from_directory(self):
        """Test loading multiple model files from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Create model file 1
            (dir_path / "model-a.toml").write_text(
                """
name = "model-alpha"
context_window = 100000
response_headroom = 4000
tokenizer = "test-a"
"""
            )
            # Create model file 2
            (dir_path / "model-b.toml").write_text(
                """
name = "model-beta"
context_window = 200000
response_headroom = 8000
tokenizer = "test-b"
"""
            )
            # Create non-toml file (should be ignored)
            (dir_path / "readme.txt").write_text("This should be ignored")

            models = load_models_directory(dir_path)

            assert len(models) == 2
            assert "model-alpha" in models
            assert "model-beta" in models
            assert models["model-alpha"].context_window == 100000
            assert models["model-beta"].context_window == 200000

    def test_load_empty_directory(self):
        """Test loading from an empty directory returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = load_models_directory(Path(tmpdir))
            assert models == {}

    def test_load_nonexistent_directory(self):
        """Test loading from nonexistent directory returns empty dict."""
        models = load_models_directory(Path("/nonexistent/models"))
        assert models == {}

    def test_load_directory_with_invalid_files(self):
        """Test that invalid files are skipped but valid ones load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Valid model
            (dir_path / "valid.toml").write_text(
                """
context_window = 50000
response_headroom = 2000
tokenizer = "valid"
"""
            )
            # Invalid TOML
            (dir_path / "invalid.toml").write_text("not { valid toml")

            models = load_models_directory(dir_path)

            # Only valid model should be loaded
            assert len(models) == 1
            assert "valid" in models

    def test_directory_models_override_registry(self):
        """Test that directory-loaded models can be added to registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            (dir_path / "dir-test-model.toml").write_text(
                """
name = "dir-test-model"
context_window = 777777
response_headroom = 7777
tokenizer = "dir-test"
"""
            )

            models = load_models_directory(dir_path)
            MODEL_REGISTRY.update(models)

            spec = get_model_spec("dir-test-model")
            assert spec.context_window == 777777
            assert spec.tokenizer == "dir-test"

            # Cleanup
            del MODEL_REGISTRY["dir-test-model"]

    def test_load_order_alphabetical(self):
        """Test that files are loaded in alphabetical order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Create files in non-alphabetical order
            (dir_path / "z-model.toml").write_text(
                """
name = "same-name"
context_window = 111111
"""
            )
            (dir_path / "a-model.toml").write_text(
                """
name = "same-name"
context_window = 222222
"""
            )

            models = load_models_directory(dir_path)

            # Since both have same name, z-model (loaded last) should win
            assert models["same-name"].context_window == 111111


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_provider_config_creation(self):
        """Test creating a ProviderConfig."""
        config = ProviderConfig(base_url="https://api.example.com/v1")
        assert config.base_url == "https://api.example.com/v1"
        assert config.api_key is None

    def test_provider_config_with_api_key(self):
        """Test ProviderConfig with API key."""
        config = ProviderConfig(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
        )
        assert config.base_url == "https://api.example.com/v1"
        assert config.api_key == "sk-test-key"

    def test_provider_config_frozen(self):
        """Test that ProviderConfig is immutable."""
        config = ProviderConfig(base_url="https://api.example.com/v1")
        with pytest.raises(AttributeError):
            config.base_url = "https://other.com"


class TestProviderRegistry:
    """Tests for the provider registry."""

    def test_registry_has_default_providers(self):
        """Test that registry has default providers."""
        assert "deepseek" in PROVIDER_REGISTRY
        assert "openai" in PROVIDER_REGISTRY
        assert "anthropic" in PROVIDER_REGISTRY
        assert "openrouter" in PROVIDER_REGISTRY

    def test_deepseek_provider_config(self):
        """Test DeepSeek provider configuration."""
        config = PROVIDER_REGISTRY["deepseek"]
        assert config.base_url == "https://api.deepseek.com/v1"

    def test_openai_provider_config(self):
        """Test OpenAI provider configuration."""
        config = PROVIDER_REGISTRY["openai"]
        assert config.base_url == "https://api.openai.com/v1"

    def test_anthropic_provider_config(self):
        """Test Anthropic provider configuration."""
        config = PROVIDER_REGISTRY["anthropic"]
        assert config.base_url == "https://api.anthropic.com"

    def test_openrouter_provider_config(self):
        """Test OpenRouter provider configuration."""
        config = PROVIDER_REGISTRY["openrouter"]
        assert config.base_url == "https://openrouter.ai/api/v1"


class TestGetProviderConfig:
    """Tests for get_provider_config function."""

    def test_get_existing_provider(self):
        """Test getting an existing provider."""
        config = get_provider_config("openai")
        assert config is not None
        assert config.base_url == "https://api.openai.com/v1"

    def test_get_nonexistent_provider(self):
        """Test getting a provider that doesn't exist."""
        config = get_provider_config("nonexistent-provider")
        assert config is None


class TestRegisterProvider:
    """Tests for register_provider function."""

    def test_register_new_provider(self):
        """Test registering a new provider."""
        custom_config = ProviderConfig(
            base_url="https://custom.llm.com/api",
            api_key="custom-key",
        )
        register_provider("custom-provider-test", custom_config)

        config = get_provider_config("custom-provider-test")
        assert config is not None
        assert config.base_url == "https://custom.llm.com/api"
        assert config.api_key == "custom-key"

        # Cleanup
        del PROVIDER_REGISTRY["custom-provider-test"]

    def test_register_provider_overwrites_existing(self):
        """Test that registering a provider overwrites existing entry."""
        original_config = get_provider_config("openai")
        original_url = original_config.base_url

        new_config = ProviderConfig(base_url="https://new.openai.com/v1")
        register_provider("openai", new_config)

        updated = get_provider_config("openai")
        assert updated.base_url == "https://new.openai.com/v1"

        # Restore original
        register_provider("openai", ProviderConfig(base_url=original_url))


class TestListProviders:
    """Tests for list_providers function."""

    def test_list_all_providers(self):
        """Test listing all providers."""
        providers = list_providers()
        assert len(providers) >= 4
        assert providers == sorted(providers)  # Should be sorted
        assert "deepseek" in providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert "openrouter" in providers


class TestLoadProvidersFromConfig:
    """Tests for load_providers_from_config function."""

    def test_load_providers_from_toml(self):
        """Test loading providers from TOML config."""
        toml_content = """
[providers.custom-prov]
api_key = "sk-custom-api-key"
base_url = "https://custom.llm.com/v1"

[providers.another-prov]
api_key = "sk-another-key"
base_url = "https://another.api.com/v1"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            providers = load_providers_from_config(config_path)

            assert len(providers) == 2
            assert "custom-prov" in providers
            assert "another-prov" in providers

            assert providers["custom-prov"].api_key == "sk-custom-api-key"
            assert providers["custom-prov"].base_url == "https://custom.llm.com/v1"
            assert providers["another-prov"].api_key == "sk-another-key"
        finally:
            config_path.unlink()

    def test_load_providers_missing_file(self):
        """Test loading from nonexistent file returns empty dict."""
        providers = load_providers_from_config(Path("/nonexistent/config.toml"))
        assert providers == {}

    def test_load_providers_without_section(self):
        """Test loading from file without [providers] section."""
        toml_content = """
provider = "deepseek"
model = "deepseek-chat"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            providers = load_providers_from_config(config_path)
            assert providers == {}
        finally:
            config_path.unlink()

    def test_load_providers_uses_default_base_url(self):
        """Test that missing base_url falls back to existing provider default."""
        toml_content = """
[providers.openai]
api_key = "sk-test-key"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            providers = load_providers_from_config(config_path)
            assert "openai" in providers
            # Should use default base_url from existing provider registry
            assert providers["openai"].base_url == "https://api.openai.com/v1"
            assert providers["openai"].api_key == "sk-test-key"
        finally:
            config_path.unlink()

    def test_load_providers_skips_unknown_without_base_url(self):
        """Test that unknown providers without base_url are skipped."""
        toml_content = """
[providers.unknown-provider]
api_key = "sk-unknown-key"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            providers = load_providers_from_config(config_path)
            assert "unknown-provider" not in providers
        finally:
            config_path.unlink()

    def test_load_providers_invalid_toml(self):
        """Test loading invalid TOML returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("this is { not valid toml")
            config_path = Path(f.name)

        try:
            providers = load_providers_from_config(config_path)
            assert providers == {}
        finally:
            config_path.unlink()

    def test_load_providers_skips_invalid_entries(self):
        """Test that invalid provider entries are skipped."""
        toml_content = """
[providers.valid-provider]
api_key = "sk-valid"
base_url = "https://valid.api.com/v1"

[providers]
invalid = "not a dict"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            providers = load_providers_from_config(config_path)
            assert "valid-provider" in providers
        finally:
            config_path.unlink()


class TestModelSpecWithProvider:
    """Tests for ModelSpec provider field."""

    def test_model_spec_with_provider(self):
        """Test ModelSpec with provider field."""
        spec = ModelSpec(
            context_window=128_000,
            response_headroom=4_000,
            tokenizer="test",
            provider="openai",
        )
        assert spec.provider == "openai"

    def test_model_spec_with_base_url(self):
        """Test ModelSpec with base_url field."""
        spec = ModelSpec(
            context_window=128_000,
            response_headroom=4_000,
            tokenizer="test",
            base_url="https://custom.api.com/v1",
        )
        assert spec.base_url == "https://custom.api.com/v1"

    def test_model_spec_default_provider_is_none(self):
        """Test ModelSpec defaults to None provider."""
        spec = ModelSpec(context_window=100_000, response_headroom=4_000, tokenizer="test")
        assert spec.provider is None
        assert spec.base_url is None


class TestLoadModelsFromFileWithProvider:
    """Tests for load_models_from_file with provider field."""

    def test_load_single_model_with_provider(self):
        """Test loading a single model with provider field."""
        toml_content = """
name = "my-provider-model"
context_window = 128000
response_headroom = 4000
tokenizer = "test"
provider = "openai"
base_url = "https://api.openai.com/v1"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            file_path = Path(f.name)

        try:
            models = load_models_from_file(file_path)
            assert "my-provider-model" in models
            spec = models["my-provider-model"]
            assert spec.provider == "openai"
            assert spec.base_url == "https://api.openai.com/v1"
        finally:
            file_path.unlink()

    def test_load_multi_model_with_file_level_provider(self):
        """Test loading multiple models with file-level provider."""
        toml_content = """
provider = "openrouter"
base_url = "https://openrouter.ai/api/v1"

["model-a"]
context_window = 128000
response_headroom = 4000
tokenizer = "test-a"

["model-b"]
context_window = 200000
response_headroom = 8000
tokenizer = "test-b"
provider = "custom"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            file_path = Path(f.name)

        try:
            models = load_models_from_file(file_path)
            assert "model-a" in models
            assert "model-b" in models

            # model-a should inherit file-level provider
            assert models["model-a"].provider == "openrouter"
            assert models["model-a"].base_url == "https://openrouter.ai/api/v1"

            # model-b should have its own provider (overrides file-level)
            assert models["model-b"].provider == "custom"
        finally:
            file_path.unlink()

    def test_load_models_from_file_nonexistent(self):
        """Test loading from nonexistent file returns empty dict."""
        models = load_models_from_file(Path("/nonexistent/models.toml"))
        assert models == {}

    def test_load_models_from_file_non_toml(self):
        """Test loading from non-.toml file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not a toml file")
            file_path = Path(f.name)

        try:
            models = load_models_from_file(file_path)
            assert models == {}
        finally:
            file_path.unlink()

    def test_load_models_from_file_invalid_toml(self):
        """Test loading invalid TOML returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("this is { not valid toml")
            file_path = Path(f.name)

        try:
            models = load_models_from_file(file_path)
            assert models == {}
        finally:
            file_path.unlink()


class TestEdgeCases:
    """Additional edge case tests for improved coverage."""

    def test_get_model_spec_normalized_match(self):
        """Test model spec lookup with normalized name (lowercase with spaces)."""
        # Add a model with specific case
        register_model(
            "Test-Model-123",
            ModelSpec(
                context_window=100_000,
                response_headroom=4_000,
                tokenizer="test",
            ),
        )
        try:
            # Look up with different case
            spec = get_model_spec("test-model-123")
            assert spec.context_window == 100_000
        finally:
            del MODEL_REGISTRY["Test-Model-123"]

    def test_load_providers_invalid_section_type(self):
        """Test loading providers when [providers] is not a dict."""
        toml_content = """
providers = "not a dict"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            providers = load_providers_from_config(config_path)
            assert providers == {}
        finally:
            config_path.unlink()

    def test_load_models_invalid_section_type(self):
        """Test loading models when [models] is not a dict."""
        toml_content = """
models = "not a dict"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            models = load_models_from_config(config_path)
            assert models == {}
        finally:
            config_path.unlink()

    def test_load_models_invalid_spec_value(self):
        """Test loading models with invalid spec that causes TypeError."""
        toml_content = """
[models.invalid-model]
context_window = "not-a-number"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            # The int() call on "not-a-number" raises ValueError
            models = load_models_from_config(config_path)
            # Invalid model should be skipped
            assert "invalid-model" not in models
        finally:
            config_path.unlink()

    def test_load_model_from_file_with_invalid_spec_value(self):
        """Test load_model_from_file with invalid spec that causes TypeError."""
        toml_content = """
name = "bad-model"
context_window = "invalid"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            file_path = Path(f.name)

        try:
            result = load_model_from_file(file_path)
            assert result is None
        finally:
            file_path.unlink()

    def test_load_models_from_file_multi_model_with_invalid_entry(self):
        """Test load_models_from_file skips invalid entries in multi-model format."""
        toml_content = """
provider = "openrouter"

["valid-model"]
context_window = 128000
response_headroom = 4000
tokenizer = "test"

["invalid-model"]
context_window = "not-a-number"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            file_path = Path(f.name)

        try:
            models = load_models_from_file(file_path)
            assert "valid-model" in models
            assert "invalid-model" not in models
        finally:
            file_path.unlink()

    def test_load_providers_with_non_dict_provider_entry(self):
        """Test loading providers skips non-dict provider entries."""
        toml_content = """
[providers]
valid = { base_url = "https://valid.api.com/v1" }
invalid = "not a dict"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            providers = load_providers_from_config(config_path)
            assert "valid" in providers
            assert "invalid" not in providers
        finally:
            config_path.unlink()

    def test_load_models_with_non_dict_model_entry(self):
        """Test loading models skips non-dict model entries."""
        toml_content = """
[models]
valid = { context_window = 128000, response_headroom = 4000, tokenizer = "test" }
invalid = "not a dict"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            models = load_models_from_config(config_path)
            assert "valid" in models
            assert "invalid" not in models
        finally:
            config_path.unlink()
