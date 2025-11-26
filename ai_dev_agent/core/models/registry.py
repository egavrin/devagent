"""
Model registry for context-aware token budgeting.

This module provides a centralized registry of LLM model specifications,
enabling accurate context window management based on the actual model being used.

Instead of using a static 100K token budget for all models, this registry
allows devagent to:
- Use the full context window for large models (Claude 200K, Gemini 1M)
- Prevent context overflow on smaller models (GPT-4 8K)
- Reserve appropriate headroom for model responses
- Track model capabilities (parallel tools, temperature support)

The registry supports provider-aware model lookup where the same model
(e.g., gpt-4o) can be accessed via different providers (OpenAI, OpenRouter).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default fallback for unknown models - conservative estimate
DEFAULT_CONTEXT_WINDOW = 100_000
DEFAULT_RESPONSE_HEADROOM = 4_000


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an LLM provider.

    Attributes:
        api_key: API key for authentication (can be None if set via env)
        base_url: API endpoint URL
    """

    base_url: str
    api_key: Optional[str] = None


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an LLM model's capabilities and limits.

    Attributes:
        context_window: Total context window size in tokens
        response_headroom: Tokens reserved for model output
        tokenizer: Tokenizer identifier for accurate token counting
        supports_tools: Whether the model supports tool/function calling
        supports_parallel_tools: Whether the model can call multiple tools at once
        supports_temperature: Whether the model accepts temperature parameter
        base_url: Default API base URL for this model/provider
        provider: Provider name (e.g., "openai", "anthropic", "openrouter")
    """

    context_window: int
    response_headroom: int
    tokenizer: str
    supports_tools: bool = True
    supports_parallel_tools: bool = True
    supports_temperature: bool = True
    base_url: Optional[str] = None
    provider: Optional[str] = None

    @property
    def effective_context(self) -> int:
        """Usable context after reserving headroom for response."""
        return self.context_window - self.response_headroom


# Provider configurations registry
PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(base_url="https://api.deepseek.com/v1"),
    "openai": ProviderConfig(base_url="https://api.openai.com/v1"),
    "anthropic": ProviderConfig(base_url="https://api.anthropic.com/v1"),
    "openrouter": ProviderConfig(base_url="https://openrouter.ai/api/v1"),
}


# ============================================================
# PROVIDER PRE-CONFIGURATIONS
# ============================================================

# DeepSeek Native API (api.deepseek.com)
DEEPSEEK_MODELS: dict[str, ModelSpec] = {
    "deepseek-chat": ModelSpec(128_000, 4_000, "deepseek", supports_parallel_tools=False),
    "deepseek-coder": ModelSpec(128_000, 4_000, "deepseek", supports_parallel_tools=False),
    "deepseek-reasoner": ModelSpec(
        64_000, 8_000, "deepseek", supports_temperature=False, supports_parallel_tools=False
    ),
}

# OpenRouter with OpenAI models
OPENROUTER_OPENAI_MODELS: dict[str, ModelSpec] = {
    "openai/gpt-4o": ModelSpec(128_000, 16_000, "o200k_base"),
    "openai/gpt-4o-mini": ModelSpec(128_000, 16_000, "o200k_base"),
    "openai/gpt-4o-2024-11-20": ModelSpec(128_000, 16_000, "o200k_base"),
    "openai/gpt-4o-2024-08-06": ModelSpec(128_000, 16_000, "o200k_base"),
    "openai/gpt-4-turbo": ModelSpec(128_000, 4_000, "cl100k_base"),
    "openai/gpt-4-turbo-preview": ModelSpec(128_000, 4_000, "cl100k_base"),
    "openai/gpt-4": ModelSpec(8_192, 1_000, "cl100k_base"),
    "openai/gpt-4-32k": ModelSpec(32_768, 4_000, "cl100k_base"),
    "openai/gpt-3.5-turbo": ModelSpec(16_385, 4_000, "cl100k_base"),
    "openai/o1": ModelSpec(200_000, 100_000, "o200k_base", supports_temperature=False),
    "openai/o1-mini": ModelSpec(128_000, 65_000, "o200k_base", supports_temperature=False),
    "openai/o1-preview": ModelSpec(128_000, 32_000, "o200k_base", supports_temperature=False),
    "openai/o3-mini": ModelSpec(200_000, 100_000, "o200k_base", supports_temperature=False),
}

# OpenRouter with Claude models
OPENROUTER_CLAUDE_MODELS: dict[str, ModelSpec] = {
    "anthropic/claude-3.5-sonnet": ModelSpec(200_000, 8_192, "claude"),
    "anthropic/claude-3.5-sonnet-20241022": ModelSpec(200_000, 8_192, "claude"),
    "anthropic/claude-3-opus": ModelSpec(200_000, 4_096, "claude"),
    "anthropic/claude-3-opus-20240229": ModelSpec(200_000, 4_096, "claude"),
    "anthropic/claude-3-sonnet": ModelSpec(200_000, 4_096, "claude"),
    "anthropic/claude-3-haiku": ModelSpec(200_000, 4_096, "claude"),
    "anthropic/claude-3-haiku-20240307": ModelSpec(200_000, 4_096, "claude"),
    # Claude 2 legacy
    "anthropic/claude-2.1": ModelSpec(200_000, 4_096, "claude"),
    "anthropic/claude-2": ModelSpec(100_000, 4_096, "claude"),
}

# OpenRouter with other popular models
OPENROUTER_OTHER_MODELS: dict[str, ModelSpec] = {
    # Meta Llama 3.1
    "meta-llama/llama-3.1-405b-instruct": ModelSpec(128_000, 4_000, "llama"),
    "meta-llama/llama-3.1-70b-instruct": ModelSpec(128_000, 4_000, "llama"),
    "meta-llama/llama-3.1-8b-instruct": ModelSpec(128_000, 4_000, "llama"),
    # Meta Llama 3.2
    "meta-llama/llama-3.2-90b-vision-instruct": ModelSpec(128_000, 4_000, "llama"),
    "meta-llama/llama-3.2-11b-vision-instruct": ModelSpec(128_000, 4_000, "llama"),
    # Meta Llama 3.3
    "meta-llama/llama-3.3-70b-instruct": ModelSpec(128_000, 4_000, "llama"),
    # Mistral
    "mistralai/mistral-large": ModelSpec(128_000, 4_000, "mistral"),
    "mistralai/mistral-large-latest": ModelSpec(128_000, 4_000, "mistral"),
    "mistralai/mistral-medium": ModelSpec(32_000, 4_000, "mistral"),
    "mistralai/mistral-small": ModelSpec(32_000, 4_000, "mistral"),
    "mistralai/codestral-latest": ModelSpec(32_000, 4_000, "mistral"),
    "mistralai/mixtral-8x22b-instruct": ModelSpec(65_536, 4_000, "mistral"),
    "mistralai/mixtral-8x7b-instruct": ModelSpec(32_768, 4_000, "mistral"),
    # Google Gemini
    "google/gemini-2.0-flash-exp": ModelSpec(1_000_000, 8_000, "gemini"),
    "google/gemini-pro-1.5": ModelSpec(1_000_000, 8_000, "gemini"),
    "google/gemini-flash-1.5": ModelSpec(1_000_000, 8_000, "gemini"),
    "google/gemini-pro": ModelSpec(32_000, 4_000, "gemini"),
    # DeepSeek via OpenRouter
    "deepseek/deepseek-chat": ModelSpec(128_000, 4_000, "deepseek", supports_parallel_tools=False),
    "deepseek/deepseek-coder": ModelSpec(128_000, 4_000, "deepseek", supports_parallel_tools=False),
    "deepseek/deepseek-r1": ModelSpec(
        64_000, 8_000, "deepseek", supports_temperature=False, supports_parallel_tools=False
    ),
    # Qwen
    "qwen/qwen-2.5-72b-instruct": ModelSpec(131_072, 8_000, "qwen"),
    "qwen/qwen-2.5-coder-32b-instruct": ModelSpec(131_072, 8_000, "qwen"),
    "qwen/qwen-2-72b-instruct": ModelSpec(131_072, 8_000, "qwen"),
    # Cohere
    "cohere/command-r-plus": ModelSpec(128_000, 4_000, "cohere"),
    "cohere/command-r": ModelSpec(128_000, 4_000, "cohere"),
}

# Direct OpenAI API models (without openai/ prefix)
OPENAI_DIRECT_MODELS: dict[str, ModelSpec] = {
    "gpt-4o": ModelSpec(128_000, 16_000, "o200k_base"),
    "gpt-4o-mini": ModelSpec(128_000, 16_000, "o200k_base"),
    "gpt-4o-2024-11-20": ModelSpec(128_000, 16_000, "o200k_base"),
    "gpt-4o-2024-08-06": ModelSpec(128_000, 16_000, "o200k_base"),
    "gpt-4-turbo": ModelSpec(128_000, 4_000, "cl100k_base"),
    "gpt-4-turbo-preview": ModelSpec(128_000, 4_000, "cl100k_base"),
    "gpt-4": ModelSpec(8_192, 1_000, "cl100k_base"),
    "gpt-4-32k": ModelSpec(32_768, 4_000, "cl100k_base"),
    "gpt-3.5-turbo": ModelSpec(16_385, 4_000, "cl100k_base"),
    "gpt-3.5-turbo-16k": ModelSpec(16_385, 4_000, "cl100k_base"),
    "o1": ModelSpec(200_000, 100_000, "o200k_base", supports_temperature=False),
    "o1-mini": ModelSpec(128_000, 65_000, "o200k_base", supports_temperature=False),
    "o1-preview": ModelSpec(128_000, 32_000, "o200k_base", supports_temperature=False),
    "o3-mini": ModelSpec(200_000, 100_000, "o200k_base", supports_temperature=False),
}

# Direct Anthropic API models (without anthropic/ prefix)
ANTHROPIC_DIRECT_MODELS: dict[str, ModelSpec] = {
    "claude-3-5-sonnet-20241022": ModelSpec(200_000, 8_192, "claude"),
    "claude-3-5-sonnet-latest": ModelSpec(200_000, 8_192, "claude"),
    "claude-3-opus-20240229": ModelSpec(200_000, 4_096, "claude"),
    "claude-3-opus-latest": ModelSpec(200_000, 4_096, "claude"),
    "claude-3-sonnet-20240229": ModelSpec(200_000, 4_096, "claude"),
    "claude-3-haiku-20240307": ModelSpec(200_000, 4_096, "claude"),
    "claude-2.1": ModelSpec(200_000, 4_096, "claude"),
    "claude-2.0": ModelSpec(100_000, 4_096, "claude"),
}

# Combine all into master registry
MODEL_REGISTRY: dict[str, ModelSpec] = {}
MODEL_REGISTRY.update(DEEPSEEK_MODELS)
MODEL_REGISTRY.update(OPENROUTER_OPENAI_MODELS)
MODEL_REGISTRY.update(OPENROUTER_CLAUDE_MODELS)
MODEL_REGISTRY.update(OPENROUTER_OTHER_MODELS)
MODEL_REGISTRY.update(OPENAI_DIRECT_MODELS)
MODEL_REGISTRY.update(ANTHROPIC_DIRECT_MODELS)


def get_model_spec(model: str, strict: bool = False) -> ModelSpec:
    """Get model specification with fuzzy matching for model variants.

    Args:
        model: Model identifier (e.g., "gpt-4o", "anthropic/claude-3.5-sonnet")
        strict: If True, raise ValueError for unknown models. If False, return
                a default spec with conservative limits.

    Returns:
        ModelSpec for the requested model

    Raises:
        ValueError: If strict=True and model is not found in registry

    Examples:
        >>> spec = get_model_spec("gpt-4o")
        >>> spec.context_window
        128000
        >>> spec = get_model_spec("anthropic/claude-3.5-sonnet")
        >>> spec.effective_context
        191808
    """
    # Exact match
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]

    # Normalize model name (lowercase, strip whitespace)
    normalized = model.lower().strip()
    if normalized in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized]

    # Fuzzy match: check if model starts with or contains a known key
    # Sort by key length descending to prefer more specific matches
    for key in sorted(MODEL_REGISTRY.keys(), key=len, reverse=True):
        if model.startswith(key) or key in model:
            logger.debug(f"Fuzzy matched model '{model}' to registry key '{key}'")
            return MODEL_REGISTRY[key]

    # No match found
    if strict:
        available = ", ".join(sorted(MODEL_REGISTRY.keys())[:10])
        raise ValueError(
            f"Unknown model '{model}'. Add to MODEL_REGISTRY or use strict=False. "
            f"Available models include: {available}..."
        )

    # Return conservative default
    logger.warning(
        f"Model '{model}' not in registry, using default context window "
        f"({DEFAULT_CONTEXT_WINDOW} tokens). Consider adding it to MODEL_REGISTRY."
    )
    return ModelSpec(
        context_window=DEFAULT_CONTEXT_WINDOW,
        response_headroom=DEFAULT_RESPONSE_HEADROOM,
        tokenizer="heuristic",
    )


def get_effective_context(model: str, strict: bool = False) -> int:
    """Get the effective (usable) context window for a model.

    This is the total context window minus the response headroom.

    Args:
        model: Model identifier
        strict: If True, raise ValueError for unknown models

    Returns:
        Usable context size in tokens
    """
    spec = get_model_spec(model, strict=strict)
    return spec.effective_context


def register_model(name: str, spec: ModelSpec) -> None:
    """Register a custom model specification.

    Use this to add models not in the default registry.

    Args:
        name: Model identifier
        spec: Model specification
    """
    MODEL_REGISTRY[name] = spec
    logger.info(f"Registered custom model '{name}' with context window {spec.context_window}")


def list_models(provider: Optional[str] = None) -> list[str]:
    """List all registered model names, optionally filtered by provider prefix.

    Args:
        provider: Optional provider prefix to filter by (e.g., "openai/", "anthropic/")

    Returns:
        Sorted list of model names
    """
    if provider:
        return sorted(name for name in MODEL_REGISTRY if name.startswith(provider))
    return sorted(MODEL_REGISTRY.keys())


def get_provider_config(provider: str) -> Optional[ProviderConfig]:
    """Get configuration for a provider.

    Args:
        provider: Provider name (e.g., "openai", "deepseek", "openrouter")

    Returns:
        ProviderConfig or None if not found
    """
    return PROVIDER_REGISTRY.get(provider)


def register_provider(name: str, config: ProviderConfig) -> None:
    """Register or update a provider configuration.

    Args:
        name: Provider name
        config: Provider configuration
    """
    PROVIDER_REGISTRY[name] = config
    logger.info(f"Registered provider '{name}' with base_url {config.base_url}")


def list_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        Sorted list of provider names
    """
    return sorted(PROVIDER_REGISTRY.keys())


def load_providers_from_config(config_path: Path) -> dict[str, ProviderConfig]:
    """Load provider configs from a TOML config file.

    The config file should have a [providers] section:

        [providers.openai]
        api_key = "sk-..."
        base_url = "https://api.openai.com/v1"

        [providers.openrouter]
        api_key = "sk-or-..."
        base_url = "https://openrouter.ai/api/v1"

    Args:
        config_path: Path to the TOML config file

    Returns:
        Dictionary mapping provider names to ProviderConfig instances
    """
    if not config_path.is_file():
        return {}

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to load provider config from {config_path}: {e}")
        return {}

    providers_section = data.get("providers", {})
    if not isinstance(providers_section, dict):
        return {}

    result: dict[str, ProviderConfig] = {}

    for name, provider_dict in providers_section.items():
        if not isinstance(provider_dict, dict):
            logger.warning(f"Invalid provider config for '{name}' in {config_path}: expected dict")
            continue

        base_url = provider_dict.get("base_url")
        if not base_url:
            # Try to get default base_url from existing provider registry
            existing = PROVIDER_REGISTRY.get(name)
            if existing:
                base_url = existing.base_url
            else:
                logger.warning(f"Provider '{name}' missing base_url in {config_path}")
                continue

        result[name] = ProviderConfig(
            base_url=base_url,
            api_key=provider_dict.get("api_key"),
        )
        logger.debug(f"Loaded provider '{name}' from {config_path}")

    return result


# ============================================================
# FILE-BASED CONFIGURATION LOADING
# ============================================================


def load_models_from_config(config_path: Path) -> dict[str, ModelSpec]:
    """Load model specs from a TOML config file.

    The config file should have a [models] section with model definitions:

        [models.my-custom-llm]
        context_window = 256000
        response_headroom = 8000
        tokenizer = "claude"
        supports_tools = true
        supports_parallel_tools = true
        supports_temperature = true

    Args:
        config_path: Path to the TOML config file

    Returns:
        Dictionary mapping model names to ModelSpec instances
    """
    if not config_path.is_file():
        return {}

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to load model config from {config_path}: {e}")
        return {}

    models_section = data.get("models", {})
    if not isinstance(models_section, dict):
        logger.warning(
            f"Invalid [models] section in {config_path}: expected dict, got {type(models_section)}"
        )
        return {}

    result: dict[str, ModelSpec] = {}

    for name, spec_dict in models_section.items():
        if not isinstance(spec_dict, dict):
            logger.warning(f"Invalid model spec for '{name}' in {config_path}: expected dict")
            continue

        try:
            result[name] = ModelSpec(
                context_window=int(spec_dict.get("context_window", DEFAULT_CONTEXT_WINDOW)),
                response_headroom=int(
                    spec_dict.get("response_headroom", DEFAULT_RESPONSE_HEADROOM)
                ),
                tokenizer=str(spec_dict.get("tokenizer", "heuristic")),
                supports_tools=bool(spec_dict.get("supports_tools", True)),
                supports_parallel_tools=bool(spec_dict.get("supports_parallel_tools", True)),
                supports_temperature=bool(spec_dict.get("supports_temperature", True)),
            )
            logger.debug(f"Loaded model '{name}' from {config_path}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid model spec for '{name}' in {config_path}: {e}")
            continue

    return result


def load_model_from_file(file_path: Path) -> Optional[tuple[str, ModelSpec]]:
    """Load a single model spec from a TOML file.

    Each file defines one model. The model name is taken from the 'name' field
    in the file, or defaults to the filename (without .toml extension).

    Example file content:
        name = "my-custom-llm"
        context_window = 256000
        response_headroom = 8000
        tokenizer = "claude"

    Args:
        file_path: Path to the TOML file

    Returns:
        Tuple of (model_name, ModelSpec) or None if loading fails
    """
    if not file_path.is_file() or file_path.suffix != ".toml":
        return None

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]

    try:
        with file_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to load model file {file_path}: {e}")
        return None

    # Model name from 'name' field or filename
    name = data.get("name", file_path.stem)

    try:
        spec = ModelSpec(
            context_window=int(data.get("context_window", DEFAULT_CONTEXT_WINDOW)),
            response_headroom=int(data.get("response_headroom", DEFAULT_RESPONSE_HEADROOM)),
            tokenizer=str(data.get("tokenizer", "heuristic")),
            supports_tools=bool(data.get("supports_tools", True)),
            supports_parallel_tools=bool(data.get("supports_parallel_tools", True)),
            supports_temperature=bool(data.get("supports_temperature", True)),
        )
        logger.debug(f"Loaded model '{name}' from {file_path}")
        return (name, spec)
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid model spec in {file_path}: {e}")
        return None


def load_models_from_file(file_path: Path) -> dict[str, ModelSpec]:
    """Load model specs from a TOML file containing multiple model definitions.

    Supports two formats:
    1. Multi-model format with sections:
        provider = "openrouter"
        base_url = "https://openrouter.ai/api/v1"

        ["anthropic/claude-3.5-sonnet"]
        context_window = 200000
        response_headroom = 8192
        tokenizer = "claude"

    2. Single model format (for backward compatibility):
        name = "my-model"
        provider = "custom"
        context_window = 128000
        ...

    The provider field can be set:
    - At file level (applies to all models in the file)
    - At model level (overrides file-level provider)

    Args:
        file_path: Path to the TOML file

    Returns:
        Dictionary mapping model names to ModelSpec instances
    """
    if not file_path.is_file() or file_path.suffix != ".toml":
        return {}

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]

    try:
        with file_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to load model file {file_path}: {e}")
        return {}

    result: dict[str, ModelSpec] = {}

    # Get file-level defaults (applies to all models in this file)
    file_base_url = data.get("base_url")
    file_provider = data.get("provider")

    # Check if this is a single-model file (has context_window at root level)
    if "context_window" in data:
        name = data.get("name", file_path.stem)
        try:
            spec = ModelSpec(
                context_window=int(data.get("context_window", DEFAULT_CONTEXT_WINDOW)),
                response_headroom=int(data.get("response_headroom", DEFAULT_RESPONSE_HEADROOM)),
                tokenizer=str(data.get("tokenizer", "heuristic")),
                supports_tools=bool(data.get("supports_tools", True)),
                supports_parallel_tools=bool(data.get("supports_parallel_tools", True)),
                supports_temperature=bool(data.get("supports_temperature", True)),
                base_url=data.get("base_url") or file_base_url,
                provider=data.get("provider") or file_provider,
            )
            result[name] = spec
            logger.debug(f"Loaded model '{name}' from {file_path}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid model spec in {file_path}: {e}")
        return result

    # Multi-model format: each section is a model
    for name, spec_dict in data.items():
        if not isinstance(spec_dict, dict):
            continue  # Skip non-dict entries (like base_url, provider at file level)

        try:
            # Model-level values override file-level defaults
            model_base_url = spec_dict.get("base_url") or file_base_url
            model_provider = spec_dict.get("provider") or file_provider
            spec = ModelSpec(
                context_window=int(spec_dict.get("context_window", DEFAULT_CONTEXT_WINDOW)),
                response_headroom=int(
                    spec_dict.get("response_headroom", DEFAULT_RESPONSE_HEADROOM)
                ),
                tokenizer=str(spec_dict.get("tokenizer", "heuristic")),
                supports_tools=bool(spec_dict.get("supports_tools", True)),
                supports_parallel_tools=bool(spec_dict.get("supports_parallel_tools", True)),
                supports_temperature=bool(spec_dict.get("supports_temperature", True)),
                base_url=model_base_url,
                provider=model_provider,
            )
            result[name] = spec
            logger.debug(f"Loaded model '{name}' (provider={model_provider}) from {file_path}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid model spec for '{name}' in {file_path}: {e}")
            continue

    return result


def load_models_directory(dir_path: Path) -> dict[str, ModelSpec]:
    """Load all model specs from a directory of TOML files.

    Each .toml file can contain one or multiple model definitions.
    Files are processed in sorted order for deterministic behavior.

    Args:
        dir_path: Path to the directory containing model TOML files

    Returns:
        Dictionary mapping model names to ModelSpec instances
    """
    if not dir_path.is_dir():
        return {}

    result: dict[str, ModelSpec] = {}
    for toml_file in sorted(dir_path.glob("*.toml")):
        models = load_models_from_file(toml_file)
        result.update(models)

    if result:
        logger.debug(f"Loaded {len(result)} model(s) from {dir_path}")

    return result


def load_all_configs() -> None:
    """Load model and provider configs from all config file locations.

    Load order (later configs override earlier):
    0. Package-level models (models/*.toml in the devagent package) - lowest priority
    1. User home configs (~/.config/devagent/config.toml, ~/.devagent.toml) - [models] and [providers] sections
    2. User home model files (~/.config/devagent/models/*.toml)
    3. Project config (.devagent.toml in project root) - [models] and [providers] sections
    4. Project model files (models/*.toml) - highest priority

    This allows project-specific model and provider overrides to take precedence.
    The models/ directory is at project root (visible, version-controlled).
    """
    # Import here to avoid circular imports
    try:
        from ai_dev_agent.core.utils.config import (
            CONFIG_FILENAMES,
            DEFAULT_CONFIG_PATHS,
            find_config_in_parents,
        )
    except ImportError:
        logger.debug("Config utilities not available, skipping file-based model loading")
        return

    # 0. Load from package-level models directory (lowest priority)
    # This is the built-in model definitions shipped with devagent
    package_models_dir = Path(__file__).parent.parent.parent.parent / "models"
    if package_models_dir.is_dir():
        models = load_models_directory(package_models_dir)
        if models:
            MODEL_REGISTRY.update(models)
            logger.debug(f"Loaded {len(models)} built-in model(s) from {package_models_dir}")

    # 1. Load from user home configs - [models] and [providers] sections
    for path in DEFAULT_CONFIG_PATHS:
        models = load_models_from_config(path)
        if models:
            MODEL_REGISTRY.update(models)
            logger.info(f"Loaded {len(models)} model(s) from {path}")
        providers = load_providers_from_config(path)
        if providers:
            PROVIDER_REGISTRY.update(providers)
            logger.info(f"Loaded {len(providers)} provider(s) from {path}")

    # 2. Load from user home model files directory
    user_models_dir = Path.home() / ".config" / "devagent" / "models"
    models = load_models_directory(user_models_dir)
    if models:
        MODEL_REGISTRY.update(models)
        logger.info(f"Loaded {len(models)} model(s) from {user_models_dir}")

    # 3. Load from project config - [models] and [providers] sections
    try:
        project_config = find_config_in_parents(Path.cwd(), CONFIG_FILENAMES)
        if project_config:
            models = load_models_from_config(project_config)
            if models:
                MODEL_REGISTRY.update(models)
                logger.info(f"Loaded {len(models)} model(s) from {project_config}")
            providers = load_providers_from_config(project_config)
            if providers:
                PROVIDER_REGISTRY.update(providers)
                logger.info(f"Loaded {len(providers)} provider(s) from {project_config}")

            # 4. Load from project model files directory (highest priority)
            project_models_dir = project_config.parent / "models"
            # Skip if this is the same as package models dir (avoid double-loading)
            if project_models_dir != package_models_dir:
                models = load_models_directory(project_models_dir)
                if models:
                    MODEL_REGISTRY.update(models)
                    logger.info(f"Loaded {len(models)} model(s) from {project_models_dir}")
    except Exception as e:
        logger.debug(f"Could not load project config: {e}")

    # Also check models/ in cwd if no project config found
    cwd_models_dir = Path.cwd() / "models"
    if cwd_models_dir.is_dir() and cwd_models_dir != package_models_dir:
        models = load_models_directory(cwd_models_dir)
        if models:
            MODEL_REGISTRY.update(models)
            logger.info(f"Loaded {len(models)} model(s) from {cwd_models_dir}")


def reload_configs() -> None:
    """Reload model configs from all config file locations.

    This clears any previously loaded file-based configs and reloads them.
    Built-in model definitions are preserved.
    """
    load_all_configs()


# Load configs at module initialization
load_all_configs()
