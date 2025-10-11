"""Factory helpers for LLM providers."""
from __future__ import annotations

from .base import (
    HTTPChatLLMClient,
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
    Message,
    RetryConfig,
    StreamHooks,
    ToolCall,
    ToolCallResult,
)
from .deepseek import DeepSeekClient, DEFAULT_BASE_URL as DEEPSEEK_DEFAULT_BASE_URL
from .openrouter import OpenRouterClient, DEFAULT_BASE_URL as OPENROUTER_DEFAULT_BASE_URL

_PROVIDER_MAP = {
    "deepseek": {
        "client": DeepSeekClient,
        "default_base_url": DEEPSEEK_DEFAULT_BASE_URL,
    },
    "openrouter": {
        "client": OpenRouterClient,
        "default_base_url": OPENROUTER_DEFAULT_BASE_URL,
    },
    "cerebras": {
        "client": OpenRouterClient,
        "default_base_url": OPENROUTER_DEFAULT_BASE_URL,
        "init_kwargs": {"provider_only": ("Cerebras",)},
    },
}


def create_client(
    provider: str,
    api_key: str,
    model: str,
    base_url: str | None = None,
    **provider_kwargs,
) -> LLMClient:
    key = provider.lower()
    try:
        provider_entry = _PROVIDER_MAP[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM provider: {provider}") from exc

    client_cls = provider_entry["client"]
    init_kwargs = dict(provider_entry.get("init_kwargs", {}))

    if key in {"openrouter", "cerebras"}:
        # Allow callers to override or extend provider-specific kwargs.
        init_kwargs.update(provider_kwargs)
    else:
        # Ignore provider-specific kwargs for other providers to avoid unexpected errors.
        init_kwargs.update({k: v for k, v in provider_kwargs.items() if k in {"timeout", "retry_config"}})

    effective_base_url = base_url or provider_entry.get("default_base_url")
    if effective_base_url:
        init_kwargs.setdefault("base_url", effective_base_url)

    return client_cls(api_key=api_key, model=model, **init_kwargs)


__all__ = [
    "create_client",
    "HTTPChatLLMClient",
    "LLMClient",
    "LLMError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMRetryExhaustedError",
    "Message",
    "StreamHooks",
    "RetryConfig",
    "ToolCall",
    "ToolCallResult",
    "DEEPSEEK_DEFAULT_BASE_URL",
    "OPENROUTER_DEFAULT_BASE_URL",
]
