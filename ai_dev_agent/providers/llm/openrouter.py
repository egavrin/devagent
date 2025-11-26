"""OpenRouter API client implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import HTTPChatLLMClient, Message, RetryConfig, supports_temperature

# Import model registry for model-aware context limits
try:
    from ai_dev_agent.core.models.registry import get_model_spec
except ImportError:
    get_model_spec = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Default context limit for unknown models (conservative)
DEFAULT_CONTEXT_LIMIT = 100_000


class OpenRouterClient(HTTPChatLLMClient):
    """Chat-completions client for the OpenRouter API with retry and tool support."""

    # Context limit - will be set per-model from registry
    _MAX_CONTEXT_TOKENS = DEFAULT_CONTEXT_LIMIT

    # Most OpenRouter models support parallel tool calls
    _SUPPORTS_PARALLEL_TOOL_CALLS = True

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        retry_config: RetryConfig | None = None,
        provider_only: Sequence[str] | None = None,
        provider_config: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            "OpenRouter",
            api_key,
            model,
            base_url=base_url,
            timeout=timeout,
            retry_config=retry_config,
        )
        self._default_headers = dict(default_headers or {})
        self._provider_config = self._merge_provider_config(provider_only, provider_config)

        # Get model-specific context limit from registry
        if get_model_spec is not None:
            try:
                spec = get_model_spec(model, strict=False)
                self._MAX_CONTEXT_TOKENS = spec.effective_context
                self._SUPPORTS_PARALLEL_TOOL_CALLS = spec.supports_parallel_tools
                logger.debug(
                    f"OpenRouter client using model registry for {model}: "
                    f"context={spec.context_window}, headroom={spec.response_headroom}, "
                    f"parallel_tools={spec.supports_parallel_tools}"
                )
            except Exception:
                pass  # Use default values

    @staticmethod
    def _merge_provider_config(
        provider_only: Sequence[str] | None,
        provider_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = dict(provider_config or {})
        if provider_only:
            config = {**config, "only": list(provider_only)}
        return config

    def _build_headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        headers = super()._build_headers()
        headers.update(
            {
                "HTTP-Referer": "https://github.com/egavrin/devagent",
                "X-Title": "devagent",
            }
        )
        if self._default_headers:
            headers.update(self._default_headers)
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _prepare_payload(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_payload() for message in messages],
        }
        # Only include temperature if the model supports it
        # (e.g., o1/o3 reasoning models don't support temperature)
        if supports_temperature(self.model):
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self._provider_config:
            payload["provider"] = self._provider_config
        return payload


__all__ = ["DEFAULT_BASE_URL", "OpenRouterClient"]
