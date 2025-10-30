"""OpenRouter API client implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import HTTPChatLLMClient, Message, RetryConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(HTTPChatLLMClient):
    """Chat-completions client for the OpenRouter API with retry and tool support."""

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
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self._provider_config:
            payload["provider"] = self._provider_config
        return payload


__all__ = ["DEFAULT_BASE_URL", "OpenRouterClient"]
