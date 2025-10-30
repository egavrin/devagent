"""DeepSeek API client implementation."""

from __future__ import annotations

import random  # - used for monkeypatch compatibility in tests
from typing import TYPE_CHECKING, Any

from .base import HTTPChatLLMClient, Message, RetryConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

# DeepSeek's published context limit is 131,072 tokens, but we use a conservative
# limit to account for: (1) estimation inaccuracy, (2) response generation headroom,
# (3) API overhead. This prevents "context length exceeded" errors.
DEEPSEEK_SAFE_CONTEXT_LIMIT = 120_000


class DeepSeekClient(HTTPChatLLMClient):
    """Chat-completions client for the DeepSeek API with retry and streaming support."""

    # DeepSeek does not support parallel_tool_calls parameter (as of 2025-01)
    _SUPPORTS_PARALLEL_TOOL_CALLS = False

    # DeepSeek's effective context limit (with safety margin)
    _MAX_CONTEXT_TOKENS = DEEPSEEK_SAFE_CONTEXT_LIMIT

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        super().__init__(
            "DeepSeek",
            api_key,
            model,
            base_url=base_url,
            timeout=timeout,
            retry_config=retry_config,
        )

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
        return payload


__all__ = ["DEFAULT_BASE_URL", "DeepSeekClient"]
