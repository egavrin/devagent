"""OpenAI API client implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import HTTPChatLLMClient, Message, RetryConfig, supports_temperature

# Import model registry for context-aware limits
try:
    from ai_dev_agent.core.models.registry import get_model_spec
except ImportError:
    get_model_spec = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CONTEXT_LIMIT = 100_000


class OpenAIClient(HTTPChatLLMClient):
    """Chat-completions client for the OpenAI API with tool support."""

    _MAX_CONTEXT_TOKENS = DEFAULT_CONTEXT_LIMIT
    _SUPPORTS_PARALLEL_TOOL_CALLS = True

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
            "OpenAI",
            api_key,
            model,
            base_url=base_url,
            timeout=timeout,
            retry_config=retry_config,
        )

        if get_model_spec is not None:
            try:
                spec = get_model_spec(model, strict=False)
                self._MAX_CONTEXT_TOKENS = spec.effective_context
                self._SUPPORTS_PARALLEL_TOOL_CALLS = spec.supports_parallel_tools
                logger.debug(
                    "OpenAI client using model registry: context=%s headroom=%s parallel_tools=%s",
                    spec.context_window,
                    spec.response_headroom,
                    spec.supports_parallel_tools,
                )
            except Exception:
                logger.debug("OpenAI model registry lookup failed; using defaults")

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
        if supports_temperature(self.model):
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload


__all__ = ["DEFAULT_BASE_URL", "OpenAIClient"]
