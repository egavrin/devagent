"""Anthropic API client implementation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from .base import (
    HTTPChatLLMClient,
    LLMError,
    Message,
    RetryConfig,
    ToolCallResult,
    supports_temperature,
)

# Import model registry for context-aware limits
try:
    from ai_dev_agent.core.models.registry import get_model_spec
except ImportError:
    get_model_spec = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_CONTEXT_LIMIT = 200_000
DEFAULT_MAX_TOKENS = 1_024
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient(HTTPChatLLMClient):
    """Messages API client for Anthropic with tool-call support."""

    _COMPLETIONS_PATH = "/v1/messages"
    _MAX_CONTEXT_TOKENS = DEFAULT_CONTEXT_LIMIT
    _SUPPORTS_PARALLEL_TOOL_CALLS = False  # Anthropic executes tools sequentially

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
            "Anthropic",
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
                    "Anthropic client using model registry: context=%s headroom=%s parallel_tools=%s",
                    spec.context_window,
                    spec.response_headroom,
                    spec.supports_parallel_tools,
                )
            except Exception:
                logger.debug("Anthropic model registry lookup failed; using defaults")

    def _build_headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _prepare_payload(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        system_prompts = [m.content for m in messages if m.role == "system" and m.content]
        system_prompt = "\n\n".join(system_prompts) if system_prompts else None

        converted: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue
            content_blocks: list[dict[str, Any]] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            converted.append(
                {
                    "role": msg.role,
                    "content": content_blocks or (msg.content or ""),
                }
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if supports_temperature(self.model):
            payload["temperature"] = temperature
        return payload

    def _request_url(self) -> str:
        # Avoid double /v1 prefix if base_url already contains it
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return super()._request_url()

    def _convert_tools(self, tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function") or {}
            name = function.get("name") or tool.get("name") or ""
            description = function.get("description") or tool.get("description") or ""
            params = function.get("parameters") or tool.get("parameters") or {}
            converted.append(
                {
                    "name": name,
                    "description": description,
                    "input_schema": params,
                }
            )
        return converted

    def _extract_choice_message(self, data: dict[str, Any], context: str) -> dict[str, Any]:
        content_blocks = data.get("content")
        if not isinstance(content_blocks, list):
            raise LLMError(
                f"Unexpected {self._provider_name} response structure for {context}: {data}"
            )

        text_parts = [
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ]
        tool_calls = []
        for block in content_blocks:
            if block.get("type") != "tool_use":
                continue
            arguments = block.get("input") or {}
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(arguments),
                    },
                }
            )

        return {
            "content": "\n".join(part for part in text_parts if part).strip() or None,
            "tool_calls": tool_calls,
        }

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload = self._prepare_payload(messages, temperature, max_tokens)
        if response_format:
            payload["response_format"] = response_format
        data = self._post(payload, extra_headers=extra_headers)
        message = self._extract_choice_message(data, "chat response")
        return (message.get("content") or "").strip()

    def invoke_tools(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        extra_headers: dict[str, str] | None = None,
        response_format: dict[str, Any] | None = None,
        parallel_tool_calls: bool = True,
    ) -> ToolCallResult:
        payload = self._prepare_payload(messages, temperature, max_tokens)
        payload["tools"] = self._convert_tools(tools)
        payload["tool_choice"] = tool_choice or "auto"
        if response_format:
            payload["response_format"] = response_format

        data = self._post(payload, extra_headers=extra_headers)
        message = self._extract_choice_message(data, "tool call")
        parsed_calls = self._parse_tool_calls(message.get("tool_calls", []))
        return ToolCallResult(
            calls=parsed_calls,
            message_content=message.get("content"),
            raw_tool_calls=message.get("tool_calls"),
            _raw_response=data,
        )

    def stream(  # pragma: no cover - streaming not yet supported for Anthropic client
        self,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        hooks=None,
    ):
        raise LLMError("Streaming is not supported for AnthropicClient")


__all__ = [
    "ANTHROPIC_VERSION",
    "AnthropicClient",
    "DEFAULT_BASE_URL",
    "DEFAULT_CONTEXT_LIMIT",
    "DEFAULT_MAX_TOKENS",
]
