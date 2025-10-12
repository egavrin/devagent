"""Abstractions for LLM providers."""
from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Protocol, Sequence

import requests


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: List[Dict[str, Any]] | None = None

    def to_payload(self) -> Dict[str, Any]:
        payload = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        return payload


@dataclass
class StreamHooks:
    """Hooks for streaming responses."""
    on_start: Callable[[], None] | None = None
    on_chunk: Callable[[str], None] | None = None
    on_complete: Callable[[str], None] | None = None
    on_error: Callable[[Exception], None] | None = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 5.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.1
    retryable_status_codes: set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})


@dataclass
class ToolCall:
    """Single tool/function invocation requested by the model."""

    name: str
    arguments: Dict[str, Any]
    call_id: str | None = None


@dataclass
class ToolCallResult:
    """Parsed outcome of a tool-enabled completion."""

    calls: List[ToolCall]
    message_content: str | None = None
    raw_tool_calls: List[Dict[str, Any]] | None = None
    _raw_response: Dict[str, Any] | None = None


class LLMError(RuntimeError):
    """Raised when an LLM provider encounters an error."""


class LLMRateLimitError(LLMError):
    """Raised when the provider reports a rate limit condition."""


class LLMTimeoutError(LLMError):
    """Raised when a request times out before the provider responds."""


class LLMConnectionError(LLMError):
    """Raised when the client is unable to reach the provider."""


class LLMResponseError(LLMError):
    """Raised when the provider returns a malformed or error response."""


class LLMRetryExhaustedError(LLMError):
    """Raised when retry attempts are exhausted without success."""


class LLMClient(Protocol):
    """Protocol for chat-completion capable LLM clients."""

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: Dict[str, str] | None = None,
        response_format: Dict[str, Any] | None = None,
    ) -> str:
        """Complete a chat conversation."""
        ...

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: Dict[str, str] | None = None,
        hooks: StreamHooks | None = None,
    ) -> Iterable[str]:
        """Stream a chat conversation with optional hooks."""
        ...

    def configure_retry(self, retry_config: RetryConfig) -> None:
        """Configure retry behavior."""
        ...

    def configure_timeout(self, timeout: float) -> None:
        """Configure request timeout."""
        ...

    def invoke_tools(
        self,
        messages: Sequence[Message],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tool_choice: str | Dict[str, Any] | None = "auto",
        extra_headers: Dict[str, str] | None = None,
        response_format: Dict[str, Any] | None = None,
        parallel_tool_calls: bool = True,
    ) -> "ToolCallResult":
        """Run a chat completion with tool definitions and return parsed tool calls.

        Args:
            messages: Conversation history
            tools: Tool definitions in OpenAI format
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            tool_choice: Control which tool is called ("auto", "none", or specific tool)
            extra_headers: Additional HTTP headers
            response_format: Response format specification
            parallel_tool_calls: Enable parallel tool calling (default: True)

        Returns:
            ToolCallResult containing parsed tool calls and message content
        """
        ...


class HTTPChatLLMClient(LLMClient, ABC):
    """Common HTTP/JSON client functionality shared by provider implementations."""

    _COMPLETIONS_PATH = "/chat/completions"
    _SUPPORTS_PARALLEL_TOOL_CALLS = True  # Override in subclasses if not supported

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        model: str,
        *,
        base_url: str,
        timeout: float = 120.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._provider_name = provider_name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()

    def configure_retry(self, retry_config: RetryConfig) -> None:
        """Configure retry behavior."""
        self.retry_config = retry_config

    def configure_timeout(self, timeout: float) -> None:
        """Configure request timeout."""
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    def _request_url(self) -> str:
        return f"{self.base_url}{self._COMPLETIONS_PATH}"

    def _build_headers(self, extra_headers: Dict[str, str] | None = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _error_from_status(self, status_code: int, response_text: str) -> LLMError:
        message = f"{self._provider_name} API error {status_code}: {response_text}"
        if status_code == 429:
            return LLMRateLimitError(message)
        if status_code in {408, 504}:
            return LLMTimeoutError(message)
        if status_code in {502, 503}:
            return LLMConnectionError(message)
        return LLMResponseError(message)

    def _calculate_delay(self, attempt: int) -> float:
        base_delay = min(
            self.retry_config.max_delay,
            self.retry_config.initial_delay * (self.retry_config.backoff_multiplier ** (attempt - 1)),
        )
        if base_delay <= 0:
            return 0.0
        jitter_ratio = max(0.0, self.retry_config.jitter_ratio)
        if jitter_ratio == 0:
            return base_delay
        jitter_span = base_delay * jitter_ratio
        lower = max(0.0, base_delay - jitter_span)
        upper = base_delay + jitter_span
        return random.uniform(lower, upper)

    def _wrap_transport_error(self, exc: Exception) -> LLMError:
        if isinstance(exc, requests.Timeout):
            return LLMTimeoutError(f"{self._provider_name} request timed out: {exc}")
        return LLMConnectionError(f"{self._provider_name} connection failed: {exc}")

    def _decode_json(self, response: requests.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Invalid JSON response from {self._provider_name} API") from exc

    def _post(
        self,
        payload: Dict[str, Any],
        *,
        extra_headers: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        url = self._request_url()
        headers = self._build_headers(extra_headers)
        body = json.dumps(payload)
        last_error: LLMError | None = None

        for attempt in range(1, self.retry_config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=body,
                    timeout=self.timeout,
                )

                if response.status_code in self.retry_config.retryable_status_codes:
                    error = self._error_from_status(response.status_code, response.text)
                    last_error = error
                    if attempt == self.retry_config.max_retries:
                        raise LLMRetryExhaustedError(
                            f"{self._provider_name} request exhausted retries: {error}"
                        ) from error
                    time.sleep(self._calculate_delay(attempt))
                    continue

                if response.status_code >= 400:
                    raise self._error_from_status(response.status_code, response.text)

                return self._decode_json(response)

            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = self._wrap_transport_error(exc)
                if attempt == self.retry_config.max_retries:
                    raise last_error from exc
                time.sleep(self._calculate_delay(attempt))
            except requests.RequestException as exc:
                raise LLMResponseError(f"{self._provider_name} request failed: {exc}") from exc

        if last_error is None:
            message = f"{self._provider_name} request failed for an unknown reason"
            raise LLMRetryExhaustedError(message)
        raise LLMRetryExhaustedError(
            f"{self._provider_name} request failed after {self.retry_config.max_retries} attempts: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------

    @abstractmethod
    def _prepare_payload(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> Dict[str, Any]:
        """Return the provider-specific request payload."""

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: Dict[str, str] | None = None,
        response_format: Dict[str, Any] | None = None,
    ) -> str:
        payload = self._prepare_payload(messages, temperature, max_tokens)
        if response_format:
            payload["response_format"] = response_format
        data = self._post(payload, extra_headers=extra_headers)
        message = self._extract_choice_message(data, "chat response")
        content = message.get("content")
        if not content:
            return ""
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: Dict[str, str] | None = None,
        hooks: StreamHooks | None = None,
    ) -> Iterable[str]:
        payload = self._prepare_payload(messages, temperature, max_tokens)
        payload["stream"] = True

        url = self._request_url()
        headers = self._build_headers(extra_headers)
        accumulated: List[str] = []

        try:
            if hooks and hooks.on_start:
                hooks.on_start()
            with requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
                stream=True,
            ) as response:
                if response.status_code >= 400:
                    raise self._error_from_status(response.status_code, response.text)

                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[len("data: "):]
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    delta = self._parse_stream_delta(data)
                    if not delta:
                        continue
                    accumulated.append(delta)
                    if hooks and hooks.on_chunk:
                        hooks.on_chunk(delta)
                    yield delta
        except Exception as exc:
            if hooks and hooks.on_error and isinstance(exc, Exception):
                hooks.on_error(exc)
            raise
        else:
            if hooks and hooks.on_complete:
                hooks.on_complete("".join(accumulated))

    def invoke_tools(
        self,
        messages: Sequence[Message],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tool_choice: str | Dict[str, Any] | None = "auto",
        extra_headers: Dict[str, str] | None = None,
        response_format: Dict[str, Any] | None = None,
        parallel_tool_calls: bool = True,
    ) -> ToolCallResult:
        payload = self._prepare_payload(messages, temperature, max_tokens)
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
        # Only include parallel_tool_calls if provider supports it
        if self._SUPPORTS_PARALLEL_TOOL_CALLS:
            payload["parallel_tool_calls"] = parallel_tool_calls
        if response_format:
            payload["response_format"] = response_format
        data = self._post(payload, extra_headers=extra_headers)
        message = self._extract_choice_message(data, "tool call")
        tool_calls_raw = message.get("tool_calls") or []
        parsed_calls = self._parse_tool_calls(tool_calls_raw)
        return ToolCallResult(
            calls=parsed_calls,
            message_content=message.get("content"),
            raw_tool_calls=tool_calls_raw or None,
            _raw_response=data,
        )

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    def _extract_choice_message(self, data: Dict[str, Any], context: str) -> Dict[str, Any]:
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"Unexpected {self._provider_name} response structure for {context}"
            ) from exc

    def _parse_stream_delta(self, data: Dict[str, Any]) -> str | None:
        try:
            delta = data["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, AttributeError):
            return None
        if not delta:
            return None
        if isinstance(delta, str):
            return delta
        return str(delta)

    def _parse_tool_calls(self, tool_calls_raw: Iterable[Dict[str, Any]]) -> List[ToolCall]:
        parsed: List[ToolCall] = []
        for call in tool_calls_raw:
            function_data = call.get("function", {}) or {}
            name = function_data.get("name") or ""
            raw_args = function_data.get("arguments") or "{}"
            if isinstance(raw_args, dict):
                arguments = raw_args
            else:
                try:
                    arguments = json.loads(raw_args)
                except (TypeError, json.JSONDecodeError):
                    arguments = {}

            parsed.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    call_id=call.get("id"),
                )
            )
        return parsed


__all__ = [
    "Message",
    "LLMClient",
    "HTTPChatLLMClient",
    "LLMError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMRetryExhaustedError",
    "StreamHooks",
    "RetryConfig",
    "ToolCall",
    "ToolCallResult",
]
