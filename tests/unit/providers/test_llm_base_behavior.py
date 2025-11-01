"""Focused coverage for ai_dev_agent.providers.llm.base."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterable, List
from unittest.mock import Mock

import pytest
import requests

from ai_dev_agent.providers.llm.base import (
    HTTPChatLLMClient,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
    Message,
    RetryConfig,
    StreamHooks,
    ToolCall,
    ToolCallResult,
)


class _DummyClient(HTTPChatLLMClient):
    """Concrete subclass exposing protected helpers for testing."""

    def _prepare_payload(self, messages, temperature, max_tokens):
        return {
            "messages": [message.to_payload() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }


@pytest.fixture()
def client():
    return _DummyClient(
        provider_name="BaseLLM",
        api_key="token",
        model="demo",
        base_url="https://example.com",
        retry_config=RetryConfig(
            max_retries=2,
            initial_delay=0.0,
            max_delay=0.0,
            backoff_multiplier=1.0,
            jitter_ratio=0.0,
        ),
        timeout=5.0,
    )


def test_message_to_payload_includes_optional_fields():
    message = Message(
        role="tool",
        content="result",
        tool_call_id="call-123",
        tool_calls=[{"id": "call-123", "function": {"name": "find", "arguments": "{}"}}],
    )

    payload = message.to_payload()

    assert payload["role"] == "tool"
    assert payload["content"] == "result"
    assert payload["tool_call_id"] == "call-123"
    assert payload["tool_calls"][0]["function"]["name"] == "find"


def test_message_to_payload_omits_optional_keys_when_empty():
    payload = Message(role="user").to_payload()

    assert payload == {"role": "user"}


def test_tool_call_result_populates_primary_fields():
    call = ToolCall(name="search", arguments={"query": "*.py"}, call_id="call-1")
    result = ToolCallResult(
        calls=[call],
        message_content="Searching...",
        raw_tool_calls=[{"id": "call-1"}],
        _raw_response={"choices": []},
    )

    assert result.call_id == "call-1"
    assert result.name == "search"
    assert result.arguments == {"query": "*.py"}
    assert result.message_content == "Searching..."
    assert result.raw_tool_calls == [{"id": "call-1"}]


def test_tool_call_result_legacy_arguments_and_fallback_json():
    result = ToolCallResult(
        call_id="legacy",
        name="compile",
        arguments='{"target": "all"}',
        content="Command payload",
    )

    assert result.calls[0].name == "compile"
    assert result.arguments == {"target": "all"}
    assert result.content == "Command payload"


def test_tool_call_result_invalid_json_returns_empty_arguments():
    result = ToolCallResult(arguments="not-json", content="{}")

    assert result.arguments == {}
    assert result.content == "{}"


def test_tool_call_result_arguments_only_creates_placeholder_call():
    result = ToolCallResult(arguments='{"path": "."}')

    assert result.calls[0].name == ""
    assert result.arguments == {"path": "."}


@pytest.mark.parametrize(
    "arguments,fallback,expected",
    [
        ({"value": 1}, None, {"value": 1}),
        ('{"threshold": 0.9}', None, {"threshold": 0.9}),
        (None, '{"pattern": "*.py"}', {"pattern": "*.py"}),
        ('["value"]', None, {}),
        (None, '["value"]', {}),
        ("invalid", None, {}),
        (None, "invalid", {}),
    ],
)
def test_coerce_arguments_handles_multiple_inputs(arguments, fallback, expected):
    assert ToolCallResult._coerce_arguments(arguments, fallback) == expected


def test_configure_retry_and_timeout_update_runtime(client):
    new_retry = RetryConfig(max_retries=5, initial_delay=1.0)
    client.configure_retry(new_retry)
    client.configure_timeout(42.0)

    assert client.retry_config is new_retry
    assert client.timeout == 42.0


def test_build_headers_merges_extras(client):
    headers = client._build_headers({"X-Test": "true"})

    assert headers["Authorization"] == "Bearer token"
    assert headers["X-Test"] == "true"


@pytest.mark.parametrize(
    "exc,expected",
    [
        (requests.Timeout("slow"), LLMTimeoutError),
        (requests.ConnectionError("down"), LLMConnectionError),
        (OSError("no route"), LLMConnectionError),
        (ValueError("generic"), LLMConnectionError),
    ],
)
def test_wrap_transport_error_maps_known_types(client, exc, expected):
    error = client._wrap_transport_error(exc)
    assert isinstance(error, expected)
    assert "BaseLLM" in str(error)


def test_error_from_status_maps_gateway_and_timeout(client):
    timeout_error = client._error_from_status(504, "gateway timeout")
    upstream_error = client._error_from_status(503, "downstream")

    assert isinstance(timeout_error, LLMTimeoutError)
    assert isinstance(upstream_error, LLMConnectionError)


def test_calculate_delay_with_jitter(monkeypatch, client):
    client.retry_config.initial_delay = 1.0
    client.retry_config.backoff_multiplier = 2.0
    client.retry_config.max_delay = 10.0
    client.retry_config.jitter_ratio = 0.5

    captured = {}

    def fake_uniform(low: float, high: float) -> float:
        captured["low"] = low
        captured["high"] = high
        return high

    monkeypatch.setattr("ai_dev_agent.providers.llm.base.random.uniform", fake_uniform)

    delay = client._calculate_delay(2)

    assert delay == captured["high"]
    assert captured["low"] < captured["high"]


def test_decode_json_returns_payload(client):
    response = SimpleNamespace(json=lambda: {"ok": True})
    assert client._decode_json(response)["ok"] is True


def test_complete_trims_string_and_handles_non_string(monkeypatch, client):
    payloads: List[dict[str, Any]] = [
        {"choices": [{"message": {"content": "  trimmed  "}}]},
        {"choices": [{"message": {"content": 12345}}]},
    ]
    monkeypatch.setattr(client, "_post", Mock(side_effect=payloads))

    result1 = client.complete([Message(role="user", content="hi")])
    result2 = client.complete([Message(role="user", content="number")])

    assert result1 == "trimmed"
    assert result2 == "12345"


def test_complete_returns_empty_string_when_content_missing(monkeypatch, client):
    monkeypatch.setattr(
        client,
        "_post",
        Mock(return_value={"choices": [{"message": {"content": None}}]}),
    )

    assert client.complete([Message(role="user", content="hi")]) == ""


def test_post_with_zero_retries_raises_unknown_error(monkeypatch, client):
    client.retry_config.max_retries = 0

    monkeypatch.setattr(
        "ai_dev_agent.providers.llm.base.requests.post",
        Mock(side_effect=AssertionError("should not be called")),
    )

    with pytest.raises(LLMRetryExhaustedError) as exc_info:
        client._post({"messages": []})

    assert "unknown reason" in str(exc_info.value)


class _StreamResponse:
    def __init__(self, status_code: int, lines: Iterable[str], text: str = "") -> None:
        self.status_code = status_code
        self._lines = list(lines)
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self, decode_unicode: bool = True):
        yield from self._lines


def test_stream_invokes_hooks_and_yields_chunks(monkeypatch, client):
    lines = [
        "",
        "not streamed data",
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": 123}}]}',
        "data: invalid-json",
        'data: {"choices": [{"delta": {"content": " world"}}]}',
        "data: [DONE]",
    ]
    response = _StreamResponse(200, lines)
    post_mock = Mock(return_value=response)
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.requests.post", post_mock)

    events: dict[str, Any] = {"start": 0, "chunks": [], "complete": None, "error": None}

    hooks = StreamHooks(
        on_start=lambda: events.__setitem__("start", events["start"] + 1),
        on_chunk=lambda chunk: events["chunks"].append(chunk),
        on_complete=lambda data: events.__setitem__("complete", data),
        on_error=lambda exc: events.__setitem__("error", exc),
    )

    chunks = list(client.stream([Message(role="user", content="hi")], hooks=hooks))

    assert chunks == ["Hello", "123", " world"]
    assert events["start"] == 1
    assert events["chunks"] == chunks
    assert events["complete"] == "Hello123 world"
    assert events["error"] is None
    assert post_mock.call_args.kwargs["stream"] is True


def test_stream_reports_error_and_calls_on_error(monkeypatch, client):
    response = _StreamResponse(429, [], text="rate limited")
    post_mock = Mock(return_value=response)
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.requests.post", post_mock)

    errors: list[Exception] = []
    hooks = StreamHooks(on_error=lambda exc: errors.append(exc))

    with pytest.raises(LLMRateLimitError):
        list(client.stream([Message(role="user", content="hi")], hooks=hooks))

    assert len(errors) == 1
    assert isinstance(errors[0], LLMRateLimitError)


def test_invoke_tools_handles_parallel_flag(monkeypatch, client):
    captured_payload: dict[str, Any] = {}

    def fake_post(payload, extra_headers=None):
        captured_payload.update(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": "Tool executed",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "find", "arguments": '{"path": "."}'},
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)

    result = client.invoke_tools(
        [Message(role="user", content="run tool")],
        tools=[{"type": "function", "function": {"name": "find", "parameters": {}}}],
        parallel_tool_calls=False,
    )

    assert captured_payload["parallel_tool_calls"] is False
    assert result.calls[0].name == "find"
    assert result.calls[0].arguments == {"path": "."}


def test_invoke_tools_omits_parallel_flag_when_not_supported(monkeypatch):
    class NonParallelClient(_DummyClient):
        _SUPPORTS_PARALLEL_TOOL_CALLS = False

    client = NonParallelClient(
        provider_name="BaseLLM",
        api_key="token",
        model="demo",
        base_url="https://example.com",
    )

    def fake_post(payload, extra_headers=None):
        assert "parallel_tool_calls" not in payload
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [],
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)

    result = client.invoke_tools([Message(role="user", content="hi")], tools=[])
    assert result.calls == []


def test_extract_choice_message_raises_on_malformed_payload(client):
    with pytest.raises(LLMError):
        client._extract_choice_message({}, "chat response")


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"choices": [{"delta": {"content": "chunk"}}]}, "chunk"),
        ({"choices": [{"delta": {"content": 42}}]}, "42"),
        ({}, None),
        ({"choices": [{"delta": {"content": ""}}]}, None),
    ],
)
def test_parse_stream_delta_handles_variants(client, payload, expected):
    assert client._parse_stream_delta(payload) == expected


def test_parse_tool_calls_handles_dict_and_string_arguments(client):
    calls = client._parse_tool_calls(
        [
            {"id": "1", "function": {"name": "find", "arguments": {"path": "."}}},
            {"id": "2", "function": {"name": "info", "arguments": '{"verbose": true}'}},
            {"id": "3", "function": {"name": "bad", "arguments": "not json"}},
            {"id": "4", "function": {"name": "list", "arguments": '["unexpected"]'}},
        ]
    )

    assert calls[0].arguments == {"path": "."}
    assert calls[1].arguments == {"verbose": True}
    assert calls[2].arguments == {}
    assert calls[3].arguments == ["unexpected"]
