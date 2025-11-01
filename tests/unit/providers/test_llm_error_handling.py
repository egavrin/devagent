"""Tests for HTTPChatLLMClient error handling pathways."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from ai_dev_agent.providers.llm.base import (
    HTTPChatLLMClient,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
    Message,
    RetryConfig,
)


class _DummyClient(HTTPChatLLMClient):
    """Concrete subclass to expose protected helpers for testing."""

    def _prepare_payload(self, messages, temperature, max_tokens):
        return {"messages": [message.to_payload() for message in messages]}


def _make_response(
    status_code: int,
    text: str = "",
    *,
    json_data: Any | None = None,
    json_error: Exception | None = None,
) -> SimpleNamespace:
    def _json():
        if json_error:
            raise json_error
        return json_data

    return SimpleNamespace(
        status_code=status_code,
        text=text,
        json=_json,
        __enter__=lambda self=None: self,
        __exit__=lambda *args, **kwargs: None,
        iter_lines=lambda decode_unicode=True: iter(()),
    )


@pytest.fixture()
def dummy_client():
    retry_config = RetryConfig(
        max_retries=2,
        initial_delay=0.0,
        max_delay=0.0,
        backoff_multiplier=1.0,
        jitter_ratio=0.0,
        retryable_status_codes={429, 500, 503},
    )
    client = _DummyClient(
        provider_name="TestLLM",
        api_key="key",
        model="model",
        base_url="https://example.com",
        retry_config=retry_config,
        timeout=1.0,
    )
    return client


def test_post_network_failure_exhausts_retries(monkeypatch, dummy_client):
    """Connection errors should exhaust retries then raise LLMConnectionError."""

    post_mock = Mock(side_effect=requests.ConnectionError("network down"))
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.requests.post", post_mock)
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.time.sleep", lambda *_: None)

    with pytest.raises(LLMConnectionError) as exc_info:
        dummy_client._post({"messages": []})

    assert post_mock.call_count == dummy_client.retry_config.max_retries
    assert "TestLLM connection failed" in str(exc_info.value)


def test_post_timeout_raises_timeout_error(monkeypatch, dummy_client):
    """Timeout errors should be wrapped as LLMTimeoutError."""

    post_mock = Mock(side_effect=requests.Timeout("request timed out"))
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.requests.post", post_mock)
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.time.sleep", lambda *_: None)

    with pytest.raises(LLMTimeoutError):
        dummy_client._post({"messages": []})

    assert post_mock.call_count == dummy_client.retry_config.max_retries


def test_post_malformed_response_raises_response_error(monkeypatch, dummy_client):
    """Invalid JSON payloads should produce LLMResponseError."""

    response = _make_response(
        200,
        text="not-json",
        json_error=json.JSONDecodeError("bad", "bad", 0),
    )
    monkeypatch.setattr(
        "ai_dev_agent.providers.llm.base.requests.post", lambda *args, **kwargs: response
    )

    with pytest.raises(LLMResponseError):
        dummy_client._post({"messages": []})


def test_post_filesystem_error_is_wrapped(monkeypatch, dummy_client):
    """File system level OSErrors should surface as LLMConnectionError."""

    post_mock = Mock(side_effect=OSError("No space left on device"))
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.requests.post", post_mock)
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.time.sleep", lambda *_: None)

    with pytest.raises(LLMConnectionError) as exc_info:
        dummy_client._post({"messages": []})

    assert "No space left on device" in str(exc_info.value)


def test_post_rate_limit_resource_exhaustion(monkeypatch, dummy_client):
    """Repeated rate limits should escalate to LLMRetryExhaustedError rooted in LLMRateLimitError."""

    responses = [
        _make_response(429, text="Too many requests", json_data={"error": "rate limited"})
        for _ in range(dummy_client.retry_config.max_retries)
    ]
    post_mock = Mock(side_effect=responses)
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.requests.post", post_mock)
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.time.sleep", lambda *_: None)

    with pytest.raises(LLMRetryExhaustedError) as exc_info:
        dummy_client._post({"messages": []})

    assert post_mock.call_count == dummy_client.retry_config.max_retries
    assert isinstance(exc_info.value.__cause__, LLMRateLimitError)


def test_post_request_exception_treated_as_connection_error(monkeypatch, dummy_client):
    """Generic RequestException should be treated as a transport failure and retried."""

    post_mock = Mock(side_effect=requests.RequestException("TLS negotiation failed"))
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.requests.post", post_mock)
    monkeypatch.setattr("ai_dev_agent.providers.llm.base.time.sleep", lambda *_: None)

    with pytest.raises(LLMConnectionError):
        dummy_client._post({"messages": []})

    assert post_mock.call_count == dummy_client.retry_config.max_retries


def test_post_non_retryable_http_error(monkeypatch, dummy_client):
    """Status codes outside retryable set should raise an LLMResponseError immediately."""

    response = _make_response(400, text="Bad request", json_data={"error": "invalid prompt"})
    monkeypatch.setattr(
        "ai_dev_agent.providers.llm.base.requests.post",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(LLMResponseError):
        dummy_client._post({"messages": []})


def test_complete_propagates_transport_errors(monkeypatch, dummy_client):
    """High-level complete call should surface underlying connection failures."""

    monkeypatch.setattr(
        dummy_client,
        "_post",
        Mock(side_effect=LLMConnectionError("Service unreachable")),
    )

    with pytest.raises(LLMConnectionError):
        dummy_client.complete([Message(role="user", content="Ping?")])
