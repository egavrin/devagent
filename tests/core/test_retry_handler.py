import itertools

import pytest

from ai_dev_agent.core.utils.retry_handler import (
    RetryConfig,
    RetryHandler,
    SmartRetryHandler,
    create_retry_handler,
)
from ai_dev_agent.providers.llm import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
)


def test_retry_handler_retries_and_succeeds(monkeypatch):
    handler = RetryHandler(RetryConfig(max_retries=5, initial_delay=0.01, jitter_ratio=0.5))

    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise LLMTimeoutError("timeout")
        return "ok"

    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("random.uniform", lambda a, b: a)

    result = handler.execute_with_retry(flaky)
    assert result == "ok"
    stats = handler.get_retry_stats()
    assert stats["retry_count"] == 0  # reset after success
    assert handler._is_retryable(LLMConnectionError("conn")) is True

    class HttpError(Exception):
        def __init__(self, status_code):
            super().__init__("boom")
            self.status_code = status_code

    assert handler._is_retryable(HttpError(503)) is True

    with pytest.raises(ValueError):
        handler.execute_with_retry(lambda: (_ for _ in ()).throw(ValueError("no retry")))


def test_create_retry_handler_variants():
    standard = create_retry_handler()
    smart = create_retry_handler(smart=True, max_retries=1)
    assert isinstance(standard, RetryHandler) and not isinstance(standard, SmartRetryHandler)
    assert isinstance(smart, SmartRetryHandler)


def test_smart_retry_circuit_breaker(monkeypatch):
    config = RetryConfig(max_retries=1, initial_delay=0.0, jitter_ratio=0.0)
    handler = SmartRetryHandler(config)

    time_values = itertools.count(start=1, step=1)
    monkeypatch.setattr("time.time", lambda: next(time_values))
    monkeypatch.setattr("time.sleep", lambda _: None)

    def always_timeout():
        raise LLMTimeoutError("fail")

    # Trigger consecutive failures to open circuit
    for _ in range(5):
        with pytest.raises(LLMRetryExhaustedError):
            handler.execute_with_retry(always_timeout)

    stats = handler.get_retry_stats()
    assert stats["failure_count"] >= 5 and stats["circuit_open"] is True

    # Circuit should prevent further execution until cooldown expires
    with pytest.raises(LLMRetryExhaustedError, match="Circuit breaker is open"):
        handler.execute_with_retry(always_timeout)

    # Advance time beyond cooldown and ensure success resets state
    monkeypatch.setattr("time.time", lambda: 10_000)

    success_calls = {"count": 0}

    def succeed():
        success_calls["count"] += 1
        return "done"

    result = handler.execute_with_retry(succeed)
    assert result == "done"
    stats = handler.get_retry_stats()
    assert stats["success_count"] >= 1
    assert handler._success_count >= 1
    assert handler._failure_count == 0


def test_retry_handler_is_retryable_matches_message():
    handler = RetryHandler()
    assert handler._is_retryable(Exception("internal server error")) is True
    assert handler._is_retryable(Exception("permanent failure")) is False
