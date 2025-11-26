"""Retry handler with exponential backoff for LLM requests.

Implements jittered backoff to handle transient failures with configurable retry
behavior.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from ai_dev_agent.providers.llm import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""

    max_retries: int = 5
    initial_delay: float = 0.125  # 125ms initial delay
    max_delay: float = 60.0  # 60 seconds max
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.1  # 10% jitter to avoid thundering herd

    # Retryable exceptions
    retryable_exceptions: set[type] = None

    # Retryable status codes (for HTTP errors)
    retryable_status_codes: set[int] = None

    def __post_init__(self):
        """Initialize default retryable conditions."""
        if self.retryable_exceptions is None:
            self.retryable_exceptions = {
                LLMTimeoutError,
                LLMConnectionError,
                LLMRateLimitError,
            }

        if self.retryable_status_codes is None:
            self.retryable_status_codes = {
                429,  # Too Many Requests
                500,  # Internal Server Error
                502,  # Bad Gateway
                503,  # Service Unavailable
                504,  # Gateway Timeout
            }


class RetryHandler:
    """Handle retries with exponential backoff for LLM operations.

    Key features:
    - Exponential backoff with configurable delays
    - Jitter to prevent thundering herd
    - Selective retry based on error type
    - Maximum retry limit

    Example:
        ```python
        handler = RetryHandler()
        result = handler.execute_with_retry(
            llm_client.complete,
            messages,
            temperature=0.0
        )
        ```
    """

    def __init__(self, config: RetryConfig | None = None):
        """Initialize retry handler with configuration.

        Args:
            config: Retry configuration (uses defaults if not provided)
        """
        self.config = config or RetryConfig()
        self._retry_count = 0
        self._last_error = None

    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        on_retry: Callable[[int, Exception, float], None] | None = None,
        **kwargs,
    ) -> T:
        """Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            on_retry: Optional callback called before each retry
                     (attempt_number, error, delay_seconds)
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function execution

        Raises:
            LLMRetryExhaustedError: If all retries exhausted
            Exception: If non-retryable error encountered
        """
        delay = self.config.initial_delay
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Reset retry count on success
                result = func(*args, **kwargs)
                self._retry_count = 0
                return result

            except Exception as e:
                last_error = e
                self._last_error = e

                # Check if error is retryable
                if not self._is_retryable(e):
                    raise

                # Check if we've exhausted retries
                if attempt == self.config.max_retries - 1:
                    raise LLMRetryExhaustedError(
                        f"Failed after {self.config.max_retries} attempts: {e!s}"
                    ) from e

                # Calculate delay with jitter
                jittered_delay = self._add_jitter(delay)

                # Call retry callback if provided
                if on_retry:
                    on_retry(attempt + 1, e, jittered_delay)

                # Wait before retry
                time.sleep(jittered_delay)

                # Update delay for next iteration
                delay = min(delay * self.config.backoff_multiplier, self.config.max_delay)
                self._retry_count = attempt + 1

        # Should not reach here, but handle it gracefully
        raise LLMRetryExhaustedError(
            f"Failed after {self.config.max_retries} attempts: {last_error!s}"
        ) from last_error

    def _is_retryable(self, error: Exception) -> bool:
        """Determine if an error is retryable.

        Args:
            error: The exception to check

        Returns:
            True if error should trigger retry
        """
        # Check if error type is retryable
        for exc_type in self.config.retryable_exceptions:
            if isinstance(error, exc_type):
                return True

        # Check for HTTP status codes in error
        if hasattr(error, "status_code"):
            if error.status_code in self.config.retryable_status_codes:
                return True

        # Check for specific error messages that indicate transient issues
        error_msg = str(error).lower()
        transient_indicators = [
            "timeout",
            "timed out",
            "connection",
            "rate limit",
            "too many requests",
            "service unavailable",
            "gateway timeout",
            "bad gateway",
            "internal server error",
        ]

        return any(indicator in error_msg for indicator in transient_indicators)

    def _add_jitter(self, delay: float) -> float:
        """Add jitter to delay to prevent thundering herd.

        Args:
            delay: Base delay in seconds

        Returns:
            Delay with random jitter applied
        """
        jitter_amount = delay * self.config.jitter_ratio
        jitter = random.uniform(-jitter_amount, jitter_amount)
        return max(0, delay + jitter)

    def get_retry_stats(self) -> dict:
        """Get statistics about retry attempts.

        Returns:
            Dictionary with retry statistics
        """
        return {
            "retry_count": self._retry_count,
            "last_error": str(self._last_error) if self._last_error else None,
            "max_retries": self.config.max_retries,
        }

    def reset(self):
        """Reset retry statistics."""
        self._retry_count = 0
        self._last_error = None


class SmartRetryHandler(RetryHandler):
    """Enhanced retry handler with adaptive behavior.

    Extends base RetryHandler with:
    - Adaptive delay based on error type
    - Circuit breaker pattern
    - Success rate tracking
    """

    def __init__(self, config: RetryConfig | None = None):
        """Initialize smart retry handler."""
        super().__init__(config)
        self._success_count = 0
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_open_until = 0

    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        on_retry: Callable[[int, Exception, float], None] | None = None,
        **kwargs,
    ) -> T:
        """Execute with smart retry logic including circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            on_retry: Optional callback for retry events
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful execution

        Raises:
            LLMRetryExhaustedError: If retries exhausted or circuit open
            Exception: If non-retryable error
        """
        # Check circuit breaker
        if self._circuit_open and time.time() < self._circuit_open_until:
            raise LLMRetryExhaustedError("Circuit breaker is open")

        try:
            result = super().execute_with_retry(func, *args, on_retry=on_retry, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        """Track successful execution."""
        self._success_count += 1
        self._failure_count = 0
        self._circuit_open = False

    def _on_failure(self):
        """Track failed execution and potentially open circuit."""
        self._failure_count += 1

        # Open circuit breaker after consecutive failures
        if self._failure_count >= 5:
            self._circuit_open = True
            self._circuit_open_until = time.time() + 30  # 30 second cooldown

    def get_retry_stats(self) -> dict:
        """Get enhanced statistics including circuit breaker status.

        Returns:
            Dictionary with extended retry statistics
        """
        stats = super().get_retry_stats()
        stats.update(
            {
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "circuit_open": self._circuit_open,
                "success_rate": (
                    self._success_count / (self._success_count + self._failure_count)
                    if (self._success_count + self._failure_count) > 0
                    else 0
                ),
            }
        )
        return stats


def create_retry_handler(smart: bool = False, **config_kwargs) -> RetryHandler:
    """Factory function to create appropriate retry handler.

    Args:
        smart: If True, create SmartRetryHandler with adaptive behavior
        **config_kwargs: Configuration parameters for RetryConfig

    Returns:
        Configured retry handler instance
    """
    config = RetryConfig(**config_kwargs)

    if smart:
        return SmartRetryHandler(config)
    else:
        return RetryHandler(config)


__all__ = [
    "RetryConfig",
    "RetryHandler",
    "SmartRetryHandler",
    "create_retry_handler",
]
