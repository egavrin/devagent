"""Integration module for enhanced budgeting components.

This module provides hooks to wire the new cost tracking, retry handling,
and summarization features into the execution path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ai_dev_agent.core.utils.cost_tracker import CostTracker, create_token_usage_from_response
from ai_dev_agent.core.utils.retry_handler import RetryConfig, create_retry_handler
from ai_dev_agent.core.utils.summarizer import SummarizationConfig, create_summarizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_dev_agent.core.utils.config import Settings
    from ai_dev_agent.providers.llm.base import LLMClient, Message

LOGGER = logging.getLogger(__name__)


@dataclass
class BudgetIntegration:
    """Central integration point for enhanced budgeting features.

    This class coordinates cost tracking, retry handling, and summarization
    across the execution pipeline.
    """

    settings: Settings
    cost_tracker: CostTracker | None = None
    retry_handler: Any | None = None
    summarizer: Any | None = None
    enabled: bool = True

    def __post_init__(self):
        """Initialize components based on settings."""
        # Initialize cost tracker
        if self.settings.enable_cost_tracking:
            self.cost_tracker = CostTracker()
            # Silent initialization - no logs

        # Initialize retry handler
        if self.settings.enable_retry:
            retry_config = RetryConfig(
                max_retries=self.settings.retry_max_attempts,
                initial_delay=self.settings.retry_initial_delay,
                max_delay=self.settings.retry_max_delay,
            )
            self.retry_handler = create_retry_handler(
                smart=True, **retry_config.__dict__  # Use smart handler with circuit breaker
            )
            # Silent initialization - no logs

        # Initialize summarizer (requires LLM client, deferred)
        self.summarizer = None  # Will be initialized when LLM client available

    def initialize_summarizer(self, llm_client: LLMClient):
        """Initialize the summarizer with an LLM client.

        Args:
            llm_client: LLM client for generating summaries
        """
        if self.settings.enable_summarization:
            config = SummarizationConfig(
                summarization_model=self.settings.summarization_model,
                async_summarization=False,  # Can be made configurable
            )
            # Use two_tier based on settings
            use_two_tier = getattr(self.settings, "enable_two_tier_pruning", True)
            self.summarizer = create_summarizer(
                llm_client, two_tier=use_two_tier, **config.__dict__
            )
            # Silent initialization - no logs

    def track_llm_call(
        self,
        model: str,
        response_data: dict[str, Any],
        operation: str = "completion",
        iteration: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Track cost and usage for an LLM call.

        Args:
            model: Model name used
            response_data: Response data containing usage information
            operation: Type of operation (completion, tool_call, etc.)
            iteration: Current iteration number
            phase: Current phase (exploration, investigation, etc.)
        """
        if not self.cost_tracker or not self.enabled:
            return

        try:
            usage = create_token_usage_from_response(response_data)
            self.cost_tracker.track_request(
                model=model,
                usage=usage,
                operation=operation,
                iteration=iteration,
                phase=phase,
            )

            # Cost warning check disabled - silent tracking

            # Debug logging disabled for silent operation
        except Exception:
            # Silent error handling - no logs
            pass

    def execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from function execution
        """
        if self.retry_handler and self.enabled:
            return self.retry_handler.execute_with_retry(
                func, *args, on_retry=self._on_retry, **kwargs
            )
        else:
            # No retry, execute directly
            return func(*args, **kwargs)

    def _on_retry(self, attempt: int, error: Exception, delay: float):
        """Callback for retry events.

        Args:
            attempt: Retry attempt number
            error: Error that triggered retry
            delay: Delay before retry
        """
        # Silent retry - no logs

    def summarize_if_needed(
        self,
        messages: Sequence[Message],
        target_tokens: int,
    ) -> list[Message]:
        """Summarize messages if they exceed token budget.

        Args:
            messages: Messages to potentially summarize
            target_tokens: Target token count

        Returns:
            Possibly summarized messages
        """
        if not self.summarizer or not self.enabled:
            return list(messages)

        try:
            # Check which method is available on the summarizer
            if hasattr(self.summarizer, "optimize_context"):
                # TwoTierSummarizer
                return self.summarizer.optimize_context(
                    list(messages),
                    target_tokens,
                )
            elif hasattr(self.summarizer, "summarize_if_needed"):
                # ConversationSummarizer
                return self.summarizer.summarize_if_needed(
                    list(messages),
                    target_tokens,
                )
            else:
                # Silent fallback - no logs
                return list(messages)
        except Exception:
            # Silent error handling - no logs
            return list(messages)

    def get_cost_summary(self, detailed: bool = False) -> str:
        """Get formatted cost summary.

        Args:
            detailed: Include detailed breakdown

        Returns:
            Formatted cost summary string
        """
        if not self.cost_tracker:
            return "Cost tracking disabled"

        return self.cost_tracker.format_summary(detailed=detailed)

    def get_cost_forecast(self, remaining_iterations: int) -> float:
        """Get cost forecast for remaining iterations.

        Args:
            remaining_iterations: Number of iterations remaining

        Returns:
            Estimated cost in USD
        """
        if not self.cost_tracker:
            return 0.0

        return self.cost_tracker.get_forecast(remaining_iterations)

    def reset_retry_stats(self):
        """Reset retry handler statistics."""
        if self.retry_handler:
            self.retry_handler.reset()

    def get_integration_status(self) -> dict[str, Any]:
        """Get status of all integrated components.

        Returns:
            Dictionary with component statuses
        """
        status = {
            "enabled": self.enabled,
            "cost_tracking": {
                "enabled": self.cost_tracker is not None,
            },
            "retry_handling": {
                "enabled": self.retry_handler is not None,
            },
            "summarization": {
                "enabled": self.summarizer is not None,
            },
        }

        if self.cost_tracker:
            status["cost_tracking"]["total_cost"] = self.cost_tracker.total_cost_usd
            status["cost_tracking"]["total_tokens"] = (
                self.cost_tracker.total_prompt_tokens + self.cost_tracker.total_completion_tokens
            )

        if self.retry_handler:
            status["retry_handling"]["stats"] = self.retry_handler.get_retry_stats()

        return status


def create_budget_integration(settings: Settings) -> BudgetIntegration:
    """Factory function to create budget integration from settings.

    Args:
        settings: Application settings

    Returns:
        Configured budget integration instance
    """
    return BudgetIntegration(settings=settings)


def integrate_with_executor(
    executor_func,
    budget_integration: BudgetIntegration,
):
    """Decorator to integrate budgeting with an executor function.

    Example:
        ```python
        @integrate_with_executor(budget_integration)
        def execute_react_loop(messages, tools):
            # Automatic retry and cost tracking
            return llm.complete(messages)
        ```

    Args:
        executor_func: Function to wrap
        budget_integration: Budget integration instance

    Returns:
        Wrapped function with budgeting features
    """

    def wrapped(*args, **kwargs):
        # Execute with retry
        result = budget_integration.execute_with_retry(executor_func, *args, **kwargs)

        # Track cost if response contains usage data
        if isinstance(result, dict) and "usage" in result:
            budget_integration.track_llm_call(
                model=kwargs.get("model", "unknown"),
                response_data=result,
                operation=kwargs.get("operation", "completion"),
            )

        return result

    wrapped.__name__ = executor_func.__name__
    wrapped.__doc__ = executor_func.__doc__
    return wrapped


# Example integration in executor.py:
def enhance_executor(settings: Settings, llm_client: LLMClient) -> BudgetIntegration:
    """Example of how to integrate enhanced budgeting in executor.

    This would be added to executor.py or similar:

    ```python
    # In executor.py
    from ai_dev_agent.core.utils.budget_integration import enhance_executor

    # Initialize at start of execution
    budget = enhance_executor(settings, llm_client)

    # Wrap LLM calls
    result = budget.execute_with_retry(
        llm_client.invoke_tools,
        messages,
        tools,
    )

    # Track costs
    budget.track_llm_call(
        model=settings.model,
        response_data=result,
        iteration=iteration,
        phase=context.phase,
    )

    # Get summary at end
    print(budget.get_cost_summary(detailed=True))
    ```

    Args:
        settings: Application settings
        llm_client: LLM client for summarization

    Returns:
        Configured budget integration
    """
    integration = create_budget_integration(settings)
    integration.initialize_summarizer(llm_client)
    return integration


__all__ = [
    "BudgetIntegration",
    "create_budget_integration",
    "enhance_executor",
    "integrate_with_executor",
]
