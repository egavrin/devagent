"""Test suite for enhanced budgeting features.

This combines the key tests from the standalone test files into the main test suite.
"""

from unittest.mock import MagicMock, Mock

from ai_dev_agent.cli.react.budget_control import (
    AdaptiveBudgetManager,
    IterationContext,
    get_tools_for_iteration,
)
from ai_dev_agent.core.utils.budget_integration import create_budget_integration
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.context_budget import (
    ContextBudgetConfig,
    config_from_settings,
    prune_messages,
)
from ai_dev_agent.core.utils.cost_tracker import (
    CostTracker,
    TokenUsage,
    create_token_usage_from_response,
)
from ai_dev_agent.core.utils.retry_handler import RetryConfig, RetryHandler
from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session.summarizer import LLMConversationSummarizer, create_summarizer


class TestToolFiltering:
    """Test that tool filtering works correctly in final iteration."""

    def test_final_iteration_returns_text_only_tool(self):
        """Test that final iteration returns only text submission tool."""
        available_tools = [
            {"function": {"name": "search", "description": "Search tool"}},
            {"function": {"name": "read", "description": "Read tool"}},
            {"function": {"name": "write", "description": "Write tool"}},
        ]

        final_context = IterationContext(
            number=10,
            total=10,
            remaining=0,
            percent_complete=100.0,
            phase="synthesis",
            is_final=True,
            is_penultimate=False,
        )

        tools = get_tools_for_iteration(final_context, available_tools)
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "submit_final_answer"

    def test_normal_iteration_returns_all_tools(self):
        """Test that normal iteration returns all available tools."""
        available_tools = [
            {"function": {"name": "search", "description": "Search tool"}},
            {"function": {"name": "read", "description": "Read tool"}},
        ]

        normal_context = IterationContext(
            number=5,
            total=10,
            remaining=5,
            percent_complete=50.0,
            phase="investigation",
            is_final=False,
            is_penultimate=False,
        )

        tools = get_tools_for_iteration(normal_context, available_tools)
        assert len(tools) == 2
        assert all(t in tools for t in available_tools)


class TestCostTracking:
    """Test cost tracking functionality."""

    def test_cache_token_accounting(self):
        """Test that cache tokens are correctly assigned."""
        response_data = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cache_creation_input_tokens": 200,  # Should be write
                "cache_read_input_tokens": 100,  # Should be read
            }
        }

        usage = create_token_usage_from_response(response_data)
        assert usage.cache_write_tokens == 200
        assert usage.cache_read_tokens == 100

    def test_cost_calculation_accuracy(self):
        """Test cost calculation with different token types."""
        tracker = CostTracker()
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            cache_read_tokens=100,
            cache_write_tokens=200,
        )

        tracker.track_request(
            model="gpt-4o",
            usage=usage,
            operation="test",
            iteration=1,
            phase="exploration",
        )

        assert tracker.total_prompt_tokens == 1000
        assert tracker.total_completion_tokens == 500
        assert tracker.total_cache_read_tokens == 100
        assert tracker.total_cache_write_tokens == 200
        assert tracker.total_cost_usd > 0

    def test_cost_warning_threshold(self):
        """Test cost warning threshold detection."""
        tracker = CostTracker()

        # Add some usage
        usage = TokenUsage(prompt_tokens=10000, completion_tokens=5000)
        tracker.track_request(model="gpt-4o", usage=usage)

        # Should warn if cost exceeds threshold
        assert tracker.should_warn(0.001)  # Very low threshold
        assert not tracker.should_warn(100.0)  # Very high threshold


class TestAdaptiveBudgetManager:
    """Test adaptive budget management features."""

    def test_reflection_mechanism(self):
        """Test reflection allows retry on errors."""
        manager = AdaptiveBudgetManager(
            max_iterations=10,
            enable_reflection=True,
            max_reflections=3,
        )

        # First reflection should be allowed
        assert manager.allow_reflection("Test error")
        assert manager.reflection.current_reflection == 1

        # Second reflection should be allowed
        assert manager.allow_reflection("Another error")
        assert manager.reflection.current_reflection == 2

        # Third reflection should be allowed
        assert manager.allow_reflection("Third error")
        assert manager.reflection.current_reflection == 3

        # Fourth should not be allowed (exceeded max)
        assert not manager.allow_reflection("Fourth error")

    def test_phase_adjustment(self):
        """Test dynamic phase adjustment based on progress."""
        manager = AdaptiveBudgetManager(
            max_iterations=10,
            adaptive_scaling=True,
        )

        # High success rate in exploration should extend it
        initial_exploration = manager._exploration_end
        manager.adjust_phases_for_progress(0.9)
        assert manager._exploration_end > initial_exploration

    def test_adaptive_scaling_with_model_context(self):
        """Test that phases scale based on model context window."""
        # Small model
        small_manager = AdaptiveBudgetManager(
            max_iterations=10,
            model_context_window=8000,
            adaptive_scaling=True,
        )

        # Large model
        large_manager = AdaptiveBudgetManager(
            max_iterations=10,
            model_context_window=200000,
            adaptive_scaling=True,
        )

        # Large model should have longer exploration phase
        assert large_manager._exploration_end > small_manager._exploration_end


class TestRetryHandler:
    """Test retry mechanism with exponential backoff."""

    def test_exponential_backoff(self):
        """Test that retry delays increase exponentially."""
        from ai_dev_agent.providers.llm import LLMConnectionError

        handler = RetryHandler(
            config=RetryConfig(
                max_retries=3,
                initial_delay=0.1,
                backoff_multiplier=2.0,
            )
        )

        attempt_count = 0
        delays = []

        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise LLMConnectionError("Test error")
            return "success"

        def on_retry(attempt, error, delay):
            delays.append(delay)

        result = handler.execute_with_retry(flaky_function, on_retry=on_retry)

        assert result == "success"
        assert attempt_count == 3
        assert len(delays) == 2
        # Check delays are increasing (with some jitter tolerance)
        assert delays[1] > delays[0]


class TestSummarizerIntegration:
    """Test summarizer integration and interfaces."""

    def test_conversation_summarizer_has_both_methods(self):
        """Test LLMConversationSummarizer has both interface methods."""
        mock_llm = Mock()
        mock_llm.complete = Mock(return_value="Summary")

        summarizer = create_summarizer(mock_llm)
        assert isinstance(summarizer, LLMConversationSummarizer)
        assert hasattr(summarizer, "summarize_if_needed")
        assert hasattr(summarizer, "optimize_context")

    def test_two_tier_summarizer_has_both_methods(self):
        """Test LLMConversationSummarizer has optimize_context (two-tier) method."""
        mock_llm = Mock()
        mock_llm.complete = Mock(return_value="Summary")

        summarizer = create_summarizer(mock_llm)
        assert isinstance(summarizer, LLMConversationSummarizer)
        assert hasattr(summarizer, "optimize_context")
        assert hasattr(summarizer, "summarize_if_needed")

    def test_budget_integration_works_with_unified_summarizer(self):
        """Test BudgetIntegration works with unified LLMConversationSummarizer."""
        mock_llm = Mock()
        mock_llm.complete = Mock(return_value="Summary")
        messages = [Message(role="user", content="Test message")]

        # Test with unified summarizer
        settings = Settings()
        settings.enable_summarization = True

        integration = create_budget_integration(settings)
        integration.initialize_summarizer(mock_llm)

        result = integration.summarize_if_needed(messages, 100)
        assert isinstance(result, list)


class TestContextBudgetConfig:
    """Test context budget configuration wiring."""

    def test_settings_wiring(self):
        """Test that new settings are properly wired through config_from_settings."""
        settings = Settings()
        settings.enable_two_tier_pruning = False
        settings.enable_summarization = False
        settings.prune_protect_tokens = 50000
        settings.prune_minimum_savings = 25000
        settings.summarization_model = "gpt-3.5-turbo"

        config = config_from_settings(settings)

        assert not config.enable_two_tier
        assert not config.enable_summarization
        assert config.prune_protect_tokens == 50000
        assert config.prune_minimum_savings == 25000
        assert config.summarization_model == "gpt-3.5-turbo"

    def test_two_tier_pruning(self):
        """Test two-tier pruning strategy."""
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="User request"),
        ]

        # Add many tool outputs
        for i in range(20):
            messages.append(
                Message(
                    role="tool", content=f"Tool output {i}: " + "x" * 2000, tool_call_id=f"call_{i}"
                )
            )

        config = ContextBudgetConfig(
            max_tokens=10000,
            enable_two_tier=True,
            prune_protect_tokens=5000,
            prune_minimum_savings=2000,
        )

        pruned = prune_messages(messages, config)

        # Should have pruned some tool messages
        assert len(pruned) == len(messages)
        tool_messages = [m for m in pruned if m.role == "tool"]
        truncated = [m for m in tool_messages if "pruned for context" in (m.content or "")]
        assert len(truncated) > 0  # Some should be truncated


class TestBudgetIntegration:
    """Test the complete budget integration."""

    def test_integration_initialization(self):
        """Test BudgetIntegration initializes all components correctly."""
        settings = Settings()
        settings.enable_cost_tracking = True
        settings.enable_retry = True
        settings.enable_summarization = False  # Skip summarizer for this test

        integration = create_budget_integration(settings)

        assert integration.cost_tracker is not None
        assert integration.retry_handler is not None
        assert integration.summarizer is None  # Not initialized yet

    def test_integrated_flow_with_mock_llm(self):
        """Test integrated flow with mocked LLM."""
        settings = Settings()
        settings.enable_cost_tracking = True
        settings.model = "gpt-4o"

        integration = create_budget_integration(settings)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response._raw_response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 300,
            }
        }

        # Track the call
        integration.track_llm_call(
            model="gpt-4o",
            response_data=mock_response._raw_response,
            operation="test",
            iteration=1,
            phase="exploration",
        )

        assert integration.cost_tracker.total_prompt_tokens == 1000
        assert integration.cost_tracker.total_completion_tokens == 300
        assert integration.cost_tracker.total_cost_usd > 0

        # Get summary
        summary = integration.get_cost_summary()
        assert "Total Cost" in summary


class TestRawResponseTracking:
    """Test that raw response with usage data is preserved."""

    def test_tool_call_result_preserves_raw_response(self):
        """Test that ToolCallResult includes raw response with usage data."""
        from ai_dev_agent.providers.llm.base import ToolCallResult

        # Create a mock raw response with usage data
        raw_response = {
            "choices": [{"message": {"content": "Test response", "tool_calls": []}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
            },
        }

        # Create ToolCallResult with raw response
        result = ToolCallResult(
            calls=[],
            message_content="Test response",
            raw_tool_calls=None,
            _raw_response=raw_response,
        )

        # Verify raw response is preserved
        assert result._raw_response is not None
        assert result._raw_response["usage"]["prompt_tokens"] == 100
        assert result._raw_response["usage"]["completion_tokens"] == 50
        assert result._raw_response["usage"]["cache_creation_input_tokens"] == 10
        assert result._raw_response["usage"]["cache_read_input_tokens"] == 20
