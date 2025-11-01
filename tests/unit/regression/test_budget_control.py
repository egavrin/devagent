"""Tests for cli/react/budget_control.py - iteration and budget management."""

from unittest.mock import Mock

import pytest

from ai_dev_agent.cli.react.budget_control import (
    AdaptiveBudgetManager,
    BudgetManager,
    IterationContext,
    ReflectionContext,
    auto_generate_summary,
    combine_partial_responses,
    create_text_only_tool,
    extract_text_content,
    filter_essential_tools,
    filter_non_exploratory_tools,
    get_tools_for_iteration,
)


class TestBudgetManager:
    """Test BudgetManager class."""

    def test_init_default(self):
        """Test budget manager initialization with defaults."""
        manager = BudgetManager(max_iterations=10)
        assert manager.max_iterations == 10
        assert manager.current == 0
        assert manager.model_context_window == 100000
        assert manager.adaptive_scaling is True

    def test_init_custom_thresholds(self):
        """Test initialization with custom phase thresholds."""
        manager = BudgetManager(
            max_iterations=20,
            phase_thresholds={
                "exploration_end": 40.0,
                "investigation_end": 70.0,
                "consolidation_end": 90.0,
            },
        )
        assert manager._exploration_end == 40.0
        assert manager._investigation_end == 70.0
        assert manager._consolidation_end == 90.0

    def test_init_adaptive_scaling(self):
        """Test adaptive scaling with different model context windows."""
        # Large model (200k context)
        large_model = BudgetManager(
            max_iterations=10, model_context_window=200000, adaptive_scaling=True
        )
        # Small model (50k context)
        small_model = BudgetManager(
            max_iterations=10, model_context_window=50000, adaptive_scaling=True
        )

        # Large model should have higher exploration threshold
        assert large_model._exploration_end > small_model._exploration_end

    def test_init_with_warnings(self):
        """Test initialization with warning configuration."""
        manager = BudgetManager(
            max_iterations=10,
            warnings={"warn_before_final": True, "final_warning_iterations": 2},
        )
        assert manager._warn_before_final is True
        assert manager._final_warning_iterations == 2

    def test_next_iteration_increments(self):
        """Test that next_iteration increments counter."""
        manager = BudgetManager(max_iterations=5)
        assert manager.current == 0

        ctx = manager.next_iteration()
        assert manager.current == 1
        assert ctx.number == 1

    def test_next_iteration_returns_context(self):
        """Test that next_iteration returns proper context."""
        manager = BudgetManager(max_iterations=10)

        ctx = manager.next_iteration()
        assert isinstance(ctx, IterationContext)
        assert ctx.number == 1
        assert ctx.total == 10
        assert ctx.remaining == 9
        assert ctx.percent_complete == 10.0
        assert ctx.is_final is False
        assert ctx.is_penultimate is False

    def test_next_iteration_at_budget_limit(self):
        """Test next_iteration when budget is exhausted."""
        manager = BudgetManager(max_iterations=2)

        ctx1 = manager.next_iteration()
        assert ctx1 is not None

        ctx2 = manager.next_iteration()
        assert ctx2 is not None
        assert ctx2.is_final is True

        ctx3 = manager.next_iteration()
        assert ctx3 is None  # Budget exhausted

    def test_determine_phase_exploration(self):
        """Test phase determination during exploration."""
        manager = BudgetManager(max_iterations=100)

        # First iterations should be exploration
        ctx = manager.next_iteration()
        assert ctx.phase == "exploration"

    def test_determine_phase_investigation(self):
        """Test phase determination during investigation."""
        manager = BudgetManager(max_iterations=100)

        # Advance to ~40% completion
        for _ in range(40):
            ctx = manager.next_iteration()

        assert ctx.phase == "investigation"

    def test_determine_phase_consolidation(self):
        """Test phase determination during consolidation."""
        manager = BudgetManager(max_iterations=100)

        # Advance to ~65% completion
        for _ in range(65):
            ctx = manager.next_iteration()

        assert ctx.phase == "consolidation"

    def test_determine_phase_preparation(self):
        """Test phase determination during preparation."""
        manager = BudgetManager(max_iterations=100)

        # Advance to ~90% completion
        for _ in range(90):
            ctx = manager.next_iteration()

        assert ctx.phase == "preparation"

    def test_determine_phase_final_warning(self):
        """Test final warning phase."""
        manager = BudgetManager(
            max_iterations=10, warnings={"warn_before_final": True, "final_warning_iterations": 2}
        )

        # Advance to the iteration that should trigger final_warning (remaining=2)
        for _ in range(8):
            manager.next_iteration()

        ctx = manager.next_iteration()  # This is iteration 9, remaining=1
        assert ctx.phase == "final_warning"

    def test_determine_phase_synthesis(self):
        """Test synthesis phase (final iteration)."""
        manager = BudgetManager(max_iterations=5)

        # Advance to final iteration
        for _ in range(4):
            manager.next_iteration()

        ctx = manager.next_iteration()
        assert ctx.phase == "synthesis"
        assert ctx.is_final is True


class TestGetToolsForIteration:
    """Test get_tools_for_iteration function."""

    def test_final_iteration_returns_text_only(self):
        """Test that final iteration gets only text submission tool."""
        ctx = IterationContext(
            number=10,
            total=10,
            remaining=0,
            percent_complete=100.0,
            phase="synthesis",
            is_final=True,
            is_penultimate=False,
        )

        all_tools = [
            {"name": "read"},
            {"name": "write"},
            {"name": "search"},
        ]

        tools = get_tools_for_iteration(ctx, all_tools)
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "submit_final_answer"

    def test_penultimate_filters_to_essential(self):
        """Test that penultimate iteration filters to essential tools."""
        ctx = IterationContext(
            number=9,
            total=10,
            remaining=1,
            percent_complete=90.0,
            phase="preparation",
            is_final=False,
            is_penultimate=True,
        )

        all_tools = [
            {"name": "read"},
            {"name": "write"},
            {"name": "explore"},
        ]

        tools = get_tools_for_iteration(
            ctx, all_tools, tool_config={"essential_only_in_final": True}
        )
        # Should filter to essential only
        assert len(tools) < len(all_tools)

    def test_preparation_phase_limits_exploratory(self):
        """Test that preparation phase limits exploratory tools."""
        ctx = IterationContext(
            number=8,
            total=10,
            remaining=2,
            percent_complete=80.0,
            phase="preparation",
            is_final=False,
            is_penultimate=False,
        )

        all_tools = [
            {"name": "read"},
            {"name": "search"},
            {"name": "explore"},
        ]

        tools = get_tools_for_iteration(ctx, all_tools, tool_config={"limit_in_preparation": True})
        # Should filter out exploratory tools
        tool_names = [t.get("name") for t in tools]
        assert "explore" not in tool_names

    def test_returns_all_tools_in_exploration(self):
        """Test that exploration phase returns all tools."""
        ctx = IterationContext(
            number=1,
            total=10,
            remaining=9,
            percent_complete=10.0,
            phase="exploration",
            is_final=False,
            is_penultimate=False,
        )

        all_tools = [
            {"name": "read"},
            {"name": "write"},
            {"name": "search"},
            {"name": "explore"},
        ]

        tools = get_tools_for_iteration(ctx, all_tools)
        assert len(tools) == len(all_tools)


class TestFilterFunctions:
    """Test tool filtering functions."""

    def test_filter_non_exploratory_tools(self):
        """Test filtering out exploratory tools."""
        tools = [
            {"name": "read"},
            {"name": "write"},
            {"name": "search"},
            {"name": "explore"},
            {"name": "plan"},
            {"name": "symbol_index"},
        ]

        filtered = filter_non_exploratory_tools(tools)
        names = [t["name"] for t in filtered]

        assert "read" in names
        assert "write" in names
        assert "search" not in names
        assert "explore" not in names
        assert "plan" not in names
        assert "symbol_index" not in names

    def test_filter_non_exploratory_with_function_schema(self):
        """Test filtering tools with function schema format."""
        tools = [
            {"function": {"name": "read"}},
            {"function": {"name": "search_symbols"}},
        ]

        filtered = filter_non_exploratory_tools(tools)
        assert len(filtered) == 1
        assert filtered[0]["function"]["name"] == "read"

    def test_filter_essential_tools(self):
        """Test filtering to essential tools only."""
        tools = [
            {"name": "read"},
            {"name": "fs_read_text"},
            {"name": "write"},
            {"name": "run"},
            {"name": "exec.shell"},
            {"name": "explore"},
        ]

        filtered = filter_essential_tools(tools)
        names = [t["name"] for t in filtered]

        assert "read" in names
        assert "fs_read_text" in names
        assert "run" in names
        assert "exec.shell" in names
        # Non-essential tools should be filtered out
        assert "write" not in names
        assert "explore" not in names

    def test_filter_essential_returns_all_if_none_match(self):
        """Test that filter_essential returns all tools if none are essential."""
        tools = [
            {"name": "custom_tool1"},
            {"name": "custom_tool2"},
        ]

        filtered = filter_essential_tools(tools)
        # Should return all tools as fallback
        assert len(filtered) == len(tools)


class TestHelperFunctions:
    """Test helper utility functions."""

    def test_create_text_only_tool(self):
        """Test creating text-only submission tool."""
        tool = create_text_only_tool()

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "submit_final_answer"
        assert "parameters" in tool["function"]
        assert "answer" in tool["function"]["parameters"]["properties"]

    def test_extract_text_content_with_message(self):
        """Test extracting text content from result."""
        result = Mock()
        result.message_content = "  Hello world  "

        text = extract_text_content(result)
        assert text == "Hello world"

    def test_extract_text_content_no_message(self):
        """Test extracting text when no message_content."""
        result = Mock(spec=[])  # No message_content attribute

        text = extract_text_content(result)
        assert text == ""

    def test_extract_text_content_non_string(self):
        """Test extracting text when message_content is not a string."""
        result = Mock()
        result.message_content = ["not", "a", "string"]

        text = extract_text_content(result)
        assert text == ""

    def test_combine_partial_responses(self):
        """Test combining multiple partial responses."""
        parts = ["First part", "  ", "Second part", "", "Third part"]

        combined = combine_partial_responses(*parts)
        assert combined == "First part\n\nSecond part\n\nThird part"

    def test_combine_partial_responses_empty(self):
        """Test combining when all parts are empty."""
        parts = ["", "  ", None]

        combined = combine_partial_responses(*parts)
        assert combined == ""


class TestAutoGenerateSummary:
    """Test auto_generate_summary function."""

    def test_auto_generate_with_observations(self):
        """Test summary generation with assistant observations."""
        message1 = Mock()
        message1.role = "assistant"
        message1.content = "I found that the authentication logic is in auth.py"

        message2 = Mock()
        message2.role = "assistant"
        message2.content = "The token validation happens in middleware.py"

        conversation = [message1, message2]

        summary = auto_generate_summary(conversation)
        assert "authentication logic" in summary.lower()
        assert "middleware.py" in summary

    def test_auto_generate_with_files_and_searches(self):
        """Test summary generation with file and search info."""
        conversation = []
        files = ["auth.py", "middleware.py", "config.py"]
        searches = ["def authenticate", "token validation"]

        summary = auto_generate_summary(
            conversation, files_examined=files, searches_performed=searches
        )

        assert "3 file(s)" in summary
        assert "2 search(es)" in summary

    def test_auto_generate_skips_short_messages(self):
        """Test that short or empty messages are skipped."""
        msg1 = Mock()
        msg1.role = "assistant"
        msg1.content = "Let me check"  # Too short

        msg2 = Mock()
        msg2.role = "assistant"
        msg2.content = "This is a substantial finding about the code architecture"

        conversation = [msg1, msg2]

        summary = auto_generate_summary(conversation)
        assert "Let me check" not in summary
        assert "substantial finding" in summary

    def test_auto_generate_skips_context_markers(self):
        """Test that context markers are skipped."""
        msg1 = Mock()
        msg1.role = "assistant"
        msg1.content = "[Context: Previous search results]"

        msg2 = Mock()
        msg2.role = "assistant"
        msg2.content = "Real observation here"

        conversation = [msg1, msg2]

        summary = auto_generate_summary(conversation)
        assert "Context:" not in summary
        assert "Real observation" in summary

    def test_auto_generate_truncates_long_observations(self):
        """Test that very long observations are truncated."""
        long_text = "A" * 350
        msg = Mock()
        msg.role = "assistant"
        msg.content = long_text

        conversation = [msg]

        summary = auto_generate_summary(conversation)
        assert "..." in summary
        assert len(summary) < len(long_text)

    def test_auto_generate_lists_files_when_no_observations(self):
        """Test that files are listed when there are no observations."""
        conversation = []
        files = [f"file{i}.py" for i in range(15)]

        summary = auto_generate_summary(conversation, files_examined=files)

        assert "Files examined:" in summary
        # Should show first 10 and indicate more
        assert "... and 5 more" in summary


class TestAdaptiveBudgetManager:
    """Test AdaptiveBudgetManager class."""

    def test_init(self):
        """Test adaptive budget manager initialization."""
        manager = AdaptiveBudgetManager(max_iterations=10)
        assert manager.reflection.enabled is True
        assert manager.reflection.max_reflections == 3
        assert manager._success_count == 0
        assert manager._failure_count == 0

    def test_init_custom_reflection(self):
        """Test initialization with custom reflection settings."""
        manager = AdaptiveBudgetManager(
            max_iterations=10, enable_reflection=False, max_reflections=5
        )
        assert manager.reflection.enabled is False
        assert manager.reflection.max_reflections == 5

    def test_next_iteration_includes_reflection_info(self):
        """Test that iteration context includes reflection info."""
        manager = AdaptiveBudgetManager(max_iterations=10)

        ctx = manager.next_iteration()
        assert hasattr(ctx, "reflection_count")
        assert hasattr(ctx, "reflection_allowed")
        assert ctx.reflection_count == 0
        assert ctx.reflection_allowed is True

    def test_allow_reflection_increments_counter(self):
        """Test that allow_reflection increments counter."""
        manager = AdaptiveBudgetManager(max_iterations=10)

        result = manager.allow_reflection("Test error")
        assert result is True
        assert manager.reflection.current_reflection == 1
        assert manager.reflection.last_error == "Test error"
        assert manager._failure_count == 1

    def test_allow_reflection_respects_max(self):
        """Test that reflection respects max limit."""
        manager = AdaptiveBudgetManager(max_iterations=10, max_reflections=2)

        # First two should succeed
        assert manager.allow_reflection("Error 1") is True
        assert manager.allow_reflection("Error 2") is True

        # Third should fail
        assert manager.allow_reflection("Error 3") is False

    def test_allow_reflection_when_disabled(self):
        """Test that reflection is blocked when disabled."""
        manager = AdaptiveBudgetManager(max_iterations=10, enable_reflection=False)

        result = manager.allow_reflection("Error")
        assert result is False
        assert manager.reflection.current_reflection == 0

    def test_reset_reflection(self):
        """Test resetting reflection after success."""
        manager = AdaptiveBudgetManager(max_iterations=10)

        manager.allow_reflection("Error")
        assert manager.reflection.current_reflection == 1

        manager.reset_reflection()
        assert manager.reflection.current_reflection == 0
        assert manager.reflection.last_error is None
        assert manager._success_count == 1
        assert manager._failure_count == 0

    def test_adjust_phases_for_good_progress(self):
        """Test phase adjustment with good progress."""
        manager = AdaptiveBudgetManager(max_iterations=100, adaptive_scaling=True)

        # Advance to exploration phase
        manager.next_iteration()

        initial_exploration_end = manager._exploration_end
        initial_investigation_end = manager._investigation_end

        # Simulate good progress
        manager.adjust_phases_for_progress(success_rate=0.9)

        # Thresholds should extend
        assert manager._exploration_end > initial_exploration_end
        assert manager._investigation_end > initial_investigation_end

    def test_adjust_phases_for_poor_progress(self):
        """Test phase adjustment with poor progress."""
        manager = AdaptiveBudgetManager(max_iterations=100, adaptive_scaling=True)

        # Advance to investigation phase
        for _ in range(50):
            manager.next_iteration()

        initial_consolidation_end = manager._consolidation_end

        # Simulate poor progress
        manager.adjust_phases_for_progress(success_rate=0.2)

        # Should move to consolidation earlier
        assert manager._consolidation_end < initial_consolidation_end

    def test_adjust_phases_when_not_adaptive(self):
        """Test that phase adjustment is skipped when not adaptive."""
        manager = AdaptiveBudgetManager(max_iterations=100, adaptive_scaling=False)

        manager.next_iteration()
        initial_exploration_end = manager._exploration_end

        manager.adjust_phases_for_progress(success_rate=0.9)

        # No change
        assert manager._exploration_end == initial_exploration_end

    def test_percent_complete_property(self):
        """Test percent_complete property."""
        manager = AdaptiveBudgetManager(max_iterations=10)

        assert manager.percent_complete == 0.0

        for i in range(5):
            manager.next_iteration()

        assert manager.percent_complete == 50.0

    def test_get_stats(self):
        """Test get_stats method."""
        manager = AdaptiveBudgetManager(max_iterations=10)

        manager.next_iteration()
        manager.allow_reflection("Error")
        manager.reset_reflection()

        stats = manager.get_stats()

        assert stats["current_iteration"] == 1
        assert stats["max_iterations"] == 10
        assert stats["reflection_count"] == 0  # Reset
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 0  # Reset
        assert "current_thresholds" in stats
        assert "phase_adjustments" in stats


class TestReflectionContext:
    """Test ReflectionContext dataclass."""

    def test_init_defaults(self):
        """Test ReflectionContext default values."""
        ctx = ReflectionContext()
        assert ctx.max_reflections == 3
        assert ctx.current_reflection == 0
        assert ctx.last_error is None
        assert ctx.enabled is True

    def test_init_custom(self):
        """Test ReflectionContext with custom values."""
        ctx = ReflectionContext(
            max_reflections=5, current_reflection=2, last_error="Test", enabled=False
        )
        assert ctx.max_reflections == 5
        assert ctx.current_reflection == 2
        assert ctx.last_error == "Test"
        assert ctx.enabled is False


class TestIterationContext:
    """Test IterationContext dataclass."""

    def test_frozen(self):
        """Test that IterationContext is immutable."""
        ctx = IterationContext(
            number=1,
            total=10,
            remaining=9,
            percent_complete=10.0,
            phase="exploration",
            is_final=False,
            is_penultimate=False,
        )

        # Should raise error when trying to modify
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.number = 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_budget_manager_with_zero_iterations(self):
        """Test budget manager with zero iterations (should default to 1)."""
        manager = BudgetManager(max_iterations=0)
        assert manager.max_iterations == 1

    def test_budget_manager_with_negative_iterations(self):
        """Test budget manager with negative iterations (should default to 1)."""
        manager = BudgetManager(max_iterations=-5)
        assert manager.max_iterations == 1

    def test_coerce_percentage_with_invalid_values(self):
        """Test percentage coercion with various invalid inputs."""
        from ai_dev_agent.cli.react.budget_control import _coerce_percentage

        assert _coerce_percentage(None, fallback=50.0) == 50.0
        assert _coerce_percentage("invalid", fallback=50.0) == 50.0
        assert _coerce_percentage(-10, fallback=50.0) == 50.0
        assert _coerce_percentage(150, fallback=50.0) == 99.9
        assert _coerce_percentage(75.5, fallback=50.0) == 75.5

    def test_extract_tool_name_with_various_formats(self):
        """Test tool name extraction with various tool formats."""
        from ai_dev_agent.cli.react.budget_control import _extract_tool_name

        # Direct name
        assert _extract_tool_name({"name": "read"}) == "read"

        # Function schema
        assert _extract_tool_name({"function": {"name": "write"}}) == "write"

        # No name
        assert _extract_tool_name({}) is None
        assert _extract_tool_name({"function": {}}) is None

        # Invalid input
        assert _extract_tool_name("not a dict") is None
        assert _extract_tool_name(None) is None
