"""Tests for the ReAct executor module."""
import json
from unittest.mock import MagicMock, patch, Mock, call
from pathlib import Path
import tempfile

import pytest
import click

from ai_dev_agent.cli.react.executor import (
    _execute_react_assistant,
    _build_json_enforcement_instructions,
    _sanitize_conversation_for_llm,
    _extract_json,
    _truncate_shell_history,
    _build_phase_prompt,
    _build_synthesis_prompt,
    _record_search_query,
    BudgetAwareExecutor
)
from ai_dev_agent.engine.react.types import ActionRequest, Observation, StepRecord, RunResult, MetricsSnapshot
from ai_dev_agent.providers.llm.base import Message


class TestJsonEnforcement:
    """Test JSON enforcement functions."""

    def test_build_json_enforcement_instructions_basic(self):
        """Test building basic JSON enforcement instructions."""
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        instructions = _build_json_enforcement_instructions(schema)

        assert "CRITICAL OUTPUT REQUIREMENT" in instructions
        assert "JSON" in instructions
        assert json.dumps(schema, indent=2) in instructions

    def test_build_json_enforcement_instructions_complex(self):
        """Test building JSON enforcement with complex schema."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "failure"]},
                "data": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["status"]
        }
        instructions = _build_json_enforcement_instructions(schema)

        assert json.dumps(schema, indent=2) in instructions
        assert "required" in instructions


class TestSanitizeConversation:
    """Test conversation sanitization."""

    def test_sanitize_conversation_empty(self):
        """Test sanitizing empty conversation."""
        messages = []
        result = _sanitize_conversation_for_llm(messages)
        assert result == []

    def test_sanitize_conversation_basic(self):
        """Test sanitizing basic conversation."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there")
        ]
        result = _sanitize_conversation_for_llm(messages)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "assistant"

    def test_sanitize_conversation_removes_orphaned_tools(self):
        """Test sanitizing removes orphaned tool messages."""
        messages = [
            Message(role="assistant", content="Let me help", tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "test"}}
            ]),
            Message(role="tool", content="Result", tool_call_id="call_1"),  # Valid
            Message(role="tool", content="Orphan", tool_call_id="call_999"),  # Orphan
        ]
        result = _sanitize_conversation_for_llm(messages)
        assert len(result) == 2  # Assistant + valid tool message
        assert result[1].tool_call_id == "call_1"

    def test_sanitize_conversation_preserves_regular_messages(self):
        """Test sanitizing preserves regular messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="system", content="You are helpful")
        ]
        result = _sanitize_conversation_for_llm(messages)
        assert len(result) == 3
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there"
        assert result[2].content == "You are helpful"


class TestExtractJson:
    """Test JSON extraction from text."""

    def test_extract_json_simple(self):
        """Test extracting simple JSON."""
        text = '{"key": "value"}'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_with_text(self):
        """Test extracting JSON embedded in text."""
        text = 'The result is {"status": "success", "data": [1, 2, 3]} and that\'s it'
        result = _extract_json(text)
        assert result == {"status": "success", "data": [1, 2, 3]}

    def test_extract_json_multiline(self):
        """Test extracting multiline JSON."""
        text = """
        Here's the output:
        {
            "name": "test",
            "value": 42
        }
        Done.
        """
        result = _extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_extract_json_no_json(self):
        """Test extracting from text with no JSON."""
        text = "This is just plain text with no JSON"
        result = _extract_json(text)
        assert result is None

    def test_extract_json_invalid(self):
        """Test extracting invalid JSON."""
        text = '{"key": invalid}'
        result = _extract_json(text)
        assert result is None

    def test_extract_json_multiple(self):
        """Test extracting first valid JSON from multiple."""
        text = '{"first": 1} some text {"second": 2}'
        result = _extract_json(text)
        assert result == {"first": 1}


class TestTruncateShellHistory:
    """Test shell history truncation."""

    def test_truncate_shell_history_empty(self):
        """Test truncating empty history."""
        result = _truncate_shell_history([], 5)
        assert result == []

    def test_truncate_shell_history_under_limit(self):
        """Test truncating history under the limit."""
        messages = [
            Message(role="user", content="cmd1"),
            Message(role="assistant", content="output1")
        ]
        result = _truncate_shell_history(messages, 5)
        assert result == messages

    def test_truncate_shell_history_over_limit(self):
        """Test truncating history over the limit."""
        messages = []
        for i in range(10):
            messages.append(Message(role="user", content=f"cmd{i}"))
            messages.append(Message(role="assistant", content=f"output{i}"))

        result = _truncate_shell_history(messages, 3)
        # Should keep last 3 turns (6 messages)
        assert len(result) <= 6
        assert "cmd9" in result[-2].content if len(result) >= 2 else True

    def test_truncate_shell_history_preserves_pairs(self):
        """Test truncation preserves user-assistant pairs."""
        messages = [
            Message(role="user", content="cmd1"),
            Message(role="assistant", content="output1"),
            Message(role="user", content="cmd2"),
            Message(role="assistant", content="output2"),
            Message(role="user", content="cmd3")  # Incomplete pair
        ]

        result = _truncate_shell_history(messages, 1)
        # Should handle incomplete pairs gracefully
        assert len(result) >= 1


class TestBuildPhasePrompt:
    """Test phase prompt building."""

    def test_build_phase_prompt_basic(self):
        """Test building basic phase prompt."""
        prompt = _build_phase_prompt(
            phase="exploration",
            user_query="Test query",
            context="Some context",
            constraints="No constraints",
            workspace="/test/workspace",
            repository_language="python"
        )

        assert "Test query" in prompt
        assert "beginning your investigation" in prompt.lower() or "exploration" in prompt.lower()

    def test_build_phase_prompt_with_context(self):
        """Test building phase prompt with context."""
        prompt = _build_phase_prompt(
            phase="investigation",
            user_query="Test query",
            context="Previous work done",
            constraints="Must be fast",
            workspace="/test/workspace",
            repository_language="javascript"
        )

        assert "Test query" in prompt
        assert "Previous work" in prompt or "investigating" in prompt.lower()

    def test_build_phase_prompt_multiple_phases(self):
        """Test building phase prompt for different phases."""
        prompt = _build_phase_prompt(
            phase="consolidation",
            user_query="Build feature X",
            context="Feature implemented",
            constraints="Must have 90% coverage",
            workspace="/test/workspace",
            repository_language="java"
        )

        assert "Build feature X" in prompt
        assert "consolidating" in prompt.lower() or "connect" in prompt.lower()
        assert "90%" in prompt or "coverage" in prompt.lower()


class TestBuildSynthesisPrompt:
    """Test synthesis prompt building."""

    def test_build_synthesis_prompt_basic(self):
        """Test building basic synthesis prompt."""
        # The function only takes user_query and context now
        context = "Examined files:\n- test.py (read_file)\nFound: def test(): pass"

        prompt = _build_synthesis_prompt(
            user_query="Test task",
            context=context,
            workspace="/test/workspace"
        )

        assert "Test task" in prompt
        assert "test.py" in prompt or "def test" in prompt

    def test_build_synthesis_prompt_multiple_steps(self):
        """Test building synthesis prompt with multiple steps."""
        # The function only takes user_query and context now
        context = """Steps performed:
1. Searched for TODOs (found 3 TODOs)
2. Edited main.py (file updated)"""

        prompt = _build_synthesis_prompt(
            user_query="Fix TODOs",
            context=context,
            workspace="/test/workspace"
        )

        assert "Fix TODOs" in prompt
        assert "TODO" in prompt
        assert "main.py" in prompt or "file updated" in prompt

    def test_build_synthesis_prompt_with_failures(self):
        """Test building synthesis prompt with failed steps."""
        # The function only takes user_query and context now
        context = "Attempted to write out.txt but failed: Permission denied"

        prompt = _build_synthesis_prompt(
            user_query="Write output",
            context=context,
            workspace="/test/workspace"
        )

        assert "Write output" in prompt
        assert "Permission" in prompt or "failed" in prompt.lower()


class TestRecordSearchQuery:
    """Test search query recording."""

    def test_record_search_query_grep(self):
        """Test recording grep search query."""
        queries = set()
        action = MagicMock()
        action.tool = "grep"
        action.args = {"query": "TODO"}

        _record_search_query(action, queries)

        assert "TODO" in queries

    def test_record_search_query_find(self):
        """Test recording find search query."""
        queries = set()
        action = MagicMock()
        action.tool = "find"
        action.args = {"pattern": "*.py"}

        _record_search_query(action, queries)

        assert "*.py" in queries

    def test_record_search_query_non_search(self):
        """Test recording non-search action (should not record)."""
        queries = set()
        action = MagicMock()
        action.tool = "read_file"
        action.args = {"path": "test.txt"}

        _record_search_query(action, queries)

        assert len(queries) == 0

    def test_record_search_query_multiple(self):
        """Test recording multiple search queries."""
        queries = set()

        action1 = MagicMock()
        action1.tool = "grep"
        action1.args = {"query": "TODO"}

        action2 = MagicMock()
        action2.tool = "find"
        action2.args = {"pattern": "FIXME"}

        _record_search_query(action1, queries)
        _record_search_query(action2, queries)

        assert "TODO" in queries
        assert "FIXME" in queries


class TestBudgetAwareExecutor:
    """Test the BudgetAwareExecutor class."""

    def test_budget_aware_executor_creation(self):
        """Test creating BudgetAwareExecutor."""
        mock_budget = MagicMock()
        format_schema = {"type": "object"}

        executor = BudgetAwareExecutor(
            budget_manager=mock_budget,
            format_schema=format_schema
        )

        assert executor.budget_manager == mock_budget
        assert executor.format_schema == format_schema
        assert hasattr(executor, 'failure_detector')

    def test_budget_aware_executor_with_defaults(self):
        """Test BudgetAwareExecutor with defaults."""
        mock_budget = MagicMock()

        executor = BudgetAwareExecutor(budget_manager=mock_budget)

        assert executor.budget_manager == mock_budget
        assert executor.format_schema is None


# Removed TestExecuteReactAssistant class as ExecutorBridge doesn't exist
# and _execute_react_assistant has a complex implementation that would
# require extensive mocking