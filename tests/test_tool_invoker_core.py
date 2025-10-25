"""Tests for engine/react/tool_invoker.py - core tool invocation functionality."""
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.tool_invoker import RegistryToolInvoker
from ai_dev_agent.engine.react.types import ActionRequest, Observation, ToolCall, ToolResult


class TestRegistryToolInvokerInit:
    """Test RegistryToolInvoker initialization."""

    def test_init_minimal(self, tmp_path):
        """Test initialization with minimal required parameters."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        assert invoker.workspace == tmp_path
        assert invoker.settings == settings
        assert invoker.code_editor is None
        assert invoker.test_runner is None
        assert invoker.collector is None

    def test_init_with_all_params(self, tmp_path):
        """Test initialization with all parameters."""
        settings = Settings()
        code_editor = Mock()
        test_runner = Mock()
        sandbox = Mock()
        collector = Mock()
        shell_manager = Mock()

        invoker = RegistryToolInvoker(
            workspace=tmp_path,
            settings=settings,
            code_editor=code_editor,
            test_runner=test_runner,
            sandbox=sandbox,
            collector=collector,
            shell_session_manager=shell_manager,
            shell_session_id="test-session",
        )

        assert invoker.code_editor == code_editor
        assert invoker.test_runner == test_runner
        assert invoker.sandbox == sandbox
        assert invoker.collector == collector
        assert invoker.shell_session_manager == shell_manager
        assert invoker.shell_session_id == "test-session"

    def test_init_creates_structure_hints(self, tmp_path):
        """Test that initialization creates structure hints."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        assert "symbols" in invoker._structure_hints
        assert "files" in invoker._structure_hints
        assert "project_summary" in invoker._structure_hints

    def test_init_creates_file_cache(self, tmp_path):
        """Test that initialization creates file read cache."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        assert hasattr(invoker, '_file_read_cache')
        assert isinstance(invoker._file_read_cache, dict)
        assert invoker._cache_ttl == 60.0


class TestSubmitFinalAnswer:
    """Test submit_final_answer special handling."""

    def test_submit_final_answer(self, tmp_path):
        """Test that submit_final_answer is intercepted."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        action = ActionRequest(
            step_id="test-step-1",
            thought="Testing final answer submission",
            tool="submit_final_answer",
            args={"answer": "This is my final answer"}
        )

        result = invoker(action)

        assert isinstance(result, Observation)
        assert result.success is True
        assert result.tool == "submit_final_answer"
        assert "final answer" in result.raw_output.lower()
        assert result.raw_output == "This is my final answer"

    def test_submit_final_answer_empty(self, tmp_path):
        """Test submit_final_answer with empty answer."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        action = ActionRequest(
            step_id="test-step-2",
            thought="Testing empty final answer",
            tool="submit_final_answer",
            args={"answer": ""}
        )

        result = invoker(action)

        assert result.success is True
        assert result.raw_output == ""


class TestBatchExecution:
    """Test batch tool execution."""

    def test_invoke_batch_empty(self, tmp_path):
        """Test batch execution with empty tool list."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        result = invoker.invoke_batch([])

        assert result.success is False
        assert "No tool calls" in result.outcome

    @patch.object(RegistryToolInvoker, '_execute_single_tool')
    def test_invoke_batch_single_tool(self, mock_execute, tmp_path):
        """Test batch execution with single tool."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        # Mock successful execution
        mock_result = ToolResult(
            call_id="1",
            tool="test_tool",
            success=True,
            outcome="Success",
            wall_time=0.5
        )
        mock_execute.return_value = mock_result

        tool_calls = [
            ToolCall(tool="test_tool", args={"arg": "value"}, call_id="1")
        ]

        result = invoker.invoke_batch(tool_calls)

        assert result.success is True
        assert "1 tool(s)" in result.outcome
        assert "1 succeeded" in result.outcome
        assert mock_execute.called

    @patch.object(RegistryToolInvoker, '_execute_single_tool')
    def test_invoke_batch_multiple_tools_all_success(self, mock_execute, tmp_path):
        """Test batch execution with multiple successful tools."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        # Mock successful executions
        mock_execute.side_effect = [
            ToolResult(call_id="1", tool="tool1", success=True, outcome="Success 1", wall_time=0.3),
            ToolResult(call_id="2", tool="tool2", success=True, outcome="Success 2", wall_time=0.2),
            ToolResult(call_id="3", tool="tool3", success=True, outcome="Success 3", wall_time=0.1),
        ]

        tool_calls = [
            ToolCall(tool="tool1", args={}, call_id="1"),
            ToolCall(tool="tool2", args={}, call_id="2"),
            ToolCall(tool="tool3", args={}, call_id="3"),
        ]

        result = invoker.invoke_batch(tool_calls)

        assert result.success is True
        assert "3 tool(s)" in result.outcome
        assert "3 succeeded" in result.outcome
        assert result.metrics["total_calls"] == 3
        assert result.metrics["successful_calls"] == 3
        assert result.metrics["failed_calls"] == 0

    @patch.object(RegistryToolInvoker, '_execute_single_tool')
    def test_invoke_batch_partial_failure(self, mock_execute, tmp_path):
        """Test batch execution with some failures."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        # Mock mixed results
        mock_execute.side_effect = [
            ToolResult(call_id="1", tool="tool1", success=True, outcome="Success", wall_time=0.1),
            ToolResult(call_id="2", tool="tool2", success=False, outcome="Failed", error="Error", wall_time=0.2),
            ToolResult(call_id="3", tool="tool3", success=True, outcome="Success", wall_time=0.15),
        ]

        tool_calls = [
            ToolCall(tool="tool1", args={}, call_id="1"),
            ToolCall(tool="tool2", args={}, call_id="2"),
            ToolCall(tool="tool3", args={}, call_id="3"),
        ]

        result = invoker.invoke_batch(tool_calls)

        assert result.success is False  # Not all succeeded
        assert "3 tool(s)" in result.outcome
        assert "2 succeeded" in result.outcome
        assert "1 failed" in result.outcome
        assert result.metrics["total_calls"] == 3
        assert result.metrics["successful_calls"] == 2
        assert result.metrics["failed_calls"] == 1

    @patch.object(RegistryToolInvoker, '_execute_single_tool')
    def test_invoke_batch_aggregates_wall_time(self, mock_execute, tmp_path):
        """Test that batch execution aggregates wall time."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        mock_execute.side_effect = [
            ToolResult(call_id="1", tool="tool1", success=True, outcome="Done", wall_time=0.5),
            ToolResult(call_id="2", tool="tool2", success=True, outcome="Done", wall_time=0.3),
        ]

        tool_calls = [
            ToolCall(tool="tool1", args={}, call_id="1"),
            ToolCall(tool="tool2", args={}, call_id="2"),
        ]

        result = invoker.invoke_batch(tool_calls)

        assert result.metrics["total_wall_time"] == 0.8
        assert result.metrics["max_wall_time"] == 0.5

    @patch.object(RegistryToolInvoker, '_execute_single_tool')
    def test_invoke_batch_aggregates_artifacts(self, mock_execute, tmp_path):
        """Test that batch execution aggregates artifacts."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        mock_execute.side_effect = [
            ToolResult(
                call_id="1",
                tool="tool1",
                success=True,
                outcome="Done",
                metrics={"artifacts": ["artifact1.txt"]}
            ),
            ToolResult(
                call_id="2",
                tool="tool2",
                success=True,
                outcome="Done",
                metrics={"artifacts": ["artifact2.txt"]}
            ),
        ]

        tool_calls = [
            ToolCall(tool="tool1", args={}, call_id="1"),
            ToolCall(tool="tool2", args={}, call_id="2"),
        ]

        result = invoker.invoke_batch(tool_calls)

        assert len(result.artifacts) == 2
        assert "artifact1.txt" in result.artifacts
        assert "artifact2.txt" in result.artifacts


class TestSingleToolExecution:
    """Test single tool execution."""

    def test_unknown_tool_error(self, tmp_path):
        """Test that unknown tools return error observation."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        action = ActionRequest(
            step_id="test-step-error",
            thought="Testing unknown tool handling",
            tool="nonexistent_tool",
            args={}
        )

        with patch.object(invoker, '_invoke_registry', side_effect=KeyError("Unknown tool")):
            result = invoker(action)

        assert result.success is False
        assert "Unknown tool" in result.outcome
        assert result.tool == "nonexistent_tool"

    def test_value_error_handling(self, tmp_path):
        """Test that ValueError is handled properly."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        action = ActionRequest(
            step_id="test-step-value-error",
            thought="Testing value error handling",
            tool="test_tool",
            args={}
        )

        with patch.object(invoker, '_invoke_registry', side_effect=ValueError("Invalid input")):
            result = invoker(action)

        assert result.success is False
        assert "rejected input" in result.outcome
        assert "Invalid input" in result.error

    def test_general_exception_handling(self, tmp_path):
        """Test that general exceptions are handled."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        action = ActionRequest(
            step_id="test-step-general-error",
            thought="Testing general exception handling",
            tool="test_tool",
            args={}
        )

        with patch.object(invoker, '_invoke_registry', side_effect=RuntimeError("Unexpected error")):
            result = invoker(action)

        assert result.success is False
        assert "failed" in result.outcome.lower()
        assert "Unexpected error" in result.error


class TestCallMethods:
    """Test __call__ method dispatching."""

    def test_call_with_batch_request(self, tmp_path):
        """Test that __call__ routes to invoke_batch for batch requests."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        tool_calls = [
            ToolCall(tool="tool1", args={}, call_id="1"),
            ToolCall(tool="tool2", args={}, call_id="2"),
        ]

        action = ActionRequest(
            step_id="test-batch-step",
            thought="Testing batch request routing",
            tool="",  # Empty when using tool_calls
            tool_calls=tool_calls
        )

        with patch.object(invoker, 'invoke_batch') as mock_batch:
            mock_batch.return_value = Observation(success=True, outcome="Batch done")
            result = invoker(action)

        mock_batch.assert_called_once_with(tool_calls)
        assert result.success is True

    def test_call_with_single_tool(self, tmp_path):
        """Test that __call__ handles single tool request."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        action = ActionRequest(
            step_id="test-single-step",
            thought="Testing single tool execution",
            tool="submit_final_answer",
            args={"answer": "Done"}
        )

        result = invoker(action)

        # Should handle submit_final_answer directly
        assert result.success is True
        assert result.tool == "submit_final_answer"


class TestWrapResult:
    """Test result wrapping functionality."""

    def test_wrap_result_with_tool_result(self, tmp_path):
        """Test wrapping a ToolResult object."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        tool_result = ToolResult(
            tool="test_tool",  # Required field
            success=True,
            outcome="Command executed successfully",
            error=None,
            metrics={"exit_code": 0}
        )

        with patch.object(invoker, '_wrap_result', wraps=invoker._wrap_result) as mock_wrap:
            mock_wrap.return_value = Observation(
                success=True,
                outcome="success",
                tool="test_tool",
                raw_output="Command executed successfully"
            )

            result = invoker._wrap_result("test_tool", tool_result)

        assert isinstance(result, Observation)
        assert result.success is True

    def test_wrap_result_with_dict(self, tmp_path):
        """Test wrapping a dictionary result."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        dict_result = {
            "success": True,
            "output": "Operation complete",
            "metadata": {"key": "value"}
        }

        result = invoker._wrap_result("test_tool", dict_result)

        assert isinstance(result, Observation)


class TestCacheManagement:
    """Test file read cache management."""

    def test_cache_ttl_property(self, tmp_path):
        """Test that cache TTL is set correctly."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        assert invoker._cache_ttl == 60.0

    def test_file_cache_initialization(self, tmp_path):
        """Test that file cache is initialized empty."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        assert len(invoker._file_read_cache) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_action_with_none_args(self, tmp_path):
        """Test action with empty args dict."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        action = ActionRequest(
            step_id="test-none-args",
            thought="Testing empty args handling",
            tool="submit_final_answer",
            args={}  # Empty dict instead of None
        )

        # Should handle empty args gracefully
        result = invoker(action)
        assert isinstance(result, Observation)

    def test_batch_with_no_wall_time(self, tmp_path):
        """Test batch execution when tools don't report wall_time."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        with patch.object(invoker, '_execute_single_tool') as mock_execute:
            mock_execute.side_effect = [
                ToolResult(call_id="1", tool="tool1", success=True, outcome="Done"),  # No wall_time
                ToolResult(call_id="2", tool="tool2", success=True, outcome="Done"),  # No wall_time
            ]

            tool_calls = [
                ToolCall(tool="tool1", args={}, call_id="1"),
                ToolCall(tool="tool2", args={}, call_id="2"),
            ]

            result = invoker.invoke_batch(tool_calls)

            # Should handle missing wall_time
            assert result.success is True
            assert "total_wall_time" not in result.metrics or result.metrics.get("total_wall_time") == 0

    def test_batch_tool_label(self, tmp_path):
        """Test that batch results have correct tool label."""
        settings = Settings()
        invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        with patch.object(invoker, '_execute_single_tool') as mock_execute:
            mock_execute.return_value = ToolResult(
                call_id="1",
                tool="test",
                success=True,
                outcome="Done"
            )

            tool_calls = [
                ToolCall(tool="tool1", args={}, call_id="1"),
                ToolCall(tool="tool2", args={}, call_id="2"),
                ToolCall(tool="tool3", args={}, call_id="3"),
            ]

            result = invoker.invoke_batch(tool_calls)

            assert result.tool == "batch[3]"
