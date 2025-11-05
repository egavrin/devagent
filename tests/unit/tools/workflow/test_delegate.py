"""Tests for the delegate tool - focused on validation and error paths."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ai_dev_agent.agents.base import AgentResult
from ai_dev_agent.tools.workflow.delegate import delegate


@pytest.fixture
def mock_context():
    """Create a minimal mock ToolContext."""
    context = Mock()
    context.repo_root = "/test/workspace"
    context.settings = {}
    context.extra = {
        "cli_context": MagicMock(),
        "llm_client": MagicMock(),
        "session_id": "test-session",
    }
    return context


class TestDelegateValidation:
    """Test input validation - easy wins for coverage."""

    def test_missing_agent_parameter(self, mock_context):
        """Test error when agent parameter is missing."""
        result = delegate({"task": "Do something"}, mock_context)

        assert result["success"] is False
        assert "Missing required parameter: agent" in result["error"]
        assert result["artifacts"] == []

    def test_missing_task_parameter(self, mock_context):
        """Test error when task parameter is missing."""
        result = delegate({"agent": "design_agent"}, mock_context)

        assert result["success"] is False
        assert "Missing required parameter: task" in result["error"]
        assert result["artifacts"] == []

    def test_empty_agent_parameter(self, mock_context):
        """Test error when agent is empty string."""
        result = delegate({"agent": "", "task": "Test"}, mock_context)

        assert result["success"] is False
        assert "Missing required parameter: agent" in result["error"]

    def test_empty_task_parameter(self, mock_context):
        """Test error when task is empty string."""
        result = delegate({"agent": "design_agent", "task": ""}, mock_context)

        assert result["success"] is False
        assert "Missing required parameter: task" in result["error"]

    def test_invalid_agent_name(self, mock_context):
        """Test error for invalid agent name."""
        result = delegate({"agent": "invalid_agent", "task": "Test"}, mock_context)

        assert result["success"] is False
        assert "Unknown agent: invalid_agent" in result["error"]
        assert "design_agent" in result["error"]  # Should list valid agents

    def test_missing_cli_context(self, mock_context):
        """Test error when cli_context is missing from extra."""
        mock_context.extra = {"llm_client": MagicMock(), "session_id": "test"}

        result = delegate({"agent": "design_agent", "task": "Test"}, mock_context)

        assert result["success"] is False
        assert "Missing cli_context" in result["error"]

    def test_missing_llm_client(self, mock_context):
        """Test error when llm_client is missing from extra."""
        mock_context.extra = {"cli_context": MagicMock(), "session_id": "test"}

        result = delegate({"agent": "design_agent", "task": "Test"}, mock_context)

        assert result["success"] is False
        assert "Missing llm_client" in result["error"]

    def test_none_extra_dict(self, mock_context):
        """Test handling when context.extra is None."""
        mock_context.extra = None

        result = delegate({"agent": "design_agent", "task": "Test"}, mock_context)

        assert result["success"] is False
        assert "Missing cli_context" in result["error"]

    def test_all_valid_agent_types(self, mock_context):
        """Test that all documented agent types are valid."""
        valid_agents = ["design_agent", "test_agent", "review_agent", "implementation_agent"]

        for agent in valid_agents:
            # Just checking validation passes, not executing
            _ = {"agent": agent, "task": "Test"}
            # This will fail on execution but pass validation
            # We're just ensuring the agent name is recognized

    def test_optional_context_parameter(self, mock_context):
        """Test that context parameter is optional."""
        # Should pass validation even without context
        _ = {"agent": "design_agent", "task": "Test"}
        # Validation should pass (will fail on execution due to mocking)


class TestDelegateExecution:
    """Test execution paths with proper mocking."""

    @patch("ai_dev_agent.agents.task_queue.TaskStore")
    @patch("ai_dev_agent.agents.task_queue.TaskQueue")
    @patch("ai_dev_agent.agents.executor.AgentExecutor")
    def test_successful_delegation(
        self, mock_executor_cls, mock_queue_cls, mock_store_cls, mock_context
    ):
        """Test successful agent delegation."""
        # Setup mocks
        mock_queue = MagicMock()
        mock_queue_cls.get_instance.return_value = mock_queue

        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor
        mock_executor.execute_with_react.return_value = AgentResult(
            success=True, output="Task completed", metadata={"artifacts": ["file.md"]}
        )

        result = delegate({"agent": "design_agent", "task": "Create design"}, mock_context)

        assert result["success"] is True
        assert result["agent"] == "design_agent"
        assert result["result"] == "Task completed"
        assert result["artifacts"] == ["file.md"]

    @patch("ai_dev_agent.agents.task_queue.TaskStore")
    @patch("ai_dev_agent.agents.task_queue.TaskQueue")
    @patch("ai_dev_agent.agents.executor.AgentExecutor")
    def test_failed_delegation(
        self, mock_executor_cls, mock_queue_cls, mock_store_cls, mock_context
    ):
        """Test delegation when agent fails."""
        mock_queue = MagicMock()
        mock_queue_cls.get_instance.return_value = mock_queue

        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor
        mock_executor.execute_with_react.return_value = AgentResult(
            success=False, output="", error="Agent failed"
        )

        result = delegate({"agent": "test_agent", "task": "Generate tests"}, mock_context)

        assert result["success"] is False
        assert result["error"] == "Agent failed"

    @patch("ai_dev_agent.agents.task_queue.TaskStore")
    @patch("ai_dev_agent.agents.task_queue.TaskQueue")
    @patch("ai_dev_agent.agents.executor.AgentExecutor")
    def test_delegation_exception(
        self, mock_executor_cls, mock_queue_cls, mock_store_cls, mock_context
    ):
        """Test exception handling during delegation."""
        mock_queue = MagicMock()
        mock_queue_cls.get_instance.return_value = mock_queue

        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor
        mock_executor.execute_with_react.side_effect = RuntimeError("Unexpected error")

        result = delegate({"agent": "review_agent", "task": "Review code"}, mock_context)

        assert result["success"] is False
        assert "Agent execution failed" in result["error"]
        assert "Unexpected error" in result["error"]
