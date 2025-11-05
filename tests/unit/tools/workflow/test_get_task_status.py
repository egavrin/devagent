"""Tests for get_task_status tool."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from ai_dev_agent.agents.base import AgentResult
from ai_dev_agent.agents.task_queue import Task, TaskStatus

# Import function for testing, but get module reference for patching
from ai_dev_agent.tools.workflow.get_task_status import get_task_status

# Get actual module object (not the function) for proper mocking
get_task_status_module = sys.modules["ai_dev_agent.tools.workflow.get_task_status"]


@pytest.fixture
def mock_context():
    """Create mock ToolContext."""
    context = Mock()
    context.repo_root = "/test"
    return context


def create_task(task_id: str, agent: str, prompt: str, status: TaskStatus) -> Task:
    """Helper to create a task."""
    return Task(
        id=task_id,
        agent=agent,
        prompt=prompt,
        status=status,
        created_at=datetime(2025, 1, 1, 10, 0, 0),
        context={},
    )


class TestValidation:
    """Test input validation."""

    def test_missing_task_id(self, mock_context):
        """Missing task_id should return error."""
        result = get_task_status({}, mock_context)
        assert result["success"] is False
        assert "Missing required parameter: task_id" in result["error"]

    def test_empty_task_id(self, mock_context):
        """Empty task_id should return error."""
        result = get_task_status({"task_id": ""}, mock_context)
        assert result["success"] is False
        assert "Missing required parameter: task_id" in result["error"]

    def test_none_task_id(self, mock_context):
        """None task_id should return error."""
        result = get_task_status({"task_id": None}, mock_context)
        assert result["success"] is False
        assert "Missing required parameter: task_id" in result["error"]


class TestCompletedTask:
    """Test completed task scenarios."""

    def test_completed_with_artifacts(self, mock_context):
        """Completed task with artifacts."""
        task = create_task("t1", "design_agent", "Create design", TaskStatus.COMPLETED)
        task.completed_at = datetime(2025, 1, 1, 10, 30, 0)
        task.result = AgentResult(
            success=True,
            output="Design created",
            metadata={"artifacts": ["design.md", "diagram.png"]},
        )

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t1"}, mock_context)

        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["result"] == "Design created"
        assert result["artifacts"] == ["design.md", "diagram.png"]
        assert result["error"] is None

    def test_completed_no_artifacts(self, mock_context):
        """Completed task without artifacts."""
        task = create_task("t2", "review_agent", "Review", TaskStatus.COMPLETED)
        task.result = AgentResult(success=True, output="Review done", metadata={})

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t2"}, mock_context)

        assert result["success"] is True
        assert result["artifacts"] == []

    def test_completed_no_metadata(self, mock_context):
        """Completed task with None metadata."""
        task = create_task("t3", "test_agent", "Test", TaskStatus.COMPLETED)
        task.result = AgentResult(success=True, output="Done", metadata=None)

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t3"}, mock_context)

        assert result["success"] is True
        assert result["artifacts"] == []

    def test_completed_no_output(self, mock_context):
        """Completed task with no output uses default."""
        task = create_task("t4", "impl_agent", "Impl", TaskStatus.COMPLETED)
        task.result = AgentResult(success=True, output=None, metadata={})

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t4"}, mock_context)

        assert result["success"] is True
        assert result["result"] == "Task completed"


class TestFailedTask:
    """Test failed task scenarios."""

    def test_failed_with_error(self, mock_context):
        """Failed task with error message."""
        task = create_task("t5", "test_agent", "Generate tests", TaskStatus.FAILED)
        task.error = "Syntax error in generated code"

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t5"}, mock_context)

        assert result["success"] is True
        assert result["status"] == "failed"
        assert result["result"] is None
        assert result["artifacts"] == []
        assert result["error"] == "Syntax error in generated code"

    def test_failed_no_error(self, mock_context):
        """Failed task without error message uses default."""
        task = create_task("t6", "design_agent", "Design", TaskStatus.FAILED)
        task.error = None

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t6"}, mock_context)

        assert result["success"] is True
        assert "unknown error" in result["error"].lower()


class TestRunningTask:
    """Test running task scenarios."""

    def test_running_short_prompt(self, mock_context):
        """Running task with short prompt."""
        task = create_task("t7", "review_agent", "Review code", TaskStatus.RUNNING)

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t7"}, mock_context)

        assert result["success"] is True
        assert result["status"] == "running"
        assert "Running review_agent" in result["result"]
        assert "review code" in result["result"]
        assert result["error"] is None

    def test_running_long_prompt(self, mock_context):
        """Running task with long prompt gets truncated."""
        long_prompt = "A" * 100
        task = create_task("t8", "impl_agent", long_prompt, TaskStatus.RUNNING)

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t8"}, mock_context)

        assert result["success"] is True
        assert "..." in result["result"]
        assert len(result["result"]) < len(long_prompt) + 50


class TestQueuedTask:
    """Test queued task scenarios."""

    def test_queued_task(self, mock_context):
        """Queued task shows queued status."""
        task = create_task("t9", "test_agent", "Generate tests", TaskStatus.QUEUED)

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t9"}, mock_context)

        assert result["success"] is True
        assert result["status"] == "queued"
        assert "Queued test_agent" in result["result"]


class TestCancelledTask:
    """Test cancelled task scenarios."""

    def test_cancelled_task(self, mock_context):
        """Cancelled task shows cancelled status."""
        task = create_task("t10", "design_agent", "Design system", TaskStatus.CANCELLED)

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t10"}, mock_context)

        assert result["success"] is True
        assert result["status"] == "cancelled"
        assert "Cancelled design_agent" in result["result"]


class TestExceptionHandling:
    """Test exception scenarios."""

    def test_queue_exception(self, mock_context):
        """Exception in queue access is handled."""
        mock_queue = MagicMock()
        mock_queue.get_task.side_effect = RuntimeError("Database error")

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t11"}, mock_context)

        assert result["success"] is False
        assert "Failed to retrieve task status" in result["error"]
        assert "Database error" in result["error"]

    def test_queue_get_instance_fails(self, mock_context):
        """Exception getting queue instance is handled."""
        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.side_effect = RuntimeError("No queue")
            result = get_task_status({"task_id": "t12"}, mock_context)

        assert result["success"] is False
        assert "Failed to retrieve task status" in result["error"]


class TestTimestamps:
    """Test timestamp handling."""

    def test_timestamps_present(self, mock_context):
        """Timestamps are properly formatted."""
        task = create_task("t13", "test_agent", "Test", TaskStatus.COMPLETED)
        task.completed_at = datetime(2025, 1, 1, 11, 0, 0)
        task.result = AgentResult(success=True, output="Done")

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t13"}, mock_context)

        assert result["created_at"] == "2025-01-01T10:00:00"
        assert result["completed_at"] == "2025-01-01T11:00:00"

    def test_timestamps_none(self, mock_context):
        """None timestamps handled gracefully."""
        task = create_task("t14", "test_agent", "Test", TaskStatus.QUEUED)
        task.created_at = None
        task.completed_at = None

        mock_queue = MagicMock()
        mock_queue.get_task.return_value = task

        with patch.object(get_task_status_module, "TaskQueue") as mock_queue_cls:
            mock_queue_cls.get_instance.return_value = mock_queue
            result = get_task_status({"task_id": "t14"}, mock_context)

        assert result["created_at"] is None
        assert result["completed_at"] is None
