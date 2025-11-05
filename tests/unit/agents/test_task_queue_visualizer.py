"""Unit tests for task queue visualization."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.agents.task_queue import Task, TaskQueue, TaskStatus
from ai_dev_agent.agents.task_queue_visualizer import (
    clear_visualization_callback,
    format_task_summary,
    get_task_list_for_display,
    get_task_summary,
    set_visualization_callback,
    update_task_visualization,
)


@pytest.fixture
def mock_task_queue(monkeypatch):
    """Mock TaskQueue singleton to avoid state pollution."""
    mock_queue = MagicMock(spec=TaskQueue)
    monkeypatch.setattr(
        "ai_dev_agent.agents.task_queue_visualizer.TaskQueue.get_instance",
        lambda: mock_queue,
    )
    return mock_queue


@pytest.fixture(autouse=True)
def clear_callback():
    """Clear visualization callback before and after each test."""
    clear_visualization_callback()
    yield
    clear_visualization_callback()


class TestVisualizationCallback:
    """Test callback registration and clearing."""

    def test_set_visualization_callback(self):
        """Test setting a visualization callback."""
        callback = MagicMock()
        set_visualization_callback(callback)

        # Verify callback is set (we can't directly inspect it, but we can test usage)
        # This is tested indirectly in other tests
        clear_visualization_callback()

    def test_clear_visualization_callback(self):
        """Test clearing the visualization callback."""
        callback = MagicMock()
        set_visualization_callback(callback)
        clear_visualization_callback()

        # After clearing, update should not call the callback
        # This is tested in update tests


class TestGetTaskListForDisplay:
    """Test get_task_list_for_display function."""

    def test_returns_empty_list_when_no_tasks(self, mock_task_queue):
        """Test that empty list is returned when queue has no tasks."""
        mock_task_queue.get_all_tasks.return_value = []

        result = get_task_list_for_display()

        assert result == []

    def test_returns_empty_list_on_exception(self, mock_task_queue):
        """Test that empty list is returned when queue raises exception."""
        mock_task_queue.get_all_tasks.side_effect = Exception("Queue error")

        result = get_task_list_for_display()

        assert result == []

    def test_formats_queued_task(self, mock_task_queue):
        """Test formatting of a queued task."""
        task = Task(
            id="task-1",
            agent="design_agent",
            prompt="Design REST API",
            status=TaskStatus.QUEUED,
            created_at=datetime.now(),
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        result = get_task_list_for_display()

        assert len(result) == 1
        assert result[0]["status"] == "pending"
        assert "design_agent" in result[0]["content"]
        assert "Design REST API" in result[0]["content"]
        assert "⏱️" in result[0]["activeForm"]
        assert "Queued" in result[0]["activeForm"]

    def test_formats_running_task(self, mock_task_queue):
        """Test formatting of a running task."""
        task = Task(
            id="task-2",
            agent="test_agent",
            prompt="Generate unit tests",
            status=TaskStatus.RUNNING,
            created_at=datetime.now(),
            started_at=datetime.now(),
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        result = get_task_list_for_display()

        assert len(result) == 1
        assert result[0]["status"] == "in_progress"
        assert "test_agent" in result[0]["content"]
        assert "▶️" in result[0]["activeForm"]
        assert "Running" in result[0]["activeForm"]

    def test_formats_completed_task(self, mock_task_queue):
        """Test formatting of a completed task."""
        task = Task(
            id="task-3",
            agent="review_agent",
            prompt="Review code changes",
            status=TaskStatus.COMPLETED,
            created_at=datetime.now(),
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        result = get_task_list_for_display()

        assert len(result) == 1
        assert result[0]["status"] == "completed"
        assert "review_agent" in result[0]["content"]
        assert "✅" in result[0]["activeForm"]
        assert "Completed" in result[0]["activeForm"]

    def test_formats_completed_task_with_artifacts(self, mock_task_queue):
        """Test formatting of a completed task with artifacts."""
        # Create a mock result object with metadata
        mock_result = MagicMock()
        mock_result.metadata = {"artifacts": ["src/feature.py", "tests/test_feature.py"]}

        task = Task(
            id="task-4",
            agent="impl_agent",
            prompt="Implement feature X",
            status=TaskStatus.COMPLETED,
            created_at=datetime.now(),
            result=mock_result,
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        result = get_task_list_for_display()

        assert len(result) == 1
        assert "artifacts" in result[0]["content"]
        assert "src/feature.py" in result[0]["content"]

    def test_formats_completed_task_with_many_artifacts(self, mock_task_queue):
        """Test formatting of a completed task with many artifacts (truncation)."""
        # Create a mock result object with metadata
        mock_result = MagicMock()
        mock_result.metadata = {"artifacts": ["file1.py", "file2.py", "file3.py", "file4.py"]}

        task = Task(
            id="task-5",
            agent="impl_agent",
            prompt="Implement multiple features",
            status=TaskStatus.COMPLETED,
            created_at=datetime.now(),
            result=mock_result,
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        result = get_task_list_for_display()

        assert len(result) == 1
        assert "+2 more" in result[0]["content"]

    def test_formats_failed_task(self, mock_task_queue):
        """Test formatting of a failed task."""
        task = Task(
            id="task-6",
            agent="test_agent",
            prompt="Run tests",
            status=TaskStatus.FAILED,
            created_at=datetime.now(),
            error="Tests failed: 3 failures",
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        result = get_task_list_for_display()

        assert len(result) == 1
        assert result[0]["status"] == "completed"  # Marked as completed but with error
        assert "❌" in result[0]["activeForm"]
        assert "Failed" in result[0]["activeForm"]
        assert "Error" in result[0]["content"]

    def test_formats_failed_task_with_long_error(self, mock_task_queue):
        """Test formatting of a failed task with truncated error."""
        long_error = "x" * 100
        task = Task(
            id="task-7",
            agent="test_agent",
            prompt="Run tests",
            status=TaskStatus.FAILED,
            created_at=datetime.now(),
            error=long_error,
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        result = get_task_list_for_display()

        assert len(result) == 1
        assert "..." in result[0]["content"]

    def test_formats_cancelled_task(self, mock_task_queue):
        """Test formatting of a cancelled task."""
        task = Task(
            id="task-8",
            agent="design_agent",
            prompt="Design feature",
            status=TaskStatus.CANCELLED,
            created_at=datetime.now(),
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        result = get_task_list_for_display()

        assert len(result) == 1
        assert result[0]["status"] == "completed"
        assert "⚠️" in result[0]["activeForm"]
        assert "Cancelled" in result[0]["activeForm"]
        assert "cancelled" in result[0]["content"]

    def test_truncates_long_prompt(self, mock_task_queue):
        """Test that long prompts are truncated in display."""
        long_prompt = "x" * 100
        task = Task(
            id="task-9",
            agent="test_agent",
            prompt=long_prompt,
            status=TaskStatus.QUEUED,
            created_at=datetime.now(),
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        result = get_task_list_for_display()

        assert len(result) == 1
        assert "..." in result[0]["content"]
        assert len(result[0]["content"]) < len(long_prompt)

    def test_formats_multiple_tasks(self, mock_task_queue):
        """Test formatting multiple tasks with different statuses."""
        tasks = [
            Task(
                id="task-10",
                agent="design",
                prompt="Design",
                status=TaskStatus.COMPLETED,
                created_at=datetime.now(),
            ),
            Task(
                id="task-11",
                agent="impl",
                prompt="Implement",
                status=TaskStatus.RUNNING,
                created_at=datetime.now(),
            ),
            Task(
                id="task-12",
                agent="test",
                prompt="Test",
                status=TaskStatus.QUEUED,
                created_at=datetime.now(),
            ),
        ]
        mock_task_queue.get_all_tasks.return_value = tasks

        result = get_task_list_for_display()

        assert len(result) == 3
        assert result[0]["status"] == "completed"
        assert result[1]["status"] == "in_progress"
        assert result[2]["status"] == "pending"


class TestUpdateTaskVisualization:
    """Test update_task_visualization function."""

    def test_does_nothing_when_no_tasks(self, mock_task_queue):
        """Test that nothing is printed when there are no tasks."""
        mock_task_queue.get_all_tasks.return_value = []

        # Should not raise and should not print anything
        update_task_visualization()

    def test_uses_callback_when_set(self, mock_task_queue):
        """Test that callback is used when set."""
        task = Task(
            id="task-1",
            agent="test",
            prompt="Test",
            status=TaskStatus.QUEUED,
            created_at=datetime.now(),
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        callback = MagicMock()
        set_visualization_callback(callback)

        update_task_visualization()

        callback.assert_called_once()
        todos = callback.call_args[0][0]
        assert len(todos) == 1

    def test_prints_to_stdout_when_no_callback(self, mock_task_queue, capsys):
        """Test that tasks are printed to stdout when no callback is set."""
        task = Task(
            id="task-2",
            agent="test_agent",
            prompt="Run tests",
            status=TaskStatus.RUNNING,
            created_at=datetime.now(),
        )
        mock_task_queue.get_all_tasks.return_value = [task]

        update_task_visualization()

        captured = capsys.readouterr()
        assert "Background Tasks:" in captured.out
        assert "test_agent" in captured.out
        assert "[~]" in captured.out  # in_progress icon

    def test_handles_exception_gracefully(self, mock_task_queue):
        """Test that exceptions are handled gracefully."""
        mock_task_queue.get_all_tasks.side_effect = Exception("Queue error")

        # Should not raise
        update_task_visualization()


class TestFormatTaskSummary:
    """Test format_task_summary function."""

    def test_formats_queued_task(self):
        """Test formatting queued task."""
        task = Task(
            id="task-1",
            agent="test",
            prompt="Test",
            status=TaskStatus.QUEUED,
            created_at=datetime.now(),
        )

        result = format_task_summary(task)

        assert "[ ]" in result
        assert "test" in result

    def test_formats_running_task(self):
        """Test formatting running task."""
        task = Task(
            id="task-2",
            agent="impl",
            prompt="Implement",
            status=TaskStatus.RUNNING,
            created_at=datetime.now(),
        )

        result = format_task_summary(task)

        assert "[~]" in result

    def test_formats_completed_task(self):
        """Test formatting completed task."""
        task = Task(
            id="task-3",
            agent="review",
            prompt="Review",
            status=TaskStatus.COMPLETED,
            created_at=datetime.now(),
        )

        result = format_task_summary(task)

        assert "[x]" in result

    def test_formats_failed_task(self):
        """Test formatting failed task."""
        task = Task(
            id="task-4",
            agent="test",
            prompt="Test",
            status=TaskStatus.FAILED,
            created_at=datetime.now(),
            error="Test error",
        )

        result = format_task_summary(task)

        assert "[!]" in result
        assert "failed" in result

    def test_formats_cancelled_task(self):
        """Test formatting cancelled task."""
        task = Task(
            id="task-5",
            agent="design",
            prompt="Design",
            status=TaskStatus.CANCELLED,
            created_at=datetime.now(),
        )

        result = format_task_summary(task)

        assert "[-]" in result
        assert "cancelled" in result


class TestGetTaskSummary:
    """Test get_task_summary function."""

    def test_returns_empty_string_when_no_tasks(self, mock_task_queue):
        """Test that empty string is returned when no tasks."""
        mock_task_queue.get_all_tasks.return_value = []

        result = get_task_summary()

        assert result == ""

    def test_returns_empty_string_on_exception(self, mock_task_queue):
        """Test that empty string is returned on exception."""
        mock_task_queue.get_all_tasks.side_effect = Exception("Queue error")

        result = get_task_summary()

        assert result == ""

    def test_formats_multiple_tasks(self, mock_task_queue):
        """Test formatting multiple tasks into summary."""
        tasks = [
            Task(
                id="task-1",
                agent="design",
                prompt="Design",
                status=TaskStatus.COMPLETED,
                created_at=datetime.now(),
            ),
            Task(
                id="task-2",
                agent="impl",
                prompt="Implement",
                status=TaskStatus.RUNNING,
                created_at=datetime.now(),
            ),
        ]
        mock_task_queue.get_all_tasks.return_value = tasks

        result = get_task_summary()

        assert "Background Tasks:" in result
        assert "design" in result
        assert "impl" in result
        assert "[x]" in result  # completed
        assert "[~]" in result  # running
