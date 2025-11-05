"""Unit tests for task queue infrastructure."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.agents.task_queue import (
    Task,
    TaskExecutor,
    TaskQueue,
    TaskStatus,
    TaskStore,
    create_task,
)


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_task_status_values(self):
        """Test that all expected task statuses exist."""
        assert TaskStatus.QUEUED.value == "queued"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTask:
    """Test Task dataclass."""

    def test_create_task_with_required_fields(self):
        """Test creating a task with required fields."""
        now = datetime.now()
        task = Task(
            id="task-123",
            agent="design_agent",
            prompt="Design REST API",
            status=TaskStatus.QUEUED,
            created_at=now,
        )

        assert task.id == "task-123"
        assert task.agent == "design_agent"
        assert task.prompt == "Design REST API"
        assert task.status == TaskStatus.QUEUED
        assert task.created_at == now
        assert task.result is None
        assert task.error is None
        assert task.context == {}

    def test_task_to_dict(self):
        """Test converting task to dictionary."""
        now = datetime.now()
        task = Task(
            id="task-456",
            agent="test_agent",
            prompt="Write tests",
            status=TaskStatus.COMPLETED,
            created_at=now,
            started_at=now,
            completed_at=now,
            error="Test error",
        )

        data = task.to_dict()

        assert data["id"] == "task-456"
        assert data["agent"] == "test_agent"
        assert data["prompt"] == "Write tests"
        assert data["status"] == "completed"
        assert data["created_at"] == now.isoformat()
        assert data["started_at"] == now.isoformat()
        assert data["completed_at"] == now.isoformat()
        assert data["error"] == "Test error"

    def test_task_from_dict(self):
        """Test reconstructing task from dictionary."""
        now = datetime.now()
        data = {
            "id": "task-789",
            "agent": "review_agent",
            "prompt": "Review code",
            "status": "running",
            "created_at": now.isoformat(),
            "started_at": now.isoformat(),
            "completed_at": None,
            "error": None,
        }

        task = Task.from_dict(data)

        assert task.id == "task-789"
        assert task.agent == "review_agent"
        assert task.prompt == "Review code"
        assert task.status == TaskStatus.RUNNING
        assert task.created_at == now
        assert task.started_at == now
        assert task.completed_at is None
        assert task.error is None


class TestTaskQueue:
    """Test TaskQueue."""

    def setup_method(self):
        """Reset singleton before each test."""
        TaskQueue.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        TaskQueue.reset_instance()

    def test_singleton_pattern(self):
        """Test that TaskQueue is a singleton."""
        queue1 = TaskQueue.get_instance()
        queue2 = TaskQueue.get_instance()
        assert queue1 is queue2

    def test_enqueue_and_dequeue(self):
        """Test basic enqueue and dequeue operations."""
        queue = TaskQueue.get_instance()
        task = create_task("design_agent", "Design API")

        queue.enqueue(task)
        assert queue.get_queue_size() == 1

        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.id == task.id
        assert queue.get_queue_size() == 0

    def test_dequeue_empty_queue(self):
        """Test dequeue returns None when queue is empty."""
        queue = TaskQueue.get_instance()
        assert queue.dequeue() is None

    def test_fifo_order(self):
        """Test that tasks are dequeued in FIFO order."""
        queue = TaskQueue.get_instance()
        task1 = create_task("design_agent", "Task 1")
        task2 = create_task("test_agent", "Task 2")
        task3 = create_task("review_agent", "Task 3")

        queue.enqueue(task1)
        queue.enqueue(task2)
        queue.enqueue(task3)

        assert queue.dequeue().id == task1.id
        assert queue.dequeue().id == task2.id
        assert queue.dequeue().id == task3.id

    def test_get_task(self):
        """Test retrieving task by ID."""
        queue = TaskQueue.get_instance()
        task = create_task("design_agent", "Design API")
        queue.enqueue(task)

        retrieved = queue.get_task(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id

    def test_get_nonexistent_task(self):
        """Test retrieving non-existent task returns None."""
        queue = TaskQueue.get_instance()
        assert queue.get_task("nonexistent") is None

    def test_update_task(self):
        """Test updating task status."""
        queue = TaskQueue.get_instance()
        task = create_task("design_agent", "Design API")
        queue.enqueue(task)

        task.status = TaskStatus.RUNNING
        queue.update_task(task)

        retrieved = queue.get_task(task.id)
        assert retrieved.status == TaskStatus.RUNNING

    def test_get_all_tasks(self):
        """Test retrieving all tasks."""
        queue = TaskQueue.get_instance()
        task1 = create_task("design_agent", "Task 1")
        task2 = create_task("test_agent", "Task 2")

        queue.enqueue(task1)
        queue.enqueue(task2)

        all_tasks = queue.get_all_tasks()
        assert len(all_tasks) == 2
        task_ids = {t.id for t in all_tasks}
        assert task1.id in task_ids
        assert task2.id in task_ids

    def test_thread_safety_enqueue(self):
        """Test thread-safe enqueue operations."""
        queue = TaskQueue.get_instance()
        tasks = []

        def enqueue_tasks():
            for i in range(10):
                task = create_task("design_agent", f"Task {i}")
                tasks.append(task)
                queue.enqueue(task)
                time.sleep(0.001)  # Simulate work

        threads = [threading.Thread(target=enqueue_tasks) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 50 tasks total (5 threads × 10 tasks)
        assert len(queue.get_all_tasks()) == 50

    def test_thread_safety_dequeue(self):
        """Test thread-safe dequeue operations."""
        queue = TaskQueue.get_instance()

        # Enqueue 50 tasks
        for i in range(50):
            queue.enqueue(create_task("design_agent", f"Task {i}"))

        dequeued_tasks = []
        lock = threading.Lock()

        def dequeue_tasks():
            while True:
                task = queue.dequeue()
                if task is None:
                    break
                with lock:
                    dequeued_tasks.append(task)
                time.sleep(0.001)

        threads = [threading.Thread(target=dequeue_tasks) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have dequeued all 50 tasks
        assert len(dequeued_tasks) == 50
        # No duplicates
        task_ids = [t.id for t in dequeued_tasks]
        assert len(task_ids) == len(set(task_ids))


class TestTaskStore:
    """Test TaskStore."""

    def test_save_and_get_task(self):
        """Test saving and retrieving task."""
        store = TaskStore()
        task = create_task("design_agent", "Design API")

        store.save_task(task)
        retrieved = store.get_task(task.id)

        assert retrieved is not None
        assert retrieved.id == task.id
        assert retrieved.agent == "design_agent"
        assert retrieved.prompt == "Design API"

        # Cleanup
        store.delete_task(task.id)

    def test_get_nonexistent_task(self):
        """Test getting non-existent task returns None."""
        store = TaskStore()
        assert store.get_task("nonexistent") is None

    def test_delete_task(self):
        """Test deleting task."""
        store = TaskStore()
        task = create_task("design_agent", "Design API")

        store.save_task(task)
        assert store.get_task(task.id) is not None

        store.delete_task(task.id)
        assert store.get_task(task.id) is None


class TestTaskExecutor:
    """Test TaskExecutor."""

    def setup_method(self):
        """Reset singletons before each test."""
        TaskQueue.reset_instance()
        TaskExecutor.reset_instance()

    def teardown_method(self):
        """Stop executor and reset singletons after each test."""
        try:
            executor = TaskExecutor.get_instance()
            executor.stop(timeout=2.0)
        except Exception:
            pass
        TaskQueue.reset_instance()
        TaskExecutor.reset_instance()

    def test_singleton_pattern(self):
        """Test that TaskExecutor is a singleton."""
        executor1 = TaskExecutor.get_instance()
        executor2 = TaskExecutor.get_instance()
        assert executor1 is executor2

    def test_start_and_stop(self):
        """Test starting and stopping executor."""
        executor = TaskExecutor.get_instance(num_workers=1)

        executor.start()
        assert executor._running is True
        assert len(executor.workers) == 1

        executor.stop(timeout=2.0)
        assert executor._running is False
        assert len(executor.workers) == 0

    def test_cannot_start_twice(self):
        """Test that starting executor twice logs warning."""
        executor = TaskExecutor.get_instance()

        executor.start()
        # Second start should log warning but not crash
        executor.start()

        executor.stop(timeout=2.0)

    def test_execute_task_success(self):
        """Test successful task execution."""
        queue = TaskQueue.get_instance()
        executor = TaskExecutor.get_instance(num_workers=1)

        # Mock agent execution
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "Design completed"
        mock_result.metadata = {"artifacts": ["design.md"]}
        mock_result.error = None

        with patch("ai_dev_agent.agents.executor.AgentExecutor") as mock_executor_class:
            mock_exec = mock_executor_class.return_value
            mock_exec.execute_with_react.return_value = mock_result

            # Create and enqueue task
            task = create_task(
                "design_agent",
                "Design REST API",
                context={
                    "session_id": "test-session",
                    "workspace": "/tmp",
                    "cli_context": MagicMock(),
                    "llm_client": MagicMock(),
                },
            )
            queue.enqueue(task)

            # Start executor
            executor.start()

            # Wait for task to complete
            time.sleep(2.0)

            # Check task was completed
            completed_task = queue.get_task(task.id)
            assert completed_task.status == TaskStatus.COMPLETED
            assert completed_task.result == mock_result

        executor.stop(timeout=2.0)

    def test_execute_task_failure(self):
        """Test task execution failure."""
        queue = TaskQueue.get_instance()
        executor = TaskExecutor.get_instance(num_workers=1)

        with patch("ai_dev_agent.agents.executor.AgentExecutor") as mock_executor_class:
            mock_exec = mock_executor_class.return_value
            mock_exec.execute_with_react.side_effect = RuntimeError("Execution failed")

            # Create and enqueue task
            task = create_task(
                "design_agent",
                "Design REST API",
                context={
                    "session_id": "test-session",
                    "workspace": "/tmp",
                    "cli_context": MagicMock(),
                    "llm_client": MagicMock(),
                },
            )
            queue.enqueue(task)

            # Start executor
            executor.start()

            # Wait for task to fail
            time.sleep(2.0)

            # Check task failed
            failed_task = queue.get_task(task.id)
            assert failed_task.status == TaskStatus.FAILED
            assert "Execution failed" in failed_task.error

        executor.stop(timeout=2.0)

    def test_invalid_agent_name(self):
        """Test execution with invalid agent name."""
        queue = TaskQueue.get_instance()
        executor = TaskExecutor.get_instance(num_workers=1)

        task = create_task(
            "invalid_agent",
            "Do something",
            context={
                "session_id": "test-session",
                "workspace": "/tmp",
                "cli_context": MagicMock(),
                "llm_client": MagicMock(),
            },
        )
        queue.enqueue(task)

        executor.start()
        time.sleep(2.0)

        failed_task = queue.get_task(task.id)
        assert failed_task.status == TaskStatus.FAILED
        assert "Unknown agent" in failed_task.error

        executor.stop(timeout=2.0)


class TestCreateTask:
    """Test create_task helper function."""

    def test_create_task_basic(self):
        """Test creating task with basic parameters."""
        task = create_task("design_agent", "Design API")

        assert task.id.startswith("task-")
        assert len(task.id) == 13  # "task-" + 8 hex chars
        assert task.agent == "design_agent"
        assert task.prompt == "Design API"
        assert task.status == TaskStatus.QUEUED
        assert task.context == {}

    def test_create_task_with_context(self):
        """Test creating task with context."""
        context = {"session_id": "test-123", "workspace": "/tmp"}
        task = create_task("test_agent", "Write tests", context)

        assert task.context == context

    def test_create_task_unique_ids(self):
        """Test that each task gets a unique ID."""
        task1 = create_task("design_agent", "Task 1")
        task2 = create_task("design_agent", "Task 2")

        assert task1.id != task2.id
