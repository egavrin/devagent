"""Async task queue for delegating work to specialized agents.

This module provides infrastructure for queuing agent tasks and executing them
asynchronously in background workers, avoiding nested ReAct conversation issues.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from ai_dev_agent.core.storage.short_term_memory import ShortTermMemory
from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from ai_dev_agent.agents.base import AgentResult

LOGGER = get_logger(__name__)

# Visualization support - imported dynamically to avoid circular dependencies
_visualization_module = None


def _get_visualizer():
    """Lazy-load task_queue_visualizer to avoid circular imports."""
    global _visualization_module
    if _visualization_module is None:
        try:
            from ai_dev_agent.agents import task_queue_visualizer

            _visualization_module = task_queue_visualizer
        except ImportError:
            LOGGER.debug("task_queue_visualizer not available")
            _visualization_module = False  # Mark as unavailable
    return _visualization_module if _visualization_module is not False else None


def _update_visualization():
    """Trigger task visualization update if available."""
    visualizer = _get_visualizer()
    if visualizer:
        visualizer.update_task_visualization()


class TaskStatus(Enum):
    """Status of a queued task."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a task to be executed by a specialized agent."""

    id: str
    agent: str
    prompt: str
    status: TaskStatus
    created_at: datetime
    result: Any = None  # AgentResult | None (Any to avoid circular import)
    error: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for storage."""
        return {
            "id": self.id,
            "agent": self.agent,
            "prompt": self.prompt,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            # Note: result and context are not serialized for simplicity
            # In a production system, these would be serialized separately
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Reconstruct task from dictionary."""
        return cls(
            id=data["id"],
            agent=data["agent"],
            prompt=data["prompt"],
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            error=data.get("error"),
        )


class TaskQueue:
    """Thread-safe task queue using short-term memory for agent delegation.

    Singleton pattern ensures a single queue instance across the application.
    Tasks are ephemeral and cleared when the process ends.
    """

    _instance: ClassVar[TaskQueue | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self):
        """Initialize task queue with short-term memory."""
        self._queue: deque[str] = deque()  # Task IDs only (FIFO order)
        self._queue_lock = threading.Lock()
        # Use short-term memory for task lookup (no persistence)
        self._tasks = ShortTermMemory[Task]()

    @classmethod
    def get_instance(cls) -> TaskQueue:
        """Get singleton instance of task queue."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = TaskQueue()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def enqueue(self, task: Task) -> None:
        """Add task to queue.

        Args:
            task: Task to enqueue
        """
        with self._queue_lock:
            self._queue.append(task.id)  # Store task ID

        self._tasks.set(task.id, task)  # ShortTermMemory handles locking

        # Update visualization (removes verbose logs)
        _update_visualization()

        # Keep minimal debug log
        LOGGER.debug(f"Task {task.id} enqueued for {task.agent}")

    def dequeue(self) -> Task | None:
        """Remove and return next task from queue (FIFO).

        Returns:
            Next task or None if queue is empty
        """
        with self._queue_lock:
            if not self._queue:
                return None
            task_id = self._queue.popleft()

        # Retrieve task from short-term memory
        return self._tasks.get(task_id)

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task if found, None otherwise
        """
        return self._tasks.get(task_id)

    def update_task(self, task: Task) -> None:
        """Update task status.

        Args:
            task: Updated task
        """
        self._tasks.set(task.id, task)  # ShortTermMemory handles locking

        # Don't spam visualization on every update
        # Only show on significant events (enqueue handles this)

    def get_queue_size(self) -> int:
        """Get number of tasks in queue (not yet dequeued).

        Returns:
            Queue size
        """
        with self._queue_lock:
            return len(self._queue)

    def get_all_tasks(self) -> list[Task]:
        """Get all tasks (queued, running, and completed).

        Returns:
            List of all tasks
        """
        task_ids = self._tasks.keys()
        return [task for tid in task_ids if (task := self._tasks.get(tid))]


# TaskStore class removed - tasks are now ephemeral (short-term memory only)
# Tasks are cleared when the devagent process ends, which is the intended behavior.


class TaskExecutor:
    """Background worker that executes queued tasks.

    Runs in separate threads to avoid blocking the main conversation.
    """

    _instance: ClassVar[TaskExecutor | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, queue: TaskQueue, num_workers: int = 1):
        """Initialize task executor.

        Args:
            queue: Task queue to pull from
            num_workers: Number of worker threads
        """
        self.queue = queue
        self.num_workers = num_workers
        self.workers: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._running = False

    @classmethod
    def get_instance(cls, num_workers: int = 1) -> TaskExecutor:
        """Get singleton instance of task executor.

        Args:
            num_workers: Number of worker threads (only used on first call)

        Returns:
            TaskExecutor instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    queue = TaskQueue.get_instance()
                    cls._instance = TaskExecutor(queue, num_workers)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            if cls._instance and cls._instance._running:
                cls._instance.stop()
            cls._instance = None

    def start(self) -> None:
        """Start background worker threads."""
        if self._running:
            LOGGER.warning("TaskExecutor already running")
            return

        self._stop_event.clear()
        self._running = True

        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"TaskWorker-{i}", daemon=True)
            worker.start()
            self.workers.append(worker)

        LOGGER.info(f"Started {self.num_workers} task worker(s)")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop background worker threads gracefully.

        Args:
            timeout: Maximum time to wait for workers to finish
        """
        if not self._running:
            return

        LOGGER.info("Stopping task workers...")
        self._stop_event.set()

        for worker in self.workers:
            worker.join(timeout=timeout)
            if worker.is_alive():
                LOGGER.warning(f"Worker {worker.name} did not stop gracefully")

        self.workers.clear()
        self._running = False
        LOGGER.info("Task workers stopped")

    def _worker_loop(self) -> None:
        """Main loop for worker thread."""
        LOGGER.debug(f"Worker {threading.current_thread().name} started")

        while not self._stop_event.is_set():
            task = self.queue.dequeue()
            if not task:
                # No tasks available, sleep briefly
                time.sleep(0.5)
                continue

            # Execute task
            self._execute_task(task)

        LOGGER.debug(f"Worker {threading.current_thread().name} stopped")

    def _execute_task(self, task: Task) -> None:
        """Execute a single task.

        Args:
            task: Task to execute
        """
        # Update status to running (visualization update happens in update_task)
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.queue.update_task(task)

        try:
            # Import here to avoid circular dependencies
            from ai_dev_agent.agents.base import AgentContext
            from ai_dev_agent.agents.executor import AgentExecutor
            from ai_dev_agent.agents.specialized import (
                DesignAgent,
                ImplementationAgent,
                ReviewAgent,
                TestingAgent,
            )

            # Map agent names to classes
            agent_map = {
                "design_agent": DesignAgent,
                "test_agent": TestingAgent,
                "review_agent": ReviewAgent,
                "implementation_agent": ImplementationAgent,
            }

            agent_class = agent_map.get(task.agent)
            if not agent_class:
                raise ValueError(f"Unknown agent: {task.agent}")

            # Instantiate agent
            agent_instance = agent_class()

            # Build context from task metadata
            agent_context = AgentContext(
                session_id=task.context.get("session_id", task.id),
                parent_id=task.context.get("parent_session_id"),
                working_directory=task.context.get("workspace", "."),
                metadata=task.context.get("metadata", {}),
            )

            # Execute agent
            # NOTE: This happens in a background thread, so LLM conversation
            # is isolated from the main conversation
            executor = AgentExecutor()
            result = executor.execute_with_react(
                agent=agent_instance,
                prompt=task.prompt,
                context=agent_context,
                ctx=task.context.get("cli_context"),
                cli_client=task.context.get("llm_client"),
            )

            # Store result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            # Visualization will show completion status
            LOGGER.debug(f"Task {task.id} completed successfully")

        except Exception as exc:
            # Keep ERROR log for debugging failures
            LOGGER.error(f"❌ {task.agent} failed: {exc!s}")
            task.error = str(exc)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()

        # Update task in short-term memory
        self.queue.update_task(task)

        # Note: Tasks are now ephemeral (short-term memory only).
        # They are automatically cleared when the devagent process ends.


def create_task(agent: str, prompt: str, context: dict[str, Any] | None = None) -> Task:
    """Create a new task with unique ID.

    Args:
        agent: Agent name to execute
        prompt: Task prompt/description
        context: Execution context

    Returns:
        New Task instance
    """
    task_id = f"task-{uuid4().hex[:8]}"
    return Task(
        id=task_id,
        agent=agent,
        prompt=prompt,
        status=TaskStatus.QUEUED,
        created_at=datetime.now(),
        context=context or {},
    )
