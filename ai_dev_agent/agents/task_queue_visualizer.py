"""Visual task queue tracking using TodoWrite.

This module provides integration between the async task queue and Claude Code's
TodoWrite visualization system, enabling clean visual tracking of delegated tasks.
"""

from __future__ import annotations

from typing import Any, Callable

from ai_dev_agent.agents.task_queue import Task, TaskQueue, TaskStatus
from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Global callback for visualization updates
# This will be set by the CLI/agent context to call TodoWrite
_visualization_callback: Callable[[list[dict[str, Any]]], None] | None = None


def set_visualization_callback(callback: Callable[[list[dict[str, Any]]], None]) -> None:
    """Set the callback function for task visualization updates.

    Args:
        callback: Function that takes a list of todo items and updates the display
                 (typically calls TodoWrite in the Claude Code context)
    """
    global _visualization_callback
    _visualization_callback = callback
    LOGGER.debug("Task visualization callback registered")


def clear_visualization_callback() -> None:
    """Clear the visualization callback (useful for testing)."""
    global _visualization_callback
    _visualization_callback = None


def get_task_list_for_display() -> list[dict[str, Any]]:
    """Build todo list from all tasks in queue.

    Returns:
        List of todo items in TodoWrite format with fields:
        - content: Task description
        - status: "pending" | "in_progress" | "completed"
        - activeForm: Status-aware description for in-progress display
    """
    try:
        queue = TaskQueue.get_instance()
        all_tasks = queue.get_all_tasks()
    except Exception as exc:
        LOGGER.warning(f"Failed to get tasks for visualization: {exc}")
        return []

    if not all_tasks:
        return []

    todos = []
    for task in all_tasks:
        # Truncate prompt for display
        preview = task.prompt[:60] + "..." if len(task.prompt) > 60 else task.prompt

        # Map TaskStatus to todo status and create descriptive activeForm
        if task.status == TaskStatus.QUEUED:
            status = "pending"
            active_form = f"⏱️  Queued: {task.agent}: {preview}"
            content = f"{task.agent}: {preview}"

        elif task.status == TaskStatus.RUNNING:
            status = "in_progress"
            active_form = f"▶️  Running: {task.agent}: {preview}"
            content = f"{task.agent}: {preview}"

        elif task.status == TaskStatus.COMPLETED:
            status = "completed"
            # Add artifacts info if available
            artifacts_info = ""
            if task.result and task.result.metadata:
                artifacts = task.result.metadata.get("artifacts", [])
                if artifacts:
                    artifacts_preview = ", ".join(artifacts[:2])
                    if len(artifacts) > 2:
                        artifacts_preview += f", +{len(artifacts) - 2} more"
                    artifacts_info = f" (artifacts: {artifacts_preview})"

            active_form = f"✅ Completed: {task.agent}: {preview}{artifacts_info}"
            content = f"{task.agent}: {preview}{artifacts_info}"

        elif task.status == TaskStatus.FAILED:
            status = "completed"  # Show as completed but with error indicator
            error_preview = (
                task.error[:40] + "..." if task.error and len(task.error) > 40 else task.error
            )
            active_form = f"❌ Failed: {task.agent}: {preview}"
            content = f"{task.agent}: {preview} - Error: {error_preview}"

        else:  # CANCELLED
            status = "completed"
            active_form = f"⚠️  Cancelled: {task.agent}: {preview}"
            content = f"{task.agent}: {preview} (cancelled)"

        todos.append(
            {
                "content": content,
                "status": status,
                "activeForm": active_form,
            }
        )

    return todos


def update_task_visualization() -> None:
    """Update the task visualization display.

    This function outputs the current task queue state directly to stdout
    in a clean, readable format. No external dependencies required.
    """
    try:
        todos = get_task_list_for_display()
        if not todos:
            return  # No tasks to display

        if _visualization_callback:
            # Use registered callback if provided (for testing/custom integrations)
            _visualization_callback(todos)
        else:
            # Output directly to console in a clean format
            print("\n📋 Background Tasks:", flush=True)
            for todo in todos:
                status_icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}[
                    todo["status"]
                ]
                print(f"  {status_icon} {todo['content']}", flush=True)
            print(flush=True)  # Blank line after

    except Exception as exc:
        LOGGER.warning(f"Failed to update task visualization: {exc}")


def format_task_summary(task: Task) -> str:
    """Format a single task for human-readable summary.

    Args:
        task: Task to format

    Returns:
        Human-readable task summary string
    """
    preview = task.prompt[:60] + "..." if len(task.prompt) > 60 else task.prompt

    if task.status == TaskStatus.QUEUED:
        return f"[ ] {task.agent}: {preview}"
    elif task.status == TaskStatus.RUNNING:
        return f"[~] {task.agent}: {preview}"
    elif task.status == TaskStatus.COMPLETED:
        return f"[x] {task.agent}: {preview}"
    elif task.status == TaskStatus.FAILED:
        return f"[!] {task.agent}: {preview} (failed)"
    else:  # CANCELLED
        return f"[-] {task.agent}: {preview} (cancelled)"


def get_task_summary() -> str:
    """Get a formatted summary of all tasks in the queue.

    Returns:
        Multi-line string with task summary, empty string if no tasks
    """
    try:
        queue = TaskQueue.get_instance()
        all_tasks = queue.get_all_tasks()
    except Exception as exc:
        LOGGER.warning(f"Failed to get task summary: {exc}")
        return ""

    if not all_tasks:
        return ""

    lines = ["Background Tasks:"]
    for task in all_tasks:
        lines.append(format_task_summary(task))

    return "\n".join(lines)
