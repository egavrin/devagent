"""Get task status tool for checking async delegation progress."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ai_dev_agent.agents.task_queue import TaskQueue, TaskStatus
from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from ai_dev_agent.tools.registry import ToolContext

LOGGER = get_logger(__name__)


def get_task_status(payload: Mapping[str, Any], context: "ToolContext") -> Mapping[str, Any]:
    """
    Check the status of a queued or running task.

    Args:
        payload: {"task_id": "Task identifier from delegate"}
        context: Tool execution context

    Returns:
        {"success": bool, "task_id": str, "status": str, "agent": str,
         "result": str, "artifacts": list, "error": str, ...}
    """
    task_id = payload.get("task_id")

    if not task_id:
        return {
            "success": False,
            "error": "Missing required parameter: task_id",
        }

    try:
        # Get task from queue
        task_queue = TaskQueue.get_instance()
        task = task_queue.get_task(task_id)

        if not task:
            return {
                "success": False,
                "error": f"Task {task_id} not found. It may have been deleted or never existed.",
            }

        # Build response based on task status
        response = {
            "success": True,
            "task_id": task.id,
            "status": task.status.value,
            "agent": task.agent,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }

        # Add result if completed
        if task.status == TaskStatus.COMPLETED and task.result:
            output = task.result.output or "Task completed"
            artifacts = task.result.metadata.get("artifacts", []) if task.result.metadata else []

            response["result"] = output
            response["artifacts"] = artifacts
            response["error"] = None

        elif task.status == TaskStatus.FAILED:
            response["result"] = None
            response["artifacts"] = []
            response["error"] = task.error or "Task failed with unknown error"

        elif task.status == TaskStatus.RUNNING:
            preview = task.prompt[:60] + "..." if len(task.prompt) > 60 else task.prompt
            response["result"] = f"Running {task.agent}: {preview.lower()}"
            response["artifacts"] = []
            response["error"] = None

        elif task.status == TaskStatus.QUEUED:
            preview = task.prompt[:60] + "..." if len(task.prompt) > 60 else task.prompt
            response["result"] = f"Queued {task.agent} to {preview.lower()}"
            response["artifacts"] = []
            response["error"] = None

        else:  # CANCELLED
            preview = task.prompt[:60] + "..." if len(task.prompt) > 60 else task.prompt
            response["result"] = f"Cancelled {task.agent}: {preview.lower()}"
            response["artifacts"] = []
            response["error"] = None

        return response

    except Exception as exc:
        LOGGER.exception("Failed to get task status for %s", task_id)
        return {
            "success": False,
            "error": f"Failed to retrieve task status: {exc!s}",
        }
