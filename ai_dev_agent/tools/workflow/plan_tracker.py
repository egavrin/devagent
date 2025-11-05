"""Plan execution tracker for visualizing task progress.

This module provides console visualization for plan execution without
requiring async background workers. Tasks execute synchronously but
progress is displayed to the user in real-time.
"""

from __future__ import annotations

from typing import Any


class PlanTracker:
    """Track and visualize plan execution progress."""

    def __init__(self):
        """Initialize plan tracker."""
        self._current_plan: dict[str, Any] | None = None
        self._task_status: dict[str, str] = {}  # task_id -> status

    def start_plan(self, plan: dict[str, Any]) -> None:
        """Start tracking a new plan.

        Args:
            plan: Plan structure with goal and tasks
        """
        self._current_plan = plan
        self._task_status = {task["id"]: "pending" for task in plan["tasks"]}
        self._display_plan()

    def update_task_status(self, task_id: str, status: str) -> None:
        """Update status of a specific task.

        Args:
            task_id: Task identifier
            status: New status (pending, in_progress, completed, failed)
        """
        if task_id in self._task_status:
            self._task_status[task_id] = status
            self._display_plan()

    def _display_plan(self) -> None:
        """Display current plan status with checkboxes."""
        if not self._current_plan:
            return

        print("\n📋 Work Plan Progress:", flush=True)
        print(f"Goal: {self._current_plan['goal']}", flush=True)
        print("", flush=True)

        for task in self._current_plan["tasks"]:
            task_id = task["id"]
            status = self._task_status.get(task_id, "pending")

            # Status icons
            icon = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "completed": "[✓]",
                "failed": "[✗]",
            }.get(status, "[ ]")

            agent_label = task["agent"].replace("_agent", "")
            print(f"  {icon} {task['title']} ({agent_label})", flush=True)

            # Show description for in-progress or failed tasks
            if status in ("in_progress", "failed"):
                print(f"      {task['description']}", flush=True)

        print("", flush=True)

    def clear(self) -> None:
        """Clear current plan."""
        self._current_plan = None
        self._task_status = {}


# Global singleton instance
_tracker: PlanTracker | None = None


def get_tracker() -> PlanTracker:
    """Get or create the global plan tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = PlanTracker()
    return _tracker


def start_plan_tracking(plan: dict[str, Any]) -> None:
    """Start tracking a plan.

    Args:
        plan: Plan structure with goal and tasks
    """
    get_tracker().start_plan(plan)


def update_task_status(task_id: str, status: str) -> None:
    """Update task status and refresh display.

    Args:
        task_id: Task identifier
        status: New status (pending, in_progress, completed, failed)
    """
    get_tracker().update_task_status(task_id, status)


def clear_plan() -> None:
    """Clear the current plan."""
    get_tracker().clear()


__all__ = [
    "PlanTracker",
    "get_tracker",
    "start_plan_tracking",
    "update_task_status",
    "clear_plan",
]
