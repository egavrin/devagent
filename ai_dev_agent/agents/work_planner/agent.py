"""
Work Planning Agent

Manages work planning lifecycle including task breakdown, dependency management,
and progress tracking.
"""

from datetime import datetime
from typing import Any, Optional

from .models import Priority, Task, TaskStatus, WorkPlan
from .storage import WorkPlanStorage


class WorkPlanningAgent:
    """Manages work planning lifecycle"""

    def __init__(self, storage: Optional[WorkPlanStorage] = None):
        """
        Initialize work planning agent.

        Args:
            storage: WorkPlanStorage instance. If None, creates default storage.
        """
        self.storage = storage or WorkPlanStorage()

    def create_plan(
        self,
        goal: str,
        context: Optional[dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> WorkPlan:
        """
        Break down goal into structured tasks.

        Args:
            goal: High-level objective
            context: Additional context (files, constraints, etc.)
            llm_client: LLM client for task generation (optional for MVP)

        Returns:
            WorkPlan with generated tasks
        """
        # Create plan
        plan_name = (context or {}).get("name") or goal or "Work Plan"
        plan = WorkPlan(
            name=plan_name,
            goal=goal,
            context=context.get("description", "") if context else "",
        )

        if llm_client:
            # Use LLM to generate tasks (Phase 3 feature)
            tasks = self._generate_tasks_with_llm(goal, context, llm_client)
            plan.tasks = tasks
        else:
            # Create placeholder task for MVP
            plan.tasks = [
                Task(
                    title=f"Implement: {goal}",
                    description=goal,
                    priority=Priority.HIGH,
                )
            ]

        # Save plan
        self.storage.save_plan(plan)

        return plan

    def _generate_tasks_with_llm(
        self,
        goal: str,
        context: Optional[dict[str, Any]],
        llm_client: Any,
    ) -> list[Task]:
        """
        Use LLM to generate structured task breakdown.

        This is a Phase 3 feature. For now, returns empty list.

        Args:
            goal: High-level objective
            context: Additional context
            llm_client: LLM client

        Returns:
            List of Task objects
        """
        # TODO: Implement in Phase 3
        # Will use prompts to generate structured task breakdown
        return []

    def update_plan(
        self,
        plan_id: str,
        feedback: str,
        llm_client: Optional[Any] = None,
    ) -> WorkPlan:
        """
        Refine plan based on new information.

        Args:
            plan_id: ID of plan to update
            feedback: User feedback or new requirements
            llm_client: LLM client for regeneration (optional)

        Returns:
            Updated WorkPlan

        Raises:
            ValueError: If plan not found
        """
        plan = self.storage.load_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        # Increment version
        plan.version += 1
        plan.updated_at = datetime.utcnow()

        if llm_client:
            # Use LLM to refine tasks based on feedback (Phase 3)
            refined_tasks = self._refine_tasks_with_llm(plan, feedback, llm_client)
            plan.tasks = refined_tasks

        # Save updated plan
        self.storage.save_plan(plan)

        return plan

    def _refine_tasks_with_llm(
        self,
        plan: WorkPlan,
        feedback: str,
        llm_client: Any,
    ) -> list[Task]:
        """
        Use LLM to refine tasks based on feedback.

        This is a Phase 3 feature. For now, returns existing tasks.

        Args:
            plan: Current plan
            feedback: User feedback
            llm_client: LLM client

        Returns:
            Refined list of Task objects
        """
        # TODO: Implement in Phase 3
        return plan.tasks

    def get_next_task(self, plan_id: str) -> Optional[Task]:
        """
        Get next task respecting dependencies and priorities.

        Args:
            plan_id: ID of plan

        Returns:
            Next task to work on, or None if no tasks available
        """
        plan = self.storage.load_plan(plan_id)
        if not plan:
            return None

        return plan.get_next_task()

    def mark_task_started(self, plan_id: str, task_id: str) -> None:
        """
        Mark task as in progress.

        Args:
            plan_id: ID of plan containing task
            task_id: ID of task to start

        Raises:
            ValueError: If plan or task not found
        """
        plan = self.storage.load_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        task = plan.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()

        self.storage.save_plan(plan)

    def mark_task_complete(self, plan_id: str, task_id: str) -> None:
        """
        Update task status to completed.

        Args:
            plan_id: ID of plan containing task
            task_id: ID of task to complete

        Raises:
            ValueError: If plan or task not found
        """
        plan = self.storage.load_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        task = plan.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()

        self.storage.save_plan(plan)

    def get_plan_summary(self, plan_id: str) -> str:
        """
        Generate markdown summary (Aider style).

        Args:
            plan_id: ID of plan

        Returns:
            Markdown-formatted plan summary
        """
        plan = self.storage.load_plan(plan_id)
        if not plan:
            return "Plan not found"

        return self._format_plan_as_markdown(plan)

    def _format_plan_as_markdown(self, plan: WorkPlan) -> str:
        """
        Format plan as readable markdown.

        Args:
            plan: WorkPlan to format

        Returns:
            Markdown string
        """
        lines = []
        lines.append(f"# Work Plan: {plan.name or plan.goal}")
        lines.append("")
        lines.append(f"**Goal**: {plan.goal}")
        lines.append(f"**Progress**: {plan.get_completion_percentage():.1f}%")
        lines.append("")

        if plan.context:
            lines.append("## Context")
            lines.append(plan.context)
            lines.append("")

        lines.append("## Tasks")
        lines.append("")

        for i, task in enumerate(plan.tasks, 1):
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.CANCELLED: "❌",
            }.get(task.status, "❓")

            priority_marker = {
                Priority.CRITICAL: "🔴",
                Priority.HIGH: "🟠",
                Priority.MEDIUM: "🟡",
                Priority.LOW: "🟢",
            }.get(task.priority, "")

            lines.append(
                f"{i}. {status_icon} {priority_marker} **{task.title}** "
                f"(~{task.effort_estimate})"
            )

            if task.description:
                lines.append(f"   {task.description}")

            if task.dependencies:
                dep_refs = ", ".join(f"#{d[:8]}" for d in task.dependencies)
                lines.append(f"   *Depends on: {dep_refs}*")

            lines.append("")

        return "\n".join(lines)
