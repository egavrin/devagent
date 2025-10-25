"""Integration between agent system and work planning system."""
from __future__ import annotations

import re
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from ..work_planner.models import WorkPlan, Task, TaskStatus, Priority
from ..specialized import DesignAgent, TestingAgent, ImplementationAgent, ReviewAgent, OrchestratorAgent
from ..base import AgentContext, AgentResult


class TaskAgentMapper:
    """Maps tasks to appropriate agents based on task properties."""

    def __init__(self):
        """Initialize task-agent mapper."""
        self._keyword_mappings = {
            "design": ["design", "architect", "plan", "specification"],
            "review": ["review", "analyze", "check", "audit", "inspect"],
            "test": ["test", "spec", "coverage", "unittest"],
            "implement": ["implement", "code", "write", "develop", "build"],
        }

    def map_task_to_agent(self, task: Task) -> str:
        """
        Map a task to an agent name.

        Args:
            task: Task to map

        Returns:
            Agent name (design, test, implement, review)
        """
        # First check tags
        for tag in task.tags:
            tag_lower = tag.lower()
            if tag_lower in self._keyword_mappings:
                return tag_lower

        # Check title and description for keywords
        text = (task.title + " " + task.description).lower()

        for agent_name, keywords in self._keyword_mappings.items():
            if any(keyword in text for keyword in keywords):
                return agent_name

        # Default to implementation
        return "implement"


class PlanningIntegration:
    """Integrates work planning with multi-agent system."""

    def __init__(self):
        """Initialize planning integration."""
        self.orchestrator = OrchestratorAgent()
        self.mapper = TaskAgentMapper()
        self.current_plan: Optional[WorkPlan] = None

        # Register all specialized agents
        self.orchestrator.register_subagent("design", DesignAgent())
        self.orchestrator.register_subagent("test", TestingAgent())
        self.orchestrator.register_subagent("implement", ImplementationAgent())
        self.orchestrator.register_subagent("review", ReviewAgent())

    def load_plan(self, plan: WorkPlan) -> None:
        """
        Load a work plan for execution.

        Args:
            plan: WorkPlan to load
        """
        self.current_plan = plan

    def convert_plan_to_workflow(self, plan: WorkPlan) -> Dict[str, Any]:
        """
        Convert a work plan to agent workflow.

        Args:
            plan: WorkPlan to convert

        Returns:
            Workflow specification for orchestrator
        """
        steps = []

        for task in plan.tasks:
            agent_name = self.mapper.map_task_to_agent(task)

            step = {
                "id": task.id,
                "agent": agent_name,
                "task": f"{task.title}: {task.description}",
                "depends_on": task.dependencies.copy()
            }

            steps.append(step)

        return {
            "goal": plan.goal,
            "steps": steps
        }

    def execute_plan(
        self,
        plan: WorkPlan,
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Execute a work plan using agents.

        Args:
            plan: WorkPlan to execute
            context: Execution context

        Returns:
            Execution results
        """
        self.load_plan(plan)

        # Convert to workflow
        workflow = self.convert_plan_to_workflow(plan)

        # Execute with orchestrator
        result = self.orchestrator.coordinate_workflow(workflow, context)

        return result

    def update_task_from_result(
        self,
        task_id: str,
        agent_result: AgentResult
    ) -> None:
        """
        Update task status based on agent result.

        Args:
            task_id: Task ID
            agent_result: Result from agent execution
        """
        if not self.current_plan:
            return

        task = self.current_plan.get_task(task_id)
        if not task:
            return

        if agent_result.success:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
        else:
            # Mark as blocked if failed
            task.status = TaskStatus.BLOCKED
            task.notes.append(f"Failed: {agent_result.error}")

        task.updated_at = datetime.utcnow()

    def get_next_task(self) -> Optional[Task]:
        """
        Get next task to execute.

        Returns:
            Next task or None
        """
        if not self.current_plan:
            return None

        return self.current_plan.get_next_task()

    def get_progress(self) -> Dict[str, Any]:
        """
        Get execution progress.

        Returns:
            Progress statistics
        """
        if not self.current_plan:
            return {
                "total_tasks": 0,
                "completed": 0,
                "in_progress": 0,
                "pending": 0,
                "completion_percentage": 0.0
            }

        total = len(self.current_plan.tasks)
        completed = sum(1 for t in self.current_plan.tasks if t.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for t in self.current_plan.tasks if t.status == TaskStatus.IN_PROGRESS)
        pending = sum(1 for t in self.current_plan.tasks if t.status == TaskStatus.PENDING)

        return {
            "total_tasks": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "completion_percentage": (completed / total * 100) if total > 0 else 0.0
        }


class AutomatedWorkflow:
    """Automates execution of work plans with agents."""

    def __init__(self):
        """Initialize automated workflow."""
        self.integration = PlanningIntegration()

    def execute_plan_automatically(
        self,
        plan: WorkPlan,
        context: AgentContext,
        stop_on_failure: bool = True,
        progress_callback: Optional[Callable[[str, str, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute entire plan automatically.

        Args:
            plan: WorkPlan to execute
            context: Execution context
            stop_on_failure: Stop if any task fails
            progress_callback: Callback for progress updates

        Returns:
            Execution results
        """
        self.integration.load_plan(plan)

        tasks_completed = 0
        tasks_failed = 0

        while True:
            # Get next task
            next_task = self.integration.get_next_task()

            if not next_task:
                # No more tasks available
                break

            # Mark as in progress
            next_task.status = TaskStatus.IN_PROGRESS
            next_task.started_at = datetime.utcnow()

            if progress_callback:
                progress_callback(next_task.id, "started", next_task.title)

            # Map to agent
            agent_name = self.integration.mapper.map_task_to_agent(next_task)

            # Execute task
            task_description = f"{next_task.title}: {next_task.description}"
            agent_result = self.integration.orchestrator.delegate_task(
                agent_name,
                task_description,
                context
            )

            # Update task status
            self.integration.update_task_from_result(next_task.id, agent_result)

            if agent_result.success:
                tasks_completed += 1
                if progress_callback:
                    progress_callback(next_task.id, "completed", next_task.title)
            else:
                tasks_failed += 1
                if progress_callback:
                    progress_callback(next_task.id, "failed", agent_result.error or "Unknown error")

                if stop_on_failure:
                    break

        # Calculate results
        total_tasks = len(plan.tasks)
        all_completed = tasks_completed == total_tasks

        return {
            "success": all_completed and tasks_failed == 0,
            "total_tasks": total_tasks,
            "tasks_completed": tasks_completed,
            "tasks_failed": tasks_failed,
            "completion_percentage": (tasks_completed / total_tasks * 100) if total_tasks > 0 else 0.0
        }
