"""Unit tests for the modern WorkPlanningAgent."""

from __future__ import annotations

from ai_dev_agent.agents.work_planner.agent import WorkPlanningAgent
from ai_dev_agent.agents.work_planner.models import Priority, TaskStatus


def test_create_plan_without_llm(tmp_path):
    agent = WorkPlanningAgent()
    plan = agent.create_plan("Implement feature X", context={"description": "demo"})

    assert plan.goal == "Implement feature X"
    assert plan.tasks
    first_task = plan.tasks[0]
    assert first_task.priority == Priority.HIGH


def test_mark_task_lifecycle(tmp_path):
    agent = WorkPlanningAgent()
    plan = agent.create_plan("Lifecycle test")
    task = plan.tasks[0]
    agent.mark_task_started(plan.id, task.id)
    reloaded = agent.storage.load_plan(plan.id)
    started_task = reloaded.get_task(task.id)
    assert started_task.status == TaskStatus.IN_PROGRESS

    agent.mark_task_complete(plan.id, task.id)
    finished_plan = agent.storage.load_plan(plan.id)
    assert finished_plan.get_task(task.id).status == TaskStatus.COMPLETED
