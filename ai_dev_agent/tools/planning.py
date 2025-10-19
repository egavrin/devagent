"""Planning tool for manager agent to create and execute work plans."""
from __future__ import annotations

from typing import Any, Mapping

from .registry import ToolSpec, ToolContext, registry
from ..agents.work_planner.agent import WorkPlanningAgent
from ..agents.base import AgentContext


def plan(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    """
    Create a structured work plan for a complex goal.

    Use this tool when a task is complex and would benefit from being broken down
    into smaller, ordered steps with dependencies.

    Args:
        payload: Dict with:
            - goal: The high-level goal to plan for
            - context: Optional additional context

    Returns:
        Dict with plan details including tasks, dependencies, and plan_id
    """
    goal = payload.get("goal", "")
    plan_context = payload.get("context", {})

    if not goal:
        return {
            "success": False,
            "error": "Missing required parameter: goal"
        }

    # Create work planning agent
    planner = WorkPlanningAgent()

    # Extract session_id from context.extra if available
    extra = context.extra or {}
    session_id = extra.get("session_id", "planning")

    # Create plan context dict (not AgentContext)
    plan_context_dict = payload.get("context", {})
    if not plan_context_dict:
        plan_context_dict = {"description": f"Planning for: {goal}"}

    # Create the plan (expects dict context, not AgentContext)
    try:
        plan = planner.create_plan(
            goal=goal,
            context=plan_context_dict
        )

        # Format tasks for output
        tasks = []
        for task in plan.tasks:
            tasks.append({
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "status": task.status.value,
                "priority": task.priority.value,
                "tags": task.tags,
                "dependencies": task.dependencies
            })

        return {
            "success": True,
            "plan_id": plan.id,
            "goal": plan.goal,
            "total_tasks": len(plan.tasks),
            "tasks": tasks,
            "message": f"Created plan with {len(plan.tasks)} tasks. Plan saved to disk."
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create plan: {str(e)}"
        }


# Register the planning tool
registry.register(
    ToolSpec(
        name="plan",
        handler=plan,
        request_schema_path=None,
        response_schema_path=None,
        description=(
            "Create a structured work plan for a complex goal.\n\n"
            "Use this when a task is complex and needs to be broken down into smaller steps.\n"
            "The planner will:\n"
            "- Analyze the goal and break it into logical tasks\n"
            "- Determine task dependencies (what must be done first)\n"
            "- Assign priorities (Critical, High, Medium, Low)\n"
            "- Add appropriate tags (design, test, implement, review, etc.)\n"
            "- Save the plan to disk\n\n"
            "After creating a plan, you can execute tasks by delegating to specialized agents.\n\n"
            "Usage:\n"
            "  plan(goal='Implement REST API for user management')\n"
            "  plan(goal='Add authentication system', context={'framework': 'FastAPI'})\n\n"
            "Good candidates for planning:\n"
            "- Multi-step features requiring design → test → implement → review\n"
            "- Complex refactoring with multiple dependent changes\n"
            "- New systems that need architecture planning first\n\n"
            "Not needed for:\n"
            "- Simple queries (reading code, searching files)\n"
            "- Single-step tasks (writing one function, fixing one bug)\n"
            "- Questions and explanations"
        ),
        category="workflow",
    )
)
