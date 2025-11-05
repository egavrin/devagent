"""Plan tool for creating structured work plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.tools.workflow.plan_tracker import start_plan_tracking

if TYPE_CHECKING:
    from ai_dev_agent.tools.registry import ToolContext

LOGGER = get_logger(__name__)


def plan(payload: Mapping[str, Any], context: "ToolContext") -> Mapping[str, Any]:
    """
    Create a structured work plan for a complex task.

    Args:
        payload: {"goal": "High-level goal",
                  "context": "Additional context (optional)",
                  "complexity": "simple|medium|complex (optional)"}
        context: Tool execution context

    Returns:
        {"success": bool, "plan": {...}, "error": str}
    """
    goal = payload.get("goal")
    plan_context = payload.get("context", "")
    complexity = payload.get("complexity", "medium")

    if not goal:
        return {
            "success": False,
            "error": "Missing required parameter: goal",
        }

    # For now, create a simple structured plan
    # In the future, this can integrate with PlanningIntegration from
    # ai_dev_agent/agents/integration/planning_integration.py

    try:
        # Analyze the goal and create tasks
        tasks = _generate_tasks_from_goal(goal, plan_context, complexity, context)

        plan_structure = {
            "goal": goal,
            "tasks": tasks,
            "complexity": complexity,
        }

        # Start plan tracking visualization
        start_plan_tracking(plan_structure)

        # Format result message
        result_lines = [
            f"Created {complexity} plan with {len(tasks)} tasks.",
            "Execute tasks sequentially using delegate tool.",
        ]

        return {
            "success": True,
            "plan": plan_structure,
            "result": "\n".join(result_lines),
        }

    except Exception as exc:
        LOGGER.exception("Plan creation failed")
        return {
            "success": False,
            "error": f"Plan creation failed: {exc!s}",
        }


def _generate_tasks_from_goal(
    goal: str, plan_context: str, complexity: str, tool_context: ToolContext
) -> list[dict[str, Any]]:
    """
    Generate tasks from a goal description using LLM analysis.

    The LLM analyzes the goal and creates appropriate tasks with:
    - Specific titles and descriptions
    - Correct agent assignments
    - Proper task dependencies
    - Appropriate number of tasks

    Args:
        goal: High-level goal
        plan_context: Additional context for planning
        complexity: Complexity level
        tool_context: Tool execution context with LLM client

    Returns:
        List of task dictionaries
    """
    import json

    # Get LLM client - prefer injected client from context
    client = None

    # First, try to get the already-active LLM client from tool context
    if tool_context.extra and "llm_client" in tool_context.extra:
        client = tool_context.extra["llm_client"]
        LOGGER.debug("Using injected LLM client from context")

    # Fallback: create a new client if none was provided
    if client is None:
        try:
            from ai_dev_agent.core.utils.config import load_settings
            from ai_dev_agent.providers.llm import create_client

            settings = load_settings()
            client = create_client(
                provider=settings.provider,
                api_key=settings.api_key,
                model=settings.model,
                base_url=settings.base_url if hasattr(settings, "base_url") else None,
            )
            LOGGER.debug("Created new LLM client from settings")
        except Exception as e:
            # Fallback to hardcoded if LLM unavailable
            LOGGER.debug(f"LLM client unavailable, using fallback: {e}")
            return _generate_tasks_fallback(goal, plan_context, complexity)

    # Determine target task count based on complexity
    task_counts = {"simple": 2, "medium": 4, "complex": 6}
    target_count = task_counts.get(complexity, 4)

    # Build prompt for LLM
    planning_prompt = f"""Analyze this software development goal and create a work plan with {target_count} tasks.

**Goal**: {goal}
{f"**Context**: {plan_context}" if plan_context else ""}
**Complexity**: {complexity}

Create {target_count} specific, actionable tasks to accomplish this goal. For each task:
1. Assign it to the most appropriate agent:
   - design_agent: Architecture, design documents, technical specs, API design
   - test_agent: Writing tests, TDD, test suites, test coverage
   - implementation_agent: Writing code, implementing features, bug fixes
   - review_agent: Code review, security audit, quality checks

2. Create a specific title (not generic like "Design and architecture")
3. Write a detailed description of what needs to be done
4. Specify dependencies (which tasks must complete first)

Return ONLY valid JSON in this exact format:
{{
  "tasks": [
    {{
      "title": "Specific task title here",
      "description": "Detailed description of what to do",
      "agent": "design_agent|test_agent|implementation_agent|review_agent",
      "dependencies": []
    }}
  ]
}}

Rules:
- Tasks should be specific to THIS goal, not generic
- Use appropriate agents (not all tasks need all 4 agents)
- First task usually has no dependencies: []
- Later tasks depend on earlier ones: ["task-1"] or ["task-1", "task-2"]
- Return exactly {target_count} tasks
- Output ONLY the JSON, no explanations"""

    try:
        # Call LLM to generate plan
        from ai_dev_agent.providers.llm.base import Message

        messages = [Message(role="user", content=planning_prompt)]

        response = client.complete(
            messages=messages,
            max_tokens=2000,
            temperature=0.3,  # Lower temperature for consistent planning
        )

        # Parse LLM response (complete() returns string directly)
        response_text = response.strip() if isinstance(response, str) else str(response).strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        plan_data = json.loads(response_text)

        # Convert to task format with IDs
        tasks = []
        for i, task_spec in enumerate(plan_data["tasks"], 1):
            task = {
                "id": f"task-{i}",
                "title": task_spec["title"],
                "description": task_spec["description"],
                "agent": task_spec["agent"],
                "dependencies": task_spec.get("dependencies", []),
            }
            tasks.append(task)

        # Validate we got the right number of tasks
        if len(tasks) != target_count:
            LOGGER.warning(
                f"LLM returned {len(tasks)} tasks, expected {target_count}. Using fallback."
            )
            return _generate_tasks_fallback(goal, plan_context, complexity)

        return tasks

    except Exception as e:
        LOGGER.warning(f"LLM plan generation failed: {e}. Using fallback.")
        return _generate_tasks_fallback(goal, plan_context, complexity)


def _generate_tasks_fallback(goal: str, plan_context: str, complexity: str) -> list[dict[str, Any]]:
    """Fallback task generation when LLM is unavailable.

    Creates a standard workflow scaled to match the requested complexity.
    """
    task_counts = {"simple": 2, "medium": 4, "complex": 6}
    num_tasks = task_counts.get(complexity, 4)

    goal_short = goal[:60].lower()

    # Base 4-phase workflow
    task_templates = [
        {
            "id": "task-1",
            "title": f"Design {goal_short}",
            "description": f"Design the architecture and approach for {goal}",
            "agent": "design_agent",
            "dependencies": [],
        },
        {
            "id": "task-2",
            "title": f"Write tests for {goal_short}",
            "description": f"Create comprehensive test suite for {goal}",
            "agent": "test_agent",
            "dependencies": ["task-1"],
        },
        {
            "id": "task-3",
            "title": f"Implement {goal_short}",
            "description": f"Implement the functionality for {goal}",
            "agent": "implementation_agent",
            "dependencies": ["task-2"],
        },
        {
            "id": "task-4",
            "title": f"Review {goal_short}",
            "description": f"Review the implementation of {goal}",
            "agent": "review_agent",
            "dependencies": ["task-3"],
        },
    ]

    # For complex plans (6 tasks), add integration and documentation phases
    if num_tasks > 4:
        task_templates.extend(
            [
                {
                    "id": "task-5",
                    "title": f"Integration testing for {goal_short}",
                    "description": f"Perform integration testing and validate the complete solution for {goal}",
                    "agent": "test_agent",
                    "dependencies": ["task-4"],
                },
                {
                    "id": "task-6",
                    "title": f"Documentation for {goal_short}",
                    "description": f"Create comprehensive documentation for {goal}",
                    "agent": "design_agent",
                    "dependencies": ["task-5"],
                },
            ]
        )

    return task_templates[:num_tasks]
