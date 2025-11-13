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
    Create a structured work plan for a task when needed.

    Simple tasks should be executed directly without planning.
    Only multi-step or complex tasks need explicit plans.

    Args:
        payload: {"goal": "High-level goal",
                  "context": "Additional context (optional)"}
        context: Tool execution context

    Returns:
        {"success": bool, "plan": {...}, "error": str}
    """
    goal = payload.get("goal")
    plan_context = payload.get("context", "")

    if not goal:
        return {
            "success": False,
            "error": "Missing required parameter: goal",
        }

    # Check if this is a delegated execution to prevent nested planning
    if context.extra and context.extra.get("is_delegated"):
        LOGGER.warning(
            "Attempted nested planning in delegated context - blocking to prevent recursion"
        )
        return {
            "success": False,
            "error": "Nested planning is not allowed. This task is already part of an executing plan. Execute the task directly without creating a sub-plan.",
        }

    # Check if planning is enabled
    from ai_dev_agent.core.utils.config import load_settings

    settings = load_settings()

    if not settings.planning_enabled:
        LOGGER.info("Planning is disabled in configuration")
        return {
            "success": True,
            "plan": {"goal": goal, "tasks": [], "disabled": True},
            "result": "Planning is disabled - execute task directly.",
        }

    try:
        # Check if task needs planning
        if not _needs_plan(goal, plan_context):
            LOGGER.info(f"Simple task detected, skipping plan generation: {goal[:100]}")
            return {
                "success": True,
                "plan": {"goal": goal, "tasks": [], "simple": True},
                "result": "Simple task - execute directly without plan.",
            }

        # Generate contextual plan using LLM
        tasks = _generate_tasks_from_goal(goal, plan_context, context)

        plan_structure = {"goal": goal, "tasks": tasks, "simple": False}

        # Start plan tracking visualization
        start_plan_tracking(plan_structure)

        # Format result message
        result_lines = [
            f"Created plan with {len(tasks)} task(s).",
            "Tasks will be executed based on dependencies.",
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


def _needs_plan(goal: str, context: str = "") -> bool:
    """
    Determine if a task needs explicit planning.

    Simple heuristics to avoid over-planning trivial tasks.
    """
    goal_lower = goal.lower()
    combined = (goal + " " + context).lower()

    # Indicators of simple tasks that don't need plans
    simple_indicators = [
        "fix typo",
        "rename",
        "update comment",
        "change variable",
        "fix syntax",
        "correct spelling",
        "update version",
        "add import",
        "remove unused",
        "format code",
        "add docstring",
        "update readme",
    ]

    # Check for simple task indicators
    for indicator in simple_indicators:
        if indicator in goal_lower:
            return False

    # Indicators of complex tasks that need plans
    complex_indicators = [
        "implement",
        "build",
        "create",
        "refactor",
        "design",
        "integrate",
        "migrate",
        "optimize",
        "architecture",
        "multiple",
        "system",
        "feature",
        "workflow",
        "pipeline",
        " and ",  # Multiple things
        ", ",  # List of items
    ]

    # Check for complex task indicators
    complex_count = sum(1 for indicator in complex_indicators if indicator in combined)

    # If multiple complexity indicators, definitely needs plan
    if complex_count >= 2:
        return True

    # If goal is very short (< 50 chars), probably simple
    if len(goal) < 50 and complex_count == 0:
        return False

    # If goal is long (> 200 chars) or has complexity indicator, needs plan
    if len(goal) > 200 or complex_count > 0:
        return True

    # Default to planning for safety
    return True


def _generate_tasks_from_goal(
    goal: str, plan_context: str, tool_context: ToolContext
) -> list[dict[str, Any]]:
    """
    Generate tasks from a goal using LLM.

    Fail-fast approach: if LLM is unavailable or fails, raise an error.
    No fallbacks, no templates.

    Args:
        goal: High-level goal
        plan_context: Additional context for planning
        tool_context: Tool execution context with LLM client

    Returns:
        List of task dictionaries

    Raises:
        RuntimeError: If plan generation fails
    """
    import json

    from ai_dev_agent.providers.llm.base import Message

    # Get LLM client - prefer injected client from context
    client = None

    # First, try to get the already-active LLM client from tool context
    if tool_context.extra and "llm_client" in tool_context.extra:
        client = tool_context.extra["llm_client"]
        LOGGER.debug("Using injected LLM client from context")

    # Create a new client if none was provided
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
            # Fail fast - no fallback
            raise RuntimeError(f"Cannot create LLM client for plan generation: {e}")

    # Simple, clear prompt - no complexity tiers, no rigid structure
    planning_prompt = f"""Break down this task into logical steps if needed.

**Task**: {goal}
{f"**Context**: {plan_context}" if plan_context else ""}

Guidelines:
- Use the RIGHT number of steps - as few or as many as actually needed
- Simple tasks might need just 1-2 steps
- Complex tasks might need several steps
- Don't add unnecessary steps just to pad the plan
- Don't force Design→Test→Implement→Review unless the task actually requires it

Return a JSON array of tasks. Each task should have:
- title: Clear, actionable title
- description: What needs to be done
- dependencies: Array of task IDs this depends on (or empty array)

Example format:
{{
  "tasks": [
    {{
      "title": "Update authentication logic",
      "description": "Modify the login function to support OAuth",
      "dependencies": []
    }},
    {{
      "title": "Add OAuth configuration",
      "description": "Set up OAuth provider settings",
      "dependencies": ["task-1"]
    }}
  ]
}}

Return ONLY the JSON, no explanations."""

    # Single attempt with optional retry
    max_attempts = 2  # Simplified: one try, one retry
    last_error = None

    for attempt in range(max_attempts):
        try:
            messages = [Message(role="user", content=planning_prompt)]

            response = client.complete(
                messages=messages,
                max_tokens=2000,
                temperature=0.3,
            )

            # Parse LLM response
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
                    "dependencies": task_spec.get("dependencies", []),
                }
                tasks.append(task)

            # Minimal validation - just check we got something
            if not tasks:
                raise ValueError("LLM returned empty task list")

            # Safety limit only - no strict counting
            from ai_dev_agent.core.utils.config import load_settings

            settings = load_settings()
            if len(tasks) > settings.plan_max_tasks:
                LOGGER.warning(
                    f"LLM returned {len(tasks)} tasks, exceeding limit of {settings.plan_max_tasks}. Truncating."
                )
                tasks = tasks[: settings.plan_max_tasks]

            LOGGER.info(f"Generated plan with {len(tasks)} task(s)")
            return tasks

        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON response: {e}"
            LOGGER.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {last_error}")
        except Exception as e:
            last_error = str(e)
            LOGGER.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {last_error}")

        if attempt < max_attempts - 1:
            import time

            time.sleep(1)  # Brief pause before retry

    # All attempts failed - fail fast, no fallback
    raise RuntimeError(
        f"Failed to generate plan after {max_attempts} attempts. Last error: {last_error}"
    )
