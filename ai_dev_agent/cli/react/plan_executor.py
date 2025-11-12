"""
Simplified plan-based query execution.

Uses the new simplified planning system without rigid templates.
"""

from pathlib import Path
from typing import Any

import click

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.tools.registry import ToolContext
from ai_dev_agent.tools.workflow.plan import _needs_plan
from ai_dev_agent.tools.workflow.plan import plan as create_plan


def execute_with_planning(
    ctx: click.Context, client: Any, settings: Settings, user_prompt: str, **kwargs
) -> dict[str, Any]:
    """
    Execute a query with simplified planning.

    Args:
        ctx: Click context
        client: LLM client
        settings: Settings object
        user_prompt: User's query
        **kwargs: Additional arguments passed to executor

    Returns:
        Dict with execution results
    """
    # Check if planning is needed
    if not _needs_plan(user_prompt):
        click.echo(f"⚡ Simple task detected - executing directly: {user_prompt[:50]}...")
        from .executor import _execute_react_assistant

        return _execute_react_assistant(
            ctx, client, settings, user_prompt, use_planning=False, **kwargs
        )

    click.echo("🗺️ Planning mode enabled - creating task breakdown...")

    # Create tool context with required parameters
    repo_root = Path.cwd()

    # Create a minimal ToolContext with required arguments
    tool_context = ToolContext(
        repo_root=repo_root,
        settings=settings,
        sandbox=None,  # Not needed for planning
        extra={"llm_client": client},
    )

    # Use simplified plan tool
    plan_result = create_plan({"goal": user_prompt, "context": ""}, tool_context)

    if not plan_result["success"]:
        click.echo(f"⚠️ Planning failed: {plan_result.get('error', 'Unknown error')}")
        click.echo("Falling back to direct execution...")
        from .executor import _execute_react_assistant

        return _execute_react_assistant(
            ctx, client, settings, user_prompt, use_planning=False, **kwargs
        )

    plan_data = plan_result["plan"]

    # If it's a simple task (no tasks generated), execute directly
    if plan_data.get("simple") or not plan_data.get("tasks"):
        click.echo("✓ Task is simple enough for direct execution")
        from .executor import _execute_react_assistant

        return _execute_react_assistant(
            ctx, client, settings, user_prompt, use_planning=False, **kwargs
        )

    tasks = plan_data["tasks"]
    click.echo(f"\n✓ Created dynamic plan with {len(tasks)} task(s)\n")

    # Display the plan
    click.echo("📋 Task Breakdown:")
    for i, task in enumerate(tasks, 1):
        click.echo(f"  {i}. {task['title']}")
        if task.get("description") and task["description"] != task["title"]:
            click.echo(f"     {task['description']}")

    click.echo("\n" + "=" * 70)
    click.echo("🚀 Executing tasks...\n")

    # Execute tasks sequentially
    results = []
    for i, task in enumerate(tasks, 1):
        click.echo(f"[Task {i}/{len(tasks)}] {task['title']}")
        click.echo("-" * 70)

        # Execute as a query with the task description
        task_prompt = f"{task['title']}: {task['description']}"

        from .executor import _execute_react_assistant

        # Remove conflicting parameters from kwargs
        task_kwargs = kwargs.copy()
        task_kwargs.pop("suppress_final_output", None)

        task_result = _execute_react_assistant(
            ctx,
            client,
            settings,
            task_prompt,
            use_planning=False,  # Don't plan individual tasks
            suppress_final_output=True,  # Don't show final output for each task
            **task_kwargs,
        )
        results.append(task_result)

        # Show task result if available
        final_msg = task_result.get("final_message", "")
        if final_msg and final_msg.strip():
            click.echo(
                f"\n📝 Result: {final_msg.strip()[:200]}{'...' if len(final_msg) > 200 else ''}\n"
            )

        progress = (i / len(tasks)) * 100
        click.echo(f"✓ Task completed. Progress: {progress:.0f}%\n")

    # Summarize results
    click.echo("=" * 70)
    click.echo("✅ All tasks completed!\n")

    # Return combined results
    return {
        "final_message": f"Completed {len(tasks)} tasks for: {user_prompt}",
        "result": {"tasks_completed": len(tasks), "results": results},
        "printed_final": True,
    }


def _assess_query_complexity(user_prompt: str) -> str:
    """
    Simple heuristic to determine if we should use planning.

    Returns:
        "simple" for direct execution
        "complex" for planned execution
    """
    if _needs_plan(user_prompt):
        return "complex"
    return "simple"
