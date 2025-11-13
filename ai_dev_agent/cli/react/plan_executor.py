"""
Simplified plan-based query execution.

Uses the new simplified planning system without rigid templates.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from ai_dev_agent.tools.registry import ToolContext
from ai_dev_agent.tools.workflow.plan import _needs_plan
from ai_dev_agent.tools.workflow.plan import plan as create_plan
from ai_dev_agent.tools.workflow.plan_tracker import update_task_status

if TYPE_CHECKING:
    from ai_dev_agent.core.utils.config import Settings


def _format_summary_text(summary: str | None) -> str:
    """Normalize final messages for log output."""
    if not summary:
        return ""
    text = " ".join(summary.strip().splitlines())
    return text[:197] + "..." if len(text) > 200 else text


def _check_done_condition(done_when: str, task_result: Any, all_results: list) -> bool:
    """Check if the done_when condition is met."""
    # Simple implementation: check if condition text appears in the result
    if not done_when:
        return False

    # Handle both dictionary and RunResult object
    if isinstance(task_result, dict):
        final_msg = task_result.get("final_message", "").lower()
        result_data = task_result.get("result", {})
    else:
        # Handle RunResult or other object types
        final_msg = str(getattr(task_result, "stop_reason", "")).lower()
        result_data = {}

    # Check for common conditions
    if "tests pass" in done_when.lower():
        # Check if tests passed in the result
        if "tests pass" in final_msg or result_data.get("tests_pass"):
            return True

    # Default: check if the done_when text appears in the final message
    return done_when.lower() in final_msg


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
    # First check if always_use_planning is enabled in settings
    always_plan = getattr(settings, "always_use_planning", False)

    if not always_plan and not _needs_plan(user_prompt):
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

    # Execute tasks sequentially with early stopping and error handling
    results = []
    tasks_completed = 0
    stopped_early = False
    error_occurred = False

    # Get done_when condition if present
    done_when = plan_data.get("done_when")

    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", f"task-{i}")
        click.echo(f"[Task {i}/{len(tasks)}] {task['title']}")
        click.echo("-" * 70)

        # Update status to in_progress
        update_task_status(task_id, "in_progress")

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
            suppress_final_output=False,  # Show full output for each task
            **task_kwargs,
        )
        results.append(task_result)

        # Check if task failed - task_result is a dictionary from _execute_react_assistant
        if isinstance(task_result, dict):
            # Extract the RunResult object from the dict
            run_result = task_result.get("result")
            # Check if the RunResult indicates failure
            task_failed = (
                run_result is not None
                and hasattr(run_result, "status")
                and run_result.status == "failure"
            )
            task_summary = task_result.get("final_message") or getattr(
                run_result, "stop_reason", ""
            )
        else:
            # In case it's a RunResult object directly
            run_result = task_result
            task_failed = hasattr(task_result, "status") and task_result.status == "failure"
            task_summary = getattr(task_result, "stop_reason", "")

        summary_text = _format_summary_text(task_summary)

        # Update status based on result
        if task_failed:
            update_task_status(task_id, "failed")
            error_occurred = True
        else:
            update_task_status(task_id, "completed")
            tasks_completed += 1

        # Note: Task result is already displayed by executor when suppress_final_output=False
        # No need to display it again here

        progress = (i / len(tasks)) * 100

        if task_failed:
            detail = f" Details: {summary_text}" if summary_text else ""
            click.echo(f"✗ Task failed. Progress: {progress:.0f}%{detail}\n")

            # Fail-fast: stop execution on error
            click.echo("⚠️ Stopping plan execution due to task failure\n")

            # Mark remaining tasks as skipped
            for j in range(i + 1, len(tasks) + 1):
                remaining_task_id = tasks[j - 1].get("id", f"task-{j}")
                update_task_status(remaining_task_id, "skipped")
            break
        else:
            detail = f" Summary: {summary_text}" if summary_text else ""
            click.echo(f"✓ Task completed. Progress: {progress:.0f}%{detail}\n")

            # Check early stopping condition
            if done_when and _check_done_condition(done_when, task_result, results):
                click.echo(f"✅ Early stopping: {done_when} condition met\n")
                stopped_early = True

                # Mark remaining tasks as skipped
                for j in range(i + 1, len(tasks) + 1):
                    remaining_task_id = tasks[j - 1].get("id", f"task-{j}")
                    update_task_status(remaining_task_id, "skipped")
                break

    # Summarize results
    click.echo("=" * 70)

    if error_occurred:
        click.echo("❌ Plan execution failed due to task error\n")
    elif stopped_early:
        click.echo(f"✅ Plan completed early - condition met ({done_when})\n")
    elif tasks_completed == len(tasks):
        click.echo("✅ All tasks completed!\n")
    else:
        click.echo(f"⚠️ Plan partially completed: {tasks_completed}/{len(tasks)} tasks\n")

    # Return combined results
    return {
        "final_message": f"Completed {tasks_completed}/{len(tasks)} tasks for: {user_prompt}",
        "result": {
            "tasks_completed": tasks_completed,
            "tasks_total": len(tasks),
            "stopped_early": stopped_early,
            "error_occurred": error_occurred,
            "results": results,
        },
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
