"""
Plan-based query execution.

When --plan flag is used, this module creates a structured work plan,
breaks down the query into tasks, and executes them step by step.
"""

from typing import Any, Dict, Optional, List
import click
import json
from pathlib import Path

from ai_dev_agent.agents.work_planner import (
    WorkPlanningAgent,
    Task,
    WorkPlan,
    Priority,
    TaskStatus,
)
from ai_dev_agent.core.utils.config import Settings


def execute_with_planning(
    ctx: click.Context,
    client: Any,
    settings: Settings,
    user_prompt: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a query with LLM-driven approach selection.

    Args:
        ctx: Click context
        client: LLM client
        settings: Settings object
        user_prompt: User's query
        **kwargs: Additional arguments passed to executor

    Returns:
        Dict with execution results
    """
    click.echo(f"🔍 Analyzing query complexity...")

    # Use LLM to assess query complexity
    assessment = _assess_query_with_llm(client, user_prompt)

    approach = assessment.get('approach', 'simple_plan')
    reasoning = assessment.get('reasoning', '')
    estimated_tasks = assessment.get('estimated_tasks', 2)

    click.echo(f"📊 Assessment: {approach.upper().replace('_', ' ')}")
    click.echo(f"💡 Reasoning: {reasoning}")
    click.echo()

    # Route based on assessment
    if approach == 'direct':
        click.echo("⚡ Using direct execution (no planning overhead)\n")
        from .executor import _execute_react_assistant
        return _execute_react_assistant(
            ctx, client, settings, user_prompt,
            use_planning=False,
            **kwargs
        )

    # For simple_plan or complex_plan, create and execute plan
    click.echo(f"🗺️  Creating work plan for: {user_prompt}")
    click.echo("="*70)

    # Initialize Work Planning Agent
    agent = WorkPlanningAgent()

    # Use LLM to break down the query into tasks (pass assessment for context)
    plan = _create_plan_from_query(agent, client, user_prompt, assessment)

    if not plan or not plan.tasks:
        click.echo("⚠️  Could not create a plan. Falling back to direct execution.")
        # Fall back to direct execution
        from .executor import _execute_react_assistant
        return _execute_react_assistant(
            ctx, client, settings, user_prompt,
            use_planning=False,
            **kwargs
        )

    click.echo(f"\n✓ Created plan with {len(plan.tasks)} task(s)\n")

    # Display the plan
    click.echo("📋 Work Plan:")
    for i, task in enumerate(plan.tasks, 1):
        priority_icon = {
            Priority.CRITICAL: "🔴",
            Priority.HIGH: "🟠",
            Priority.MEDIUM: "🟡",
            Priority.LOW: "🟢",
        }.get(task.priority, "")
        click.echo(f"  {i}. {priority_icon} {task.title}")
        if task.description and task.description != task.title:
            click.echo(f"     {task.description}")

    click.echo("\n" + "="*70)
    click.echo("🚀 Executing tasks...\n")

    # Execute each task
    results = []
    early_termination = False
    for i, task in enumerate(plan.tasks, 1):
        click.echo(f"[Task {i}/{len(plan.tasks)}] {task.title}")
        click.echo("-"*70)

        # Mark task as started
        agent.mark_task_started(plan.id, task.id)

        # Execute the task
        task_result = _execute_task(
            ctx, client, settings, task, user_prompt, **kwargs
        )
        results.append(task_result)

        # Mark task as completed
        agent.mark_task_complete(plan.id, task.id)

        # Show task result if available
        result_data = task_result.get("result", {})
        final_msg = result_data.get("final_message", "")
        if final_msg and final_msg.strip():
            click.echo(f"\n📝 Result: {final_msg.strip()}\n")

        # Show progress
        updated_plan = agent.storage.load_plan(plan.id)
        progress = updated_plan.get_completion_percentage()
        click.echo(f"✓ Task completed. Overall progress: {progress:.0f}%\n")

        # Check if we can stop early (only if there are remaining tasks)
        remaining_tasks = plan.tasks[i:]
        if remaining_tasks:
            click.echo("🤔 Checking if query is fully answered...")

            satisfaction = _check_if_query_satisfied(
                client, user_prompt, results, remaining_tasks
            )

            is_satisfied = satisfaction.get('is_satisfied', False)
            confidence = satisfaction.get('confidence', 0.0)
            reasoning = satisfaction.get('reasoning', '')

            if is_satisfied and confidence > 0.7:
                click.echo(f"✅ Query fully answered! (confidence: {confidence:.0%})")
                click.echo(f"💡 {reasoning}\n")
                click.echo(f"⏭️  Skipping {len(remaining_tasks)} remaining task(s)\n")

                # Mark remaining tasks as cancelled
                for remaining_task in remaining_tasks:
                    # Update task status in storage
                    updated_plan = agent.storage.load_plan(plan.id)
                    for t in updated_plan.tasks:
                        if t.id == remaining_task.id:
                            t.status = TaskStatus.CANCELLED
                    agent.storage.save_plan(updated_plan)

                early_termination = True
                break  # Exit the loop early
            else:
                click.echo(f"⏩ Continuing with remaining tasks (confidence: {confidence:.0%})\n")

    click.echo("="*70)
    # Show accurate completion message
    tasks_completed = len(results)
    if early_termination:
        tasks_cancelled = len(plan.tasks) - tasks_completed
        click.echo(f"✅ Completed {tasks_completed} of {len(plan.tasks)} tasks ({tasks_cancelled} cancelled due to early termination)")
    else:
        click.echo(f"✅ All {len(plan.tasks)} tasks completed!")

    # Show final answer
    final_answer = _synthesize_final_message(results)
    if final_answer and final_answer.strip():
        click.echo("\n" + "="*70)
        click.echo("📋 FINAL ANSWER:")
        click.echo("="*70)
        click.echo(final_answer)
        click.echo("="*70)

    # Store plan ID in context for reference
    if not isinstance(ctx.obj, dict):
        ctx.obj = {}
    ctx.obj["_last_plan_id"] = plan.id

    # Return aggregated results with accurate count
    return {
        "plan_id": plan.id,
        "tasks_completed": len(results),  # Actual tasks executed, not total planned
        "tasks_total": len(plan.tasks),
        "early_termination": early_termination,
        "task_results": results,
        "final_message": _synthesize_final_message(results),
    }


def _create_plan_from_query(
    agent: WorkPlanningAgent,
    client: Any,
    query: str,
    assessment: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Use LLM to break down the query into structured tasks.

    Args:
        agent: Work Planning Agent
        client: LLM client
        query: User's query
        assessment: Optional complexity assessment with task suggestions

    Returns:
        WorkPlan object or None
    """
    estimated_tasks = assessment.get('estimated_tasks', 3) if assessment else 3
    task_suggestions = assessment.get('task_suggestions', []) if assessment else []

    # Build prompt for LLM to generate task breakdown
    breakdown_prompt = f"""Break down this query into a structured work plan with specific, actionable tasks:

Query: "{query}"

Constraints:
- Target {estimated_tasks} task(s) (use fewer if possible, more only if truly necessary)
- Each task must add UNIQUE value - no redundant verification steps
- Combine related operations (e.g., "find and count" not "find" then "count")
- Focus on actionable steps that produce concrete results

{f"Suggested approach: {', '.join(task_suggestions)}" if task_suggestions else ""}

For each task, provide:
1. Clear, action-oriented title (verb + object)
2. Brief description of what needs to be done
3. Priority (critical/high/medium/low)

Think step-by-step about the MINIMUM work needed to answer this query.

Format your response as a numbered list of tasks."""

    try:
        # Get task breakdown from LLM
        from ai_dev_agent.providers.llm.base import Message
        messages = [Message(role="user", content=breakdown_prompt)]

        response_text = client.complete(messages=messages, temperature=0.3)
        task_descriptions = _parse_task_breakdown(response_text)

        if not task_descriptions:
            # Fallback: create a single task
            task_descriptions = [(query, "medium", "Main task")]

        # Create the plan
        plan = agent.create_plan(
            goal=query,
            context={"description": f"Automatically generated plan for: {query}"}
        )

        # Clear placeholder task and add parsed tasks
        plan.tasks = []

        for i, (title, priority_str, description) in enumerate(task_descriptions):
            # Map priority string to Priority enum
            priority_map = {
                "critical": Priority.CRITICAL,
                "high": Priority.HIGH,
                "medium": Priority.MEDIUM,
                "low": Priority.LOW,
            }
            priority = priority_map.get(priority_str.lower(), Priority.MEDIUM)

            # Add dependency on previous task for sequential execution
            dependencies = [plan.tasks[i-1].id] if i > 0 and plan.tasks else []

            task = Task(
                title=title,
                description=description,
                priority=priority,
                effort_estimate="5min",  # Default estimate
                dependencies=dependencies,
            )
            plan.tasks.append(task)

        # Save the plan
        agent.storage.save_plan(plan)

        return plan

    except Exception as e:
        click.echo(f"⚠️  Error creating plan: {e}", err=True)
        return None


def _parse_task_breakdown(response_text: str) -> list:
    """
    Parse LLM response into list of (title, priority, description) tuples.

    Args:
        response_text: LLM response with task breakdown

    Returns:
        List of (title, priority, description) tuples
    """
    tasks = []
    lines = response_text.strip().split('\n')

    current_task = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered tasks (1., 2., etc.)
        if line[0].isdigit() and '.' in line[:3]:
            # Extract task title
            title = line.split('.', 1)[1].strip()
            # Remove markdown bold/emphasis
            title = title.replace('**', '').replace('*', '')
            # Extract priority if mentioned
            priority = "medium"
            if "critical" in title.lower():
                priority = "critical"
            elif "high" in title.lower() or "important" in title.lower():
                priority = "high"
            elif "low" in title.lower():
                priority = "low"

            # Clean up title
            title = title.split('(')[0].strip()  # Remove parenthetical notes
            title = title.split('-')[0].strip()  # Remove dashes

            current_task = [title, priority, title]
            tasks.append(current_task)
        elif current_task and line:
            # Additional description for current task
            current_task[2] = line

    # Fallback: if no tasks parsed, create simple task breakdown
    if not tasks:
        # Try to extract any actionable phrases
        simple_tasks = [
            ("Analyze the query", "high", "Understand what needs to be done"),
            ("Execute the task", "high", "Perform the requested operation"),
            ("Verify the result", "medium", "Confirm the output is correct"),
        ]
        return simple_tasks

    return [(t[0], t[1], t[2]) for t in tasks]


def _execute_task(
    ctx: click.Context,
    client: Any,
    settings: Settings,
    task: Task,
    original_query: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a single task from the work plan.

    Args:
        ctx: Click context
        client: LLM client
        settings: Settings object
        task: Task to execute
        original_query: Original user query for context
        **kwargs: Additional arguments

    Returns:
        Dict with task execution results
    """
    # Build task-specific prompt
    task_prompt = f"""[Task: {task.title}]

Original query: {original_query}

Task description: {task.description}

Execute this specific task. Be focused and concise."""

    # Execute using the standard React executor
    from .executor import _execute_react_assistant

    # Remove parameters we're explicitly setting to avoid conflicts
    task_kwargs = {k: v for k, v in kwargs.items()
                   if k not in ['use_planning', 'suppress_final_output']}

    result = _execute_react_assistant(
        ctx,
        client,
        settings,
        task_prompt,
        use_planning=False,  # Don't nest planning
        suppress_final_output=False,  # Show task output
        **task_kwargs
    )

    return {
        "task_id": task.id,
        "task_title": task.title,
        "result": result,
    }


def _synthesize_final_message(task_results: list) -> str:
    """
    Synthesize a final message from all task results.

    Args:
        task_results: List of task execution results

    Returns:
        Combined message string
    """
    messages = []
    for i, task_result in enumerate(task_results, 1):
        result_data = task_result.get("result", {})
        final_msg = result_data.get("final_message", "")

        if final_msg and final_msg.strip():
            # Only include substantive results (skip empty or error messages)
            if not final_msg.startswith("ERROR:") and len(final_msg.strip()) > 10:
                messages.append(final_msg.strip())

    if messages:
        # If we have results, combine them intelligently
        if len(messages) == 1:
            return messages[0]
        else:
            # Combine multiple results with the last one typically being the most complete
            return messages[-1]  # Return the final task's result as the answer
    else:
        return "All tasks completed successfully. No detailed results were generated."


def _assess_query_with_llm(client: Any, query: str) -> Dict[str, Any]:
    """
    Use LLM to assess if query needs planning and how complex it should be.

    Args:
        client: LLM client
        query: User's query to assess

    Returns:
        {
            'approach': 'direct' | 'simple_plan' | 'complex_plan',
            'reasoning': str,
            'estimated_tasks': int,
            'can_answer_immediately': bool,
            'task_suggestions': List[str]
        }
    """

    assessment_prompt = f"""Analyze this query and determine the best execution approach:

Query: "{query}"

Consider:
1. Can this be answered in ONE direct action? (e.g., counting lines, simple calculation, finding a file)
2. Does it require 2-3 sequential steps? (e.g., find then analyze, fetch then process)
3. Does it require complex multi-step planning? (e.g., implement feature with tests, refactor with validation)

Respond in JSON format:
{{
    "approach": "direct" | "simple_plan" | "complex_plan",
    "reasoning": "Brief explanation of why",
    "estimated_tasks": <number>,
    "can_answer_immediately": true | false,
    "task_suggestions": ["optional list of task titles if planning needed"]
}}

Examples:

Query: "how many lines in commands.py"
Response: {{"approach": "direct", "reasoning": "Single file operation, can count immediately", "estimated_tasks": 1, "can_answer_immediately": true}}

Query: "find all TODO comments and count them"
Response: {{"approach": "simple_plan", "reasoning": "Need to search codebase then aggregate results", "estimated_tasks": 1, "can_answer_immediately": false, "task_suggestions": ["Search for and count all TODO comments"]}}

Query: "implement user authentication with JWT, add tests, update docs"
Response: {{"approach": "complex_plan", "reasoning": "Multiple independent components requiring coordination", "estimated_tasks": 4, "can_answer_immediately": false, "task_suggestions": ["Implement JWT authentication logic", "Add authentication middleware", "Write comprehensive tests", "Update documentation"]}}

Now analyze the given query. Prefer simpler approaches when possible - only use complex_plan for truly multi-faceted work."""

    try:
        from ai_dev_agent.providers.llm.base import Message
        messages = [Message(role="user", content=assessment_prompt)]

        response_text = client.complete(
            messages=messages,
            temperature=0.1,  # Low temperature for consistent decisions
        )

        # Try to parse JSON from response
        # Handle potential JSON wrapped in markdown code blocks
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        assessment = json.loads(response_text)

        # Validate required fields
        if 'approach' not in assessment:
            assessment['approach'] = 'simple_plan'
        if 'estimated_tasks' not in assessment:
            assessment['estimated_tasks'] = 2
        if 'reasoning' not in assessment:
            assessment['reasoning'] = 'Default assessment'

        return assessment

    except Exception as e:
        # Fallback to safe default (use simple planning)
        click.echo(f"⚠️  Assessment error: {e}")
        return {
            'approach': 'simple_plan',
            'reasoning': 'Fallback due to assessment error',
            'estimated_tasks': 2,
            'can_answer_immediately': False,
            'task_suggestions': []
        }


def _check_if_query_satisfied(
    client: Any,
    original_query: str,
    completed_tasks: List[Dict],
    remaining_tasks: List[Task]
) -> Dict[str, Any]:
    """
    Use LLM to determine if the original query has been fully answered.

    Args:
        client: LLM client
        original_query: The original user query
        completed_tasks: List of completed task results
        remaining_tasks: List of tasks not yet executed

    Returns:
        {
            'is_satisfied': bool,
            'reasoning': str,
            'confidence': float,
            'missing_aspects': List[str]
        }
    """

    # Build context of what's been done
    completed_context = "\n\n".join([
        f"Task: {t['task_title']}\nResult: {t.get('result', {}).get('final_message', 'No result')[:500]}"
        for t in completed_tasks
        if t.get('result', {}).get('final_message')
    ])

    if not completed_context:
        completed_context = "No substantive results yet."

    remaining_context = "\n".join([
        f"- {task.title}: {task.description}"
        for task in remaining_tasks
    ])

    if not remaining_context:
        remaining_context = "No remaining tasks."

    satisfaction_prompt = f"""Evaluate if the original query has been fully answered:

Original Query: "{original_query}"

Completed Work:
{completed_context}

Remaining Planned Tasks:
{remaining_context}

Questions to answer:
1. Has the original query been completely answered by the completed work?
2. Are the remaining tasks redundant or unnecessary?
3. Is there critical information still missing?

Respond in JSON:
{{
    "is_satisfied": true | false,
    "reasoning": "Explanation of decision",
    "confidence": 0.0-1.0,
    "missing_aspects": ["list of what's still needed, if any"]
}}

Be practical: If the query is "how many lines in commands.py" and we found it contains 716 lines,
we're done - no need to "verify" or "report" again. Similarly, if a task says "verify result" but
the result has already been verified, that task is redundant."""

    try:
        from ai_dev_agent.providers.llm.base import Message
        messages = [Message(role="user", content=satisfaction_prompt)]

        response_text = client.complete(
            messages=messages,
            temperature=0.1,
        )

        # Parse JSON response
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        result = json.loads(response_text)

        # Validate fields
        if 'is_satisfied' not in result:
            result['is_satisfied'] = False
        if 'confidence' not in result:
            result['confidence'] = 0.5
        if 'reasoning' not in result:
            result['reasoning'] = 'Unable to determine'

        return result

    except Exception as e:
        # Fallback: continue with remaining tasks (safe default)
        click.echo(f"⚠️  Satisfaction check error: {e}")
        return {
            'is_satisfied': False,
            'reasoning': f'Assessment failed: {e}',
            'confidence': 0.0,
            'missing_aspects': ['Could not determine']
        }
