"""Automatic multi-agent query execution.

This module provides a command that automatically:
1. Analyzes query complexity
2. Creates a work plan if needed
3. Spawns appropriate specialized agents
4. Executes and coordinates them
5. Returns results
"""

from pathlib import Path
from typing import Optional

import click

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.enhanced_registry import EnhancedAgentRegistry
from ai_dev_agent.agents.integration import AutomatedWorkflow
from ai_dev_agent.agents.specialized import (
    DesignAgent,
    ImplementationAgent,
    OrchestratorAgent,
    ReviewAgent,
    TestingAgent,
)
from ai_dev_agent.agents.work_planner import WorkPlanningAgent


def should_use_multi_agent(query: str) -> bool:
    """Determine if query should use multi-agent system.

    Args:
        query: User query

    Returns:
        True if multi-agent system should be used
    """
    # Keywords that suggest multi-agent workflow
    multi_agent_keywords = [
        "build",
        "create",
        "implement",
        "develop",
        "design and",
        "test and",
        "review",
        "add feature",
        "new feature",
        "authentication",
        "api",
        "endpoint",
        "with tests",
        "with security",
        "full",
        "complete",
        "entire",
    ]

    query_lower = query.lower()

    # Check for keywords
    keyword_match = any(keyword in query_lower for keyword in multi_agent_keywords)

    # Check for complexity indicators
    has_and = " and " in query_lower or "," in query
    is_long = len(query.split()) > 10

    return keyword_match or (has_and and is_long)


def execute_with_auto_agents(query: str, cwd: Optional[Path] = None, verbose: bool = False) -> dict:
    """Execute query with automatic agent spawning.

    Args:
        query: User query to execute
        cwd: Working directory (default: current directory)
        verbose: Show detailed progress

    Returns:
        Result dictionary with success, output, and data
    """
    if cwd is None:
        cwd = Path.cwd()

    # Step 1: Decide if we need multi-agent system
    use_multi_agent = should_use_multi_agent(query)

    if verbose:
        mode = "multi-agent" if use_multi_agent else "single-agent"
        click.echo(f"🤖 Mode: {mode}")

    if not use_multi_agent:
        # Simple query - just return instruction to use regular devagent
        return {
            "success": True,
            "mode": "single",
            "message": "Query will be handled by single agent",
            "recommendation": f'devagent query "{query}"',
        }

    # Step 2: Create a work plan
    if verbose:
        click.echo("📋 Creating work plan...")

    planner = WorkPlanningAgent()
    plan = planner.create_plan(goal=query, context={"auto_generated": True})

    if verbose:
        click.echo(f"✓ Plan created with {len(plan.tasks)} tasks")
        for i, task in enumerate(plan.tasks, 1):
            click.echo(f"  {i}. {task.title}")

    # Step 3: Set up multi-agent system
    if verbose:
        click.echo("\n🤖 Initializing agents...")

    registry = EnhancedAgentRegistry()
    agents_to_register = [DesignAgent(), TestingAgent(), ImplementationAgent(), ReviewAgent()]

    for agent in agents_to_register:
        registry.register_agent(agent)

    orchestrator = OrchestratorAgent()
    for agent in agents_to_register:
        orchestrator.register_subagent(agent.name, agent)

    if verbose:
        click.echo(f"✓ Registered {len(agents_to_register)} specialized agents")

    # Step 4: Execute with workflow
    if verbose:
        click.echo("\n⚡ Executing workflow...\n")

    workflow = AutomatedWorkflow()
    workflow.orchestrator = orchestrator
    context = AgentContext(session_id=f"auto-{plan.id[:8]}")

    # Progress callback
    def progress(task_id, status, message):
        if verbose:
            symbols = {"started": "→", "completed": "✓", "failed": "✗"}
            click.echo(f"{symbols.get(status, '•')} {message}")

    result = workflow.execute_plan_automatically(
        plan, context, stop_on_failure=False, progress_callback=progress if verbose else None
    )

    # Step 5: Return results
    if verbose:
        click.echo(f"\n✓ Completed {result['tasks_completed']}/{result['total_tasks']} tasks")

    return {
        "success": result["success"],
        "mode": "multi-agent",
        "plan_id": plan.id,
        "tasks_completed": result["tasks_completed"],
        "total_tasks": result["total_tasks"],
        "completion_rate": plan.get_completion_percentage(),
        "message": f"Executed with {len(agents_to_register)} specialized agents",
    }


@click.command(name="auto")
@click.argument("query", nargs=-1, required=True)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
@click.option("--force-multi", is_flag=True, help="Force multi-agent mode")
@click.option("--force-single", is_flag=True, help="Force single-agent mode")
def auto_agent_command(query: tuple, verbose: bool, force_multi: bool, force_single: bool):
    """Execute query with automatic agent selection and spawning.

    This command automatically:
    - Analyzes query complexity
    - Decides whether to use single or multi-agent approach
    - Creates work plan if needed
    - Spawns appropriate specialized agents
    - Coordinates execution
    - Returns results

    Examples:
        devagent auto "Build REST API with authentication"
        devagent auto "Fix bug in user login"
        devagent auto --verbose "Create complete CRUD system"
    """
    query_str = " ".join(query)

    if not query_str:
        raise click.UsageError("Query cannot be empty")

    click.echo(f"⚡ Query: {query_str}\n")

    # Override automatic detection if forced
    if force_multi:
        click.echo("🤖 Forced multi-agent mode\n")
        use_multi = True
    elif force_single:
        click.echo("🤖 Forced single-agent mode\n")
        click.echo('Use: devagent query "' + query_str + '"')
        return
    else:
        use_multi = should_use_multi_agent(query_str)

    if not use_multi and not force_multi:
        click.echo("💡 This query can be handled by a single agent.")
        click.echo('   Run: devagent query "' + query_str + '"')
        click.echo("\n   Or use --force-multi to use multi-agent system anyway.")
        return

    # Execute with multi-agent system
    result = execute_with_auto_agents(query_str, verbose=verbose)

    if result["mode"] == "multi-agent":
        click.echo(f"\n{'='*60}")
        click.echo("✓ Multi-agent execution complete!")
        click.echo(f"{'='*60}")
        click.echo(f"Success: {result['success']}")
        click.echo(f"Completion: {result['completion_rate']:.0f}%")
        click.echo(f"Tasks: {result['tasks_completed']}/{result['total_tasks']}")
