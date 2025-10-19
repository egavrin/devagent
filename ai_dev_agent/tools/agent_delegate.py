"""Agent delegation tool for manager agent to delegate to specialized agents."""
from __future__ import annotations

from typing import Any, Mapping

from .registry import ToolSpec, ToolContext, registry
from ..agents.base import AgentContext
from ..agents.specialized.executor_bridge import execute_agent_with_react
from ..agents.specialized import (
    DesignAgent,
    TestingAgent,
    ImplementationAgent,
    ReviewAgent,
)


def delegate(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    """
    Delegate a task to a specialized agent.

    This tool allows the manager agent to delegate tasks to specialized agents
    that have specific expertise in design, testing, implementation, or review.

    Args:
        payload: Dict with 'agent' (agent name) and 'task' (task description)
        context: Tool execution context

    Returns:
        Dict with execution results
    """
    agent_name = payload.get("agent", "")
    task_description = payload.get("task", "")

    if not agent_name:
        return {
            "success": False,
            "error": "Missing required parameter: agent"
        }

    if not task_description:
        return {
            "success": False,
            "error": "Missing required parameter: task"
        }

    # Map agent names to agent instances
    agent_map = {
        "design_agent": DesignAgent(),
        "test_agent": TestingAgent(),
        "implementation_agent": ImplementationAgent(),
        "review_agent": ReviewAgent(),
    }

    if agent_name not in agent_map:
        return {
            "success": False,
            "error": f"Unknown agent: {agent_name}. Available: {', '.join(agent_map.keys())}"
        }

    # Get the specialized agent
    specialized_agent = agent_map[agent_name]

    # Extract required values from context.extra
    extra = context.extra or {}
    session_id = extra.get("session_id", "delegation")
    cli_context = extra.get("cli_context")
    llm_client = extra.get("llm_client")

    if not cli_context or not llm_client:
        return {
            "success": False,
            "error": "Delegation requires cli_context and llm_client in context.extra"
        }

    # Create agent context from tool context
    agent_context = AgentContext(
        session_id=session_id,
        working_directory=context.repo_root
    )

    # Execute the task using the specialized agent
    # This will use the ReAct workflow with the agent's system prompt
    try:
        result = execute_agent_with_react(
            agent=specialized_agent,
            prompt=task_description,
            context=agent_context,
            ctx=cli_context,
            cli_client=llm_client,
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Agent execution failed: {str(e)}"
        }

    return {
        "success": result.success,
        "output": result.output,
        "metadata": result.metadata or {},
        "error": result.error
    }


# Register the delegation tool
registry.register(
    ToolSpec(
        name="delegate",
        handler=delegate,
        request_schema_path=None,  # Will use minimal schema
        response_schema_path=None,
        description=(
            "Delegate a task to a specialized agent.\n\n"
            "Available specialized agents:\n"
            "- design_agent: Creates technical designs, analyzes architecture, extracts patterns from references\n"
            "- test_agent: Generates comprehensive tests following TDD workflow, validates coverage\n"
            "- implementation_agent: Implements code from designs using TDD principles, makes tests pass\n"
            "- review_agent: Reviews code for quality, security, and best practices (read-only)\n\n"
            "Usage:\n"
            "  delegate(agent='design_agent', task='Design REST API for user management')\n"
            "  delegate(agent='test_agent', task='Generate tests for authentication module with 90% coverage')\n"
            "  delegate(agent='review_agent', task='Review security of payment processing code')\n\n"
            "The specialized agent will use its expertise and specialized system prompt to complete the task."
        ),
        category="agents",
    )
)
