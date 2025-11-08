"""Delegate tool for spawning specialized agents synchronously.

This tool executes specialized agents in isolated sessions and returns
their results to the calling agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping

from ai_dev_agent.agents.task_queue import TaskQueue, TaskStatus, create_task
from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from ai_dev_agent.tools.registry import ToolContext

LOGGER = get_logger(__name__)


def delegate(payload: Mapping[str, Any], context: "ToolContext") -> Mapping[str, Any]:
    """
    Delegate a task to a specialized agent for synchronous execution.

    The specialized agent runs in an isolated session and returns results
    immediately. Session isolation prevents conversation conflicts.

    Args:
        payload: {"agent": "design_agent|test_agent|review_agent|implementation_agent",
                  "task": "Task description",
                  "context": {optional context dict}}
        context: Tool execution context with cli_context, llm_client in extra

    Returns:
        {"success": bool, "agent": str, "result": str, "artifacts": list,
         "error": str, "metadata": dict}
    """
    agent_name = payload.get("agent")
    task_prompt = payload.get("task")
    extra_context = payload.get("context", {})

    if not agent_name:
        return {
            "success": False,
            "error": "Missing required parameter: agent",
            "artifacts": [],
        }

    if not task_prompt:
        return {
            "success": False,
            "error": "Missing required parameter: task",
            "artifacts": [],
        }

    # Extract required context from ToolContext.extra
    extra = context.extra or {}
    cli_context = extra.get("cli_context")
    llm_client = extra.get("llm_client")
    session_id = extra.get("session_id")

    if not cli_context:
        return {
            "success": False,
            "agent": agent_name,
            "error": "Missing cli_context in tool execution context. Delegation requires active CLI session.",
            "artifacts": [],
        }

    if not llm_client:
        return {
            "success": False,
            "agent": agent_name,
            "error": "Missing llm_client in tool execution context. Delegation requires LLM client.",
            "artifacts": [],
        }

    # Validate agent name
    valid_agents = ["design_agent", "test_agent", "review_agent", "implementation_agent"]
    if agent_name not in valid_agents:
        valid_agents_str = ", ".join(valid_agents)
        return {
            "success": False,
            "agent": agent_name,
            "error": f"Unknown agent: {agent_name}. Valid agents: {valid_agents_str}",
            "artifacts": [],
        }

    task_queue = TaskQueue.get_instance()
    task = create_task(
        agent_name,
        task_prompt,
        {
            "session_id": session_id,
            "workspace": str(context.repo_root),
            "metadata": extra_context,
        },
    )
    task_queue.update_task(task)

    try:
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task_queue.update_task(task)

        # Import here to avoid circular dependencies
        from ai_dev_agent.agents.base import AgentContext
        from ai_dev_agent.agents.executor import AgentExecutor
        from ai_dev_agent.agents.specialized import (
            DesignAgent,
            ImplementationAgent,
            ReviewAgent,
            TestingAgent,
        )

        agent_map = {
            "design_agent": DesignAgent,
            "test_agent": TestingAgent,
            "review_agent": ReviewAgent,
            "implementation_agent": ImplementationAgent,
        }
        agent_class = agent_map.get(agent_name)
        if not agent_class:
            raise ValueError(f"Agent not found in map: {agent_name}")

        agent_instance = agent_class()

        metadata = {
            "workspace_root": context.repo_root,
            "settings": context.settings,
            **extra_context,
        }

        agent_context = AgentContext(
            session_id=f"{session_id}-delegate" if session_id else "delegate",
            parent_id=session_id,
            working_directory=str(context.repo_root),
            metadata=metadata,
        )

        task_id = extra_context.get("task_id")
        if task_id:
            from ai_dev_agent.tools.workflow.plan_tracker import update_task_status

            update_task_status(task_id, "in_progress")

        executor = AgentExecutor()
        result = executor.execute_with_react(
            agent=agent_instance,
            prompt=task_prompt,
            context=agent_context,
            ctx=cli_context,
            cli_client=llm_client,
        )

        task.result = result
        task.completed_at = datetime.now()
        task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        task.error = result.error
        task_queue.update_task(task)

        if task_id:
            from ai_dev_agent.tools.workflow.plan_tracker import update_task_status

            status = "completed" if result.success else "failed"
            update_task_status(task_id, status)

        artifacts = []
        if result.metadata:
            artifacts = result.metadata.get("artifacts", [])

        # Note: Task is already in short-term memory (ephemeral, process-scoped)

        return {
            "success": result.success,
            "agent": agent_name,
            "result": result.output,
            "artifacts": artifacts,
            "error": result.error,
            "task_id": task.id,
            "metadata": result.metadata or {},
        }

    except Exception as exc:
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.error = str(exc)
        task_queue.update_task(task)
        # Note: Task is in short-term memory (ephemeral, process-scoped)
        LOGGER.exception("Failed to execute %s", agent_name)
        return {
            "success": False,
            "agent": agent_name,
            "error": f"Agent execution failed: {exc!s}",
            "artifacts": [],
            "task_id": task.id,
        }
