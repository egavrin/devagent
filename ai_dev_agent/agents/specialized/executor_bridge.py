"""Bridge between BaseAgent and ReAct execution engine.

This module connects the specialized agent framework (BaseAgent) with the
existing ReAct workflow execution system, enabling real tool usage and LLM calls.
"""

from __future__ import annotations

from typing import Any

import click

from ai_dev_agent.agents import AgentRegistry
from ai_dev_agent.agents.base import AgentContext, AgentResult, BaseAgent
from ai_dev_agent.core.utils.config import Settings


class AgentExecutor:
    """Bridges BaseAgent interface to ReAct execution engine."""

    def __init__(self):
        """Initialize the agent executor bridge."""
        self._settings_cache: Settings | None = None

    def execute_with_react(
        self,
        agent: BaseAgent,
        prompt: str,
        context: AgentContext,
        ctx: click.Context | None = None,
        cli_client: Any | None = None,
    ) -> AgentResult:
        """Execute agent task using the ReAct workflow.

        Args:
            agent: BaseAgent instance to execute
            prompt: Task description/prompt
            context: Agent execution context
            ctx: Click context (optional, will create if not provided)
            cli_client: LLM client (optional, will create if not provided)

        Returns:
            AgentResult with execution outcome
        """
        try:
            # Import here to avoid circular dependencies
            from uuid import uuid4

            from ai_dev_agent.cli.react.executor import _execute_react_assistant
            from ai_dev_agent.cli.utils import get_llm_client

            # Get or create settings
            settings = self._get_settings()

            # Get or create Click context
            if ctx is None:
                ctx = self._create_click_context(settings)
            else:
                # Even if ctx provided, ensure unique session for nested execution
                if not isinstance(ctx.obj, dict):
                    ctx.obj = {}
                # Force unique session ID to avoid message flow conflicts
                ctx.obj["_session_id"] = f"delegate-{uuid4()}"
                ctx.obj["silent_mode"] = True

            # Get or create LLM client
            if cli_client is None:
                cli_client = get_llm_client(ctx)

            # Get agent spec from registry to get system prompt
            agent_spec = None
            if AgentRegistry.has_agent(agent.name):
                agent_spec = AgentRegistry.get(agent.name)

            # Build enhanced prompt with agent role
            enhanced_prompt = self._build_agent_prompt(agent, prompt, agent_spec)

            # Execute using ReAct workflow
            result_dict = _execute_react_assistant(
                ctx=ctx,
                client=cli_client,
                settings=settings,
                user_prompt=enhanced_prompt,
                use_planning=False,  # Don't nest planning
                agent_type=agent.name if AgentRegistry.has_agent(agent.name) else "manager",
                suppress_final_output=True,  # We'll handle output
            )

            # Convert ReAct result to AgentResult
            return self._convert_result(result_dict, agent.name)

        except Exception as e:
            # Return failure result
            return AgentResult(success=False, output=f"Agent execution failed: {e!s}", error=str(e))

    def _get_settings(self) -> Settings:
        """Get or create Settings instance."""
        if self._settings_cache is None:
            self._settings_cache = Settings()
        return self._settings_cache

    def _create_click_context(self, settings: Settings) -> click.Context:
        """Create a Click context for execution."""
        from uuid import uuid4

        from ai_dev_agent.cli.utils import _build_context

        # Create minimal context
        ctx_obj = _build_context(settings)

        # CRITICAL: Create unique session for nested execution
        # This prevents message flow conflicts with parent session
        ctx_obj["_session_id"] = f"delegate-{uuid4()}"

        # Disable status messages for nested execution
        ctx_obj["silent_mode"] = True

        # Create Click context
        ctx = click.Context(click.Command("agent-executor"), obj=ctx_obj)

        return ctx

    def _build_agent_prompt(
        self, agent: BaseAgent, prompt: str, agent_spec: Any | None = None
    ) -> str:
        """Build enhanced prompt with agent role and constraints.

        Args:
            agent: BaseAgent instance
            prompt: Original user prompt
            agent_spec: AgentSpec from registry (if registered)

        Returns:
            Enhanced prompt with role definition
        """
        # Start with base prompt
        enhanced = ""

        # Add agent role
        if agent_spec and agent_spec.system_prompt_suffix:
            # Use registered system prompt
            enhanced = agent_spec.system_prompt_suffix + "\n\n"
        else:
            # Create basic role description
            enhanced = f"# {agent.description}\n\n"
            if agent.capabilities:
                enhanced += f"Your capabilities: {', '.join(agent.capabilities)}\n\n"

        # Add permission constraints
        if agent.permissions:
            denied = [tool for tool, perm in agent.permissions.items() if perm == "deny"]
            if denied:
                enhanced += f"IMPORTANT: You CANNOT use these tools: {', '.join(denied)}\n"
                enhanced += "You are read-only and can only analyze, not modify.\n\n"

        # Add the actual task
        enhanced += f"# Task\n{prompt}"

        return enhanced

    def _convert_result(self, react_result: dict[str, Any], agent_name: str) -> AgentResult:
        """Convert ReAct execution result to AgentResult.

        Args:
            react_result: Result dict from _execute_react_assistant
            agent_name: Name of the agent

        Returns:
            AgentResult instance
        """
        # Extract final message
        final_message = react_result.get("final_message", "")

        # Extract execution metadata
        run_result = react_result.get("result")

        # Determine success
        success = True
        error_msg = None

        if run_result:
            # Check if execution had errors
            if hasattr(run_result, "exception"):
                success = False
                error_msg = str(run_result.exception) if run_result.exception else None
            elif hasattr(run_result, "stop_condition"):
                # Budget exhaustion might indicate partial failure
                if run_result.stop_condition == "budget" and (
                    not final_message or len(final_message.strip()) < 10
                ):
                    success = False
                    error_msg = "Execution reached iteration limit without completing task"

        # Build metadata
        metadata = {
            "agent": agent_name,
            "stop_condition": getattr(run_result, "stop_condition", None) if run_result else None,
            "steps_taken": len(getattr(run_result, "steps", [])) if run_result else 0,
        }

        # Add JSON output if present
        if "final_json" in react_result:
            metadata["json_output"] = react_result["final_json"]

        return AgentResult(
            success=success,
            output=final_message or "Task completed",
            metadata=metadata,
            error=error_msg,
        )


# Singleton instance for convenience
_executor = AgentExecutor()


def execute_agent_with_react(
    agent: BaseAgent,
    prompt: str,
    context: AgentContext,
    ctx: click.Context | None = None,
    cli_client: Any | None = None,
) -> AgentResult:
    """Convenience function to execute agent with ReAct.

    Args:
        agent: BaseAgent instance
        prompt: Task prompt
        context: Agent context
        ctx: Click context (optional)
        cli_client: LLM client (optional)

    Returns:
        AgentResult
    """
    return _executor.execute_with_react(agent, prompt, context, ctx, cli_client)
