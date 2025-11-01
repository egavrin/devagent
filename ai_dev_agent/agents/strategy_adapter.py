"""Adapters that expose AgentStrategy instances as BaseAgent implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from ai_dev_agent.agents.base import BaseAgent
from ai_dev_agent.agents.executor import execute_agent_with_react

if TYPE_CHECKING:
    from ai_dev_agent.agents.base import AgentContext, AgentResult
    from ai_dev_agent.agents.strategies import AgentStrategy


class StrategyAgentAdapter(BaseAgent):
    """Minimal BaseAgent wrapper around an AgentStrategy."""

    def __init__(
        self,
        strategy: "AgentStrategy",
        *,
        tools: Iterable[str] | None = None,
        capabilities: Iterable[str] | None = None,
        max_iterations: int = 30,
        permissions: dict[str, str] | None = None,
    ):
        self.strategy = strategy
        super().__init__(
            name=f"{strategy.name}_agent",
            description=strategy.description,
            tools=list(tools or []),
            capabilities=list(capabilities or []),
            max_iterations=max_iterations,
            permissions=permissions or {},
            metadata={"strategy": strategy.name},
        )

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------
    def set_strategy_context(self, context: dict[str, Any]) -> None:
        """Replace the strategy context entirely."""
        self.strategy.set_context(context)

    def update_strategy_context(self, updates: dict[str, Any]) -> None:
        """Update individual keys inside the strategy context."""
        self.strategy.update_context(updates)

    def get_strategy_context(self) -> dict[str, Any]:
        """Return a copy of the current strategy context."""
        return self.strategy.get_context()

    def build_prompt(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Build a prompt via the underlying strategy."""
        return self.strategy.build_prompt(task, context)

    def validate_input(self, task: str, context: dict[str, Any] | None = None) -> bool:
        """Validate inputs via the underlying strategy."""
        return self.strategy.validate_input(task, context)

    def process_output(self, output: str) -> dict[str, Any]:
        """Normalise LLM output via the strategy."""
        return self.strategy.process_output(output)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(
        self,
        prompt: str,
        context: "AgentContext",
        *,
        ctx=None,
        cli_client=None,
    ) -> "AgentResult":
        """Execute the agent task by delegating to the shared executor."""
        return execute_agent_with_react(
            agent=self,
            prompt=prompt,
            context=context,
            ctx=ctx,
            cli_client=cli_client,
        )
