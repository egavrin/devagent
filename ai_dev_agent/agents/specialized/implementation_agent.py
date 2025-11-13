"""Thin adapter exposing the implementation strategy as a BaseAgent."""

from __future__ import annotations

from ai_dev_agent.agents.strategies.implementation import ImplementationAgentStrategy
from ai_dev_agent.agents.strategy_adapter import StrategyAgentAdapter


class ImplementationAgent(StrategyAgentAdapter):
    """Backwards-compatible implementation agent backed by the strategy implementation."""

    def __init__(self):
        super().__init__(
            ImplementationAgentStrategy(),
            tools=["read", "edit", "grep", "find", "run"],
            capabilities=[
                "code_implementation",
                "incremental_development",
                "status_tracking",
                "error_handling",
            ],
            max_iterations=40,
        )
