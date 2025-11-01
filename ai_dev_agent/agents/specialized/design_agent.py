"""Thin adapter exposing the design strategy as a BaseAgent."""

from __future__ import annotations

from ai_dev_agent.agents.strategies.design import DesignAgentStrategy
from ai_dev_agent.agents.strategy_adapter import StrategyAgentAdapter


class DesignAgent(StrategyAgentAdapter):
    """Backwards-compatible design agent backed by the strategy implementation."""

    def __init__(self):
        super().__init__(
            DesignAgentStrategy(),
            tools=["read", "write", "grep", "find", "symbols"],
            capabilities=[
                "technical_design",
                "reference_analysis",
                "architecture_design",
                "pattern_extraction",
            ],
            max_iterations=30,
        )
