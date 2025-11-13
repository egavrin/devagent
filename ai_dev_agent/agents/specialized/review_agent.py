"""Thin adapter exposing the review strategy as a BaseAgent."""

from __future__ import annotations

from ai_dev_agent.agents.strategies.review import ReviewAgentStrategy
from ai_dev_agent.agents.strategy_adapter import StrategyAgentAdapter


class ReviewAgent(StrategyAgentAdapter):
    """Backwards-compatible review agent backed by the strategy implementation."""

    def __init__(self):
        super().__init__(
            ReviewAgentStrategy(),
            tools=["read", "grep", "find", "symbols"],
            capabilities=[
                "code_quality",
                "security_analysis",
                "performance_review",
                "best_practices",
            ],
            max_iterations=30,
            permissions={"edit": "deny", "run": "deny"},
        )
