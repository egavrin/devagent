"""Thin adapter exposing the test generation strategy as a BaseAgent."""

from __future__ import annotations

from ai_dev_agent.agents.strategies.test import TestGenerationAgentStrategy
from ai_dev_agent.agents.strategy_adapter import StrategyAgentAdapter


class TestingAgent(StrategyAgentAdapter):
    """Backwards-compatible testing agent backed by the strategy implementation."""

    __test__ = False

    def __init__(self):
        super().__init__(
            TestGenerationAgentStrategy(),
            tools=["read", "edit", "grep", "find", "run"],
            capabilities=[
                "test_generation",
                "tdd_workflow",
                "coverage_analysis",
                "fixture_creation",
            ],
            max_iterations=25,
        )
