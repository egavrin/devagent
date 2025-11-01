"""Guards to ensure pytest does not attempt to collect production agent classes."""

from ai_dev_agent.agents.specialized.testing_agent import TestingAgent
from ai_dev_agent.agents.strategies.test import TestGenerationAgentStrategy


def test_test_generation_strategy_opt_outs_pytest_collection() -> None:
    """The strategy class must opt out of pytest's Test* auto-discovery."""
    assert hasattr(TestGenerationAgentStrategy, "__test__")
    assert getattr(TestGenerationAgentStrategy, "__test__") is False


def test_testing_agent_adapter_opt_outs_pytest_collection() -> None:
    """The adapter agent must opt out of pytest's Test* auto-discovery."""
    assert hasattr(TestingAgent, "__test__")
    assert getattr(TestingAgent, "__test__") is False
