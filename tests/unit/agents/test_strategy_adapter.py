"""Tests for the strategy-backed agent adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.agents.base import AgentContext, AgentResult
from ai_dev_agent.agents.strategies import (
    DesignAgentStrategy,
    ImplementationAgentStrategy,
    ReviewAgentStrategy,
    TestGenerationAgentStrategy,
)
from ai_dev_agent.agents.strategy_adapter import StrategyAgentAdapter


class _StubStrategy(DesignAgentStrategy):
    """Minimal strategy stub overriding prompt/output handling for deterministic tests."""

    def __init__(self):
        super().__init__()
        self.build_prompt_called_with = None
        self.process_output_called_with = None

    def build_prompt(self, task: str, context: dict | None = None) -> str:
        self.build_prompt_called_with = (task, context)
        return f"PROMPT::{task}"

    def process_output(self, output: str) -> dict:
        self.process_output_called_with = output
        return {"processed": output.upper()}


@pytest.fixture
def agent_context():
    """Factory for a reusable agent context."""
    ctx = AgentContext(session_id="session-test")
    ctx.metadata.update({"cli_state": MagicMock(), "settings": MagicMock()})
    return ctx


@pytest.mark.parametrize(
    "strategy_cls",
    [
        DesignAgentStrategy,
        TestGenerationAgentStrategy,
        ImplementationAgentStrategy,
        ReviewAgentStrategy,
    ],
)
def test_adapter_initialises_from_strategy(strategy_cls):
    """Adapter should derive name/description/tools from the supplied strategy."""
    strategy = strategy_cls()
    adapter = StrategyAgentAdapter(strategy, tools=["read"], capabilities=["prompting"])

    assert adapter.name == f"{strategy.name}_agent"
    assert strategy.description.lower() in adapter.description.lower()
    assert "read" in adapter.tools
    assert "prompting" in adapter.capabilities


def test_adapter_executes_via_executor(agent_context):
    """execute() should delegate to the shared executor bridge."""
    strategy = _StubStrategy()
    adapter = StrategyAgentAdapter(strategy)

    with patch("ai_dev_agent.agents.strategy_adapter.execute_agent_with_react") as mock_execute:
        mock_execute.return_value = AgentResult(success=True, output="done", metadata={"x": 1})
        result = adapter.execute("PROMPT::task", agent_context)

    mock_execute.assert_called_once_with(
        agent=adapter, prompt="PROMPT::task", context=agent_context, ctx=None, cli_client=None
    )
    assert result.success is True
    assert result.metadata["x"] == 1


def test_adapter_build_prompt_uses_strategy_context(agent_context):
    """build_prompt() should ask the strategy to generate the prompt."""
    strategy = _StubStrategy()
    adapter = StrategyAgentAdapter(strategy)

    adapter.set_strategy_context({"workspace": "/repo"})
    prompt = adapter.build_prompt("Design feature", {"priority": "high"})

    assert prompt == "PROMPT::Design feature"
    assert strategy.build_prompt_called_with == ("Design feature", {"priority": "high"})


def test_adapter_process_output_invokes_strategy(agent_context):
    """process_output() should use the strategy to normalise responses."""
    strategy = _StubStrategy()
    adapter = StrategyAgentAdapter(strategy)

    processed = adapter.process_output("success")
    assert processed == {"processed": "SUCCESS"}
    assert strategy.process_output_called_with == "success"
