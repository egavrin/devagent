"""Runtime helpers for executing AgentStrategy-backed adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable

from ai_dev_agent.agents.factory import AgentFactory
from ai_dev_agent.agents.strategy_adapter import StrategyAgentAdapter

if TYPE_CHECKING:
    from ai_dev_agent.agents.base import AgentContext, AgentResult

_AGENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "design": {
        "tools": ["read", "edit", "grep", "find", "symbols"],
        "capabilities": [
            "technical_design",
            "reference_analysis",
            "architecture_design",
            "pattern_extraction",
        ],
        "max_iterations": 30,
    },
    "test": {
        "tools": ["read", "edit", "grep", "find", "run"],
        "capabilities": [
            "test_generation",
            "tdd_workflow",
            "coverage_analysis",
            "fixture_creation",
        ],
        "max_iterations": 25,
    },
    "implementation": {
        "tools": ["read", "edit", "grep", "find", "run"],
        "capabilities": [
            "code_implementation",
            "incremental_development",
            "status_tracking",
            "error_handling",
        ],
        "max_iterations": 40,
    },
    "review": {
        "tools": ["read", "grep", "find", "symbols"],
        "capabilities": [
            "code_quality",
            "security_analysis",
            "performance_review",
            "best_practices",
        ],
        "max_iterations": 30,
        "permissions": {"edit": "deny", "run": "deny"},
    },
}


def create_strategy_agent(
    agent_type: str,
    *,
    prompt_loader=None,
    tools: Iterable[str] | None = None,
    capabilities: Iterable[str] | None = None,
    max_iterations: int | None = None,
    permissions: dict[str, str] | None = None,
) -> StrategyAgentAdapter:
    """Instantiate a StrategyAgentAdapter for the requested agent type."""
    factory = AgentFactory(prompt_loader=prompt_loader)
    strategy = factory.create_agent(agent_type)

    defaults = _AGENT_DEFAULTS.get(agent_type, {})
    adapter = StrategyAgentAdapter(
        strategy,
        tools=list(tools) if tools is not None else defaults.get("tools"),
        capabilities=(
            list(capabilities) if capabilities is not None else defaults.get("capabilities")
        ),
        max_iterations=max_iterations or defaults.get("max_iterations", 30),
        permissions=permissions or defaults.get("permissions"),
    )
    return adapter


def execute_strategy(
    agent_type: str,
    prompt: str,
    context: "AgentContext",
    *,
    prompt_loader=None,
    strategy_context: dict[str, Any] | None = None,
    ctx=None,
    cli_client=None,
    adapter: StrategyAgentAdapter | None = None,
) -> "AgentResult":
    """Execute a strategy-backed agent in the ReAct runtime."""
    agent = adapter or create_strategy_agent(agent_type, prompt_loader=prompt_loader)
    if strategy_context:
        agent.set_strategy_context(strategy_context)
    return agent.execute(prompt, context, ctx=ctx, cli_client=cli_client)
