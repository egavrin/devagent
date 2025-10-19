"""Tests for agent registry system."""
import pytest
from ai_dev_agent.agents import AgentRegistry, AgentSpec


def test_agent_registry_has_default_agents():
    """Verify default agents are registered."""
    agents = AgentRegistry.list_agents()
    assert "manager" in agents
    assert "reviewer" in agents


def test_agent_registry_get_manager():
    """Verify manager agent configuration."""
    manager = AgentRegistry.get("manager")
    assert manager.name == "manager"
    assert "read" in manager.tools
    assert "write" in manager.tools
    assert "run" in manager.tools
    assert manager.max_iterations == 40  # Updated from 25 for complex queries
    assert manager.system_prompt_suffix is not None  # Manager has enhanced prompt
    assert "Intelligent Task Router" in manager.system_prompt_suffix
    assert "delegate" in manager.system_prompt_suffix.lower()


def test_agent_registry_get_reviewer():
    """Verify reviewer agent configuration."""
    reviewer = AgentRegistry.get("reviewer")
    assert reviewer.name == "reviewer"
    assert "read" in reviewer.tools
    assert "parse_patch" not in reviewer.tools
    assert "write" not in reviewer.tools  # Reviewers can't write
    assert "run" not in reviewer.tools  # Reviewers can't execute
    assert reviewer.max_iterations == 30
    assert reviewer.system_prompt_suffix is not None
    assert "Patch Dataset" in reviewer.system_prompt_suffix


def test_agent_registry_unknown_agent():
    """Verify error when getting unknown agent."""
    with pytest.raises(KeyError, match="Unknown agent type"):
        AgentRegistry.get("unknown_agent")


def test_agent_registry_has_agent():
    """Verify has_agent check."""
    assert AgentRegistry.has_agent("manager")
    assert AgentRegistry.has_agent("reviewer")
    assert not AgentRegistry.has_agent("unknown")


def test_agent_registry_register_custom():
    """Verify custom agent registration."""
    original_agents = set(AgentRegistry.list_agents())

    custom_spec = AgentSpec(
        name="test_agent",
        tools=["read"],
        max_iterations=5,
        description="Test agent"
    )
    AgentRegistry.register(custom_spec)

    assert "test_agent" in AgentRegistry.list_agents()
    retrieved = AgentRegistry.get("test_agent")
    assert retrieved.name == "test_agent"
    assert retrieved.tools == ["read"]

    # Cleanup
    AgentRegistry.clear()
    from ai_dev_agent.agents.registry import _register_default_agents
    _register_default_agents()

    # Verify original agents restored
    assert set(AgentRegistry.list_agents()) == original_agents
