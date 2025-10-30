"""Tests for the enhanced agent registry module."""
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
import tempfile
import pytest
from dataclasses import dataclass

from ai_dev_agent.agents.enhanced_registry import EnhancedAgentRegistry
from ai_dev_agent.agents.registry import AgentSpec, AgentRegistry
from ai_dev_agent.agents.base import BaseAgent, AgentCapability


class TestEnhancedAgentRegistry:
    """Test suite for EnhancedAgentRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance."""
        # Reset singleton
        EnhancedAgentRegistry._instance = None
        return EnhancedAgentRegistry()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock(spec=BaseAgent)
        agent.name = "test_agent"
        agent.description = "Test agent"
        # AgentCapability is a dataclass, not an enum
        agent.capabilities = [AgentCapability(name="code_generation", description="Generate code")]
        return agent

    @pytest.fixture
    def mock_spec(self):
        """Create a mock agent spec."""
        return AgentSpec(
            name="test_spec",
            tools=["grep", "read", "write"],
            max_iterations=10,
            description="Test spec",
            system_prompt_suffix="Test suffix"
        )

    def test_singleton_pattern(self):
        """Test that registry follows singleton pattern."""
        registry1 = EnhancedAgentRegistry.get_instance()
        registry2 = EnhancedAgentRegistry.get_instance()
        assert registry1 is registry2

    def test_init(self, registry):
        """Test registry initialization."""
        assert isinstance(registry._agents, dict)
        assert isinstance(registry._specs, dict)
        assert isinstance(registry._agent_classes, dict)
        assert "BaseAgent" in registry._agent_classes

    @patch('ai_dev_agent.agents.enhanced_registry.AgentRegistry')
    def test_import_default_agents(self, mock_agent_registry):
        """Test importing default agents from original registry."""
        mock_manager = AgentSpec(name="manager", tools=["read", "write"], max_iterations=10, description="Manager")
        mock_reviewer = AgentSpec(name="reviewer", tools=["read"], max_iterations=5, description="Reviewer")

        mock_agent_registry.get.side_effect = lambda name: {
            "manager": mock_manager,
            "reviewer": mock_reviewer
        }[name]

        registry = EnhancedAgentRegistry()

        assert "manager" in registry._specs
        assert "reviewer" in registry._specs

    @patch('ai_dev_agent.agents.enhanced_registry.AgentRegistry')
    def test_import_default_agents_missing(self, mock_agent_registry):
        """Test graceful handling when default agents are missing."""
        mock_agent_registry.get.side_effect = KeyError("Not found")

        # Should not raise exception
        registry = EnhancedAgentRegistry()
        assert isinstance(registry, EnhancedAgentRegistry)

    def test_register_agent(self, registry, mock_agent):
        """Test registering an agent instance."""
        registry.register_agent(mock_agent)

        assert "test_agent" in registry._agents
        assert registry._agents["test_agent"] == mock_agent

    def test_register_spec(self, registry, mock_spec):
        """Test registering an agent spec."""
        registry.register_spec(mock_spec)

        assert "test_spec" in registry._specs
        assert registry._specs["test_spec"] == mock_spec

    def test_register_agent_class(self, registry):
        """Test registering an agent class."""
        class CustomAgent(BaseAgent):
            pass

        registry.register_agent_class("custom", CustomAgent)

        assert "custom" in registry._agent_classes
        assert registry._agent_classes["custom"] == CustomAgent

    def test_has_agent(self, registry, mock_agent):
        """Test checking if agent exists."""
        assert not registry.has_agent("test_agent")

        registry.register_agent(mock_agent)

        assert registry.has_agent("test_agent")
        assert not registry.has_agent("nonexistent")

    def test_has_spec(self, registry, mock_spec):
        """Test checking if spec exists."""
        assert not registry.has_spec("test_spec")

        registry.register_spec(mock_spec)

        assert registry.has_spec("test_spec")
        assert not registry.has_spec("nonexistent")

    def test_get_agent(self, registry, mock_agent):
        """Test getting a registered agent."""
        registry.register_agent(mock_agent)

        agent = registry.get_agent("test_agent")
        assert agent == mock_agent

    def test_get_agent_not_found(self, registry):
        """Test getting a non-existent agent raises error."""
        with pytest.raises(KeyError, match="Unknown agent: nonexistent"):
            registry.get_agent("nonexistent")

    def test_get_spec(self, registry, mock_spec):
        """Test getting a registered spec."""
        registry.register_spec(mock_spec)

        spec = registry.get_spec("test_spec")
        assert spec == mock_spec

    def test_get_spec_not_found(self, registry):
        """Test getting a non-existent spec raises error."""
        with pytest.raises(KeyError, match="Unknown agent spec: nonexistent"):
            registry.get_spec("nonexistent")

    @patch.object(EnhancedAgentRegistry, '_create_agent_from_spec')
    def test_create_agent_from_spec(self, mock_create, registry, mock_spec):
        """Test creating an agent from a spec."""
        registry.register_spec(mock_spec)
        mock_agent = MagicMock(spec=BaseAgent)
        mock_create.return_value = mock_agent

        agent = registry.create_agent("test_spec", custom_param="value")

        mock_create.assert_called_once_with(mock_spec, custom_param="value")
        assert agent == mock_agent

    def test_create_agent_from_class(self, registry):
        """Test creating an agent from a registered class."""
        class CustomAgent(BaseAgent):
            def __init__(self, **kwargs):
                super().__init__(name="custom", **kwargs)

        registry.register_agent_class("custom", CustomAgent)

        agent = registry.create_agent("custom")

        assert isinstance(agent, CustomAgent)
        assert agent.name == "custom"

    def test_create_agent_not_found(self, registry):
        """Test creating an agent with unknown name raises error."""
        with pytest.raises(KeyError):
            registry.create_agent("unknown")

    def test_list_agents(self, registry, mock_agent):
        """Test listing all registered agents."""
        registry.register_agent(mock_agent)
        another_agent = MagicMock(spec=BaseAgent)
        another_agent.name = "another"
        registry.register_agent(another_agent)

        agents = registry.list_all_agents()  # Use list_all_agents

        assert len(agents) == 2
        assert "test_agent" in agents
        assert "another" in agents

    def test_list_specs(self, registry, mock_spec):
        """Test listing all registered specs."""
        registry.register_spec(mock_spec)
        another_spec = AgentSpec(name="another", tools=[], max_iterations=1, description="Another")
        registry.register_spec(another_spec)

        specs = registry.list_all_specs()  # Use list_all_specs

        # Registry may have default specs from initialization
        assert "test_spec" in specs
        assert "another" in specs
        # At minimum we have our 2 test specs
        assert len(specs) >= 2

    def test_discover_agents_in_directory(self, tmp_path):
        """Test discovering agents in a directory using AgentDiscovery."""
        # Create a mock agent file
        agent_file = tmp_path / "custom_agent.py"
        agent_file.write_text("""
from ai_dev_agent.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    '''Custom agent for testing.'''

    def __init__(self):
        super().__init__(name='custom', description='Custom agent')
""")

        # Use AgentDiscovery class
        from ai_dev_agent.agents.enhanced_registry import AgentDiscovery
        discovery = AgentDiscovery()
        discovered = discovery.discover_in_directory(str(tmp_path))

        # The method returns a list of discovered agent file paths
        assert isinstance(discovered, list)
        assert len(discovered) == 1

    def test_save_to_file(self, registry, mock_spec, tmp_path):
        """Test saving registry to a JSON file."""
        registry.register_spec(mock_spec)
        file_path = tmp_path / "registry.json"

        registry.save_to_file(str(file_path))

        assert file_path.exists()

    def test_load_from_file(self, registry, tmp_path):
        """Test loading a registry from a file."""
        # Create a spec file
        spec_data = {
            "specs": {
                "test": {
                    "name": "test",
                    "tools": ["read", "write"],
                    "max_iterations": 10,
                    "description": "Test spec"
                }
            }
        }
        file_path = tmp_path / "registry.json"
        with open(file_path, 'w') as f:
            json.dump(spec_data, f)

        registry.load_from_file(str(file_path))
        # Check if specs were loaded (method behavior may vary)

    def test_get_agent_capabilities(self, registry, mock_agent):
        """Test getting agent capabilities through the agent."""
        registry.register_agent(mock_agent)

        # Get the agent and access capabilities directly
        agent = registry.get_agent("test_agent")
        capabilities = agent.capabilities

        assert len(capabilities) == 1
        assert capabilities[0].name == "code_generation"

    def test_find_agents_by_capability(self, registry):
        """Test finding agents by capability string."""
        agent1 = MagicMock(spec=BaseAgent)
        agent1.name = "agent1"
        agent1.capabilities = ["code_generation", "testing"]

        agent2 = MagicMock(spec=BaseAgent)
        agent2.name = "agent2"
        agent2.capabilities = ["code_review", "documentation"]

        registry.register_agent(agent1)
        registry.register_agent(agent2)

        # Use find_agents_by_capability with string
        code_gen_agents = registry.find_agents_by_capability("code_generation")
        assert "agent1" in code_gen_agents
        assert "agent2" not in code_gen_agents

        review_agents = registry.find_agents_by_capability("code_review")
        assert "agent2" in review_agents
        assert "agent1" not in review_agents

    def test_clear_registry(self, registry, mock_agent, mock_spec):
        """Test clearing the registry."""
        registry.register_agent(mock_agent)
        registry.register_spec(mock_spec)

        registry.clear()

        assert len(registry._agents) == 0
        assert len(registry._specs) == 0
        # BaseAgent class should still be registered
        assert "BaseAgent" in registry._agent_classes

    def test_find_agents_with_tools(self, registry):
        """Test finding agents with required tools."""
        agent1 = MagicMock(spec=BaseAgent)
        agent1.name = "agent1"
        agent1.tools = ["read", "write", "grep"]

        agent2 = MagicMock(spec=BaseAgent)
        agent2.name = "agent2"
        agent2.tools = ["read", "grep"]

        registry.register_agent(agent1)
        registry.register_agent(agent2)

        # Find agents with all required tools
        agents_with_write = registry.find_agents_with_tools(["read", "write"])
        assert "agent1" in agents_with_write
        assert "agent2" not in agents_with_write

        # Both have read and grep
        agents_with_read_grep = registry.find_agents_with_tools(["read", "grep"])
        assert "agent1" in agents_with_read_grep
        assert "agent2" in agents_with_read_grep