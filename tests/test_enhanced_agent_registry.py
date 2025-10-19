"""Tests for enhanced agent registry with dynamic discovery."""
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch
import tempfile
import json
import os

from ai_dev_agent.agents.registry import AgentSpec
from ai_dev_agent.agents.enhanced_registry import (
    EnhancedAgentRegistry,
    AgentLoader,
    AgentDiscovery
)
from ai_dev_agent.agents.base import BaseAgent, AgentCapability


class TestEnhancedAgentRegistry:
    """Test enhanced agent registry functionality."""

    def test_register_base_agent(self):
        """Test registering a BaseAgent instance."""
        registry = EnhancedAgentRegistry()

        agent = BaseAgent(
            name="test_agent",
            description="Test agent",
            tools=["read", "write"]
        )

        registry.register_agent(agent)

        assert registry.has_agent("test_agent")
        retrieved = registry.get_agent("test_agent")
        assert retrieved.name == "test_agent"
        assert retrieved.description == "Test agent"

    def test_register_agent_spec(self):
        """Test registering an agent spec."""
        registry = EnhancedAgentRegistry()

        spec = AgentSpec(
            name="spec_agent",
            tools=["grep", "find"],
            max_iterations=20
        )

        registry.register_spec(spec)

        assert registry.has_spec("spec_agent")
        retrieved_spec = registry.get_spec("spec_agent")
        assert retrieved_spec.name == "spec_agent"

    def test_create_agent_from_spec(self):
        """Test creating an agent instance from spec."""
        registry = EnhancedAgentRegistry()

        spec = AgentSpec(
            name="spec_agent",
            tools=["read"],
            max_iterations=15,
            description="Created from spec"
        )

        registry.register_spec(spec)
        agent = registry.create_agent("spec_agent")

        assert agent.name == "spec_agent"
        assert agent.tools == ["read"]
        assert agent.max_iterations == 15
        assert agent.description == "Created from spec"

    def test_list_agents_by_capability(self):
        """Test listing agents by capability."""
        registry = EnhancedAgentRegistry()

        # Register agents with different capabilities
        agent1 = BaseAgent(name="agent1", capabilities=["code_review"])
        agent2 = BaseAgent(name="agent2", capabilities=["test_generation"])
        agent3 = BaseAgent(name="agent3", capabilities=["code_review", "test_generation"])

        registry.register_agent(agent1)
        registry.register_agent(agent2)
        registry.register_agent(agent3)

        # Find agents with code_review capability
        review_agents = registry.find_agents_by_capability("code_review")
        assert len(review_agents) == 2
        assert "agent1" in review_agents
        assert "agent3" in review_agents

        # Find agents with test_generation capability
        test_agents = registry.find_agents_by_capability("test_generation")
        assert len(test_agents) == 2
        assert "agent2" in test_agents
        assert "agent3" in test_agents

    def test_list_agents_by_tools(self):
        """Test finding agents by required tools."""
        registry = EnhancedAgentRegistry()

        agent1 = BaseAgent(name="agent1", tools=["read", "write"])
        agent2 = BaseAgent(name="agent2", tools=["read", "grep", "find"])
        agent3 = BaseAgent(name="agent3", tools=["read"])

        registry.register_agent(agent1)
        registry.register_agent(agent2)
        registry.register_agent(agent3)

        # Find agents that have both read and write
        agents = registry.find_agents_with_tools(["read", "write"])
        assert len(agents) == 1
        assert "agent1" in agents

        # Find agents that have read
        agents = registry.find_agents_with_tools(["read"])
        assert len(agents) == 3

    def test_agent_metadata(self):
        """Test agent metadata handling."""
        registry = EnhancedAgentRegistry()

        agent = BaseAgent(
            name="meta_agent",
            metadata={
                "version": "1.0",
                "author": "test",
                "tags": ["experimental", "ml"]
            }
        )

        registry.register_agent(agent)
        retrieved = registry.get_agent("meta_agent")

        assert retrieved.metadata["version"] == "1.0"
        assert "experimental" in retrieved.metadata["tags"]

    def test_registry_persistence(self):
        """Test saving and loading registry state."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            registry = EnhancedAgentRegistry()

            # Add some agents
            spec1 = AgentSpec(name="persist1", tools=["read"], max_iterations=10)
            spec2 = AgentSpec(name="persist2", tools=["write"], max_iterations=10)
            registry.register_spec(spec1)
            registry.register_spec(spec2)

            # Save registry
            registry.save_to_file(temp_file)

            # Create new registry and load
            new_registry = EnhancedAgentRegistry()
            new_registry.load_from_file(temp_file)

            # Verify agents were loaded
            assert new_registry.has_spec("persist1")
            assert new_registry.has_spec("persist2")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_singleton_registry(self):
        """Test that registry can work as singleton."""
        registry1 = EnhancedAgentRegistry.get_instance()
        registry2 = EnhancedAgentRegistry.get_instance()

        assert registry1 is registry2

        # Register agent in one instance
        agent = BaseAgent(name="singleton_test")
        registry1.register_agent(agent)

        # Should be available in other instance
        assert registry2.has_agent("singleton_test")


class TestAgentLoader:
    """Test agent loader functionality."""

    def test_load_agent_from_module(self):
        """Test loading agent from Python module."""
        loader = AgentLoader()

        # Mock module import
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.TestAgent = type('TestAgent', (BaseAgent,), {
                '__init__': lambda self: BaseAgent.__init__(self, name="loaded_agent")
            })
            mock_import.return_value = mock_module

            agent_class = loader.load_agent_class("test_module", "TestAgent")
            agent = agent_class()

            assert agent.name == "loaded_agent"

    def test_load_agent_from_config(self):
        """Test loading agent from config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "name": "config_agent",
                "type": "BaseAgent",
                "tools": ["read", "write"],
                "max_iterations": 25,
                "description": "Loaded from config",
                "capabilities": ["analysis"]
            }
            json.dump(config, f)
            temp_file = f.name

        try:
            loader = AgentLoader()
            agent = loader.load_from_config(temp_file)

            assert agent.name == "config_agent"
            assert agent.tools == ["read", "write"]
            assert agent.max_iterations == 25
            assert "analysis" in agent.capabilities

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestAgentDiscovery:
    """Test agent auto-discovery."""

    def test_discover_agents_in_directory(self):
        """Test discovering agent modules in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test agent files
            agent1_code = """
from ai_dev_agent.agents.base import BaseAgent

class CustomAgent1(BaseAgent):
    def __init__(self):
        super().__init__(name="custom1", tools=["read"])
"""
            agent2_code = """
from ai_dev_agent.agents.base import BaseAgent

class CustomAgent2(BaseAgent):
    def __init__(self):
        super().__init__(name="custom2", tools=["write"])
"""
            with open(os.path.join(tmpdir, "agent1.py"), 'w') as f:
                f.write(agent1_code)
            with open(os.path.join(tmpdir, "agent2.py"), 'w') as f:
                f.write(agent2_code)

            discovery = AgentDiscovery()
            agents = discovery.discover_in_directory(tmpdir)

            # Should find 2 agent files
            assert len(agents) >= 2

    def test_discover_builtin_agents(self):
        """Test discovering built-in agents."""
        discovery = AgentDiscovery()
        builtin = discovery.discover_builtin_agents()

        # Should have at least the default agents
        assert "manager" in builtin
        assert "reviewer" in builtin

    def test_validate_agent_class(self):
        """Test validating agent class."""
        discovery = AgentDiscovery()

        # Valid agent class
        class ValidAgent(BaseAgent):
            pass

        assert discovery.is_valid_agent_class(ValidAgent)

        # Invalid agent class
        class InvalidAgent:
            pass

        assert not discovery.is_valid_agent_class(InvalidAgent)

    def test_auto_register_discovered_agents(self):
        """Test auto-registering discovered agents."""
        registry = EnhancedAgentRegistry()
        discovery = AgentDiscovery()

        # Create a test agent class
        class DiscoveredAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="discovered", tools=["grep"])

        # Simulate discovery and registration
        discovery.register_agent_class(registry, DiscoveredAgent)

        # Should be able to create the agent
        agent = registry.create_agent("discovered")
        assert agent.name == "discovered"
        assert "grep" in agent.tools