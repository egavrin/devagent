"""Tests for the agent factory."""

from unittest.mock import Mock, patch

import pytest

from ai_dev_agent.agents.factory import AgentFactory, create_agent, get_agent_factory
from ai_dev_agent.agents.strategies import (
    AgentStrategy,
    DesignAgentStrategy,
    ImplementationAgentStrategy,
    ReviewAgentStrategy,
)
from ai_dev_agent.agents.strategies import TestAgentStrategy as TestGenerationStrategy


class TestAgentFactory:
    """Test the AgentFactory class."""

    def test_init(self):
        """Test factory initialization."""
        factory = AgentFactory()
        assert factory.prompt_loader is not None
        assert len(factory._instances) == 0

    def test_init_with_custom_loader(self):
        """Test factory with custom prompt loader."""
        mock_loader = Mock()
        factory = AgentFactory(prompt_loader=mock_loader)
        assert factory.prompt_loader == mock_loader

    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = AgentFactory.list_strategies()
        assert "design" in strategies
        assert "test" in strategies
        assert "implementation" in strategies
        assert "review" in strategies

    def test_register_strategy(self):
        """Test registering a custom strategy."""

        class CustomStrategy(AgentStrategy):
            @property
            def name(self):
                return "custom"

            @property
            def description(self):
                return "Custom strategy"

            def build_prompt(self, task, context=None):
                return "custom prompt"

            def validate_input(self, task, context=None):
                return True

            def process_output(self, output):
                return {"processed": output}

        # Register the strategy
        AgentFactory.register_strategy("custom", CustomStrategy)

        # Verify it's registered
        assert "custom" in AgentFactory.list_strategies()

        # Create instance
        factory = AgentFactory()
        agent = factory.create_agent("custom")
        assert isinstance(agent, CustomStrategy)

        # Clean up
        del AgentFactory._strategies["custom"]

    def test_create_agent(self):
        """Test creating agents."""
        factory = AgentFactory()

        # Create design agent
        design_agent = factory.create_agent("design")
        assert isinstance(design_agent, DesignAgentStrategy)

        # Create test agent
        test_agent = factory.create_agent("test")
        assert isinstance(test_agent, TestGenerationStrategy)

        # Create implementation agent
        impl_agent = factory.create_agent("implementation")
        assert isinstance(impl_agent, ImplementationAgentStrategy)

        # Create review agent
        review_agent = factory.create_agent("review")
        assert isinstance(review_agent, ReviewAgentStrategy)

    def test_create_agent_invalid_type(self):
        """Test creating agent with invalid type raises error."""
        factory = AgentFactory()
        with pytest.raises(ValueError, match="Unknown agent type"):
            factory.create_agent("invalid_type")

    def test_create_agent_with_cache(self):
        """Test agent caching."""
        factory = AgentFactory()

        # Create agent with cache (default)
        agent1 = factory.create_agent("design", use_cache=True)
        agent2 = factory.create_agent("design", use_cache=True)

        # Should return same instance
        assert agent1 is agent2
        assert len(factory._instances) == 1

    def test_create_agent_without_cache(self):
        """Test creating agents without cache."""
        factory = AgentFactory()

        # Create agents without cache
        agent1 = factory.create_agent("design", use_cache=False)
        agent2 = factory.create_agent("design", use_cache=False)

        # Should return different instances
        assert agent1 is not agent2
        assert len(factory._instances) == 0

    def test_specific_agent_creation_methods(self):
        """Test convenience methods for specific agent types."""
        factory = AgentFactory()

        design = factory.create_design_agent()
        assert isinstance(design, DesignAgentStrategy)

        test = factory.create_test_agent()
        assert isinstance(test, TestGenerationStrategy)

        impl = factory.create_implementation_agent()
        assert isinstance(impl, ImplementationAgentStrategy)

        review = factory.create_review_agent()
        assert isinstance(review, ReviewAgentStrategy)

    def test_clear_cache(self):
        """Test clearing the agent cache."""
        factory = AgentFactory()

        # Create cached agents
        factory.create_agent("design", use_cache=True)
        factory.create_agent("test", use_cache=True)
        assert len(factory._instances) == 2

        # Clear cache
        factory.clear_cache()
        assert len(factory._instances) == 0


class TestGlobalFactory:
    """Test global factory functions."""

    def test_get_agent_factory_singleton(self):
        """Test that get_agent_factory returns singleton."""
        factory1 = get_agent_factory()
        factory2 = get_agent_factory()
        assert factory1 is factory2

    def test_create_agent_convenience(self):
        """Test convenience function for creating agents."""
        # Create agent without context
        agent = create_agent("design")
        assert isinstance(agent, DesignAgentStrategy)

        # Create agent with context
        context = {"workspace": "/test", "key": "value"}
        agent = create_agent("test", context=context)
        assert isinstance(agent, TestGenerationStrategy)
        assert agent.get_context() == context
