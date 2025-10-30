"""Agent factory for creating agents with the strategy pattern."""

import logging
from typing import Dict, Optional, Type

from ai_dev_agent.agents.strategies import (
    AgentStrategy,
    DesignAgentStrategy,
    ImplementationAgentStrategy,
    ReviewAgentStrategy,
    TestGenerationAgentStrategy,
)
from ai_dev_agent.prompts import PromptLoader

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating agent strategies."""

    # Registry of available agent strategies
    _strategies: Dict[str, Type[AgentStrategy]] = {
        "design": DesignAgentStrategy,
        "test": TestGenerationAgentStrategy,
        "implementation": ImplementationAgentStrategy,
        "review": ReviewAgentStrategy,
    }

    def __init__(self, prompt_loader: Optional[PromptLoader] = None):
        """Initialize the agent factory.

        Args:
            prompt_loader: Optional PromptLoader instance to share across agents.
        """
        self.prompt_loader = prompt_loader or PromptLoader()
        self._instances: Dict[str, AgentStrategy] = {}

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[AgentStrategy]):
        """Register a new agent strategy.

        Args:
            name: Name for the strategy
            strategy_class: Strategy class to register
        """
        cls._strategies[name] = strategy_class
        logger.info(f"Registered agent strategy: {name}")

    @classmethod
    def list_strategies(cls) -> list:
        """List available agent strategies.

        Returns:
            List of available strategy names.
        """
        return list(cls._strategies.keys())

    def create_agent(self, agent_type: str, use_cache: bool = True) -> AgentStrategy:
        """Create an agent with the specified strategy.

        Args:
            agent_type: Type of agent to create (e.g., "design", "test")
            use_cache: Whether to cache and reuse agent instances

        Returns:
            The agent strategy instance.

        Raises:
            ValueError: If agent_type is not registered.
        """
        if agent_type not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available types: {available}")

        # Check cache if enabled
        if use_cache and agent_type in self._instances:
            logger.debug(f"Returning cached agent: {agent_type}")
            return self._instances[agent_type]

        # Create new instance
        strategy_class = self._strategies[agent_type]
        agent = strategy_class(prompt_loader=self.prompt_loader)
        logger.info(f"Created agent with strategy: {agent_type}")

        # Cache if enabled
        if use_cache:
            self._instances[agent_type] = agent

        return agent

    def create_design_agent(self) -> DesignAgentStrategy:
        """Create a design agent.

        Returns:
            Design agent strategy instance.
        """
        return self.create_agent("design")

    def create_test_agent(self) -> TestGenerationAgentStrategy:
        """Create a test generation agent.

        Returns:
            Test agent strategy instance.
        """
        return self.create_agent("test")

    def create_implementation_agent(self) -> ImplementationAgentStrategy:
        """Create an implementation agent.

        Returns:
            Implementation agent strategy instance.
        """
        return self.create_agent("implementation")

    def create_review_agent(self) -> ReviewAgentStrategy:
        """Create a code review agent.

        Returns:
            Review agent strategy instance.
        """
        return self.create_agent("review")

    def clear_cache(self):
        """Clear the agent instance cache."""
        self._instances.clear()
        logger.debug("Agent cache cleared")


# Global factory instance
_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Get the global agent factory instance.

    Returns:
        The agent factory singleton.
    """
    global _factory
    if _factory is None:
        _factory = AgentFactory()
    return _factory


def create_agent(agent_type: str, context: Optional[Dict] = None) -> AgentStrategy:
    """Convenience function to create an agent with context.

    Args:
        agent_type: Type of agent to create
        context: Optional context to set on the agent

    Returns:
        Configured agent strategy instance.
    """
    factory = get_agent_factory()
    agent = factory.create_agent(agent_type)

    if context:
        agent.set_context(context)

    return agent
