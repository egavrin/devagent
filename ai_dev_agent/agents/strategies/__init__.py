"""Agent strategy pattern implementation."""

from .base import AgentStrategy
from .design import DesignAgentStrategy
from .implementation import ImplementationAgentStrategy
from .review import ReviewAgentStrategy
from .test import TestGenerationAgentStrategy

# Keep TestAgentStrategy alias for backward compatibility
TestAgentStrategy = TestGenerationAgentStrategy

__all__ = [
    "AgentStrategy",
    "DesignAgentStrategy",
    "TestAgentStrategy",
    "TestGenerationAgentStrategy",
    "ImplementationAgentStrategy",
    "ReviewAgentStrategy",
]
