"""Base strategy interface for agents."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ai_dev_agent.prompts import PromptLoader

logger = logging.getLogger(__name__)


class AgentStrategy(ABC):
    """Abstract base class for agent strategies."""

    def __init__(self, prompt_loader: Optional[PromptLoader] = None):
        """Initialize the agent strategy.

        Args:
            prompt_loader: Optional PromptLoader instance. Creates one if not provided.
        """
        self.prompt_loader = prompt_loader or PromptLoader()
        self._context: Dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the agent name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the agent description."""
        pass

    @abstractmethod
    def build_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the complete prompt for the agent.

        Args:
            task: The task to perform
            context: Additional context for the prompt

        Returns:
            The complete prompt string.
        """
        pass

    @abstractmethod
    def validate_input(self, task: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate input before processing.

        Args:
            task: The task to validate
            context: Additional context

        Returns:
            True if input is valid, False otherwise.
        """
        pass

    @abstractmethod
    def process_output(self, output: str) -> Dict[str, Any]:
        """Process the agent's output.

        Args:
            output: Raw output from the LLM

        Returns:
            Processed output as a dictionary.
        """
        pass

    def set_context(self, context: Dict[str, Any]):
        """Set the agent's context.

        Args:
            context: Context dictionary to set
        """
        self._context = context

    def update_context(self, updates: Dict[str, Any]):
        """Update the agent's context.

        Args:
            updates: Dictionary of updates to apply
        """
        self._context.update(updates)

    def get_context(self) -> Dict[str, Any]:
        """Get the current context.

        Returns:
            The current context dictionary.
        """
        return self._context.copy()

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.

        Returns:
            The system prompt string.
        """
        return self.prompt_loader.compose_prompt(
            [("system/base_context", self._context), "system/error_handling", "system/react_loop"]
        )

    def get_agent_prompt(self) -> str:
        """Get the agent-specific prompt.

        Returns:
            The agent-specific prompt string.
        """
        return self.prompt_loader.load_agent_prompt(self.name, self._context)

    def get_format_guidelines(self, format_type: str) -> str:
        """Get format guidelines for output.

        Args:
            format_type: Type of format (e.g., "whole_file", "edit_block", "unified_diff")

        Returns:
            The format guidelines string.
        """
        return self.prompt_loader.load_format_prompt(format_type)
