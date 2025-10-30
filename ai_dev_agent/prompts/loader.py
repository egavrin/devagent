"""Loader for markdown-based prompt templates."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _read_prompt_file(path: str) -> str:
    """Read a prompt file from disk with shared caching."""
    full_path = Path(path)
    with full_path.open("r", encoding="utf-8") as handle:
        content = handle.read()
        logger.debug(f"Loaded prompt from {full_path}")
        return content


class PromptLoader:
    """Loads and manages markdown-based prompt templates."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing prompt files. Defaults to project prompts/.
        """
        if prompts_dir is None:
            # Find project root (where .git is)
            current = Path.cwd()
            while current != current.parent:
                if (current / ".git").exists():
                    prompts_dir = current / "prompts"
                    break
                current = current.parent
            else:
                # Fallback to relative path
                prompts_dir = Path(__file__).parent.parent.parent / "prompts"

        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            self.prompts_dir.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, str] = {}
        logger.info(f"PromptLoader initialized with directory: {self.prompts_dir}")

    def load_prompt(self, prompt_path: str) -> str:
        """Load a prompt from a markdown file.

        Args:
            prompt_path: Path to prompt file relative to prompts_dir (e.g., "agents/design.md")

        Returns:
            The prompt content as a string.

        Raises:
            FileNotFoundError: If the prompt file doesn't exist.
        """
        full_path = self.prompts_dir / prompt_path

        if not full_path.exists():
            # Try with .md extension if not provided
            if not prompt_path.endswith(".md"):
                full_path = self.prompts_dir / f"{prompt_path}.md"

            if not full_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        cache_key = str(full_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            content = _read_prompt_file(cache_key)
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_path}: {e}")
            raise

        self._cache[cache_key] = content
        return content

    def render_prompt(self, prompt_path: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Load and render a prompt with context variables.

        Args:
            prompt_path: Path to prompt file relative to prompts_dir
            context: Dictionary of variables to substitute in the prompt

        Returns:
            The rendered prompt with variables substituted.
        """
        template = self.load_prompt(prompt_path)

        if context is None:
            return template

        # Simple variable substitution using {variable} syntax
        rendered = template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, str(value))
                logger.debug(f"Replaced {placeholder} in prompt")

        return rendered

    def load_agent_prompt(self, agent_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Load a prompt for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., "design", "test", "implementation")
            context: Optional context variables for rendering

        Returns:
            The agent's prompt, rendered with context if provided.
        """
        prompt_path = f"agents/{agent_name}.md"
        return self.render_prompt(prompt_path, context)

    def load_system_prompt(
        self, system_name: str = "base_context", context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Load a system prompt.

        Args:
            system_name: Name of the system prompt (default: "base_context")
            context: Optional context variables for rendering

        Returns:
            The system prompt, rendered with context if provided.
        """
        prompt_path = f"system/{system_name}.md"
        return self.render_prompt(prompt_path, context)

    def load_format_prompt(self, format_name: str) -> str:
        """Load a format specification prompt.

        Args:
            format_name: Name of the format (e.g., "whole_file", "edit_block", "unified_diff")

        Returns:
            The format specification prompt.
        """
        prompt_path = f"formats/{format_name}.md"
        return self.load_prompt(prompt_path)

    def compose_prompt(self, components: list, separator: str = "\n\n---\n\n") -> str:
        """Compose multiple prompt components into a single prompt.

        Args:
            components: List of prompt paths or (path, context) tuples
            separator: String to separate components (default: markdown separator)

        Returns:
            The composed prompt string.
        """
        parts = []

        for component in components:
            if isinstance(component, tuple):
                path, context = component
                parts.append(self.render_prompt(path, context))
            else:
                parts.append(self.load_prompt(component))

        return separator.join(parts)

    def list_prompts(self, category: Optional[str] = None) -> list:
        """List available prompts.

        Args:
            category: Optional category to filter by (e.g., "agents", "system", "formats")

        Returns:
            List of available prompt paths.
        """
        prompts = []

        if category:
            category_dir = self.prompts_dir / category
            if category_dir.exists():
                prompts = [f"{category}/{f.name}" for f in category_dir.glob("*.md")]
        else:
            # List all prompts in all categories
            for category_dir in self.prompts_dir.iterdir():
                if category_dir.is_dir():
                    prompts.extend(
                        [f"{category_dir.name}/{f.name}" for f in category_dir.glob("*.md")]
                    )

        return sorted(prompts)

    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()
        _read_prompt_file.cache_clear()
        logger.debug("Prompt cache cleared")
