"""Loader for markdown-based prompt templates.

Uses {{PLACEHOLDER}} syntax via TemplateEngine to avoid conflicts with
Python code examples containing single braces.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ai_dev_agent.prompts.templates.template_engine import TemplateEngine

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
    """Loads and manages markdown-based prompt templates.

    Uses {{PLACEHOLDER}} syntax for variable substitution, which avoids
    conflicts with Python dicts, f-strings, and JSON in code examples.

    Placeholder names should use SCREAMING_SNAKE_CASE by convention.
    """

    SENTINEL = Path("system") / "base_context.md"

    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialise the prompt loader using well-defined prompt directories."""
        self.prompts_dir = self._resolve_prompts_dir(prompts_dir)
        self._engine = TemplateEngine()
        logger.debug("PromptLoader initialised with directory: %s", self.prompts_dir)

    @classmethod
    def _resolve_prompts_dir(cls, explicit: Optional[Path]) -> Path:
        """Resolve the directory that contains the markdown prompts."""
        if explicit is not None:
            candidate = Path(explicit).resolve()
            if (candidate / cls.SENTINEL).exists():
                return candidate
            raise FileNotFoundError(
                f"Prompt directory '{candidate}' does not contain {cls.SENTINEL}"
            )

        package_prompts = Path(__file__).resolve().parent
        project_prompts = package_prompts.parent.parent / "prompts"

        for candidate in (project_prompts.resolve(), package_prompts):
            if (candidate / cls.SENTINEL).exists():
                return candidate

        raise FileNotFoundError(
            "Unable to locate prompt definitions. "
            f"Checked: {project_prompts.resolve()} and {package_prompts}"
        )

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

        try:
            return _read_prompt_file(str(full_path))
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_path}: {e}")
            raise

    def render_prompt(
        self,
        prompt_path: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        strict: bool = True,
    ) -> str:
        """Load and render a prompt with context variables.

        Uses {{PLACEHOLDER}} syntax for substitution. Context keys should be
        UPPERCASE to match placeholder names.

        Args:
            prompt_path: Path to prompt file relative to prompts_dir
            context: Dictionary of variables to substitute (keys should be UPPERCASE)
            strict: If True, raise on missing placeholders (default: True)

        Returns:
            The rendered prompt with variables substituted.

        Raises:
            ValueError: If strict=True and placeholders are missing from context
        """
        template = self.load_prompt(prompt_path)

        if context is None:
            return template

        # Normalize context keys to uppercase
        normalized_context = {k.upper(): v for k, v in context.items()}

        return self._engine.resolve(template, normalized_context, strict=strict)

    def load_agent_prompt(
        self,
        agent_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Load a prompt for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., "design", "test", "implementation")
            context: Optional context variables for rendering

        Returns:
            The agent's prompt, rendered with context if provided.
        """
        prompt_path = f"agents/{agent_name}.md"
        return self.render_prompt(prompt_path, context, strict=False)

    def load_system_prompt(
        self,
        system_name: str = "base_context",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Load a system prompt.

        Args:
            system_name: Name of the system prompt (default: "base_context")
            context: Optional context variables for rendering

        Returns:
            The system prompt, rendered with context if provided.
        """
        prompt_path = f"system/{system_name}.md"
        return self.render_prompt(prompt_path, context, strict=False)

    def load_format_prompt(self, format_name: str) -> str:
        """Load a format specification prompt.

        Args:
            format_name: Name of the format (e.g., "whole_file", "edit_block", "unified_diff")

        Returns:
            The format specification prompt.
        """
        prompt_path = f"formats/{format_name}.md"
        return self.load_prompt(prompt_path)

    def compose_prompt(
        self,
        components: List[Union[str, Tuple[str, Dict[str, Any]]]],
        separator: str = "\n\n---\n\n",
    ) -> str:
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
                parts.append(self.render_prompt(path, context, strict=False))
            else:
                parts.append(self.load_prompt(component))

        return separator.join(parts)

    def list_prompts(self, category: Optional[str] = None) -> List[str]:
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

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        _read_prompt_file.cache_clear()
        logger.debug("Prompt cache cleared")

    def extract_placeholders(self, prompt_path: str) -> Set[str]:
        """Extract all placeholder names from a prompt.

        Args:
            prompt_path: Path to prompt file relative to prompts_dir

        Returns:
            Set of placeholder names (without braces)
        """
        template = self.load_prompt(prompt_path)
        return self._engine.extract_placeholders(template)

    def validate_context(
        self,
        prompt_path: str,
        context: Dict[str, Any],
    ) -> Tuple[bool, Set[str], Set[str]]:
        """Validate that context provides all required placeholders.

        Args:
            prompt_path: Path to prompt file relative to prompts_dir
            context: Values to substitute

        Returns:
            Tuple of (is_valid, missing_placeholders, unused_context_keys)
        """
        template = self.load_prompt(prompt_path)
        normalized_context = {k.upper(): v for k, v in context.items()}
        result = self._engine.validate(template, normalized_context)
        return result.is_valid, result.missing, result.unused
