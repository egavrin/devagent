"""Template engine for prompt rendering with {{VAR}} syntax.

Uses double-brace syntax to avoid conflicts with Python dicts, f-strings, and JSON.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of template validation."""

    is_valid: bool
    missing: Set[str] = field(default_factory=set)
    unused: Set[str] = field(default_factory=set)


class TemplateEngine:
    """Resolves {{PLACEHOLDER}} syntax, avoiding Python code conflicts.

    Uses double-brace syntax (e.g., {{TOOL_NAME}}) which doesn't conflict with:
    - Python dicts: {"key": "value"}
    - F-strings: f"{variable}"
    - JSON: {"field": 123}

    Placeholders use SCREAMING_SNAKE_CASE by convention for visual distinction.
    """

    # Match {{PLACEHOLDER}} but not {single_brace}
    PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")

    def resolve(
        self,
        template: str,
        context: Dict[str, Any],
        *,
        strict: bool = True,
    ) -> str:
        """Resolve placeholders in template.

        Args:
            template: Template string with {{PLACEHOLDER}} syntax
            context: Values to substitute (keys should be UPPERCASE)
            strict: If True, raise on missing placeholders

        Returns:
            Resolved template string

        Raises:
            ValueError: If strict=True and placeholders are missing from context
        """
        if strict:
            validation = self.validate(template, context)
            if not validation.is_valid:
                raise ValueError(f"Missing placeholders in template: {validation.missing}")
            if validation.unused:
                logger.warning(f"Unused context keys: {validation.unused}")

        def replacer(match: re.Match) -> str:
            key = match.group(1)
            if key in context:
                return str(context[key])
            if strict:
                raise ValueError(f"Missing placeholder: {key}")
            return match.group(0)  # Keep unresolved

        return self.PLACEHOLDER_PATTERN.sub(replacer, template)

    def extract_placeholders(self, template: str) -> Set[str]:
        """Extract all placeholder names from template.

        Args:
            template: Template string with {{PLACEHOLDER}} syntax

        Returns:
            Set of placeholder names (without braces)
        """
        return {match.group(1) for match in self.PLACEHOLDER_PATTERN.finditer(template)}

    def validate(
        self,
        template: str,
        context: Dict[str, Any],
    ) -> ValidationResult:
        """Validate that context provides all required placeholders.

        Args:
            template: Template string with {{PLACEHOLDER}} syntax
            context: Values to substitute

        Returns:
            ValidationResult with missing and unused placeholders
        """
        required = self.extract_placeholders(template)
        # Normalize context keys to uppercase for comparison
        provided = {k.upper() for k in context}

        missing = required - provided
        unused = provided - required

        return ValidationResult(
            is_valid=len(missing) == 0,
            missing=missing,
            unused=unused,
        )

    def has_placeholders(self, template: str) -> bool:
        """Check if template contains any placeholders.

        Args:
            template: Template string to check

        Returns:
            True if template contains {{PLACEHOLDER}} patterns
        """
        return bool(self.PLACEHOLDER_PATTERN.search(template))
