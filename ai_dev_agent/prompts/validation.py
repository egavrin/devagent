"""Prompt validation utilities for fail-fast variable checking.

Provides functions to validate template rendering context and detect
unsubstituted placeholders. Follows fail-fast philosophy - raises
immediately on validation failures rather than silently passing.
"""

import logging
from typing import Any, Dict, Optional, Set

from ai_dev_agent.prompts.templates.template_engine import TemplateEngine, ValidationResult

logger = logging.getLogger(__name__)


def extract_placeholders(template: str) -> Set[str]:
    """Extract all {{PLACEHOLDER}} patterns from a template.

    Args:
        template: Template string with {{PLACEHOLDER}} syntax

    Returns:
        Set of placeholder names (without braces)
    """
    engine = TemplateEngine()
    return engine.extract_placeholders(template)


def validate_render_context(
    template: str,
    context: Dict[str, Any],
    *,
    strict: bool = True,
    prompt_path: Optional[str] = None,
) -> ValidationResult:
    """Validate that context provides all required placeholders.

    Args:
        template: Template string with {{PLACEHOLDER}} syntax
        context: Values to substitute
        strict: If True, raise ValueError on missing placeholders
        prompt_path: Optional path for error messages

    Returns:
        ValidationResult with missing and unused placeholders

    Raises:
        ValueError: If strict=True and placeholders are missing
    """
    engine = TemplateEngine()
    result = engine.validate(template, context)

    path_info = f" in {prompt_path}" if prompt_path else ""

    if result.missing:
        msg = f"Missing required placeholder(s){path_info}: {result.missing}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    if result.unused:
        logger.warning(f"Unused context key(s) provided{path_info}: {result.unused}")

    return result


def check_unsubstituted_placeholders(
    rendered: str,
    *,
    prompt_path: Optional[str] = None,
) -> None:
    """Check for placeholders that weren't substituted.

    Args:
        rendered: Rendered template string to check
        prompt_path: Optional path for error messages

    Raises:
        ValueError: If unsubstituted placeholders are found
    """
    remaining = extract_placeholders(rendered)

    if remaining:
        path_info = f" in {prompt_path}" if prompt_path else ""
        raise ValueError(f"Unsubstituted placeholder(s){path_info}: {remaining}")


# Prompt variable manifest - documents required variables for each prompt
PROMPT_VARIABLE_MANIFEST: Dict[str, Dict[str, bool]] = {
    # System prompts
    "system/base_context.md": {
        "TOOL_READ": True,
        "TOOL_EDIT": True,
        "TOOL_RUN": True,
        "TOOL_FIND": True,
        "TOOL_GREP": True,
        "TOOL_SYMBOLS": True,
        "ITERATION_NOTE": False,  # Optional
        "LANGUAGE_HINT": False,  # Optional
    },
    "system/react_loop.md": {
        "ITERATION_NOTE": False,  # Optional
    },
    "system/planner_user.md": {
        "GOAL": True,
        "CONTEXT_BLOCK": True,
    },
    "system/conversation_summary_user.md": {
        "CONVERSATION": True,
        "MAX_CHARS": True,
    },
    "system/json_enforcement.md": {
        "FORMAT_SCHEMA": True,
    },
    # Format prompts
    "formats/edit_block.md": {
        "TOOL_READ": True,
    },
    # Agent prompts
    "agents/design.md": {
        "TASK": True,
        "CONTEXT": True,
        "REPO_CONTEXT": True,
    },
    "agents/implementation.md": {
        "TASK": True,
        "CONTEXT": True,
        "REPO_CONTEXT": True,
    },
    "agents/test.md": {
        "TASK": True,
        "CONTEXT": True,
        "REPO_CONTEXT": True,
    },
    "agents/review.md": {
        "FILE_PATH": False,  # Optional
        "RULE_FILE": False,  # Optional
        "CHANGE_TYPE": False,  # Optional
    },
}


def get_required_variables(prompt_path: str) -> Set[str]:
    """Get required variables for a prompt path.

    Args:
        prompt_path: Path to the prompt file (e.g., "system/base_context.md")

    Returns:
        Set of required variable names
    """
    if prompt_path not in PROMPT_VARIABLE_MANIFEST:
        return set()

    manifest = PROMPT_VARIABLE_MANIFEST[prompt_path]
    return {var for var, required in manifest.items() if required}


def get_optional_variables(prompt_path: str) -> Set[str]:
    """Get optional variables for a prompt path.

    Args:
        prompt_path: Path to the prompt file (e.g., "system/base_context.md")

    Returns:
        Set of optional variable names
    """
    if prompt_path not in PROMPT_VARIABLE_MANIFEST:
        return set()

    manifest = PROMPT_VARIABLE_MANIFEST[prompt_path]
    return {var for var, required in manifest.items() if not required}
