"""Pydantic schemas for prompt context validation.

These schemas ensure type safety and validation for prompt rendering contexts.
Keys are UPPERCASE to match the {{PLACEHOLDER}} convention.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PromptContext(BaseModel):
    """Base context for system prompt rendering.

    Contains tool names and optional configuration used in base_context.md.
    """

    # Required tool names
    TOOL_READ: str = Field(description="Name of the read tool")
    TOOL_EDIT: str = Field(description="Name of the edit tool")
    TOOL_RUN: str = Field(description="Name of the run/shell tool")
    TOOL_FIND: str = Field(description="Name of the find/glob tool")
    TOOL_GREP: str = Field(description="Name of the grep/search tool")
    TOOL_SYMBOLS: str = Field(description="Name of the symbols tool")

    # Optional fields with defaults
    ITERATION_NOTE: str = Field(default="", description="Note about iteration limits")
    LANGUAGE_HINT: str = Field(default="", description="Programming language hint")

    class Config:
        extra = "allow"  # Allow additional fields for extensibility

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with UPPERCASE keys for template rendering."""
        return self.model_dump(exclude_unset=False)


class AgentPromptContext(BaseModel):
    """Context for agent-specific prompts (design, implementation, test).

    Used in agents/design.md, agents/implementation.md, agents/test.md.
    """

    TASK: str = Field(min_length=1, description="Task description")
    CONTEXT: str = Field(default="", description="Additional context")
    REPO_CONTEXT: str = Field(default="{}", description="Repository context as JSON")

    class Config:
        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with UPPERCASE keys for template rendering."""
        return self.model_dump(exclude_unset=False)


class ReviewPromptContext(BaseModel):
    """Context for code review prompts.

    Used in agents/review.md.
    """

    FILE_PATH: str = Field(default="", description="Path to file being reviewed")
    RULE_FILE: str = Field(default="", description="Path to rule file")
    CHANGE_TYPE: str = Field(default="", description="Type of change being reviewed")

    class Config:
        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with UPPERCASE keys for template rendering."""
        return self.model_dump(exclude_unset=False)


class PlannerContext(BaseModel):
    """Context for planner prompts.

    Used in system/planner_user.md.
    """

    GOAL: str = Field(min_length=1, description="Goal to plan for")
    CONTEXT_BLOCK: str = Field(default="", description="Context block for planning")

    class Config:
        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with UPPERCASE keys for template rendering."""
        return self.model_dump(exclude_unset=False)


class SummaryContext(BaseModel):
    """Context for conversation summary prompts.

    Used in system/conversation_summary_user.md.
    """

    CONVERSATION: str = Field(min_length=1, description="Conversation to summarize")
    MAX_CHARS: int = Field(ge=100, description="Maximum characters for summary")

    class Config:
        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with UPPERCASE keys for template rendering."""
        return self.model_dump(exclude_unset=False)


def create_system_context(
    tool_read: str = "read",
    tool_edit: str = "edit",
    tool_run: str = "run",
    tool_find: str = "find",
    tool_grep: str = "grep",
    tool_symbols: str = "symbols",
    iteration_note: str = "",
    language_hint: str = "",
) -> Dict[str, Any]:
    """Create a system prompt context dict with UPPERCASE keys.

    Convenience function for creating context for base_context.md rendering.

    Args:
        tool_read: Name of the read tool
        tool_edit: Name of the edit tool
        tool_run: Name of the run tool
        tool_find: Name of the find tool
        tool_grep: Name of the grep tool
        tool_symbols: Name of the symbols tool
        iteration_note: Optional iteration note
        language_hint: Optional language hint

    Returns:
        Dict with UPPERCASE keys ready for template rendering
    """
    ctx = PromptContext(
        TOOL_READ=tool_read,
        TOOL_EDIT=tool_edit,
        TOOL_RUN=tool_run,
        TOOL_FIND=tool_find,
        TOOL_GREP=tool_grep,
        TOOL_SYMBOLS=tool_symbols,
        ITERATION_NOTE=iteration_note,
        LANGUAGE_HINT=language_hint,
    )
    return ctx.to_dict()
