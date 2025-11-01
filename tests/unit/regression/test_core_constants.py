"""Tests for core constants module."""

from ai_dev_agent.core.utils.constants import (
    DEFAULT_IGNORED_REPO_DIRS,
    DEFAULT_KEEP_LAST_ASSISTANT,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_TOOL_MESSAGES,
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    DEFAULT_RESPONSE_HEADROOM,
    MAX_HISTORY_ENTRIES,
    MAX_METRICS_ENTRIES,
    MIN_TOOL_OUTPUT_CHARS,
    RUN_STDERR_TAIL_CHARS,
    RUN_STDOUT_TAIL_CHARS,
)


def test_default_ignored_repo_dirs():
    """Test default ignored repository directories."""
    assert ".git" in DEFAULT_IGNORED_REPO_DIRS
    assert "node_modules" in DEFAULT_IGNORED_REPO_DIRS
    assert "__pycache__" in DEFAULT_IGNORED_REPO_DIRS
    assert ".venv" in DEFAULT_IGNORED_REPO_DIRS
    assert "venv" in DEFAULT_IGNORED_REPO_DIRS

    # Should be immutable (frozenset)
    assert isinstance(DEFAULT_IGNORED_REPO_DIRS, frozenset)


def test_tool_execution_limits():
    """Test tool execution limit constants."""
    assert MAX_HISTORY_ENTRIES == 50
    assert MIN_TOOL_OUTPUT_CHARS == 256
    assert DEFAULT_MAX_TOOL_OUTPUT_CHARS == 4000
    assert RUN_STDOUT_TAIL_CHARS == 16000
    assert RUN_STDERR_TAIL_CHARS == 4000
    assert MAX_METRICS_ENTRIES == 500

    # Sensible relationships
    assert DEFAULT_MAX_TOOL_OUTPUT_CHARS > MIN_TOOL_OUTPUT_CHARS
    assert RUN_STDOUT_TAIL_CHARS > RUN_STDERR_TAIL_CHARS


def test_conversation_context_defaults():
    """Test conversation context default constants."""
    assert DEFAULT_MAX_CONTEXT_TOKENS == 100000
    assert DEFAULT_RESPONSE_HEADROOM == 2000
    assert DEFAULT_MAX_TOOL_MESSAGES == 10
    assert DEFAULT_KEEP_LAST_ASSISTANT == 4

    # Sensible relationships
    assert DEFAULT_MAX_CONTEXT_TOKENS > DEFAULT_RESPONSE_HEADROOM
