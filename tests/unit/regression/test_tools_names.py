"""Validation tests for canonical tool name exports."""

from __future__ import annotations

from ai_dev_agent.tools.names import (
    ALL_TOOLS,
    DELEGATE,
    EDIT,
    FIND,
    GET_TASK_STATUS,
    GREP,
    PLAN,
    READ,
    RUN,
    SYMBOLS,
)
from ai_dev_agent.tools.registry import registry


def test_constants_match_expected_strings() -> None:
    """Ensure the canonical tool name exports stay stable."""
    assert FIND == "find"
    assert GREP == "grep"
    assert READ == "read"
    assert RUN == "run"
    assert SYMBOLS == "symbols"
    assert DELEGATE == "delegate"
    assert GET_TASK_STATUS == "get_task_status"
    assert PLAN == "plan"
    assert EDIT == "edit"
    # WRITE has been removed - EDIT is now the universal file editing tool
    assert ALL_TOOLS == (
        READ,
        EDIT,
        RUN,
        FIND,
        GREP,
        SYMBOLS,
        DELEGATE,
        GET_TASK_STATUS,
        PLAN,
    )


def test_all_tools_are_registered() -> None:
    """Every exported tool name should be available in the registry."""
    available = set(registry.available())
    for name in ALL_TOOLS:
        assert name in available, f"{name} is missing from the registry"

    # Ensure WRITE is not registered (it's been removed)
    assert "write" not in available, "write tool should be removed from registry"
