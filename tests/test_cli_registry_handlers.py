"""Tests for CLI registry handlers."""

from __future__ import annotations

import click
import pytest

from ai_dev_agent.cli.handlers.registry_handlers import (
    _build_find_payload,
    _build_grep_payload,
    _build_symbols_payload,
)
from ai_dev_agent.core.utils.config import Settings


def _make_context(command: str = "tool") -> click.Context:
    ctx = click.Context(click.Command(command))
    ctx.obj = {"settings": Settings()}
    return ctx


def test_find_requires_query() -> None:
    ctx = _make_context("find")
    with pytest.raises(click.ClickException):
        _build_find_payload(ctx, {})


def test_find_builds_payload_with_options() -> None:
    ctx = _make_context("find")
    payload, meta = _build_find_payload(
        ctx, {"query": "*.py", "path": "src", "limit": "5", "fuzzy": False}
    )
    assert payload == {"query": "*.py", "path": "src", "limit": 5, "fuzzy": False}
    assert meta == {}


@pytest.mark.parametrize(
    "pattern",
    [
        "Compile.*method",
        r"(?P<name>Task)",
        r"^TaskManager",
    ],
)
def test_grep_auto_regex_triggers(pattern: str) -> None:
    ctx = _make_context("grep")
    payload, _ = _build_grep_payload(ctx, {"pattern": pattern})

    assert payload["regex"] is True


@pytest.mark.parametrize(
    "pattern",
    [
        "def greet",
        "main.py",
        "TaskManager::CreateTaskQueue",
    ],
)
def test_grep_keeps_literal_patterns(pattern: str) -> None:
    ctx = _make_context("grep")
    payload, _ = _build_grep_payload(ctx, {"pattern": pattern})

    assert payload.get("regex") is None


def test_grep_respects_explicit_regex_flag() -> None:
    ctx = _make_context("grep")
    payload, _ = _build_grep_payload(ctx, {"pattern": "foo", "regex": False})
    assert payload["regex"] is False


def test_symbols_requires_name() -> None:
    ctx = _make_context("symbols")
    with pytest.raises(click.ClickException):
        _build_symbols_payload(ctx, {})


def test_symbols_payload_includes_optional_fields() -> None:
    ctx = _make_context("symbols")
    payload, meta = _build_symbols_payload(
        ctx,
        {
            "name": "MyClass",
            "path": "src",
            "limit": "10",
            "kind": "class",
            "lang": "python",
        },
    )
    assert payload == {
        "name": "MyClass",
        "path": "src",
        "limit": 10,
        "kind": "class",
        "lang": "python",
    }
    assert meta == {}
