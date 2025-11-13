"""Smoke tests for the tool registry entry points."""

from __future__ import annotations

from pathlib import Path

from ai_dev_agent.tools import EDIT, FIND, GREP, READ, RUN, SYMBOLS
from ai_dev_agent.tools.registry import ToolContext, registry


def test_tool_registry_exposes_expected_tools():
    available = set(registry.available())
    for tool in (FIND, GREP, READ, RUN, SYMBOLS, EDIT):
        assert tool in available


def test_tool_registry_invokes_registered_tool(tmp_path: Path):
    context = ToolContext(repo_root=tmp_path, settings=None, sandbox=None)
    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello world\n", encoding="utf-8")

    result = registry.invoke(
        READ,
        {"paths": [str(file_path)]},
        context,
    )

    assert result["files"][0]["path"].endswith("hello.txt")
