"""Integration tests for the ripgrep-based find tool."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_dev_agent.tools.registry import ToolContext, registry


def _make_context(root: Path) -> ToolContext:
    return ToolContext(repo_root=root, settings=None, sandbox=None)


def test_find_returns_sorted_results_and_truncation_metadata(tmp_path: Path) -> None:
    """Find should return results sorted by mtime and note truncation."""
    repo_root = tmp_path
    src = repo_root / "src"
    src.mkdir()
    newer = src / "new.py"
    older = src / "old.py"
    newer.write_text("print('new')\n", encoding="utf-8")
    older.write_text("print('old')\n", encoding="utf-8")

    now = time.time()
    os.utime(newer, (now, now))
    os.utime(older, (now - 7200, now - 7200))

    completed = subprocess.CompletedProcess(
        args=["rg"],
        returncode=0,
        stdout="src/old.py\nsrc/new.py\n",
        stderr="",
    )

    context = _make_context(repo_root)
    with patch("ai_dev_agent.tools.find.subprocess.run", return_value=completed):
        result = registry.invoke("find", {"query": "*.py", "limit": 1}, context)

    assert result["files"][0]["path"] == "src/new.py"
    assert result["files"][0]["lines"] > 0
    assert result["files"][0]["size_bytes"] > 0
    assert result["truncated"] is True
    assert result["total_files"] == 2
    assert "Showing 1 of 2" in result["message"]


def test_find_returns_error_when_rg_fails(tmp_path: Path) -> None:
    """Find should surface ripgrep errors to the caller."""
    repo_root = tmp_path
    completed = subprocess.CompletedProcess(
        args=["rg"],
        returncode=2,
        stdout="",
        stderr="ripgrep failure",
    )

    context = _make_context(repo_root)
    with patch("ai_dev_agent.tools.find.subprocess.run", return_value=completed):
        result = registry.invoke("find", {"query": "*.py"}, context)

    assert result == {"error": "ripgrep failure", "files": []}


def test_find_returns_timeout_error(tmp_path: Path) -> None:
    """Timeouts should be reported as tool errors."""
    repo_root = tmp_path

    context = _make_context(repo_root)

    def _raise_timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        cmd = args[0] if args else ["rg"]
        timeout = kwargs.get("timeout", 10)
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    with patch("ai_dev_agent.tools.find.subprocess.run", side_effect=_raise_timeout):
        result = registry.invoke("find", {"query": "*.py"}, context)

    assert result == {"error": "Search timeout", "files": []}
