"""Integration tests exercising the ripgrep-backed grep tool."""

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


def test_grep_groups_matches_by_file(tmp_path: Path) -> None:
    """Grep should group matches per file and order by recency."""
    repo_root = tmp_path
    src = repo_root / "src"
    src.mkdir()
    old_file = src / "legacy.py"
    new_file = src / "latest.py"
    old_file.write_text("print('legacy')\n", encoding="utf-8")
    new_file.write_text("print('latest')\n", encoding="utf-8")

    now = time.time()
    os.utime(old_file, (now - 7200, now - 7200))
    os.utime(new_file, (now, now))

    stdout = "src/legacy.py:5:first\nsrc/latest.py:8:second\nsrc/latest.py:9:third\n"
    completed = subprocess.CompletedProcess(
        args=["rg"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )

    context = _make_context(repo_root)

    with patch("ai_dev_agent.tools.grep.subprocess.run", return_value=completed):
        result = registry.invoke("grep", {"pattern": "needle"}, context)

    files = [entry["file"] for entry in result["matches"]]
    assert files[0] == "src/latest.py"
    assert [match["line"] for match in result["matches"][0]["matches"]] == [8, 9]


def test_grep_respects_max_files_limit(tmp_path: Path) -> None:
    repo_root = tmp_path
    context = _make_context(repo_root)

    stdout = "\n".join(f"src/file{i}.py:{i}:match" for i in range(5)) + "\n"
    completed = subprocess.CompletedProcess(
        args=["rg"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )

    with patch("ai_dev_agent.tools.grep.subprocess.run", return_value=completed):
        result = registry.invoke("grep", {"pattern": "match", "max_files": 2}, context)

    assert len(result["matches"]) == 2
    assert result["truncated"] is True
    assert result["total_files"] == 5


def test_grep_returns_error_when_rg_fails(tmp_path: Path) -> None:
    repo_root = tmp_path
    context = _make_context(repo_root)
    completed = subprocess.CompletedProcess(
        args=["rg"],
        returncode=2,
        stdout="",
        stderr="invalid pattern",
    )

    with patch("ai_dev_agent.tools.grep.subprocess.run", return_value=completed):
        result = registry.invoke("grep", {"pattern": "bad"}, context)

    assert result == {"error": "invalid pattern", "matches": []}


def test_grep_returns_empty_when_pattern_missing(tmp_path: Path) -> None:
    context = _make_context(tmp_path)

    with patch("ai_dev_agent.tools.grep.subprocess.run") as mock_run:
        result = registry.invoke("grep", {"pattern": ""}, context)

    assert result == {"matches": []}
    mock_run.assert_not_called()
