"""Unit tests for simplified search tools."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai_dev_agent.tools.find import find as find_tool
from ai_dev_agent.tools.grep import grep as grep_tool
from ai_dev_agent.tools.registry import ToolContext
from ai_dev_agent.tools.symbols import symbols as symbols_tool


def _make_context(root: Path) -> ToolContext:
    return ToolContext(
        repo_root=root,
        settings=None,
        sandbox=None,
        devagent_config=None,
        metrics_collector=None,
        extra={},
    )


def test_find_sorts_by_mtime_and_limits(tmp_path: Path) -> None:
    repo_root = tmp_path
    src = repo_root / "src"
    src.mkdir()
    newer = src / "new.py"
    older = src / "old.py"
    newer.write_text("print('new')\n", encoding="utf-8")
    older.write_text("print('old')\n", encoding="utf-8")

    now = time.time()
    os.utime(newer, (now, now))
    os.utime(older, (now - 3600, now - 3600))

    context = _make_context(repo_root)
    completed = subprocess.CompletedProcess([], 0, stdout="src/old.py\nsrc/new.py\n", stderr="")

    with patch("ai_dev_agent.tools.find.subprocess.run", return_value=completed):
        result = find_tool({"query": "*.py", "limit": 5}, context)
        limited = find_tool({"query": "*.py", "limit": 1}, context)

    paths = [entry["path"] for entry in result["files"]]
    assert paths == ["src/new.py", "src/old.py"]

    assert len(limited["files"]) == 1
    assert limited["files"][0]["path"] == "src/new.py"


def test_find_returns_error_message_on_failure(tmp_path: Path) -> None:
    repo_root = tmp_path
    context = _make_context(repo_root)
    completed = subprocess.CompletedProcess([], 2, stdout="", stderr="ripgrep failure")

    with patch("ai_dev_agent.tools.find.subprocess.run", return_value=completed):
        result = find_tool({"query": "*.py"}, context)

    assert result["files"] == []
    assert result["error"] == "ripgrep failure"


def test_grep_groups_results_and_sorts_by_mtime(tmp_path: Path) -> None:
    repo_root = tmp_path
    src = repo_root / "src"
    src.mkdir()
    file_a = src / "a.py"
    file_b = src / "b.py"
    file_a.write_text("print('a')\n", encoding="utf-8")
    file_b.write_text("print('b')\n", encoding="utf-8")

    now = time.time()
    os.utime(file_a, (now - 7200, now - 7200))
    os.utime(file_b, (now, now))

    context = _make_context(repo_root)
    stdout = "src/a.py:5:first match\nsrc/b.py:3:second\nsrc/b.py:9:third\n"
    completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    with patch("ai_dev_agent.tools.grep.subprocess.run", return_value=completed):
        result = grep_tool({"pattern": "match"}, context)

    files = [entry["file"] for entry in result["matches"]]
    assert files == ["src/b.py", "src/a.py"]

    b_matches = result["matches"][0]["matches"]
    assert [m["line"] for m in b_matches] == [3, 9]
    assert b_matches[0]["text"] == "second"


def test_grep_returns_empty_when_pattern_missing(tmp_path: Path) -> None:
    context = _make_context(tmp_path)

    with patch("ai_dev_agent.tools.grep.subprocess.run") as mock_run:
        result = grep_tool({"pattern": ""}, context)

    assert result == {"matches": []}
    mock_run.assert_not_called()


def test_grep_returns_error_on_failure(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    completed = subprocess.CompletedProcess([], 2, stdout="", stderr="bad pattern")

    with patch("ai_dev_agent.tools.grep.subprocess.run", return_value=completed):
        result = grep_tool({"pattern": "value"}, context)

    assert result["matches"] == []
    assert result["error"] == "bad pattern"


def test_symbols_reads_existing_tags_and_prioritizes_exact_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path
    module_path = repo_root / "pkg" / "mod.py"
    module_path.parent.mkdir()
    module_path.write_text("def target():\n    pass\n", encoding="utf-8")

    tags_file = repo_root / ".tags"
    tags_content = "\n".join(
        [
            "!_TAG_FILE_FORMAT\t2\t/extended format/",
            f'target\t{module_path}\t/^def target():$/;"\tkind:function\tline:1',
            f'target_helper\t{module_path}\t/^def target_helper():$/;"\tkind:function\tline:5',
        ]
    )
    tags_file.write_text(tags_content + "\n", encoding="utf-8")

    now = 1_700_000_000
    os.utime(tags_file, (now, now))

    completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    monkeypatch.setattr(
        "ai_dev_agent.tools.symbols.subprocess.run", lambda *args, **kwargs: completed
    )
    monkeypatch.setattr("ai_dev_agent.tools.symbols.time.time", lambda: now)

    context = _make_context(repo_root)
    result = symbols_tool({"name": "target"}, context)

    symbols = result["symbols"]
    assert [entry["name"] for entry in symbols] == ["target", "target_helper"]
    assert symbols[0]["file"] == "pkg/mod.py"
    assert symbols[0]["line"] == 1


def test_symbols_returns_error_when_ctags_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("missing")

    monkeypatch.setattr("ai_dev_agent.tools.symbols.subprocess.run", _raise)

    context = _make_context(tmp_path)
    result = symbols_tool({"name": "thing"}, context)

    assert result["symbols"] == []
    assert "ctags not found" in result["error"]
