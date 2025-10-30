"""Tests for registry-backed tools."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.constants import RUN_STDOUT_TAIL_CHARS
from ai_dev_agent.tools import READ, RUN, WRITE, ToolContext
from ai_dev_agent.tools import registry as tool_registry


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)


def _make_context(root: Path, sandbox=None) -> ToolContext:
    settings = Settings(workspace_root=root)
    return ToolContext(
        repo_root=root,
        settings=settings,
        sandbox=sandbox,
        devagent_config=None,
        metrics_collector=None,
        extra={},
    )


def test_read_and_write(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    target = tmp_path / "hello.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    ctx = _make_context(tmp_path)

    read_result = tool_registry.invoke(READ, {"paths": ["hello.py"]}, ctx)
    assert read_result["files"][0]["content"].startswith("print"), read_result

    diff = """diff --git a/hello.py b/hello.py
--- a/hello.py
+++ b/hello.py
@@ -1 +1 @@
-print('hello')
+print('world')
"""
    apply_result = tool_registry.invoke(WRITE, {"diff": diff}, ctx)
    assert apply_result["applied"] is True
    assert apply_result["diff_stats"]["lines"] == 2
    assert (tmp_path / "hello.py").read_text(encoding="utf-8").strip() == "print('world')"


def test_read_missing_file_returns_value_error(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    ctx = _make_context(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        tool_registry.invoke(READ, {"paths": ["missing.txt"]}, ctx)

    assert "missing.txt" in str(exc_info.value)


def test_find_returns_matching_files(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py", "b.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add files"], cwd=tmp_path, check=True)

    ctx = _make_context(tmp_path)
    result = tool_registry.invoke("find", {"query": "*.py"}, ctx)
    files = {Path(entry["path"]).name for entry in result.get("files", [])}
    assert "a.py" in files
    assert "b.txt" not in files


def test_grep_returns_grouped_matches(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    target = tmp_path / "notes.md"
    target.write_text("hello\nhello world\n", encoding="utf-8")
    subprocess.run(["git", "add", "notes.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add notes"], cwd=tmp_path, check=True)

    ctx = _make_context(tmp_path)
    result = tool_registry.invoke("grep", {"pattern": "hello"}, ctx)
    matches = result.get("matches") or []
    assert matches, result
    first = matches[0]
    assert first.get("file", "").endswith("notes.md")
    lines = [m.get("line") for m in first.get("matches", [])]
    assert lines == [1, 2]


def _has_universal_ctags() -> bool:
    path = shutil.which("ctags")
    if not path:
        return False
    proc = subprocess.run([path, "--version"], capture_output=True, text=True)
    return "Universal Ctags" in proc.stdout


@pytest.mark.skipif(not _has_universal_ctags(), reason="Universal Ctags not available")
def test_symbols_returns_results(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    target = tmp_path / "lib.py"
    target.write_text("""def sample():\n    return 1\n""", encoding="utf-8")
    subprocess.run(["git", "add", "lib.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add lib"], cwd=tmp_path, check=True)

    ctx = _make_context(tmp_path)
    result = tool_registry.invoke("symbols", {"name": "sample"}, ctx)
    symbols = result.get("symbols") or []
    assert symbols, result
    names = {entry.get("name") for entry in symbols}
    assert any(name and name.lower().startswith("sample") for name in names)


def test_exec_tool(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    result = tool_registry.invoke(RUN, {"cmd": "echo", "args": ["hello"]}, ctx)
    assert result["exit_code"] == 0
    assert "hello" in result["stdout_tail"]


def test_exec_tool_handles_pipes(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    (tmp_path / "one.txt").write_text("a", encoding="utf-8")
    (tmp_path / "two.txt").write_text("b", encoding="utf-8")

    command = "find . -maxdepth 1 -type f -print | wc -l"
    result = tool_registry.invoke(RUN, {"cmd": command}, ctx)

    assert result["exit_code"] == 0
    assert result["stdout_tail"].strip() == "2"


def test_exec_tool_handles_redirection(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    target = tmp_path / "redirected.txt"

    result = tool_registry.invoke(RUN, {"cmd": 'echo "hello" >> redirected.txt'}, ctx)

    assert result["exit_code"] == 0
    assert target.exists()
    assert target.read_text(encoding="utf-8").strip() == "hello"


def test_exec_tool_truncates_with_marker(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    oversize = "x" * (RUN_STDOUT_TAIL_CHARS + 250)
    result = tool_registry.invoke(
        RUN,
        {
            "cmd": sys.executable,
            "args": ["-c", f"print('{oversize}')"],
        },
        ctx,
    )

    stdout_tail = result["stdout_tail"]
    assert "... truncated:" in stdout_tail
    assert "characters omitted" in stdout_tail
    assert stdout_tail.rstrip().endswith("characters omitted ...]")
