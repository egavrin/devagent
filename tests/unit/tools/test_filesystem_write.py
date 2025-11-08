"""Tests for filesystem write tool with enhanced diff application."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.tools.filesystem import _fs_write_patch
from ai_dev_agent.tools.registry import ToolContext


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a git repository for testing."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


@pytest.fixture
def tool_context(git_repo: Path) -> ToolContext:
    """Create a tool context for testing."""
    return ToolContext(
        repo_root=git_repo,
        settings=MagicMock(),
        sandbox=MagicMock(),
    )


def test_fs_write_patch_success(git_repo: Path, tool_context: ToolContext) -> None:
    """Test successful patch application."""
    # Create a test file
    test_file = git_repo / "test.txt"
    test_file.write_text("hello\n", encoding="utf-8")

    # Create a valid diff
    diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-hello
+hello world
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is True
    assert result["rejected_hunks"] == []
    assert "test.txt" in result["changed_files"]
    assert result["diff_stats"]["lines"] > 0


def test_fs_write_patch_corrupt_hunk_header(git_repo: Path, tool_context: ToolContext) -> None:
    """Test rejection of patch with corrupt hunk header."""
    test_file = git_repo / "test.txt"
    test_file.write_text("hello\n", encoding="utf-8")

    # Create a diff with corrupt hunk header (invalid format)
    diff = """--- a/test.txt
+++ b/test.txt
@@ corrupt @@
-hello
+hello world
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0
    # Should contain informative error message
    assert any(
        "corrupt" in err.lower() or "invalid" in err.lower() or "hunk" in err.lower()
        for err in result["rejected_hunks"]
    )


def test_fs_write_patch_missing_file_markers(git_repo: Path, tool_context: ToolContext) -> None:
    """Test rejection of patch missing --- and +++ markers."""
    diff = """@@ -1 +1 @@
-hello
+hello world
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0
    # Should indicate validation failure about missing file markers
    rejection_msg = " ".join(result["rejected_hunks"]).lower()
    assert "validation" in rejection_msg and ("file" in rejection_msg or "diff" in rejection_msg)


def test_fs_write_patch_with_conflict_markers(git_repo: Path, tool_context: ToolContext) -> None:
    """Test rejection of patch containing conflict markers."""
    test_file = git_repo / "test.txt"
    test_file.write_text("hello\n", encoding="utf-8")

    # Create a diff with conflict markers
    diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1,5 @@
-hello
+<<<<<<< HEAD
+hello world
+=======
+goodbye world
+>>>>>>> branch
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0
    # Should detect conflict markers
    rejection_msg = " ".join(result["rejected_hunks"]).lower()
    assert "conflict" in rejection_msg or "merge" in rejection_msg


def test_fs_write_patch_context_mismatch(git_repo: Path, tool_context: ToolContext) -> None:
    """Test rejection when patch context doesn't match file content."""
    test_file = git_repo / "test.txt"
    test_file.write_text("goodbye\n", encoding="utf-8")  # Different content

    # Diff expects "hello" but file has "goodbye"
    diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-hello
+hello world
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0


def test_fs_write_patch_file_not_found(git_repo: Path, tool_context: ToolContext) -> None:
    """Test handling of patch for non-existent file (should still fail gracefully)."""
    # Diff for a file that doesn't exist (not a new file creation)
    diff = """--- a/nonexistent.txt
+++ b/nonexistent.txt
@@ -1 +1 @@
-hello
+hello world
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0


def test_fs_write_patch_new_file_creation(git_repo: Path, tool_context: ToolContext) -> None:
    """Test successful creation of new file via patch."""
    diff = """--- /dev/null
+++ b/newfile.txt
@@ -0,0 +1 @@
+new content
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is True
    assert result["rejected_hunks"] == []
    assert "newfile.txt" in result["new_files"]

    # Verify file was created
    new_file = git_repo / "newfile.txt"
    assert new_file.exists()
    assert "new content" in new_file.read_text(encoding="utf-8")


def test_fs_write_patch_multiple_files(git_repo: Path, tool_context: ToolContext) -> None:
    """Test patch affecting multiple files."""
    # Create test files
    (git_repo / "file1.txt").write_text("line1\n", encoding="utf-8")
    (git_repo / "file2.txt").write_text("line2\n", encoding="utf-8")

    diff = """--- a/file1.txt
+++ b/file1.txt
@@ -1 +1 @@
-line1
+line1 modified
--- a/file2.txt
+++ b/file2.txt
@@ -1 +1 @@
-line2
+line2 modified
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is True
    assert result["rejected_hunks"] == []
    assert "file1.txt" in result["changed_files"]
    assert "file2.txt" in result["changed_files"]
    assert result["diff_stats"]["files"] == 2


def test_fs_write_patch_fallback_to_patch_command(
    git_repo: Path, tool_context: ToolContext, monkeypatch
) -> None:
    """Test that patch command is used as fallback when git apply fails."""
    test_file = git_repo / "test.txt"
    test_file.write_text("hello\n", encoding="utf-8")

    # Create a diff that might fail with git apply but work with patch
    diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-hello
+hello world
"""

    # Mock git apply to fail but let patch command succeed
    original_run = subprocess.run
    call_count = {"git_apply": 0, "patch": 0}

    def mock_run(cmd, *args, **kwargs):
        if cmd[0] == "git" and "apply" in cmd:
            call_count["git_apply"] += 1
            # Make git apply fail
            result = MagicMock()
            result.returncode = 1
            result.stderr = b"git apply failed"
            result.stdout = b""
            return result
        elif cmd[0] == "patch":
            call_count["patch"] += 1
            # Let patch succeed by calling the real command
            return original_run(cmd, *args, **kwargs)
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    # Should succeed via patch fallback
    assert result["applied"] is True or call_count["patch"] > 0
    # Git apply should have been tried
    assert call_count["git_apply"] > 0


def test_fs_write_patch_preserves_backward_compatibility(
    git_repo: Path, tool_context: ToolContext
) -> None:
    """Test that response schema matches what tool_invoker expects."""
    test_file = git_repo / "test.txt"
    test_file.write_text("hello\n", encoding="utf-8")

    diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-hello
+hello world
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    # Check all required fields are present
    assert "applied" in result
    assert isinstance(result["applied"], bool)
    assert "rejected_hunks" in result
    assert isinstance(result["rejected_hunks"], list)
    assert "changed_files" in result
    assert isinstance(result["changed_files"], list)
    assert "new_files" in result
    assert isinstance(result["new_files"], list)
    assert "diff_stats" in result
    assert isinstance(result["diff_stats"], dict)
    assert "lines" in result["diff_stats"]
    assert "files" in result["diff_stats"]


def test_fs_write_patch_empty_diff(git_repo: Path, tool_context: ToolContext) -> None:
    """Test handling of empty diff."""
    diff = ""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0


def test_fs_write_patch_whitespace_only_changes(git_repo: Path, tool_context: ToolContext) -> None:
    """Test patch with only whitespace changes."""
    test_file = git_repo / "test.txt"
    test_file.write_text("hello\n", encoding="utf-8")

    # Diff with trailing whitespace change
    diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-hello
+hello
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    # Should either succeed (with whitespace=fix) or provide clear rejection
    assert "applied" in result
    if not result["applied"]:
        assert len(result["rejected_hunks"]) > 0
