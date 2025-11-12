"""Tests for file existence validation in patch application."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.tools.code.code_edit.diff_utils import DiffError, DiffProcessor
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


def test_create_file_that_already_exists(git_repo: Path, tool_context: ToolContext) -> None:
    """Test that creating a file that already exists gives a helpful error."""
    # Create an existing file
    existing_file = git_repo / "existing.txt"
    existing_file.write_text("original content\n", encoding="utf-8")

    # Try to create it as a new file
    diff = """--- /dev/null
+++ b/existing.txt
@@ -0,0 +1 @@
+new content
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0
    error_msg = result["rejected_hunks"][0]

    # Check for helpful error message
    assert "File already exists" in error_msg
    assert "--- a/existing.txt" in error_msg  # Shows correct format
    assert "+++ b/existing.txt" in error_msg
    assert "--- /dev/null" in error_msg  # Explains the problem


def test_modify_file_that_does_not_exist(git_repo: Path, tool_context: ToolContext) -> None:
    """Test that modifying a non-existent file gives a helpful error."""
    # Try to modify a file that doesn't exist
    diff = """--- a/nonexistent.txt
+++ b/nonexistent.txt
@@ -1,3 +1,3 @@
 line1
-line2
+modified_line2
 line3
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0
    error_msg = result["rejected_hunks"][0]

    # Check for helpful error message
    assert "File does not exist" in error_msg
    assert "--- /dev/null" in error_msg  # Shows how to create new file
    assert "+++ b/nonexistent.txt" in error_msg


def test_create_new_file_success(git_repo: Path, tool_context: ToolContext) -> None:
    """Test that creating a new file works when it doesn't exist."""
    diff = """--- /dev/null
+++ b/newfile.txt
@@ -0,0 +1,3 @@
+line1
+line2
+line3
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is True
    assert result["rejected_hunks"] == []
    assert "newfile.txt" in result["new_files"]

    # Verify file was created with correct content
    new_file = git_repo / "newfile.txt"
    assert new_file.exists()
    assert new_file.read_text(encoding="utf-8") == "line1\nline2\nline3\n"


def test_modify_existing_file_success(git_repo: Path, tool_context: ToolContext) -> None:
    """Test that modifying an existing file works correctly."""
    # Create a file to modify
    test_file = git_repo / "modify.txt"
    test_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    # Modify it
    diff = """--- a/modify.txt
+++ b/modify.txt
@@ -1,3 +1,3 @@
 line1
-line2
+modified_line2
 line3
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is True
    assert result["rejected_hunks"] == []
    assert "modify.txt" in result["changed_files"]

    # Verify modification
    assert test_file.read_text(encoding="utf-8") == "line1\nmodified_line2\nline3\n"


def test_delete_file_with_dev_null(git_repo: Path, tool_context: ToolContext) -> None:
    """Test that deleting a file using /dev/null works."""
    # Create a file to delete
    test_file = git_repo / "delete_me.txt"
    test_file.write_text("content to delete\n", encoding="utf-8")

    # Delete it
    diff = """--- a/delete_me.txt
+++ /dev/null
@@ -1 +0,0 @@
-content to delete
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is True
    assert result["rejected_hunks"] == []

    # Verify file was deleted
    assert not test_file.exists()


def test_error_message_includes_guidance(git_repo: Path) -> None:
    """Test that error messages include actionable guidance."""
    processor = DiffProcessor(git_repo)

    # Create an existing file
    existing = git_repo / "test.py"
    existing.write_text("print('hello')\n", encoding="utf-8")

    # Try to create it as new
    diff = """--- /dev/null
+++ b/test.py
@@ -0,0 +1 @@
+print('world')
"""

    with pytest.raises(DiffError) as exc_info:
        processor.apply_diff_safely(diff)

    error_msg = str(exc_info.value)

    # Check that error message includes:
    # 1. Clear problem statement
    assert "Cannot create new file" in error_msg
    assert "File already exists" in error_msg

    # 2. Explanation
    assert "This patch is trying to create a new file" in error_msg

    # 3. Solution
    assert "To modify an existing file, use:" in error_msg
    assert "--- a/test.py" in error_msg

    # 4. Alternative
    assert "delete the existing file first" in error_msg


def test_multiple_files_with_mixed_operations(git_repo: Path, tool_context: ToolContext) -> None:
    """Test a patch with multiple files including creates and modifies."""
    # Create one file that will be modified
    existing = git_repo / "existing.txt"
    existing.write_text("original\n", encoding="utf-8")

    # Multi-file patch
    diff = """--- a/existing.txt
+++ b/existing.txt
@@ -1 +1 @@
-original
+modified
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+new file content
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is True
    assert "existing.txt" in result["changed_files"]
    assert "new.txt" in result["new_files"]

    # Verify both operations succeeded
    assert existing.read_text(encoding="utf-8") == "modified\n"
    assert (git_repo / "new.txt").read_text(encoding="utf-8") == "new file content\n"


def test_subdirectory_file_creation(git_repo: Path, tool_context: ToolContext) -> None:
    """Test creating a file in a subdirectory."""
    # Create subdirectory
    subdir = git_repo / "src" / "utils"
    subdir.mkdir(parents=True, exist_ok=True)

    # Create file in subdirectory
    diff = """--- /dev/null
+++ b/src/utils/helper.py
@@ -0,0 +1,2 @@
+def helper():
+    return "help"
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is True
    assert "src/utils/helper.py" in result["new_files"]

    helper_file = git_repo / "src" / "utils" / "helper.py"
    assert helper_file.exists()
    assert "def helper():" in helper_file.read_text(encoding="utf-8")
