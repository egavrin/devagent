"""Test improvements to patch write tool error messages and validation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_improved_error_message_corrupt_patch(git_repo: Path, tool_context: ToolContext) -> None:
    """Test improved error message for corrupt patch."""
    test_file = git_repo / "test.txt"
    test_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    # Create a malformed diff
    diff = """--- a/test.txt
+++ b/test.txt
@@ -2,2 +2,2 @@
-line2
+modified_line2
"""  # Missing context lines

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0
    error_msg = result["rejected_hunks"][0]

    # Check for clear, actionable error message
    assert "Malformed patch format" in error_msg or "patch" in error_msg.lower()


def test_improved_error_message_file_not_found(git_repo: Path, tool_context: ToolContext) -> None:
    """Test improved error message when file doesn't exist."""
    # Try to patch a file that doesn't exist
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

    # Check for specific file not found message
    assert "does not exist" in error_msg or "not found" in error_msg.lower()
    assert "nonexistent.txt" in error_msg or "specified file" in error_msg


def test_improved_error_message_context_mismatch(git_repo: Path, tool_context: ToolContext) -> None:
    """Test improved error message when context doesn't match."""
    test_file = git_repo / "test.txt"
    test_file.write_text("actual_line1\nactual_line2\nactual_line3\n", encoding="utf-8")

    # Create a diff with wrong context
    diff = """--- a/test.txt
+++ b/test.txt
@@ -1,3 +1,3 @@
 wrong_line1
-wrong_line2
+modified_line2
 wrong_line3
"""

    payload = {"diff": diff}
    result = _fs_write_patch(payload, tool_context)

    assert result["applied"] is False
    assert len(result["rejected_hunks"]) > 0
    error_msg = result["rejected_hunks"][0]

    # Check for context mismatch message
    assert (
        ("context" in error_msg.lower() and "match" in error_msg.lower())
        or "does not apply" in error_msg.lower()
        or "Patch context doesn't match" in error_msg
    )


def test_validation_insufficient_context_warning(git_repo: Path) -> None:
    """Test validation warns about insufficient context."""
    processor = DiffProcessor(git_repo)

    # Diff with minimal context (only 1 line)
    diff = """--- a/test.txt
+++ b/test.txt
@@ -1,1 +1,1 @@
-old_line
+new_line
"""

    validation = processor._validate_diff(diff)

    # Should have warning about insufficient context
    assert any("context" in warning.lower() for warning in validation.warnings)


def test_validation_missing_file_headers(git_repo: Path) -> None:
    """Test validation detects missing file headers."""
    processor = DiffProcessor(git_repo)

    # Diff without proper file headers
    diff = """@@ -1,3 +1,3 @@
 line1
-line2
+modified_line2
 line3
"""

    validation = processor._validate_diff(diff)

    assert not validation.is_valid
    assert any("No files found" in error or "headers" in error for error in validation.errors)


def test_no_redundant_error_wrapping(git_repo: Path) -> None:
    """Test that DiffError isn't wrapped multiple times."""
    processor = DiffProcessor(git_repo)

    # Force a DiffError by creating a file that git apply will reject
    test_file = git_repo / "test.txt"
    test_file.write_text("line1\n", encoding="utf-8")

    diff = """--- a/test.txt
+++ b/test.txt
@@ corrupt header @@
-line1
+modified
"""

    with pytest.raises(DiffError) as exc_info:
        processor.apply_diff_safely(diff)

    # Error message should be clear and not have multiple layers
    error_msg = str(exc_info.value)

    # Should not have "Unexpected error applying diff: Failed to apply diff:"
    assert "Unexpected error applying diff: Failed to apply diff:" not in error_msg
    # Should have a clear message
    assert (
        "Invalid hunk header" in error_msg
        or "Malformed patch format" in error_msg
        or "Patch application failed" in error_msg
        or "Diff validation failed" in error_msg
    )


def test_actionable_error_messages(git_repo: Path, tool_context: ToolContext) -> None:
    """Test that error messages provide actionable guidance."""
    test_file = git_repo / "test.txt"
    test_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    # Test various error conditions
    test_cases = [
        # Missing --- header
        (
            """@@ -1,3 +1,3 @@
 line1
-line2
+modified_line2
 line3
""",
            ["headers", "---", "+++"],
        ),
        # Corrupt hunk header
        (
            """--- a/test.txt
+++ b/test.txt
@@ invalid @@
 line1
-line2
+modified_line2
 line3
""",
            ["Invalid hunk header", "hunk", "format"],
        ),
        # File doesn't exist
        (
            """--- a/missing.txt
+++ b/missing.txt
@@ -1,1 +1,1 @@
-old
+new
""",
            ["does not exist", "missing.txt"],
        ),
    ]

    for diff, expected_keywords in test_cases:
        payload = {"diff": diff}
        result = _fs_write_patch(payload, tool_context)

        assert result["applied"] is False
        error_msg = " ".join(result["rejected_hunks"]).lower()

        # Check that at least one expected keyword is present
        assert any(
            keyword.lower() in error_msg for keyword in expected_keywords
        ), f"Expected one of {expected_keywords} in error: {error_msg}"
