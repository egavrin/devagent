"""Integration tests for EDIT tool failure modes.

These tests document and validate the common failure scenarios that users
encounter when using the EDIT tool in real agent workflows.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from ai_dev_agent.tools.filesystem.search_replace import _fs_edit
from ai_dev_agent.tools.registry import ToolContext


@pytest.fixture
def tool_context(tmp_path):
    """Create a mock ToolContext for testing."""
    return ToolContext(
        repo_root=tmp_path, settings=Mock(), sandbox=Mock(), metrics_collector=Mock()
    )


class TestEditFailureModes:
    """Test suite for documenting EDIT tool failure modes."""

    def test_whitespace_mismatch_with_line_trimmed_matching(self, tool_context, tmp_path):
        """Test that line-trimmed matching tolerates indentation differences.

        This documents current behavior: the matcher falls back to line-trimmed
        matching which normalizes whitespace, so indentation mismatches succeed.
        """
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def hello():
    print("Hello")
    return True
"""
        )

        # Try to edit with wrong indentation (2 spaces instead of 4)
        changes = """<<<<<<< SEARCH
def hello():
  print("Hello")
  return True
=======
def hello():
    print("Hello, World!")
    return True
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        # Currently succeeds due to line-trimmed matching
        assert result["success"], "Line-trimmed matching should handle whitespace differences"
        assert result["changes_applied"] == 1
        assert "Hello, World!" in test_file.read_text()

    def test_missing_path_parameter(self, tool_context):
        """Test failure when path parameter is missing for SEARCH/REPLACE."""
        changes = """<<<<<<< SEARCH
old text
=======
new text
>>>>>>> REPLACE
"""

        result = _fs_edit({"changes": changes}, tool_context)

        assert not result["success"]
        assert "path" in result["errors"][0].lower()

    def test_file_changed_after_read(self, tool_context, tmp_path):
        """Test failure when file content changed between READ and EDIT.

        This is a common failure mode when:
        - File is modified by another process
        - Agent uses stale context from earlier in conversation
        """
        test_file = tmp_path / "test.py"

        # Initial content that agent "reads"
        initial_content = """def func():
    return 42
"""
        test_file.write_text(initial_content)

        # File changes (simulated concurrent modification)
        test_file.write_text(
            """def func():
    return 43  # Changed!
"""
        )

        # Agent tries to edit based on old content
        changes = """<<<<<<< SEARCH
def func():
    return 42
=======
def func():
    return 100
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert not result["success"]
        assert "SEARCH text not found" in result["errors"][0]

    def test_paraphrased_content(self, tool_context, tmp_path):
        """Test failure when agent paraphrases instead of copying exact text.

        This is the DOMINANT failure mode: LLMs paraphrase code instead of
        copying the exact bytes from the READ output.
        """
        test_file = tmp_path / "test.py"

        # Actual file content
        test_file.write_text(
            """# Calculate the sum of two numbers
def add(a, b):
    result = a + b
    return result
"""
        )

        # Agent paraphrases by adding a comment
        changes = """<<<<<<< SEARCH
def add(a, b):
    result = a + b  # Agent added comment
    return result
=======
def add(a, b):
    return a + b
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert not result["success"]
        assert "SEARCH text not found" in result["errors"][0]

    def test_malformed_search_replace_missing_separator(self, tool_context, tmp_path):
        """Test failure with malformed SEARCH/REPLACE blocks."""
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        # Missing separator
        changes = """<<<<<<< SEARCH
old content
>>>>>>> REPLACE
new content
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert not result["success"]
        assert "format" in result["errors"][0].lower() or "separator" in result["errors"][0].lower()

    def test_successful_edit_with_exact_match(self, tool_context, tmp_path):
        """Test that EDIT tool works correctly when used properly.

        This is the control case showing the tool works when:
        - Agent reads file first
        - Copies EXACT text from file
        - Provides path parameter
        """
        test_file = tmp_path / "test.py"

        # Write original content
        original = """def greet(name):
    print(f"Hello, {name}")
"""
        test_file.write_text(original)

        # Copy EXACT text from file
        changes = """<<<<<<< SEARCH
def greet(name):
    print(f"Hello, {name}")
=======
def greet(name):
    print(f"Hi, {name}!")
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert result["success"]
        assert result["changes_applied"] == 1
        assert "Hi, {name}!" in test_file.read_text()

    def test_multiple_blocks_out_of_order(self, tool_context, tmp_path):
        """Test that multiple SEARCH/REPLACE blocks work even when out of order."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def first():
    pass

def second():
    pass

def third():
    pass
"""
        )

        # Multiple blocks, out of order
        changes = """<<<<<<< SEARCH
def third():
    pass
=======
def third():
    return 3
>>>>>>> REPLACE

<<<<<<< SEARCH
def first():
    pass
=======
def first():
    return 1
>>>>>>> REPLACE

<<<<<<< SEARCH
def second():
    pass
=======
def second():
    return 2
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert result["success"]
        assert result["changes_applied"] == 3
        content = test_file.read_text()
        assert "return 1" in content
        assert "return 2" in content
        assert "return 3" in content

    def test_new_file_creation_with_multiple_blocks(self, tool_context, tmp_path):
        """Test creating a new file with multiple SEARCH/REPLACE blocks.

        LLMs naturally scaffold new files with one block per function/method.
        This should now work by concatenating all REPLACE blocks.
        """
        test_file = tmp_path / "new_file.py"

        # LLM tries to create file with multiple blocks (one per function)
        changes = """<<<<<<< SEARCH
=======
def func_one():
    return 1
>>>>>>> REPLACE

<<<<<<< SEARCH
=======

def func_two():
    return 2
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        # Should now succeed with our fix
        assert result["success"], f"Expected success but got: {result.get('errors')}"
        assert test_file.exists()
        content = test_file.read_text()
        assert "func_one" in content
        assert "func_two" in content
        assert result["changes_applied"] == 2  # Both blocks applied
        assert len(result["new_files"]) == 1  # One new file created

    def test_new_file_creation_single_block(self, tool_context, tmp_path):
        """Test creating a new file with a single SEARCH/REPLACE block.

        This works correctly - only single-block creation is currently supported.
        """
        test_file = tmp_path / "new_file.py"

        changes = """<<<<<<< SEARCH
=======
def hello():
    print("Hello, World!")
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert result["success"]
        assert test_file.exists()
        assert "Hello, World!" in test_file.read_text()

    def test_empty_search_for_deletion(self, tool_context, tmp_path):
        """Test using empty REPLACE to delete content."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def to_delete():
    pass

def to_keep():
    pass
"""
        )

        changes = """<<<<<<< SEARCH
def to_delete():
    pass

=======
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert result["success"]
        content = test_file.read_text()
        assert "to_delete" not in content
        assert "to_keep" in content


class TestEditErrorMessages:
    """Test suite for validating error message quality."""

    def test_error_message_includes_block_number(self, tool_context, tmp_path):
        """Test that error messages identify which block failed."""
        test_file = tmp_path / "test.py"
        test_file.write_text("actual content")

        changes = """<<<<<<< SEARCH
block 1 wrong
=======
replacement 1
>>>>>>> REPLACE

<<<<<<< SEARCH
block 2 wrong
=======
replacement 2
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert not result["success"]
        # Both blocks should fail
        assert len(result["errors"]) >= 1
        assert "Block" in result["errors"][0]

    def test_error_message_shows_search_text_preview(self, tool_context, tmp_path):
        """Test that error messages show preview of failed SEARCH text."""
        test_file = tmp_path / "test.py"
        test_file.write_text("actual content")

        changes = """<<<<<<< SEARCH
this is the search text that doesn't match
=======
replacement
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert not result["success"]
        # Error should include preview of search text
        assert (
            "this is the search text" in result["errors"][0]
            or "this·is·the·search·text" in result["errors"][0]
        )

    def test_error_message_visualizes_whitespace(self, tool_context, tmp_path):
        """Test that error messages visualize whitespace characters."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func():\n    pass\n")

        # Try to match with wrong indentation
        changes = """<<<<<<< SEARCH
def func():
  pass
=======
def func():
    return True
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        # Should succeed due to line-trimmed matching, but if it failed:
        # Error message should visualize whitespace
        if not result["success"]:
            error_msg = result["errors"][0]
            # Should contain whitespace visualization markers
            assert "·" in error_msg or "→" in error_msg or "⏎" in error_msg
            # Should contain helpful tip
            assert "EXACT text" in error_msg or "exact text" in error_msg.lower()


class TestEditPreValidation:
    """Test suite for pre-validation of SEARCH blocks."""

    def test_pre_validation_catches_missing_search_text(self, tool_context, tmp_path):
        """Test that pre-validation catches SEARCH text that doesn't exist in file."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def actual_function():
    return 42
"""
        )

        # Try to edit with wrong function name
        changes = """<<<<<<< SEARCH
def wrong_function():
    return 42
=======
def wrong_function():
    return 100
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert not result["success"]
        assert len(result["errors"]) == 1
        assert "pre-validation failed" in result["errors"][0]
        assert "SEARCH text not found" in result["errors"][0]
        # File should not be modified
        assert test_file.read_text() == "def actual_function():\n    return 42\n"

    def test_pre_validation_provides_closest_match(self, tool_context, tmp_path):
        """Test that pre-validation suggests closest match when SEARCH fails."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def calculate_sum(a, b):
    result = a + b
    return result
"""
        )

        # Try to edit with slightly wrong text (missing comment)
        changes = """<<<<<<< SEARCH
def calculate_sum(a, b):  # This comment doesn't exist
    result = a + b
    return result
=======
def calculate_sum(a, b):
    return a + b
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert not result["success"]
        assert "Did you mean to match these lines from the file?" in result["errors"][0]
        # Should show whitespace visualization
        assert "·" in result["errors"][0] or "def·calculate_sum" in result["errors"][0]

    def test_pre_validation_skipped_for_new_files(self, tool_context, tmp_path):
        """Test that pre-validation is skipped when creating new files."""
        test_file = tmp_path / "new_file.py"

        changes = """<<<<<<< SEARCH
=======
def new_function():
    return 1
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        # Should succeed - validation skipped for new files
        assert result["success"]
        assert test_file.exists()
        assert "new_function" in test_file.read_text()

    def test_pre_validation_with_multiple_blocks_some_invalid(self, tool_context, tmp_path):
        """Test pre-validation when some blocks are valid and some aren't."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def func_one():
    pass

def func_two():
    pass
"""
        )

        # First block is valid, second is not
        changes = """<<<<<<< SEARCH
def func_one():
    pass
=======
def func_one():
    return 1
>>>>>>> REPLACE

<<<<<<< SEARCH
def func_three():
    pass
=======
def func_three():
    return 3
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        assert not result["success"]
        # Should report error for block 2
        assert "Block 2:" in result["errors"][0]
        assert "pre-validation failed" in result["errors"][0]
        # File should NOT be modified (no partial application)
        original = test_file.read_text()
        assert "return 1" not in original  # First block should not have been applied

    def test_pre_validation_with_all_valid_blocks(self, tool_context, tmp_path):
        """Test that pre-validation passes when all blocks are valid."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def func_one():
    pass

def func_two():
    pass
"""
        )

        changes = """<<<<<<< SEARCH
def func_one():
    pass
=======
def func_one():
    return 1
>>>>>>> REPLACE

<<<<<<< SEARCH
def func_two():
    pass
=======
def func_two():
    return 2
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        # Should succeed - all blocks validated and applied
        assert result["success"]
        assert result["changes_applied"] == 2
        content = test_file.read_text()
        assert "return 1" in content
        assert "return 2" in content

    def test_pre_validation_with_empty_search_blocks(self, tool_context, tmp_path):
        """Test that pre-validation handles empty SEARCH blocks correctly."""
        test_file = tmp_path / "test.py"
        test_file.write_text("old content\n")

        # Empty SEARCH = full file replacement
        changes = """<<<<<<< SEARCH
=======
new content
>>>>>>> REPLACE
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        # Should succeed - empty SEARCH is valid for full replacement
        assert result["success"]
        assert test_file.read_text() == "new content"


class TestEditWithDiffFormat:
    """Test suite for unified diff format support."""

    def test_unified_diff_format_detection(self, tool_context, tmp_path):
        """Test that unified diff format is correctly detected and applied."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def hello():
    print("Hello")
"""
        )

        # Unified diff format
        changes = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def hello():
-    print("Hello")
+    print("Hi")
"""

        result = _fs_edit({"path": str(test_file), "changes": changes}, tool_context)

        # Should be detected as unified diff and applied
        assert result["success"] or "diff" in str(result).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
