"""Tests for SEARCH/REPLACE format file editing tool."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.tools.filesystem.search_replace import (
    SearchReplaceBlock,
    SearchReplaceMatcher,
    _fs_edit,
    apply_replacements,
    parse_search_replace_blocks,
)
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


def test_parse_search_replace_blocks_aider_style() -> None:
    """Test parsing SEARCH/REPLACE blocks in Aider style."""
    changes = """<<<<<<< SEARCH
def old_function():
    return "old"
=======
def new_function():
    return "new"
>>>>>>> REPLACE"""

    blocks = parse_search_replace_blocks(changes)

    assert len(blocks) == 1
    assert blocks[0].search == 'def old_function():\n    return "old"'
    assert blocks[0].replace == 'def new_function():\n    return "new"'


def test_parse_search_replace_blocks_cline_style() -> None:
    """Test parsing SEARCH/REPLACE blocks in Cline style."""
    changes = """------- SEARCH
import os
=======
import os
import sys
+++++++ REPLACE"""

    blocks = parse_search_replace_blocks(changes)

    assert len(blocks) == 1
    assert blocks[0].search == "import os"
    assert blocks[0].replace == "import os\nimport sys"


def test_parse_multiple_blocks() -> None:
    """Test parsing multiple SEARCH/REPLACE blocks."""
    changes = """<<<<<<< SEARCH
first_old
=======
first_new
>>>>>>> REPLACE

<<<<<<< SEARCH
second_old
=======
second_new
>>>>>>> REPLACE"""

    blocks = parse_search_replace_blocks(changes)

    assert len(blocks) == 2
    assert blocks[0].search == "first_old"
    assert blocks[0].replace == "first_new"
    assert blocks[1].search == "second_old"
    assert blocks[1].replace == "second_new"


def test_exact_match() -> None:
    """Test exact string matching."""
    content = """def function1():
    return 1

def function2():
    return 2"""

    matcher = SearchReplaceMatcher(content)
    match = matcher.find_match("def function2():\n    return 2")

    assert match is not None
    start, end = match
    assert content[start:end] == "def function2():\n    return 2"


def test_line_trimmed_match() -> None:
    """Test matching with whitespace differences."""
    content = """def function():
    if True:
        return "value"
    return None"""

    matcher = SearchReplaceMatcher(content)

    # Search with different indentation
    search = """def function():
if True:
    return "value"
return None"""

    match = matcher.find_match(search)

    assert match is not None  # Should find via line-trimmed matching
    start, end = match
    assert len(content[start:end]) == len(content)  # Matched entire content


def test_anchor_match() -> None:
    """Test matching with first/last line anchors."""
    content = """def complex_function():
    # Some comment
    x = 1
    y = 2
    # Another comment
    return x + y"""

    matcher = SearchReplaceMatcher(content)

    # Search with middle content slightly different
    search = """def complex_function():
    # Different comment
    x = 1
    y = 2
    # Yet another comment
    return x + y"""

    match = matcher.find_match(search)

    # With current implementation, this should not match
    # (anchors must have same content between them)
    assert match is None


def test_apply_single_replacement(git_repo: Path) -> None:
    """Test applying a single SEARCH/REPLACE block."""
    test_file = git_repo / "test.py"
    test_file.write_text(
        """def old_function():
    return "old"

def keep_this():
    return "keep"
""",
        encoding="utf-8",
    )

    blocks = [
        SearchReplaceBlock(
            search='def old_function():\n    return "old"',
            replace='def new_function():\n    return "new"',
        )
    ]

    new_content, applied, errors = apply_replacements(test_file, blocks)

    assert len(applied) == 1
    assert len(errors) == 0
    assert "def new_function():" in new_content
    assert 'return "new"' in new_content
    assert "def keep_this():" in new_content


def test_apply_multiple_replacements(git_repo: Path) -> None:
    """Test applying multiple SEARCH/REPLACE blocks."""
    test_file = git_repo / "test.py"
    test_file.write_text(
        """import os

def function1():
    return 1

def function2():
    return 2
""",
        encoding="utf-8",
    )

    blocks = [
        SearchReplaceBlock(search="import os", replace="import os\nimport sys"),
        SearchReplaceBlock(search="return 1", replace="return 10"),
        SearchReplaceBlock(search="return 2", replace="return 20"),
    ]

    new_content, applied, errors = apply_replacements(test_file, blocks)

    assert len(applied) == 3
    assert len(errors) == 0
    assert "import sys" in new_content
    assert "return 10" in new_content
    assert "return 20" in new_content


def test_search_not_found_error(git_repo: Path) -> None:
    """Test error when SEARCH text is not found."""
    test_file = git_repo / "test.py"
    test_file.write_text("actual content", encoding="utf-8")

    blocks = [SearchReplaceBlock(search="nonexistent", replace="replacement")]

    new_content, applied, errors = apply_replacements(test_file, blocks)

    assert len(applied) == 0
    assert len(errors) == 1
    assert "not found" in errors[0]


def test_empty_search_block_full_replacement(git_repo: Path) -> None:
    """Test empty SEARCH block for full file replacement."""
    test_file = git_repo / "test.py"
    test_file.write_text("old content", encoding="utf-8")

    blocks = [SearchReplaceBlock(search="", replace="brand new content")]

    new_content, applied, errors = apply_replacements(test_file, blocks)

    assert new_content == "brand new content"
    assert len(applied) == 1
    assert "Replaced entire file" in applied[0]


def test_empty_replace_block_deletion(git_repo: Path) -> None:
    """Test empty REPLACE block for deletion."""
    test_file = git_repo / "test.py"
    test_file.write_text(
        """line1
delete_this
line3""",
        encoding="utf-8",
    )

    blocks = [SearchReplaceBlock(search="delete_this\n", replace="")]

    new_content, applied, errors = apply_replacements(test_file, blocks)

    assert "delete_this" not in new_content
    assert "line1\nline3" in new_content


def test_fs_edit_success(git_repo: Path, tool_context: ToolContext) -> None:
    """Test successful file editing via _fs_edit handler."""
    test_file = git_repo / "test.py"
    test_file.write_text("old_content", encoding="utf-8")

    payload = {
        "path": "test.py",
        "changes": """<<<<<<< SEARCH
old_content
=======
new_content
>>>>>>> REPLACE""",
    }

    result = _fs_edit(payload, tool_context)

    assert result["success"] is True
    assert result["changes_applied"] == 1
    assert len(result["errors"]) == 0

    # Verify file was actually modified
    assert test_file.read_text(encoding="utf-8") == "new_content"


def test_fs_edit_file_not_found(tool_context: ToolContext) -> None:
    """Test that edit tool creates new files when they don't exist."""
    new_file = Path(tool_context.repo_root) / "new_file.py"

    # Ensure file doesn't exist
    if new_file.exists():
        new_file.unlink()

    payload = {
        "path": "new_file.py",
        "changes": """<<<<<<< SEARCH
=======
def hello():
    return "Hello, World!"
>>>>>>> REPLACE""",
    }

    result = _fs_edit(payload, tool_context)

    # Should succeed by creating the file
    assert result["success"] is True
    assert result["changes_applied"] == 1
    assert new_file.exists()
    assert "hello" in new_file.read_text()

    # Clean up
    new_file.unlink()


def test_fs_edit_invalid_format(git_repo: Path, tool_context: ToolContext) -> None:
    """Test error with invalid SEARCH/REPLACE format."""
    test_file = git_repo / "test.py"
    test_file.write_text("content", encoding="utf-8")

    payload = {
        "path": "test.py",
        "changes": """<<<<<<< SEARCH
something
=======
Missing REPLACE marker""",
    }

    result = _fs_edit(payload, tool_context)

    assert result["success"] is False
    assert "Invalid SEARCH/REPLACE format" in result["errors"][0]


def test_fs_edit_partial_success(git_repo: Path, tool_context: ToolContext) -> None:
    """Test partial success with multiple blocks."""
    test_file = git_repo / "test.py"
    test_file.write_text(
        """line1
line2
line3""",
        encoding="utf-8",
    )

    payload = {
        "path": "test.py",
        "changes": """<<<<<<< SEARCH
line1
=======
modified1
>>>>>>> REPLACE

<<<<<<< SEARCH
nonexistent
=======
replacement
>>>>>>> REPLACE

<<<<<<< SEARCH
line3
=======
modified3
>>>>>>> REPLACE""",
    }

    result = _fs_edit(payload, tool_context)

    assert result["success"] is False  # Not all blocks succeeded
    assert result["changes_applied"] == 2  # 2 out of 3 succeeded
    assert len(result["errors"]) == 1
    assert len(result["warnings"]) == 1
    assert "2/3 blocks succeeded" in result["warnings"][0]


def test_out_of_order_replacements(git_repo: Path) -> None:
    """Test replacements that appear out of order in file."""
    test_file = git_repo / "test.py"
    test_file.write_text(
        """line1
line2
line3
line4
line5""",
        encoding="utf-8",
    )

    # Search for line3 first, then line1 (out of order)
    blocks = [
        SearchReplaceBlock(search="line3", replace="modified3"),
        SearchReplaceBlock(search="line1", replace="modified1"),
    ]

    new_content, applied, errors = apply_replacements(test_file, blocks)

    assert len(applied) == 2
    assert len(errors) == 0
    assert "modified1" in new_content
    assert "modified3" in new_content
