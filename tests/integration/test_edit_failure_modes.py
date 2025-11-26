"""Integration tests for the SEARCH/REPLACE EDIT tool."""

from unittest.mock import Mock

import pytest

from ai_dev_agent.tools.filesystem.search_replace import _fs_edit
from ai_dev_agent.tools.registry import ToolContext


def make_search_replace(path: str, search: str, replace: str) -> str:
    """Create a SEARCH/REPLACE block."""
    return f"""{path}
```
<<<<<<< SEARCH
{search}=======
{replace}>>>>>>> REPLACE
```"""


@pytest.fixture
def tool_context(tmp_path):
    """Create a mock ToolContext for testing."""
    return ToolContext(
        repo_root=tmp_path, settings=Mock(), sandbox=Mock(), metrics_collector=Mock()
    )


class TestSearchReplaceSuccess:
    """Happy-path tests for the EDIT tool with SEARCH/REPLACE format."""

    def test_update_existing_file(self, tool_context, tmp_path):
        target = tmp_path / "app.py"
        target.write_text(
            "def greet(name):\n" '    return f"Hello, {name}"\n',
            encoding="utf-8",
        )

        patch = make_search_replace(
            "app.py",
            'def greet(name):\n    return f"Hello, {name}"\n',
            'def greet(name: str) -> str:\n    return f"Hi, {name}!"\n',
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert target.read_text(encoding="utf-8") == (
            "def greet(name: str) -> str:\n" '    return f"Hi, {name}!"\n'
        )

    def test_add_new_file_with_empty_search(self, tool_context, tmp_path):
        """Empty SEARCH creates new file."""
        patch = make_search_replace(
            "src/new_module.py",
            "",  # Empty SEARCH = create/append
            'def added():\n    return "ok"\n',
        )

        result = _fs_edit({"patch": patch}, tool_context)

        new_file = tmp_path / "src" / "new_module.py"
        assert result["success"]
        assert new_file.exists()
        assert 'return "ok"' in new_file.read_text()

    def test_delete_content_with_empty_replace(self, tool_context, tmp_path):
        """Empty REPLACE deletes the matched content."""
        target = tmp_path / "file.py"
        target.write_text("line1\ndelete_me\nline3\n", encoding="utf-8")

        patch = make_search_replace(
            "file.py",
            "delete_me\n",
            "",  # Empty REPLACE = delete
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        # The deleted line is replaced with empty, leaving an empty line
        assert target.read_text(encoding="utf-8") == "line1\n\nline3\n"


class TestSearchReplaceFailures:
    """Failure scenarios with actionable diagnostics."""

    def test_missing_patch_parameter(self, tool_context):
        result = _fs_edit({}, tool_context)

        assert not result["success"]
        assert any("patch" in err.lower() for err in result["errors"])

    def test_invalid_patch_structure(self, tool_context):
        result = _fs_edit({"patch": "not a patch"}, tool_context)

        assert not result["success"]
        # Should fail to find any SEARCH/REPLACE blocks
        assert len(result["errors"]) > 0

    def test_search_content_not_found(self, tool_context, tmp_path):
        target = tmp_path / "service.py"
        target.write_text("def run():\n    return 1\n", encoding="utf-8")

        # Search for content that doesn't exist
        patch = make_search_replace(
            "service.py",
            "def run():\n    return 2\n",  # Wrong - file has return 1
            "def run():\n    return 3\n",
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert any(
            "not found" in err.lower() or "search" in err.lower() for err in result["errors"]
        )
        # File should be unchanged
        assert target.read_text(encoding="utf-8") == "def run():\n    return 1\n"


class TestDiagnostics:
    """Ensure the tool provides actionable diagnostics back to the agent."""

    def test_provides_helpful_error_on_mismatch(self, tool_context, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")

        patch = make_search_replace(
            "file.py",
            "VALUE = 2\n",  # Doesn't match
            "VALUE = 3\n",
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        # Should suggest using empty SEARCH for insertions
        error_text = " ".join(result["errors"])
        assert "SEARCH" in error_text or "not found" in error_text.lower()

    def test_anchor_not_found_shows_actual_content(self, tool_context, tmp_path):
        target = tmp_path / "doc.md"
        target.write_text("# Title\n\n## Existing\nBody\n", encoding="utf-8")

        patch = make_search_replace(
            "doc.md",
            "@@BEFORE: ## Missing",
            "\n## Patch Workflow\nContent\n",
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        error_text = " ".join(result["errors"])
        assert "anchor" in error_text.lower() or "missing" in error_text.lower()
        assert "Actual file content" in error_text
