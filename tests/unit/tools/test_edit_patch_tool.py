"""Unit tests for the SEARCH/REPLACE EDIT tool implementation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.tools.filesystem.search_replace import (
    EditBlock,
    EditBlockApplier,
    ParseResult,
    PatchFormatError,
    SearchReplaceParser,
    _fs_edit,
)
from ai_dev_agent.tools.registry import ToolContext


def wrap_search_replace(path: str, search: str, replace: str, lang: str = "python") -> str:
    """Helper to create a properly formatted SEARCH/REPLACE block."""
    return f"""{path}
```{lang}
<<<<<<< SEARCH
{search}=======
{replace}>>>>>>> REPLACE
```"""


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README.md").write_text("# test repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.fixture
def tool_context(git_repo: Path) -> ToolContext:
    return ToolContext(
        repo_root=git_repo,
        settings=MagicMock(),
        sandbox=MagicMock(),
    )


class TestSearchReplaceParser:
    def test_parse_basic_replacement(self):
        text = """src/app.py
```python
<<<<<<< SEARCH
old_value = 1
=======
new_value = 2
>>>>>>> REPLACE
```"""
        parser = SearchReplaceParser(text)
        blocks = parser.parse()

        assert len(blocks) == 1
        assert blocks[0].path == "src/app.py"
        assert blocks[0].search == "old_value = 1"
        assert blocks[0].replace == "new_value = 2"

    def test_parse_insertion_empty_search(self):
        text = """README.md
```markdown
<<<<<<< SEARCH
=======
## New Section
Content here.
>>>>>>> REPLACE
```"""
        parser = SearchReplaceParser(text)
        blocks = parser.parse()

        assert len(blocks) == 1
        assert blocks[0].search == ""
        assert "## New Section" in blocks[0].replace

    def test_parse_deletion_empty_replace(self):
        text = """file.py
```python
<<<<<<< SEARCH
debug_code()
=======
>>>>>>> REPLACE
```"""
        parser = SearchReplaceParser(text)
        blocks = parser.parse()

        assert len(blocks) == 1
        assert blocks[0].search == "debug_code()"
        assert blocks[0].replace == ""

    def test_parse_multiple_blocks(self):
        text = """config.py
```python
<<<<<<< SEARCH
DEBUG = False
=======
DEBUG = True
>>>>>>> REPLACE
```

config.py
```python
<<<<<<< SEARCH
TIMEOUT = 30
=======
TIMEOUT = 60
>>>>>>> REPLACE
```"""
        parser = SearchReplaceParser(text)
        blocks = parser.parse()

        assert len(blocks) == 2
        assert blocks[0].search == "DEBUG = False"
        assert blocks[1].search == "TIMEOUT = 30"

    def test_parse_without_fences(self):
        """Parser should handle blocks without fence markers."""
        text = """file.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE"""
        parser = SearchReplaceParser(text)
        result = parser.parse_with_warnings()

        assert len(result.blocks) == 1
        assert result.blocks[0].search == "old"
        assert "without fence markers" in result.warnings[0]

    def test_parse_anchor_directive(self):
        text = """README.md
```markdown
<<<<<<< SEARCH
@@BEFORE: ## Target
=======
## Patch
>>>>>>> REPLACE
```"""
        parser = SearchReplaceParser(text)
        blocks = parser.parse()

        assert blocks[0].anchor_mode == "before"
        assert blocks[0].anchor == "## Target"
        assert blocks[0].search == ""

    def test_invalid_format_raises(self):
        parser = SearchReplaceParser("no markers here")
        with pytest.raises(PatchFormatError):
            parser.parse()

    def test_empty_input_raises(self):
        parser = SearchReplaceParser("")
        with pytest.raises(PatchFormatError):
            parser.parse()


class TestEditBlockApplier:
    def test_apply_replacement(self, git_repo: Path):
        target = git_repo / "test.py"
        target.write_text("value = 1\n", encoding="utf-8")

        applier = EditBlockApplier(git_repo)
        blocks = [EditBlock(path="test.py", search="value = 1", replace="value = 2")]
        result = applier.apply(blocks, dry_run=False)

        assert result["success"]
        assert "value = 2" in target.read_text()

    def test_apply_insertion_append(self, git_repo: Path):
        target = git_repo / "test.py"
        target.write_text("line1\n", encoding="utf-8")

        applier = EditBlockApplier(git_repo)
        blocks = [EditBlock(path="test.py", search="", replace="line2")]
        result = applier.apply(blocks, dry_run=False)

        assert result["success"]
        content = target.read_text()
        assert "line1" in content
        assert "line2" in content

    def test_insert_before_anchor_directive(self, git_repo: Path):
        target = git_repo / "doc.md"
        target.write_text("# Title\nIntro\n## Target\nBody\n", encoding="utf-8")

        applier = EditBlockApplier(git_repo)
        blocks = [
            EditBlock(
                path="doc.md",
                search="@@BEFORE: ## Target",
                replace="## Patch\nContent\n",
            )
        ]
        result = applier.apply(blocks, dry_run=False)

        assert result["success"]
        content = target.read_text(encoding="utf-8")
        assert "## Patch" in content
        assert content.index("## Patch") < content.index("## Target")

    def test_apply_deletion(self, git_repo: Path):
        target = git_repo / "test.py"
        target.write_text("keep\ndelete_me\n", encoding="utf-8")

        applier = EditBlockApplier(git_repo)
        blocks = [EditBlock(path="test.py", search="delete_me\n", replace="")]
        result = applier.apply(blocks, dry_run=False)

        assert result["success"]
        content = target.read_text()
        assert "keep" in content
        assert "delete_me" not in content

    def test_create_new_file(self, git_repo: Path):
        applier = EditBlockApplier(git_repo)
        blocks = [EditBlock(path="new_file.py", search="", replace="print('hello')")]
        result = applier.apply(blocks, dry_run=False)

        assert result["success"]
        new_file = git_repo / "new_file.py"
        assert new_file.exists()
        assert "print('hello')" in new_file.read_text()

    def test_missing_file_error(self, git_repo: Path):
        applier = EditBlockApplier(git_repo)
        blocks = [EditBlock(path="missing.py", search="old", replace="new")]
        result = applier.apply(blocks, dry_run=False)

        assert not result["success"]
        assert "missing" in result["errors"][0].lower()

    def test_search_not_found_error(self, git_repo: Path):
        target = git_repo / "test.py"
        target.write_text("actual_content\n", encoding="utf-8")

        applier = EditBlockApplier(git_repo)
        blocks = [EditBlock(path="test.py", search="wrong_content", replace="new")]
        result = applier.apply(blocks, dry_run=False)

        assert not result["success"]
        assert "not found" in result["errors"][0].lower()

    def test_dry_run_does_not_modify(self, git_repo: Path):
        target = git_repo / "test.py"
        target.write_text("original\n", encoding="utf-8")

        applier = EditBlockApplier(git_repo)
        blocks = [EditBlock(path="test.py", search="original", replace="modified")]
        result = applier.apply(blocks, dry_run=True)

        assert result["success"]
        assert target.read_text() == "original\n"  # Unchanged


class TestFsEdit:
    def test_apply_search_replace_success(self, tool_context: ToolContext, git_repo: Path):
        target = git_repo / "src" / "svc.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def fn():\n    return 1\n", encoding="utf-8")

        patch = """src/svc.py
```python
<<<<<<< SEARCH
def fn():
    return 1
=======
def fn():
    return 2
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert target.read_text(encoding="utf-8") == "def fn():\n    return 2\n"

    def test_missing_patch_parameter(self, tool_context: ToolContext):
        result = _fs_edit({}, tool_context)
        assert not result["success"]
        assert "patch" in (result["errors"][0].lower())

    def test_patch_failure_reports_errors(self, tool_context: ToolContext):
        patch = """src/missing.py
```python
<<<<<<< SEARCH
old_content
=======
new_content
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)
        assert not result["success"]
        assert "missing" in result["errors"][0].lower()

    def test_search_mismatch_reported_without_writing(
        self, tool_context: ToolContext, git_repo: Path
    ):
        target = git_repo / "module.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")

        patch = """module.py
```python
<<<<<<< SEARCH
VALUE = 2
=======
VALUE = 3
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert any("not found" in err.lower() for err in result["errors"])
        assert target.read_text(encoding="utf-8") == "VALUE = 1\n"

    def test_insertion_with_empty_search(self, tool_context: ToolContext, git_repo: Path):
        """Empty SEARCH section appends content at end of file."""
        target = git_repo / "README.md"
        target.write_text("# Title\n\nExisting content.\n", encoding="utf-8")

        patch = """README.md
```markdown
<<<<<<< SEARCH
=======

## New Section
Appended at end.
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        content = target.read_text(encoding="utf-8")
        assert "# Title" in content
        assert "Existing content." in content
        assert "## New Section" in content
        assert "Appended at end." in content
        # Preview should describe the append operation
        preview = result.get("preview") or []
        assert preview
        assert preview[0]["operation"] in {"append", "create"}
        assert preview[0]["path"] == "README.md"

    def test_preview_only_does_not_write(self, tool_context: ToolContext, git_repo: Path):
        target = git_repo / "file.py"
        target.write_text("value = 1\n", encoding="utf-8")

        patch = """file.py
```python
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch, "preview_only": True}, tool_context)

        assert result["success"]
        assert result.get("preview")
        # Under preview-only mode, file should remain unchanged
        assert target.read_text(encoding="utf-8") == "value = 1\n"

    def test_anchor_insertion_via_patch(self, tool_context: ToolContext, git_repo: Path):
        target = git_repo / "README.md"
        target.write_text("# Title\n\n## Existing\nText\n", encoding="utf-8")

        patch = """README.md
```markdown
<<<<<<< SEARCH
@@AFTER: # Title
=======

## Patch Workflow
Content here.
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        content = target.read_text(encoding="utf-8")
        assert "## Patch Workflow" in content
        assert content.index("## Patch Workflow") < content.index("## Existing")
        preview = result.get("preview") or []
        assert preview and preview[0]["operation"] == "insert_after"
        assert preview[0]["anchor"] == "# Title"


class TestFuzzyMatching:
    """Tests for layered fuzzy matching and whitespace tolerance."""

    def test_trailing_whitespace_tolerance(self, tool_context: ToolContext, git_repo: Path):
        """Trailing whitespace in file doesn't break match."""
        target = git_repo / "module.py"
        target.write_text("value = 1   \n", encoding="utf-8")

        patch = """module.py
```python
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert "value = 2" in target.read_text(encoding="utf-8")

    def test_trailing_whitespace_tolerance_multiline(
        self, tool_context: ToolContext, git_repo: Path
    ):
        """Multiline content with trailing whitespace differences."""
        target = git_repo / "module.py"
        target.write_text("def fn():   \n    return 1   \n", encoding="utf-8")

        patch = """module.py
```python
<<<<<<< SEARCH
def fn():
    return 1
=======
def fn():
    return 2
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert "return 2" in target.read_text(encoding="utf-8")
        assert any("stripping" in w.lower() for w in result.get("warnings", []))

    def test_leading_whitespace_tolerance(self, tool_context: ToolContext, git_repo: Path):
        """Uniform indentation differences are tolerated."""
        target = git_repo / "module.py"
        target.write_text("    def fn():\n        return 1\n", encoding="utf-8")

        patch = """module.py
```python
<<<<<<< SEARCH
  def fn():
      return 1
=======
  def fn():
      return 2
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert "return 2" in target.read_text(encoding="utf-8")
        assert any("indent" in w.lower() for w in result.get("warnings", []))

    def test_exact_match_preferred_over_fuzzy(self, tool_context: ToolContext, git_repo: Path):
        """When exact match exists, no fuzzy match warning is generated."""
        target = git_repo / "module.py"
        target.write_text("value = 1\n", encoding="utf-8")

        patch = """module.py
```python
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        warnings = result.get("warnings", [])
        assert not any("whitespace" in w.lower() for w in warnings)
        assert not any("indent" in w.lower() for w in warnings)


class TestSimilarLineSuggestions:
    """Tests for error diagnostics with similar line suggestions."""

    def test_similar_lines_suggested_on_mismatch(self, tool_context: ToolContext, git_repo: Path):
        """When SEARCH doesn't match, error suggests similar lines."""
        target = git_repo / "config.py"
        target.write_text("DEBUG = False\nTIMEOUT = 30\n", encoding="utf-8")

        patch = """config.py
```python
<<<<<<< SEARCH
debug = False
=======
debug = True
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        error_text = " ".join(result.get("errors", []))
        assert "Did you mean" in error_text or "DEBUG" in error_text

    def test_insertion_hint_for_invented_content(self, tool_context: ToolContext, git_repo: Path):
        """Error provides insertion hint when SEARCH content is invented."""
        target = git_repo / "README.md"
        target.write_text("# Title\n", encoding="utf-8")

        patch = """README.md
```markdown
<<<<<<< SEARCH
## Nonexistent Section
=======
## New Section
>>>>>>> REPLACE
```"""

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        error_text = " ".join(result.get("errors", []))
        # Should suggest using empty SEARCH for insertions
        assert "empty SEARCH" in error_text.lower() or "insertion" in error_text.lower()
