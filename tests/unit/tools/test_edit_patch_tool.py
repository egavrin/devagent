"""Unit tests for the patch-based EDIT tool implementation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.tools.filesystem.search_replace import (
    ParseResult,
    PatchChunk,
    PatchParser,
    _fs_edit,
)
from ai_dev_agent.tools.registry import ToolContext

PATCH_BEGIN = "*** Begin Patch"
PATCH_END = "*** End Patch"


def wrap_patch(body: str) -> str:
    return f"{PATCH_BEGIN}\n{body}\n{PATCH_END}"


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


class TestPatchParser:
    def test_parse_update_chunk(self):
        parser = PatchParser(
            wrap_patch(
                """*** Update File: src/app.py
@@
-old
+new
"""
            )
        )

        actions = parser.parse()
        assert len(actions) == 1
        action = actions[0]
        assert action.type == "update"
        assert action.path == "src/app.py"
        assert len(action.chunks) == 1
        chunk = action.chunks[0]
        assert isinstance(chunk, PatchChunk)
        assert chunk.old_lines == ["old"]
        assert chunk.new_lines == ["new"]

    def test_parse_add_and_delete(self):
        parser = PatchParser(
            wrap_patch(
                """*** Add File: new.txt
+hello
+world
*** Delete File: old.txt
"""
            )
        )
        actions = parser.parse()
        assert [a.type for a in actions] == ["add", "delete"]
        assert actions[0].content == "hello\nworld\n"
        assert actions[1].path == "old.txt"

    def test_parse_move(self):
        parser = PatchParser(
            wrap_patch(
                """*** Update File: src/old.py
*** Move to: src/new.py
@@
-VALUE = "old"
+VALUE = "new"
"""
            )
        )
        action = parser.parse()[0]
        assert action.move_path == "src/new.py"

    def test_update_header_missing_colon_auto_corrected(self):
        """Missing colon is auto-corrected and a warning is returned."""
        parser = PatchParser(
            wrap_patch(
                """*** Update File src/app.py
@@
-old
+new
"""
            )
        )

        # Auto-correction happens; parse_with_warnings returns result with warnings
        result = parser.parse_with_warnings()
        assert len(result.actions) == 1
        assert result.actions[0].path == "src/app.py"
        assert len(result.warnings) == 1
        assert "Auto-corrected" in result.warnings[0]
        # Warning format: "Auto-corrected '*** Update File' to '*** Update File:' for path 'src/app.py'"
        assert "*** Update File:" in result.warnings[0]
        assert "src/app.py" in result.warnings[0]

    def test_invalid_patch_raises(self):
        parser = PatchParser("no markers")
        with pytest.raises(ValueError):
            parser.parse()


class TestFsEdit:
    def test_apply_patch_success(self, tool_context: ToolContext, git_repo: Path):
        target = git_repo / "src" / "svc.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def fn():\n    return 1\n", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File: src/svc.py
@@
-def fn():
-    return 1
+def fn():
+    return 2
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert target.read_text(encoding="utf-8") == "def fn():\n    return 2\n"

    def test_missing_patch_parameter(self, tool_context: ToolContext):
        result = _fs_edit({}, tool_context)
        assert not result["success"]
        assert "patch" in (result["errors"][0].lower())

    def test_patch_failure_reports_errors(self, tool_context: ToolContext):
        patch = wrap_patch(
            """*** Update File: src/missing.py
@@
-a
+b
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)
        assert not result["success"]
        assert "missing" in result["errors"][0].lower()

    def test_context_mismatch_reported_without_writing(
        self, tool_context: ToolContext, git_repo: Path
    ):
        target = git_repo / "module.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File: module.py
@@
-VALUE = 2
+VALUE = 3
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert any("context" in err.lower() for err in result["errors"])
        assert target.read_text(encoding="utf-8") == "VALUE = 1\n"

    def test_missing_colon_auto_corrected_with_warning(
        self, tool_context: ToolContext, git_repo: Path
    ):
        """Missing colon is auto-corrected and edit succeeds with warning."""
        target = git_repo / "src" / "app.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old\n", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File src/app.py
@@
-old
+new
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert target.read_text(encoding="utf-8") == "new\n"
        # Warnings should be present indicating auto-correction
        assert "warnings" in result
        assert len(result["warnings"]) == 1
        assert "Auto-corrected" in result["warnings"][0]

    def test_validation_detects_move_conflicts_before_apply(
        self, tool_context: ToolContext, git_repo: Path
    ):
        source = git_repo / "a.py"
        dest = git_repo / "b.py"
        source.write_text("print('A')\n", encoding="utf-8")
        dest.write_text("print('B')\n", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File: a.py
*** Move to: b.py
@@
-print('A')
+print('moved')
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert any("already exists" in err for err in result["errors"])
        assert source.read_text(encoding="utf-8") == "print('A')\n"
        assert dest.read_text(encoding="utf-8") == "print('B')\n"


class TestPatchApplierEdgeCases:
    def test_add_file_conflict(self, tool_context: ToolContext, git_repo: Path):
        target = git_repo / "dup.py"
        target.write_text("print('original')\n", encoding="utf-8")

        patch = wrap_patch(
            """*** Add File: dup.py
+print("new")
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert any("already exists" in err for err in result["errors"])

    def test_delete_missing_file_error(self, tool_context: ToolContext):
        patch = wrap_patch(
            """*** Delete File: nope.py
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert any("missing file" in err.lower() for err in result["errors"])

    def test_move_target_conflict(self, tool_context: ToolContext, git_repo: Path):
        src = git_repo / "src.py"
        src.write_text("def value():\n    return 1\n", encoding="utf-8")
        dest = git_repo / "dest.py"
        dest.write_text("def value():\n    return 0\n", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File: src.py
*** Move to: dest.py
@@
-def value():
-    return 1
+def value():
+    return 2
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert any("already exists" in err for err in result["errors"])

    def test_insert_chunk_without_old_lines(self, tool_context: ToolContext, git_repo: Path):
        target = git_repo / "module.py"
        target.write_text("HEADER", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File: module.py
@@ HEADER
+MIDDLE
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert target.read_text(encoding="utf-8") == "HEADERMIDDLE"

    def test_partial_success_counts(self, tool_context: ToolContext, git_repo: Path):
        target = git_repo / "file.py"
        target.write_text("flag = False\n", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File: file.py
@@
-flag = False
+flag = True
*** Add File: file.py
+print("duplicate")
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert result["changes_applied"] == 1
        assert "file.py" in result["changed_files"]

    def test_eof_marker_appends_content(self, tool_context: ToolContext, git_repo: Path):
        """*** End of File marker appends content at end of file."""
        target = git_repo / "README.md"
        target.write_text("# Title\n\nExisting content.\n", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File: README.md
@@
+
+## New Section
+Appended at end.
*** End of File
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        content = target.read_text(encoding="utf-8")
        assert "# Title" in content
        assert "Existing content." in content
        assert "## New Section" in content
        assert "Appended at end." in content


class TestFuzzyMatching:
    """Tests for layered fuzzy matching and whitespace tolerance."""

    def test_trailing_whitespace_tolerance(self, tool_context: ToolContext, git_repo: Path):
        """Trailing whitespace in file doesn't break exact substring match."""
        target = git_repo / "module.py"
        # File has trailing spaces after "value = 1"
        target.write_text("value = 1   \n", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File: module.py
@@
-value = 1
+value = 2
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        # Should succeed - exact substring match works with trailing whitespace
        assert result["success"]
        assert "value = 2" in target.read_text(encoding="utf-8")
        # No fuzzy matching warning when exact match works
        warnings = result.get("warnings", [])
        assert not any("stripping" in w.lower() for w in warnings)

    def test_trailing_whitespace_tolerance_multiline(
        self, tool_context: ToolContext, git_repo: Path
    ):
        """When multiline patch lines have trailing whitespace differences, fuzzy matching helps."""
        target = git_repo / "module.py"
        # File has trailing spaces on each line
        target.write_text("def fn():   \n    return 1   \n", encoding="utf-8")

        # Patch doesn't have trailing spaces (common LLM output)
        patch = wrap_patch(
            """*** Update File: module.py
@@
-def fn():
-    return 1
+def fn():
+    return 2
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        # Should succeed via fuzzy matching
        assert result["success"]
        assert "return 2" in target.read_text(encoding="utf-8")
        # Should have a warning about fuzzy match
        assert "warnings" in result
        # Warning message: "Chunk 1 in 'module.py': Matched after stripping trailing whitespace"
        assert any("stripping" in w.lower() for w in result.get("warnings", []))

    def test_leading_whitespace_tolerance(self, tool_context: ToolContext, git_repo: Path):
        """Patch matches even with uniform indentation differences."""
        target = git_repo / "module.py"
        # File has 4-space indent
        target.write_text("    def fn():\n        return 1\n", encoding="utf-8")

        # Patch has 2-space indent (uniform difference)
        patch = wrap_patch(
            """*** Update File: module.py
@@
-  def fn():
-      return 1
+  def fn():
+      return 2
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert "return 2" in target.read_text(encoding="utf-8")
        # Should have a warning about indentation adjustment
        assert "warnings" in result
        assert any("indent" in w.lower() for w in result.get("warnings", []))

    def test_exact_match_preferred_over_fuzzy(self, tool_context: ToolContext, git_repo: Path):
        """When exact match exists, no fuzzy match warning is generated."""
        target = git_repo / "module.py"
        target.write_text("value = 1\n", encoding="utf-8")

        patch = wrap_patch(
            """*** Update File: module.py
@@
-value = 1
+value = 2
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        # No fuzzy matching warnings when exact match works
        warnings = result.get("warnings", [])
        assert not any("whitespace" in w.lower() for w in warnings)
        assert not any("indent" in w.lower() for w in warnings)


class TestSimilarLineSuggestions:
    """Tests for find_similar_lines error diagnostics."""

    def test_similar_lines_suggested_on_mismatch(self, tool_context: ToolContext, git_repo: Path):
        """When context doesn't match, error suggests similar lines."""
        target = git_repo / "config.py"
        target.write_text("DEBUG = False\nTIMEOUT = 30\n", encoding="utf-8")

        # Slightly wrong - "debug" instead of "DEBUG"
        patch = wrap_patch(
            """*** Update File: config.py
@@
-debug = False
+debug = True
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        # Error should include "Did you mean" suggestion with actual content
        error_text = " ".join(result.get("errors", []))
        assert "Did you mean" in error_text or "DEBUG" in error_text

    def test_no_similar_lines_when_very_different(self, tool_context: ToolContext, git_repo: Path):
        """When content is completely different, no misleading suggestions."""
        target = git_repo / "config.py"
        target.write_text("DEBUG = False\n", encoding="utf-8")

        # Completely unrelated content
        patch = wrap_patch(
            """*** Update File: config.py
@@
-xyz_totally_unrelated_content
+abc_replacement
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        # Should mention context not found but not suggest unrelated lines
        error_text = " ".join(result.get("errors", []))
        assert "context" in error_text.lower() or "not found" in error_text.lower()


class TestAutoCorrection:
    """Tests for directive auto-correction behavior."""

    def test_add_file_missing_colon_auto_corrected(self):
        """*** Add File without colon is auto-corrected."""
        parser = PatchParser(
            wrap_patch(
                """*** Add File new.txt
+hello world
"""
            )
        )

        result = parser.parse_with_warnings()
        assert len(result.actions) == 1
        assert result.actions[0].type == "add"
        assert result.actions[0].path == "new.txt"
        assert len(result.warnings) == 1
        assert "Auto-corrected" in result.warnings[0]

    def test_delete_file_missing_colon_auto_corrected(self):
        """*** Delete File without colon is auto-corrected."""
        parser = PatchParser(
            wrap_patch(
                """*** Delete File old.txt
"""
            )
        )

        result = parser.parse_with_warnings()
        assert len(result.actions) == 1
        assert result.actions[0].type == "delete"
        assert result.actions[0].path == "old.txt"
        assert len(result.warnings) == 1
        assert "Auto-corrected" in result.warnings[0]

    def test_move_to_missing_colon_auto_corrected(self):
        """*** Move to without colon is auto-corrected."""
        parser = PatchParser(
            wrap_patch(
                """*** Update File: src/old.py
*** Move to src/new.py
@@
-VALUE = "old"
+VALUE = "new"
"""
            )
        )

        result = parser.parse_with_warnings()
        assert len(result.actions) == 1
        assert result.actions[0].move_path == "src/new.py"
        assert len(result.warnings) == 1
        assert "Auto-corrected" in result.warnings[0]

    def test_multiple_auto_corrections_accumulated(self):
        """Multiple missing colons result in multiple warnings."""
        parser = PatchParser(
            wrap_patch(
                """*** Add File new1.txt
+line1
*** Add File new2.txt
+line2
"""
            )
        )

        result = parser.parse_with_warnings()
        assert len(result.actions) == 2
        assert len(result.warnings) == 2
        assert all("Auto-corrected" in w for w in result.warnings)
