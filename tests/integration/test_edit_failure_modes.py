"""Integration tests for the patch-based EDIT tool."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from ai_dev_agent.tools.filesystem.search_replace import _fs_edit
from ai_dev_agent.tools.registry import ToolContext

PATCH_HEADER = "*** Begin Patch"
PATCH_FOOTER = "*** End Patch"


def make_patch(body: str) -> str:
    """Wrap patch body with begin/end sentinels."""
    return f"{PATCH_HEADER}\n{body}\n{PATCH_FOOTER}"


@pytest.fixture
def tool_context(tmp_path):
    """Create a mock ToolContext for testing."""
    return ToolContext(
        repo_root=tmp_path, settings=Mock(), sandbox=Mock(), metrics_collector=Mock()
    )


class TestApplyPatchSuccess:
    """Happy-path tests for the EDIT tool."""

    def test_update_existing_file(self, tool_context, tmp_path):
        target = tmp_path / "app.py"
        target.write_text(
            "def greet(name):\n" '    return f"Hello, {name}"\n',
            encoding="utf-8",
        )

        patch = make_patch(
            """*** Update File: app.py
@@
-def greet(name):
-    return f"Hello, {name}"
+def greet(name: str) -> str:
+    return f"Hi, {name}!"
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert result["changes_applied"] == 1
        assert target.read_text(encoding="utf-8") == (
            "def greet(name: str) -> str:\n" '    return f"Hi, {name}!"\n'
        )

    def test_add_new_file(self, tool_context, tmp_path):
        patch = make_patch(
            """*** Add File: src/new_module.py
+def added():
+    return "ok"
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        new_file = tmp_path / "src" / "new_module.py"
        assert result["success"]
        assert new_file.exists()
        assert 'return "ok"' in new_file.read_text()
        assert result["new_files"] == ["src/new_module.py"]

    def test_delete_file(self, tool_context, tmp_path):
        target = tmp_path / "old.txt"
        target.write_text("legacy", encoding="utf-8")

        patch = make_patch(
            """*** Delete File: old.txt
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        assert not target.exists()
        assert "old.txt" in result["changed_files"]

    def test_move_file(self, tool_context, tmp_path):
        target = tmp_path / "pkg" / "models" / "old_name.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("IDENTITY = 'old'\n", encoding="utf-8")

        patch = make_patch(
            """*** Update File: pkg/models/old_name.py
*** Move to: pkg/models/new_name.py
@@
-IDENTITY = 'old'
+IDENTITY = 'new'
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert result["success"]
        new_path = tmp_path / "pkg" / "models" / "new_name.py"
        assert new_path.exists()
        assert "IDENTITY = 'new'\n" == new_path.read_text()


class TestApplyPatchFailures:
    """Failure scenarios with actionable diagnostics."""

    def test_missing_patch_parameter(self, tool_context):
        result = _fs_edit({}, tool_context)

        assert not result["success"]
        assert "patch" in result["errors"][0].lower()

    def test_invalid_patch_structure(self, tool_context):
        result = _fs_edit({"patch": "not a patch"}, tool_context)

        assert not result["success"]
        assert "begin patch" in result["errors"][0].lower()

    def test_update_missing_file(self, tool_context):
        patch = make_patch(
            """*** Update File: no_such_file.py
@@
-print("hi")
+print("bye")
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert "missing file" in result["errors"][0].lower()

    def test_chunk_context_not_found(self, tool_context, tmp_path):
        target = tmp_path / "service.py"
        target.write_text("def run():\n    return 1\n", encoding="utf-8")

        patch = make_patch(
            """*** Update File: service.py
@@
-def run():
-    return 2
+def run():
+    return 3
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert "context" in result["errors"][0].lower()
        assert target.read_text(encoding="utf-8") == "def run():\n    return 1\n"

    def test_add_file_that_exists(self, tool_context, tmp_path):
        existing = tmp_path / "dup.py"
        existing.write_text("print('original')\n", encoding="utf-8")

        patch = make_patch(
            """*** Add File: dup.py
+print("new")
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert "already exists" in result["errors"][0]

    def test_delete_missing_file(self, tool_context):
        patch = make_patch(
            """*** Delete File: nope.py
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert "missing file" in result["errors"][0].lower()


class TestDiagnostics:
    """Ensure the tool provides actionable diagnostics back to the agent."""

    def test_reports_multiple_errors(self, tool_context, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("def func():\n    return 1\n", encoding="utf-8")

        patch = make_patch(
            """*** Update File: file.py
@@
-def func():
-    return 2
+def func():
+    return 3
*** Delete File: missing.py
"""
        )

        result = _fs_edit({"patch": patch}, tool_context)

        assert not result["success"]
        assert len(result["errors"]) >= 2
        assert "context" in result["errors"][0].lower()
        assert "missing" in result["errors"][1].lower()

    def test_returns_changed_files_on_partial_success(self, tool_context, tmp_path):
        target = tmp_path / "file.py"
        target.write_text("flag = False\n", encoding="utf-8")

        patch = make_patch(
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
        assert "file.py" in result["changed_files"]
        assert "already exists" in result["errors"][-1]
