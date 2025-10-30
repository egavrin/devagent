import subprocess
from pathlib import Path

from ai_dev_agent.tools.code.code_edit.diff_utils import DiffProcessor


def test_diff_preview_handles_file_addition(tmp_path: Path) -> None:
    diff_text = """--- /dev/null
+++ b/new_file.txt
@@ -0,0 +1,2 @@
+line one
+line two
"""
    processor = DiffProcessor(tmp_path)
    preview = processor.create_preview(diff_text)

    assert "new_file.txt" in preview.file_changes
    changes = preview.file_changes["new_file.txt"]
    assert changes["added"] == 2
    assert changes["removed"] == 0
    assert preview.validation_result.affected_files == ["new_file.txt"]


def test_diff_preview_handles_file_deletion(tmp_path: Path) -> None:
    target = tmp_path / "old_file.txt"
    target.write_text("line one\nline two\n", encoding="utf-8")

    diff_text = """--- a/old_file.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-line one
-line two
"""
    processor = DiffProcessor(tmp_path)
    preview = processor.create_preview(diff_text)

    assert "old_file.txt" in preview.file_changes
    changes = preview.file_changes["old_file.txt"]
    assert changes["removed"] == 2
    assert changes["added"] == 0
    assert "old_file.txt" in preview.validation_result.affected_files


def test_apply_diff_deletion_fallback(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "remove_me.txt"
    target.write_text("line one\nline two\n", encoding="utf-8")

    diff_text = """--- a/remove_me.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-line one
-line two
"""

    processor = DiffProcessor(tmp_path)

    class FailingProcess:
        returncode = 1
        stdout = b""
        stderr = b"failure"

    def fake_run(*args, **kwargs):
        return FailingProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert processor.apply_diff_safely(diff_text)
    assert not target.exists()
