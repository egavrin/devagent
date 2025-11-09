import subprocess
from pathlib import Path

import pytest

from ai_dev_agent.tools.code.code_edit.diff_utils import (
    DiffError,
    DiffProcessor,
    DiffValidationResult,
    apply_patch,
    extract_diff,
)


def test_extract_diff_from_code_block(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)
    diff_text = """Here is a change:
```diff
--- a/sample.txt
+++ b/sample.txt
@@ -1 +1 @@
-old
+new
```
"""
    extracted, validation = processor.extract_and_validate_diff(diff_text)

    assert extracted.startswith("--- a/sample.txt")
    assert extracted.endswith("+new\n")
    assert validation.affected_files == ["sample.txt"]
    assert validation.lines_added == 1
    assert validation.lines_removed == 1


def test_extract_diff_without_markers_raises(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)

    with pytest.raises(DiffError):
        processor.extract_and_validate_diff("No diff content present here.")


def test_validate_diff_detects_conflict_markers(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)
    conflict_diff = """--- a/sample.txt
+++ b/sample.txt
@@ -1,3 +1,6 @@
 line1
-line2
+<<<<<<< HEAD
+line2-change
+=======
+line2-other
+>>>>>>> feature
 line3
"""

    validation = processor._validate_diff(conflict_diff)

    assert validation.errors and any("conflict" in error.lower() for error in validation.errors)


def test_apply_diff_safely_rejects_conflicts(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)
    conflict_diff = """--- a/sample.txt
+++ b/sample.txt
@@ -1,3 +1,6 @@
 line1
-line2
+<<<<<<< HEAD
+line2-change
+=======
+line2-other
+>>>>>>> feature
 line3
"""

    with pytest.raises(DiffError):
        processor.apply_diff_safely(conflict_diff)


def test_validate_diff_reports_invalid_hunk(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)
    invalid_hunk_diff = """--- a/invalid.txt
+++ b/invalid.txt
@@ -X,Y +1,1 @@
-line
+line
"""

    validation = processor._validate_diff(invalid_hunk_diff)

    assert validation.errors and "invalid hunk header" in validation.errors[0].lower()


def test_apply_diff_reports_subprocess_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = DiffProcessor(tmp_path)
    target = tmp_path / "existing.txt"
    target.write_text("old\n", encoding="utf-8")

    diff_text = """--- a/existing.txt
+++ b/existing.txt
@@ -1 +1 @@
-old
+new
"""

    class FakeResult:
        def __init__(self, returncode: int, stderr: bytes) -> None:
            self.returncode = returncode
            self.stdout = b""
            self.stderr = stderr

    call_count = {"count": 0}

    def fake_run(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return FakeResult(1, b"git conflict")
        return FakeResult(1, b"patch failed")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DiffError) as excinfo:
        processor.apply_diff_safely(diff_text)

    message = str(excinfo.value)
    # The error message should contain the actual git error
    assert "git conflict" in message or "Patch application failed" in message


def test_validation_has_issues_property() -> None:
    result = DiffValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        affected_files=[],
        lines_added=0,
        lines_removed=0,
    )

    assert result.has_issues is False
    result.warnings.append("notice")
    assert result.has_issues is True


def test_normalize_diff_path_preserves_relative(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)

    assert processor._normalize_diff_path("sample.txt") == "sample.txt"
    assert processor._normalize_diff_path("a/sample.txt") == "sample.txt"


def test_extract_diff_selects_first_block(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)
    mixed_text = """context line
--- a/first.txt
+++ b/first.txt
@@ -1 +1 @@
-old
+new

--- a/second.txt
+++ b/second.txt
@@ -1 +1 @@
-foo
+bar
"""

    extracted, validation = processor.extract_and_validate_diff(mixed_text)

    assert "--- a/second.txt" not in extracted
    assert validation.affected_files == ["first.txt"]


def test_create_preview_includes_summary_warnings(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)
    diff_text = """--- a/missing.txt
+++ b/missing.txt
@@ -1 +1 @@
-old value
+new value
"""

    preview = processor.create_preview(diff_text)

    assert preview.validation_result.has_issues is True
    # We now have 2 warnings: file doesn't exist + insufficient context
    assert "🔶 2 warnings" in preview.summary
    assert preview.file_changes["missing.txt"]["added"] == 1


def test_extract_diff_legacy_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    text = """intro
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old
+new
"""

    extracted = extract_diff(text)

    assert extracted.startswith("--- a/file.txt")
    assert extracted.endswith("+new\n")


def test_apply_diff_uses_git_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = DiffProcessor(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("old\n", encoding="utf-8")

    diff_text = """--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old
+new
"""

    class FakeResult:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode
            self.stdout = b""
            self.stderr = b""

    commands = []

    def fake_run(cmd, *args, **kwargs):
        commands.append(cmd[0])
        return FakeResult(0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert processor.apply_diff_safely(diff_text) is True
    assert commands == ["git"]


def test_apply_diff_uses_patch_on_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = DiffProcessor(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("old\n", encoding="utf-8")

    diff_text = """--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old
+new
"""

    class FakeResult:
        def __init__(self, returncode: int, stderr: bytes = b"") -> None:
            self.returncode = returncode
            self.stdout = b""
            self.stderr = stderr

    commands = []

    def fake_run(cmd, *args, **kwargs):
        commands.append(cmd[0])
        if len(commands) == 1:
            return FakeResult(1, b"failure")
        return FakeResult(0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert processor.apply_diff_safely(diff_text) is True
    assert commands == ["git", "patch"]


def test_apply_simple_fallback_removes_directory(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)
    target_dir = tmp_path / "folder"
    target_dir.mkdir()
    (target_dir / "nested.txt").write_text("data\n", encoding="utf-8")

    diff_text = """--- a/folder
+++ /dev/null
"""

    assert processor._apply_simple_fallback(diff_text)
    assert not target_dir.exists()


def test_apply_simple_fallback_handles_missing_target(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)
    diff_text = """--- a/missing.txt
+++ /dev/null
"""

    assert processor._apply_simple_fallback(diff_text) is False


def test_apply_simple_fallback_rejects_additions(tmp_path: Path) -> None:
    processor = DiffProcessor(tmp_path)
    diff_text = """--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old
+new
"""

    assert processor._apply_simple_fallback(diff_text) is False


def test_apply_patch_legacy_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diff_text = """--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old
+new
"""

    class FakeResult:
        def __init__(self, returncode: int, stderr: bytes = b"") -> None:
            self.returncode = returncode
            self.stdout = b""
            self.stderr = stderr

    def fake_run(*args, **kwargs):
        return FakeResult(1, b"git failure")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DiffError):
        apply_patch(diff_text, tmp_path)
