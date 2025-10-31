"""Tests for patch coverage computation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Iterator

import pytest

from ai_dev_agent.engine.metrics.coverage import PatchCoverageResult, compute_patch_coverage


class CompletedProcess:
    """Lightweight stub mimicking subprocess.CompletedProcess for git diff."""

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def _write_coverage_xml(path: Path, *, covered: list[int], uncovered: list[int]) -> None:
    covered_lines = "".join(f'<line number="{line}" hits="1"/>\n' for line in sorted(set(covered)))
    uncovered_lines = "".join(
        f'<line number="{line}" hits="0"/>\n' for line in sorted(set(uncovered))
    )
    xml = f"""\
    <coverage>
        <packages>
            <package name="pkg">
                <classes>
                    <class filename="src/module.py">
                        <lines>
                            {covered_lines}{uncovered_lines}
                        </lines>
                    </class>
                </classes>
            </package>
        </packages>
    </coverage>
    """
    path.write_text(dedent(xml), encoding="utf-8")


def test_compute_patch_coverage_missing_xml(tmp_path: Path) -> None:
    """When no coverage XML is present the helper should return None."""
    result = compute_patch_coverage(tmp_path)
    assert result is None


def test_compute_patch_coverage_merges_git_diff_and_xml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collect changed lines from git diff and combine them with coverage XML."""
    coverage_xml = tmp_path / "coverage.xml"
    _write_coverage_xml(coverage_xml, covered=[10], uncovered=[11, 12])

    diff_output = dedent(
        """\
        diff --git a/src/module.py b/src/module.py
        --- a/src/module.py
        +++ b/src/module.py
        @@ -9,0 +10 @@
        +print("covered")
        @@ -10,0 +11,2 @@
        +print("uncovered one")
        +print("uncovered two")
        """
    )

    def fake_run(
        command: list[str], cwd: str, capture_output: bool, text: bool, check: bool
    ) -> CompletedProcess:
        assert Path(cwd) == tmp_path
        assert command[:3] == ["git", "diff", "HEAD"]
        return CompletedProcess(stdout=diff_output)

    monkeypatch.setattr("ai_dev_agent.engine.metrics.coverage.subprocess.run", fake_run)

    result = compute_patch_coverage(tmp_path, coverage_xml=coverage_xml)
    assert isinstance(result, PatchCoverageResult)
    assert result.total_lines == 3
    assert result.covered_lines == 1
    assert pytest.approx(result.ratio) == 1 / 3
    per_file = result.per_file["src/module.py"]
    assert per_file["covered"] == [10]
    assert per_file["uncovered"] == [11, 12]


def test_compute_patch_coverage_handles_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gracefully handle git diff failures by reporting an empty patch with perfect ratio."""
    coverage_xml = tmp_path / "coverage.xml"
    _write_coverage_xml(coverage_xml, covered=[5], uncovered=[6])

    def fake_run(*args, **kwargs) -> CompletedProcess:
        return CompletedProcess(stdout="", returncode=1)

    monkeypatch.setattr("ai_dev_agent.engine.metrics.coverage.subprocess.run", fake_run)

    result = compute_patch_coverage(tmp_path, coverage_xml=coverage_xml)
    assert isinstance(result, PatchCoverageResult)
    assert result.total_lines == 0
    assert result.covered_lines == 0
    assert result.ratio == 1.0
    assert result.per_file == {}
