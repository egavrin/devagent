"""Tests for patch-level coverage utilities."""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_dev_agent.engine.metrics.coverage import PatchCoverageResult, compute_patch_coverage


def write_coverage_xml(root: Path, filename: str, *, covered: list[int], total: int = 10) -> Path:
    template = [
        '<?xml version="1.0" ?>',
        '<coverage branch-rate="0" line-rate="0" version="7.3.2">',
        "  <packages>",
        '    <package name="ai_dev_agent">',
        "      <classes>",
        f'        <class name="sample" filename="{filename}">',
        "          <lines>",
    ]
    template.extend(
        f'            <line hits="{1 if i in covered else 0}" number="{i}"/>'
        for i in range(1, total + 1)
    )
    template.extend(
        [
            "          </lines>",
            "        </class>",
            "      </classes>",
            "    </package>",
            "  </packages>",
            "</coverage>",
        ]
    )
    path = root / "coverage.xml"
    path.write_text("\n".join(template), encoding="utf-8")
    return path


def test_compute_patch_coverage_counts_changed_lines(tmp_path, monkeypatch):
    coverage_file = write_coverage_xml(tmp_path, "ai_dev_agent/module.py", covered=[2, 3])

    def fake_collect(repo_root: Path, compare_ref: str | None) -> dict[str, set[int]]:
        assert repo_root == tmp_path
        return {"ai_dev_agent/module.py": {2, 4}}

    monkeypatch.setattr(
        "ai_dev_agent.engine.metrics.coverage._collect_changed_lines",
        fake_collect,
    )

    result = compute_patch_coverage(tmp_path, coverage_xml=coverage_file)
    assert isinstance(result, PatchCoverageResult)
    assert result.total_lines == 2
    assert result.covered_lines == 1
    assert result.ratio == pytest.approx(0.5)
    assert result.per_file["ai_dev_agent/module.py"]["covered"] == [2]
    assert result.per_file["ai_dev_agent/module.py"]["uncovered"] == [4]


def test_compute_patch_coverage_handles_missing_xml(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ai_dev_agent.engine.metrics.coverage._collect_changed_lines",
        lambda repo_root, compare_ref: {"a.py": {1}},
    )
    assert compute_patch_coverage(tmp_path, coverage_xml=tmp_path / "missing.xml") is None


def test_compute_patch_coverage_no_changes(tmp_path, monkeypatch):
    coverage_file = write_coverage_xml(tmp_path, "ai_dev_agent/module.py", covered=[1, 2])
    monkeypatch.setattr(
        "ai_dev_agent.engine.metrics.coverage._collect_changed_lines",
        lambda repo_root, compare_ref: {},
    )
    result = compute_patch_coverage(tmp_path, coverage_xml=coverage_file)
    assert result.covered_lines == 0
    assert result.total_lines == 0
    assert result.ratio == pytest.approx(1.0)
