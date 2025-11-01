"""Tests for metrics collectors module."""

from __future__ import annotations

from pathlib import Path

from ai_dev_agent.engine.metrics.collectors import MetricsCollector, build_metrics_snapshot
from ai_dev_agent.tools.execution.testing.local_tests import TestResult


def test_build_metrics_snapshot_empty():
    """Test building metrics snapshot with no data."""
    snapshot = build_metrics_snapshot()

    assert snapshot.tests_passed is None
    assert snapshot.lint_errors is None
    assert snapshot.type_errors is None
    assert snapshot.format_errors is None
    assert snapshot.compile_errors is None
    assert snapshot.diff_lines is None
    assert snapshot.diff_files is None
    assert snapshot.secrets_found is None
    assert snapshot.sandbox_violations is None
    assert snapshot.flaky_tests is None
    assert snapshot.tokens_cost is None
    assert snapshot.wall_time is None

    # Check gate notes are populated for missing data
    assert "patch_coverage" in snapshot.gate_notes
    assert "diff_limits" in snapshot.gate_notes
    assert "tests" in snapshot.gate_notes


def test_build_metrics_snapshot_with_test_result():
    """Test building metrics snapshot with test result."""
    test_result = TestResult(
        command=["pytest"],
        returncode=0,
        stdout="test output",
        stderr="",
    )

    snapshot = build_metrics_snapshot(test_result=test_result)

    assert snapshot.tests_passed is True
    assert snapshot.raw.get("test_command") == ["pytest"]
    assert snapshot.raw.get("test_stdout") == "test output"
    assert snapshot.raw.get("test_stderr") == ""


def test_build_metrics_snapshot_with_errors():
    """Test building metrics snapshot with error counts."""
    snapshot = build_metrics_snapshot(
        lint_errors=5,
        type_errors=3,
        format_errors=2,
        compile_errors=1,
        secrets_found=0,
        sandbox_violations=0,
        flaky_tests=1,
    )

    assert snapshot.lint_errors == 5
    assert snapshot.type_errors == 3
    assert snapshot.format_errors == 2
    assert snapshot.compile_errors == 1
    assert snapshot.secrets_found == 0
    assert snapshot.sandbox_violations == 0
    assert snapshot.flaky_tests == 1

    # These should not be in gate_notes since they were provided
    assert "lint" not in snapshot.gate_notes
    assert "types" not in snapshot.gate_notes
    assert "format" not in snapshot.gate_notes
    assert "compile" not in snapshot.gate_notes


def test_build_metrics_snapshot_with_costs():
    """Test building metrics snapshot with cost metrics."""
    snapshot = build_metrics_snapshot(
        tokens_cost=1000.5,
        wall_time=123.45,
    )

    assert snapshot.tokens_cost == 1000.5
    assert snapshot.wall_time == 123.45


def test_metrics_collector_init():
    """Test MetricsCollector initialization."""
    repo_root = Path.cwd()
    collector = MetricsCollector(repo_root=repo_root)

    assert collector.repo_root == repo_root
    assert collector.diff_base is None


def test_metrics_collector_with_diff_base():
    """Test MetricsCollector with diff base."""
    repo_root = Path.cwd()
    collector = MetricsCollector(repo_root=repo_root, diff_base="main")

    assert collector.repo_root == repo_root
    assert collector.diff_base == "main"
