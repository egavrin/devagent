"""Metric aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_dev_agent.engine.react.types import MetricsSnapshot

from .coverage import PatchCoverageResult, compute_patch_coverage
from .diff import DiffMetrics, compute_diff_metrics

if TYPE_CHECKING:
    from pathlib import Path

    from ai_dev_agent.tools.execution.testing.local_tests import TestResult


@dataclass
class MetricsCollector:
    """Collects metrics from repo state and tool results."""

    repo_root: Path
    diff_base: str | None = None

    def collect(
        self,
        *,
        test_result: TestResult | None = None,
        lint_errors: int | None = None,
        type_errors: int | None = None,
        format_errors: int | None = None,
        compile_errors: int | None = None,
        coverage_xml: Path | None = None,
        secrets_found: int | None = None,
        sandbox_violations: int | None = None,
        flaky_tests: int | None = None,
        tokens_cost: float | None = None,
        wall_time: float | None = None,
    ) -> MetricsSnapshot:
        diff_metrics = compute_diff_metrics(self.repo_root, compare_ref=self.diff_base)
        coverage_result = compute_patch_coverage(
            self.repo_root,
            coverage_xml=coverage_xml,
            compare_ref=self.diff_base,
        )
        return build_metrics_snapshot(
            test_result=test_result,
            lint_errors=lint_errors,
            type_errors=type_errors,
            format_errors=format_errors,
            compile_errors=compile_errors,
            diff_metrics=diff_metrics,
            coverage=coverage_result,
            secrets_found=secrets_found,
            sandbox_violations=sandbox_violations,
            flaky_tests=flaky_tests,
            tokens_cost=tokens_cost,
            wall_time=wall_time,
        )


def build_metrics_snapshot(
    *,
    test_result: TestResult | None = None,
    lint_errors: int | None = None,
    type_errors: int | None = None,
    format_errors: int | None = None,
    compile_errors: int | None = None,
    diff_metrics: DiffMetrics | None = None,
    coverage: PatchCoverageResult | None = None,
    secrets_found: int | None = None,
    sandbox_violations: int | None = None,
    flaky_tests: int | None = None,
    tokens_cost: float | None = None,
    wall_time: float | None = None,
) -> MetricsSnapshot:
    snapshot = MetricsSnapshot(
        tests_passed=test_result.success if test_result else None,
        lint_errors=lint_errors,
        type_errors=type_errors,
        format_errors=format_errors,
        compile_errors=compile_errors,
        diff_lines=diff_metrics.total_lines if diff_metrics else None,
        diff_files=diff_metrics.file_count if diff_metrics else None,
        diff_concentration=diff_metrics.concentration if diff_metrics else None,
        patch_coverage=coverage.ratio if coverage else None,
        secrets_found=secrets_found,
        sandbox_violations=sandbox_violations,
        flaky_tests=flaky_tests,
        tokens_cost=tokens_cost,
        wall_time=wall_time,
    )
    if coverage:
        snapshot.raw.setdefault("coverage_details", coverage.per_file)
    else:
        snapshot.gate_notes.setdefault("patch_coverage", "Patch coverage not available.")
    if diff_metrics:
        snapshot.raw.setdefault("diff_files", diff_metrics.files)
    else:
        snapshot.gate_notes.setdefault("diff_limits", "Diff metrics not available.")
    if test_result:
        snapshot.raw.setdefault("test_command", test_result.command)
        snapshot.raw.setdefault("test_stdout", test_result.stdout)
        snapshot.raw.setdefault("test_stderr", test_result.stderr)
    else:
        snapshot.gate_notes.setdefault("tests", "Tests not executed yet.")
    if lint_errors is None:
        snapshot.gate_notes.setdefault("lint", "Lint results missing.")
    if type_errors is None:
        snapshot.gate_notes.setdefault("types", "Type-check results missing.")
    if format_errors is None:
        snapshot.gate_notes.setdefault("format", "Formatter not run.")
    if compile_errors is None:
        snapshot.gate_notes.setdefault("compile", "Build step not run.")
    if secrets_found is None:
        snapshot.gate_notes.setdefault("secrets", "Secret scan not executed.")
    if sandbox_violations is None:
        snapshot.gate_notes.setdefault("sandbox", "Sandbox metrics unavailable.")
    if flaky_tests is None:
        snapshot.gate_notes.setdefault("flaky", "Flake check not run.")
    return snapshot


__all__ = ["MetricsCollector", "build_metrics_snapshot"]
