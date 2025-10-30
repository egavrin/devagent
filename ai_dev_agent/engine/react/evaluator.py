"""Gate evaluation and progress tracking for the ReAct loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import EvaluationResult, GateConfig, MetricsSnapshot, StepRecord

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass
class GateEvaluator:
    """Evaluate metrics against hard gates and detect lack of progress."""

    config: GateConfig

    def evaluate(self, metrics: MetricsSnapshot, history: Iterable[StepRecord]) -> EvaluationResult:
        history_list = list(history)
        gates = self._build_gates(metrics)
        notes = metrics.gate_notes.copy()

        # Determine which gates are required by profile
        required = self._required_gate_names()
        failing_required = [name for name in required if not gates.get(name, False)]
        all_passed = len(required) > 0 and not failing_required

        status = "in_progress"
        should_stop = False
        stop_reason: str | None = None
        next_hint: str | None = None

        if all_passed:
            should_stop = True
            status = "success"
            stop_reason = "All mandatory gates satisfied."
        else:
            if failing_required:
                next_hint = self._hint_for_gate(failing_required[0])

            if self._stuck(history_list, metrics):
                should_stop = True
                status = "blocked"
                stop_reason = "No improvement across critical metrics; consider hand-off."

            elif self._exceeded_budget(history_list):
                should_stop = True
                status = "failed"
                stop_reason = "Step budget exhausted without passing gates."

        improved = self._improvements(history_list, metrics)
        if notes:
            for key, note in notes.items():
                if key not in failing_required:
                    continue
                next_hint = f"{next_hint} ({note})" if next_hint else note

        required_map = dict.fromkeys(self._required_gate_names(), True)
        return EvaluationResult(
            gates=gates,
            required_gates=required_map,
            should_stop=should_stop,
            stop_reason=stop_reason,
            next_action_hint=next_hint,
            improved_metrics=improved,
            status=status,
        )

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    def _build_gates(self, metrics: MetricsSnapshot) -> dict[str, bool]:
        gates: dict[str, bool] = {}
        gates["tests"] = metrics.tests_passed is True
        gates["lint"] = metrics.lint_errors == 0 if metrics.lint_errors is not None else False
        gates["types"] = metrics.type_errors == 0 if metrics.type_errors is not None else False
        gates["format"] = metrics.format_errors == 0 if metrics.format_errors is not None else False
        gates["compile"] = (
            metrics.compile_errors == 0 if metrics.compile_errors is not None else False
        )

        diff_ok = True
        if metrics.diff_lines is None or metrics.diff_files is None:
            diff_ok = False
        else:
            diff_ok = (
                metrics.diff_lines <= self.config.diff_limit_lines
                and metrics.diff_files <= self.config.diff_limit_files
            )
        gates["diff_limits"] = diff_ok

        if metrics.patch_coverage is None:
            gates["patch_coverage"] = False
        else:
            gates["patch_coverage"] = metrics.patch_coverage >= self.config.patch_coverage_target

        gates["secrets"] = (
            metrics.secrets_found == 0 if metrics.secrets_found is not None else False
        )
        gates["sandbox"] = (
            metrics.sandbox_violations == 0 if metrics.sandbox_violations is not None else False
        )
        gates["flaky"] = metrics.flaky_tests == 0 if metrics.flaky_tests is not None else False

        # Extended/optional gates (default PASS when not required)
        design_ok = bool(metrics.raw.get("design_doc_ok")) if metrics.raw is not None else False
        gates["design_doc"] = design_ok if self.config.require_design_doc else True
        perf_ok = bool(metrics.raw.get("perf_ok")) if metrics.raw is not None else False
        gates["perf"] = perf_ok if self.config.require_perf else True
        return gates

    def _hint_for_gate(self, gate: str) -> str:
        hints = {
            "tests": "Investigate failing tests and rerun until green.",
            "lint": "Resolve lint violations (run lint tool).",
            "types": "Fix typing errors detected by type checker.",
            "format": "Run formatter to align with style guide.",
            "compile": "Fix compilation/packaging issues.",
            "diff_limits": "Reduce diff size or split changes into smaller patches.",
            "patch_coverage": "Add or extend tests to raise diff coverage above target.",
            "secrets": "Remove secrets or credentials from changes.",
            "sandbox": "Avoid sandbox violations; adjust commands/policies.",
            "flaky": "Stabilize flaky tests before proceeding.",
        }
        return hints.get(gate, f"Address gate '{gate}'.")

    def _required_gate_names(self) -> list[str]:
        cfg = self.config
        mapping = [
            ("tests", cfg.require_tests),
            ("lint", cfg.require_lint),
            ("types", cfg.require_types),
            ("format", cfg.require_format),
            ("compile", cfg.require_compile),
            ("diff_limits", cfg.require_diff_limits),
            ("patch_coverage", cfg.require_patch_coverage),
            ("secrets", cfg.require_secrets),
            ("sandbox", cfg.require_sandbox),
            ("flaky", cfg.require_flaky),
            ("design_doc", cfg.require_design_doc),
            ("perf", cfg.require_perf),
        ]
        return [name for name, required in mapping if required]

    def _stuck(self, history: list[StepRecord], current: MetricsSnapshot) -> bool:
        if not history:
            return False
        window = self.config.stuck_threshold
        if window <= 0:
            return False
        metrics_sequence = [record.metrics for record in history[-window:]] + [current]
        if len(metrics_sequence) <= 1:
            return False

        streak = 0
        for prev, curr in zip(metrics_sequence[:-1], metrics_sequence[1:]):
            if self._has_progress(prev, curr):
                streak = 0
            else:
                streak += 1
            if streak >= window:
                return True
        return False

    def _exceeded_budget(self, history: list[StepRecord]) -> bool:
        budget = self.config.steps_budget
        if budget <= 0:
            return False
        return len(history) + 1 >= budget

    def _has_progress(self, prev: MetricsSnapshot, curr: MetricsSnapshot) -> bool:
        if prev.tests_passed is not True and curr.tests_passed is True:
            return True
        if self._metric_improved(prev.lint_errors, curr.lint_errors, lower_is_better=True):
            return True
        if self._metric_improved(prev.type_errors, curr.type_errors, lower_is_better=True):
            return True
        if self._metric_improved(prev.diff_lines, curr.diff_lines, lower_is_better=True):
            return True
        return bool(
            self._metric_improved(prev.patch_coverage, curr.patch_coverage, lower_is_better=False)
        )

    def _metric_improved(
        self,
        prev_value: float | int | bool | None,
        curr_value: float | int | bool | None,
        *,
        lower_is_better: bool,
    ) -> bool:
        if prev_value is None or curr_value is None:
            return False
        if lower_is_better:
            return curr_value < prev_value
        return curr_value > prev_value

    def _improvements(self, history: list[StepRecord], current: MetricsSnapshot) -> dict[str, bool]:
        if not history:
            return {}
        prev = history[-1].metrics
        improvements: dict[str, bool] = {}
        improvements["tests_passed"] = (
            prev.tests_passed is not True and current.tests_passed is True
        )
        improvements["lint_errors"] = self._metric_improved(
            prev.lint_errors, current.lint_errors, lower_is_better=True
        )
        improvements["type_errors"] = self._metric_improved(
            prev.type_errors, current.type_errors, lower_is_better=True
        )
        improvements["diff_lines"] = self._metric_improved(
            prev.diff_lines, current.diff_lines, lower_is_better=True
        )
        improvements["patch_coverage"] = self._metric_improved(
            prev.patch_coverage, current.patch_coverage, lower_is_better=False
        )
        return {key: value for key, value in improvements.items() if value}


__all__ = ["GateEvaluator"]
