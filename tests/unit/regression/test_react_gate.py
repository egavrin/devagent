from __future__ import annotations

from ai_dev_agent.engine.react.evaluator import GateEvaluator
from ai_dev_agent.engine.react.types import (
    ActionRequest,
    EvaluationResult,
    GateConfig,
    MetricsSnapshot,
    Observation,
    StepRecord,
)


def _make_step(index: int, metrics: MetricsSnapshot) -> StepRecord:
    action = ActionRequest(step_id=f"S{index}", thought="", tool="noop", args={}, metadata={})
    observation = Observation(success=True, outcome="", metrics=metrics.model_dump(), tool="noop")
    evaluation = EvaluationResult(gates={}, should_stop=False, status="in_progress")
    return StepRecord(
        action=action,
        observation=observation,
        metrics=metrics,
        evaluation=evaluation,
        step_index=index,
    )


def test_gate_success_when_all_metrics_satisfied() -> None:
    evaluator = GateEvaluator(
        GateConfig(
            diff_limit_lines=10,
            diff_limit_files=2,
            patch_coverage_target=0.8,
            stuck_threshold=3,
            steps_budget=5,
        )
    )
    metrics = MetricsSnapshot(
        tests_passed=True,
        lint_errors=0,
        type_errors=0,
        format_errors=0,
        compile_errors=0,
        diff_lines=4,
        diff_files=1,
        patch_coverage=0.95,
        secrets_found=0,
        sandbox_violations=0,
        flaky_tests=0,
    )
    result = evaluator.evaluate(metrics, [])
    assert result.should_stop is True
    assert result.status == "success"
    assert all(result.gates.values())


def test_gate_failure_highlights_first_failing_gate() -> None:
    evaluator = GateEvaluator(
        GateConfig(diff_limit_lines=10, diff_limit_files=2, patch_coverage_target=0.8)
    )
    metrics = MetricsSnapshot(
        tests_passed=True,
        lint_errors=2,
        type_errors=0,
        format_errors=0,
        compile_errors=0,
        diff_lines=3,
        diff_files=1,
        patch_coverage=0.9,
        secrets_found=0,
        sandbox_violations=0,
        flaky_tests=0,
    )
    result = evaluator.evaluate(metrics, [])
    assert result.should_stop is False
    assert result.gates["lint"] is False
    assert result.next_action_hint is not None
    assert "lint" in result.next_action_hint.lower()


def test_gate_marks_stuck_after_no_progress_streak() -> None:
    config = GateConfig(
        diff_limit_lines=10,
        diff_limit_files=2,
        patch_coverage_target=0.8,
        stuck_threshold=3,
        steps_budget=10,
    )
    evaluator = GateEvaluator(config)
    base_metrics = MetricsSnapshot(
        tests_passed=False,
        lint_errors=2,
        type_errors=1,
        format_errors=1,
        compile_errors=0,
        diff_lines=5,
        diff_files=1,
        patch_coverage=0.3,
        secrets_found=0,
        sandbox_violations=0,
        flaky_tests=0,
    )
    history = [_make_step(index + 1, base_metrics) for index in range(3)]
    result = evaluator.evaluate(base_metrics, history)
    assert result.should_stop is True
    assert result.status == "blocked"
    assert result.stop_reason is not None
