"""Reactive execution loop orchestrating action → observation → evaluation."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Callable, Union

from pydantic import ValidationError

from .types import (
    ActionRequest,
    EvaluationResult,
    MetricsSnapshot,
    Observation,
    RunResult,
    StepRecord,
    TaskSpec,
)

if TYPE_CHECKING:
    from .evaluator import GateEvaluator

ActionProvider = Callable[[TaskSpec, Sequence[StepRecord]], Union[ActionRequest, dict]]
ToolInvoker = Callable[[ActionRequest], Union[Observation, dict]]


class ReactiveExecutor:
    """Coordinates ReAct steps until gates succeed or budgets expire."""

    def __init__(
        self,
        evaluator: GateEvaluator | None = None,
        *,
        default_max_steps: int = 25,
    ) -> None:
        self.evaluator = evaluator
        self.default_max_steps = max(1, int(default_max_steps or 1))

    def run(
        self,
        task: TaskSpec,
        action_provider: ActionProvider,
        tool_invoker: ToolInvoker,
        *,
        prior_steps: Iterable[StepRecord] | None = None,
        max_steps: int | None = None,
    ) -> RunResult:
        start_time = time.perf_counter()
        history: list[StepRecord] = list(prior_steps or [])
        steps: list[StepRecord] = []
        last_eval: EvaluationResult | None = None
        exc_info: Exception | None = None
        stop_condition: str | None = None
        steps_budget = self._resolve_steps_budget(max_steps)
        try:
            step_index = len(history) + 1
            while step_index <= steps_budget:
                try:
                    action_payload = action_provider(task, history + steps)
                except StopIteration:
                    stop_condition = "stop_iteration"
                    break
                try:
                    action = self._ensure_action(action_payload, step_index)
                except ValidationError as exc:
                    raise ValueError(f"Action provider returned invalid payload: {exc}") from exc

                observation = self._invoke_tool(tool_invoker, action)
                metrics = self._metrics_from_observation(observation)
                if self.evaluator:
                    evaluation = self.evaluator.evaluate(metrics, history + steps)
                else:
                    evaluation = EvaluationResult(
                        gates={},
                        required_gates={},
                        should_stop=False,
                        stop_reason=None,
                        next_action_hint=None,
                        improved_metrics={},
                        status="in_progress",
                    )
                record = StepRecord(
                    action=action,
                    observation=observation,
                    metrics=metrics,
                    evaluation=evaluation,
                    step_index=step_index,
                )
                steps.append(record)
                last_eval = evaluation

                if self.evaluator and evaluation.should_stop:
                    stop_condition = "gates"
                    break
                step_index += 1
            else:
                stop_condition = "budget"

        except Exception as exc:
            exc_info = exc
            if not last_eval:
                last_eval = EvaluationResult(
                    gates={},
                    required_gates={},
                    should_stop=True,
                    stop_reason=f"Exception: {exc}",
                    next_action_hint=None,
                    improved_metrics={},
                    status="failed",
                )
            stop_condition = "exception"
        finally:
            runtime = time.perf_counter() - start_time
            gates = last_eval.gates if last_eval else {}
            required_gates = last_eval.required_gates if last_eval else {}
            status, stop_reason = self._derive_status(
                stop_condition,
                last_eval,
                bool(self.evaluator),
                steps,
            )
            metrics_dict = steps[-1].metrics.model_dump() if steps else {}

            result = RunResult(
                task_id=task.identifier,
                status=status,
                steps=history + steps,
                gates=gates,
                required_gates=required_gates,
                stop_reason=stop_reason,
                runtime_seconds=round(runtime, 3),
                metrics=metrics_dict,
            )

        if exc_info:
            raise exc_info
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_action(self, payload: ActionRequest | dict, step_index: int) -> ActionRequest:
        if isinstance(payload, ActionRequest):
            return payload
        if not isinstance(payload, dict):
            raise TypeError(f"Action provider returned unsupported type: {type(payload)!r}")
        payload.setdefault("step_id", f"S{step_index}")
        return ActionRequest.model_validate(payload)

    def _invoke_tool(self, invoker: ToolInvoker, action: ActionRequest) -> Observation:
        try:
            raw = invoker(action)
        except Exception as exc:
            return Observation(
                success=False,
                outcome="Tool invocation raised exception.",
                tool=action.tool,
                error=str(exc),
            )
        if isinstance(raw, Observation):
            if raw.tool is None:
                raw.tool = action.tool
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"Tool invoker returned unsupported type: {type(raw)!r}")
        raw.setdefault("tool", action.tool)
        return Observation.model_validate(raw)

    def _metrics_from_observation(self, observation: Observation) -> MetricsSnapshot:
        metrics = observation.metrics or {}
        if isinstance(metrics, MetricsSnapshot):
            return metrics
        if not isinstance(metrics, dict):
            raise TypeError(f"Observation metrics must be dict-like, received {type(metrics)!r}")
        return MetricsSnapshot.model_validate(metrics)

    def _resolve_steps_budget(self, max_steps: int | None) -> int:
        if self.evaluator:
            budget = self.evaluator.config.steps_budget
            if max_steps is not None:
                try:
                    override = int(max_steps)
                except (TypeError, ValueError):
                    override = budget
                else:
                    override = max(1, override)
                budget = min(budget, override)
            return max(1, budget)

        if max_steps is not None:
            try:
                parsed = int(max_steps)
            except (TypeError, ValueError):
                parsed = self.default_max_steps
            else:
                if parsed <= 0:
                    parsed = self.default_max_steps
            return max(1, parsed)
        return self.default_max_steps

    def _derive_status(
        self,
        stop_condition: str | None,
        evaluation: EvaluationResult | None,
        has_evaluator: bool,
        steps: Sequence[StepRecord],
    ) -> tuple[str, str | None]:
        condition = stop_condition or "unknown"
        if has_evaluator:
            if evaluation:
                status = evaluation.status
                stop_reason = evaluation.stop_reason
                if not evaluation.should_stop and status == "in_progress":
                    status = "failed"
                    stop_reason = stop_reason or "Execution stopped before gates were satisfied."
                return status, stop_reason
            if condition == "stop_iteration":
                return "success", "Action provider requested stop."
            if condition == "budget":
                return "failed", "Step budget exhausted."
            if condition == "exception":
                return "failed", "Exception raised during execution."
            return "failed", "Execution ended without evaluation."

        if condition == "stop_iteration":
            return "success", "Completed"
        if condition == "budget":
            if steps:
                return "failed", "Step budget exhausted."
            return "failed", "No actions were executed before the step budget was reached."
        if condition == "exception":
            stop_reason = (
                evaluation.stop_reason if evaluation else "Exception raised during execution."
            )
            return "failed", stop_reason
        if condition == "gates":
            stop_reason = evaluation.stop_reason if evaluation else "Completed"
            if evaluation and evaluation.status == "success" and steps:
                return "success", stop_reason
            return "failed", stop_reason

        stop_reason = evaluation.stop_reason if evaluation else None
        status = "success" if steps else "failed"
        return status, stop_reason


__all__ = ["ActionProvider", "ReactiveExecutor", "ToolInvoker"]
