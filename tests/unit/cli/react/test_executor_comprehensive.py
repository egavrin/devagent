"""Comprehensive scenarios for the CLI ReAct executor behaviour."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.cli.react.budget_control import BudgetManager
from ai_dev_agent.cli.react.executor import BudgetAwareExecutor
from ai_dev_agent.engine.react.types import (
    ActionRequest,
    EvaluationResult,
    MetricsSnapshot,
    Observation,
    StepRecord,
    TaskSpec,
    ToolResult,
)
from ai_dev_agent.providers.llm.base import Message


class StubSession:
    """Simple session container mimicking SessionState."""

    def __init__(self) -> None:
        self.history: list[Message] = []
        self.metadata: dict[str, Any] = {}


class StubSessionManager:
    """Minimal session manager for executor tests."""

    def __init__(self) -> None:
        self._session = StubSession()

    def ensure_session(self, session_id, system_messages=None, metadata=None):
        if system_messages:
            self._session.history.extend(system_messages)
        if metadata:
            self._session.metadata.update(metadata)
        return self._session

    def extend_history(self, session_id, messages):
        self._session.history.extend(messages)

    def compose(self, session_id):
        return list(self._session.history)

    def add_system_message(self, session_id, message, location="system"):
        self._session.history.append(message)

    def get_session(self, session_id):
        return self._session


class InvokeToolsResult:
    """Return object used by the mocked invoke_tools path."""

    def __init__(self, message_content: str) -> None:
        self.message_content = message_content

    def __str__(self) -> str:
        return self.message_content


def build_metrics(**values: Any) -> MetricsSnapshot:
    """Convenience helper for constructing metrics snapshots."""
    return MetricsSnapshot.model_validate(values)


def build_evaluation(stop_reason: str | None = None, status: str = "success") -> EvaluationResult:
    return EvaluationResult(
        gates={},
        required_gates={},
        should_stop=bool(stop_reason),
        stop_reason=stop_reason,
        next_action_hint=None,
        improved_metrics={},
        status=status,
    )


def test_executor_error_recovery():
    """Tool failure followed by forced synthesis should complete via invoke_tools fallback."""

    class RecoveringProvider:
        def __init__(self):
            self.session_manager = StubSessionManager()
            self.session_id = "session-err"
            system_msg = Message(role="system", content="Diagnostics")
            self.session_manager.ensure_session(self.session_id, [system_msg], {})
            self.client = SimpleNamespace(
                invoke_tools=MagicMock(return_value=InvokeToolsResult("Recovered answer."))
            )
            self.calls = 0

        def update_phase(self, phase: str, *, is_final: bool = False) -> None:
            pass

        def __call__(self, task: TaskSpec, history):
            if self.calls == 0:
                self.calls += 1
                return {
                    "step_id": "S1",
                    "thought": "Attempt read",
                    "tool": "read",
                    "args": {"path": "README.md"},
                }
            self.calls += 1
            raise StopIteration("LLM stopped without final tool call.")

        def last_response_text(self) -> str:
            return ""

    class FailingInvoker:
        def __call__(self, action: ActionRequest) -> Observation:
            return Observation(
                success=False,
                outcome="Tool execution timed out.",
                tool=action.tool,
                error="TimeoutError",
            )

    executor = BudgetAwareExecutor(BudgetManager(2, adaptive_scaling=False))
    action_provider = RecoveringProvider()
    tool_invoker = FailingInvoker()

    task = TaskSpec(identifier="task-err", goal="Handle failure")
    result = executor.run(task, action_provider, tool_invoker)

    assert result.status == "success"
    assert result.stop_reason in {"provider_stop", "Forced synthesis"}
    assert action_provider.client.invoke_tools.called

    first_step, forced_step = result.steps
    assert first_step.observation.success is False
    assert "timed out" in (first_step.observation.outcome or "").lower()
    assert forced_step.observation.tool == "submit_final_answer"
    assert forced_step.observation.raw_output == "Recovered answer."


def test_executor_multi_step_reasoning():
    """Complex multi-step execution should track history and phase hooks."""

    class SequencedProvider:
        def __init__(self):
            self.session_manager = StubSessionManager()
            self.session_id = "session-multi"
            self.client = MagicMock()
            self.actions = [
                {
                    "step_id": "S1",
                    "thought": "Survey repository",
                    "tool": "find",
                    "args": {"query": "tests/"},
                },
                {
                    "step_id": "S2",
                    "thought": "Inspect file",
                    "tool": "read",
                    "args": {"path": "README.md"},
                },
                {
                    "step_id": "S3",
                    "thought": "Provide summary",
                    "tool": "submit_final_answer",
                    "args": {"answer": "Summary complete."},
                },
            ]
            self.calls = 0
            self.history_lengths: list[int] = []

        def update_phase(self, phase: str, *, is_final: bool = False) -> None:
            pass

        def __call__(self, task: TaskSpec, history):
            self.history_lengths.append(len(history))
            action = self.actions[self.calls]
            self.calls += 1
            return action

        def last_response_text(self) -> str:
            return "Summary complete."

    class SuccessInvoker:
        def __call__(self, action: ActionRequest) -> Observation:
            return Observation(
                success=True,
                outcome=f"{action.tool} executed",
                tool=action.tool,
                metrics={"tokens_used": action.step_id.count("S")},
            )

    executor = BudgetAwareExecutor(BudgetManager(3, adaptive_scaling=False))
    action_provider = SequencedProvider()
    tool_invoker = SuccessInvoker()

    iteration_phases: list[tuple[int, str, bool]] = []

    def capture_iteration(context, history):
        iteration_phases.append((context.number, context.phase, context.is_final))

    task = TaskSpec(identifier="task-multi", goal="Run multi-step plan")
    result = executor.run(
        task,
        action_provider,
        tool_invoker,
        iteration_hook=capture_iteration,
    )

    assert action_provider.history_lengths == [0, 1, 2]
    assert len(iteration_phases) == 3
    assert iteration_phases[-1][2] is True  # last iteration is final

    assert [step.observation.tool for step in result.steps] == [
        "find",
        "read",
        "submit_final_answer",
    ]
    assert result.status == "success"
    assert result.steps[-1].observation.raw_output is None


@pytest.mark.parametrize("scenario", ["success", "max_iterations", "user_cancel"])
def test_executor_termination_conditions(scenario):
    """Validate executor behaviour across different termination scenarios."""

    if scenario == "success":
        executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
        action_provider = MagicMock(
            return_value={
                "step_id": "S1",
                "thought": "Finish up",
                "tool": "submit_final_answer",
                "args": {"answer": "Done."},
            }
        )
        action_provider.session_manager = StubSessionManager()
        action_provider.session_id = "session-done"
        action_provider.client = MagicMock()
        action_provider.last_response_text.return_value = "Done."

        tool_invoker = MagicMock(
            return_value=Observation(success=True, outcome="ok", tool="submit_final_answer")
        )
        task = TaskSpec(identifier="task-done", goal="Complete")
        result = executor.run(task, action_provider, tool_invoker)

        assert result.status == "success"
        assert result.stop_reason in {"final_iteration", "Completed"}

    elif scenario == "max_iterations":
        manager = BudgetManager(1, adaptive_scaling=False)
        manager._current = manager.max_iterations  # Force exhaustion before run
        executor = BudgetAwareExecutor(manager)
        action_provider = MagicMock(side_effect=AssertionError("Should not be called"))
        tool_invoker = MagicMock(side_effect=AssertionError("Should not be called"))

        task = TaskSpec(identifier="task-budget", goal="No actions")
        result = executor.run(task, action_provider, tool_invoker)

        assert result.status == "failed"
        assert result.stop_reason == "No actions executed"

    else:  # scenario == "user_cancel"
        executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
        action_provider = MagicMock(side_effect=KeyboardInterrupt("Cancelled by user"))
        action_provider.session_manager = StubSessionManager()
        action_provider.session_id = "session-cancel"
        action_provider.client = MagicMock()

        with pytest.raises(KeyboardInterrupt):
            executor.run(
                TaskSpec(identifier="task-cancel", goal="Abort"),
                action_provider,
                MagicMock(),
            )


def test_executor_state_management():
    """Ensure prior steps and batch results are preserved across execution."""

    prior_action = ActionRequest(
        step_id="S1",
        thought="Initial analysis",
        tool="find",
        args={"query": "TODO"},
    )
    prior_observation = Observation(
        success=True,
        outcome="Seed context",
        tool="find",
        metrics={"tokens_used": 1},
    )
    prior_record = StepRecord(
        action=prior_action,
        observation=prior_observation,
        metrics=build_metrics(tokens_used=1),
        evaluation=build_evaluation(),
        step_index=1,
    )

    class StatefulProvider:
        def __init__(self):
            self.session_manager = StubSessionManager()
            self.session_id = "session-state"
            self.client = MagicMock()
            self.calls = 0
            self.history_lengths: list[int] = []

        def update_phase(self, phase: str, *, is_final: bool = False) -> None:
            pass

        def __call__(self, task: TaskSpec, history):
            self.history_lengths.append(len(history))
            if self.calls == 0:
                self.calls += 1
                return {
                    "step_id": "S2",
                    "thought": "Run batch tools",
                    "tool": "batch",
                    "tool_calls": [
                        {"tool": "find", "args": {"query": "tests/"}},
                        {"tool": "grep", "args": {"pattern": "TODO"}},
                    ],
                }
            self.calls += 1
            return {
                "step_id": "S3",
                "thought": "Submit final response",
                "tool": "submit_final_answer",
                "args": {"answer": '{"status": "cached"}'},
            }

        def last_response_text(self) -> str:
            return '{"status": "cached"}'

    class StatefulInvoker:
        def __init__(self):
            self.calls = 0

        def __call__(self, action: ActionRequest) -> Observation:
            self.calls += 1
            if action.tool == "batch":
                return Observation(
                    success=True,
                    outcome="Executed 2 tool(s): 2 succeeded",
                    tool="batch[2]",
                    metrics={"tokens_used": 2},
                    results=[
                        ToolResult(tool="find", success=True, outcome="Found tests", error=None),
                        ToolResult(
                            tool="grep", success=True, outcome="Matches located", error=None
                        ),
                    ],
                )
            return Observation(
                success=True,
                outcome="Final answer submitted",
                tool="submit_final_answer",
                raw_output='{"status": "cached"}',
                metrics={"tokens_used": 5},
            )

    executor = BudgetAwareExecutor(BudgetManager(2, adaptive_scaling=False))
    action_provider = StatefulProvider()
    tool_invoker = StatefulInvoker()

    task = TaskSpec(identifier="task-state", goal="Preserve state")
    result = executor.run(
        task,
        action_provider,
        tool_invoker,
        prior_steps=[prior_record],
    )

    assert action_provider.history_lengths == [1, 2]
    assert len(result.steps) == 3
    assert result.steps[0].action.step_id == "S1"
    assert result.steps[1].observation.tool == "batch[2]"
    assert len(result.steps[1].observation.results) == 2

    final_output = result.steps[-1].observation.raw_output
    assert final_output == '{"status": "cached"}'
    assert json.loads(final_output) == {"status": "cached"}

    assert result.metrics["tokens_used"] == 5
