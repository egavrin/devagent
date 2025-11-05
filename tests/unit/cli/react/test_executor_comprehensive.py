"""Comprehensive scenarios for the CLI ReAct executor behaviour."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.cli.react.budget_control import BudgetManager
from ai_dev_agent.cli.react.executor import BudgetAwareExecutor, _extract_json
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


def test_extract_json_handles_multiple_sources():
    """_extract_json should parse direct text, fenced blocks, and nested braces safely."""

    direct = ' { "answer": 42 } \n'
    assert _extract_json(direct) == {"answer": 42}

    fenced = '```json\n{\n  "status": "ok"\n}\n```'
    assert _extract_json(fenced) == {"status": "ok"}

    nested = 'Some reply {"note": "contains {braces} and [lists] inside strings"} trailing'
    assert _extract_json(nested) == {"note": "contains {braces} and [lists] inside strings"}

    escaped = 'leading {"path": "C:\\\\logs\\\\file.json", "quote": "He said \\"hi\\"."}'
    assert _extract_json(escaped) == {
        "path": "C:\\logs\\file.json",
        "quote": 'He said "hi".',
    }

    malformed_fence = "```json\n{invalid: true]\n```"
    assert _extract_json(malformed_fence) is None


def test_executor_accepts_json_response_without_forcing(monkeypatch):
    """Provider JSON inside markdown fences should bypass forced synthesis when schema set."""

    class JsonProvider:
        def __init__(self) -> None:
            self.session_manager = StubSessionManager()
            self.session_id = "session-json"
            self.calls = 0

        def update_phase(self, phase: str, *, is_final: bool = False) -> None:  # pragma: no cover
            pass

        def __call__(self, task: TaskSpec, history):
            self.calls += 1
            raise StopIteration("LLM stopped with JSON block.")

        def last_response_text(self) -> str:
            return '```json\n{"final": "done"}\n```'

    executor = BudgetAwareExecutor(
        BudgetManager(1, adaptive_scaling=False), format_schema={"type": "object"}
    )
    provider = JsonProvider()

    result = executor.run(
        TaskSpec(identifier="json", goal="Test JSON shortcut"), provider, MagicMock()
    )

    assert result.status == "success"
    assert result.steps == []
    assert result.stop_reason == "Completed"


def test_executor_repeated_failures_surface_diagnostics():
    """BudgetAwareExecutor should append failure detector guidance after repeated failures."""

    class RepeatingProvider:
        def __init__(self) -> None:
            self.session_manager = StubSessionManager()
            self.session_id = "repeat"
            self.calls = 0

        def update_phase(self, phase: str, *, is_final: bool = False) -> None:  # pragma: no cover
            pass

        def __call__(self, task: TaskSpec, history):
            self.calls += 1
            return {
                "step_id": f"S{self.calls}",
                "thought": "Retry search",
                "tool": "find",
                "args": {"query": "missing_symbol"},
            }

        def last_response_text(self) -> str:
            return ""

    class AlwaysFailInvoker:
        def __call__(self, action: ActionRequest) -> Observation:
            return Observation(
                success=False,
                outcome="Pattern not found.",
                tool=action.tool,
                error="NotFound",
            )

    executor = BudgetAwareExecutor(BudgetManager(3, adaptive_scaling=False))
    provider = RepeatingProvider()
    invoker = AlwaysFailInvoker()

    result = executor.run(
        TaskSpec(identifier="repeat", goal="Trigger diagnostics"), provider, invoker
    )

    assert result.status == "failed"
    assert len(result.steps) == 3
    final_outcome = result.steps[-1].observation.outcome or ""
    assert "Repeated Failure Detected" in final_outcome


def test_executor_forced_synthesis_empty_response(capsys):
    """Forced synthesis should report empty completions for visibility."""

    class SilentProvider:
        def __init__(self) -> None:
            self.session_manager = StubSessionManager()
            self.session_id = "silent"
            self.client = SimpleNamespace(complete=MagicMock(return_value="   "))
            self.calls = 0

        def update_phase(self, phase: str, *, is_final: bool = False) -> None:  # pragma: no cover
            pass

        def __call__(self, task: TaskSpec, history):
            self.calls += 1
            raise StopIteration("Stopped without answer.")

        def last_response_text(self) -> str:
            return ""

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    provider = SilentProvider()

    result = executor.run(
        TaskSpec(identifier="silent", goal="Force fallback"), provider, MagicMock()
    )

    err = capsys.readouterr().err
    assert "LLM returned empty response" in err
    provider.client.complete.assert_called_once()
    assert result.status == "success"
    assert result.stop_reason == "Completed"


def test_executor_forced_synthesis_with_integration(monkeypatch):
    """Forced synthesis path should route through budget integration when available."""

    class TrackingIntegration:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def execute_with_retry(self, fn, *args, **kwargs):
            self.calls.append((fn, args, kwargs))
            return fn(*args, **kwargs)

    class MixedProvider:
        def __init__(self) -> None:
            self.session_manager = StubSessionManager()
            self.session_id = "mixed"
            self.client = SimpleNamespace(complete=MagicMock(return_value='{"final": "value"}'))
            self.budget_integration = TrackingIntegration()
            self.actions = [
                {
                    "step_id": "S1",
                    "thought": "Attempt read",
                    "tool": "read",
                    "args": {"path": "missing.md"},
                }
            ]
            self.calls = 0

        def update_phase(self, phase: str, *, is_final: bool = False) -> None:  # pragma: no cover
            pass

        def __call__(self, task: TaskSpec, history):
            if self.calls < len(self.actions):
                action = self.actions[self.calls]
                self.calls += 1
                return action
            self.calls += 1
            raise StopIteration("Need to synthesize.")

        def last_response_text(self) -> str:
            return ""

    class FailThenPassInvoker:
        def __call__(self, action: ActionRequest) -> Observation:
            return Observation(success=False, outcome="read failed", tool=action.tool)

    executor = BudgetAwareExecutor(
        BudgetManager(2, adaptive_scaling=False), format_schema={"type": "object"}
    )
    provider = MixedProvider()
    invoker = FailThenPassInvoker()

    result = executor.run(TaskSpec(identifier="mixed", goal="Integration path"), provider, invoker)

    assert result.status == "success"
    assert result.steps[-1].observation.tool == "submit_final_answer"
    assert result.steps[-1].observation.raw_output == '{"final": "value"}'
    assert provider.client.complete.called
    assert provider.budget_integration.calls, "Integration should wrap the completion call"


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
