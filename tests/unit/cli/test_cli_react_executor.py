"""Focused tests for the CLI ReAct executor helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.cli.react.budget_control import (
    BudgetManager,
    create_text_only_tool,
    get_tools_for_iteration,
)
from ai_dev_agent.cli.react.executor import BudgetAwareExecutor
from ai_dev_agent.engine.react.types import ActionRequest, Observation, TaskSpec


def test_budget_manager_iteration_progression():
    """Budget manager should yield phases until the budget is exhausted."""
    manager = BudgetManager(3, adaptive_scaling=False)

    contexts = []
    for _ in range(3):
        ctx = manager.next_iteration()
        assert ctx is not None
        contexts.append(ctx)

    assert contexts[-1].is_final is True
    assert manager.next_iteration() is None


def test_get_tools_for_iteration_returns_text_only_on_final():
    """Final iterations should restrict tools to the text-only submission."""
    final_context = manager_context(is_final=True, is_penultimate=False, phase="synthesis")
    available = [
        {"type": "function", "function": {"name": "run", "description": "Execute shell"}},
        {"type": "function", "function": {"name": "read", "description": "Read file"}},
    ]

    selected = get_tools_for_iteration(final_context, available)
    assert selected == [create_text_only_tool()]


def manager_context(
    *,
    is_final: bool,
    is_penultimate: bool,
    phase: str,
    number: int = 1,
    total: int = 1,
) -> any:
    """Helper to create an IterationContext without importing the class directly."""
    return type(
        "IterationContext",
        (),
        {
            "number": number,
            "total": total,
            "remaining": 0 if is_final else max(total - number, 0),
            "percent_complete": 100.0 if is_final else (number / total) * 100,
            "phase": phase,
            "is_final": is_final,
            "is_penultimate": is_penultimate,
            "reflection_count": 0,
            "reflection_allowed": False,
        },
    )()


class StubInvoker:
    """Simple tool invoker returning a successful observation."""

    def __call__(self, action: ActionRequest) -> Observation:
        return Observation(success=True, outcome="ok", tool=action.tool, raw_output="ok")


class StubActionProvider:
    """Minimal action provider implementation for driving the executor."""

    def __init__(self):
        self.session_manager = StubSessionManager()
        self.session_id = "session-1"
        self.client = MagicMock()
        self.client.complete.return_value = "All done"
        self.phase_updates: list[tuple[str, bool]] = []
        self.calls = 0

    def update_phase(self, phase: str, *, is_final: bool = False) -> None:
        self.phase_updates.append((phase, is_final))

    def __call__(self, task: TaskSpec, history):
        self.calls += 1
        return {
            "thought": "complete",
            "tool": "submit_final_answer",
            "args": {"answer": "All done"},
        }

    def last_response_text(self) -> str:
        return "All done"


class StubSession:
    """Simple session container mimicking SessionState."""

    def __init__(self):
        self.history = []
        self.metadata = {}


class StubSessionManager:
    """Minimal session manager that stores a single session."""

    def __init__(self):
        self._session = StubSession()

    def ensure_session(self, session_id, system_messages, metadata):
        self._session.history.extend(system_messages)
        self._session.metadata.update(metadata)
        return self._session

    def extend_history(self, session_id, messages):
        self._session.history.extend(messages)

    def compose(self, session_id):
        return list(self._session.history)

    def remove_system_messages(self, session_id, predicate):
        self._session.history = [msg for msg in self._session.history if not predicate(msg)]

    def get_session(self, session_id):
        return self._session

    def add_system_message(self, session_id, message, location="system"):
        self._session.history.append(message)


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_budget_aware_executor_runs_single_iteration(mock_synth, mock_session_manager):
    """BudgetAwareExecutor should honour the iteration budget and produce a RunResult."""
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda history, current_step, **_: ""
    )

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProvider()
    tool_invoker = StubInvoker()

    task = TaskSpec(identifier="task-123", goal="Do the thing")
    result = executor.run(task, action_provider, tool_invoker)

    assert result.status in {"success", "completed"}
    assert result.stop_reason.lower() in {"final_iteration", "provider_stop", "completed"}
    assert action_provider.calls == 1


@patch("ai_dev_agent.cli.react.executor.SessionManager")
def test_budget_aware_executor_handles_tool_exceptions(mock_session_manager):
    """BudgetAwareExecutor._invoke_tool should convert tool exceptions to failed observations."""
    mock_session_manager.get_instance.return_value = StubSessionManager()

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    failing_invoker = MagicMock(side_effect=RuntimeError("boom"))
    action = ActionRequest(step_id="S1", thought="test", tool="run", args={})

    observation = executor._invoke_tool(failing_invoker, action)

    assert observation.success is False
    failure_text = (observation.error or observation.outcome or "").lower()
    assert "boom" in failure_text or "exception" in failure_text


class StubActionProviderWithStopIteration:
    """Action provider that returns text without tool calls, triggering StopIteration."""

    def __init__(self, response_text: str = "The answer is 42 lines."):
        self.session_manager = StubSessionManager()
        self.session_id = "session-1"
        self.client = MagicMock()
        self.phase_updates: list[tuple[str, bool]] = []
        self.calls = 0
        self._response_text = response_text

    def update_phase(self, phase: str, *, is_final: bool = False) -> None:
        self.phase_updates.append((phase, is_final))

    def __call__(self, task: TaskSpec, history):
        self.calls += 1
        # Simulate the LLM returning text without tool calls
        raise StopIteration("No tool calls - synthesis complete")

    def last_response_text(self) -> str:
        # Return the text response that was already generated
        return self._response_text


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_uses_existing_response_when_llm_stops_without_tools(
    mock_synth, mock_session_manager, capsys
):
    """When LLM raises StopIteration with existing text response, should use that instead of forcing synthesis."""
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda history, current_step, **_: "",
        get_redundant_operations=lambda history: [],
        build_constraints_section=lambda redundant_ops: "",
    )

    response_text = "The file contains 142 lines of code."
    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProviderWithStopIteration(response_text)
    tool_invoker = StubInvoker()

    task = TaskSpec(identifier="task-123", goal="Count lines in file")
    result = executor.run(task, action_provider, tool_invoker)

    # Should complete successfully without calling client.complete() for forced synthesis
    assert result.status == "success"
    assert result.stop_reason in {"provider_stop", "Completed"}

    # Should not have attempted forced synthesis (which would print an error)
    captured = capsys.readouterr()
    assert "ERROR: LLM returned empty response" not in captured.err
    assert "ERROR: Failed to force synthesis" not in captured.err

    # Verify the action provider's client.complete was not called
    # (since we should use the existing response instead)
    action_provider.client.complete.assert_not_called()
