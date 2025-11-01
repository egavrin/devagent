"""Budget-focused tests for the CLI ReAct executor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.cli.react.budget_control import BudgetManager
from ai_dev_agent.cli.react.executor import BudgetAwareExecutor
from ai_dev_agent.engine.react.types import Observation, TaskSpec
from ai_dev_agent.providers.llm.base import Message


class _StubSessionManager:
    """Minimal session manager stub used by budget tests."""

    def __init__(self) -> None:
        self._history: list[Message] = [Message(role="user", content="Investigate")]

    def compose(self, _session_id: str) -> list[Message]:
        return list(self._history)

    def add_assistant_message(
        self,
        _session_id: str,
        _content: str | None,
        tool_calls: Any | None = None,
    ) -> None:
        if tool_calls:
            self._history.append(Message(role="assistant", content="", tool_calls=tool_calls))


class _StubBudgetIntegration:
    """Capture execute_with_retry calls for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def execute_with_retry(self, func, *args, **kwargs):
        self.calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    def track_llm_call(self, *args, **kwargs) -> None:  # pragma: no cover - tracked indirectly
        self.calls.append(("track", args, kwargs))


class _ForcedSynthesisProvider:
    """LLM action provider stub that triggers forced synthesis."""

    def __init__(self, budget_integration: _StubBudgetIntegration) -> None:
        self.session_manager = _StubSessionManager()
        self.session_id = "budget-session"
        self.client = SimpleNamespace(model="stub-model", complete=self._complete)
        self.budget_integration = budget_integration
        self._complete_calls: int = 0

    def update_phase(self, _phase: str, *, is_final: bool = False) -> None:
        self._is_final = is_final

    def __call__(self, _task: TaskSpec, _history):
        raise StopIteration("Provider stopped without final tool call")

    def last_response_text(self) -> str:
        return ""

    def _complete(self, conversation, temperature: float = 0.1) -> str:
        self._complete_calls += 1
        assert temperature == 0.1
        assert all(isinstance(msg, Message) for msg in conversation)
        return "Forced answer."


def _build_executor(max_iterations: int = 1) -> BudgetAwareExecutor:
    manager = BudgetManager(max_iterations, adaptive_scaling=False)
    return BudgetAwareExecutor(manager)


def test_budget_executor_forced_synthesis_uses_budget_integration(monkeypatch):
    """Forced synthesis should route through the budget integration helper."""

    integration = _StubBudgetIntegration()
    provider = _ForcedSynthesisProvider(integration)

    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._sanitize_conversation_for_llm",
        lambda messages: messages,
    )

    executor = _build_executor()

    result = executor.run(
        TaskSpec(identifier="task-budget", goal="Collect final answer"),
        provider,
        MagicMock(),
    )

    assert any(
        call[0] == provider.client.complete for call in integration.calls
    ), "Forced synthesis should leverage budget integration execute_with_retry"
    assert result.steps[-1].observation.raw_output == "Forced answer."


def test_budget_executor_records_failure_patterns():
    """A failing tool observation should register with the failure detector."""

    executor = _build_executor()
    detector = MagicMock()
    detector.should_give_up.return_value = (False, "")
    executor.failure_detector = detector

    provider = MagicMock()
    provider.update_phase.side_effect = lambda *args, **kwargs: None
    provider.session_manager = _StubSessionManager()
    provider.session_id = "session-failure"
    provider.client = SimpleNamespace()
    provider.last_response_text.return_value = "Final thought"
    provider.return_value = {
        "step_id": "S1",
        "thought": "Attempt action",
        "tool": "run",
        "args": {"command": "ls"},
    }

    observation = Observation(success=False, outcome="Tool crashed", tool="run", error="Boom")

    result = executor.run(
        TaskSpec(identifier="task-failure", goal="Trigger failure"),
        provider,
        MagicMock(return_value=observation),
    )

    detector.record_failure.assert_called_once()
    assert result.status == "failed"
    assert "Tool crashed" in (result.stop_reason or "")


def test_budget_executor_handles_immediate_budget_exhaustion():
    """If the budget is already exhausted the executor should exit gracefully."""

    manager = BudgetManager(1, adaptive_scaling=False)
    manager._current = manager.max_iterations

    executor = BudgetAwareExecutor(manager)

    provider = MagicMock()
    provider.update_phase.side_effect = AssertionError("update_phase should not be called")
    provider.session_manager = _StubSessionManager()
    provider.session_id = "session-exhausted"
    provider.client = SimpleNamespace()
    provider.last_response_text.return_value = ""

    result = executor.run(
        TaskSpec(identifier="task-none", goal="Budget exceeded"),
        provider,
        MagicMock(side_effect=AssertionError("tool_invoker should not run")),
    )

    assert result.status == "failed"
    assert result.stop_reason == "No actions executed"
