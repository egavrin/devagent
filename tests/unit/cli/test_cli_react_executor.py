from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ai_dev_agent.agents.schemas import VIOLATION_SCHEMA
from ai_dev_agent.cli.react.executor import (
    BudgetAwareExecutor,
    BudgetManager,
    _record_search_query,
    _sanitize_conversation_for_llm,
)
from ai_dev_agent.engine.react.types import ActionRequest, Observation, TaskSpec
from ai_dev_agent.providers.llm.base import Message


class _DummyActionProvider:
    def __init__(self, actions: Sequence[ActionRequest]) -> None:
        self._actions = list(actions)
        self._invocations = 0

    def update_phase(
        self, phase: str, *, is_final: bool = False
    ) -> None:  # pragma: no cover - no-op stub
        return None

    def __call__(self, task: TaskSpec, history: Sequence[Any]) -> ActionRequest:
        if self._invocations >= len(self._actions):
            raise StopIteration()
        action = self._actions[self._invocations]
        self._invocations += 1
        return action


class _DummyToolInvoker:
    def __init__(self, observations: Sequence[Observation]) -> None:
        self._observations = list(observations)
        self._invocations = 0

    def __call__(self, action: ActionRequest) -> Observation:
        observation = self._observations[self._invocations]
        self._invocations += 1
        return observation


def _make_action(tool: str = "run") -> ActionRequest:
    return ActionRequest(
        step_id="S1",
        thought="test",
        tool=tool,
        args={},
    )


def _make_observation(success: bool, outcome: str) -> Observation:
    return Observation(
        success=success,
        outcome=outcome,
        tool="run",
        metrics={"exit_code": 0 if success else 1},
    )


class _RecordingClient:
    def __init__(self, response: str | None = None) -> None:
        self.captured_conversation: Sequence[Message] | None = None
        self._response = response

    def complete(self, conversation: Sequence[Message], *, temperature: float = 0.1) -> str:
        self.captured_conversation = conversation
        if self._response is not None:
            return self._response
        return json.dumps(
            {
                "violations": [],
                "summary": {
                    "total_violations": 0,
                    "files_reviewed": 0,
                    "rule_name": "TEST_RULE",
                },
            }
        )


class _MockSessionManager:
    def __init__(self) -> None:
        self._base_conversation = [
            Message(role="system", content="base-system"),
            Message(role="user", content="base-user"),
        ]

    def compose(self, session_id: str) -> list[Message]:
        return list(self._base_conversation)


def test_sanitize_conversation_for_llm_removes_orphaned_tool_messages():
    assistant_calls = [{"id": "tool-keep"}]
    messages = [
        Message(role="assistant", content=None, tool_calls=assistant_calls),
        Message(role="tool", content="kept", tool_call_id="tool-keep"),
        Message(role="tool", content="orphan", tool_call_id="tool-drop"),
        Message(role="assistant", content="summary"),
    ]

    sanitized = _sanitize_conversation_for_llm(messages)

    assert len(sanitized) == 3
    assert all(getattr(msg, "tool_call_id", None) != "tool-drop" for msg in sanitized)


class _ForcedStopActionProvider:
    def __init__(self, session_manager: _MockSessionManager, client: _RecordingClient) -> None:
        self.session_manager = session_manager
        self.session_id = "session-1"
        self.client = client

    def update_phase(
        self, phase: str, *, is_final: bool = False
    ) -> None:  # pragma: no cover - no-op stub
        return None

    def __call__(self, task: TaskSpec, history: Sequence[Any]) -> ActionRequest:
        raise StopIteration("No tool calls - synthesis complete")


class _NoopToolInvoker:
    def __call__(self, action: ActionRequest) -> Observation:  # pragma: no cover - defensive guard
        raise AssertionError("Tool invoker should not be called during forced synthesis")


class _SessionBackedActionProvider:
    def __init__(
        self,
        actions: Sequence[ActionRequest],
        *,
        session_manager: _MockSessionManager | None = None,
        client: _RecordingClient | None = None,
        session_id: str = "session-test",
    ) -> None:
        self._actions = list(actions)
        self._index = 0
        self.session_manager = session_manager or _MockSessionManager()
        self.client = client or _RecordingClient()
        self.session_id = session_id

    def update_phase(
        self, phase: str, *, is_final: bool = False
    ) -> None:  # pragma: no cover - no-op
        return None

    def __call__(self, task: TaskSpec, history: Sequence[Any]) -> ActionRequest:
        if self._index >= len(self._actions):
            raise StopIteration()
        action = self._actions[self._index]
        self._index += 1
        return action


def test_budget_executor_marks_failure_when_last_observation_fails() -> None:
    manager = BudgetManager(1)
    executor = BudgetAwareExecutor(manager)
    action_provider = _DummyActionProvider([_make_action()])
    tool_invoker = _DummyToolInvoker([_make_observation(False, "Command exited with 1")])
    task = TaskSpec(identifier="T1", goal="Run command", category="assist")

    result = executor.run(task, action_provider, tool_invoker)

    assert result.status == "failed"
    assert "Command exited with 1" in (result.stop_reason or "")


def test_budget_executor_reports_success_after_final_success() -> None:
    manager = BudgetManager(1)
    executor = BudgetAwareExecutor(manager)
    action_provider = _DummyActionProvider([_make_action()])
    tool_invoker = _DummyToolInvoker([_make_observation(True, "Command exited with 0")])
    task = TaskSpec(identifier="T2", goal="Run command", category="assist")

    result = executor.run(task, action_provider, tool_invoker)

    assert result.status == "success"
    assert result.stop_reason == "Completed"


def test_record_search_query_prefers_query_and_pattern() -> None:
    recorded: set[str] = set()
    action = ActionRequest(
        step_id="S1",
        thought="search",
        tool="grep",
        args={"pattern": "TODO"},
    )

    _record_search_query(action, recorded)

    assert "TODO" in recorded


def test_forced_synthesis_enforces_json_schema_instructions() -> None:
    manager = BudgetManager(1)
    client = _RecordingClient()
    session_manager = _MockSessionManager()
    action_provider = _ForcedStopActionProvider(session_manager, client)
    executor = BudgetAwareExecutor(manager, format_schema=VIOLATION_SCHEMA)
    task = TaskSpec(identifier="T3", goal="Force synthesis", category="assist")

    result = executor.run(task, action_provider, _NoopToolInvoker())

    assert result.status == "success"
    assert client.captured_conversation is not None
    instructions_message = client.captured_conversation[-1]
    assert instructions_message.role == "system"
    instructions_text = instructions_message.content
    assert "CRITICAL RULES" in instructions_text
    assert json.dumps(VIOLATION_SCHEMA, indent=2) in instructions_text


def test_failure_detector_appends_should_give_up_message(monkeypatch) -> None:
    executor = BudgetAwareExecutor(BudgetManager(1))

    def fake_should_give_up(tool_name, target):
        return True, "Repeated Failure Detected"

    monkeypatch.setattr(executor.failure_detector, "should_give_up", fake_should_give_up)

    action = ActionRequest(
        step_id="S1",
        thought="Investigate",
        tool="grep",
        args={"pattern": "TODO"},
    )

    def failing_invoker(call: ActionRequest) -> Observation:
        assert call is action
        return Observation(success=False, outcome="Command failed", tool="grep")

    observation = executor._invoke_tool(failing_invoker, action)

    assert observation.success is False
    assert "Repeated Failure Detected" in (observation.outcome or "")


def test_post_loop_forced_synthesis_adds_submit_final_answer():
    manager = BudgetManager(1)
    session_manager = _MockSessionManager()
    client = _RecordingClient(response="Forced final answer")
    action_provider = _SessionBackedActionProvider(
        [
            ActionRequest(
                step_id="S1",
                thought="Investigate",
                tool="run",
                args={"cmd": "pytest"},
            )
        ],
        session_manager=session_manager,
        client=client,
    )
    observation = Observation(
        success=True,
        outcome="Executed pytest",
        tool="run",
        raw_output="ok",
        metrics={"exit_code": 0},
    )
    tool_invoker = _DummyToolInvoker([observation])
    executor = BudgetAwareExecutor(manager)
    task = TaskSpec(identifier="T-success", goal="Run tests", category="assist")

    result = executor.run(task, action_provider, tool_invoker)

    assert result.status == "success"
    last_step = result.steps[-1]
    assert last_step.action.tool == "submit_final_answer"
    assert last_step.observation.raw_output == "Forced final answer"
