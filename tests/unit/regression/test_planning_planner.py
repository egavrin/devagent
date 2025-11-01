"""Tests for the modern planner module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import pytest

from ai_dev_agent.engine.planning.planner import (
    Planner,
    PlanningContext,
    PlanResult,
    PlanTask,
    _normalize_int_list,
)
from ai_dev_agent.providers.llm.base import LLMError, Message


class DummyLock:
    """Simple re-entrant lock used by the stub session."""

    def __enter__(self) -> "DummyLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@dataclass
class StubSession:
    """Minimal session object used by tests."""

    lock: DummyLock = field(default_factory=DummyLock)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[Message] = field(default_factory=list)
    system_messages: list[Message] = field(default_factory=list)


class StubSessionManager:
    """SessionManager double capturing interactions."""

    def __init__(self) -> None:
        self.session = StubSession()
        self.user_messages: list[str] = []
        self.assistant_messages: list[str] = []
        self.system_messages: list[str] = []

    def ensure_session(
        self,
        session_id: str | None,
        *,
        system_messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StubSession:
        if system_messages is not None:
            self.session.system_messages = list(system_messages)
        if metadata:
            self.session.metadata.update(metadata)
        return self.session

    def add_user_message(self, session_id: str, content: str) -> Message:
        self.user_messages.append(content)
        message = Message(role="user", content=content)
        self.session.history.append(message)
        return message

    def compose(self, session_id: str) -> list[Message]:
        return [*self.session.system_messages, *self.session.history]

    def add_assistant_message(
        self,
        session_id: str,
        content: str | None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> Message:
        text = content or ""
        self.assistant_messages.append(text)
        message = Message(role="assistant", content=text, tool_calls=tool_calls)
        self.session.history.append(message)
        return message

    def add_system_message(
        self, session_id: str, content: str, *, location: str = "history"
    ) -> Message:
        self.system_messages.append(content)
        message = Message(role="system", content=content)
        self.session.history.append(message)
        return message


class FakeLLM:
    """LLM stub returning pre-recorded responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], temperature: float = 0.1) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise LLMError("no responses configured")
        return self.responses.pop(0)


@pytest.fixture()
def stub_manager(monkeypatch: pytest.MonkeyPatch) -> StubSessionManager:
    """Provide a stubbed SessionManager."""
    manager = StubSessionManager()
    monkeypatch.setattr(
        "ai_dev_agent.engine.planning.planner.SessionManager.get_instance",
        classmethod(lambda cls: manager),
    )
    monkeypatch.setattr(
        "ai_dev_agent.engine.planning.planner.build_system_messages",
        lambda include_react_guidance, extra_messages, workspace_root: [
            Message(role="system", content="system"),
            *extra_messages,
        ],
    )
    return manager


def test_plan_task_normalizes_dependencies_and_fields() -> None:
    """PlanTask should convert identifiers and normalize optional fields."""
    task = PlanTask(
        step_number=None,
        title="Implement feature",
        description="Do the thing",
        dependencies=["2", 3, "foo"],
        deliverables=("report.md",),
        commands=("make test",),
        identifier=None,
        risk_mitigation=None,
    )

    assert task.step_number == 1
    assert task.dependencies == [2, 3]
    assert task.deliverables == ["report.md"]
    assert task.commands == ["make test"]
    assert task.identifier.startswith("T")


def test_planning_context_generates_prompt_block() -> None:
    """PlanningContext produces a multi-section prompt for the LLM."""
    context = PlanningContext(
        project_structure="src/\n tests/",
        repository_metrics="LOC: 1234",
        dominant_language="Python",
    )
    block = context.as_prompt_block()
    assert "Repository Metrics" in block
    assert "Python" in block
    assert "Project Structure Outline" in block


def test_planner_generate_returns_plan(
    monkeypatch: pytest.MonkeyPatch, stub_manager: StubSessionManager
) -> None:
    """Planner.generate should parse JSON payloads and produce PlanResult."""
    payload = """
    Here is your plan:
    ```json
    {
        "summary": "Tidy summary",
        "complexity": "medium",
        "tasks": [
            {"id": "T1", "title": "Step One", "description": "Do work", "dependencies": [], "commands": ["make lint"]},
            {"id": "T2", "title": "Step Two", "description": "Verify", "dependencies": [1]}
        ],
        "success_criteria": ["All tests green"]
    }
    ```
    """
    planner = Planner(FakeLLM([payload]))

    result = planner.generate("Ship feature X", project_structure="src/")

    assert isinstance(result, PlanResult)
    assert result.summary == "Tidy summary"
    assert len(result.tasks) == 2
    assert result.tasks[0].commands == ["make lint"]
    assert result.tasks[1].dependencies == [1]
    assert result.complexity == "medium"
    assert result.success_criteria == ["All tests green"]
    assert stub_manager.user_messages  # ensure session interaction occurred


def test_planner_generate_fallback_on_llm_error(
    monkeypatch: pytest.MonkeyPatch, stub_manager: StubSessionManager
) -> None:
    """Planner should produce a fallback plan when the LLM fails."""

    class FailingLLM(FakeLLM):
        def complete(self, messages: list[Message], temperature: float = 0.1) -> str:
            raise LLMError("timeout")

    planner = Planner(FailingLLM([]))

    result = planner.generate("Investigate outage")

    assert result.fallback_reason == "timeout"
    assert len(result.tasks) == 3
    assert all(isinstance(task, PlanTask) for task in result.tasks)
    assert stub_manager.system_messages, "Fallback should record system notice"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, []),
        (1, [1]),
        ("3", [3]),
        (["1", 2, "bad"], [1, 2]),
        ("not-a-number", []),
    ],
)
def test_normalize_int_list(value: Any, expected: list[int]) -> None:
    """Utility should coerce a variety of inputs into integer lists."""
    assert _normalize_int_list(value) == expected
