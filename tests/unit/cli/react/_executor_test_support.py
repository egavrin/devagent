"""Shared stubs used across executor tests."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from types import SimpleNamespace
from typing import Any

from ai_dev_agent.engine.react.types import (
    ActionRequest,
    EvaluationResult,
    MetricsSnapshot,
    Observation,
    RunResult,
    StepRecord,
)
from ai_dev_agent.providers.llm.base import Message


class StubSession:
    def __init__(self) -> None:
        self.lock = RLock()
        self.history: list[Message] = []
        self.system_messages: list[Message] = []
        self.metadata: dict[str, Any] = {}


class StubSessionManager:
    def __init__(self) -> None:
        self._session = StubSession()

    def ensure_session(self, session_id: str, system_messages=None, metadata=None):
        if system_messages:
            with self._session.lock:
                self._session.system_messages = list(system_messages)
        if metadata:
            with self._session.lock:
                self._session.metadata.update(metadata)
        return self._session

    def get_session(self, session_id: str) -> StubSession:
        return self._session

    def compose(self, session_id: str) -> list[Message]:
        with self._session.lock:
            return [*self._session.system_messages, *self._session.history]

    def extend_history(self, session_id: str, messages):
        with self._session.lock:
            self._session.history.extend(messages)

    def remove_system_messages(self, session_id: str, predicate):
        with self._session.lock:
            self._session.system_messages = [
                msg for msg in self._session.system_messages if not predicate(msg)
            ]
            self._session.history = [
                msg
                for msg in self._session.history
                if not (msg.role == "system" and predicate(msg))
            ]

    def add_system_message(self, session_id: str, content: str, *, location: str = "history"):
        message = Message(role="system", content=content)
        with self._session.lock:
            if location == "system":
                self._session.system_messages.append(message)
            else:
                self._session.history.append(message)
        return message

    def add_assistant_message(self, session_id: str, content: str | None, tool_calls=None):
        message = Message(role="assistant", content=content, tool_calls=tool_calls)
        with self._session.lock:
            self._session.history.append(message)
        return message


class StubContextSynthesizer:
    def synthesize_previous_steps(self, *_args, **_kwargs) -> str:
        return "Prior context summary"

    def get_redundant_operations(self, *_args, **_kwargs):
        return []

    def build_constraints_section(self, *_args, **_kwargs) -> str:
        return ""


class StubDynamicContextTracker:
    def __init__(self, _repo_root: Any) -> None:
        self.updated_steps: list[int] = []

    def update_from_step(self, record: StepRecord) -> None:
        self.updated_steps.append(record.step_index)

    def should_refresh_repomap(self) -> bool:
        return False

    def get_context_summary(self) -> dict[str, Any]:
        return {"files": [], "symbols": [], "total_mentions": 0}


@dataclass
class _RecordedRun:
    search_queries: set[str]


class StubExecutor:
    def __init__(self, _budget_manager, format_schema=None, skip_intermediate_synthesis=False):
        self.format_schema = format_schema
        self.skip_intermediate_synthesis = skip_intermediate_synthesis
        self._record = _RecordedRun(set())

    def run(self, task, action_provider, tool_invoker, iteration_hook=None, step_hook=None):
        find_step = StepRecord(
            action=ActionRequest(
                step_id="S1",
                thought="Search repository",
                tool="find",
                args={"query": "pytest"},
            ),
            observation=Observation(success=True, outcome="Found matches", tool="find"),
            metrics=MetricsSnapshot.model_validate({}),
            evaluation=EvaluationResult(
                gates={},
                required_gates={},
                should_stop=False,
                stop_reason=None,
                status="in_progress",
            ),
            step_index=1,
        )
        answer_step = StepRecord(
            action=ActionRequest(
                step_id="S2",
                thought="Provide answer",
                tool="submit_final_answer",
                args={"answer": "All good."},
            ),
            observation=Observation(
                success=True,
                outcome="Answer sent",
                tool="submit_final_answer",
                raw_output="Execution finished",
            ),
            metrics=MetricsSnapshot.model_validate({"tokens_used": 42}),
            evaluation=EvaluationResult(
                gates={},
                required_gates={},
                should_stop=True,
                stop_reason="Completed",
                status="success",
            ),
            step_index=2,
        )

        if iteration_hook:
            iteration_hook(
                SimpleNamespace(phase="exploration", is_final=False, number=1, total=2), []
            )
            iteration_hook(
                SimpleNamespace(phase="synthesis", is_final=True, number=2, total=2), [find_step]
            )
        if step_hook:
            step_hook(find_step, SimpleNamespace(phase="exploration", is_final=False))
            step_hook(answer_step, SimpleNamespace(phase="synthesis", is_final=True))

        return RunResult(
            task_id=task.identifier,
            status="success",
            steps=[find_step, answer_step],
            gates={},
            stop_reason="Completed",
            metrics={"tokens_used": 42},
        )


class DummyRouter:
    def __init__(self, *_args, **_kwargs) -> None:
        self.session_id = "router-session"
        self.tools = [{"name": "find"}]

    def route(self, prompt: str):
        return SimpleNamespace(tool="submit_final_answer", arguments={"text": "done"})
