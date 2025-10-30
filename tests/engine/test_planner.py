from types import SimpleNamespace

import pytest

from ai_dev_agent.engine.planning.planner import Planner, PlanningContext, PlanTask
from ai_dev_agent.providers.llm import LLMError
from ai_dev_agent.session.manager import SessionManager


class StubLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, conversation, temperature=0.1):
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


def _cleanup_sessions(created_ids):
    manager = SessionManager.get_instance()
    with manager._lock:  # type: ignore[attr-defined]
        for session_id in created_ids:
            manager._sessions.pop(session_id, None)  # type: ignore[attr-defined]


def test_planner_generates_plan_with_context(monkeypatch):
    manager = SessionManager.get_instance()
    existing_sessions = set(manager.list_sessions())

    response = """```json
    {
      "summary": "Implement feature",
      "complexity": "medium",
      "success_criteria": ["All tests green"],
      "tasks": [
        {
          "id": "T1",
          "step_number": 1,
          "title": "Plan",
          "description": "Draft approach",
          "dependencies": [],
          "deliverables": "Design doc",
          "commands": ["pytest"]
        },
        {
          "step_number": 2,
          "title": "Implement",
          "description": "Write code",
          "dependencies": ["1"],
          "deliverables": ["Code", "Tests"]
        }
      ]
    }
    ```"""
    planner = Planner(StubLLM([response]))
    context = PlanningContext(repository_metrics="Large repo", dominant_language="Python")

    result = planner.generate("Add feature", project_structure="src/\n  - app.py", context=context)
    created_sessions = set(manager.list_sessions()) - existing_sessions

    try:
        assert result.summary == "Implement feature"
        assert result.complexity == "medium"
        assert result.success_criteria == ["All tests green"]
        assert len(result.tasks) == 2
        first_task: PlanTask = result.tasks[0]
        assert first_task.step_number == 1
        assert first_task.deliverables == ["Design doc"]
        second_task = result.tasks[1]
        assert second_task.dependencies == [1]
        assert second_task.deliverables == ["Code", "Tests"]
        assert result.context_snapshot and "Repository Metrics" in result.context_snapshot
    finally:
        _cleanup_sessions(created_sessions)


def test_planner_fallback_on_llm_error():
    manager = SessionManager.get_instance()
    existing_sessions = set(manager.list_sessions())

    planner = Planner(StubLLM([LLMError("failure")]))
    result = planner.generate("Improve docs")
    created_sessions = set(manager.list_sessions()) - existing_sessions

    try:
        assert result.fallback_reason == "failure"
        assert len(result.tasks) == 3
        assert all(task.step_number for task in result.tasks)
        assert result.summary.startswith("Fallback plan")
    finally:
        _cleanup_sessions(created_sessions)


def test_planner_raises_on_invalid_json():
    manager = SessionManager.get_instance()
    existing_sessions = set(manager.list_sessions())
    planner = Planner(StubLLM(['{"summary": "a", bad}']))

    with pytest.raises(LLMError):
        planner.generate("Broken response")

    created_sessions = set(manager.list_sessions()) - existing_sessions
    _cleanup_sessions(created_sessions)
