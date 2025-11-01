import pytest

from ai_dev_agent.engine.planning.planner import (
    Planner,
    PlanningContext,
    PlanResult,
    PlanTask,
    _normalize_int_list,
)
from ai_dev_agent.providers.llm import LLMConnectionError, LLMError, LLMTimeoutError
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


def test_plan_task_normalizes_identifiers_and_fields():
    task_with_identifier = PlanTask(
        step_number=None,
        title="Refine scope",
        description="Clarify the deliverable",
        dependencies=["2", "not-an-int", 3, None],
        deliverables=["Annotated doc"],
        commands=["pytest tests/unit"],
        effort=5,
        reach=3,
        impact=4,
        confidence=0.85,
        risk_mitigation="Peer review the plan",
        identifier="T7",
    )
    assert task_with_identifier.step_number == 7
    assert task_with_identifier.dependencies == [2, 3]
    assert task_with_identifier.deliverables == ["Annotated doc"]
    assert task_with_identifier.commands == ["pytest tests/unit"]

    task_without_identifier = PlanTask(step_number=None, title="Bootstrap")
    assert task_without_identifier.step_number == 1
    assert task_without_identifier.identifier == "T1"

    task_data = task_with_identifier.to_dict()
    assert task_data["id"] == "T7"
    assert task_data["effort"] == 5
    assert task_data["risk_mitigation"] == "Peer review the plan"
    assert "deliverables" in task_data and task_data["deliverables"] == ["Annotated doc"]
    assert "commands" in task_data and task_data["commands"] == ["pytest tests/unit"]


def test_plan_result_to_dict_includes_optional_fields():
    task = PlanTask(step_number=1, title="Draft")
    result = PlanResult(
        goal="Ship feature",
        summary="End-to-end delivery",
        tasks=[task],
        raw_response="{}",
        fallback_reason="temporary outage",
        project_structure="src/\n  - main.py",
        context_snapshot="Repository Metrics:\nOK",
        complexity="high",
        success_criteria=["All tests green", "Docs updated"],
    )

    serialized = result.to_dict()
    assert serialized["status"] == "planned"
    assert serialized["fallback_reason"] == "temporary outage"
    assert serialized["project_structure"] == "src/\n  - main.py"
    assert serialized["context_snapshot"] == "Repository Metrics:\nOK"
    assert serialized["complexity"] == "high"
    assert serialized["success_criteria"] == ["All tests green", "Docs updated"]


def test_planner_retries_and_preserves_context(monkeypatch):
    manager = SessionManager.get_instance()
    existing_sessions = set(manager.list_sessions())

    responses = [
        LLMTimeoutError("slow model"),
        LLMConnectionError("network hiccup"),
        """{
          "summary": "Stabilize feature",
          "complexity": "low",
          "success_criteria": "Regression tests pass",
          "tasks": [
            {
              "title": "Investigate",
              "dependencies": ["1", "invalid"],
              "deliverables": "Report",
              "commands": "pytest -k regression"
            }
          ]
        }""",
    ]
    planner = Planner(StubLLM(responses))

    time_values = iter([1000.0, 1012.0, 1025.0])

    def fake_time():
        return next(time_values, 1035.0)

    monkeypatch.setattr("ai_dev_agent.engine.planning.planner.time.time", fake_time)

    result = planner.generate("Improve stability")
    created_sessions = set(manager.list_sessions()) - existing_sessions

    try:
        assert result.summary == "Stabilize feature"
        assert result.complexity == "low"
        assert result.success_criteria == ["Regression tests pass"]
        assert result.tasks[0].dependencies == [1]
        assert result.tasks[0].deliverables == ["Report"]
        assert result.tasks[0].commands == ["pytest -k regression"]
        assert result.context_snapshot and "Primary Language" in result.context_snapshot
    finally:
        _cleanup_sessions(created_sessions)


def test_normalize_int_list_handles_various_inputs():
    assert _normalize_int_list(None) == []
    assert _normalize_int_list(4) == [4]
    assert _normalize_int_list("5") == [5]
    assert _normalize_int_list(["1", "oops", 2]) == [1, 2]
    assert _normalize_int_list(5.5) == []
