from pathlib import Path

from ai_dev_agent.cli import update_task_state
from ai_dev_agent.core.utils.state import StateStore
from ai_dev_agent.engine.planning.reasoning import TaskReasoning, ToolUse


def test_task_reasoning_serializes_steps_and_adjustments():
    reasoning = TaskReasoning(task_id="T1", goal="Ship feature", task_title="Implement logic")
    step = reasoning.start_step(
        "Investigate context",
        "Inspect repository structure",
        tool=ToolUse(name="rg", command="rg pattern"),
    )
    step.complete("Context gathered")
    reasoning.record_adjustment("Add verification", "Include regression test for new logic")

    snapshot = reasoning.to_dict()
    assert snapshot["task_id"] == "T1"
    assert snapshot["steps"][0]["title"] == "Investigate context"
    assert snapshot["adjustments"][0]["summary"] == "Add verification"

    plan = {"adjustments": []}
    task = {}
    reasoning.apply_to_task(task)
    reasoning.merge_into_plan(plan)
    assert task["reasoning_log"][0]["status"] == "completed"
    assert plan["adjustments"], "Expected adjustment to be recorded"

    # Merging again should not duplicate adjustments
    reasoning.merge_into_plan(plan)
    assert len(plan["adjustments"]) == 1


def test_update_task_state_persists_reasoning(tmp_path):
    state_path = Path(tmp_path) / "state.json"
    store = StateStore(state_path)

    plan = {
        "goal": "Improve feature",
        "status": "planned",
        "tasks": [{"id": "T1", "title": "Do work", "status": "pending"}],
    }
    store.save({"last_plan": plan, "last_updated": "0"})

    task = plan["tasks"][0]
    reasoning = TaskReasoning(task_id="T1", goal="Improve feature", task_title="Do work")
    step = reasoning.start_step("Execute step", "Apply change")
    step.complete("Done")
    reasoning.record_adjustment("Review follow-up", "Confirm documentation updates")

    update_task_state(store, plan, task, {"status": "completed"}, reasoning=reasoning)

    persisted = store.load()
    updated_task = persisted["last_plan"]["tasks"][0]
    assert updated_task["status"] == "completed"
    assert updated_task["reasoning_log"][0]["title"] == "Execute step"
    assert persisted["last_plan"]["status"] == "completed"
    assert persisted["last_plan"]["adjustments"][0]["summary"] == "Review follow-up"
