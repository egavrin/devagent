from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ai_dev_agent.core.utils.state import InMemoryStateStore, PlanSession, StateStore


def _assert_session_dict(session_dict: dict[str, Any], session: PlanSession) -> None:
    assert session_dict["session_id"] == session.session_id
    assert session_dict["goal"] == session.goal
    assert "created_at" in session_dict
    assert "updated_at" in session_dict


def test_plan_session_serialization_round_trip() -> None:
    created_at = datetime.utcnow().isoformat()
    updated_at = datetime.utcnow().isoformat()
    session = PlanSession(
        session_id="session-1",
        goal="Ship rock-solid features",
        status="active",
        current_task_id="T123",
        created_at=created_at,
        updated_at=updated_at,
    )

    snapshot = session.to_dict()
    restored = PlanSession.from_dict(snapshot)

    assert restored.session_id == "session-1"
    assert restored.goal == "Ship rock-solid features"
    assert restored.status == "active"
    assert restored.current_task_id == "T123"
    assert restored.created_at == created_at
    assert restored.updated_at == updated_at


def test_in_memory_session_lifecycle_updates_history() -> None:
    store = InMemoryStateStore()

    session = store.start_session("Refine planner", session_id="session-1")
    assert store.can_resume() is True
    assert store.get_current_session() == session

    previous_updated_at = session.updated_at
    updated_session = store.update_session(status="paused", current_task_id="T42")
    assert updated_session is not None
    assert updated_session.status == "paused"
    assert updated_session.current_task_id == "T42"
    assert updated_session.updated_at != previous_updated_at

    store.end_session(status="interrupted")
    assert store.can_resume() is False

    history = store.load()["session_history"]
    assert len(history) == 1
    _assert_session_dict(history[0], updated_session)
    assert history[0]["status"] == "interrupted"


def test_state_store_persists_and_recovers(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    store = StateStore(state_path)

    session = store.start_session("Persist planner state", session_id="persisted-session")
    store.update_session(status="active", current_task_id="T1")

    plan = {
        "goal": "Persist planner state",
        "status": "in_progress",
        "tasks": [{"id": "T1", "title": "Schedule agents", "status": "pending"}],
    }
    snapshot = store.load()
    snapshot["last_plan"] = plan
    store.save(snapshot)

    restored_store = StateStore(state_path)
    reloaded = restored_store.load()

    current_session = restored_store.get_current_session()
    assert current_session is not None
    assert current_session.session_id == session.session_id
    assert current_session.status == "active"
    assert reloaded["last_plan"]["tasks"][0]["title"] == "Schedule agents"

    # Corrupt the on-disk state and ensure defaults are recreated
    state_path.write_text("{ this is not valid json }", encoding="utf-8")
    recovered_store = StateStore(state_path)
    recovered_data = recovered_store.load()
    assert recovered_data["version"] == "1.0"
    assert recovered_data["current_session"] is None
    assert recovered_store.get_current_session() is None


def test_state_store_handles_concurrent_history_and_metrics(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    store = StateStore(state_path)

    history_updates = 120
    metrics_updates = 80

    def append_history(index: int) -> None:
        store.append_history({"index": index}, limit=None)

    def record_metric(index: int) -> None:
        store.record_metric({"metric": index}, limit=None)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = []
        for idx in range(history_updates):
            futures.append(pool.submit(append_history, idx))
        for idx in range(metrics_updates):
            futures.append(pool.submit(record_metric, idx))

        for future in futures:
            future.result()

    data = store.load()
    history_indices = {entry["index"] for entry in data["command_history"]}
    metric_indices = {entry["metric"] for entry in data["metrics"]}

    assert len(history_indices) == history_updates
    assert len(metric_indices) == metrics_updates
    assert history_indices == set(range(history_updates))
    assert metric_indices == set(range(metrics_updates))


def test_in_memory_store_logs_ignored_state_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    ignored_path = tmp_path / "ignored.json"
    store = InMemoryStateStore(state_file=ignored_path)
    assert store.state_file == ignored_path
    assert "will be ignored" in caplog.text


def test_start_session_generates_identifier() -> None:
    store = InMemoryStateStore()
    session = store.start_session("Auto generated session")
    assert session.session_id.startswith("session_")


def test_update_session_without_active_session(caplog: pytest.LogCaptureFixture) -> None:
    store = InMemoryStateStore()
    caplog.set_level("WARNING")
    result = store.update_session(status="paused")
    assert result is None
    assert "No active session to update" in caplog.text


def test_end_session_without_active_session() -> None:
    store = InMemoryStateStore()
    store.end_session(status="completed")  # Should be a no-op without raising


def test_end_session_replaces_matching_history_only() -> None:
    store = InMemoryStateStore()
    session = store.start_session("Track replacement", session_id="session-main")
    store.update(
        current_session=session.to_dict(),
        session_history=[
            {"session_id": "session-old", "goal": "Legacy", "status": "completed"},
            session.to_dict(),
        ],
    )

    store.end_session(status="completed")
    history = store._get_session_history()
    assert {"session_id": "session-old", "goal": "Legacy", "status": "completed"} in history
    updated = next(entry for entry in history if entry["session_id"] == "session-main")
    assert updated["status"] == "completed"


def test_get_resumable_tasks_filters_statuses() -> None:
    store = InMemoryStateStore()
    store.update(
        last_plan={
            "tasks": [
                {"id": "T1", "status": "pending"},
                {"id": "T2", "status": "in_progress"},
                {"id": "T3", "status": "needs_attention"},
                {"id": "T4", "status": "completed"},
            ]
        }
    )
    resumable = store.get_resumable_tasks()
    assert {task["id"] for task in resumable} == {"T1", "T2", "T3"}


def test_validate_state_adds_last_updated() -> None:
    store = InMemoryStateStore()
    payload: dict[str, Any] = {}
    store._validate_state(payload)
    assert "last_updated" in payload


def test_get_session_history_returns_copy() -> None:
    store = InMemoryStateStore()
    session = store.start_session("History copy", session_id="sess-copy")
    history = store._get_session_history()
    assert history[0]["session_id"] == session.session_id
    history.append({"session_id": "mutation"})
    fresh_history = store._get_session_history()
    assert all(entry.get("session_id") != "mutation" for entry in fresh_history)


def test_state_store_load_creates_default_when_missing(tmp_path: Path) -> None:
    state_path = tmp_path / "missing.json"
    store = StateStore(state_path)
    data = store.load()
    assert data["version"] == "1.0"
    assert state_path.exists()


def test_state_store_read_existing_file(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    store = StateStore(state_path)
    store.save(store.load())

    second = StateStore(state_path)
    data = second.load()
    assert data["version"] == "1.0"


def test_state_store_save_without_path_uses_memory() -> None:
    store = StateStore()
    store.save({"created_at": "0", "last_updated": "0"})
    assert store.load()["created_at"] == "0"


def test_state_store_write_cache_no_path_safe() -> None:
    store = StateStore()
    store._write_cache_to_disk_locked()
