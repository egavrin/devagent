from ai_dev_agent.core.utils.state import StateStore


def test_append_history_trims_entries():
    store = StateStore()
    for index in range(60):
        store.append_history(
            {
                "command_path": ["run"],
                "params": {"index": index},
                "timestamp": str(index),
            },
            limit=5,
        )
    history = store.load().get("command_history")
    assert history is not None
    assert len(history) == 5
    assert history[0]["params"]["index"] == 55
    assert history[-1]["params"]["index"] == 59


def test_record_metric_limits_entries():
    store = StateStore()
    for index in range(12):
        store.record_metric(
            {
                "task_id": f"T{index}",
                "outcome": "completed",
                "duration": index,
            },
            limit=10,
        )
    metrics = store.load().get("metrics")
    assert metrics is not None
    assert len(metrics) == 10
    assert metrics[0]["task_id"] == "T2"
    assert metrics[-1]["task_id"] == "T11"
