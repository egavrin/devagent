from types import SimpleNamespace

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.tool_invoker import RegistryToolInvoker
from ai_dev_agent.engine.react.types import ActionRequest, ToolCall
from ai_dev_agent.tools import READ, RUN, WRITE, ToolSpec, registry


@pytest.fixture(autouse=True)
def _restore_registry():
    original_read = registry.get(READ)
    original_write = registry.get(WRITE)
    original_run = registry.get(RUN)
    try:
        yield
    finally:
        registry.register(original_read)
        registry.register(original_write)
        registry.register(original_run)


def _make_invoker(tmp_path, *, extra=None):
    settings = Settings()
    settings.workspace_root = tmp_path

    return RegistryToolInvoker(
        workspace=tmp_path,
        settings=settings,
        sandbox=None,
        collector=None,
        pipeline_commands=None,
        devagent_cfg=None,
        shell_session_manager=None,
        shell_session_id=None,
    )


def test_read_cache_and_write_invalidation(tmp_path):
    file_path = tmp_path / "foo.txt"
    file_path.write_text("one")

    read_calls = {"count": 0}

    def read_handler(payload, context):
        read_calls["count"] += 1
        path = payload["paths"][0]
        file_on_disk = context.repo_root / path
        return {"files": [{"path": path, "content": file_on_disk.read_text()}]}

    def write_handler(payload, context):
        path = payload["path"]
        (context.repo_root / path).write_text(payload.get("content", ""))
        return {
            "applied": True,
            "changed_files": [path],
            "diff_stats": {"additions": 1, "deletions": 0},
        }

    registry.register(
        ToolSpec(
            name=READ, handler=read_handler, request_schema_path=None, response_schema_path=None
        )
    )
    registry.register(
        ToolSpec(
            name=WRITE, handler=write_handler, request_schema_path=None, response_schema_path=None
        )
    )

    invoker = _make_invoker(tmp_path)

    read_action = ActionRequest(step_id="1", thought="read", tool=READ, args={"paths": ["foo.txt"]})
    obs1 = invoker(read_action)
    assert obs1.success
    assert read_calls["count"] == 1

    obs_cached = invoker(read_action)
    assert obs_cached.success
    assert read_calls["count"] == 1, "second read should hit cache"

    write_action = ActionRequest(
        step_id="2",
        thought="write",
        tool=WRITE,
        args={"path": "foo.txt", "content": "two"},
    )
    obs_write = invoker(write_action)
    assert obs_write.success

    obs_after_write = invoker(read_action)
    assert obs_after_write.success
    assert read_calls["count"] == 2
    assert '"two"' in obs_after_write.raw_output


def test_run_clears_cache(tmp_path):
    file_path = tmp_path / "foo.txt"
    file_path.write_text("data")

    def read_handler(payload, context):
        return {"files": [{"path": payload["paths"][0], "content": "cached"}]}

    def run_handler(payload, context):
        return {"stdout": "done"}

    registry.register(
        ToolSpec(
            name=READ, handler=read_handler, request_schema_path=None, response_schema_path=None
        )
    )
    registry.register(
        ToolSpec(name=RUN, handler=run_handler, request_schema_path=None, response_schema_path=None)
    )

    invoker = _make_invoker(tmp_path)

    action_read = ActionRequest(step_id="x", thought="read", tool=READ, args={"paths": ["foo.txt"]})
    invoker(action_read)
    assert invoker._file_read_cache

    run_action = ActionRequest(step_id="y", thought="run", tool=RUN, args={"command": "echo"})
    invoker(run_action)
    assert invoker._file_read_cache == {}


def test_invoke_batch_mix_success_and_failure(tmp_path):
    def read_handler(payload, context):
        return {"files": [{"path": payload["paths"][0], "content": "ok"}]}

    registry.register(
        ToolSpec(
            name=READ, handler=read_handler, request_schema_path=None, response_schema_path=None
        )
    )

    invoker = _make_invoker(tmp_path)

    tool_calls = [
        ToolCall(tool=READ, args={"paths": ["foo.txt"]}, call_id="a"),
        ToolCall(tool="nonexistent", args={}, call_id="b"),
    ]

    observation = invoker.invoke_batch(tool_calls)
    assert observation.tool == "batch[2]"
    assert observation.metrics["total_calls"] == 2
    assert observation.metrics["failed_calls"] == 1
    assert any(result.success for result in observation.results)
    assert any(not result.success for result in observation.results)


def test_tool_invoker_unknown_and_value_error(tmp_path):
    invoker = _make_invoker(tmp_path)

    unknown = ActionRequest(step_id="0", thought="unknown", tool="missing", args={})
    obs = invoker(unknown)
    assert obs.success is False
    assert "not registered" in obs.error

    def rejecting_handler(payload, context):
        raise ValueError("bad input")

    registry.register(
        ToolSpec(
            name="reject",
            handler=rejecting_handler,
            request_schema_path=None,
            response_schema_path=None,
        )
    )
    try:
        rejection = ActionRequest(step_id="1", thought="reject", tool="reject", args={})
        obs_reject = invoker(rejection)
        assert obs_reject.success is False
        assert "rejected" in obs_reject.outcome
    finally:
        registry._tools.pop("reject", None)
        registry._rebuild_indices()


def test_submit_final_answer_and_cache_expiry(tmp_path):
    read_calls = {"count": 0}

    def read_handler(payload, context):
        read_calls["count"] += 1
        return {"files": [{"path": payload["paths"][0], "content": "v"}]}

    registry.register(
        ToolSpec(
            name=READ, handler=read_handler, request_schema_path=None, response_schema_path=None
        )
    )

    invoker = _make_invoker(tmp_path)
    invoker._cache_ttl = -1  # Force expiry on next access

    read_action = ActionRequest(step_id="r", thought="read", tool=READ, args={"paths": ["f.txt"]})
    invoker(read_action)
    invoker(read_action)
    assert read_calls["count"] == 2, "cache should expire immediately"

    final = ActionRequest(
        step_id="f", thought="done", tool="submit_final_answer", args={"answer": "ok"}
    )
    obs_final = invoker(final)
    assert obs_final.success is True
    assert obs_final.tool == "submit_final_answer"
