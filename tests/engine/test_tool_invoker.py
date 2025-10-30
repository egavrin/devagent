from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest

import ai_dev_agent.engine.react.tool_invoker as tool_invoker_module
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.tool_invoker import RegistryToolInvoker, SessionAwareToolInvoker
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


def _make_session_invoker(tmp_path, *, session_manager, session_id="session-1"):
    settings = Settings()
    settings.workspace_root = tmp_path
    return SessionAwareToolInvoker(
        workspace=tmp_path,
        settings=settings,
        sandbox=None,
        collector=None,
        pipeline_commands=None,
        devagent_cfg=None,
        session_manager=session_manager,
        session_id=session_id,
        shell_session_manager=None,
        shell_session_id=None,
        cli_context=None,
        llm_client=None,
    )


class DummySessionManager:
    def __init__(self):
        self.messages = []

    def add_tool_message(self, session_id, tool_call_id, content):
        self.messages.append((session_id, tool_call_id, content))


@contextmanager
def _temporary_tool(name, handler):
    try:
        original = registry.get(name)
    except KeyError:
        original = None
    registry.register(
        ToolSpec(name=name, handler=handler, request_schema_path=None, response_schema_path=None)
    )
    try:
        yield
    finally:
        if original is not None:
            registry.register(original)
        else:
            registry._tools.pop(name, None)
            registry._rebuild_indices()


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


def test_tool_invoker_wrap_result_formats_misc_tools(tmp_path):
    invoker = _make_invoker(tmp_path)

    def find_handler(payload, context):
        return {"files": [{"path": "src/a.py"}, "src/b.py"]}

    def grep_handler(payload, context):
        return {
            "matches": [
                {"file": "src/a.py", "matches": [{"line": 1}]},
                {"file": "src/b.py", "matches": []},
            ]
        }

    def symbols_handler(payload, context):
        return {"symbols": [{"file": "src/a.py", "symbol": "Foo"}, {"file": "src/b.py"}]}

    with ExitStack() as stack:
        stack.enter_context(_temporary_tool("find", find_handler))
        stack.enter_context(_temporary_tool("grep", grep_handler))
        stack.enter_context(_temporary_tool("symbols", symbols_handler))
        find_action = ActionRequest(step_id="f", thought="find", tool="find", args={"query": "foo"})
        obs_find = invoker(find_action)
        assert obs_find.success
        assert obs_find.metrics["files"] == 2
        assert obs_find.artifacts == ["src/a.py", "src/b.py"]
        assert '"path": "src/a.py"' in obs_find.raw_output

        grep_action = ActionRequest(
            step_id="g", thought="grep", tool="grep", args={"pattern": "foo"}
        )
        obs_grep = invoker(grep_action)
        assert obs_grep.metrics["match_counts"]["src/a.py"] == 1
        assert obs_grep.artifacts[0] == "src/a.py"

        symbols_action = ActionRequest(
            step_id="s", thought="symbols", tool="symbols", args={"path": "src/a.py"}
        )
        obs_symbols = invoker(symbols_action)
        assert obs_symbols.metrics["symbols"] == 2
        assert "src/a.py" in obs_symbols.artifacts


def test_tool_invoker_write_rejection_sets_reason(tmp_path):
    def rejecting_write(payload, context):
        return {
            "applied": False,
            "rejected_hunks": ["Patch failure\nDetailed mismatch"],
            "diff_stats": {"additions": 0, "deletions": 0},
        }

    registry.register(
        ToolSpec(
            name=WRITE,
            handler=rejecting_write,
            request_schema_path=None,
            response_schema_path=None,
        )
    )

    invoker = _make_invoker(tmp_path)
    action = ActionRequest(step_id="w", thought="write", tool=WRITE, args={"path": "file.txt"})
    observation = invoker(action)

    assert observation.success is False
    assert observation.metrics["rejected_hunks"] == 1
    assert observation.metrics["rejection_reason"] == "Patch failure"
    assert "Patch rejected" in observation.outcome


def test_tool_invoker_run_observation_formats_output(tmp_path):
    def failing_run(payload, context):
        return {
            "exit_code": 1,
            "duration_ms": 12,
            "stdout_tail": "line1\nline2",
            "stderr_tail": "err1\nerr2",
        }

    registry.register(
        ToolSpec(name=RUN, handler=failing_run, request_schema_path=None, response_schema_path=None)
    )

    invoker = _make_invoker(tmp_path)
    action = ActionRequest(step_id="r", thought="run", tool=RUN, args={"cmd": "echo x"})
    observation = invoker(action)

    assert observation.success is False
    assert observation.metrics["exit_code"] == 1
    assert "STDOUT" in observation.raw_output
    assert "line1" in observation.raw_output


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


def test_session_invoker_find_formats_and_records(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def find_handler(payload, context):
        return {"files": [{"path": "src/a.py"}, {"path": "src/b.py"}]}

    with _temporary_tool("find", find_handler):
        action = ActionRequest(
            step_id="1", thought="find files", tool="find", args={"query": "foo"}
        )
        observation = invoker(action)

        assert observation.display_message.startswith("🔍 find")
        assert observation.formatted_output.startswith("Files found")
        assert session_manager.messages
        recorded = session_manager.messages[-1][2]
        assert "Files found" in recorded


def test_session_invoker_run_summarizes_and_writes_artifact(tmp_path, monkeypatch):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    long_output = "stdout line\n" * 10

    def run_handler(payload, context):
        return {"exit_code": 0, "stdout_tail": long_output, "stderr_tail": ""}

    registry.register(
        ToolSpec(name=RUN, handler=run_handler, request_schema_path=None, response_schema_path=None)
    )

    monkeypatch.setattr(
        tool_invoker_module,
        "summarize_text",
        lambda text, limit: "short" if len(text) > 10 else text,
    )

    saved_path = tmp_path / ".devagent" / "artifacts" / "saved.txt"

    def fake_write(content, *, suffix=".txt", root=None):
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path.write_text(content)
        return saved_path

    monkeypatch.setattr(tool_invoker_module, "write_artifact", fake_write)

    action = ActionRequest(step_id="2", thought="run", tool=RUN, args={"cmd": "echo hi"})
    observation = invoker(action)

    assert observation.formatted_output.endswith(
        "Full output saved to .devagent/artifacts/saved.txt"
    )
    assert observation.artifact_path == ".devagent/artifacts/saved.txt"
    assert session_manager.messages
    assert "(See .devagent/artifacts/saved.txt" in session_manager.messages[-1][2]


def test_session_invoker_batch_records_each_result(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def read_handler(payload, context):
        return {"files": [{"path": payload["paths"][0], "content": "text"}]}

    def write_handler(payload, context):
        return {"applied": True, "changed_files": [payload["path"]], "diff_stats": {}}

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

    tool_calls = [
        ToolCall(tool=READ, args={"paths": ["foo.txt"]}, call_id="read-1"),
        ToolCall(tool=WRITE, args={"path": "bar.txt", "content": "v"}, call_id="write-1"),
    ]

    action = ActionRequest(
        step_id="batch", thought="execute batch", tool="batch", tool_calls=tool_calls
    )
    observation = invoker(action)

    assert observation.tool == "batch[2]"
    assert len(observation.results) == 2
    assert len(session_manager.messages) == 2
