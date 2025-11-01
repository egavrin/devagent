import asyncio
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest

import ai_dev_agent.engine.react.tool_invoker as tool_invoker_module
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.tool_invoker import RegistryToolInvoker, SessionAwareToolInvoker
from ai_dev_agent.engine.react.types import ActionRequest, CLIObservation, ToolCall, ToolResult
from ai_dev_agent.tools import READ, RUN, WRITE, ToolSpec, registry
from ai_dev_agent.tools.execution.shell_session import ShellSessionManager


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


def test_session_invoker_find_formats_and_records(tmp_path, monkeypatch):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def find_handler(payload, context):
        return {"files": [{"path": "src/a.py"}, {"path": "src/b.py"}]}

    monkeypatch.setenv("DEVAGENT_DEBUG_TOOLS", "1")
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


def test_invoke_batch_uses_async_execution_when_loop_available(monkeypatch, tmp_path):
    invoker = _make_invoker(tmp_path)

    loop = asyncio.new_event_loop()
    async_called = {"flag": False}
    call_threads: list[str] = []

    original_async = RegistryToolInvoker._execute_batch_async

    async def wrapped_async(self, tool_calls):
        async_called["flag"] = True
        return await original_async(self, tool_calls)

    monkeypatch.setattr(RegistryToolInvoker, "_execute_batch_async", wrapped_async)

    def fake_execute_single_tool(self, call):
        call_threads.append(call.call_id)
        return ToolResult(
            call_id=call.call_id,
            tool=call.tool,
            success=True,
            outcome="ok",
            error=None,
            metrics={},
            wall_time=0.0,
        )

    monkeypatch.setattr(RegistryToolInvoker, "_execute_single_tool", fake_execute_single_tool)

    tool_calls = [
        ToolCall(tool="noop", args={}, call_id="first"),
        ToolCall(tool="noop", args={}, call_id="second"),
    ]

    try:
        asyncio.set_event_loop(loop)
        observation = invoker.invoke_batch(tool_calls)
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    assert async_called["flag"] is True
    assert observation.metrics["total_calls"] == 2
    assert observation.metrics["successful_calls"] == 2
    assert call_threads == ["first", "second"]
    assert all(result.success for result in observation.results)


@pytest.mark.asyncio
async def test_invoke_batch_sequential_in_running_loop_handles_failures(monkeypatch, tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    async def unexpected_async(self, tool_calls):
        raise AssertionError("async path should not run with active loop")

    monkeypatch.setattr(RegistryToolInvoker, "_execute_batch_async", unexpected_async)

    def ok_handler(payload, context):
        return {"files": [{"path": "foo.txt", "content": "ok"}]}

    def failing_handler(payload, context):
        raise RuntimeError("boom")

    with ExitStack() as stack:
        stack.enter_context(_temporary_tool("ok", ok_handler))
        stack.enter_context(_temporary_tool("boom", failing_handler))

        tool_calls = [
            ToolCall(tool="boom", args={}, call_id="fail"),
            ToolCall(tool="ok", args={}, call_id="succeed"),
        ]
        action = ActionRequest(
            step_id="batch",
            thought="handle tools",
            tool="batch",
            tool_calls=tool_calls,
        )
        observation = invoker(action)

    assert observation.success is False
    assert observation.metrics["total_calls"] == 2
    assert observation.metrics["failed_calls"] == 1
    failure_result = next(res for res in observation.results if not res.success)
    assert failure_result.error == "boom"
    assert any(res.success for res in observation.results)
    # Ensure session messages captured both paths, including error surface
    assert len(session_manager.messages) == 2
    assert "Error: boom" in session_manager.messages[0][2]


def test_execute_single_tool_converts_general_exception(tmp_path, monkeypatch):
    invoker = _make_invoker(tmp_path)

    def boom(tool_name, payload):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(invoker, "_invoke_registry", boom)

    result = invoker._execute_single_tool(ToolCall(tool="noop", args={}, call_id="test"))
    assert result.success is False
    assert result.error == "kaboom"
    assert result.outcome.startswith("Tool noop failed")


def test_session_invoker_records_stdout_from_raw_output(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    action = ActionRequest(step_id="1", thought="run command", tool=RUN, args={"cmd": "echo hi"})
    observation = CLIObservation(
        success=True,
        outcome="Command exited with 0",
        tool=RUN,
        metrics={"exit_code": 0, "stdout_tail": None, "stderr_tail": None},
        raw_output="STDOUT:\nhello world\n\nSTDERR:\nwarning line\n",
        artifact_path=".devagent/artifacts/run.txt",
        formatted_output=None,
        display_message=None,
    )

    invoker._record_tool_message(action, observation)

    assert session_manager.messages
    _, tool_call_id, content = session_manager.messages[-1]
    assert tool_call_id.startswith("tool-exec-")
    assert "STDOUT:\nhello world" in content
    assert "STDERR:\nwarning line" in content
    assert "(See .devagent/artifacts/run.txt" in content


def test_tool_invoker_handles_runtime_exception(tmp_path):
    invoker = _make_invoker(tmp_path)

    def exploding_handler(payload, context):
        raise RuntimeError("explode")

    with _temporary_tool("explode", exploding_handler):
        action = ActionRequest(step_id="1", thought="boom", tool="explode", args={})
        observation = invoker(action)

    assert observation.success is False
    assert observation.error == "explode"
    assert observation.outcome == "Tool explode failed"


def test_invoke_batch_without_calls_returns_error(tmp_path):
    invoker = _make_invoker(tmp_path)

    observation = invoker.invoke_batch([])

    assert observation.success is False
    assert observation.error == "Empty batch request"
    assert observation.outcome == "No tool calls provided"


def test_execute_single_tool_handles_submit_final_answer(tmp_path):
    invoker = _make_invoker(tmp_path)
    result = invoker._execute_single_tool(
        ToolCall(tool="submit_final_answer", args={"answer": "done"}, call_id="final")
    )

    assert result.success is True
    assert result.tool == "submit_final_answer"
    assert result.metrics["raw_output"] == "done"


def test_invoke_registry_passes_shell_context(tmp_path, monkeypatch):
    settings = Settings()
    settings.workspace_root = tmp_path
    manager = ShellSessionManager()
    invoker = RegistryToolInvoker(
        workspace=tmp_path,
        settings=settings,
        sandbox=None,
        collector=None,
        pipeline_commands=None,
        devagent_cfg=None,
        shell_session_manager=manager,
        shell_session_id="shell-123",
    )

    captured = {}

    def fake_invoke(tool_name, payload, ctx):
        captured["extra"] = ctx.extra
        return {"ok": True}

    monkeypatch.setattr(tool_invoker_module.registry, "invoke", fake_invoke)

    invoker._invoke_registry("custom", {})

    assert captured["extra"]["shell_session_manager"] is manager
    assert captured["extra"]["shell_session_id"] == "shell-123"


def test_session_invoker_write_display_message(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def write_handler(payload, context):
        return {
            "applied": True,
            "changed_files": [f"src/file_{i}.py" for i in range(5)],
            "diff_stats": {},
        }

    with _temporary_tool(WRITE, write_handler):
        action = ActionRequest(
            step_id="w",
            thought="apply patch",
            tool=WRITE,
            args={"path": "src/file_0.py", "content": "new"},
        )
        observation = invoker(action)

    assert observation.display_message.startswith("📝 write")
    assert "(+2)" in observation.display_message


def test_session_invoker_batch_generates_call_id_for_missing(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def read_handler(payload, context):
        return {"files": [{"path": payload["paths"][0], "content": "text"}]}

    with _temporary_tool(READ, read_handler):
        tool_calls = [ToolCall(tool=READ, args={"paths": ["foo.txt"]})]
        action = ActionRequest(
            step_id="batch",
            thought="missing id",
            tool="batch",
            tool_calls=tool_calls,
        )
        observation = invoker(action)

    assert observation.metrics["total_calls"] == 1
    assert session_manager.messages
    _, tool_call_id, content = session_manager.messages[-1]
    assert tool_call_id.startswith("tool-batch-")
    assert "Read 1 file(s)" in content


def test_tool_invocation_error_scenarios(tmp_path):
    invoker = _make_invoker(tmp_path)

    def value_error_handler(payload, context):
        raise ValueError("schema mismatch")

    def timeout_handler(payload, context):
        raise TimeoutError("execution expired")

    with ExitStack() as stack:
        stack.enter_context(_temporary_tool("value_error_tool", value_error_handler))
        stack.enter_context(_temporary_tool("timeout_tool", timeout_handler))

        tool_calls = [
            ToolCall(tool="value_error_tool", args={}, call_id="value"),
            ToolCall(tool="timeout_tool", args={}, call_id="timeout"),
            ToolCall(tool="missing_tool", args={}, call_id="missing"),
        ]

        observation = invoker.invoke_batch(tool_calls)

    assert observation.success is False
    assert observation.metrics["failed_calls"] == 3
    results_by_id = {result.call_id: result for result in observation.results}

    assert results_by_id["value"].metrics.get("error_type") == "ValueError"
    assert "schema mismatch" in results_by_id["value"].error

    assert results_by_id["timeout"].metrics.get("error_type") == "TimeoutError"
    assert "execution expired" in results_by_id["timeout"].error

    assert results_by_id["missing"].metrics.get("error_type") == "KeyError"
    assert "missing_tool" in results_by_id["missing"].error


def test_tool_parallel_execution(tmp_path):
    invoker = _make_invoker(tmp_path)
    thread_ids = set()

    def slow_handler(payload, context):
        thread_ids.add(threading.get_ident())
        time.sleep(0.1)
        return {"ok": payload["label"]}

    with _temporary_tool("slow_tool", slow_handler):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            start = time.perf_counter()
            tool_calls = [
                ToolCall(tool="slow_tool", args={"label": "first"}, call_id="first"),
                ToolCall(tool="slow_tool", args={"label": "second"}, call_id="second"),
            ]
            observation = invoker.invoke_batch(tool_calls)
            elapsed = time.perf_counter() - start
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    assert observation.success is True
    assert observation.metrics["total_calls"] == 2
    # Ensure execution happened with concurrency (significantly less than sequential 0.24s)
    assert elapsed < 0.18
    assert len(thread_ids) >= 2


def test_tool_result_validation(tmp_path):
    invoker = _make_invoker(tmp_path)

    def messy_find(payload, context):
        return {
            "files": [
                "src/a.py",
                {"path": "src/b.py", "content": "print('ok')"},
                42,
                None,
                {"file": "src/c.py"},
                Path("src/d.py"),
                {"path": None},
            ]
        }

    def messy_write(payload, context):
        return {
            "applied": True,
            "changed_files": ["src/a.py", None, 123],
            "diff_stats": {"additions": 1, "deletions": 0},
        }

    def messy_run(payload, context):
        return {
            "exit_code": 1,
            "duration_ms": 5,
            "stdout_tail": "",
            "stderr_tail": "problem line\n" + "x" * 200,
        }

    with ExitStack() as stack:
        stack.enter_context(_temporary_tool("find", messy_find))
        stack.enter_context(_temporary_tool(WRITE, messy_write))
        stack.enter_context(_temporary_tool(RUN, messy_run))

        find_action = ActionRequest(
            step_id="find", thought="look", tool="find", args={"query": "src"}
        )
        find_observation = invoker(find_action)
        assert find_observation.metrics["files"] == 4
        assert find_observation.artifacts == [
            "src/a.py",
            "src/b.py",
            "src/c.py",
            "src/d.py",
        ]
        assert "42" not in find_observation.artifacts

        write_action = ActionRequest(
            step_id="write",
            thought="apply",
            tool=WRITE,
            args={"path": "src/a.py", "content": "new"},
        )
        write_observation = invoker(write_action)
        assert write_observation.success is True
        assert write_observation.artifacts == ["src/a.py"]
        assert write_observation.metrics["applied"] is True

        run_action = ActionRequest(
            step_id="run",
            thought="exec",
            tool=RUN,
            args={"cmd": "echo broken"},
        )
        run_observation = invoker(run_action)
        assert run_observation.success is False
        assert "stderr" in run_observation.outcome.lower()
        assert run_observation.metrics["stderr_tail"].endswith("x" * 200)
        assert run_observation.metrics["stderr_preview"].endswith("…")
        assert run_observation.raw_output.endswith("x" * 200 + "\n")


def test_tool_result_validation_handles_non_iterable_find(tmp_path):
    invoker = _make_invoker(tmp_path)

    def weird_find(payload, context):
        return {"files": "not-a-list"}

    with _temporary_tool("find", weird_find):
        action = ActionRequest(step_id="find", thought="weird", tool="find", args={"query": "foo"})
        observation = invoker(action)

    assert observation.success is False
    assert observation.metrics["files"] == 0
    assert observation.artifacts == []


def test_session_invoker_read_display_message(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def read_handler(payload, context):
        return {"files": [{"path": payload["paths"][0], "content": "line1\nline2\n"}]}

    with _temporary_tool(READ, read_handler):
        action = ActionRequest(
            step_id="read",
            thought="inspect",
            tool=READ,
            args={"paths": ["src/file.txt"]},
        )
        observation = invoker(action)

    assert observation.display_message.startswith("📖 read")
    assert "2 lines" in observation.display_message
    assert session_manager.messages
    _, _, content = session_manager.messages[-1]
    assert "📖" in content


def test_session_invoker_run_display_message_prefers_stderr(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def run_handler(payload, context):
        return {
            "exit_code": 1,
            "stdout_tail": "",
            "stderr_tail": "failure occurred\nsecond line",
        }

    with _temporary_tool(RUN, run_handler):
        action = ActionRequest(step_id="run", thought="exec", tool=RUN, args={"cmd": "do"})
        observation = invoker(action)

    assert "stderr" in observation.display_message.lower()
    assert "failure occurred" in observation.display_message
    assert session_manager.messages
    _, _, content = session_manager.messages[-1]
    assert "STDERR:" in content


def test_session_invoker_record_tool_message_captures_streams(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def run_handler(payload, context):
        return {
            "exit_code": 0,
            "stdout_tail": "line one\nline two\n",
            "stderr_tail": "warn\n",
        }

    with _temporary_tool(RUN, run_handler):
        action = ActionRequest(step_id="run", thought="stdout", tool=RUN, args={"cmd": "list"})
        observation = invoker(action)

    assert observation.metrics["stdout_tail"].startswith("line one")
    assert session_manager.messages
    _, _, content = session_manager.messages[-1]
    assert "STDOUT:\nline one" in content
    assert "STDERR:\nwarn" in content


def test_session_invoker_batch_records_error_result(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(
        tmp_path, session_manager=session_manager, session_id="session-x"
    )

    failing_result = ToolResult(
        call_id=None,
        tool="broken",
        success=False,
        outcome="Failed to execute",
        error="boom",
        metrics={},
        wall_time=None,
    )

    invoker._record_batch_tool_message(failing_result)

    assert session_manager.messages
    _, call_id, content = session_manager.messages[-1]
    assert call_id.startswith("tool-batch-")
    assert "Error: boom" in content


def test_run_stream_preview_handles_non_string_tail(tmp_path):
    class BadStr:
        def __str__(self):
            raise RuntimeError("no string")

    invoker = _make_invoker(tmp_path)

    def run_handler(payload, context):
        return {"exit_code": 0, "stdout_tail": BadStr(), "stderr_tail": BadStr()}

    with _temporary_tool(RUN, run_handler):
        action = ActionRequest(step_id="run", thought="preview", tool=RUN, args={"cmd": "noop"})
        observation = invoker(action)

    assert observation.metrics["stdout_tail"] is None
    assert observation.metrics["stderr_tail"] is None
    assert observation.raw_output.endswith("\n")


def test_run_wrap_preserves_full_streams(tmp_path):
    invoker = _make_invoker(tmp_path)

    long_text = "X" * 200

    def run_handler(payload, context):
        return {"exit_code": 0, "stdout_tail": long_text, "stderr_tail": long_text}

    with _temporary_tool(RUN, run_handler):
        action = ActionRequest(step_id="run", thought="preview", tool=RUN, args={"cmd": "noop"})
        observation = invoker(action)

    assert observation.metrics["stdout_tail"] == long_text
    assert observation.metrics["stderr_tail"] == long_text
    assert observation.metrics["stdout_preview"].endswith("…")
    assert observation.metrics["stderr_preview"].endswith("…")
    assert long_text in observation.raw_output


def test_session_invoker_passes_cli_context_and_llm(tmp_path, monkeypatch):
    captured = {}

    def fake_invoke(tool_name, payload, ctx):
        captured.update(ctx.extra)
        return {"files": []} if tool_name == READ else {"files": []}

    monkeypatch.setattr(tool_invoker_module.registry, "invoke", fake_invoke)
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(
        tmp_path,
        session_manager=session_manager,
        session_id="sess-1",
    )
    invoker.cli_context = {"mode": "cli"}
    invoker.llm_client = object()

    action = ActionRequest(
        step_id="read",
        thought="context",
        tool=READ,
        args={"paths": ["foo.txt"]},
    )
    invoker(action)

    assert captured["cli_context"] == {"mode": "cli"}
    assert captured["session_id"] == "sess-1"
    assert "llm_client" in captured


def test_session_invoker_find_formats_large_result(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def find_handler(payload, context):
        return {"files": [{"path": f"src/file_{i}.py"} for i in range(60)]}

    with _temporary_tool("find", find_handler):
        action = ActionRequest(
            step_id="find",
            thought="many files",
            tool="find",
            args={"query": "src"},
        )
        observation = invoker(action)

    assert "... and 40 more files" in observation.formatted_output
    assert "Tip: Use more specific pattern" in observation.formatted_output


def test_session_invoker_grep_formats_large_result(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def grep_handler(payload, context):
        matches = []
        for i in range(55):
            matches.append({"file": f"src/file_{i}.py", "matches": [{"line": 1}]})
        return {"matches": matches}

    with _temporary_tool("grep", grep_handler):
        action = ActionRequest(
            step_id="grep",
            thought="large grep",
            tool="grep",
            args={"pattern": "foo"},
        )
        observation = invoker(action)

    assert observation.metrics["files"] == 55
    assert "... and 35 more files" in observation.formatted_output
    assert "Tip: Use more specific pattern" in observation.formatted_output


def test_session_invoker_final_answer_formatting(tmp_path):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    action = ActionRequest(
        step_id="final",
        thought="answer",
        tool="submit_final_answer",
        args={"answer": "All tasks complete."},
    )

    observation = invoker(action)

    assert observation.formatted_output == "All tasks complete."
    assert observation.display_message.startswith("✅")
    assert session_manager.messages
    _, _, content = session_manager.messages[-1]
    assert "All tasks complete." in content


def test_session_invoker_grep_formats_with_counts(tmp_path, monkeypatch):
    session_manager = DummySessionManager()
    invoker = _make_session_invoker(tmp_path, session_manager=session_manager)

    def messy_grep(payload, context):
        return {
            "matches": [
                {"file": "src/a.py", "matches": [{"line": 1}, {"line": 2}]},
                {"file": "src/b.py", "matches": []},
                "garbage",
                {"matches": []},
                {"file": "src/c.py", "matches": [{"line": 1}]},
            ]
        }

    monkeypatch.setenv("DEVAGENT_DEBUG_TOOLS", "1")
    with _temporary_tool("grep", messy_grep):
        action = ActionRequest(
            step_id="grep",
            thought="search",
            tool="grep",
            args={"pattern": "foo"},
        )
        observation = invoker(action)

    assert observation.formatted_output.startswith("Files with matches:")
    assert "- src/a.py (2 matches)" in observation.formatted_output
    assert "- src/b.py" in observation.formatted_output
    assert observation.metrics["match_counts"]["src/b.py"] == 0
    assert "garbage" not in observation.formatted_output
