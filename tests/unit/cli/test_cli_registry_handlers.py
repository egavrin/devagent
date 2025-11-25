"""Tests for CLI registry handlers."""

from __future__ import annotations

import click
import pytest

from ai_dev_agent.cli.handlers.registry_handlers import (
    RegistryIntent,
    _build_exec_payload,
    _build_find_payload,
    _build_grep_payload,
    _build_read_payload,
    _build_symbols_payload,
    _handle_exec_result,
    _handle_read_result,
    _handle_simple_result,
    _handle_symbols_index_result,
    _should_enable_regex,
)
from ai_dev_agent.core.utils.config import Settings


def _make_context(command: str = "tool") -> click.Context:
    ctx = click.Context(click.Command(command))
    ctx.obj = {"settings": Settings()}
    return ctx


def test_find_requires_query() -> None:
    ctx = _make_context("find")
    with pytest.raises(click.ClickException):
        _build_find_payload(ctx, {})


def test_find_builds_payload_with_options() -> None:
    ctx = _make_context("find")
    payload, meta = _build_find_payload(
        ctx, {"query": "*.py", "path": "src", "limit": "5", "fuzzy": False}
    )
    assert payload == {"query": "*.py", "path": "src", "limit": 5, "fuzzy": False}
    assert meta == {}


@pytest.mark.parametrize(
    "pattern",
    [
        "Compile.*method",
        r"(?P<name>Task)",
        r"^TaskManager",
    ],
)
def test_grep_auto_regex_triggers(pattern: str) -> None:
    ctx = _make_context("grep")
    payload, _ = _build_grep_payload(ctx, {"pattern": pattern})

    assert payload["regex"] is True


@pytest.mark.parametrize(
    "pattern",
    [
        "def greet",
        "main.py",
        "TaskManager::CreateTaskQueue",
    ],
)
def test_grep_keeps_literal_patterns(pattern: str) -> None:
    ctx = _make_context("grep")
    payload, _ = _build_grep_payload(ctx, {"pattern": pattern})

    assert payload.get("regex") is None


def test_grep_respects_explicit_regex_flag() -> None:
    ctx = _make_context("grep")
    payload, _ = _build_grep_payload(ctx, {"pattern": "foo", "regex": False})
    assert payload["regex"] is False


def test_symbols_requires_name() -> None:
    ctx = _make_context("symbols")
    with pytest.raises(click.ClickException):
        _build_symbols_payload(ctx, {})


def test_symbols_payload_includes_optional_fields() -> None:
    ctx = _make_context("symbols")
    payload, meta = _build_symbols_payload(
        ctx,
        {
            "name": "MyClass",
            "path": "src",
            "limit": "10",
            "kind": "class",
            "lang": "python",
        },
    )
    assert payload == {
        "name": "MyClass",
        "path": "src",
        "limit": 10,
        "kind": "class",
        "lang": "python",
    }
    assert meta == {}


def test_should_enable_regex_detects_anchors_and_patterns() -> None:
    assert _should_enable_regex("^hello")
    assert _should_enable_regex("world$")
    assert _should_enable_regex(r"\d{3}")
    assert not _should_enable_regex(r"\^escaped")
    assert _should_enable_regex("foo|bar")


def test_should_enable_regex_ignores_blank_queries() -> None:
    assert _should_enable_regex("") is False
    assert _should_enable_regex("   ") is False


def test_find_payload_rejects_invalid_limit() -> None:
    ctx = _make_context("find")
    with pytest.raises(click.ClickException):
        _build_find_payload(ctx, {"query": "foo", "limit": "abc"})


def test_grep_payload_requires_pattern() -> None:
    ctx = _make_context("grep")
    with pytest.raises(click.ClickException):
        _build_grep_payload(ctx, {"pattern": ""})


def test_grep_payload_rejects_invalid_limit() -> None:
    ctx = _make_context("grep")
    with pytest.raises(click.ClickException):
        _build_grep_payload(ctx, {"pattern": "foo", "limit": "not-int"})


def test_grep_payload_includes_path_argument() -> None:
    ctx = _make_context("grep")
    payload, _ = _build_grep_payload(ctx, {"pattern": "foo", "path": "src"})

    assert payload["path"] == "src"


def test_symbols_payload_rejects_invalid_limit() -> None:
    ctx = _make_context("symbols")
    with pytest.raises(click.ClickException):
        _build_symbols_payload(ctx, {"name": "Thing", "limit": "oops"})


def test_read_payload_requires_paths() -> None:
    ctx = _make_context("read")
    with pytest.raises(click.ClickException):
        _build_read_payload(ctx, {})


def test_read_payload_normalises_arguments() -> None:
    ctx = _make_context("read")
    payload, meta = _build_read_payload(
        ctx,
        {
            "path": "README.md",
            "paths": ["docs/one.md", "docs/two.md"],
            "context_lines": "5",
            "byte_range": (1, 8),
        },
    )

    assert payload["paths"] == ["docs/one.md", "docs/two.md"]
    assert payload["context_lines"] == 5
    assert payload["byte_range"] == [1, 8]
    assert meta == {}


def test_handle_read_result_displays_default_window(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("read")
    ctx.obj["settings"].fs_read_default_max_lines = 2

    _handle_read_result(
        ctx,
        {},
        {"files": [{"path": "sample.txt", "content": "one\ntwo\nthree"}]},
        {},
    )

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured[0] == "== sample.txt =="
    assert captured[1].endswith(": one")
    assert captured[2].endswith(": two")
    assert captured[3].startswith("... (1 more lines")


def test_handle_read_result_handles_empty_files(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("read")

    _handle_read_result(
        ctx,
        {},
        {"files": [{"path": "empty.txt", "content": ""}]},
        {},
    )

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured[0] == "== empty.txt =="
    assert captured[1] == "(empty)"


def test_handle_read_result_no_content_returns_message(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("read")

    _handle_read_result(ctx, {}, {"files": []}, {})

    assert "No content returned." in capsys.readouterr().out


def test_handle_read_result_with_line_range(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("read")

    _handle_read_result(
        ctx,
        {"start_line": 2, "end_line": 3},
        {"files": [{"path": "sample.txt", "content": "one\ntwo\nthree\nfour"}]},
        {},
    )

    output = capsys.readouterr().out
    assert "Reading sample.txt (lines 2-3 of 4)" in output
    assert "    2: two" in output
    assert "    3: three" in output


def test_handle_read_result_normalises_start_line_below_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx = _make_context("read")

    _handle_read_result(
        ctx,
        {"start_line": 0, "end_line": 1},
        {"files": [{"path": "sample.txt", "content": "one\ntwo"}]},
        {},
    )

    output = capsys.readouterr().out
    assert "Reading sample.txt (lines 1-1 of 2)" in output


def test_handle_read_result_enforces_end_line_not_less_than_start() -> None:
    ctx = _make_context("read")

    with pytest.raises(click.ClickException, match="greater than or equal to"):
        _handle_read_result(
            ctx,
            {"start_line": 3, "end_line": 2},
            {"files": [{"path": "sample.txt", "content": "one\ntwo\nthree"}]},
            {},
        )


def test_handle_read_result_reports_empty_file_with_slicing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx = _make_context("read")

    _handle_read_result(
        ctx,
        {"start_line": 1, "end_line": 2},
        {"files": [{"path": "empty.txt", "content": ""}]},
        {},
    )

    output = capsys.readouterr().out
    assert "empty.txt is empty." in output


def test_handle_read_result_enforces_numeric_limits() -> None:
    ctx = _make_context("read")

    with pytest.raises(click.ClickException, match="start_line must be an integer"):
        _handle_read_result(
            ctx,
            {"start_line": "nope"},
            {"files": [{"path": "a.py", "content": "data"}]},
            {},
        )

    with pytest.raises(click.ClickException, match="end_line must be an integer"):
        _handle_read_result(
            ctx,
            {"start_line": 1, "end_line": "bad"},
            {"files": [{"path": "a.py", "content": "one"}]},
            {},
        )

    with pytest.raises(click.ClickException, match="max_lines must be an integer"):
        _handle_read_result(
            ctx,
            {"start_line": 1, "max_lines": "bad"},
            {"files": [{"path": "a.py", "content": "one"}]},
            {},
        )


def test_handle_read_result_handles_out_of_range_start(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("read")

    _handle_read_result(
        ctx,
        {"start_line": 10},
        {"files": [{"path": "data.txt", "content": "one\ntwo"}]},
        {},
    )

    output = capsys.readouterr().out
    assert "File data.txt has only 2 lines" in output


def test_handle_read_result_normalises_max_lines(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("read")

    _handle_read_result(
        ctx,
        {"start_line": 1, "max_lines": 0},
        {"files": [{"path": "d.txt", "content": "one\ntwo"}]},
        {},
    )

    output = capsys.readouterr().out
    assert "Reading d.txt (lines 1-2 of 2)" in output


def test_handle_symbols_index_result_reports_stats(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("symbols_index")

    _handle_symbols_index_result(
        ctx,
        {},
        {"stats": {"files_indexed": 3, "symbols": 7}, "db_path": "/tmp/index.db"},
        {},
    )

    output = capsys.readouterr().out
    assert "Symbol index updated." in output
    assert "Indexed 3 file(s), 7 symbol(s)" in output
    assert "Index written to /tmp/index.db" in output


def test_handle_symbols_index_result_without_stats(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("symbols_index")

    _handle_symbols_index_result(ctx, {}, {"stats": {}, "db_path": ""}, {})

    output = capsys.readouterr().out
    assert "Symbol index updated." in output


def test_build_exec_payload_populates_arguments(monkeypatch) -> None:
    ctx = _make_context("run")

    monkeypatch.setattr(
        "ai_dev_agent.cli.handlers.registry_handlers.build_system_context",
        lambda: {"os": "Linux"},
    )

    payload, meta = _build_exec_payload(
        ctx,
        {"cmd": "echo hello", "args": ["hello"], "cwd": "/tmp", "timeout": "5"},
    )

    assert payload == {"cmd": "echo hello", "args": ["hello"], "cwd": "/tmp", "timeout_sec": 5}
    assert meta == {}


def test_build_exec_payload_accepts_timeout_sec(monkeypatch) -> None:
    ctx = _make_context("run")

    monkeypatch.setattr(
        "ai_dev_agent.cli.handlers.registry_handlers.build_system_context",
        lambda: {"os": "Linux"},
    )

    payload, _ = _build_exec_payload(ctx, {"cmd": "echo hi", "timeout_sec": "7"})

    assert payload["timeout_sec"] == 7


def test_build_exec_payload_rejects_invalid_timeout() -> None:
    ctx = _make_context("run")

    with pytest.raises(click.ClickException, match="timeout must be an integer"):
        _build_exec_payload(ctx, {"cmd": "echo", "timeout": "abc"})


def test_registry_intent_invokes_registry(monkeypatch) -> None:
    ctx = _make_context("tool")
    calls: list[tuple[dict[str, str], dict[str, str]]] = []

    def fake_payload_builder(_ctx, arguments):
        return {"arg": arguments["value"]}, {"meta": "info"}

    def fake_result_handler(_ctx, _arguments, result, extras):
        calls.append((result, extras))

    captured: dict[str, object] = {}

    def fake_invoke(ctx, tool_name, payload, *, with_sandbox):
        captured["ctx"] = ctx
        captured["tool"] = tool_name
        captured["payload"] = payload
        captured["with_sandbox"] = with_sandbox
        return {"ok": True}

    monkeypatch.setattr(
        "ai_dev_agent.cli.handlers.registry_handlers._invoke_registry_tool", fake_invoke
    )

    handler = RegistryIntent(
        tool_name="demo",
        payload_builder=fake_payload_builder,
        result_handler=fake_result_handler,
        with_sandbox=True,
    )

    result = handler(ctx, {"value": "42"})

    assert result == {"ok": True}
    assert calls == [({"ok": True}, {"meta": "info"})]
    assert captured["tool"] == "demo"
    assert captured["payload"] == {"arg": "42"}
    assert captured["with_sandbox"] is True


def test_registry_intent_recovery_handler(monkeypatch) -> None:
    ctx = _make_context("tool")

    def fake_payload_builder(_ctx, _arguments):
        return {"payload": True}, {}

    def fake_recovery(_ctx, _arguments, payload, extras, exc):
        assert payload == {"payload": True}
        assert extras == {}
        assert isinstance(exc, RuntimeError)
        return {"recovered": True}

    monkeypatch.setattr(
        "ai_dev_agent.cli.handlers.registry_handlers._invoke_registry_tool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    handler = RegistryIntent(
        tool_name="demo",
        payload_builder=fake_payload_builder,
        result_handler=lambda *_: None,
        with_sandbox=False,
        recovery_handler=fake_recovery,
    )

    result = handler(ctx, {})
    assert result == {"recovered": True}


def test_handle_simple_result_prints_files_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("find")

    _handle_simple_result(
        ctx,
        {},
        {"files": [{"path": "a.txt", "lines": 10, "size_bytes": 2048, "score": 0.6}]},
        {},
    )

    _handle_simple_result(ctx, {}, {"error": "bad things"}, {})

    output = capsys.readouterr().out
    assert "a.txt (10 lines, 2.0 KB, score 0.6)" in output
    assert "Error: bad things" in output


def test_handle_simple_result_prints_matches_and_symbols(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx = _make_context("grep")

    _handle_simple_result(
        ctx,
        {},
        {
            "matches": [
                {
                    "file": "app.py",
                    "matches": [{"line": 3, "text": "print('hi')"}],
                }
            ]
        },
        {},
    )

    _handle_simple_result(
        ctx,
        {},
        {"symbols": [{"file": "app.py", "line": 1, "name": "main", "kind": "function"}]},
        {},
    )

    output = capsys.readouterr().out
    assert "app.py:" in output
    assert "  3: print('hi')" in output
    assert "app.py:1 - main (function)" in output


def test_handle_simple_result_handles_string_match_groups(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx = _make_context("grep")

    _handle_simple_result(ctx, {}, {"matches": ["raw\nmatch"]}, {})

    output = capsys.readouterr().out
    assert "raw" in output


def test_handle_simple_result_handles_plain_file_entries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx = _make_context("find")

    _handle_simple_result(ctx, {}, {"files": ["notes.txt"]}, {})

    assert "notes.txt" in capsys.readouterr().out


def test_handle_exec_result_prints_streams(capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _make_context("run")

    _handle_exec_result(
        ctx,
        {},
        {"exit_code": 3, "stdout_tail": "output\n", "stderr_tail": "error\n"},
        {},
    )

    output = capsys.readouterr().out
    assert "Exit: 3" in output
    assert "output" in output
    assert "error" in output


# WRITE tool tests removed - WRITE has been deprecated in favor of EDIT
# which now consumes canonical apply_patch payloads
