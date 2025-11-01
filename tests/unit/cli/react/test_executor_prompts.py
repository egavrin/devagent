import click

from ai_dev_agent.cli.react import executor
from ai_dev_agent.cli.react.executor import (
    _build_phase_prompt,
    _build_synthesis_prompt,
    _execute_react_assistant,
)
from ai_dev_agent.cli.router import IntentDecision
from ai_dev_agent.core.utils.config import Settings


def test_build_phase_prompt_basic():
    """Test basic phase prompt construction."""
    result = _build_phase_prompt(
        phase="exploration",
        user_query="Fix the bug",
        context="",
        constraints="",
        workspace="/tmp/test",
        repository_language=None,
    )
    assert "TASK: Fix the bug" in result
    assert "WORKSPACE: /tmp/test" in result


def test_build_phase_prompt_with_language():
    """Test prompt with repository language."""
    result = _build_phase_prompt(
        phase="exploration",
        user_query="Test",
        context="",
        constraints="",
        workspace="/tmp",
        repository_language="python",
    )
    assert "LANGUAGE: python" in result
    assert "Python-specific tooling" in result


def test_build_phase_prompt_with_context():
    """Test prompt with previous discoveries."""
    result = _build_phase_prompt(
        phase="exploration",
        user_query="Test",
        context="Found 3 files",
        constraints="",
        workspace="/tmp",
        repository_language=None,
    )
    assert "PREVIOUS DISCOVERIES" in result
    assert "Found 3 files" in result


def test_build_phase_prompt_with_constraints():
    """Test prompt with constraints."""
    result = _build_phase_prompt(
        phase="exploration",
        user_query="Test",
        context="",
        constraints="Must use Python 3.8",
        workspace="/tmp",
        repository_language=None,
    )
    assert "CONSTRAINTS" in result
    assert "Must use Python 3.8" in result


def test_build_synthesis_prompt():
    """Test synthesis prompt construction."""
    result = _build_synthesis_prompt(
        user_query="Test query",
        context="Some context",
        workspace="/tmp/workspace",
    )
    assert "Test query" in result
    assert "Some context" in result
    assert "Workspace: /tmp/workspace" in result


def test_terminal_output_planning_mode(monkeypatch, capsys):
    """Test terminal output in planning mode."""

    def fake_execute_with_planning(ctx, client, settings, user_prompt, **kwargs):
        return {"result": "planning"}

    monkeypatch.setattr(
        "ai_dev_agent.cli.react.plan_executor.execute_with_planning",
        fake_execute_with_planning,
    )

    ctx = click.Context(click.Command("react"))
    ctx.obj = {}

    result = _execute_react_assistant(
        ctx=ctx,
        client=object(),
        settings=Settings(),
        user_prompt="Investigate issue",
        use_planning=True,
    )

    output = capsys.readouterr().out
    assert "🗺️ Planning: Investigate issue" in output
    assert "🗺️ Planning mode enabled" in output
    assert result == {"result": "planning"}


class _DummyRouter:
    def __init__(self, *_args, **_kwargs):
        pass

    def route(self, _prompt):
        return IntentDecision(tool=None, arguments={"text": "Task complete"})


def test_terminal_output_direct_mode(monkeypatch, capsys):
    """Test terminal output in direct mode."""

    monkeypatch.setattr(executor, "_resolve_intent_router", lambda: _DummyRouter)
    monkeypatch.setattr(executor, "_detect_repository_language", lambda *a, **k: (None, None))
    monkeypatch.setattr(executor, "_get_structure_hints_state", lambda _ctx: {})
    monkeypatch.setattr(executor, "resolve_prompt_input", lambda value: value or "")

    ctx = click.Context(click.Command("react"))
    ctx.meta["_emit_status_messages"] = True
    ctx.obj = {}

    _execute_react_assistant(
        ctx=ctx,
        client=object(),
        settings=Settings(),
        user_prompt="Handle request",
        use_planning=False,
    )

    output = capsys.readouterr().out
    assert "⚡ Executing: Handle request" in output
    assert "⚡ Direct execution mode" in output
    assert "Task complete" in output
    assert "(direct)" in output
