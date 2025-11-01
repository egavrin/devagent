"""Tests for the new CLI runtime entrypoint scaffolding."""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from ai_dev_agent.agents.base import AgentResult
from ai_dev_agent.cli.runtime.commands import query as query_module


@pytest.mark.usefixtures("monkeypatch")
def test_cli_help_lists_query_command():
    """Ensure the CLI runtime entrypoint exposes the query command."""
    from ai_dev_agent.cli.runtime.main import cli  # (import within test)

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "query" in result.output


def test_cli_query_command_delegates(monkeypatch):
    """Ensure the query wrapper forwards to the legacy callback."""
    from ai_dev_agent.cli.runtime.main import cli  # (import within test)

    captured: dict[str, object] = {}

    def fake_execute(ctx, state, prompt, force_plan, direct, agent):
        captured["prompt"] = prompt
        captured["ctx_obj"] = ctx.obj
        captured["force_plan"] = force_plan

    monkeypatch.setattr(query_module, "execute_query", fake_execute)

    runner = CliRunner()
    result = runner.invoke(cli, ["query", "hello", "world"])

    assert result.exit_code == 0
    assert captured["prompt"] == ("hello", "world")
    assert "settings" in captured["ctx_obj"]


def test_cli_nl_fallback_invokes_query(monkeypatch):
    """Ensure natural language fallback routes into query."""
    from ai_dev_agent.cli.runtime.main import cli  # (import within test)

    captured: dict[str, object] = {}

    def fake_execute(ctx, state, prompt, force_plan, direct, agent):
        captured["pending_prompt"] = ctx.meta.get("_pending_nl_prompt")

    monkeypatch.setattr(query_module, "execute_query", fake_execute)

    runner = CliRunner()
    result = runner.invoke(cli, ["investigate a bug"])

    assert result.exit_code == 0
    assert captured["pending_prompt"] == "investigate a bug"


def test_cli_plan_flag_sets_default_use_planning(monkeypatch):
    """Global --plan flag should propagate to legacy context."""
    from ai_dev_agent.cli.runtime.main import cli  # (import within test)

    captured: dict[str, object] = {}

    def fake_execute(ctx, state, prompt, force_plan, direct, agent):
        captured["default_use_planning"] = ctx.obj.get("default_use_planning")
        captured["force_plan"] = force_plan

    monkeypatch.setattr(query_module, "execute_query", fake_execute)

    runner = CliRunner()
    result = runner.invoke(cli, ["--plan", "query", "plan", "work"])

    assert result.exit_code == 0
    assert captured["default_use_planning"] is True
    assert captured["force_plan"] is False


def test_cli_review_command_executes(tmp_path, monkeypatch):
    """Review command should execute using the runtime implementation."""
    from ai_dev_agent.cli.runtime.main import cli  # (import within test)

    target = tmp_path / "example.py"
    target.write_text("# sample file\n")

    captured: dict[str, object] = {}

    def fake_execute_strategy(agent_type, prompt, agent_context, **kwargs):
        captured["prompt"] = prompt
        captured["metadata"] = dict(agent_context.metadata)
        return AgentResult(success=True, output="Looks good", metadata={"issues_found": 0})

    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.commands.review.execute_strategy",
        fake_execute_strategy,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["review", str(target)])

    assert result.exit_code == 0
    assert str(target) in captured["prompt"]
    assert captured["metadata"]["file_path"] == str(target)


def test_cli_chat_command_runs(monkeypatch):
    """Chat command should start and terminate a shell session."""
    from ai_dev_agent.cli.runtime.main import cli  # (import within test)

    fake_manager = MagicMock()
    fake_manager.create_session.return_value = "session-1"

    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.commands.chat.ShellSessionManager",
        lambda **_: fake_manager,
    )

    prompts = iter(["exit"])
    monkeypatch.setattr("click.prompt", lambda *_, **__: next(prompts))

    runner = CliRunner()
    result = runner.invoke(cli, ["chat"])

    assert result.exit_code == 0
    fake_manager.create_session.assert_called_once()
    fake_manager.close_all.assert_called_once()
