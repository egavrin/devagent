"""Tests for the new CLI runtime entrypoint scaffolding."""

import importlib
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import ai_dev_agent.cli
from ai_dev_agent.agents.base import AgentResult
from ai_dev_agent.cli.runtime.commands import query as query_module


@pytest.mark.usefixtures("monkeypatch")
def test_initialise_state_smoke(monkeypatch, tmp_path):
    """Smoke test that CLI startup wiring populates shared state."""
    from ai_dev_agent.cli.runtime import main as main_module
    from ai_dev_agent.core.utils.config import Settings

    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}
    settings = Settings()

    def fake_load_settings(config_path):
        captured["config_path"] = config_path
        return settings

    configure_calls: list[tuple[str, bool]] = []

    def fake_configure_logging(level, *, structured):
        configure_calls.append((level, structured))

    class DummyPromptLoader:
        def __init__(self):
            self.prompts_dir = tmp_path / "prompts"

    class DummyBuilder:
        def __init__(self, workspace):
            captured["workspace"] = workspace

        def build_system_context(self):
            return {"sys": "ctx"}

        def build_project_context(self):
            return {"project": "ctx"}

    def fake_build_context(in_settings):
        assert in_settings is settings
        return {"state": {"session": "ok"}}

    monkeypatch.setattr(main_module, "load_settings", fake_load_settings)
    monkeypatch.setattr(main_module, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(main_module, "PromptLoader", lambda: DummyPromptLoader())
    monkeypatch.setattr(main_module, "ContextBuilder", DummyBuilder)
    monkeypatch.setattr(main_module, "_build_context", fake_build_context)

    config_path = tmp_path / "devagent.toml"
    result_settings, cli_context, state = main_module._initialise_state(
        config_path,
        verbose=2,
        quiet=False,
        repomap_debug=True,
    )

    assert result_settings is settings
    assert captured["config_path"] == config_path
    assert captured["workspace"] == Path(tmp_path)
    assert settings.log_level == "DEBUG"
    assert settings.repomap_debug_stdout is True
    assert configure_calls == [("DEBUG", settings.structured_logging)]
    assert cli_context["state"] == {"session": "ok"}
    assert state.system_context == {"sys": "ctx"}
    assert state.project_context == {"project": "ctx"}
    assert isinstance(state.prompt_loader, DummyPromptLoader)


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


def test_cli_direct_flag_sets_pending_prompt(monkeypatch):
    """--direct flag should be accepted and disable planning fallback."""
    from ai_dev_agent.cli.runtime.main import cli  # (import within test)

    captured: dict[str, object] = {}

    def fake_execute(ctx, state, prompt, force_plan, direct, agent):
        captured["pending_prompt"] = ctx.meta.get("_pending_nl_prompt")
        captured["use_planning"] = ctx.meta.get("_use_planning")
        captured["force_plan"] = force_plan
        captured["direct"] = direct
        captured["default_use_planning"] = ctx.obj.get("default_use_planning")

    monkeypatch.setattr(query_module, "execute_query", fake_execute)

    runner = CliRunner()
    result = runner.invoke(cli, ["--direct", "inspect logs"])

    assert result.exit_code == 0
    assert captured["pending_prompt"] == "inspect logs"
    assert captured["use_planning"] is False
    assert captured["force_plan"] is False
    assert captured["default_use_planning"] is False
    assert captured["direct"] is True


def test_cli_unknown_flag_surfaces_error():
    """Unknown CLI flags should return an informative error."""
    from ai_dev_agent.cli.runtime.main import cli  # (import within test)

    runner = CliRunner()
    result = runner.invoke(cli, ["--not-a-real-flag"])

    assert result.exit_code != 0
    assert "No such option: --not-a-real-flag" in result.output


def test_cli_module_entrypoint_invokes_main():
    """`python -m ai_dev_agent.cli` should call cli.main()."""
    with patch("ai_dev_agent.cli.main") as mock_main:
        runpy.run_module("ai_dev_agent.cli", run_name="__main__")
    mock_main.assert_called_once_with()


def test_cli_module_import_exposes_main():
    """Importing the __main__ module should provide the cli.main callable."""
    module = importlib.import_module("ai_dev_agent.cli.__main__")

    assert hasattr(module, "main")
    assert module.main is ai_dev_agent.cli.main


def test_cli_module_help_executes(tmp_path):
    """Running the module as a script should render help output."""
    # Use isolated HOME so we do not pollute user cache directories
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "ai_dev_agent.cli", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0
    assert "DevAgent - AI-powered development assistant" in result.stdout


def test_cli_plan_and_direct_flags_conflict():
    """Passing both --plan and --direct should surface a usage error."""
    from ai_dev_agent.cli.runtime.main import cli  # (import within test)

    runner = CliRunner()
    result = runner.invoke(cli, ["--plan", "--direct", "summarize logs"])

    assert result.exit_code != 0
    assert "Use either --plan or --direct" in result.output
