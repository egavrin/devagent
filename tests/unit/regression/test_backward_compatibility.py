"""Backward compatibility tests for primary CLI entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

import ai_dev_agent.cli as cli_module
import ai_dev_agent.cli.review as cli_review
from ai_dev_agent.cli import cli
from ai_dev_agent.core.utils.config import Settings


@pytest.fixture
def cli_test_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CliRunner, list[dict[str, Any]], Settings, Path]:
    """Provide a configured CLI runner with stubbed LLM execution."""

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    settings = Settings()
    settings.workspace_root = repo_root
    settings.state_file = repo_root / ".devagent" / "state.json"
    settings.ensure_state_dir()
    settings.api_key = "dummy-key"

    # Ensure CLI loads the predictable settings and stub client.
    monkeypatch.setattr(cli_module, "load_settings", lambda path=None: settings)
    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.main.load_settings",
        lambda path=None: settings,
    )
    dummy_client = object()
    monkeypatch.setattr(cli_module, "get_llm_client", lambda ctx: dummy_client)
    monkeypatch.setattr("ai_dev_agent.cli.utils.get_llm_client", lambda ctx: dummy_client)

    recorded_calls: list[dict[str, Any]] = []

    def fake_execute_query(ctx, state, prompt, force_plan, direct, agent):
        pending = " ".join(prompt).strip()
        if not pending:
            pending = str(ctx.meta.get("_pending_nl_prompt", "")).strip()
        default_plan = bool(ctx.obj.get("default_use_planning", False))
        if force_plan:
            use_planning = True
        elif direct:
            use_planning = False
        else:
            use_planning = default_plan
        recorded_calls.append(
            {
                "prompt": pending,
                "use_planning": use_planning,
                "system_extension": None,
                "format_schema": None,
                "agent_type": agent,
            }
        )
        click.echo("ok")

    monkeypatch.setattr("ai_dev_agent.cli.runtime.commands.query.execute_query", fake_execute_query)

    runner = CliRunner()
    return runner, recorded_calls, settings, repo_root


class TestBackwardCompatibility:
    """Ensure legacy CLI behaviours remain stable."""

    def test_natural_language_query(
        self, cli_test_env: tuple[CliRunner, list[dict[str, Any]], Settings, Path]
    ) -> None:
        runner, calls, *_ = cli_test_env

        result = runner.invoke(cli, ["inspect", "repo"])

        assert result.exit_code == 0, result.output
        assert "ok" in result.output
        assert calls and calls[-1]["prompt"] == "inspect repo"

    def test_plan_flag_routes_to_planning_mode(
        self,
        cli_test_env: tuple[CliRunner, list[dict[str, Any]], Settings, Path],
    ) -> None:
        runner, calls, *_ = cli_test_env

        result = runner.invoke(cli, ["--plan", "summarize", "changes"])

        assert result.exit_code == 0, result.output
        assert calls and calls[-1]["use_planning"] is True

    def test_review_command_outputs_json(
        self,
        cli_test_env: tuple[CliRunner, list[dict[str, Any]], Settings, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner, _, _, repo_root = cli_test_env

        patch_file = repo_root / "changes.patch"
        rule_file = repo_root / "rule.md"
        patch_file.write_text("diff --git a/foo b/foo\n", encoding="utf-8")
        rule_file.write_text("# Rule\n", encoding="utf-8")

        monkeypatch.setattr(
            cli_review,
            "run_review",
            lambda ctx, patch_file, rule_file, json_output, settings: {
                "violations": [],
                "summary": {"total_violations": 0, "files_reviewed": 1, "rule_name": "Rule"},
            },
        )

        result = runner.invoke(
            cli,
            ["review", str(patch_file), "--rule", str(rule_file), "--json"],
        )

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["summary"]["files_reviewed"] == 1
