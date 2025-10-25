"""Backward compatibility tests for primary CLI entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import pytest
from click.testing import CliRunner

import ai_dev_agent.cli as cli_module
import ai_dev_agent.cli.commands as cli_commands
from ai_dev_agent.cli import cli
from ai_dev_agent.core.utils.config import Settings


@pytest.fixture
def cli_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tuple[CliRunner, List[Dict[str, Any]], Settings, Path]:
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
    monkeypatch.setattr(cli_module, "get_llm_client", lambda ctx: object())

    recorded_calls: List[Dict[str, Any]] = []

    def fake_execute(
        ctx: click.Context,
        client: object,
        active_settings: Settings,
        prompt: str,
        *,
        use_planning: Optional[bool] = None,
        system_extension: Optional[str] = None,
        format_schema: Optional[Dict[str, Any]] = None,
        agent_type: str = "manager",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        recorded_calls.append(
            {
                "prompt": prompt,
                "use_planning": use_planning,
                "system_extension": system_extension,
                "format_schema": format_schema,
                "agent_type": agent_type,
            }
        )
        click.echo("ok")
        return {"final_message": "ok", "final_json": None, "result": {"status": "success"}}

    monkeypatch.setattr(cli_commands, "_execute_react_assistant", fake_execute)

    runner = CliRunner()
    return runner, recorded_calls, settings, repo_root


class TestBackwardCompatibility:
    """Ensure legacy CLI behaviours remain stable."""

    def test_natural_language_query(self, cli_test_env: Tuple[CliRunner, List[Dict[str, Any]], Settings, Path]) -> None:
        runner, calls, *_ = cli_test_env

        result = runner.invoke(cli, ["inspect", "repo"])

        assert result.exit_code == 0, result.output
        assert "ok" in result.output
        assert calls and calls[-1]["prompt"] == "inspect repo"

    def test_plan_flag_routes_to_planning_mode(
        self,
        cli_test_env: Tuple[CliRunner, List[Dict[str, Any]], Settings, Path],
    ) -> None:
        runner, calls, *_ = cli_test_env

        result = runner.invoke(cli, ["--plan", "summarize", "changes"])

        assert result.exit_code == 0, result.output
        assert calls and calls[-1]["use_planning"] is True

    def test_review_command_outputs_json(
        self,
        cli_test_env: Tuple[CliRunner, List[Dict[str, Any]], Settings, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner, _, _, repo_root = cli_test_env

        patch_file = repo_root / "changes.patch"
        rule_file = repo_root / "rule.md"
        patch_file.write_text("diff --git a/foo b/foo\n", encoding="utf-8")
        rule_file.write_text("# Rule\n", encoding="utf-8")

        monkeypatch.setattr(
            cli_commands,
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
