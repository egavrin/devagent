"""Tests for CLI runtime specialised commands."""

from click.testing import CliRunner

import ai_dev_agent.cli.runtime.main
from ai_dev_agent.agents.base import AgentResult


def _patch_shared_context(monkeypatch):
    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.main.ContextBuilder.build_project_context",
        lambda self, include_outline=False: {"workspace": "/fake/workspace"},
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.main.ContextBuilder.build_system_context",
        lambda self: {"os": "TestOS"},
    )


def _patch_prompt_loader(monkeypatch):
    original_render = ai_dev_agent.cli.runtime.main.PromptLoader.render_prompt

    def fake_render(self, prompt_path, context=None):
        if prompt_path.startswith("agents/"):
            name = prompt_path.split("/")[-1].split(".")[0].upper()
            return f"{name} TEMPLATE"
        return original_render(self, prompt_path, context)

    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.main.PromptLoader.render_prompt",
        fake_render,
    )


def test_create_design_command_uses_cli_state(monkeypatch, tmp_path):
    """Ensure create-design command composes prompt with shared context."""
    from ai_dev_agent.cli.runtime.main import cli

    _patch_shared_context(monkeypatch)
    _patch_prompt_loader(monkeypatch)

    captured = {}

    def fake_execute_strategy(agent_type, prompt, agent_context, **kwargs):
        captured["prompt"] = prompt
        captured["metadata"] = dict(agent_context.metadata)
        return AgentResult(success=True, output="DESIGN OUTPUT", metadata={"foo": "bar"})

    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.commands.design.execute_strategy",
        fake_execute_strategy,
    )

    runner = CliRunner()
    output_file = tmp_path / "design.md"
    result = runner.invoke(
        cli,
        ["create-design", "Payments", "--context", "extra details", "--output", str(output_file)],
    )

    assert result.exit_code == 0
    # Check for actual prompt content (fallback prompt when template not found)
    assert "# Design Prompt" in captured["prompt"] or "DESIGN TEMPLATE" in captured["prompt"]
    assert "Design solution for Payments" in captured["prompt"]
    assert "## Repository Context" in captured["prompt"]
    assert '"workspace": "/fake/workspace"' in captured["prompt"]
    assert captured["metadata"]["feature"] == "Payments"
    assert captured["metadata"]["cli_state"] is not None
    assert captured["metadata"]["system_context"]["os"] == "TestOS"
    assert output_file.read_text() == "DESIGN OUTPUT"


def test_generate_tests_command_uses_cli_state(monkeypatch):
    """Ensure generate-tests command composes prompt with shared context."""
    from ai_dev_agent.cli.runtime.main import cli

    _patch_shared_context(monkeypatch)
    _patch_prompt_loader(monkeypatch)

    captured = {}

    def fake_execute_strategy(agent_type, prompt, agent_context, **kwargs):
        captured["prompt"] = prompt
        captured["metadata"] = dict(agent_context.metadata)
        metadata = {"test_files_created": ["tests/test_feature.py"]}
        return AgentResult(success=True, output="Generated tests", metadata=metadata)

    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.commands.generate_tests.execute_strategy",
        fake_execute_strategy,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["generate-tests", "Search Service", "--coverage", "95", "--type", "integration"],
    )

    assert result.exit_code == 0
    # Check for actual prompt content (fallback prompt when template not found)
    assert "# Test Generation Prompt" in captured["prompt"] or "TEST TEMPLATE" in captured["prompt"]
    assert "Target coverage: 95%" in captured["prompt"]
    assert "Primary focus: integration" in captured["prompt"]
    assert "## Repository Context" in captured["prompt"]
    assert '"workspace": "/fake/workspace"' in captured["prompt"]
    assert captured["metadata"]["target_coverage"] == 95
    assert captured["metadata"]["cli_state"] is not None


def test_write_code_command_uses_cli_state(monkeypatch, tmp_path):
    """Ensure write-code command uses shared prompt/context information."""
    from ai_dev_agent.cli.runtime.main import cli

    _patch_shared_context(monkeypatch)
    _patch_prompt_loader(monkeypatch)

    captured = {}

    def fake_execute_strategy(agent_type, prompt, agent_context, **kwargs):
        captured["prompt"] = prompt
        captured["metadata"] = dict(agent_context.metadata)
        metadata = {"files_created": ["src/module.py"]}
        return AgentResult(success=True, output="Implementation diff", metadata=metadata)

    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.commands.write_code.execute_strategy",
        fake_execute_strategy,
    )

    runner = CliRunner()
    design_file = tmp_path / "design.md"
    design_file.write_text("Design doc")
    test_file = tmp_path / "test_module.py"
    test_file.write_text("Test file")

    result = runner.invoke(
        cli,
        ["write-code", str(design_file), "--test-file", str(test_file)],
    )

    assert result.exit_code == 0
    # Check for actual prompt content (fallback prompt when template not found)
    assert (
        "# Implementation Prompt" in captured["prompt"]
        or "IMPLEMENTATION TEMPLATE" in captured["prompt"]
    )
    assert str(design_file) in captured["prompt"]
    assert str(test_file) in captured["prompt"]
    assert "## Repository Context" in captured["prompt"]
    assert '"workspace": "/fake/workspace"' in captured["prompt"]
    assert captured["metadata"]["design_file"] == str(design_file)
    assert captured["metadata"]["cli_state"] is not None


def test_review_command_uses_cli_state(monkeypatch, tmp_path):
    """Ensure review command integrates CLIState context."""
    from ai_dev_agent.cli.runtime.main import cli

    _patch_shared_context(monkeypatch)
    _patch_prompt_loader(monkeypatch)

    captured = {}

    def fake_execute_strategy(agent_type, prompt, agent_context, **kwargs):
        captured["prompt"] = prompt
        captured["metadata"] = dict(agent_context.metadata)
        metadata = {"issues_found": 0, "quality_score": 1.0}
        return AgentResult(success=True, output="No issues", metadata=metadata)

    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.commands.review.execute_strategy",
        fake_execute_strategy,
    )

    runner = CliRunner()
    target_file = tmp_path / "module.py"
    target_file.write_text("print('hello')\n")

    result = runner.invoke(
        cli,
        ["review", str(target_file)],
    )

    assert result.exit_code == 0
    assert "# Code Review Agent" in captured["prompt"] or "REVIEW TEMPLATE" in captured["prompt"]
    assert str(target_file) in captured["prompt"]
    assert "## Repository Context" in captured["prompt"]
    assert '"workspace": "/fake/workspace"' in captured["prompt"]
    assert captured["metadata"]["file_path"] == str(target_file)
    assert captured["metadata"]["cli_state"] is not None
