"""Generate-tests command implementation for CLI runtime."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

import click

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.specialized import TestingAgent
from ai_dev_agent.agents.strategies.test import TestGenerationAgentStrategy

from .._compat import get_cli_state

if TYPE_CHECKING:  # pragma: no cover
    from ..main import CLIState


def _build_prompt(state: "CLIState", feature: str, coverage: int, test_type: str) -> str:
    """Compose testing prompt using shared context."""
    strategy = TestGenerationAgentStrategy(prompt_loader=state.prompt_loader)
    base_context = {
        "workspace": str(state.context_builder.workspace),
        "system_context": state.system_context,
        "project_context": state.project_context,
    }
    strategy.set_context(base_context)

    prompt_context = {
        "coverage_target": coverage,
        "test_type": test_type,
    }

    try:
        prompt = strategy.build_prompt(feature, context=prompt_context)
    except FileNotFoundError:
        prompt = (
            "# Test Generation Prompt\n"
            "Create thorough tests following TDD and report coverage gaps.\n\n"
            f"## Task\n\nGenerate {test_type} tests for {feature} with {coverage}% coverage."
        )
    else:
        prompt += (
            f"\n\n## Coverage Goal\nTarget coverage: {coverage}%"
            f"\n\n## Test Type\nPrimary focus: {test_type}"
        )

    repo_snapshot = {
        "workspace": base_context["workspace"],
        "project": base_context["project_context"],
        "system": base_context["system_context"],
    }
    prompt += "\n\n## Repository Context\n```json\n"
    prompt += json.dumps(repo_snapshot, indent=2, sort_keys=True)
    prompt += "\n```"

    return prompt


@click.command(
    name="generate-tests",
    help="Generate tests for a feature.",
    short_help="Generate tests with coverage targets.",
)
@click.argument("feature", required=True)
@click.option("--coverage", "-c", default=90, show_default=True, help="Target coverage percentage")
@click.option(
    "--type",
    "-t",
    "test_type",
    type=click.Choice(["unit", "integration", "all"]),
    default="all",
    show_default=True,
)
@click.pass_context
def generate_tests_command(
    ctx: click.Context,
    feature: str,
    coverage: int,
    test_type: str,
) -> None:
    """Generate tests using CLIState prompt loader and context builder."""
    state = get_cli_state(ctx)
    execute_generate_tests(ctx, state, feature, coverage, test_type)


def execute_generate_tests(
    ctx: click.Context,
    state: "CLIState",
    feature: str,
    coverage: int,
    test_type: str,
) -> None:
    """Shared implementation for test generation."""
    prompt = _build_prompt(state, feature, coverage, test_type)

    agent = TestingAgent()
    agent_context = AgentContext(session_id=f"test-{feature}")
    agent_context.metadata.update(
        {
            "feature": feature,
            "target_coverage": coverage,
            "test_type": test_type,
            "workspace": str(state.context_builder.workspace),
            "system_context": state.system_context,
            "project_context": state.project_context,
            "cli_state": state,
            "prompt_loader": state.prompt_loader,
            "context_builder": state.context_builder,
            "state_store": ctx.obj.get("state"),
            "parent_ctx": ctx,
        }
    )

    json_output = ctx.obj.get("json_output", False)
    if not json_output:
        click.echo(f"🧪 Generating tests for '{feature}'...")

    result = agent.execute(prompt, agent_context)

    if result.success:
        if json_output:
            click.echo(
                json.dumps({"success": True, "output": result.output, "metadata": result.metadata})
            )
        else:
            click.echo(click.style("✓ Tests generated", fg="green"))
            click.echo(f"\n{result.output}")

            files = result.metadata.get("test_files_created")
            if files:
                click.echo("\nFiles created:")
                for file_path in files:
                    click.echo(f"  - {file_path}")
    else:
        if json_output:
            click.echo(json.dumps({"success": False, "error": result.error}))
        else:
            click.echo(click.style(f"✗ Failed: {result.error}", fg="red"))
        raise click.Abort()
