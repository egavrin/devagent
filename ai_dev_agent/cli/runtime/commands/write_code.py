"""Write-code command implementation for CLI runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.specialized import ImplementationAgent
from ai_dev_agent.agents.strategies.implementation import ImplementationAgentStrategy

from .._compat import get_cli_state

if TYPE_CHECKING:  # pragma: no cover
    from ..main import CLIState


def _build_prompt(
    state: "CLIState",
    design_file: str,
    test_file: Optional[str],
) -> str:
    """Compose implementation prompt with repository context."""
    strategy = ImplementationAgentStrategy(prompt_loader=state.prompt_loader)
    base_context = {
        "workspace": str(state.context_builder.workspace),
        "system_context": state.system_context,
        "project_context": state.project_context,
    }
    strategy.set_context(base_context)

    task = f"Implement solution from {design_file}"
    prompt_context = {
        "design_file": design_file,
    }
    if test_file:
        prompt_context["test_file"] = test_file

    try:
        prompt = strategy.build_prompt(task, context=prompt_context)
    except FileNotFoundError:
        prompt = (
            "# Implementation Prompt\n"
            "Implement the requested feature following tests and existing patterns.\n\n"
            f"## Task\n\nImplement solution from {design_file}."
        )
        if test_file:
            prompt += f"\n\n## Test File\n\nEnsure tests pass in: {test_file}"
    else:
        prompt += "\n"

    repo_snapshot = {
        "workspace": base_context["workspace"],
        "project": base_context["project_context"],
        "system": base_context["system_context"],
    }
    prompt += "\n## Repository Context\n```json\n"
    prompt += json.dumps(repo_snapshot, indent=2, sort_keys=True)
    prompt += "\n```"

    return prompt


@click.command(
    name="write-code",
    help="Implement code from a design file.",
    short_help="Implement code from design/test artifacts.",
)
@click.argument("design_file", required=True)
@click.option("--test-file", "-t", help="Path to test file")
@click.pass_context
def write_code_command(
    ctx: click.Context,
    design_file: str,
    test_file: Optional[str],
) -> None:
    """Implement code using design/test files with repo context."""
    state = get_cli_state(ctx)
    execute_write_code(ctx, state, design_file, test_file)


def execute_write_code(
    ctx: click.Context,
    state: "CLIState",
    design_file: str,
    test_file: Optional[str],
) -> None:
    """Shared implementation for code writing routine."""
    prompt = _build_prompt(state, design_file, test_file)

    agent = ImplementationAgent()
    agent_context = AgentContext(session_id=f"implement-{Path(design_file).stem}")
    agent_context.metadata.update(
        {
            "design_file": design_file,
            "test_file": test_file,
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
        click.echo(f"⚙️  Implementing from '{design_file}'...")

    result = agent.execute(prompt, agent_context)

    if result.success:
        if json_output:
            click.echo(
                json.dumps({"success": True, "output": result.output, "metadata": result.metadata})
            )
        else:
            click.echo(click.style("✓ Implementation completed", fg="green"))
            click.echo(f"\n{result.output}")

            created = result.metadata.get("files_created")
            if created:
                click.echo(f"\nFiles created: {len(created)}")
    else:
        if json_output:
            click.echo(json.dumps({"success": False, "error": result.error}))
        else:
            click.echo(click.style(f"✗ Failed: {result.error}", fg="red"))
        raise click.Abort()
