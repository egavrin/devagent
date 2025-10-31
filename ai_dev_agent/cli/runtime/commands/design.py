"""Design command implementation for CLI runtime using shared CLI state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.specialized import DesignAgent
from ai_dev_agent.agents.strategies.design import DesignAgentStrategy

from .._compat import get_cli_state

if TYPE_CHECKING:  # pragma: no cover - type hinting only
    from ..main import CLIState


def _build_prompt(state: "CLIState", feature: str, extra_context: Optional[str]) -> str:
    """Compose the prompt using the design strategy and shared context."""
    strategy = DesignAgentStrategy(prompt_loader=state.prompt_loader)
    base_context = {
        "workspace": str(state.context_builder.workspace),
        "system_context": state.system_context,
        "project_context": state.project_context,
    }
    strategy.set_context(base_context)

    task = f"Design solution for {feature}"
    prompt_context = {"feature_name": feature}
    if extra_context:
        prompt_context["additional_context"] = extra_context

    try:
        prompt = strategy.build_prompt(task, context=prompt_context)
    except FileNotFoundError:
        prompt = (
            "# Design Prompt\n"
            "You create comprehensive technical designs including architecture and risks.\n\n"
            f"## Task\n\n{task}"
        )
        if extra_context:
            prompt += f"\n\n## Additional Context\n{extra_context}"
    else:
        if extra_context:
            prompt += f"\n\n## Additional Context\n{extra_context}"

    # Attach repository context snapshot for grounding
    repo_snapshot = {
        "workspace": base_context["workspace"],
        "project": base_context["project_context"],
        "system": base_context["system_context"],
    }
    prompt += "\n\n## Repository Context\n```json\n"
    prompt += json.dumps(repo_snapshot, indent=2, sort_keys=True)
    prompt += "\n```"

    return prompt


def execute_create_design(
    ctx: click.Context,
    state: "CLIState",
    feature: str,
    output: Optional[str],
    context: Optional[str],
) -> None:
    """Shared implementation for creating a design document."""
    prompt = _build_prompt(state, feature, context)

    agent = DesignAgent()
    agent_context = AgentContext(session_id=f"design-{feature}")
    agent_context.metadata.update(
        {
            "feature": feature,
            "extra_context": context,
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
        click.echo(f"🎨 Creating design for '{feature}'...")

    result = agent.execute(prompt, agent_context)

    if result.success:
        if json_output:
            click.echo(
                json.dumps({"success": True, "output": result.output, "metadata": result.metadata})
            )
        else:
            click.echo(click.style("✓ Design completed", fg="green"))
            click.echo(f"\n{result.output}")

            if output:
                Path(output).write_text(result.output)
                click.echo(f"\nSaved to: {output}")
    else:
        if json_output:
            click.echo(json.dumps({"success": False, "error": result.error}))
        else:
            click.echo(click.style(f"✗ Failed: {result.error}", fg="red"))
        raise click.Abort()


@click.command(
    name="create-design",
    help="Create a technical design for a feature.",
    short_help="Create a technical design document.",
)
@click.argument("feature", required=True)
@click.option("--context", "-c", help="Additional context")
@click.option("--output", "-o", help="Output path for design document")
@click.pass_context
def create_design_command(
    ctx: click.Context,
    feature: str,
    output: Optional[str],
    context: Optional[str],
) -> None:
    """Create design document using CLI runtime state (prompt loader + context builder)."""
    state = get_cli_state(ctx)
    execute_create_design(ctx, state, feature, output, context)
