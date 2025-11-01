"""Review command implementation for CLI runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import click

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.runtime import create_strategy_agent, execute_strategy

from ... import review as review_module
from .._compat import get_cli_state

if TYPE_CHECKING:  # pragma: no cover
    from ..main import CLIState


def _build_review_prompt(
    state: "CLIState",
    file_path: str,
    *,
    rule_file: Optional[str] = None,
    patch_data: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Compose review prompt using shared context and strategy."""
    adapter = create_strategy_agent("review", prompt_loader=state.prompt_loader)
    base_context = {
        "workspace": str(state.context_builder.workspace),
        "system_context": state.system_context,
        "project_context": state.project_context,
    }
    adapter.set_strategy_context(base_context)

    task = f"Review '{file_path}' for quality, security, and performance issues."
    prompt_context = {
        "file_path": file_path,
        "rule_file": rule_file,
        "patch_data": patch_data,
    }

    try:
        prompt = adapter.build_prompt(task, context=prompt_context)
    except FileNotFoundError:
        prompt = (
            "# Review Prompt\n"
            "You review code for quality, security, and performance issues.\n\n"
            f"## Review Task\n\n{task}"
        )
        if rule_file:
            prompt += f"\n\n## Rule File\n{rule_file}"
        if patch_data:
            prompt += f"\n\n## Patch Dataset\n{patch_data}"

    repo_snapshot = {
        "workspace": base_context["workspace"],
        "project": base_context["project_context"],
        "system": base_context["system_context"],
    }
    prompt += "\n\n## Repository Context\n```json\n"
    prompt += json.dumps(repo_snapshot, indent=2, sort_keys=True)
    prompt += "\n```"

    return prompt, adapter.get_strategy_context()


def execute_review(
    ctx: click.Context,
    state: "CLIState",
    file_path: str,
    rule: Optional[str],
    report: Optional[str],
    json_flag: bool,
) -> None:
    """Execute review command leveraging CLI runtime state."""
    json_output = json_flag or ctx.obj.get("json_output", False)

    # Rule-based review uses existing helper for consistency
    if rule:
        settings = ctx.obj["settings"]

        cli_pkg = sys.modules.get("ai_dev_agent.cli")
        if cli_pkg is not None:
            patched_command = getattr(cli_pkg, "review", None)
            if hasattr(patched_command, "run_review"):
                run_review_fn = patched_command.run_review
            else:
                run_review_fn = review_module.run_review
        else:
            run_review_fn = review_module.run_review

        result = run_review_fn(
            ctx,
            patch_file=file_path,
            rule_file=rule,
            json_output=json_output,
            settings=settings,
        )

        if json_output:
            click.echo(json.dumps(result, indent=2))
            return

        summary = result.get("summary", {})
        files_reviewed = summary.get("files_reviewed", 0)
        total_violations = summary.get("total_violations", 0)
        click.echo(f"Files reviewed: {files_reviewed}")
        click.echo(f"Violations: {total_violations}")
        discarded = summary.get("discarded_violations", 0)
        if discarded:
            click.echo(f"Discarded violations: {discarded} (ignored due to invalid references)")

        violations = result.get("violations") or []
        if violations:
            click.echo("\nFindings:")
            for violation in violations:
                severity = violation.get("severity", "warning")
                location = f"{violation.get('file')}:{violation.get('line')}"
                message = violation.get("message", "")
                click.echo(f"- [{severity.upper()}] {location} - {message}")
        else:
            click.echo("No violations reported.")
        return

    # General review via ReviewAgent
    prompt, strategy_context = _build_review_prompt(state, file_path)

    agent_context = AgentContext(session_id=f"review-{Path(file_path).stem}")
    agent_context.metadata.update(
        {
            "file_path": file_path,
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

    if not json_output:
        click.echo(f"🔍 Reviewing '{file_path}'...")

    result = execute_strategy(
        "review",
        prompt,
        agent_context,
        prompt_loader=state.prompt_loader,
        strategy_context=strategy_context,
        ctx=ctx,
        cli_client=ctx.obj.get("llm_client"),
    )

    if result.success:
        issues = result.metadata.get("issues_found", 0)
        score = result.metadata.get("quality_score", 0.0)

        if json_output:
            click.echo(
                json.dumps(
                    {
                        "success": True,
                        "issues": issues,
                        "score": score,
                        "output": result.output,
                        "metadata": result.metadata,
                    }
                )
            )
            return

        if issues == 0:
            click.echo(click.style("✓ No issues found", fg="green"))
        else:
            click.echo(click.style(f"⚠ Found {issues} issue(s)", fg="yellow"))

        click.echo(f"Quality score: {score:.2f}/1.00")
        click.echo(f"\n{result.output}")

        if report:
            Path(report).write_text(result.output)
            click.echo(f"\nReport saved to: {report}")
    else:
        if json_output:
            click.echo(json.dumps({"success": False, "error": result.error}))
        else:
            click.echo(click.style(f"✗ Review failed: {result.error}", fg="red"))
        raise click.Abort()


@click.command(
    name="review",
    help="Review code for quality, security, and best practices.",
    short_help="Review code for issues.",
)
@click.argument("file_path", required=True)
@click.option(
    "--rule",
    type=click.Path(exists=True),
    help="Path to coding rule file (for patch/rule-based review)",
)
@click.option("--report", "-r", help="Output path for review report")
@click.option("--json", "json_flag", is_flag=True, help="Output in JSON format")
@click.pass_context
def review_command(
    ctx: click.Context,
    file_path: str,
    rule: Optional[str],
    report: Optional[str],
    json_flag: bool,
) -> None:
    """Review command building prompts from CLIState context."""
    state = get_cli_state(ctx)
    execute_review(ctx, state, file_path, rule, report, json_flag)
