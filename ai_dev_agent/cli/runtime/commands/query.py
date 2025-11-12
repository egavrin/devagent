"""Query command for the modern CLI runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import click

from ai_dev_agent.agents import AgentRegistry
from ai_dev_agent.cli.react.executor import _execute_react_assistant
from ai_dev_agent.cli.utils import _record_invocation, get_llm_client
from ai_dev_agent.core.utils.logger import get_logger

from .._compat import get_cli_state

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..main import CLIState

QUERY_HELP = "Execute a natural-language query using the ReAct workflow."
QUERY_SHORT_HELP = "Run a natural-language query."

LOGGER = get_logger(__name__)


def _resolve_pending_prompt(ctx: click.Context, prompt: Tuple[str, ...]) -> str:
    """Resolve the pending prompt from CLI arguments or natural language fallback."""
    pending = " ".join(prompt).strip()
    if not pending:
        pending = str(ctx.meta.pop("_pending_nl_prompt", "")).strip()
    if not pending:
        pending = str(ctx.obj.pop("_pending_nl_prompt", "")).strip()
    return pending


def _prepare_repomap_context(ctx: click.Context, pending: str) -> None:
    """Populate RepoMap context when available."""
    try:
        from ai_dev_agent.cli.context_enhancer import enhance_query

        workspace = Path.cwd()
        _original_query, repomap_messages = enhance_query(pending, workspace)
        if repomap_messages:
            ctx.obj["_repomap_messages"] = repomap_messages
    except Exception as exc:  # pragma: no cover - best effort enrichment
        LOGGER.debug("RepoMap context unavailable: %s", exc)


def _resolve_planning_flag(ctx: click.Context, force_plan: bool, direct: bool) -> bool:
    """Resolve whether planning mode should be enabled."""
    use_planning = ctx.obj.get("default_use_planning", False)
    if force_plan:
        return True
    if direct:
        return False
    return bool(use_planning)


def execute_query(
    ctx: click.Context,
    state: "CLIState",
    prompt: Tuple[str, ...],
    force_plan: bool,
    direct: bool,
    agent: str,
) -> None:
    """Execute the query command using the shared CLI state."""
    pending = _resolve_pending_prompt(ctx, prompt)
    if not pending:
        raise click.UsageError("Provide a request for the assistant.")

    _prepare_repomap_context(ctx, pending)

    if not AgentRegistry.has_agent(agent):
        available = ", ".join(AgentRegistry.list_agents())
        raise click.UsageError(f"Unknown agent type '{agent}'. Available: {available}")

    _record_invocation(ctx, overrides={"prompt": pending, "mode": "query"})
    settings = state.settings

    use_planning = _resolve_planning_flag(ctx, force_plan, direct)

    if not settings.api_key:
        raise click.ClickException(
            "No API key configured (DEVAGENT_API_KEY). Natural language assistance requires an LLM."
        )

    try:
        client = get_llm_client(ctx)
    except click.ClickException as exc:
        raise click.ClickException(f"Failed to create LLM client: {exc}") from exc

    _execute_react_assistant(
        ctx,
        client,
        settings,
        pending,
        use_planning=use_planning,
        system_extension=None,
        format_schema=None,
        agent_type=agent,
    )


@click.command(name="query", help=QUERY_HELP, short_help=QUERY_SHORT_HELP)
@click.argument("prompt", nargs=-1)
@click.option("--plan", "force_plan", is_flag=True, help="Force planning for this query")
@click.option("--direct", is_flag=True, help="Force direct execution (no planning)")
@click.option("--agent", default="manager", help="Agent type: manager, reviewer (default: manager)")
@click.pass_context
def query_command(
    ctx: click.Context,
    prompt: Tuple[str, ...],
    force_plan: bool,
    direct: bool,
    agent: str,
) -> None:
    """Run the query command using the shared CLI runtime."""
    state = get_cli_state(ctx)
    inherited_direct = bool(ctx.obj.get("default_direct", False))
    meta_direct = bool(ctx.meta.get("_force_direct", False))
    effective_direct = direct or meta_direct or inherited_direct
    execute_query(ctx, state, prompt, force_plan, effective_direct, agent)
