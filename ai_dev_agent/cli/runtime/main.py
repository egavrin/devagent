"""CLI runtime main entrypoint preparing shared state and delegating to commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import click

from ai_dev_agent.core.context.builder import ContextBuilder
from ai_dev_agent.core.utils.logger import configure_logging, get_logger
from ai_dev_agent.prompts import PromptLoader

if TYPE_CHECKING:
    from ai_dev_agent.core.utils.config import Settings

from .. import load_settings
from ..utils import _build_context  # Shared context builder

LOGGER = get_logger(__name__)


class NaturalLanguageGroup(click.Group):
    """Group that falls back to NL intent routing when no command matches."""

    def resolve_command(self, ctx: click.Context, args: list[str]):  # type: ignore[override]
        planning_flag: bool | None = None
        filtered_args: list[str] = []
        index = 0

        while index < len(args):
            arg = args[index]
            if arg == "--plan":
                planning_flag = True
                index += 1
            elif arg == "--direct":
                planning_flag = False
                index += 1
            else:
                filtered_args.append(arg)
                index += 1

        if planning_flag is not None:
            ctx.meta["_use_planning"] = planning_flag

        try:
            return super().resolve_command(ctx, filtered_args)
        except click.UsageError as exc:
            if not filtered_args:
                raise
            if len(filtered_args) == 1 and "-" in filtered_args[0]:
                raise exc
            if any(arg.startswith("-") for arg in filtered_args):
                raise
            query = " ".join(filtered_args).strip()
            if not query:
                raise
            ctx.meta["_pending_nl_prompt"] = query
            ctx.meta["_emit_status_messages"] = True
            return super().resolve_command(ctx, ["query"])


@dataclass
class CLIState:
    """Holds shared objects for CLI runtime commands."""

    settings: "Settings"
    prompt_loader: PromptLoader
    context_builder: ContextBuilder
    system_context: Dict[str, Any]
    project_context: Dict[str, Any]


def _resolve_log_level(verbose: int, quiet: bool) -> str:
    """Resolve log level based on verbosity flags."""
    if quiet:
        return "WARNING"
    if verbose >= 2:
        return "DEBUG"
    if verbose == 1:
        return "INFO"
    return "INFO"


def _initialise_state(
    config_path: Optional[Path],
    *,
    verbose: int,
    quiet: bool,
    repomap_debug: bool,
) -> tuple["Settings", dict, CLIState]:
    """Load settings, configure logging, and prepare shared state."""
    settings = load_settings(config_path)

    log_level = _resolve_log_level(verbose, quiet)
    if repomap_debug:
        settings.repomap_debug_stdout = True
        log_level = "DEBUG"

    settings.log_level = log_level
    configure_logging(log_level, structured=settings.structured_logging)

    prompt_loader = PromptLoader()
    context_builder = ContextBuilder(Path.cwd())
    system_context = context_builder.build_system_context()
    project_context = context_builder.build_project_context()

    cli_context = _build_context(settings)
    state = CLIState(
        settings=settings,
        prompt_loader=prompt_loader,
        context_builder=context_builder,
        system_context=system_context,
        project_context=project_context,
    )

    LOGGER.debug("CLI runtime state initialised with prompts at %s", prompt_loader.prompts_dir)
    return settings, cli_context, state


@click.group(cls=NaturalLanguageGroup, invoke_without_command=True)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    help="Path to configuration file.",
)
@click.option("-v", "--verbose", count=True, help="Verbosity: -v (info), -vv (debug), -vvv (trace)")
@click.option("-q", "--quiet", is_flag=True, help="Minimal output")
@click.option("--json", "json_output", is_flag=True, help="Emit JSON output")
@click.option("--plan", is_flag=True, help="Use planning mode by default")
@click.option(
    "--repomap-debug",
    is_flag=True,
    envvar="DEVAGENT_REPOMAP_DEBUG",
    help="Enable RepoMap debug logging",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Optional[Path],
    verbose: int,
    quiet: bool,
    json_output: bool,
    plan: bool,
    repomap_debug: bool,
) -> None:
    """DevAgent - AI-powered development assistant.

    Execute natural language queries, generate code, review changes, and more.
    Unrecognized commands are automatically interpreted as queries.

    Examples:
      devagent "fix the bug in auth.py"
      devagent query --plan "refactor the database layer"
      devagent chat
      devagent review src/app.py
    """
    ctx.ensure_object(dict)

    settings, cli_context, state = _initialise_state(
        config_path,
        verbose=verbose,
        quiet=quiet,
        repomap_debug=repomap_debug,
    )

    ctx.obj.update(
        {
            "cli_state": state,
            "settings": settings,
            "state": cli_context.get("state"),
            "cli_ctx": cli_context,
            "json_output": json_output,
            "default_use_planning": plan,
            "verbosity_level": verbose,
            "prompt_loader": state.prompt_loader,
            "context_builder": state.context_builder,
            "system_context": state.system_context,
            "project_context": state.project_context,
        }
    )

    cli_context["json_output"] = json_output
    cli_context["default_use_planning"] = plan
    cli_context["verbosity_level"] = verbose


def _register_commands() -> None:
    """Register CLI runtime commands (lazy import to avoid cycles)."""
    from .commands import (
        chat_command,
        create_design_command,
        generate_tests_command,
        query_command,
        review_command,
        write_code_command,
    )

    for command in (
        query_command,
        review_command,
        chat_command,
        create_design_command,
        generate_tests_command,
        write_code_command,
    ):
        if command.name not in cli.commands:
            cli.add_command(command)


# Ensure commands are registered on import so help output is complete.
_register_commands()
