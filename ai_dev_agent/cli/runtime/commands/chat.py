"""Interactive chat command implemented for CLI runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from ai_dev_agent.tools.execution.shell_session import ShellSessionError, ShellSessionManager

from .._compat import get_cli_state


def execute_chat(ctx: click.Context, state) -> None:
    """Shared chat implementation leveraging CLIState."""
    settings = state.settings

    manager = ShellSessionManager(
        shell=getattr(settings, "shell_executable", None),
        default_timeout=getattr(settings, "shell_session_timeout", None),
        cpu_time_limit=getattr(settings, "shell_session_cpu_time_limit", None),
        memory_limit_mb=getattr(settings, "shell_session_memory_limit_mb", None),
    )

    try:
        session_id = manager.create_session(cwd=Path.cwd())
    except ShellSessionError as exc:  # pragma: no cover - defensive
        raise click.ClickException(f"Failed to start chat session: {exc}") from exc

    previous_manager: Optional[ShellSessionManager] = ctx.obj.get("_shell_session_manager")
    previous_session = ctx.obj.get("_shell_session_id")
    previous_history = ctx.obj.get("_shell_conversation_history")
    ctx.obj["_shell_session_manager"] = manager
    ctx.obj["_shell_session_id"] = session_id
    ctx.obj["_shell_conversation_history"] = []

    click.echo("DevAgent Chat")
    click.echo("Ask questions, request features, or get help. Type 'exit' to quit.")
    click.echo("=" * 60)

    try:
        while True:
            try:
                user_input = click.prompt(
                    "DevAgent> ", prompt_suffix="", show_default=False
                ).strip()
            except (KeyboardInterrupt, EOFError):
                click.echo("\nGoodbye!")
                break

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in {"exit", "quit", "q"}:
                click.echo("Goodbye!")
                break

            try:
                parent_ctx = ctx.parent or ctx
                query_command = parent_ctx.command.get_command(parent_ctx, "query")
                if query_command is None:  # pragma: no cover - defensive guard
                    raise click.ClickException("Query command unavailable during chat session")
                parent_ctx.invoke(query_command, prompt=(user_input,))
            except ShellSessionError as exc:
                click.echo(f"Chat session error: {exc}")
                break
            except TimeoutError as exc:
                click.echo(f"Command timed out: {exc}")
            except click.ClickException as exc:
                click.echo(f"Error: {exc}")
    finally:
        if previous_manager is not None:
            ctx.obj["_shell_session_manager"] = previous_manager
        else:
            ctx.obj.pop("_shell_session_manager", None)

        if previous_session is not None:
            ctx.obj["_shell_session_id"] = previous_session
        else:
            ctx.obj.pop("_shell_session_id", None)

        if previous_history is not None:
            ctx.obj["_shell_conversation_history"] = previous_history
        else:
            ctx.obj.pop("_shell_conversation_history", None)

        manager.close_all()


@click.command(
    name="chat", help="Start an interactive chat session.", short_help="Interactive chat"
)
@click.pass_context
def chat_command(ctx: click.Context) -> None:
    """Start an interactive chat session with CLIState context."""
    state = get_cli_state(ctx)
    execute_chat(ctx, state)
