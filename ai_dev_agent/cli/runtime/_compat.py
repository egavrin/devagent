"""Compatibility helpers for CLI runtime commands."""

from __future__ import annotations

import click


def get_cli_state(ctx: click.Context):
    """Retrieve CLIState from context, ensuring it exists."""
    state = ctx.obj.get("cli_state") if ctx.obj else None
    if state is None:
        raise click.ClickException("CLI state missing; CLI runtime not initialised correctly.")
    return state
