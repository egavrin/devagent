"""Command line interface for the development agent."""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from ai_dev_agent.agents import AgentRegistry
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.logger import configure_logging, get_logger
from ai_dev_agent.tools.execution.shell_session import ShellSessionError, ShellSessionManager
from ai_dev_agent.session import SessionManager

from .react.executor import _execute_react_assistant
from .review import run_review
from .utils import _build_context, get_llm_client, _record_invocation

LOGGER = get_logger(__name__)


class NaturalLanguageGroup(click.Group):
    """Group that falls back to NL intent routing when no command matches."""

    def resolve_command(self, ctx: click.Context, args: List[str]):  # type: ignore[override]
        planning_flag: Optional[bool] = None
        system_value: Optional[str] = None
        prompt_value: Optional[str] = None
        format_value: Optional[str] = None
        filtered_args: List[str] = []
        i = 0

        while i < len(args):
            arg = args[i]
            if arg == "--plan":
                planning_flag = True
                i += 1
            elif arg == "--direct":
                planning_flag = False
                i += 1
            elif arg == "--system" and i + 1 < len(args):
                system_value = args[i + 1]
                i += 2
            elif arg == "--prompt" and i + 1 < len(args):
                prompt_value = args[i + 1]
                i += 2
            elif arg == "--format" and i + 1 < len(args):
                format_value = args[i + 1]
                i += 2
            else:
                filtered_args.append(arg)
                i += 1

        # Check if context already has global options set (from Click's parser)
        # ctx.params is populated after group options are parsed
        has_global_system = ctx.params.get('system') is not None if hasattr(ctx, 'params') else False
        has_global_prompt = ctx.params.get('prompt_global') is not None if hasattr(ctx, 'params') else False
        has_global_format = ctx.params.get('format_global') is not None if hasattr(ctx, 'params') else False

        # If we captured any of the new custom options (system/prompt/format), auto-route to query
        has_custom_opts = (system_value is not None or prompt_value is not None or
                          format_value is not None or has_global_system or
                          has_global_prompt or has_global_format)

        # Store captured values in ctx.meta so they're available regardless of routing path
        if planning_flag is not None:
            ctx.meta["_use_planning"] = planning_flag
        if system_value is not None:
            ctx.meta["_system_extension"] = system_value
        if prompt_value is not None:
            ctx.meta["_prompt_value"] = prompt_value
        if format_value is not None:
            ctx.meta["_format_file"] = format_value

        # If we have custom options but no command, auto-route to query
        if has_custom_opts and not filtered_args:
            ctx.meta["_emit_status_messages"] = True
            return super().resolve_command(ctx, ["query"])

        try:
            return super().resolve_command(ctx, filtered_args)
        except click.UsageError:
            # Original NL fallback logic for natural language queries
            if not filtered_args:
                raise
            if any(arg.startswith("-") for arg in filtered_args):
                raise
            query = " ".join(filtered_args).strip()
            if not query:
                raise
            ctx.meta["_pending_nl_prompt"] = query
            ctx.meta["_emit_status_messages"] = True
            return super().resolve_command(ctx, ["query"])


@click.group(cls=NaturalLanguageGroup, invoke_without_command=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), help="Path to config file.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging output.")
@click.option("--plan", is_flag=True, help="Use planning mode for all queries")
@click.option("--silent", is_flag=True, help="Suppress status messages and tool output (JSON-only mode)")
@click.option("--system", help="System prompt extension (string or file path)")
@click.option("--prompt", "prompt_global", help="User prompt from file or string")
@click.option("--format", "format_global", help="Output format JSON schema file path")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool, plan: bool, silent: bool,
        system: Optional[str], prompt_global: Optional[str], format_global: Optional[str]) -> None:
    """AI-assisted development agent CLI."""
    from ai_dev_agent.cli import load_settings as _load_settings

    settings = _load_settings(config_path)
    if verbose:
        settings.log_level = "DEBUG"
    configure_logging(settings.log_level, structured=settings.structured_logging)
    if not settings.api_key:
        LOGGER.warning("No API key configured. Some commands may fail.")
    ctx.obj = _build_context(settings)
    ctx.obj["default_use_planning"] = plan
    ctx.obj["silent_mode"] = silent

    # Store global options for use by query command
    if system:
        ctx.obj["_global_system"] = system
    if prompt_global:
        ctx.obj["_global_prompt"] = prompt_global
    if format_global:
        ctx.obj["_global_format"] = format_global

    # If custom options provided but no subcommand, auto-invoke query
    if ctx.invoked_subcommand is None and (system or prompt_global or format_global):
        ctx.invoke(query)


@cli.command(name="query")
@click.argument("prompt", nargs=-1)
@click.option("--plan", "force_plan", is_flag=True, help="Force planning for this query")
@click.option("--direct", is_flag=True, help="Force direct execution (no planning)")
@click.option("--agent", default="manager", help="Agent type: manager, reviewer (default: manager)")
@click.option("--system", help="System prompt extension (string or file path)")
@click.option("--prompt", "prompt_file", help="User prompt from file or string")
@click.option("--format", "format_file", help="Output format JSON schema file path")
@click.pass_context
def query(
    ctx: click.Context,
    prompt: Tuple[str, ...],
    force_plan: bool,
    direct: bool,
    agent: str,
    system: Optional[str],
    prompt_file: Optional[str],
    format_file: Optional[str],
) -> None:
    """Execute a natural-language query using the ReAct workflow."""
    # Check for values from NaturalLanguageGroup fallback
    meta_prompt = ctx.meta.pop("_prompt_value", None)
    meta_system = ctx.meta.pop("_system_extension", None)
    meta_format = ctx.meta.pop("_format_file", None)

    # Resolve prompt: --prompt option > meta > global > CLI args > context
    if prompt_file:
        pending = _resolve_input(prompt_file)
    elif meta_prompt:
        pending = _resolve_input(meta_prompt)
    elif ctx.obj.get("_global_prompt"):
        pending = _resolve_input(ctx.obj["_global_prompt"])
    else:
        pending = " ".join(prompt).strip()
        if not pending:
            pending = str(ctx.meta.pop("_pending_nl_prompt", "")).strip()
        if not pending:
            pending = str(ctx.obj.pop("_pending_nl_prompt", "")).strip()

    if not pending:
        raise click.UsageError("Provide a request for the assistant.")

    # Validate agent type
    if not AgentRegistry.has_agent(agent):
        available = ", ".join(AgentRegistry.list_agents())
        raise click.UsageError(f"Unknown agent type '{agent}'. Available: {available}")

    # Resolve system prompt extension (command option > meta > global)
    if system:
        system_extension = _resolve_input(system)
    elif meta_system:
        system_extension = _resolve_input(meta_system)
    elif ctx.obj.get("_global_system"):
        system_extension = _resolve_input(ctx.obj["_global_system"])
    else:
        system_extension = None

    # Load format schema (command option > meta > global)
    if format_file:
        format_schema = _load_json_schema(format_file)
    elif meta_format:
        format_schema = _load_json_schema(meta_format)
    elif ctx.obj.get("_global_format"):
        format_schema = _load_json_schema(ctx.obj["_global_format"])
    else:
        format_schema = None

    _record_invocation(ctx, overrides={"prompt": pending, "mode": "query"})

    settings: Settings = ctx.obj["settings"]

    planning_pref = ctx.meta.pop("_use_planning", None)
    if planning_pref is None:
        planning_pref = ctx.obj.get("default_use_planning", False)

    use_planning = bool(planning_pref)
    if getattr(settings, "always_use_planning", False):
        use_planning = True

    if force_plan:
        use_planning = True
    elif direct:
        use_planning = False

    if not settings.api_key:
        raise click.ClickException(
            "No API key configured (DEVAGENT_API_KEY). Natural language assistance requires an LLM."
        )

    try:
        cli_pkg = import_module('ai_dev_agent.cli')
        llm_factory = getattr(cli_pkg, 'get_llm_client', get_llm_client)
    except ModuleNotFoundError:
        llm_factory = get_llm_client
    try:
        client = llm_factory(ctx)
    except click.ClickException as exc:
        raise click.ClickException(f'Failed to create LLM client: {exc}') from exc
    _execute_react_assistant(
        ctx, client, settings, pending,
        use_planning=use_planning,
        system_extension=system_extension,
        format_schema=format_schema,
        agent_type=agent
    )


@cli.command()
@click.argument("patch_file", type=click.Path(exists=True))
@click.option("--rule", type=click.Path(exists=True), required=True, help="Path to coding rule file")
@click.option("--json", "json_output", is_flag=True, help="Output violations as JSON only")
@click.pass_context
def review(
    ctx: click.Context,
    patch_file: str,
    rule: str,
    json_output: bool,
) -> None:
    """Review a patch file against a coding rule.

    This is a specialized command that uses the reviewer agent to analyze
    patches for violations of coding standards or rules.

    Example:
        devagent review changes.patch --rule rules/jsdoc-required.md
        devagent review changes.patch --rule rules/jsdoc-required.md --json
    """
    settings: Settings = ctx.obj["settings"]

    validated = run_review(
        ctx,
        patch_file=patch_file,
        rule_file=rule,
        json_output=json_output,
        settings=settings,
    )

    if json_output:
        click.echo(json.dumps(validated, indent=2))
    else:
        click.echo(json.dumps(validated, indent=2))


@cli.command()
@click.pass_context
def shell(ctx: click.Context) -> None:
    """Start an interactive shell session with persistent context."""
    settings: Settings = ctx.obj["settings"]

    manager = ShellSessionManager(
        shell=getattr(settings, "shell_executable", None),
        default_timeout=getattr(settings, "shell_session_timeout", None),
        cpu_time_limit=getattr(settings, "shell_session_cpu_time_limit", None),
        memory_limit_mb=getattr(settings, "shell_session_memory_limit_mb", None),
    )

    try:
        session_id = manager.create_session(cwd=Path.cwd())
    except ShellSessionError as exc:
        raise click.ClickException(f"Failed to start shell session: {exc}") from exc

    previous_manager = ctx.obj.get("_shell_session_manager")
    previous_session = ctx.obj.get("_shell_session_id")
    previous_history = ctx.obj.get("_shell_conversation_history")
    ctx.obj["_shell_session_manager"] = manager
    ctx.obj["_shell_session_id"] = session_id
    ctx.obj["_shell_conversation_history"] = []

    click.echo("DevAgent Interactive Shell")
    click.echo("Type a question or command, 'help' for guidance, and 'exit' to quit.")
    click.echo("=" * 50)

    try:
        while True:
            try:
                user_input = click.prompt("DevAgent> ", prompt_suffix="", show_default=False).strip()
            except (KeyboardInterrupt, EOFError):
                click.echo("\nGoodbye!")
                break

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in {"exit", "quit", "q"}:
                click.echo("Goodbye!")
                break

            if lowered == "help":
                click.echo("Enter any natural-language request to run `devagent query`.")
                click.echo("Use 'exit' to leave the shell.")
                continue

            try:
                ctx.invoke(query, prompt=(user_input,))
            except ShellSessionError as exc:
                click.echo(f"Shell session error: {exc}")
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


@cli.command(name="diagnostics")
@click.option("--session", "session_id", help="Inspect a specific session ID (defaults to CLI session).")
@click.option("--plan", is_flag=True, help="Show planner sessions as well." )
@click.option("--router", is_flag=True, help="Include intent router history.")
@click.pass_context
def diagnostics(
    ctx: click.Context,
    session_id: Optional[str],
    plan: bool,
    router: bool,
) -> None:
    """Display conversation and metadata recorded by the session service."""

    manager = SessionManager.get_instance()

    target_ids = []
    if session_id:
        if not manager.has_session(session_id):
            raise click.ClickException(f"Session '{session_id}' not found")
        target_ids.append(session_id)
    else:
        cli_session = ctx.obj.get("_session_id")
        if cli_session and manager.has_session(cli_session):
            target_ids.append(cli_session)
        if plan:
            plan_session = ctx.obj.get("_planner_session_id")
            if plan_session and manager.has_session(plan_session):
                target_ids.append(plan_session)
        if router:
            router_session = getattr(ctx.obj.get("_router_state", {}), "get", lambda _x: None)("session_id")
            if router_session and manager.has_session(router_session):
                target_ids.append(router_session)

    if not target_ids:
        raise click.ClickException("No sessions available to inspect. Provide --session or run a query first.")

    for idx, sid in enumerate(dict.fromkeys(target_ids), start=1):
        session = manager.get_session(sid)
        click.echo(f"\n=== Session {idx}: {sid} ===")
        with session.lock:
            if session.metadata:
                click.echo("Metadata:")
                for key, value in session.metadata.items():
                    click.echo(f"  - {key}: {value}")
            else:
                click.echo("Metadata: <none>")

            if session.system_messages:
                click.echo("\nSystem Prompts:")
                for message in session.system_messages:
                    click.echo(f"  [{message.role}] {message.content[:200]}" + ("..." if message.content and len(message.content) > 200 else ""))

            if session.history:
                click.echo("\nHistory:")
                for message in session.history:
                    snippet = (message.content or "").strip()
                    if snippet and len(snippet) > 200:
                        snippet = snippet[:197] + "..."
                    if message.role == "tool" and message.tool_call_id:
                        click.echo(f"  [tool:{message.tool_call_id}] {snippet}")
                    elif message.role == "assistant" and message.tool_calls:
                        click.echo(f"  [assistant tool-calls] {snippet}")
                    else:
                        click.echo(f"  [{message.role}] {snippet}")
            else:
                click.echo("\nHistory: <empty>")


def _resolve_input(value: str) -> str:
    """Resolve input: if path exists and is a file, read it; otherwise return as-is."""
    if not value:
        return ""
    path = Path(value).expanduser()
    if path.is_file():
        try:
            return path.read_text(encoding='utf-8')
        except Exception as exc:
            raise click.ClickException(f"Failed to read file '{value}': {exc}") from exc
    return value


def _load_json_schema(path: str) -> Optional[Dict[str, Any]]:
    """Load and parse JSON schema from file."""
    if not path:
        return None
    schema_path = Path(path).expanduser()
    if not schema_path.is_absolute():
        schema_path = Path.cwd() / schema_path
    if not schema_path.is_file():
        raise click.ClickException(f"Schema file not found: {path}")
    try:
        return json.loads(schema_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in schema file '{path}': {exc}") from exc
    except Exception as exc:
        raise click.ClickException(f"Failed to read schema file '{path}': {exc}") from exc


def main() -> None:
    cli(prog_name="devagent")


if __name__ == "__main__":  # pragma: no cover
    main()
