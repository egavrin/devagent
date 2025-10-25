"""Command line interface for the development agent."""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from ai_dev_agent.agents import AgentRegistry
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.logger import configure_logging, get_logger
from ai_dev_agent.tools.execution.shell_session import ShellSessionError, ShellSessionManager
from ai_dev_agent.session import SessionManager

logger = get_logger(__name__)

from .react.executor import _execute_react_assistant
from .review import run_review
from .utils import _build_context, get_llm_client, _record_invocation
from .agent_commands import agent_group
from .auto_agent_query import auto_agent_command

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
        except click.UsageError as exc:
            # Original NL fallback logic for natural language queries
            if not filtered_args:
                raise
            # Treat single CLI-like tokens (eg `foo-bar`) as real commands and surface the original error
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


@click.group(cls=NaturalLanguageGroup, invoke_without_command=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), help="Path to config file.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging output.")
@click.option("--plan", is_flag=True, help="Use planning mode for all queries")
@click.option("--silent", is_flag=True, help="Suppress status messages and tool output (JSON-only mode)")
@click.option("--repomap-debug", is_flag=True, envvar="DEVAGENT_REPOMAP_DEBUG", help="Enable RepoMap debug logging")
@click.option("--system", help="System prompt extension (string or file path)")
@click.option("--prompt", "prompt_global", help="User prompt from file or string")
@click.option("--format", "format_global", help="Output format JSON schema file path")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool, plan: bool, silent: bool,
        repomap_debug: bool, system: Optional[str], prompt_global: Optional[str], format_global: Optional[str]) -> None:
    """AI-assisted development agent CLI."""
    from ai_dev_agent.cli import load_settings as _load_settings

    settings = _load_settings(config_path)
    if verbose:
        settings.log_level = "DEBUG"
    if repomap_debug:
        settings.repomap_debug_stdout = True
        settings.log_level = "DEBUG"  # Enable DEBUG level when repomap debug is on
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
        ctx.invoke(
            query,
            prompt=(),
            force_plan=False,
            direct=False,
            agent="manager",
            system=system,
            prompt_file=prompt_global,
            format_file=format_global,
        )
        return

    # When no subcommand or natural language input supplied, show help text
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return


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

    # Get RepoMap context as conversation messages (Aider's approach)
    repomap_messages = None
    try:
        from ai_dev_agent.cli.context_enhancer import enhance_query
        workspace = Path.cwd()
        original_query, repomap_messages = enhance_query(pending, workspace)
        if repomap_messages:
            # Store messages in ctx for executor to inject into conversation
            ctx.obj["_repomap_messages"] = repomap_messages
            logger.debug("RepoMap context prepared as conversation messages")
    except Exception as e:
        logger.debug(f"Could not prepare RepoMap context: {e}")
        # Continue without RepoMap context

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
        start_session = getattr(manager, "start_session", None)
        if callable(start_session):
            session_id = start_session(cwd=Path.cwd())
        else:
            session_id = manager.create_session(cwd=Path.cwd())
    except ShellSessionError as exc:
        raise click.ClickException(f"Failed to start shell session: {exc}") from exc

    previous_manager = ctx.obj.get("_shell_session_manager")
    previous_session = ctx.obj.get("_shell_session_id")
    previous_history = ctx.obj.get("_shell_conversation_history")
    ctx.obj["_shell_session_manager"] = manager
    ctx.obj["_shell_session_id"] = session_id
    ctx.obj["_shell_conversation_history"] = []

    is_active_fn = getattr(manager, "is_session_active", None)
    if callable(is_active_fn):
        try:
            is_active = bool(is_active_fn(session_id))
        except TypeError:
            is_active = True
        if not is_active:
            click.echo("Shell session is not active.")
            return

    click.echo("DevAgent Interactive Shell")
    click.echo("Type a question or command, 'help' for guidance, and 'exit' to quit.")
    click.echo("=" * 50)

    try:
        while True:
            if callable(is_active_fn):
                try:
                    if not is_active_fn(session_id):
                        click.echo("Shell session closed.")
                        break
                except TypeError:
                    pass
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
        click.echo("No sessions available to inspect. Provide --session or run a session-producing query first.")
        return

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


# =============================================================================
# Work Planning Commands
# =============================================================================

@cli.group(name="plan")
def plan_group():
    """Manage work plans and tasks."""
    pass


def _resolve_plan_id(agent, partial_id: str):
    """
    Resolve a partial plan ID to a full plan, checking for ambiguity.

    Returns:
        tuple: (plan, error_message)
        - If successful: (WorkPlan, None)
        - If failed: (None, error_string)
    """
    plans = agent.storage.list_plans()
    matching_plans = [p for p in plans if p.id.startswith(partial_id)]

    if not matching_plans:
        return None, f"❌ Plan not found: {partial_id}"

    if len(matching_plans) > 1:
        error = f"❌ Ambiguous plan ID. Multiple plans match '{partial_id}':\n"
        for p in matching_plans:
            error += f"   - {p.id}: {p.goal}\n"
        error += "Please provide a longer prefix to uniquely identify the plan."
        return None, error

    return matching_plans[0], None


@plan_group.command(name="create")
@click.argument("goal")
@click.option("--context", "-c", help="Additional context for the plan")
@click.pass_context
def plan_create(ctx: click.Context, goal: str, context: Optional[str]):
    """Create a new work plan.

    Example:
        devagent plan create "Implement user authentication" --context "JWT-based auth with refresh tokens"
    """
    from ai_dev_agent.agents.work_planner import WorkPlanningAgent

    agent = WorkPlanningAgent()
    plan = agent.create_plan(
        goal=goal,
        context={"description": context} if context else {}
    )

    click.echo(f"✓ Created work plan: {plan.id}")
    click.echo(f"  Goal: {plan.goal}")
    click.echo(f"  Tasks: {len(plan.tasks)}")
    click.echo(f"\nUse 'devagent plan show {plan.id}' to view the plan")
    click.echo(f"Use 'devagent plan next {plan.id}' to get the next task")
    click.echo(f"Use 'devagent plan start {plan.id} <task_id>' to start a task")


@plan_group.command(name="list")
def plan_list():
    """List all work plans."""
    from ai_dev_agent.agents.work_planner import WorkPlanningAgent

    agent = WorkPlanningAgent()
    plans = agent.storage.list_plans()

    if not plans:
        click.echo("No work plans found.")
        return

    click.echo(f"Found {len(plans)} work plan(s):\n")
    for i, plan in enumerate(plans, 1):
        progress = plan.get_completion_percentage()
        status = "✅" if progress == 100 else "🔄" if progress > 0 else "⏳"
        click.echo(f"{i}. {status} {plan.id[:8]}... - {plan.goal}")
        click.echo(f"   Progress: {progress:.0f}% ({len([t for t in plan.tasks if t.status.value == 'completed'])}/{len(plan.tasks)} tasks)")


@plan_group.command(name="show")
@click.argument("plan_id")
def plan_show(plan_id: str):
    """Show details of a work plan.

    Example:
        devagent plan show abc123
    """
    from ai_dev_agent.agents.work_planner import WorkPlanningAgent

    agent = WorkPlanningAgent()

    # Try to find plan by partial ID
    plans = agent.storage.list_plans()
    matching_plans = [p for p in plans if p.id.startswith(plan_id)]

    if not matching_plans:
        click.echo(f"❌ Plan not found: {plan_id}")
        return

    if len(matching_plans) > 1:
        click.echo(f"❌ Ambiguous plan ID. Multiple plans match '{plan_id}':")
        for p in matching_plans:
            click.echo(f"   - {p.id}: {p.goal}")
        return

    plan = matching_plans[0]
    summary = agent.get_plan_summary(plan.id)
    click.echo(summary)


@plan_group.command(name="next")
@click.argument("plan_id")
def plan_next(plan_id: str):
    """Get the next task to work on.

    Example:
        devagent plan next abc123
    """
    from ai_dev_agent.agents.work_planner import WorkPlanningAgent

    agent = WorkPlanningAgent()

    # Resolve plan ID with ambiguity check
    plan, error = _resolve_plan_id(agent, plan_id)
    if error:
        click.echo(error)
        return

    next_task = agent.get_next_task(plan.id)

    if not next_task:
        click.echo("✅ No more tasks available - plan complete!")
        return

    click.echo(f"Next task: {next_task.title}")
    click.echo(f"  Priority: {next_task.priority.value}")
    click.echo(f"  Effort: {next_task.effort_estimate}")
    click.echo(f"  Description: {next_task.description}")
    if next_task.acceptance_criteria:
        click.echo(f"  Acceptance criteria:")
        for criterion in next_task.acceptance_criteria:
            click.echo(f"    - {criterion}")
    click.echo(f"\nTask ID: {next_task.id}")
    click.echo(f"Use 'devagent plan start {plan.id} {next_task.id}' to begin")


@plan_group.command(name="start")
@click.argument("plan_id")
@click.argument("task_id")
def plan_start(plan_id: str, task_id: str):
    """Mark a task as started.

    Example:
        devagent plan start abc123 def456
    """
    from ai_dev_agent.agents.work_planner import WorkPlanningAgent

    agent = WorkPlanningAgent()

    # Resolve plan ID with ambiguity check
    plan, error = _resolve_plan_id(agent, plan_id)
    if error:
        click.echo(error)
        return

    # Find task
    task = plan.get_task(task_id) or next((t for t in plan.tasks if t.id.startswith(task_id)), None)

    if not task:
        click.echo(f"❌ Task not found: {task_id}")
        return

    agent.mark_task_started(plan.id, task.id)
    click.echo(f"✓ Started task: {task.title}")


@plan_group.command(name="complete")
@click.argument("plan_id")
@click.argument("task_id")
def plan_complete(plan_id: str, task_id: str):
    """Mark a task as completed.

    Example:
        devagent plan complete abc123 def456
    """
    from ai_dev_agent.agents.work_planner import WorkPlanningAgent

    agent = WorkPlanningAgent()

    # Resolve plan ID with ambiguity check
    plan, error = _resolve_plan_id(agent, plan_id)
    if error:
        click.echo(error)
        return

    # Find task
    task = plan.get_task(task_id) or next((t for t in plan.tasks if t.id.startswith(task_id)), None)

    if not task:
        click.echo(f"❌ Task not found: {task_id}")
        return

    agent.mark_task_complete(plan.id, task.id)

    # Show progress
    updated_plan = agent.storage.load_plan(plan.id)
    progress = updated_plan.get_completion_percentage()

    click.echo(f"✓ Completed task: {task.title}")
    click.echo(f"  Overall progress: {progress:.0f}%")


@plan_group.command(name="delete")
@click.argument("plan_id")
@click.confirmation_option(prompt="Are you sure you want to delete this plan?")
def plan_delete(plan_id: str):
    """Delete a work plan.

    Example:
        devagent plan delete abc123
    """
    from ai_dev_agent.agents.work_planner import WorkPlanningAgent

    agent = WorkPlanningAgent()

    # Resolve plan ID with ambiguity check
    plan, error = _resolve_plan_id(agent, plan_id)
    if error:
        click.echo(error)
        return

    if agent.storage.delete_plan(plan.id):
        click.echo(f"✓ Deleted plan: {plan.goal}")
    else:
        click.echo(f"❌ Failed to delete plan")


# =============================================================================
# Multi-Agent System Commands
# =============================================================================

# Register the agent command group from agent_commands.py
cli.add_command(agent_group)

# Register the auto agent command for automatic agent spawning
cli.add_command(auto_agent_command)


def main() -> None:
    cli(prog_name="devagent")


if __name__ == "__main__":  # pragma: no cover
    main()
