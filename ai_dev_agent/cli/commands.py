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

LOGGER = get_logger(__name__)


class NaturalLanguageGroup(click.Group):
    """Group that falls back to NL intent routing when no command matches."""

    def resolve_command(self, ctx: click.Context, args: List[str]):  # type: ignore[override]
        planning_flag: Optional[bool] = None
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
            else:
                filtered_args.append(arg)
                i += 1

        # Store planning flag in ctx.meta if set
        if planning_flag is not None:
            ctx.meta["_use_planning"] = planning_flag

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
@click.option("-v", "--verbose", count=True, help="Verbosity: -v (info), -vv (debug), -vvv (trace)")
@click.option("-q", "--quiet", is_flag=True, help="Minimal output")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
@click.option("--plan", is_flag=True, help="Use planning mode for all queries")
@click.option("--repomap-debug", is_flag=True, envvar="DEVAGENT_REPOMAP_DEBUG", help="Enable RepoMap debug logging")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: int, quiet: bool, json_output: bool,
        plan: bool, repomap_debug: bool) -> None:
    """AI-assisted development agent CLI."""
    from ai_dev_agent.cli import load_settings as _load_settings

    settings = _load_settings(config_path)

    # Handle verbosity levels
    if quiet:
        settings.log_level = "WARNING"
    elif verbose == 0:
        settings.log_level = "INFO"
    elif verbose == 1:
        settings.log_level = "INFO"
    elif verbose == 2:
        settings.log_level = "DEBUG"
    elif verbose >= 3:
        settings.log_level = "DEBUG"

    if repomap_debug:
        settings.repomap_debug_stdout = True
        settings.log_level = "DEBUG"

    configure_logging(settings.log_level, structured=settings.structured_logging)

    if not settings.api_key:
        LOGGER.warning("No API key configured. Some commands may fail.")

    ctx.obj = _build_context(settings)
    ctx.obj["default_use_planning"] = plan
    ctx.obj["json_output"] = json_output
    ctx.obj["verbosity_level"] = verbose

    # When no subcommand or natural language input supplied, show help text
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return


@cli.command(name="query")
@click.argument("prompt", nargs=-1)
@click.option("--plan", "force_plan", is_flag=True, help="Force planning for this query")
@click.option("--direct", is_flag=True, help="Force direct execution (no planning)")
@click.option("--agent", default="manager", help="Agent type: manager, reviewer (default: manager)")
@click.pass_context
def query(
    ctx: click.Context,
    prompt: Tuple[str, ...],
    force_plan: bool,
    direct: bool,
    agent: str,
) -> None:
    """Execute a natural-language query using the ReAct workflow."""
    # Resolve prompt from args or context
    pending = " ".join(prompt).strip()
    if not pending:
        pending = str(ctx.meta.pop("_pending_nl_prompt", "")).strip()
    if not pending:
        pending = str(ctx.obj.pop("_pending_nl_prompt", "")).strip()

    if not pending:
        raise click.UsageError("Provide a request for the assistant.")

    # Get RepoMap context
    repomap_messages = None
    try:
        from ai_dev_agent.cli.context_enhancer import enhance_query
        workspace = Path.cwd()
        original_query, repomap_messages = enhance_query(pending, workspace)
        if repomap_messages:
            ctx.obj["_repomap_messages"] = repomap_messages
            logger.debug("RepoMap context prepared")
    except Exception as e:
        logger.debug(f"Could not prepare RepoMap context: {e}")

    # Validate agent type
    if not AgentRegistry.has_agent(agent):
        available = ", ".join(AgentRegistry.list_agents())
        raise click.UsageError(f"Unknown agent type '{agent}'. Available: {available}")

    _record_invocation(ctx, overrides={"prompt": pending, "mode": "query"})

    settings: Settings = ctx.obj["settings"]

    # Determine planning mode
    use_planning = ctx.obj.get("default_use_planning", False)
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
        system_extension=None,
        format_schema=None,
        agent_type=agent
    )


@cli.command()
@click.argument("file_path", required=True)
@click.option("--rule", type=click.Path(exists=True), help="Path to coding rule file (for patch/rule-based review)")
@click.option("--report", "-r", help="Output path for review report")
@click.option("--json", "json_flag", is_flag=True, help="Output in JSON format")
@click.pass_context
def review(
    ctx: click.Context,
    file_path: str,
    rule: Optional[str],
    report: Optional[str],
    json_flag: bool,
) -> None:
    """Review code for quality, security, and best practices.

    Can review source files or patches. When --rule is provided, performs
    rule-based patch review. Otherwise, performs general code quality review.

    Examples:
        devagent review src/auth.py
        devagent review changes.patch --rule rules/style.md
        devagent review src/app.ts --report review.md
    """
    settings: Settings = ctx.obj["settings"]
    json_output = json_flag or ctx.obj.get("json_output", False)

    # Rule-based patch review
    if rule:
        validated = run_review(
            ctx,
            patch_file=file_path,
            rule_file=rule,
            json_output=json_output,
            settings=settings,
        )

        if json_output:
            click.echo(json.dumps(validated, indent=2))
        else:
            summary = validated.get("summary", {})
            files_reviewed = summary.get("files_reviewed", 0)
            total_violations = summary.get("total_violations", 0)
            click.echo(f"Files reviewed: {files_reviewed}")
            click.echo(f"Violations: {total_violations}")
            discarded = summary.get("discarded_violations", 0)
            if discarded:
                click.echo(f"Discarded violations: {discarded} (ignored due to invalid references)")

            violations = validated.get("violations") or []
            if violations:
                click.echo("")
                click.echo("Findings:")
                for violation in violations:
                    severity = violation.get("severity", "warning")
                    location = f"{violation.get('file')}:{violation.get('line')}"
                    message = violation.get("message", "")
                    click.echo(f"- [{severity.upper()}] {location} - {message}")
            else:
                click.echo("No violations reported.")
    else:
        # General code quality review
        from ai_dev_agent.agents.specialized import ReviewAgent
        from ai_dev_agent.agents.base import AgentContext

        agent = ReviewAgent()
        agent_context = AgentContext(session_id=f"review-{Path(file_path).stem}")

        prompt = f"Review the code at {file_path} for security and performance issues"

        if not json_output:
            click.echo(f"🔍 Reviewing '{file_path}'...")

        result = agent.execute(prompt, agent_context)

        if result.success:
            issues = result.metadata.get("issues_found", 0)
            score = result.metadata.get("quality_score", 0.0)

            if json_output:
                click.echo(json.dumps({
                    "success": True,
                    "issues": issues,
                    "score": score,
                    "output": result.output,
                    "metadata": result.metadata
                }))
            else:
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




@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start an interactive chat session with persistent context.

    In chat mode, you can have a back-and-forth conversation with DevAgent.
    Your conversation history is maintained across queries in the same session.

    Type 'exit', 'quit', or 'q' to leave chat mode.
    """
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
        raise click.ClickException(f"Failed to start chat session: {exc}") from exc

    previous_manager = ctx.obj.get("_shell_session_manager")
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

            try:
                ctx.invoke(query, prompt=(user_input,))
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


@cli.command(name="create-design")
@click.argument("feature", required=True)
@click.option("--context", "-c", help="Additional context")
@click.option("--output", "-o", help="Output path for design document")
@click.pass_context
def create_design(ctx: click.Context, feature: str, output: Optional[str], context: Optional[str]):
    """Create a technical design for a feature.

    Example:
        devagent create-design "User Authentication System"
        devagent create-design "REST API" --context "CRUD for blog posts"
    """
    from ai_dev_agent.agents.specialized import DesignAgent
    from ai_dev_agent.agents.base import AgentContext

    agent = DesignAgent()
    agent_context = AgentContext(session_id=f"design-{feature}")

    prompt = f"Design {feature}"
    if context:
        prompt += f"\nContext: {context}"

    json_output = ctx.obj.get("json_output", False)
    if not json_output:
        click.echo(f"🎨 Creating design for '{feature}'...")

    result = agent.execute(prompt, agent_context)

    if result.success:
        if json_output:
            click.echo(json.dumps({
                "success": True,
                "output": result.output,
                "metadata": result.metadata
            }))
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


@cli.command(name="generate-tests")
@click.argument("feature", required=True)
@click.option("--coverage", "-c", default=90, help="Target coverage percentage")
@click.option("--type", "-t", type=click.Choice(["unit", "integration", "all"]), default="all")
@click.pass_context
def generate_tests(ctx: click.Context, feature: str, coverage: int, type: str):
    """Generate tests for a feature.

    Example:
        devagent generate-tests "authentication module"
        devagent generate-tests "payment processing" --coverage 95 --type integration
    """
    from ai_dev_agent.agents.specialized import TestingAgent
    from ai_dev_agent.agents.base import AgentContext

    agent = TestingAgent()
    agent_context = AgentContext(session_id=f"test-{feature}")

    prompt = f"Create {type} tests for {feature} with {coverage}% coverage"

    json_output = ctx.obj.get("json_output", False)
    if not json_output:
        click.echo(f"🧪 Generating tests for '{feature}'...")

    result = agent.execute(prompt, agent_context)

    if result.success:
        if json_output:
            click.echo(json.dumps({
                "success": True,
                "output": result.output,
                "metadata": result.metadata
            }))
        else:
            click.echo(click.style("✓ Tests generated", fg="green"))
            click.echo(f"\n{result.output}")

            if "test_files_created" in result.metadata:
                files = result.metadata["test_files_created"]
                click.echo(f"\nFiles created:")
                for f in files:
                    click.echo(f"  - {f}")
    else:
        if json_output:
            click.echo(json.dumps({"success": False, "error": result.error}))
        else:
            click.echo(click.style(f"✗ Failed: {result.error}", fg="red"))
        raise click.Abort()


@cli.command(name="write-code")
@click.argument("design_file", required=True)
@click.option("--test-file", "-t", help="Path to test file")
@click.pass_context
def write_code(ctx: click.Context, design_file: str, test_file: Optional[str]):
    """Implement code from a design file.

    Example:
        devagent write-code design.md
        devagent write-code design/auth.md --test-file tests/test_auth.py
    """
    from ai_dev_agent.agents.specialized import ImplementationAgent
    from ai_dev_agent.agents.base import AgentContext

    agent = ImplementationAgent()
    agent_context = AgentContext(session_id=f"implement-{Path(design_file).stem}")

    prompt = f"Implement the design at {design_file}"
    if test_file:
        prompt += f" with tests at {test_file}"

    json_output = ctx.obj.get("json_output", False)
    if not json_output:
        click.echo(f"⚙️  Implementing from '{design_file}'...")

    result = agent.execute(prompt, agent_context)

    if result.success:
        if json_output:
            click.echo(json.dumps({
                "success": True,
                "output": result.output,
                "metadata": result.metadata
            }))
        else:
            click.echo(click.style("✓ Implementation completed", fg="green"))
            click.echo(f"\n{result.output}")

            if "files_created" in result.metadata:
                click.echo(f"\nFiles created: {len(result.metadata['files_created'])}")
    else:
        if json_output:
            click.echo(json.dumps({"success": False, "error": result.error}))
        else:
            click.echo(click.style(f"✗ Failed: {result.error}", fg="red"))
        raise click.Abort()

def main() -> None:
    cli(prog_name="devagent")


if __name__ == "__main__":  # pragma: no cover
    main()
