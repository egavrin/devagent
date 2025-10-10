"""Execution helpers for the CLI ReAct workflow."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import uuid4

import click

logger = logging.getLogger(__name__)

from ai_dev_agent.agents import AgentRegistry
from ai_dev_agent.cli.handlers import INTENT_HANDLERS
from ai_dev_agent.cli.utils import (
    _collect_project_structure_outline,
    _detect_repository_language,
    _get_structure_hints_state,
    _merge_structure_hints_state,
    _update_files_discovered,
)
from ai_dev_agent.cli.dynamic_context import DynamicContextTracker
from ai_dev_agent.core.utils.budget_integration import BudgetIntegration, create_budget_integration
from ai_dev_agent.core.utils.config import DEFAULT_MAX_ITERATIONS, Settings
from ai_dev_agent.core.utils.devagent_config import load_devagent_yaml
from ai_dev_agent.providers.llm import LLMError
from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session import SessionManager
from ai_dev_agent.session.context_synthesis import ContextSynthesizer

from ai_dev_agent.engine.react.loop import ReactiveExecutor
from ai_dev_agent.engine.react.tool_invoker import SessionAwareToolInvoker
from ai_dev_agent.engine.react.types import (
    EvaluationResult,
    MetricsSnapshot,
    RunResult,
    StepRecord,
    TaskSpec,
)

from ..router import IntentDecision, IntentRouter as _DEFAULT_INTENT_ROUTER
from .action_provider import LLMActionProvider
from .budget_control import AdaptiveBudgetManager, BudgetManager, PHASE_PROMPTS, auto_generate_summary

__all__ = ["_execute_react_assistant"]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text, handling markdown code fences and surrounding text."""
    if not text:
        return None

    # Try to parse the entire text as JSON first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code fence (greedy to get full content)
    code_fence_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if code_fence_match:
        try:
            return json.loads(code_fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find JSON by matching braces/brackets with proper nesting
    # Try objects first (most common), then arrays
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        # Find first occurrence of start character
        start_pos = text.find(start_char)
        if start_pos == -1:
            continue

        # Track nesting depth to find matching closing character
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_pos, len(text)):
            char = text[i]

            # Handle string literals (JSON strings can contain braces/brackets)
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue

            # Only count braces/brackets outside of strings
            if not in_string:
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        # Found complete JSON structure
                        candidate = text[start_pos:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # This wasn't valid JSON, keep searching
                            break

    return None


class BudgetAwareExecutor(ReactiveExecutor):
    """Thin wrapper around the engine executor that honours the CLI budget manager."""

    def __init__(self, budget_manager: BudgetManager) -> None:
        super().__init__(evaluator=None)
        self.budget_manager = budget_manager

    def run(
        self,
        task: TaskSpec,
        action_provider: LLMActionProvider,
        tool_invoker: SessionAwareToolInvoker,
        *,
        prior_steps: Optional[Iterable[StepRecord]] = None,
        iteration_hook: Optional[Callable[[Any, Sequence[StepRecord]], None]] = None,
        step_hook: Optional[Callable[[StepRecord, Any], None]] = None,
    ) -> RunResult:
        history: List[StepRecord] = list(prior_steps or [])
        steps: List[StepRecord] = []
        natural_stop = False
        stop_condition: Optional[str] = None
        start_time = time.perf_counter()

        while True:
            context = self.budget_manager.next_iteration()
            if context is None:
                stop_condition = "budget_exhausted"
                break

            action_provider.update_phase(context.phase, is_final=context.is_final)
            if iteration_hook:
                iteration_hook(context, history + steps)

            try:
                action_payload = action_provider(task, history + steps)
            except StopIteration:
                stop_condition = "provider_stop"
                natural_stop = True
                break

            step_index = len(history) + len(steps) + 1
            action = self._ensure_action(action_payload, step_index)
            observation = self._invoke_tool(tool_invoker, action)
            metrics = self._metrics_from_observation(observation)
            evaluation = EvaluationResult(
                gates={},
                required_gates={},
                should_stop=context.is_final,
                stop_reason="Budget exhausted" if context.is_final else None,
                next_action_hint=None,
                improved_metrics={},
                status="success" if (context.is_final and observation.success) else "in_progress",
            )
            record = StepRecord(
                action=action,
                observation=observation,
                metrics=metrics,
                evaluation=evaluation,
                step_index=step_index,
            )
            steps.append(record)
            if step_hook:
                step_hook(record, context)
            if context.is_final:
                stop_condition = "final_iteration"
                natural_stop = True
                break
            else:
                stop_condition = "next_iteration"

        runtime = time.perf_counter() - start_time
        last_record = steps[-1] if steps else None
        last_observation = last_record.observation if last_record else None

        successful_result = False
        if last_observation is not None:
            successful_result = last_observation.success and stop_condition in {"final_iteration", "provider_stop"}
        else:
            successful_result = natural_stop and stop_condition in {"provider_stop"}

        status = "success" if successful_result else "failed"

        if not last_record:
            if successful_result:
                stop_reason = "Completed"
            elif stop_condition == "budget_exhausted":
                stop_reason = "No actions executed"
            else:
                stop_reason = "No actions executed"
        elif not last_observation.success:
            stop_reason = (
                last_observation.outcome
                or last_observation.error
                or last_record.evaluation.stop_reason
                or "Tool execution failed"
            )
        elif last_record.evaluation.stop_reason:
            eval_reason = last_record.evaluation.stop_reason
            if eval_reason == "Budget exhausted" and last_observation.success:
                stop_reason = "Completed"
            else:
                stop_reason = eval_reason
        elif stop_condition == "budget_exhausted":
            stop_reason = "Budget exhausted"
        else:
            stop_reason = "Completed"

        metrics_dict = steps[-1].metrics.model_dump() if steps else {}
        return RunResult(
            task_id=task.identifier,
            status=status,
            steps=history + steps,
            gates={},
            required_gates={},
            stop_reason=stop_reason,
            runtime_seconds=round(runtime, 3),
            metrics=metrics_dict,
        )


def _resolve_intent_router() -> type[_DEFAULT_INTENT_ROUTER]:
    try:
        cli_module = import_module("ai_dev_agent.cli")
    except ModuleNotFoundError:
        return _DEFAULT_INTENT_ROUTER
    return getattr(cli_module, "IntentRouter", _DEFAULT_INTENT_ROUTER)


def _truncate_shell_history(history: List[Message], max_turns: int) -> List[Message]:
    if max_turns <= 0:
        return []

    turns: List[tuple[Message, Message]] = []
    pending_user: Message | None = None
    for msg in history:
        if msg.role == "user":
            pending_user = msg
        elif msg.role == "assistant" and msg.content is not None:
            if pending_user is not None:
                turns.append((pending_user, msg))
                pending_user = None

    trimmed: List[Message] = []
    for user_msg, assistant_msg in turns[-max_turns:]:
        trimmed.extend([user_msg, assistant_msg])

    if pending_user is not None:
        if not trimmed or trimmed[-1] is not pending_user:
            trimmed.append(pending_user)

    return trimmed


def _build_phase_prompt(
    phase: str,
    user_query: str,
    context: str,
    constraints: str,
    *,
    workspace: str,
    repository_language: Optional[str],
) -> str:
    guidance = PHASE_PROMPTS.get(phase) or PHASE_PROMPTS.get("exploration", "Focus on the task at hand.")
    lang_hint = ""
    if repository_language:
        hint_map = {
            "python": "Consider Python-specific tooling and packaging.",
            "javascript": "Inspect package.json and JS/TS conventions.",
            "java": "Check build configuration and Java idioms.",
            "go": "Review go.mod and Go formatting rules.",
        }
        lang_hint = hint_map.get(repository_language.lower(), "")

    show_discoveries = bool(context.strip())

    prompt_parts = [
        "You are a development assistant analysing a codebase.",
        "",
        f"TASK: {user_query}",
        "",
        "APPROACH:",
        guidance,
        "",
        f"WORKSPACE: {workspace}",
    ]
    if repository_language:
        prompt_parts.append(f"LANGUAGE: {repository_language}{f' — {lang_hint}' if lang_hint else ''}")
    if show_discoveries:
        prompt_parts.extend(["", "PREVIOUS DISCOVERIES:", context])
    if constraints.strip():
        prompt_parts.extend(["", "CONSTRAINTS:", constraints])
    return "\n".join(prompt_parts)


def _build_synthesis_prompt(
    user_query: str,
    context: str,
    *,
    workspace: str,
) -> str:
    guidance = PHASE_PROMPTS.get("synthesis", "Provide your final response.")
    prompt = [
        "📋 FINAL SYNTHESIS ONLY",
        "",
        f"Task: {user_query}",
        "",
        guidance,
        "",
        f"Workspace: {workspace}",
        "",
        "Investigation Summary:",
        context if context else "No prior findings recorded.",
        "",
        "Instructions:",
        "- Respond with a complete, self-contained answer.",
        "- Cite specific files, paths, and relevant context.",
        "- Call out open questions or risks still unresolved.",
        "- DO NOT request more iterations or attempt tool usage.",
        "",
        "Begin your final answer now.",
    ]
    return "\n".join(prompt)


def _record_search_query(action: ActionRequest, search_queries: set[str]) -> None:
    if action.tool not in {"find", "grep"}:
        return
    candidate: Optional[str] = None
    for key in ("query", "pattern"):
        value = action.args.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            break
    if candidate:
        search_queries.add(candidate)


def _execute_react_assistant(
    ctx: click.Context,
    client,
    settings: Settings,
    user_prompt: str,
    use_planning: bool = False,
    system_extension: Optional[str] = None,
    format_schema: Optional[Dict[str, Any]] = None,
    agent_type: str = "manager",
    suppress_final_output: bool = False,
    ) -> Dict[str, Any]:
    """Execute the CLI ReAct loop using the shared engine primitives.

    Returns a dictionary containing:
        final_message: Raw assistant text from the last iteration (if any)
        final_json: Parsed JSON payload when format_schema enforced
        result: RunResult instance summarizing execution
        printed_final: Whether JSON/text was already emitted to stdout
    """

    start_time = time.time()
    planning_active = bool(use_planning)
    supports_tool_calls = hasattr(client, "invoke_tools")
    truncated_prompt = user_prompt if len(user_prompt) <= 50 else f"{user_prompt[:50]}..."

    # Check for silent mode
    if not isinstance(getattr(ctx, "obj", None), dict):
        ctx.obj = {}
    ctx_obj: Dict[str, Any] = ctx.obj
    ctx_obj["settings"] = settings  # Store settings for access by action_provider
    silent_mode = ctx_obj.get("silent_mode", False)

    should_emit_status = (planning_active or supports_tool_calls or bool(ctx.meta.pop("_emit_status_messages", False))) and not silent_mode
    execution_mode = "with planning" if planning_active else "direct"

    if should_emit_status:
        if planning_active:
            click.echo(f"🗺️ Planning: {truncated_prompt}")
            click.echo("🗺️ Planning mode enabled")
        else:
            click.echo(f"⚡ Executing: {truncated_prompt}")
            click.echo("⚡ Direct execution mode")

    history_raw = ctx_obj.get("_shell_conversation_history")
    history_enabled = isinstance(history_raw, list)
    history_messages: List[Message] = [msg for msg in history_raw or [] if isinstance(msg, Message)] if history_enabled else []
    max_history_turns = max(1, getattr(settings, "keep_last_assistant_messages", 4))

    # Initialize dynamic context tracker for adaptive RepoMap
    dynamic_context = ctx_obj.get("_dynamic_context")
    if dynamic_context is None:
        repo_root = Path.cwd()
        dynamic_context = DynamicContextTracker(repo_root)
        ctx_obj["_dynamic_context"] = dynamic_context

    # Inject RepoMap messages before user query (Aider's approach)
    repomap_messages_raw = ctx_obj.get("_repomap_messages")
    if repomap_messages_raw:
        for msg in repomap_messages_raw:
            history_messages.append(Message(role=msg["role"], content=msg["content"]))

    user_message = Message(role="user", content=user_prompt)

    repo_root = Path.cwd()
    structure_state = _get_structure_hints_state(ctx)

    repository_language = ctx_obj.get("_detected_language")
    repository_size_estimate = ctx_obj.get("_repo_file_count")
    if repository_language is None or repository_size_estimate is None:
        detected_language, file_count = _detect_repository_language(repo_root)
        if repository_language is None:
            repository_language = detected_language
            ctx_obj["_detected_language"] = detected_language
        if repository_size_estimate is None and file_count is not None:
            repository_size_estimate = file_count
            ctx_obj["_repo_file_count"] = file_count

    project_profile: Dict[str, Any] = {
        "workspace_root": str(settings.workspace_root or repo_root),
        "language": repository_language,
        "repository_size": repository_size_estimate,
        "project_summary": ctx_obj.get("_project_structure_summary"),
        "active_plan_complexity": ctx_obj.get("_active_plan_complexity"),
    }
    discovered_files_snapshot = structure_state.get("files") if isinstance(structure_state, dict) else {}
    if isinstance(discovered_files_snapshot, dict) and discovered_files_snapshot:
        project_profile["recent_files"] = sorted(discovered_files_snapshot.keys())[:6]
    style_notes = ctx_obj.get("_latest_style_profile")
    if style_notes:
        project_profile["style_notes"] = style_notes
    project_profile = {k: v for k, v in project_profile.items() if v}

    router_cls = _resolve_intent_router()
    router = router_cls(
        client,
        settings,
        agent_type=agent_type,
        project_profile=project_profile,
        tool_success_history=ctx_obj.get("_tool_success_history"),
    )
    ctx_obj.setdefault("_router_state", {})["session_id"] = getattr(router, "session_id", None)
    available_tools = getattr(router, "tools", [])

    if not supports_tool_calls:
        decision: IntentDecision = router.route(user_prompt)
        if not decision.tool:
            text = str(decision.arguments.get("text", "")).strip()
            if text and not silent_mode:
                click.echo(text)
            if history_enabled:
                updated = history_messages + [user_message]
                if text:
                    updated.append(Message(role="assistant", content=text))
                ctx.obj["_shell_conversation_history"] = _truncate_shell_history(updated, max_history_turns)
            if should_emit_status:
                elapsed = time.time() - start_time
                click.echo(f"\n✅ Completed in {elapsed:.1f}s ({execution_mode})")
            return

        handler = INTENT_HANDLERS.get(decision.tool)
        if not handler:
            raise click.ClickException(f"Intent tool '{decision.tool}' is not supported yet.")
        handler(ctx, decision.arguments)
        if history_enabled:
            ctx.obj["_shell_conversation_history"] = _truncate_shell_history(
                history_messages + [user_message],
                max_history_turns,
            )
        if should_emit_status:
            elapsed = time.time() - start_time
            click.echo(f"\n✅ Completed in {elapsed:.1f}s ({execution_mode})")
        return

    devagent_cfg = ctx.obj.get("devagent_config")
    if devagent_cfg is None:
        devagent_cfg = load_devagent_yaml()
        ctx.obj["devagent_config"] = devagent_cfg

    agent_spec = AgentRegistry.get(agent_type)

    config_cap = getattr(devagent_cfg, "react_iteration_global_cap", None) if devagent_cfg else None
    settings_cap = getattr(settings, "max_iterations", None)
    iteration_cap = (
        settings_cap
        if isinstance(settings_cap, int) and settings_cap > 0
        else DEFAULT_MAX_ITERATIONS
    )
    if (
        (not isinstance(settings_cap, int) or settings_cap <= 0)
        and isinstance(config_cap, int)
        and config_cap > 0
    ):
        iteration_cap = config_cap

    budget_settings: Dict[str, Any] = {}
    if devagent_cfg and getattr(devagent_cfg, "budget_control", None):
        if isinstance(devagent_cfg.budget_control, dict):
            budget_settings = dict(devagent_cfg.budget_control)

    configured_cap = budget_settings.get("max_iterations")
    if isinstance(configured_cap, int) and configured_cap > 0:
        iteration_cap = min(iteration_cap, configured_cap)

    # Apply agent-specific iteration cap
    if agent_spec and agent_spec.max_iterations > 0:
        iteration_cap = min(iteration_cap, agent_spec.max_iterations)

    phase_thresholds = budget_settings.get("phases")
    warning_settings = budget_settings.get("warnings")
    synthesis_settings = budget_settings.get("synthesis") or {}
    auto_summary_enabled = bool(synthesis_settings.get("auto_summary_on_failure", True))

    model_context_window = getattr(settings, "model_context_window", 100_000)

    use_adaptive = getattr(settings, "enable_reflection", True) or getattr(settings, "adaptive_budget_scaling", True)
    if use_adaptive:
        budget_manager: BudgetManager = AdaptiveBudgetManager(
            iteration_cap,
            phase_thresholds=phase_thresholds if isinstance(phase_thresholds, Mapping) else None,
            warnings=warning_settings if isinstance(warning_settings, Mapping) else None,
            model_context_window=model_context_window,
            adaptive_scaling=getattr(settings, "adaptive_budget_scaling", True),
            enable_reflection=getattr(settings, "enable_reflection", True),
            max_reflections=getattr(settings, "max_reflections", 3),
        )
    else:
        budget_manager = BudgetManager(
            iteration_cap,
            phase_thresholds=phase_thresholds if isinstance(phase_thresholds, Mapping) else None,
            warnings=warning_settings if isinstance(warning_settings, Mapping) else None,
        )

    budget_integration: Optional[BudgetIntegration] = None
    is_test_mode = os.environ.get("PYTEST_CURRENT_TEST") is not None
    if not is_test_mode and (
        settings.enable_cost_tracking or settings.enable_retry or settings.enable_summarization
    ):
        budget_integration = create_budget_integration(settings)
        if settings.enable_summarization and hasattr(client, "complete"):
            budget_integration.initialize_summarizer(client)

    session_manager = SessionManager.get_instance()
    session_id = ctx_obj.get("_session_id")
    if not session_id:
        session_id = f"cli-{uuid4()}"
        ctx_obj["_session_id"] = session_id

    system_messages = [Message(role="system", content="Initializing DEVAGENT assistant...")]
    session = session_manager.ensure_session(
        session_id,
        system_messages=system_messages,
        metadata={
            "iteration_cap": iteration_cap,
            "repository_language": repository_language,
        },
    )

    if history_enabled and history_messages and not session.metadata.get("existing_history_loaded"):
        session_manager.extend_history(session_id, history_messages)
        session.metadata["existing_history_loaded"] = True

    session_manager.extend_history(session_id, [user_message])
    with session.lock:
        session.metadata["history_anchor"] = len(session.history)

    project_structure = ctx.obj.get("_project_structure_summary")
    if not project_structure:
        project_structure = _collect_project_structure_outline(repo_root)
        if project_structure:
            ctx.obj["_project_structure_summary"] = project_structure
            structure_state["project_summary"] = project_structure

    task = TaskSpec(
        identifier=str(uuid4()),
        goal=user_prompt,
        category="assistance",
    )

    # Store user_prompt in ctx_obj for RepoMap refresh
    ctx_obj["_user_prompt"] = user_prompt

    action_provider = LLMActionProvider(
        llm_client=client,
        session_manager=session_manager,
        session_id=session_id,
        tools=available_tools,
        budget_integration=budget_integration,
        format_schema=format_schema,
        ctx_obj=ctx_obj,
    )

    tool_invoker = SessionAwareToolInvoker(
        workspace=repo_root,
        settings=settings,
        session_manager=session_manager,
        session_id=session_id,
        shell_session_manager=ctx.obj.get("_shell_session_manager"),
        shell_session_id=ctx.obj.get("_shell_session_id"),
    )

    executor = BudgetAwareExecutor(budget_manager)

    synthesizer = ContextSynthesizer()
    files_discovered: set[str] = set()
    search_queries: set[str] = set()

    def _update_system_prompt(phase: str, is_final: bool) -> None:
        session_manager.remove_system_messages(session_id, lambda _msg: True)
        session_local = session_manager.get_session(session_id)
        user_query = user_prompt
        with session_local.lock:
            assistant_messages = [msg for msg in session_local.history if msg.role == "assistant"]
            context = synthesizer.synthesize_previous_steps(
                session_local.history,
                current_step=len(assistant_messages),
            )
            redundant_ops = synthesizer.get_redundant_operations(session_local.history)
            constraints = synthesizer.build_constraints_section(redundant_ops)

        if is_final:
            prompt = _build_synthesis_prompt(
                user_query=user_query,
                context=context,
                workspace=str(repo_root),
            )
        else:
            prompt = _build_phase_prompt(
                phase=phase,
                user_query=user_query,
                context=context,
                constraints=constraints,
                workspace=str(repo_root),
                repository_language=repository_language,
            )

        # Append custom system extension if provided
        if system_extension:
            prompt += f"\n\n# Custom Instructions\n{system_extension}"

        # Add agent-specific system prompt suffix
        agent_spec = AgentRegistry.get(agent_type)
        if agent_spec.system_prompt_suffix:
            prompt += f"\n\n{agent_spec.system_prompt_suffix}"

        # Add format schema instructions when present
        if format_schema:
            if is_final:
                # Final iteration - must output JSON now
                prompt += "\n\n# Output Format (OVERRIDES ALL OTHER FORMAT INSTRUCTIONS)\n"
                prompt += "CRITICAL: Your response must be ONLY valid JSON conforming to this schema.\n"
                prompt += "This JSON format requirement SUPERSEDES any other output format instructions above.\n"
                prompt += "Do not include ANY explanatory text, thinking process, markdown formatting, code fences, or line-based output.\n"
                prompt += "Do not write anything before or after the JSON.\n"
                prompt += "Start your response with { and end with }\n"
                prompt += "Output raw JSON only - nothing else.\n\n"
                prompt += "Required JSON Schema:\n"
                prompt += json.dumps(format_schema, indent=2)
            else:
                # Exploration phase - remind about eventual JSON output
                prompt += "\n\n# Final Output Format Requirement\n"
                prompt += "When you have completed your analysis and are ready to provide your final answer,\n"
                prompt += "you MUST format your response as JSON conforming to this schema:\n"
                prompt += json.dumps(format_schema, indent=2)
                prompt += "\n\nDuring exploration, use tools normally. Only output JSON when providing your final answer."

        session_manager.add_system_message(session_id, prompt, location="system")

    def iteration_hook(context, _history):
        _update_system_prompt(context.phase, context.is_final)

    def step_hook(record: StepRecord, context) -> None:
        observation = record.observation
        if not silent_mode:
            display_message = getattr(observation, "display_message", None)
            if display_message:
                click.echo(display_message)
            elif observation.outcome:
                tool_name = observation.tool or record.action.tool
                click.echo(f"{tool_name}: {observation.outcome}")

        metrics_payload = observation.metrics if isinstance(observation.metrics, Mapping) else {}
        _update_files_discovered(files_discovered, metrics_payload if isinstance(metrics_payload, Dict) else {})
        for artifact in observation.artifacts or []:
            files_discovered.add(str(artifact))

        _record_search_query(record.action, search_queries)

        # Dynamic RepoMap: Track context from this step
        if dynamic_context:
            try:
                dynamic_context.update_from_step(record)

                # Check if RepoMap should be refreshed, but DON'T inject yet
                if dynamic_context.should_refresh_repomap():
                    if settings.repomap_debug_stdout:
                        logger.debug(f"RepoMap refresh scheduled after step {record.step_index}")
                        summary = dynamic_context.get_context_summary()
                        logger.debug(f"Context: {summary['total_mentions']} mentions "
                                   f"({len(summary['files'])} files, {len(summary['symbols'])} symbols)")

                    # Store the pending refresh - will be applied before next LLM call
                    ctx_obj["_repomap_refresh_pending"] = True

            except Exception as e:
                # Don't fail the execution if context tracking fails
                if settings.repomap_debug_stdout:
                    logger.debug(f"Context tracking error: {e}")

    try:
        result = executor.run(
            task=task,
            action_provider=action_provider,
            tool_invoker=tool_invoker,
            iteration_hook=iteration_hook,
            step_hook=step_hook,
        )
    except LLMError as exc:
        raise click.ClickException(f"LLM invocation failed: {exc}") from exc

    execution_completed = result.status == "success"

    def _commit_shell_history() -> None:
        if not history_enabled:
            return
        session_local = session_manager.get_session(session_id)
        anchor = session_local.metadata.get("history_anchor", len(session_local.history))
        new_entries: List[Message] = []
        if user_message.content:
            new_entries.append(user_message)
        with session_local.lock:
            for msg in session_local.history[anchor:]:
                if msg.role == "assistant" and msg.content and not msg.tool_calls:
                    new_entries.append(msg)
            session_local.metadata["history_anchor"] = len(session_local.history)
        ctx.obj["_shell_conversation_history"] = _truncate_shell_history(
            history_messages + new_entries,
            max_history_turns,
        )

    _commit_shell_history()
    _merge_structure_hints_state(ctx.obj, structure_state)

    final_message = action_provider.last_response_text()

    # Fallback: if no response text, check for submit_final_answer in last step
    if not final_message and result.steps:
        last_step = result.steps[-1]
        if last_step.observation and last_step.observation.tool == "submit_final_answer":
            final_message = last_step.observation.raw_output or last_step.observation.formatted_output

    final_json: Optional[Dict[str, Any]] = None
    printed_final = False
    if final_message:
        # If format schema provided, extract and validate JSON
        if format_schema:
            extracted_json = _extract_json(final_message)
            if extracted_json:
                final_json = extracted_json
                if not suppress_final_output and not silent_mode:
                    click.echo("")
                    click.echo(json.dumps(extracted_json, indent=2))
                    printed_final = True
            else:
                raise click.ClickException(
                    "Assistant response did not contain valid JSON matching the required schema."
                )
        else:
            if not suppress_final_output and not silent_mode:
                click.echo("")
                click.echo(final_message)
                printed_final = True

    if budget_integration and budget_integration.cost_tracker:
        session = session_manager.get_session(session_id)
        session.metadata["cost_summary"] = {
            "total_cost": budget_integration.cost_tracker.total_cost_usd,
            "total_tokens": (
                budget_integration.cost_tracker.total_prompt_tokens
                + budget_integration.cost_tracker.total_completion_tokens
            ),
            "phase_costs": budget_integration.cost_tracker.phase_costs,
            "model_costs": budget_integration.cost_tracker.model_costs,
        }

    if result.status != "success" and auto_summary_enabled and not silent_mode:
        conversation = session_manager.compose(session_id)
        fallback = auto_generate_summary(
            conversation,
            files_examined=files_discovered,
            searches_performed=search_queries,
        )
        if fallback:
            click.echo("")
            click.echo(fallback)

    if should_emit_status and not silent_mode:
        elapsed = time.time() - start_time
        status_icon = "✅" if execution_completed else "⚠️"
        message = result.stop_reason or ("Completed" if execution_completed else "Execution stopped")
        click.echo(f"\n{status_icon} {message} in {elapsed:.1f}s ({execution_mode})")

    return {
        "final_message": final_message,
        "final_answer": final_message,  # Alias for consistency
        "final_json": final_json,
        "result": result,
        "printed_final": printed_final,
    }
