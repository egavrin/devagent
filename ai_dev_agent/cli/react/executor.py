"""Execution helpers for the CLI ReAct workflow."""
from __future__ import annotations

import contextlib
import io
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import click

from ai_dev_agent.cli.handlers import INTENT_HANDLERS
from ai_dev_agent.cli.utils import (
    _collect_project_structure_outline,
    _detect_repository_language,
    _get_structure_hints_state,
    _merge_structure_hints_state,
    _update_files_discovered,
)
from ai_dev_agent.core.utils.artifacts import write_artifact
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.context_budget import DEFAULT_MAX_TOOL_OUTPUT_CHARS, summarize_text
from ai_dev_agent.core.utils.constants import MIN_TOOL_OUTPUT_CHARS
from ai_dev_agent.core.utils.devagent_config import load_devagent_yaml
from ai_dev_agent.core.utils.keywords import extract_keywords
from ai_dev_agent.core.utils.tool_utils import (
    FILE_READ_TOOLS,
    SEARCH_TOOLS,
    canonical_tool_name,
    tool_signature,
)
from ai_dev_agent.engine.planning.planner import Planner
from ai_dev_agent.engine.react.tool_strategy import (
    TaskType,
    ToolContext as StrategyToolContext,
    ToolSelectionStrategy,
)
from ai_dev_agent.providers.llm import (
    LLMConnectionError,
    LLMError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
)
from ai_dev_agent.providers.llm.base import Message

from ..analysis.formatting import _format_enhanced_tool_log
from ..analysis.research import _handle_question_without_llm
from ..router import IntentDecision, IntentRouter as _DEFAULT_INTENT_ROUTER


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    return normalized not in {"0", "false", "off", "no"}


_DEBUG_CONTEXT_ENABLED = _is_truthy(os.getenv("DEVAGENT_DEBUG_REACT_CONTEXT"))


def _format_message_preview(message: Message, *, limit: int = 200) -> str:
    content = message.content
    if content is None:
        preview = "<no content>"
    else:
        preview = str(content).strip()
        if len(preview) > limit:
            preview = preview[: limit - 1] + "…"

    suffix_parts: List[str] = []
    if message.tool_call_id:
        suffix_parts.append(f"call_id={message.tool_call_id}")
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        suffix_parts.append(f"tool_calls={len(tool_calls)}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"[{message.role}] {preview}{suffix}"


def _emit_context_snapshot(messages: Sequence[Message], iteration: int) -> None:
    if not _DEBUG_CONTEXT_ENABLED:
        return

    iteration_label = "initial context" if iteration == 0 else f"iteration {iteration}"
    click.echo(f"\n🔎 [debug] ReAct context snapshot before {iteration_label}:")
    for index, msg in enumerate(messages, start=1):
        click.echo(f"   {index:02d}. {_format_message_preview(msg)}")



def _resolve_intent_router():
    try:
        cli_module = import_module("ai_dev_agent.cli")
    except ModuleNotFoundError:
        return _DEFAULT_INTENT_ROUTER
    return getattr(cli_module, "IntentRouter", _DEFAULT_INTENT_ROUTER)


def _truncate_shell_history(history: List[Message], max_turns: int) -> List[Message]:
    """Keep only the most recent conversation turns within the configured budget."""

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

def _execute_react_assistant(
    ctx: click.Context,
    client,
    settings: Settings,
    user_prompt: str,
    use_planning: bool = False,
) -> None:
    """Execute multi-step ReAct reasoning for natural language queries."""

    start_time = time.time()
    execution_completed = False
    planning_active = bool(use_planning)
    emit_status_requested = bool(ctx.meta.pop("_emit_status_messages", False))
    supports_tool_calls = hasattr(client, "invoke_tools")
    should_emit_status = planning_active or supports_tool_calls or emit_status_requested
    execution_mode = "with planning" if planning_active else "direct"

    user_message = Message(role="user", content=user_prompt)

    history_messages: List[Message] = []
    history_enabled = False
    if isinstance(getattr(ctx, "obj", None), dict):
        history_raw = ctx.obj.get("_shell_conversation_history")
        if isinstance(history_raw, list):
            history_messages = [msg for msg in history_raw if isinstance(msg, Message)]
            history_enabled = True
    max_history_turns = max(1, getattr(settings, "keep_last_assistant_messages", 4))
    existing_history = list(history_messages)

    truncated_prompt = user_prompt if len(user_prompt) <= 50 else f"{user_prompt[:50]}..."
    direct_mode_announced = not planning_active if should_emit_status else True

    if should_emit_status:
        if planning_active:
            click.echo(f"🗺️ Planning: {truncated_prompt}")
            click.echo("🗺️ Planning mode enabled")
        else:
            click.echo(f"⚡ Executing: {truncated_prompt}")
            click.echo("⚡ Direct execution mode")
            direct_mode_announced = True

    def _finalize() -> None:
        if execution_completed and should_emit_status:
            execution_time = time.time() - start_time
            mode_label = execution_mode
            click.echo(f"\n✅ Completed in {execution_time:.1f}s ({mode_label})")

    router_cls = _resolve_intent_router()
    router = router_cls(client, settings)
    available_tools = getattr(router, "tools", [])

    if not supports_tool_calls:
        if planning_active and not direct_mode_announced:
            execution_mode = "direct"
            planning_active = False
            click.echo("⚡ Direct execution mode")
            direct_mode_announced = True
        decision: IntentDecision = router.route(user_prompt)
        if not decision.tool:
            text = str(decision.arguments.get("text", "")).strip()
            if text:
                click.echo(text)
            execution_completed = True
            if history_enabled:
                updated = existing_history + [user_message]
                if text:
                    updated.append(Message(role="assistant", content=text))
                ctx.obj["_shell_conversation_history"] = _truncate_shell_history(updated, max_history_turns)
            _finalize()
            return

        handler = INTENT_HANDLERS.get(decision.tool)
        if not handler:
            raise click.ClickException(f"Intent tool '{decision.tool}' is not supported yet.")
        handler(ctx, decision.arguments)
        execution_completed = True
        if history_enabled:
            ctx.obj["_shell_conversation_history"] = _truncate_shell_history(
                existing_history + [user_message],
                max_history_turns,
            )
        _finalize()
        return

    strategy = ToolSelectionStrategy()
    structure_state = _get_structure_hints_state(ctx)
    repo_root = Path.cwd()
    repository_language = ctx.obj.get("_detected_language")
    repository_size_estimate = ctx.obj.get("_repo_file_count")
    if repository_language is None or repository_size_estimate is None:
        detected_language, file_count = _detect_repository_language(strategy, repo_root)
        if repository_language is None:
            repository_language = detected_language
            ctx.obj["_detected_language"] = detected_language
        if repository_size_estimate is None:
            repository_size_estimate = file_count
            ctx.obj["_repo_file_count"] = file_count

    task_keywords = set(extract_keywords(user_prompt)) if user_prompt else set()
    query_lower = user_prompt.lower()
    tool_task_type = strategy.detect_task_type(user_prompt)

    enumerative_markers = [
        "list", "show all", "find all", "enumerate", "which files", "what files",
        "what methods", "which methods", "all methods", "every", "identify all",
        "remove which", "candidates", "unused",
    ]
    computational_markers = ["count", "how many", "number of", "total", "sum"]
    informational_markers = ["what is", "where is", "when", "who", "definition"]
    research_markers = ["explain", "describe", "tell me about", "how does", "why"]

    if any(marker in query_lower for marker in computational_markers):
        task_type = "computational"
    elif any(marker in query_lower for marker in enumerative_markers):
        task_type = "enumeration"
    elif tool_task_type in {TaskType.RESEARCH, TaskType.CODE_EXPLORATION} or any(
        marker in query_lower for marker in research_markers
    ):
        task_type = "research"
    elif tool_task_type == TaskType.TESTING:
        task_type = "computational"
    elif any(marker in query_lower for marker in informational_markers):
        task_type = "informational"
    else:
        task_type = "general"

    devagent_cfg = ctx.obj.get("devagent_config")
    if devagent_cfg is None:
        devagent_cfg = load_devagent_yaml()
        ctx.obj["devagent_config"] = devagent_cfg

    config_global_cap = getattr(devagent_cfg, "react_iteration_global_cap", None) if devagent_cfg else None

    env_cap_value: Optional[int] = None
    env_cap_raw = os.getenv("DEVAGENT_MAX_ITERATIONS")
    if env_cap_raw:
        try:
            env_cap_value = int(env_cap_raw)
        except ValueError:
            env_cap_value = None

    default_global_cap = 120
    global_max_iterations = default_global_cap
    if isinstance(config_global_cap, int) and config_global_cap > 0:
        global_max_iterations = config_global_cap
    if isinstance(env_cap_value, int) and env_cap_value > 0:
        global_max_iterations = env_cap_value

    iteration_cap = global_max_iterations

    planner_enabled = planning_active and getattr(settings, "react_enable_planner", True)
    structured_plan = None

    if planning_active and not planner_enabled:
        planning_active = False
        execution_mode = "direct"
        if not direct_mode_announced:
            click.echo("⚡ Direct execution mode")
            direct_mode_announced = True

    if planner_enabled:
        try:
            project_structure = ctx.obj.get("_project_structure_summary")
            if "_project_structure_summary" not in ctx.obj:
                project_structure = _collect_project_structure_outline(Path.cwd())
                ctx.obj["_project_structure_summary"] = project_structure
            if project_structure:
                structure_state["project_summary"] = project_structure
            planner = Planner(client)
            structured_plan = planner.generate(user_prompt, project_structure=project_structure)
        except LLMError as exc:
            click.echo(f"Planner unavailable: {exc}. Switching to direct execution.")
            planning_active = False
            execution_mode = "direct"
            if not direct_mode_announced:
                click.echo("⚡ Direct execution mode")
                direct_mode_announced = True
            structured_plan = None

    if planning_active and structured_plan and getattr(structured_plan, "tasks", None):
        summary_text = structured_plan.summary or structured_plan.goal or "Structured plan"
        total_tasks = len(structured_plan.tasks)
        click.echo(
            f"🗺️ Plan created ({total_tasks} task{'s' if total_tasks != 1 else ''}): {summary_text}"
        )
        for task in structured_plan.tasks:
            click.echo(f"   Step {task.step_number}: {task.title}")
    elif planning_active:
        planning_active = False
        execution_mode = "direct"
        if not direct_mode_announced:
            click.echo("⚡ Direct execution mode")
            direct_mode_announced = True

    consecutive_fails = 0
    used_tools: Set[str] = set()
    file_reads: Set[str] = set()
    search_queries: Set[str] = set()
    tool_repeat_counts: Dict[str, int] = {}

    def _normalize_read_targets(arguments: Dict[str, Any]) -> List[str]:
        targets: List[str] = []
        path_value = arguments.get("path")
        if path_value:
            targets.append(str(path_value))
        paths_value = arguments.get("paths")
        if isinstance(paths_value, list):
            targets.extend(str(item) for item in paths_value if item)
        elif paths_value:
            targets.append(str(paths_value))
        return targets

    tool_guidance = ""
    if task_type == "computational":
        tool_guidance = (
            "\nFOR COMPUTATIONAL TASKS (counting, measuring, calculating):\n"
            "- Prefer shell commands (sandbox_exec) for bulk file operations\n"
            "- Generate and run scripts for complex computations\n"
            "- Use 'find', 'wc', 'ls -la', etc. for file system analysis\n"
            "- Avoid reading individual files when aggregate data is needed"
        )
    elif task_type == "enumeration":
        tool_guidance = (
            "\nFOR ENUMERATION TASKS (listing, finding all):\n"
            "- Use code_search for finding patterns across files\n"
            "- Use sandbox_exec for directory traversal and filtering\n"
            "- Combine multiple search strategies if needed\n"
            "- Aggregate results before presenting"
        )

    messages = [
        Message(
            role="system",
            content=(
                "You are a helpful assistant for a software development CLI tool called devagent. "
                "Use the available tools efficiently to answer the user's question. "
                f"You have a budget of {iteration_cap} iterations to complete this task. "
                f"Plan your tool usage accordingly and prioritize synthesis as you approach the limit. "
                "IMPORTANT GUIDELINES:\n"
                "1. Choose the most appropriate tool for the task type\n"
                "2. Avoid calling the same tool with identical arguments repeatedly\n"
                "3. Don't read individual files when bulk operations are more efficient\n"
                "4. When you have sufficient information, provide a final response WITHOUT calling more tools\n"
                "5. If a tool fails, try a different approach rather than repeating\n"
                "6. For file system operations, prefer shell commands over individual file reads\n"
                "7. Generate scripts when you need to perform complex computations"
                + tool_guidance
            ),
        ),
        user_message,
    ]

    project_structure = ctx.obj.get("_project_structure_summary")
    if not project_structure:
        project_structure = _collect_project_structure_outline(repo_root)
        if project_structure:
            ctx.obj["_project_structure_summary"] = project_structure
    if project_structure:
        structure_state["project_summary"] = project_structure
        messages.insert(
            1,
            Message(
                role="system",
                content="Project structure outline:\n" + project_structure,
            ),
        )

    if history_enabled and existing_history:
        insert_at = len(messages) - 1
        messages[insert_at:insert_at] = existing_history

    history_start_index = len(messages) - 1

    def _commit_shell_history() -> None:
        if not history_enabled or not isinstance(ctx.obj, dict):
            return
        new_entries: List[Message] = []
        if user_message.content:
            new_entries.append(user_message)
        for msg in messages[history_start_index + 1 :]:
            if msg.role == "assistant" and msg.content and not msg.tool_calls:
                new_entries.append(msg)
        ctx.obj["_shell_conversation_history"] = _truncate_shell_history(
            existing_history + new_entries,
            max_history_turns,
        )

    _emit_context_snapshot(messages, 0)

    iteration = 0
    tool_history: List[str] = []
    has_symbol_index = False
    files_discovered_set: Set[str] = set()
    last_tool_success = True

    while iteration < iteration_cap:
        iteration += 1

        strategy_context = StrategyToolContext(
            task_type=tool_task_type,
            language=repository_language,
            has_symbol_index=has_symbol_index,
            files_discovered=set(files_discovered_set),
            tools_used=list(tool_history),
            last_tool_success=last_tool_success,
            iteration_count=iteration - 1,
            task_keywords=set(task_keywords),
            repository_size=repository_size_estimate,
            iteration_budget=iteration_cap,
        )

        tools_for_iteration = available_tools
        try:
            sanitized_to_entry: Dict[str, Dict[str, Any]] = {}
            sanitized_to_canonical: Dict[str, str] = {}

            _emit_context_snapshot(messages, iteration)

            for entry in available_tools:
                fn = entry.get("function", {})
                sanitized = fn.get("name")
                if not sanitized:
                    continue
                canonical = canonical_tool_name(sanitized)
                sanitized_to_entry[sanitized] = entry
                sanitized_to_canonical[sanitized] = canonical

            canonical_available = [canonical for canonical in sanitized_to_canonical.values()]
            prioritized_canonical = strategy.prioritize_tools(canonical_available, strategy_context)

            prioritized_sanitized: List[str] = []
            used_sanitized: Set[str] = set()
            for canonical in prioritized_canonical:
                for sanitized, mapped in sanitized_to_canonical.items():
                    if mapped == canonical and sanitized not in used_sanitized:
                        prioritized_sanitized.append(sanitized)
                        used_sanitized.add(sanitized)
                        break

            for sanitized in sanitized_to_entry:
                if sanitized not in used_sanitized:
                    prioritized_sanitized.append(sanitized)

            tools_for_iteration = [
                sanitized_to_entry[name]
                for name in prioritized_sanitized
                if name in sanitized_to_entry
            ] or available_tools
        except Exception:
            tools_for_iteration = available_tools

        try:
            result = client.invoke_tools(messages, tools=tools_for_iteration, temperature=0.1)

            assistant_message = Message(
                role="assistant",
                content=result.message_content,
                tool_calls=result.raw_tool_calls,
            )
            if assistant_message.content is not None or assistant_message.tool_calls:
                messages.append(assistant_message)

            if not result.calls:
                if result.message_content:
                    click.echo(result.message_content)
                else:
                    click.echo("I was unable to provide a complete answer.")
                execution_completed = True
                _commit_shell_history()
                _finalize()
                return

            redundant_calls = 0
            last_tool_failed = consecutive_fails > 0

            for tool_call in result.calls:
                call_signature = tool_signature(tool_call)
                if tool_call.name in FILE_READ_TOOLS:
                    targets = _normalize_read_targets(tool_call.arguments)
                    if targets and all(target in file_reads for target in targets) and not last_tool_failed:
                        redundant_calls += 1
                elif tool_call.name in SEARCH_TOOLS:
                    query = tool_call.arguments.get("query")
                    if query in search_queries and not last_tool_failed:
                        redundant_calls += 1
                elif call_signature in used_tools and not last_tool_failed:
                    redundant_calls += 1

            if redundant_calls >= len(result.calls) * 0.7 and len(result.calls) > 1:
                click.echo("🚫 Detected redundant tool usage. Stopping to avoid loops.")
                execution_completed = True
                _commit_shell_history()
                _finalize()
                return

            total_calls = len(result.calls)
            successful_calls = 0
            file_reading_context: Dict[str, Any] = {"last_file_read": None, "file_line_counts": {}}

            for call_index, tool_call in enumerate(result.calls, 1):
                call_signature = tool_signature(tool_call)
                canonical_name = canonical_tool_name(tool_call.name)

                used_tools.add(call_signature)

                if tool_call.name in FILE_READ_TOOLS:
                    for target in _normalize_read_targets(tool_call.arguments):
                        file_reads.add(target)
                        files_discovered_set.add(target)
                elif tool_call.name in SEARCH_TOOLS and tool_call.arguments.get("query"):
                    search_queries.add(tool_call.arguments["query"])

                handler = INTENT_HANDLERS.get(tool_call.name)
                if not handler:
                    error_msg = f"Tool '{tool_call.name}' is not supported."
                    tool_call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", "unknown")
                    messages.append(
                        Message(
                            role="tool",
                            content=f"Error: {error_msg}",
                            tool_call_id=tool_call_id,
                        )
                    )
                    click.echo(f"❌ {tool_call.name} → Tool not supported")
                    consecutive_fails += 1
                    continue

                captured_output = io.StringIO()
                repeat_count = tool_repeat_counts.get(call_signature, 0) + 1
                tool_repeat_counts[call_signature] = repeat_count

                tool_output_for_message: Optional[str] = None
                handler_result: Optional[Mapping[str, Any]] = None
                call_success = False

                try:
                    start_call = time.time()
                    with contextlib.redirect_stdout(captured_output):
                        handler_result = handler(ctx, tool_call.arguments)
                    tool_output_raw = captured_output.getvalue()
                    tool_output = tool_output_raw.strip() or "Tool executed successfully (no output)"

                    max_chars = max(
                        MIN_TOOL_OUTPUT_CHARS,
                        getattr(settings, "max_tool_output_chars", DEFAULT_MAX_TOOL_OUTPUT_CHARS),
                    )
                    summarized_output = summarize_text(tool_output, max_chars)
                    if summarized_output != tool_output:
                        summary_note = f" (truncated to {max_chars} chars)"
                        artifact_reference = ""
                        try:
                            artifact_path = write_artifact(tool_output)
                            try:
                                display_path = artifact_path.relative_to(Path.cwd())
                            except ValueError:
                                display_path = artifact_path
                            artifact_reference = f"\nFull output saved to {display_path}"
                        except Exception:
                            artifact_reference = ""
                        tool_output_for_message = f"{summarized_output}{summary_note}{artifact_reference}"
                    else:
                        tool_output_for_message = tool_output

                    execution_time = time.time() - start_call
                    log_message = _format_enhanced_tool_log(
                        tool_call,
                        repeat_count,
                        execution_time,
                        tool_output,
                        success=True,
                        file_context=file_reading_context,
                    )
                    click.echo(log_message)

                    successful_calls += 1
                    consecutive_fails = 0
                    call_success = True

                except FileNotFoundError as exc:
                    execution_time = time.time() - start_time
                    tool_output = f"File not found: {exc}"
                    log_message = _format_enhanced_tool_log(
                        tool_call,
                        repeat_count,
                        execution_time,
                        tool_output,
                        success=False,
                        file_context=file_reading_context,
                    )
                    click.echo(log_message)
                    consecutive_fails += 1
                    tool_output_for_message = tool_output
                    call_success = False

                except Exception as exc:
                    execution_time = time.time() - start_time
                    tool_output = f"Error executing {tool_call.name}: {exc}"
                    log_message = _format_enhanced_tool_log(
                        tool_call,
                        repeat_count,
                        execution_time,
                        tool_output,
                        success=False,
                        file_context=file_reading_context,
                    )
                    click.echo(log_message)
                    consecutive_fails += 1
                    tool_output_for_message = tool_output
                    call_success = False

                tool_call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", None)
                tool_message = Message(
                    role="tool",
                    content=tool_output_for_message or tool_output,
                    tool_call_id=tool_call_id,
                )
                messages.append(tool_message)

                if handler_result:
                    _merge_structure_hints_state(structure_state, handler_result)
                    _update_files_discovered(files_discovered_set, handler_result)
                    if structure_state.get("project_summary"):
                        ctx.obj["_project_structure_summary"] = structure_state["project_summary"]

                if canonical_name == "symbols.index" and call_success:
                    has_symbol_index = True

                tool_history.append(canonical_name)
                last_tool_success = call_success

            if consecutive_fails >= 3:
                click.echo("🚫 Multiple consecutive tool failures. Stopping to avoid loops.")
                execution_completed = True
                _commit_shell_history()
                _finalize()
                return

            if (task_type == "informational" and iteration >= 6 and successful_calls >= 2) or (
                task_type in ["computational", "enumeration"] and iteration >= 8 and successful_calls >= 3
            ):
                has_concrete_results = any(
                    "completed" in str(msg.content).lower()
                    or "found" in str(msg.content).lower()
                    or "result" in str(msg.content).lower()
                    for msg in messages[-3:]
                    if hasattr(msg, "content") and msg.content
                )

                if has_concrete_results:
                    hint_message = Message(
                        role="user",
                        content="Based on the information gathered above, please provide a complete answer to my original question.",
                    )
                    messages.append(hint_message)

        except (LLMConnectionError, LLMTimeoutError, LLMRetryExhaustedError) as exc:
            click.echo(f"⚠️  Unable to reach the LLM: {exc}")
            click.echo("   Falling back to offline analysis using local heuristics.")
            _handle_question_without_llm(user_prompt, reason="LLM unavailable")
            execution_mode = "direct"
            execution_completed = True
            _commit_shell_history()
            _finalize()
            return
        except LLMError as exc:
            click.echo(f"❌ ReAct execution failed: {exc}")
            break
        except Exception as exc:
            click.echo(f"❌ ReAct execution failed: {exc}")
            break

    _commit_shell_history()

    if iteration >= iteration_cap:
        click.echo(f"⚠️  Reached maximum iteration limit ({iteration_cap}).")
        click.echo("Please refine your request or increase DEVAGENT_MAX_ITERATIONS if you need more steps.")
        execution_completed = True
        _finalize()


__all__ = ["_execute_react_assistant"]
