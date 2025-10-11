"""Tool invoker that routes ReAct tool calls to registry-backed implementations."""
from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ai_dev_agent.tools.code.code_edit.editor import CodeEditor
from ai_dev_agent.engine.react.pipeline import PipelineCommands
from ai_dev_agent.tools.execution.testing.local_tests import TestRunner
from ai_dev_agent.tools.execution.shell_session import ShellSessionManager
from ai_dev_agent.tools import ToolContext, registry, READ, WRITE, RUN
from ai_dev_agent.core.utils.devagent_config import DevAgentConfig, load_devagent_yaml
from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.metrics import MetricsCollector
from ai_dev_agent.core.utils.artifacts import write_artifact
from ai_dev_agent.core.utils.constants import MIN_TOOL_OUTPUT_CHARS
from ai_dev_agent.core.utils.context_budget import DEFAULT_MAX_TOOL_OUTPUT_CHARS, summarize_text
from ai_dev_agent.core.utils.tool_utils import canonical_tool_name
from ai_dev_agent.session import SessionManager
from .types import ActionRequest, CLIObservation, Observation, ToolCall, ToolResult

LOGGER = get_logger(__name__)


class RegistryToolInvoker:
    """Invoke registered tools backed by the shared registry."""

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        code_editor: Optional[CodeEditor] = None,
        test_runner: Optional[TestRunner] = None,
        sandbox=None,
        collector: Optional[MetricsCollector] = None,
        pipeline_commands: Optional[PipelineCommands] = None,
        devagent_cfg: Optional[DevAgentConfig] = None,
        shell_session_manager: Optional[ShellSessionManager] = None,
        shell_session_id: Optional[str] = None,
    ) -> None:
        self.workspace = workspace
        self.settings = settings
        self.code_editor = code_editor
        self.test_runner = test_runner
        self.sandbox = sandbox
        self.collector = collector
        self.pipeline_commands = pipeline_commands
        self.devagent_cfg = devagent_cfg or load_devagent_yaml()
        self.shell_session_manager = shell_session_manager
        self.shell_session_id = shell_session_id
        self._structure_hints: Dict[str, Any] = {
            "symbols": set(),
            "files": {},
            "project_summary": None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, action: ActionRequest) -> Observation:
        # Check if this is a batch request
        if action.tool_calls:
            return self.invoke_batch(action.tool_calls)

        # Single-tool mode (backward compatible)
        payload = action.args or {}
        tool_name = action.tool

        # Intercept submit_final_answer before registry lookup
        if tool_name == "submit_final_answer":
            answer = payload.get("answer", "")
            return Observation(
                success=True,
                outcome="success",
                tool="submit_final_answer",
                raw_output=answer,
                display_message="✓ Final answer ready",
            )

        try:
            result = self._invoke_registry(tool_name, payload)
        except KeyError:
            return Observation(
                success=False,
                outcome=f"Unknown tool: {tool_name}",
                tool=tool_name,
                error=f"Tool '{tool_name}' is not registered",
            )
        except ValueError as exc:
            return Observation(
                success=False,
                outcome=f"Tool {tool_name} rejected input",
                tool=tool_name,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Tool %s execution failed", tool_name)
            return Observation(
                success=False,
                outcome=f"Tool {tool_name} failed",
                tool=tool_name,
                error=str(exc),
            )

        return self._wrap_result(tool_name, result)

    def invoke_batch(self, tool_calls: List[ToolCall]) -> Observation:
        """Execute multiple tools in parallel and return aggregated observation."""
        if not tool_calls:
            return Observation(
                success=False,
                outcome="No tool calls provided",
                error="Empty batch request",
            )

        # Run batch execution synchronously using ThreadPoolExecutor
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, can't use asyncio.run
                # Fall back to sequential execution
                results = []
                for call in tool_calls:
                    results.append(self._execute_single_tool(call))
            else:
                # Use asyncio.run for concurrent execution
                results = asyncio.run(self._execute_batch_async(tool_calls))
        except RuntimeError:
            # No event loop available, execute sequentially
            results = []
            for call in tool_calls:
                results.append(self._execute_single_tool(call))

        # Aggregate results
        all_success = all(r.success for r in results)
        total_calls = len(results)
        success_count = sum(1 for r in results if r.success)

        outcome_parts = [f"Executed {total_calls} tool(s): {success_count} succeeded"]
        if success_count < total_calls:
            outcome_parts.append(f"{total_calls - success_count} failed")

        # Aggregate metrics
        aggregated_metrics: Dict[str, Any] = {
            "total_calls": total_calls,
            "successful_calls": success_count,
            "failed_calls": total_calls - success_count,
        }

        # Sum up costs and wall times
        total_wall_time = sum(r.wall_time for r in results if r.wall_time)
        if total_wall_time > 0:
            aggregated_metrics["total_wall_time"] = total_wall_time
            aggregated_metrics["max_wall_time"] = max((r.wall_time for r in results if r.wall_time), default=0)

        # Collect all artifacts
        all_artifacts = []
        for r in results:
            if r.metrics.get("artifacts"):
                all_artifacts.extend(r.metrics["artifacts"])

        return Observation(
            success=all_success,
            outcome=", ".join(outcome_parts),
            metrics=aggregated_metrics,
            artifacts=all_artifacts,
            tool=f"batch[{total_calls}]",
            results=results,
        )

    async def _execute_batch_async(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute tool calls concurrently using asyncio."""
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 10)) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, self._execute_single_tool, call)
                for call in tool_calls
            ]
            return await asyncio.gather(*tasks)

    def _execute_single_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool and return its result."""
        start_time = time.time()
        tool_name = call.tool
        payload = call.args or {}

        # Intercept submit_final_answer in batch mode too
        if tool_name == "submit_final_answer":
            answer = payload.get("answer", "")
            return ToolResult(
                call_id=call.call_id,
                tool="submit_final_answer",
                success=True,
                outcome="success",
                error=None,
                metrics={"raw_output": answer},
                wall_time=time.time() - start_time,
            )

        try:
            result = self._invoke_registry(tool_name, payload)
            observation = self._wrap_result(tool_name, result)

            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=observation.success,
                outcome=observation.outcome,
                error=observation.error,
                metrics={
                    **observation.metrics,
                    "artifacts": observation.artifacts,
                },
                wall_time=time.time() - start_time,
            )
        except KeyError:
            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=False,
                outcome=f"Unknown tool: {tool_name}",
                error=f"Tool '{tool_name}' is not registered",
                wall_time=time.time() - start_time,
            )
        except ValueError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=False,
                outcome=f"Tool {tool_name} rejected input",
                error=str(exc),
                wall_time=time.time() - start_time,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Tool %s execution failed", tool_name)
            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=False,
                outcome=f"Tool {tool_name} failed",
                error=str(exc),
                wall_time=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Registry invocation helpers
    # ------------------------------------------------------------------

    def _invoke_registry(self, tool_name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        extra: Dict[str, Any] = {
            "code_editor": self.code_editor,
            "test_runner": self.test_runner,
            "pipeline_commands": self.pipeline_commands,
            "structure_hints": self._export_structure_hints(),
        }

        if isinstance(self.shell_session_manager, ShellSessionManager) and isinstance(self.shell_session_id, str):
            extra["shell_session_manager"] = self.shell_session_manager
            extra["shell_session_id"] = self.shell_session_id

        ctx = ToolContext(
            repo_root=self.workspace,
            settings=self.settings,
            sandbox=self.sandbox,
            devagent_config=self.devagent_cfg,
            metrics_collector=self.collector,
            extra=extra,
        )
        return registry.invoke(tool_name, payload, ctx)

    def _wrap_result(self, tool_name: str, result: Mapping[str, Any]) -> Observation:
        success = True
        outcome = f"Executed {tool_name}"
        metrics: Dict[str, Any] = {}
        artifacts: list[str] = []
        raw_output: Optional[str] = None

        if tool_name == READ:
            files = result.get("files", [])
            outcome = f"Read {len(files)} file(s)"
            total_lines = 0
            for entry in files:
                if isinstance(entry, Mapping):
                    path_value = entry.get("path")
                    if path_value:
                        artifacts.append(str(path_value))
                    content = entry.get("content")
                    if isinstance(content, str):
                        total_lines += len(content.splitlines())
                elif isinstance(entry, str):
                    artifacts.append(entry)
            metrics = {"files": len(files), "lines_read": total_lines}
            raw_output = json.dumps(result, indent=2)
        elif tool_name == WRITE:
            applied = bool(result.get("applied"))
            rejected = result.get("rejected_hunks") or []
            success = applied and not rejected

            if applied:
                outcome = "Patch applied"
                rejection_reason = None
            else:
                first_reason = next((msg for msg in rejected if msg), "")
                if first_reason:
                    # Keep the reason compact so the observation stays readable for the agent.
                    first_line = first_reason.splitlines()[0][:200]
                    rejection_reason = first_line
                    outcome = f"Patch rejected: {first_line}"
                else:
                    rejection_reason = None
                    outcome = "Patch rejected: no changes detected"

            metrics = {
                "applied": applied,
                "rejected_hunks": len(rejected),
                **(result.get("diff_stats") or {}),
            }
            if rejection_reason:
                metrics["rejection_reason"] = rejection_reason
            artifacts = result.get("changed_files") or []
            raw_output = json.dumps(result, indent=2)
        elif tool_name == "find":
            files = result.get("files", [])
            success = bool(files)
            outcome = f"Found {len(files)} file(s)"
            artifacts: List[str] = []
            for entry in files:
                if isinstance(entry, Mapping):
                    candidate = entry.get("path") or entry.get("file")
                    if candidate:
                        artifacts.append(str(candidate))
                elif entry:
                    artifacts.append(str(entry))
            metrics = {"files": len(files), "paths": artifacts[:10]}
            raw_output = json.dumps(files[:20], indent=2)
        elif tool_name == "grep":
            matches = result.get("matches", [])
            success = bool(matches)
            outcome = f"Found matches in {len(matches)} file(s)"

            # Extract file paths and match counts for better context
            artifacts = []
            match_counts = {}
            for group in matches:
                if isinstance(group, Mapping) and group.get("file"):
                    file_path = group.get("file")
                    artifacts.append(file_path)
                    # Count matches in this file
                    file_matches = group.get("matches", [])
                    match_counts[file_path] = len(file_matches) if isinstance(file_matches, list) else 0

            metrics = {
                "files": len(matches),
                "match_counts": match_counts  # Add match counts to metrics
            }
            raw_output = json.dumps(matches[:10], indent=2)
        elif tool_name == "symbols":
            symbols = result.get("symbols", [])
            outcome = f"Found {len(symbols)} symbol(s)"
            metrics = {"symbols": len(symbols)}
            artifacts = [
                entry.get("file")
                for entry in symbols
                if isinstance(entry, Mapping) and entry.get("file")
            ]
            raw_output = json.dumps(symbols[:20], indent=2)
        elif tool_name == RUN:
            exit_code = result.get("exit_code", 0)
            success = exit_code == 0
            outcome = f"Command exited with {exit_code}"
            metrics = {
                "exit_code": exit_code,
                "duration_ms": result.get("duration_ms"),
                "stdout_tail": result.get("stdout_tail"),
                "stderr_tail": result.get("stderr_tail"),
            }
            raw_output = "STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}".format(
                stdout=result.get("stdout_tail", ""),
                stderr=result.get("stderr_tail", ""),
            )
        else:
            metrics = dict(result)
            raw_output = json.dumps(result, indent=2)

        observations_kwargs: Dict[str, Any] = {
            "success": success,
            "outcome": outcome,
            "metrics": metrics,
            "artifacts": [item for item in artifacts if item],
            "tool": tool_name,
            "raw_output": raw_output,
        }

        structure_payload = self._export_structure_hints()
        if structure_payload["symbols"] or structure_payload["files"] or structure_payload["project_summary"]:
            observations_kwargs["structure_hints"] = structure_payload

        return Observation(**observations_kwargs)

    def _export_structure_hints(self) -> Dict[str, Any]:
        files_payload: Dict[str, Any] = {}
        file_hints = self._structure_hints.get("files") or {}
        for path, info in file_hints.items():
            outline = info.get("outline") or []
            symbols = info.get("symbols") or []
            files_payload[path] = {
                "outline": outline[:20],
                "symbols": sorted(set(symbols))[:20],
            }

        symbols = sorted(set(self._structure_hints.get("symbols") or []))[:50]
        return {
            "symbols": symbols,
            "files": files_payload,
            "project_summary": self._structure_hints.get("project_summary"),
        }


class SessionAwareToolInvoker(RegistryToolInvoker):
    """Tool invoker that integrates tool output with session history."""

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        code_editor: Optional[CodeEditor] = None,
        test_runner: Optional[TestRunner] = None,
        sandbox=None,
        collector: Optional[MetricsCollector] = None,
        pipeline_commands: Optional[PipelineCommands] = None,
        devagent_cfg: Optional[DevAgentConfig] = None,
        *,
        session_manager: Optional[SessionManager] = None,
        session_id: Optional[str] = None,
        shell_session_manager: Optional[ShellSessionManager] = None,
        shell_session_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            workspace=workspace,
            settings=settings,
            code_editor=code_editor,
            test_runner=test_runner,
            sandbox=sandbox,
            collector=collector,
            pipeline_commands=pipeline_commands,
            devagent_cfg=devagent_cfg,
            shell_session_manager=shell_session_manager,
            shell_session_id=shell_session_id,
        )
        self.session_manager = session_manager or (SessionManager.get_instance() if session_id else None)
        self.session_id = session_id
        setting_value = getattr(settings, "max_tool_output_chars", DEFAULT_MAX_TOOL_OUTPUT_CHARS)
        try:
            parsed_setting = int(setting_value)
        except (TypeError, ValueError):
            parsed_setting = DEFAULT_MAX_TOOL_OUTPUT_CHARS
        self._max_tool_output_chars = max(MIN_TOOL_OUTPUT_CHARS, parsed_setting)

    def __call__(self, action: ActionRequest) -> CLIObservation:
        base_observation = super().__call__(action)
        cli_observation = self._to_cli_observation(action, base_observation)
        self._record_tool_message(action, cli_observation)
        return cli_observation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_cli_observation(self, action: ActionRequest, observation: Observation) -> CLIObservation:
        payload = observation.model_dump()

        raw_text = (observation.raw_output or "").strip()
        outcome_text = (observation.outcome or "").strip()
        canonical = canonical_tool_name(action.tool)

        # Special handling for submit_final_answer - show the actual answer
        if canonical == "submit_final_answer":
            summary_source = raw_text  # The answer is in raw_output
            summary_text = raw_text
            formatted_output = raw_text
            display_message = self._format_display_message(action, observation, canonical)
            payload.update(
                formatted_output=formatted_output,
                display_message=display_message,
            )
            return CLIObservation.model_validate(payload)

        if canonical == RUN and raw_text:
            summary_source = raw_text
        else:
            summary_source = raw_text or outcome_text

        summary_text = summary_source
        artifact_path: Optional[Path] = None

        if summary_source:
            summarized = summarize_text(summary_source, self._max_tool_output_chars)
            if summarized != summary_source:
                summary_text = summarized
                try:
                    artifact_path = write_artifact(summary_source)
                except Exception:  # noqa: BLE001 - best-effort artifact creation
                    artifact_path = None
            else:
                summary_text = summary_source

        formatted_output = summary_text or outcome_text or None

        # For find/grep, provide file lists to LLM (not just display message)
        if canonical == "find" and observation.artifacts:
            # Dynamic limit based on total results
            total_files = len(observation.artifacts)
            if total_files <= 10:
                limit = total_files  # Show all for small sets
            elif total_files <= 50:
                limit = 30  # Standard limit for medium sets
            else:
                limit = 20  # Smaller limit for large sets

            file_list = observation.artifacts[:limit]
            formatted_output = "Files found:\n" + "\n".join(f"- {f}" for f in file_list)
            if total_files > limit:
                remaining = total_files - limit
                formatted_output += f"\n... and {remaining} more files"
                if total_files > 50:
                    formatted_output += "\n(Tip: Use more specific pattern to narrow results)"
            import os
            if os.environ.get("DEVAGENT_DEBUG_TOOLS"):
                print(f"[DEBUG-FIND] Formatted output for LLM: {len(file_list)}/{total_files} files", flush=True)
        elif canonical == "grep" and observation.artifacts:
            # For grep, show file list with match counts for better context
            total_files = len(observation.artifacts)
            if total_files <= 10:
                limit = total_files
            elif total_files <= 50:
                limit = 30
            else:
                limit = 20

            file_list = observation.artifacts[:limit]
            match_counts = observation.metrics.get("match_counts", {})

            formatted_lines = []
            for f in file_list:
                count = match_counts.get(f, 0)
                if count > 0:
                    formatted_lines.append(f"- {f} ({count} match{'es' if count != 1 else ''})")
                else:
                    formatted_lines.append(f"- {f}")

            formatted_output = "Files with matches:\n" + "\n".join(formatted_lines)
            if total_files > limit:
                remaining = total_files - limit
                formatted_output += f"\n... and {remaining} more files"
                if total_files > 50:
                    formatted_output += "\n(Tip: Use more specific pattern to narrow results)"
            import os
            if os.environ.get("DEVAGENT_DEBUG_TOOLS"):
                print(f"[DEBUG-GREP] Formatted output for LLM: {len(file_list)}/{total_files} files with counts", flush=True)

        display_message = self._format_display_message(action, observation, canonical)

        artifact_display: Optional[str] = None
        if artifact_path:
            artifact_display = self._normalize_artifact_path(artifact_path)
            if formatted_output:
                formatted_output = f"{formatted_output}\nFull output saved to {artifact_display}"
            else:
                formatted_output = f"Full output saved to {artifact_display}"

        payload.update(
            formatted_output=formatted_output,
            artifact_path=artifact_display,
            display_message=display_message,
        )
        return CLIObservation.model_validate(payload)

    def _format_display_message(
        self,
        action: ActionRequest,
        observation: Observation,
        canonical_name: str,
    ) -> str:
        success = observation.success
        status_ok = "✓"
        status_fail = "✗"
        base_icon_map = {
            "find": "🔍",
            "grep": "🔎",
            "symbols": "🧭",
            READ: "📖",
            RUN: "⚡",
            "write": "📝",
            "submit_final_answer": "✅",
        }
        icon = base_icon_map.get(canonical_name, status_ok if success else status_fail)
        status_suffix = status_ok if success else status_fail

        def quote(value: Optional[str]) -> str:
            if not value:
                return ""
            return f" \"{value}\""

        if canonical_name == "find":
            query = (
                action.args.get("query")
                or action.args.get("pattern")
                or action.args.get("path")
                or action.args.get("name")
            )
            matches = observation.metrics.get("files")
            if matches is None and observation.artifacts:
                matches = len(observation.artifacts)
            if isinstance(matches, (int, float)):
                matches_text = f"{int(matches)} match{'es' if int(matches) != 1 else ''} found"
            else:
                matches_text = observation.outcome or ("matches found" if success else "no matches")
            path_hint = None
            if observation.artifacts:
                path_hint = observation.artifacts[0]
            elif isinstance(observation.metrics.get("raw"), Mapping):
                raw_files = observation.metrics["raw"].get("files")
                if isinstance(raw_files, list) and raw_files:
                    first = raw_files[0]
                    if isinstance(first, Mapping):
                        path_hint = first.get("path")
                    elif isinstance(first, str):
                        path_hint = first
            suffix = f" ({path_hint})" if path_hint else ""
            return f"{icon} find{quote(str(query) if query else None)} → {matches_text}{suffix}"

        if canonical_name == "grep":
            query = action.args.get("query") or action.args.get("pattern")
            matches = observation.metrics.get("files")
            if isinstance(matches, (int, float)):
                matches_text = f"{int(matches)} file{'s' if int(matches) != 1 else ''}"
            else:
                matches_text = observation.outcome or ("matches located" if success else "no matches")
            return f"{icon} grep{quote(str(query) if query else None)} → {matches_text}"

        if canonical_name == READ:
            path = action.args.get("path")
            if not path:
                paths = action.args.get("paths")
                if isinstance(paths, list) and paths:
                    path = paths[0]
                elif isinstance(paths, str):
                    path = paths
            lines_read = observation.metrics.get("lines_read")
            if isinstance(lines_read, (int, float)) and lines_read > 0:
                detail = f"{int(lines_read)} line{'s' if int(lines_read) != 1 else ''} read"
            else:
                detail = observation.outcome or ("content captured" if success else "read failed")
            return f"{icon} read{quote(str(path) if path else None)} → {detail}"

        if canonical_name == RUN:
            cmd = action.args.get("cmd") or action.args.get("command")
            if not cmd:
                args = action.args.get("args")
                if isinstance(args, (list, tuple)) and args:
                    cmd = " ".join(str(item) for item in args)
                else:
                    cmd = str(args) if args else None
            exit_code = observation.metrics.get("exit_code")
            if success:
                status = status_ok
            else:
                status = f"{status_fail} exit {exit_code}" if exit_code is not None else status_fail
            preview_value: Optional[str] = None
            preview_label = "stdout"

            stdout_tail = observation.metrics.get("stdout_tail")
            if isinstance(stdout_tail, str):
                stripped = stdout_tail.strip()
                if stripped:
                    preview_value = stripped.splitlines()[0].strip()

            if preview_value is None:
                stderr_tail = observation.metrics.get("stderr_tail")
                if isinstance(stderr_tail, str):
                    stripped_err = stderr_tail.strip()
                    if stripped_err:
                        preview_value = stripped_err.splitlines()[0].strip()
                        preview_label = "stderr"

            if preview_value:
                if len(preview_value) > 120:
                    preview_value = f"{preview_value[:117]}..."
                return f"{icon} run{quote(cmd)} → {status} ({preview_label}: {preview_value})"
            return f"{icon} run{quote(cmd)} → {status}"

        if canonical_name == "write":
            targets = observation.metrics.get("artifacts") or observation.artifacts
            if isinstance(targets, list) and targets:
                detail = ", ".join(str(item) for item in targets[:3])
                if len(targets) > 3:
                    detail += f" (+{len(targets) - 3})"
            else:
                detail = observation.outcome or ("changes applied" if success else "no changes")
            return f"{icon} write → {detail}"

        return f"{icon if success else status_fail} {action.tool}{' → ' + (observation.outcome or status_suffix) if observation.outcome else ''}"

    def _normalize_artifact_path(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.workspace)
        except ValueError:
            try:
                relative = path.relative_to(Path.cwd())
            except ValueError:
                relative = path
        return str(relative)

    def _record_tool_message(self, action: ActionRequest, observation: CLIObservation) -> None:
        if not self.session_manager or not self.session_id:
            return

        canonical = canonical_tool_name(action.tool)
        tool_call_id = (
            action.metadata.get("tool_call_id")
            or action.metadata.get("call_id")
            or action.metadata.get("id")
            or None
        )
        content_parts: List[str] = []
        if observation.display_message:
            content_parts.append(observation.display_message)
        elif observation.outcome:
            content_parts.append(observation.outcome)

        # For find/grep, include the formatted file list for LLM
        if canonical in {"find", "grep"} and observation.formatted_output:
            content_parts.append(observation.formatted_output)
            import os
            if os.environ.get("DEVAGENT_DEBUG_TOOLS"):
                print(f"[DEBUG-{canonical.upper()}] Sending to LLM: {len(content_parts[-1])} chars", flush=True)

        if canonical == RUN:
            stdout_preview = observation.metrics.get("stdout_tail")
            if not stdout_preview and observation.raw_output:
                stdout_section = observation.raw_output.split("STDERR:", 1)[0]
                lines = [line for line in stdout_section.splitlines()[1:] if line.strip()]
                stdout_preview = "\n".join(lines)
            if isinstance(stdout_preview, str):
                stdout_preview = stdout_preview.strip()
            if stdout_preview:
                content_parts.append(f"STDOUT:\n{stdout_preview}")

            stderr_preview = observation.metrics.get("stderr_tail")
            if not stderr_preview and observation.raw_output and "STDERR:" in observation.raw_output:
                stderr_section = observation.raw_output.split("STDERR:", 1)[1]
                stderr_preview = stderr_section.strip()
            if isinstance(stderr_preview, str):
                stderr_preview = stderr_preview.strip()
            if stderr_preview:
                content_parts.append(f"STDERR:\n{stderr_preview}")

        if observation.artifact_path:
            content_parts.append(f"(See {observation.artifact_path} for full output)")
        content = "\n".join(part for part in content_parts if part) or observation.outcome or ""
        if not content:
            content = f"{action.tool} completed"

        try:
            self.session_manager.add_tool_message(self.session_id, tool_call_id, content)
        except Exception:  # noqa: BLE001 - do not fail loop for logging issues
            LOGGER.debug("Failed to record tool message for %s", action.tool, exc_info=True)



def create_tool_invoker(
    workspace: Path,
    settings: Settings,
    code_editor: Optional[CodeEditor] = None,
    test_runner: Optional[TestRunner] = None,
    sandbox=None,
    collector: Optional[MetricsCollector] = None,
    pipeline_commands: Optional[PipelineCommands] = None,
    devagent_cfg: Optional[DevAgentConfig] = None,
) -> RegistryToolInvoker:
    """Factory to create a configured tool invoker."""

    return RegistryToolInvoker(
        workspace=workspace,
        settings=settings,
        code_editor=code_editor,
        test_runner=test_runner,
        sandbox=sandbox,
        collector=collector,
        pipeline_commands=pipeline_commands,
        devagent_cfg=devagent_cfg,
    )


__all__ = ["RegistryToolInvoker", "SessionAwareToolInvoker", "create_tool_invoker"]
