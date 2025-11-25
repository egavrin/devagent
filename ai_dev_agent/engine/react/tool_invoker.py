"""Tool invoker that routes ReAct tool calls to registry-backed implementations."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_dev_agent.core.utils.artifacts import write_artifact
from ai_dev_agent.core.utils.constants import MIN_TOOL_OUTPUT_CHARS
from ai_dev_agent.core.utils.context_budget import DEFAULT_MAX_TOOL_OUTPUT_CHARS, summarize_text
from ai_dev_agent.core.utils.devagent_config import DevAgentConfig, load_devagent_yaml
from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.core.utils.tool_utils import canonical_tool_name
from ai_dev_agent.session import SessionManager
from ai_dev_agent.tools import EDIT, READ, RUN, ToolContext, registry
from ai_dev_agent.tools.execution.shell_session import ShellSessionManager
from ai_dev_agent.tools.filesystem.search_replace import PatchApplier, PatchFormatError, PatchParser

# Pipeline module removed - functionality integrated into tool invoker
from .types import ActionRequest, CLIObservation, Observation, ToolCall, ToolResult

if TYPE_CHECKING:
    from ai_dev_agent.core.utils.config import Settings
    from ai_dev_agent.engine.metrics import MetricsCollector
    from ai_dev_agent.tools.code.code_edit.editor import CodeEditor
    from ai_dev_agent.tools.execution.testing.local_tests import TestRunner

LOGGER = get_logger(__name__)
STREAM_PREVIEW_CHAR_LIMIT = 120


class RegistryToolInvoker:
    """Invoke registered tools backed by the shared registry."""

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        code_editor: CodeEditor | None = None,
        test_runner: TestRunner | None = None,
        sandbox=None,
        collector: MetricsCollector | None = None,
        pipeline_commands: Any | None = None,  # Removed - kept for compatibility
        devagent_cfg: DevAgentConfig | None = None,
        shell_session_manager: ShellSessionManager | None = None,
        shell_session_id: str | None = None,
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
        self._structure_hints: dict[str, Any] = {
            "symbols": set(),
            "files": {},
            "project_summary": None,
        }
        # File read cache: path -> (result, timestamp)
        self._file_read_cache: dict[str, tuple[dict[str, Any], float]] = {}
        self._cache_ttl = 60.0  # Cache for 60 seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _sanitize_file_entries(self, entries: Any) -> list[str]:
        """Normalize tool-provided file listings into a list of string paths."""
        sanitized: list[str] = []
        if not isinstance(entries, (list, tuple)):
            return sanitized

        for entry in entries:
            if isinstance(entry, Mapping):
                candidate = entry.get("path") or entry.get("file")
                if candidate:
                    sanitized.append(str(candidate))
            elif isinstance(entry, (str, Path)):
                sanitized.append(str(entry))
        return sanitized

    @staticmethod
    def _sanitize_artifact_list(values: Any) -> list[str]:
        """Normalize arbitrary artifact collections into a clean list of strings."""
        sanitized: list[str] = []
        if not isinstance(values, (list, tuple, set)):
            return sanitized
        for entry in values:
            if isinstance(entry, (str, Path)):
                if entry:
                    sanitized.append(str(entry))
        return sanitized

    @staticmethod
    def _coerce_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value
        else:
            try:
                text = str(value)
            except Exception:
                return None
        stripped = text.strip("\n")
        return stripped if stripped else None

    @staticmethod
    def _truncate_stream_preview(text: str | None) -> str | None:
        if not text:
            return None
        if len(text) <= STREAM_PREVIEW_CHAR_LIMIT:
            return text
        return text[: STREAM_PREVIEW_CHAR_LIMIT - 1] + "…"

    @staticmethod
    def _dump_json(value: Any) -> str:
        try:
            return json.dumps(value, indent=2)
        except TypeError:
            return json.dumps(value, default=str, indent=2)

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
                metrics={"error_type": "KeyError"},
            )
        except ValueError as exc:
            return Observation(
                success=False,
                outcome=f"Tool {tool_name} rejected input",
                tool=tool_name,
                error=str(exc),
                metrics={"error_type": "ValueError"},
            )
        except Exception as exc:
            LOGGER.exception("Tool %s execution failed", tool_name)
            return Observation(
                success=False,
                outcome=f"Tool {tool_name} failed",
                tool=tool_name,
                error=str(exc),
                metrics={"error_type": exc.__class__.__name__},
            )

        return self._wrap_result(tool_name, result)

    def invoke_batch(self, tool_calls: list[ToolCall]) -> Observation:
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
        aggregated_metrics: dict[str, Any] = {
            "total_calls": total_calls,
            "successful_calls": success_count,
            "failed_calls": total_calls - success_count,
        }

        # Sum up costs and wall times
        total_wall_time = sum(r.wall_time for r in results if r.wall_time)
        if total_wall_time > 0:
            aggregated_metrics["total_wall_time"] = total_wall_time
            aggregated_metrics["max_wall_time"] = max(
                (r.wall_time for r in results if r.wall_time), default=0
            )

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

    async def _execute_batch_async(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
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
                metrics={"error_type": "KeyError"},
                wall_time=time.time() - start_time,
            )
        except ValueError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=False,
                outcome=f"Tool {tool_name} rejected input",
                error=str(exc),
                metrics={"error_type": "ValueError"},
                wall_time=time.time() - start_time,
            )
        except Exception as exc:
            LOGGER.exception("Tool %s execution failed", tool_name)
            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=False,
                outcome=f"Tool {tool_name} failed",
                error=str(exc),
                metrics={"error_type": exc.__class__.__name__},
                wall_time=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Registry invocation helpers
    # ------------------------------------------------------------------

    def _invoke_registry(self, tool_name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        # Check cache for READ operations
        if tool_name == READ:
            cache_key = self._get_read_cache_key(payload)
            if cache_key:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    LOGGER.debug(f"Cache hit for READ: {cache_key}")
                    return cached_result

        extra: dict[str, Any] = {
            "code_editor": self.code_editor,
            "test_runner": self.test_runner,
            "pipeline_commands": self.pipeline_commands,
            "structure_hints": self._export_structure_hints(),
        }

        if isinstance(self.shell_session_manager, ShellSessionManager) and isinstance(
            self.shell_session_id, str
        ):
            extra["shell_session_manager"] = self.shell_session_manager
            extra["shell_session_id"] = self.shell_session_id

        # Pass cli_context and llm_client for agent delegation (set by SessionAwareToolInvoker)
        if hasattr(self, "cli_context") and self.cli_context is not None:
            extra["cli_context"] = self.cli_context
            # Pass delegation flag from context to prevent nested planning
            if hasattr(self.cli_context, "obj") and isinstance(self.cli_context.obj, dict):
                extra["is_delegated"] = self.cli_context.obj.get("is_delegated", False)
        if hasattr(self, "llm_client") and self.llm_client is not None:
            extra["llm_client"] = self.llm_client
        if hasattr(self, "session_id") and self.session_id is not None:
            extra["session_id"] = self.session_id

        if tool_name in (EDIT, "edit"):
            LOGGER.debug(
                "EDIT payload received keys=%s",
                list(payload.keys()),
            )
            checklist_failure = self._validate_edit_payload(payload)
            if checklist_failure is not None:
                return checklist_failure

        ctx = ToolContext(
            repo_root=self.workspace,
            settings=self.settings,
            sandbox=self.sandbox,
            devagent_config=self.devagent_cfg,
            metrics_collector=self.collector,
            extra=extra,
        )
        result = registry.invoke(tool_name, payload, ctx)

        # Cache successful READ results (READ returns {"files": [...]}, not {"success": True})
        if tool_name == READ and isinstance(result, dict) and "files" in result:
            cache_key = self._get_read_cache_key(payload)
            if cache_key:
                self._add_to_cache(cache_key, result)
                LOGGER.debug(f"Cached READ result: {cache_key}")

        # Invalidate cache for EDIT/RUN operations (they may not have "success" field either)
        # For EDIT/RUN, any non-exception result means success - they would have raised otherwise
        if tool_name in (EDIT, "edit", RUN) and isinstance(result, dict):
            self._invalidate_cache_for_write_or_run(tool_name, payload, result)

        return result

    def _validate_edit_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Run a pre-flight checklist for EDIT payloads before invoking registry tools.

        Uses parse_with_warnings() to auto-correct minor format issues (like missing colons)
        and layered fuzzy matching for whitespace tolerance.
        """
        patch_value = payload.get("patch")
        if not isinstance(patch_value, str) or not patch_value.strip():
            LOGGER.warning(
                "EDIT invocation missing patch payload; received keys: %s", list(payload.keys())
            )
            return {
                "success": False,
                "errors": [
                    "EDIT tool requires a 'patch' string containing the canonical "
                    "*** Begin Patch ... *** End Patch payload."
                ],
                "warnings": [],
                "changed_files": [],
                "new_files": [],
            }

        all_warnings: list[str] = []

        try:
            parser = PatchParser(patch_value)
            parse_result = parser.parse_with_warnings()
            actions = parse_result.actions
            all_warnings.extend(parse_result.warnings)
        except PatchFormatError as exc:
            return {
                "success": False,
                "errors": [str(exc)],
                "warnings": [],
                "changed_files": [],
                "new_files": [],
            }

        validator = PatchApplier(self.workspace)
        validation = validator.apply(actions, dry_run=True)
        if not validation["success"]:
            # Combine warnings and add recovery steps
            all_warnings.extend(validation.get("warnings", []))
            return {
                "success": False,
                "errors": validation["errors"],
                "warnings": all_warnings,
                "changed_files": [],
                "new_files": [],
            }

        # If we got parser warnings (auto-corrections), log them
        if all_warnings:
            LOGGER.info("EDIT pre-validation passed with warnings: %s", all_warnings)

        return None

    def _get_read_cache_key(self, payload: Mapping[str, Any]) -> str | None:
        """Generate cache key for READ operations.

        Supports both single and multiple file reads by creating composite keys.
        """
        import hashlib

        paths = payload.get("paths", [])
        if not paths:
            return None

        # Single file - use path directly for readability
        if len(paths) == 1:
            return str(paths[0])

        # Multiple files - create composite key using hash
        sorted_paths = sorted(str(p) for p in paths)
        composite = "|".join(sorted_paths)
        # Use hash to keep key size manageable
        return "multi_" + hashlib.md5(composite.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> dict[str, Any] | None:
        """Get cached result if not expired."""
        if key not in self._file_read_cache:
            return None

        result, timestamp = self._file_read_cache[key]
        if time.time() - timestamp > self._cache_ttl:
            # Expired, remove from cache
            del self._file_read_cache[key]
            return None

        return result

    def _add_to_cache(self, key: str, result: dict[str, Any]) -> None:
        """Add result to cache with current timestamp."""
        self._file_read_cache[key] = (result, time.time())

    def _invalidate_cache(self, path: str) -> None:
        """Invalidate cache entry for a specific file path."""
        if path in self._file_read_cache:
            del self._file_read_cache[path]
            LOGGER.debug(f"Invalidated cache for: {path}")

    def _invalidate_cache_for_write_or_run(
        self, tool_name: str, payload: Mapping[str, Any], result: Mapping[str, Any]
    ) -> None:
        """Invalidate cache entries that might be affected by EDIT or RUN operations."""
        if tool_name in (EDIT, "edit"):
            # EDIT returns changed_files and new_files - invalidate those
            if result.get("success"):
                changed = result.get("changed_files", [])
                new = result.get("new_files", [])
                for file_path in set(changed + new):
                    self._invalidate_cache(file_path)
        elif tool_name == RUN:
            # RUN might modify files - only clear cache if command likely modifies files
            command = payload.get("command", "")
            # Detect file-modifying commands
            file_modifying_keywords = [
                "write",
                "mv",
                "rm",
                "cp",
                "touch",
                "sed",
                "awk",
                ">",
                ">>",
                "tee",
                "dd",
                "rsync",
            ]
            likely_modifies_files = any(keyword in command for keyword in file_modifying_keywords)

            if likely_modifies_files:
                LOGGER.debug(f"Clearing file cache: RUN command may modify files: {command[:50]}")
                self._file_read_cache.clear()
            else:
                LOGGER.debug(
                    f"Preserving cache: RUN command unlikely to modify files: {command[:50]}"
                )

    def _wrap_result(self, tool_name: str, result: Mapping[str, Any]) -> Observation:
        success = True
        outcome = f"Executed {tool_name}"
        metrics: dict[str, Any] = {}
        artifacts: list[str] = []
        raw_output: str | None = None

        if tool_name == READ:
            raw_files = result.get("files")
            files_iterable = raw_files if isinstance(raw_files, (list, tuple)) else []
            sanitized_paths: list[str] = []
            total_lines = 0
            for entry in files_iterable:
                if isinstance(entry, Mapping):
                    path_value = entry.get("path")
                    if path_value:
                        sanitized_paths.append(str(path_value))
                    content = entry.get("content")
                    if isinstance(content, str):
                        total_lines += len(content.splitlines())
                elif isinstance(entry, (str, Path)):
                    sanitized_paths.append(str(entry))
            artifacts = sanitized_paths
            outcome = f"Read {len(sanitized_paths)} file(s)"
            metrics = {"files": len(sanitized_paths), "lines_read": total_lines}
            if sanitized_paths:
                metrics["artifacts"] = sanitized_paths
            raw_output = self._dump_json(result)
        elif tool_name == "find":
            sanitized_paths = self._sanitize_file_entries(result.get("files"))
            success = bool(sanitized_paths)
            outcome = f"Found {len(sanitized_paths)} file(s)"
            artifacts = sanitized_paths
            metrics = {"files": len(sanitized_paths), "paths": sanitized_paths[:10]}
            raw_output = self._dump_json((result.get("files") or [])[:20])
        elif tool_name == "grep":
            matches = result.get("matches", [])
            matching_groups = matches if isinstance(matches, (list, tuple)) else []

            # Extract file paths and match counts for better context
            artifacts = []
            match_counts = {}
            for group in matching_groups:
                if isinstance(group, Mapping) and group.get("file"):
                    file_path = str(group.get("file"))
                    artifacts.append(file_path)
                    # Count matches in this file
                    file_matches = group.get("matches", [])
                    match_counts[file_path] = (
                        len(file_matches) if isinstance(file_matches, list) else 0
                    )

            metrics = {
                "files": len(artifacts),
                "match_counts": match_counts,  # Add match counts to metrics
            }
            success = bool(artifacts)
            outcome = f"Found matches in {len(artifacts)} file(s)"
            raw_output = self._dump_json(list(matching_groups[:10]))
        elif tool_name == "symbols":
            symbols = result.get("symbols", [])
            outcome = f"Found {len(symbols)} symbol(s)"
            metrics = {"symbols": len(symbols)}
            artifacts = [
                entry.get("file")
                for entry in symbols
                if isinstance(entry, Mapping) and entry.get("file")
            ]
            raw_output = self._dump_json(symbols[:20])
        elif tool_name == "edit" or tool_name == EDIT:
            success = bool(result.get("success", True))
            errors = result.get("errors", [])

            if success:
                outcome = "Executed edit"
                changed_files = self._sanitize_artifact_list(result.get("changed_files"))
                new_files = self._sanitize_artifact_list(result.get("new_files"))
                combined = self._sanitize_artifact_list(
                    list(dict.fromkeys(changed_files + new_files))
                )
                if combined:
                    artifacts = combined
            else:
                if errors:
                    first_error = errors[0] if isinstance(errors, list) else str(errors)
                    outcome = f"Edit failed: {first_error[:200]}"
                else:
                    outcome = "Edit failed"

            metrics = dict(result)
            raw_output = self._dump_json(result)
        elif tool_name == RUN:
            exit_code = result.get("exit_code", 0)
            success = exit_code == 0
            stdout_text = self._coerce_text(result.get("stdout_tail"))
            stderr_text = self._coerce_text(result.get("stderr_tail"))
            stdout_preview = self._truncate_stream_preview(stdout_text)
            stderr_preview = self._truncate_stream_preview(stderr_text)
            if exit_code != 0 and stderr_preview:
                outcome = f"Command exited with {exit_code} (stderr available)"
            elif stdout_preview:
                outcome = f"Command exited with {exit_code} (stdout available)"
            else:
                outcome = f"Command exited with {exit_code}"
            metrics = {
                "exit_code": exit_code,
                "duration_ms": result.get("duration_ms"),
                "stdout_tail": stdout_text,
                "stderr_tail": stderr_text,
            }
            if stdout_preview and stdout_preview != stdout_text:
                metrics["stdout_preview"] = stdout_preview
            if stderr_preview and stderr_preview != stderr_text:
                metrics["stderr_preview"] = stderr_preview
            raw_output = "STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n".format(
                stdout=stdout_text or "",
                stderr=stderr_text or "",
            )
        else:
            if isinstance(result, Mapping):
                metrics = dict(result)
                raw_output = self._dump_json(result)
            else:
                metrics = {"raw": result}
                raw_output = self._dump_json(result)

        observations_kwargs: dict[str, Any] = {
            "success": success,
            "outcome": outcome,
            "metrics": metrics,
            "artifacts": [item for item in artifacts if item],
            "tool": tool_name,
            "raw_output": raw_output,
        }

        structure_payload = self._export_structure_hints()
        if (
            structure_payload["symbols"]
            or structure_payload["files"]
            or structure_payload["project_summary"]
        ):
            observations_kwargs["structure_hints"] = structure_payload

        return Observation(**observations_kwargs)

    def _export_structure_hints(self) -> dict[str, Any]:
        files_payload: dict[str, Any] = {}
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
        code_editor: CodeEditor | None = None,
        test_runner: TestRunner | None = None,
        sandbox=None,
        collector: MetricsCollector | None = None,
        pipeline_commands: Any | None = None,  # Removed - kept for compatibility
        devagent_cfg: DevAgentConfig | None = None,
        *,
        session_manager: SessionManager | None = None,
        session_id: str | None = None,
        shell_session_manager: ShellSessionManager | None = None,
        shell_session_id: str | None = None,
        cli_context: Any | None = None,
        llm_client: Any | None = None,
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
        self.session_manager = session_manager or (
            SessionManager.get_instance() if session_id else None
        )
        self.session_id = session_id
        self.cli_context = cli_context
        self.llm_client = llm_client
        setting_value = getattr(settings, "max_tool_output_chars", DEFAULT_MAX_TOOL_OUTPUT_CHARS)
        try:
            parsed_setting = int(setting_value)
        except (TypeError, ValueError):
            parsed_setting = DEFAULT_MAX_TOOL_OUTPUT_CHARS
        self._max_tool_output_chars = max(MIN_TOOL_OUTPUT_CHARS, parsed_setting)

    def __call__(self, action: ActionRequest) -> CLIObservation:
        base_observation = super().__call__(action)
        cli_observation = self._to_cli_observation(action, base_observation)

        # For batch execution, record a tool message for each result
        if cli_observation.results:
            LOGGER.debug(
                "Tool %s: batch mode, recording %d tool messages (session_id=%s)",
                action.tool,
                len(cli_observation.results),
                self.session_id,
            )
            for result in cli_observation.results:
                self._record_batch_tool_message(result)
        else:
            # Single tool mode - record the primary action
            LOGGER.debug(
                "Tool %s: single mode, recording tool message (session_id=%s)",
                action.tool,
                self.session_id,
            )
            self._record_tool_message(action, cli_observation)

        return cli_observation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_cli_observation(
        self, action: ActionRequest, observation: Observation
    ) -> CLIObservation:
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

        summary_source = raw_text if canonical == RUN and raw_text else raw_text or outcome_text

        summary_text = summary_source
        artifact_path: Path | None = None

        if summary_source:
            summarized = summarize_text(summary_source, self._max_tool_output_chars)
            if summarized != summary_source:
                summary_text = summarized
                try:
                    artifact_path = write_artifact(summary_source)
                except Exception:
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
                print(
                    f"[DEBUG-FIND] Formatted output for LLM: {len(file_list)}/{total_files} files",
                    flush=True,
                )
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
                print(
                    f"[DEBUG-GREP] Formatted output for LLM: {len(file_list)}/{total_files} files with counts",
                    flush=True,
                )

        display_message = self._format_display_message(action, observation, canonical)

        artifact_display: str | None = None
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
            EDIT: "📝",
            "submit_final_answer": "✅",
        }
        icon = base_icon_map.get(canonical_name, status_ok if success else status_fail)
        status_suffix = status_ok if success else status_fail

        def quote(value: str | None) -> str:
            if not value:
                return ""
            return f' "{value}"'

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
                matches_text = observation.outcome or (
                    "matches located" if success else "no matches"
                )
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
            preview_value: str | None = None
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

        if canonical_name == EDIT:
            artifacts = observation.metrics.get("artifacts", observation.artifacts)
            if success:
                if artifacts and isinstance(artifacts, list) and len(artifacts) > 0:
                    first_file = artifacts[0]
                    count_suffix = f" +{len(artifacts) - 1}" if len(artifacts) > 1 else ""
                    return f"{icon} edit → {first_file}{count_suffix}"
                return f"{icon} edit → applied"

            errors = observation.metrics.get("errors")
            if isinstance(errors, list) and errors:
                first_error = str(errors[0]).strip()
                first_line = first_error.splitlines()[0]
                if len(first_line) > 120:
                    first_line = first_line[:117] + "..."
                return f"{icon} edit → failed ({first_line})"
            return f"{icon} edit → failed"

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
            LOGGER.warning(
                "Skipping tool message recording for %s: session_manager=%s, session_id=%s",
                action.tool,
                "present" if self.session_manager else "missing",
                self.session_id or "missing",
            )
            return

        canonical = canonical_tool_name(action.tool)
        tool_call_id = (
            action.metadata.get("tool_call_id")
            or action.metadata.get("call_id")
            or action.metadata.get("id")
        )

        # If no tool_call_id found, generate one as fallback
        # This ensures tool messages always have an ID
        if not tool_call_id:
            import uuid

            tool_call_id = f"tool-exec-{uuid.uuid4().hex[:8]}"
            LOGGER.debug("No tool_call_id found for %s, generated: %s", action.tool, tool_call_id)
        content_parts: list[str] = []
        if observation.display_message:
            content_parts.append(observation.display_message)
        elif observation.outcome:
            content_parts.append(observation.outcome)

        if canonical == "submit_final_answer" and observation.formatted_output:
            content_parts.append(observation.formatted_output)

        # For find/grep, include the formatted file list for LLM
        if canonical in {"find", "grep"} and observation.formatted_output:
            content_parts.append(observation.formatted_output)
            import os

            if os.environ.get("DEVAGENT_DEBUG_TOOLS"):
                print(
                    f"[DEBUG-{canonical.upper()}] Sending to LLM: {len(content_parts[-1])} chars",
                    flush=True,
                )

        # For EDIT failures, include detailed error information so LLM can fix the patch
        if canonical == EDIT and not observation.success:
            errors = observation.metrics.get("errors")
            if isinstance(errors, list) and errors:
                error_details = "\n".join(f"  - {err}" for err in errors)
                content_parts.append(f"Error details:\n{error_details}")
            elif observation.formatted_output:
                content_parts.append(observation.formatted_output)
            elif observation.raw_output:
                import json

                try:
                    error_data = json.loads(observation.raw_output)
                    raw_errors = error_data.get("errors", [])
                    if raw_errors:
                        error_details = "\n".join(f"  - {err}" for err in raw_errors)
                        content_parts.append(f"Error details:\n{error_details}")
                except (json.JSONDecodeError, AttributeError):
                    content_parts.append(f"Raw error: {observation.raw_output[:500]}")

            # Add recovery steps to help LLM self-correct
            content_parts.append(
                "\nRECOVERY STEPS:\n"
                "1. Use READ to view the current file content\n"
                "2. Ensure directive headers have colons (*** Update File: path)\n"
                "3. Copy exact lines from READ output into your - lines\n"
                "4. Rebuild the patch and retry"
            )

        # For EDIT successes with warnings (auto-corrections), surface them
        if canonical == EDIT and observation.success:
            warnings = observation.metrics.get("warnings")
            if isinstance(warnings, list) and warnings:
                warning_details = "\n".join(f"  - {w}" for w in warnings)
                content_parts.append(f"Note (auto-corrected):\n{warning_details}")

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
            if (
                not stderr_preview
                and observation.raw_output
                and "STDERR:" in observation.raw_output
            ):
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
        except Exception as e:
            LOGGER.warning(
                "Failed to record tool message for %s (tool_call_id=%s, session_id=%s): %s",
                action.tool,
                tool_call_id,
                self.session_id,
                e,
                exc_info=True,
            )

    def _record_batch_tool_message(self, result: ToolResult) -> None:
        """Record a tool message for a single result from batch execution."""
        if not self.session_manager or not self.session_id:
            return

        # Build content from result
        content_parts = []
        if result.outcome:
            content_parts.append(result.outcome)

        if result.error:
            content_parts.append(f"Error: {result.error}")

        content = "\n".join(content_parts) or f"{result.tool} completed"

        # Ensure tool_call_id is never None
        tool_call_id = result.call_id
        if not tool_call_id:
            import uuid

            tool_call_id = f"tool-batch-{uuid.uuid4().hex[:8]}"
            LOGGER.debug(
                "No call_id found for batch tool %s, generated: %s", result.tool, tool_call_id
            )

        try:
            self.session_manager.add_tool_message(self.session_id, tool_call_id, content)
        except Exception as e:
            LOGGER.warning(
                "Failed to record batch tool message for %s (tool_call_id=%s, session_id=%s): %s",
                result.tool,
                tool_call_id,
                self.session_id,
                e,
                exc_info=True,
            )


def create_tool_invoker(
    workspace: Path,
    settings: Settings,
    code_editor: CodeEditor | None = None,
    test_runner: TestRunner | None = None,
    sandbox=None,
    collector: MetricsCollector | None = None,
    pipeline_commands: Any | None = None,  # Removed - kept for compatibility
    devagent_cfg: DevAgentConfig | None = None,
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
