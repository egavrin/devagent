"""LLM-backed action provider bridging the CLI to the engine ReAct loop."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence
from uuid import uuid4

from ai_dev_agent.core.utils.budget_integration import BudgetIntegration

logger = logging.getLogger(__name__)
from ai_dev_agent.engine.react.types import ActionRequest, StepRecord, TaskSpec, ToolCall
from ai_dev_agent.providers.llm.base import LLMClient, ToolCallResult
from ai_dev_agent.session import SessionManager

from .budget_control import create_text_only_tool


class LLMActionProvider:
    """Produce actions for the engine's ReAct loop by invoking the CLI LLM client."""

    def __init__(
        self,
        llm_client: LLMClient,
        session_manager: SessionManager,
        session_id: str,
        *,
        tools: Sequence[Dict[str, Any]] | None = None,
        budget_integration: Optional[BudgetIntegration] = None,
        format_schema: Optional[Dict[str, Any]] = None,
        ctx_obj: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client = llm_client
        self.session_manager = session_manager
        self.session_id = session_id
        self.budget_integration = budget_integration
        self.format_schema = format_schema
        self._base_tools = list(tools or [])
        self._current_phase: str = "exploration"
        self._is_final_iteration: bool = False
        self._last_response: Optional[ToolCallResult] = None
        self._ctx_obj = ctx_obj or {}

    def update_phase(self, phase: str, *, is_final: bool = False) -> None:
        """Record the active behavioural phase for subsequent prompts."""

        self._current_phase = phase
        self._is_final_iteration = bool(is_final)

    def __call__(self, task: TaskSpec, history: Sequence[StepRecord]) -> ActionRequest:
        """Invoke the LLM to obtain the next tool action."""

        iteration_index = len(history) + 1

        # Check if RepoMap refresh is pending and inject before LLM call
        if self._ctx_obj.get("_repomap_refresh_pending"):
            self._inject_repomap_update()
            self._ctx_obj["_repomap_refresh_pending"] = False

        conversation = self.session_manager.compose(self.session_id)
        tools = self._get_tools_for_phase(history)

        # Build kwargs for LLM call
        llm_kwargs = {
            "temperature": 0.1,
        }

        # When format_schema present, set response_format on all iterations
        # but only switch to text-only mode on final iteration
        if self.format_schema:
            schema_root_type = self.format_schema.get("type", "object")
            if schema_root_type == "object":
                # Always request JSON output when schema present
                llm_kwargs["response_format"] = {"type": "json_object"}

        # On final iteration with format_schema, switch to text-only mode
        if self.format_schema and self._is_final_iteration:
            # Call complete instead of invoke_tools for JSON-only mode
            if self.budget_integration:
                content = self.budget_integration.execute_with_retry(
                    self.client.complete,
                    conversation,
                    **llm_kwargs,
                )
            else:
                content = self.client.complete(
                    conversation,
                    **llm_kwargs,
                )
            # Wrap in ToolCallResult format for compatibility
            result = ToolCallResult(calls=[], message_content=content)
        else:
            llm_kwargs["tools"] = tools
            if self.budget_integration:
                result: ToolCallResult = self.budget_integration.execute_with_retry(
                    self.client.invoke_tools,
                    conversation,
                    **llm_kwargs,
                )
            else:
                result = self.client.invoke_tools(
                    conversation,
                    **llm_kwargs,
                )

        self._last_response = result
        normalized_tool_calls = self._normalize_tool_calls(result.raw_tool_calls, result.calls)
        if normalized_tool_calls is not None:
            result.raw_tool_calls = normalized_tool_calls

        if result.message_content is not None or normalized_tool_calls:
            self.session_manager.add_assistant_message(
                self.session_id,
                result.message_content,
                tool_calls=normalized_tool_calls,
            )

        if self.budget_integration and getattr(result, "_raw_response", None):
            try:
                model_name = getattr(self.client, "model", "unknown")
                self.budget_integration.track_llm_call(
                    model=model_name,
                    response_data=result._raw_response,  # type: ignore[attr-defined]
                    operation="tool_invocation",
                    iteration=iteration_index,
                    phase=self._current_phase,
                )
            except Exception:
                # Tracking should never break iteration flow.
                pass

        used_call_ids: set[str] = set()
        for entry in normalized_tool_calls or []:
            if isinstance(entry, Mapping):
                existing = entry.get("id") or entry.get("tool_call_id")
                if isinstance(existing, str) and existing:
                    used_call_ids.add(existing)

        if not result.calls:
            self._record_dummy_tool_messages(normalized_tool_calls)
            raise StopIteration("No tool calls - synthesis complete")

        tool_calls = self._convert_tool_calls(result.calls, used_call_ids)
        primary_call = tool_calls[0]
        metadata: Dict[str, Any] = {
            "iteration": iteration_index,
            "phase": self._current_phase,
        }
        if primary_call.call_id:
            metadata["tool_call_id"] = primary_call.call_id

        tool_calls_payload = tool_calls if len(tool_calls) > 1 else []

        return ActionRequest(
            step_id=f"S{iteration_index}",
            thought=result.message_content or "",
            tool=primary_call.tool,
            args=primary_call.args,
            metadata=metadata,
            tool_calls=tool_calls_payload,
        )

    def last_response_text(self) -> Optional[str]:
        """Return the most recent assistant message content, if available."""

        if not self._last_response or self._last_response.message_content is None:
            return None
        text = self._last_response.message_content.strip()
        return text or None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tools_for_phase(self, _history: Sequence[StepRecord]) -> List[Dict[str, Any]]:
        if self._is_final_iteration:
            return [create_text_only_tool()]
        return list(self._base_tools)

    def _convert_tool_calls(self, calls: Sequence[Any], used_ids: set[str]) -> List[ToolCall]:
        normalized: List[ToolCall] = []
        for index, call in enumerate(calls):
            name = getattr(call, "name", None)
            arguments = getattr(call, "arguments", {}) or {}
            call_id = getattr(call, "call_id", None) or getattr(call, "id", None)
            if not isinstance(arguments, dict):
                try:
                    arguments = dict(arguments)  # type: ignore[arg-type]
                except Exception:
                    arguments = {"value": arguments}
            if not call_id:
                call_id = self._generate_call_identifier(index, used_ids)
            else:
                used_ids.add(call_id)
            normalized.append(
                ToolCall(
                    tool=str(name),
                    args=arguments,
                    call_id=call_id,
                )
            )
        return normalized

    def _normalize_tool_calls(
        self,
        raw_tool_calls: Optional[Sequence[Any]],
        parsed_calls: Optional[Sequence[Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not raw_tool_calls:
            return raw_tool_calls  # type: ignore[return-value]

        used_ids: set[str] = set()
        for entry in raw_tool_calls:
            if isinstance(entry, Dict):
                existing = str(entry.get("id") or entry.get("tool_call_id") or "").strip()
                if existing:
                    used_ids.add(existing)

        normalized: List[Dict[str, Any]] = []
        for index, entry in enumerate(raw_tool_calls):
            if isinstance(entry, Dict):
                entry_dict: Dict[str, Any] = dict(entry)
            else:
                entry_dict = getattr(entry, "__dict__", {}).copy()
                if not entry_dict:
                    entry_dict = {"raw": entry}

            call_id = entry_dict.get("id") or entry_dict.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                call_id = self._generate_call_identifier(index, used_ids)
            else:
                used_ids.add(call_id)
            entry_dict["id"] = call_id
            normalized.append(entry_dict)

            if parsed_calls and index < len(parsed_calls):
                parsed_call = parsed_calls[index]
                parsed_id = getattr(parsed_call, "call_id", None) or getattr(parsed_call, "id", None)
                if parsed_id != call_id:
                    setattr(parsed_call, "call_id", call_id)

        return normalized

    def _generate_call_identifier(self, index: int, used_ids: set[str]) -> str:
        while True:
            candidate = f"tool-{index}-{uuid4().hex[:8]}"
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate

    def _inject_repomap_update(self) -> None:
        """Inject RepoMap update into session before next LLM call."""
        from pathlib import Path

        try:
            # Get context objects
            dynamic_context = self._ctx_obj.get("_dynamic_context")
            user_prompt = self._ctx_obj.get("_user_prompt", "")
            settings = self._ctx_obj.get("settings")
            repo_root = Path.cwd()

            if not dynamic_context:
                return

            # Generate fresh RepoMap with accumulated context
            from ai_dev_agent.cli.context_enhancer import get_context_enhancer
            enhancer = get_context_enhancer(repo_root, settings)
            system_msg, _ = enhancer.get_repomap_messages(
                query=user_prompt,
                max_files=15,
                additional_files=dynamic_context.mentioned_files,
                additional_symbols=dynamic_context.mentioned_symbols
            )

            if system_msg:
                # Inject as system message to avoid breaking tool call sequences
                self.session_manager.add_system_message(
                    self.session_id,
                    system_msg,
                    location="after_initial"  # Add after initial system prompts
                )
                if settings and settings.repomap_debug_stdout:
                    logger.debug("Updated RepoMap injected as system message")

        except Exception as e:
            settings = self._ctx_obj.get("settings")
            if settings and settings.repomap_debug_stdout:
                logger.debug(f"Failed to inject RepoMap update: {e}")

    def _record_dummy_tool_messages(self, raw_tool_calls: Optional[Sequence[Any]]) -> None:
        if not raw_tool_calls:
            return
        for entry in raw_tool_calls:
            if not isinstance(entry, Mapping):
                continue
            function_payload = entry.get("function") if isinstance(entry.get("function"), Mapping) else {}
            tool_name = (
                function_payload.get("name")
                or entry.get("name")
                or "tool"
            )
            call_id = entry.get("id") or entry.get("tool_call_id")
            message = (
                f"Tool '{tool_name}' was referenced but not executed."
            )
            try:
                self.session_manager.add_tool_message(self.session_id, call_id, message)
            except Exception:
                pass
