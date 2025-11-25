"""Natural-language intent routing leveraging LLM tool calling."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ai_dev_agent.agents import AgentRegistry, AgentSpec
from ai_dev_agent.cli.utils import build_system_context

# Tool metadata is registered directly under canonical names
from ai_dev_agent.providers.llm.base import LLMClient, LLMError, Message, ToolCall, ToolCallResult
from ai_dev_agent.session import SessionManager, build_system_messages
from ai_dev_agent.tools import EDIT, FIND, GREP, READ, RUN, SYMBOLS
from ai_dev_agent.tools import registry as tool_registry

if TYPE_CHECKING:
    from ai_dev_agent.core.utils.config import Settings

DEFAULT_TOOLS: list[dict[str, Any]] = []

_SYSTEM_CONTEXT_DEFAULTS: dict[str, Any] = {
    "os": "unknown",
    "os_friendly": "Unknown OS",
    "os_version": "unknown",
    "shell": "/bin/sh",
    "shell_type": "unix",
    "architecture": "unknown",
    "home_dir": str(Path.home()),
    "cwd": ".",
    "python_version": "",
    "available_tools": [],
    "is_unix": True,
    "path_separator": "/",
    "command_separator": "&&",
    "null_device": "/dev/null",
    "temp_dir": "/tmp",
    "command_mappings": {},
    "platform_examples": "",
}


@dataclass
class IntentDecision:
    """Result returned by the intent router."""

    tool: str | None
    arguments: dict[str, Any]
    rationale: str | None = None


class IntentRoutingError(RuntimeError):
    """Raised when the router cannot derive a suitable intent."""


class IntentRouter:
    """Use LLM tool-calling to map natural language prompts onto CLI intents."""

    def __init__(
        self,
        client: LLMClient | None,
        settings: Settings,
        agent_type: str = "manager",
        tools: list[dict[str, Any]] | None = None,
        project_profile: dict[str, Any] | None = None,
        tool_success_history: dict[str, Any] | None = None,
        is_delegated: bool = False,
    ) -> None:
        self.client = client
        self.settings = settings
        self.agent_spec = AgentRegistry.get(agent_type)
        self._is_delegated_context = is_delegated  # Track if this is a delegated execution
        try:
            system_context = build_system_context()
        except Exception:
            system_context = {}
        self._system_context = self._normalise_system_context(system_context)
        self.tools = (
            list(tools) if tools is not None else self._build_tool_list(settings, self.agent_spec)
        )
        self.project_profile = project_profile or {}
        self.tool_success_history = tool_success_history or {}
        self._session_manager = SessionManager.get_instance()
        self._session_id = f"router-{uuid4()}"

    def _build_tool_list(self, settings: Settings, agent_spec: AgentSpec) -> list[dict[str, Any]]:
        """Combine core tools with selected registry tools, avoiding duplicates."""
        combined: list[dict[str, Any]] = []
        used_names: set[str] = set()
        for entry in DEFAULT_TOOLS:
            fn = entry.get("function", {})
            name = fn.get("name")
            if name:
                used_names.add(name)
            combined.append(entry)

        combined.extend(self._build_registry_tools(settings, agent_spec, used_names))
        return combined

    def _normalise_system_context(self, context: Any) -> dict[str, Any]:
        """Ensure the system context is a dict with expected fields."""
        normalized = dict(_SYSTEM_CONTEXT_DEFAULTS)

        if isinstance(context, dict):
            normalized.update(context)

        available_tools = normalized.get("available_tools")
        if isinstance(available_tools, list):
            normalized["available_tools"] = [str(tool) for tool in available_tools if tool]
        elif available_tools:
            normalized["available_tools"] = [str(available_tools)]
        else:
            normalized["available_tools"] = []

        command_mappings = normalized.get("command_mappings")
        if not isinstance(command_mappings, dict):
            normalized["command_mappings"] = {}

        for key in (
            "os",
            "os_friendly",
            "os_version",
            "shell",
            "shell_type",
            "architecture",
            "python_version",
        ):
            normalized[key] = str(normalized.get(key) or _SYSTEM_CONTEXT_DEFAULTS.get(key, ""))

        normalized["cwd"] = str(normalized.get("cwd") or ".")
        normalized["home_dir"] = str(normalized.get("home_dir") or "")
        normalized["command_separator"] = str(
            normalized.get("command_separator") or _SYSTEM_CONTEXT_DEFAULTS["command_separator"]
        )
        normalized["path_separator"] = str(
            normalized.get("path_separator") or _SYSTEM_CONTEXT_DEFAULTS["path_separator"]
        )
        normalized["null_device"] = str(
            normalized.get("null_device") or _SYSTEM_CONTEXT_DEFAULTS["null_device"]
        )
        normalized["temp_dir"] = str(
            normalized.get("temp_dir") or _SYSTEM_CONTEXT_DEFAULTS["temp_dir"]
        )
        normalized["platform_examples"] = str(
            normalized.get("platform_examples")
            or _SYSTEM_CONTEXT_DEFAULTS.get("platform_examples", "")
        )
        normalized["is_unix"] = bool(normalized.get("is_unix"))

        return normalized

    def _build_registry_tools(
        self, settings: Settings, agent_spec: AgentSpec, used_names: set[str]
    ) -> list[dict[str, Any]]:
        """Translate registry specs into LLM tool definitions filtered by agent's allowed tools."""
        # Full safelist of all available tools (WRITE removed - use EDIT)
        all_tools = [FIND, GREP, SYMBOLS, READ, RUN, EDIT]

        # Add workflow tools (plan, delegate) only if not in a delegated context
        # This prevents nested planning attempts
        is_delegated = getattr(self, "_is_delegated_context", False)
        if not is_delegated:
            # Workflow tools are only available at top level
            workflow_tools = ["plan", "delegate"]
            all_tools.extend(workflow_tools)

        # Filter to only include tools allowed for this agent
        safelist = [name for name in all_tools if name in agent_spec.tools]

        tools: list[dict[str, Any]] = []
        for name in safelist:
            try:
                spec = tool_registry.get(name)
            except KeyError:
                continue
            if name in used_names:
                continue
            try:
                with spec.request_schema_path.open("r", encoding="utf-8") as handle:
                    params_schema = json.load(handle)
            except Exception:
                params_schema = {"type": "object", "properties": {}, "additionalProperties": True}

            if name == RUN:
                params_schema = self._augment_run_schema(params_schema)

            description = spec.description or ""
            if name == RUN:
                description = self._augment_run_description(description)
            # WRITE has been removed - EDIT now handles canonical apply_patch payloads

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": params_schema,
                    },
                }
            )
            used_names.add(name)
        return tools

    def route_prompt(self, prompt: str) -> IntentDecision:
        """Legacy entrypoint used by the CLI router tests."""
        return self._route_internal(prompt, prefer_generate=True)

    def route(self, prompt: str) -> IntentDecision:
        """Default entrypoint that prefers invoke_tools when available."""
        return self._route_internal(prompt, prefer_generate=False)

    def _route_internal(self, prompt: str, *, prefer_generate: bool) -> IntentDecision:
        text = prompt.strip()
        if not text:
            raise IntentRoutingError("Empty prompt provided for intent routing.")

        if self.client is None:
            raise IntentRoutingError("No LLM client available; fallback routing is disabled.")

        system_messages = build_system_messages(
            include_react_guidance=False,
            extra_messages=[self._system_prompt()],
            provider=getattr(self.settings, "provider", None),
            model=getattr(self.settings, "model", None),
            workspace_root=getattr(self.settings, "workspace_root", None),
            settings=self.settings,
        )
        if not system_messages:
            system_messages = [Message(role="system", content=self._system_prompt())]
        session = self._session_manager.ensure_session(
            self._session_id,
            system_messages=system_messages,
            metadata={
                "mode": "intent-router",
                "project_profile": self.project_profile,
            },
        )
        with session.lock:
            session.metadata["last_prompt"] = text

        self._session_manager.add_user_message(self._session_id, text)

        conversation_payload = self._session_manager.compose(self._session_id)
        if isinstance(conversation_payload, Sequence):
            conversation = list(conversation_payload)
        else:
            conversation = [*list(system_messages), Message(role="user", content=text)]

        try:
            invocation = self._invoke_model(conversation, prefer_generate)
        except LLMError as exc:
            self._session_manager.add_system_message(
                self._session_id, f"Intent routing error: {exc}"
            )
            raise IntentRoutingError(f"Intent routing failed: {exc}") from exc
        except IntentRoutingError:
            raise
        except Exception as exc:
            self._session_manager.add_system_message(
                self._session_id, f"Unexpected intent routing failure: {exc}"
            )
            raise IntentRoutingError(f"Unexpected routing error: {exc}") from exc

        result = self._coerce_tool_call_result(invocation)
        raw_tool_calls = result.raw_tool_calls or self._build_raw_tool_calls(result.calls)

        if result.calls:
            call = result.calls[0]
            arguments = self._parse_arguments(result, call)
            rationale = self._normalize_rationale(result.message_content)
            self._session_manager.add_assistant_message(
                self._session_id,
                result.message_content,
                tool_calls=raw_tool_calls,
            )
            return IntentDecision(tool=call.name, arguments=arguments, rationale=rationale)

        message = self._normalize_rationale(result.message_content)
        if prefer_generate:
            raise IntentRoutingError("Could not determine a tool from the model response.")
        if message:
            self._session_manager.add_assistant_message(self._session_id, message)
            return IntentDecision(tool=None, arguments={"text": message})

        raise IntentRoutingError("Could not determine a tool from the model response.")

    def _invoke_model(self, messages: Sequence[Message], prefer_generate: bool):
        kwargs = {"tools": self.tools, "temperature": 0.1}
        if prefer_generate and hasattr(self.client, "generate_with_tools"):
            return self.client.generate_with_tools(messages, **kwargs)
        if hasattr(self.client, "invoke_tools"):
            return self.client.invoke_tools(messages, **kwargs)
        if hasattr(self.client, "generate_with_tools"):
            return self.client.generate_with_tools(messages, **kwargs)
        raise IntentRoutingError("LLM client does not support tool routing.")

    def _coerce_tool_call_result(self, data: Any) -> ToolCallResult:
        if isinstance(data, ToolCallResult):
            return data

        if isinstance(data, tuple) and len(data) == 2:
            message, entries = data
            calls: list[ToolCall] = []
            raw_tool_calls: list[dict[str, Any]] = []
            for entry in entries or []:
                if isinstance(entry, ToolCallResult):
                    calls.extend(entry.calls)
                    if entry.raw_tool_calls:
                        raw_tool_calls.extend(entry.raw_tool_calls)
                    elif entry.calls:
                        payload = self._build_raw_tool_calls(entry.calls)
                        if payload:
                            raw_tool_calls.extend(payload)
                elif isinstance(entry, ToolCall):
                    calls.append(entry)
                elif isinstance(entry, dict):
                    name = entry.get("name") or entry.get("function", {}).get("name")
                    arguments = entry.get("arguments") or entry.get("function", {}).get("arguments")
                    parsed_args = self._ensure_dict(arguments) or {}
                    calls.append(
                        ToolCall(
                            name=name or "",
                            arguments=parsed_args,
                            call_id=entry.get("call_id") or entry.get("id"),
                        )
                    )
                    raw_tool_calls.append(
                        {
                            "id": entry.get("call_id")
                            or entry.get("id")
                            or f"tool-{len(raw_tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": name or "",
                                "arguments": json.dumps(parsed_args),
                            },
                        }
                    )
            return ToolCallResult(
                calls=calls,
                message_content=message,
                raw_tool_calls=raw_tool_calls or None,
            )

        raise IntentRoutingError("Received unsupported response from intent model.")

    def _build_raw_tool_calls(self, calls: Sequence[ToolCall]) -> list[dict[str, Any]] | None:
        if not calls:
            return None
        payload: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            name = getattr(call, "name", "") or ""
            arguments = getattr(call, "arguments", {}) or {}
            if isinstance(arguments, str):
                try:
                    json.loads(arguments)
                    arguments_str = arguments
                except json.JSONDecodeError:
                    arguments_str = json.dumps({})
            else:
                try:
                    arguments_str = json.dumps(arguments)
                except (TypeError, ValueError):
                    arguments_str = json.dumps({})
            payload.append(
                {
                    "id": getattr(call, "call_id", None) or f"tool-{index}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_str,
                    },
                }
            )
        return payload

    def _parse_arguments(self, result: ToolCallResult, call: ToolCall) -> dict[str, Any]:
        candidates = [
            getattr(call, "arguments", None),
            getattr(result, "arguments", None),
            result.content,
        ]
        for candidate in candidates:
            parsed = self._ensure_dict(candidate)
            if parsed is not None:
                return parsed
        return {}

    @staticmethod
    def _ensure_dict(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return None
            if isinstance(decoded, dict):
                return decoded
        return None

    @staticmethod
    def _normalize_rationale(content: str | None) -> str | None:
        if not content:
            return None
        text = content.strip()
        return text or None

    @property
    def session_id(self) -> str:
        return self._session_id

    def _system_prompt(self) -> str:
        workspace = str(self.settings.workspace_root or ".")
        ctx = self._system_context
        available_tools = (
            ", ".join(ctx["available_tools"]) if ctx["available_tools"] else "none detected"
        )
        command_mappings = ", ".join(
            f"{key}={value}" for key, value in ctx["command_mappings"].items()
        )

        lines = [
            "You route developer requests for the DevAgent CLI.",
            "SYSTEM CONTEXT:",
            f"- Operating System: {ctx['os_friendly']} {ctx['os_version']} ({ctx['os']})",
            f"- Architecture: {ctx['architecture']}",
            f"- Python Version: {ctx['python_version']}",
            f"- Shell: {ctx['shell']} ({ctx['shell_type']} syntax)",
            f"- Working Directory: {ctx['cwd']}",
            f"- Home Directory: {ctx['home_dir']}",
            f"- Available Tools: {available_tools}",
            (
                "- Command Separator: '"
                f"{ctx['command_separator']}' | Path Separator: '{ctx['path_separator']}' | Null Device: {ctx['null_device']}"
            ),
            f"- Temp Directory: {ctx['temp_dir']}",
            f"- Platform Command Mappings: {command_mappings}",
            f"- Platform Examples: {ctx['platform_examples']}",
            "PROJECT CONTEXT:",
            *self._project_context_lines(workspace),
            "TOOL PERFORMANCE SIGNALS:",
            *self._tool_performance_lines(),
            "IMPORTANT:",
            f"- Use {'Unix' if ctx['is_unix'] else 'Windows'} command syntax and validate commands exist before invoking {RUN}.",
            f"- Never emit empty 'cmd' values for {RUN}; include all required arguments.",
            f"- Prefer registry tools over '{RUN}'; only run commands when no dedicated tool applies.",
            f"- Locate files with '{FIND}', '{GREP}', or '{SYMBOLS}' before using '{READ}'.",
            f"- Use '{READ}' with specific paths and optional 'context_lines' to keep outputs small.",
            f"- The repository root is at '{workspace}'. Return concise rationales when helpful.",
            "- Exploit tools with higher historical success before falling back to slower options.",
            "- When success rates are low or uncertain, capture rationale and propose safer alternatives.",
            "- Stop early if the user's request is fulfilled; otherwise, escalate with a structured plan.",
        ]
        return "\n".join(lines)

    def _project_context_lines(self, workspace: str) -> list[str]:
        """Build prompt lines with repository-specific context for better routing."""

        project = self.project_profile or {}
        if not project:
            return ["- Repository context not supplied; confirm assumptions when necessary."]

        lines: list[str] = []
        language = project.get("language") or project.get("dominant_language")
        if language:
            lines.append(f"- Dominant language: {language}")

        repo_size = project.get("repository_size") or project.get("file_count")
        if repo_size:
            lines.append(f"- Approximate file count: {repo_size}")

        plan_complexity = project.get("active_plan_complexity")
        if plan_complexity:
            lines.append(f"- Current plan complexity: {plan_complexity}")

        recent_files = project.get("recent_files") or []
        if isinstance(recent_files, list) and recent_files:
            preview = ", ".join(str(item) for item in recent_files[:4])
            if len(recent_files) > 4:
                preview += ", …"
            lines.append(f"- Recently touched files: {preview}")

        style_notes = project.get("style_notes")
        if style_notes:
            lines.append(f"- Style highlights: {style_notes}")

        summary = project.get("project_summary")
        if summary:
            flattened = " ".join(summary.split())
            lines.append(
                f"- Structure summary: {flattened[:200]}{'…' if len(flattened) > 200 else ''}"
            )

        workspace_hint = project.get("workspace_root")
        if workspace_hint and workspace_hint != workspace:
            lines.append(f"- Override workspace root: {workspace_hint}")

        return lines or ["- Repository context not supplied; confirm assumptions when necessary."]

    def _tool_performance_lines(self) -> list[str]:
        """Surface historical tool performance to steer selection."""

        history = self.tool_success_history or {}
        metrics: list[tuple[float, float, float, str]] = []

        for name, raw in history.items():
            if not isinstance(raw, dict):
                continue
            success = float(raw.get("success", 0))
            failure = float(raw.get("failure", 0))
            count = raw.get("count")
            if count is None:
                count = success + failure
            if count <= 0:
                continue
            success_rate = success / count
            avg_duration = float(
                raw.get("avg_duration", raw.get("total_duration", 0.0) / count if count else 0.0)
            )
            metrics.append((count, success_rate, avg_duration, name))

        if not metrics:
            return ["- No historical tool metrics captured; treat all tools as neutral."]

        metrics.sort(reverse=True)
        lines = []
        for count, success_rate, avg_duration, name in metrics[:4]:
            duration_text = f", avg {avg_duration:.1f}s" if avg_duration else ""
            lines.append(
                f"- {name}: {success_rate:.0%} success over {int(count)} runs{duration_text}"
            )

        if len(metrics) > 4:
            lines.append("- Additional tools tracked; use stored metrics when selecting fallbacks.")

        return lines

    def _augment_run_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        ctx = self._system_context
        updated = dict(schema)
        updated["description"] = (
            "Execute a shell command. "
            f"Platform: {ctx['os_friendly']} {ctx['os_version']} ({ctx['os']}). "
            f"Shell: {ctx['shell']} ({ctx['shell_type']} syntax). "
            f"Path separator: '{ctx['path_separator']}'. Command separator: '{ctx['command_separator']}'. "
            f"Examples: {ctx['platform_examples']}."
        )
        properties = dict(updated.get("properties") or {})
        cmd_schema = dict(properties.get("cmd") or {"type": "string"})
        cmd_schema["description"] = (
            "Primary command string. Never leave blank and ensure the binary exists for this platform. "
            f"Null device: {ctx['null_device']}."
        )
        properties["cmd"] = cmd_schema
        updated["properties"] = properties
        return updated

    def _augment_run_description(self, description: str) -> str:
        ctx = self._system_context
        base = description or "Execute a shell command."
        available_tools = (
            ", ".join(ctx["available_tools"]) if ctx["available_tools"] else "none detected"
        )
        return (
            f"{base} Platform: {ctx['os_friendly']} {ctx['os_version']} ({ctx['os']}). "
            f"Shell: {ctx['shell']} ({ctx['shell_type']} syntax). Available tools: {available_tools}. "
            f"Examples: {ctx['platform_examples']}."
        )

    # No keyword-based fallback routing


__all__ = ["DEFAULT_TOOLS", "IntentDecision", "IntentRouter", "IntentRoutingError"]
