"""Natural-language intent routing leveraging LLM tool calling."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ai_dev_agent.agents import AgentRegistry, AgentSpec
from ai_dev_agent.cli.utils import build_system_context
from ai_dev_agent.core.utils.config import Settings
# Tool metadata is registered directly under canonical names
from ai_dev_agent.providers.llm.base import LLMClient, LLMError, ToolCallResult
from ai_dev_agent.tools import (
    registry as tool_registry,
    READ,
    WRITE,
    RUN,
    FIND,
    GREP,
    SYMBOLS,
)
from ai_dev_agent.session import SessionManager, build_system_messages

DEFAULT_TOOLS: List[Dict[str, Any]] = []


@dataclass
class IntentDecision:
    """Result returned by the intent router."""

    tool: Optional[str]
    arguments: Dict[str, Any]
    rationale: Optional[str] = None


class IntentRoutingError(RuntimeError):
    """Raised when the router cannot derive a suitable intent."""


class IntentRouter:
    """Use LLM tool-calling to map natural language prompts onto CLI intents."""

    def __init__(
        self,
        client: Optional[LLMClient],
        settings: Settings,
        agent_type: str = "manager",
        tools: Optional[List[Dict[str, Any]]] = None,
        project_profile: Optional[Dict[str, Any]] = None,
        tool_success_history: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.agent_spec = AgentRegistry.get(agent_type)
        self._system_context = build_system_context()
        self.tools = tools or self._build_tool_list(settings, self.agent_spec)
        self.project_profile = project_profile or {}
        self.tool_success_history = tool_success_history or {}
        self._session_manager = SessionManager.get_instance()
        self._session_id = f"router-{uuid4()}"

    def _build_tool_list(self, settings: Settings, agent_spec: AgentSpec) -> List[Dict[str, Any]]:
        """Combine core tools with selected registry tools, avoiding duplicates."""
        combined: List[Dict[str, Any]] = []
        used_names: set[str] = set()
        for entry in DEFAULT_TOOLS:
            fn = entry.get("function", {})
            name = fn.get("name")
            if name:
                used_names.add(name)
            combined.append(entry)

        combined.extend(self._build_registry_tools(settings, agent_spec, used_names))
        return combined

    def _build_registry_tools(self, settings: Settings, agent_spec: AgentSpec, used_names: set[str]) -> List[Dict[str, Any]]:
        """Translate registry specs into LLM tool definitions filtered by agent's allowed tools."""
        # Full safelist of all available tools
        all_tools = [FIND, GREP, SYMBOLS, READ, RUN, WRITE]

        # Filter to only include tools allowed for this agent
        safelist = [name for name in all_tools if name in agent_spec.tools]

        tools: List[Dict[str, Any]] = []
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
            if name == WRITE and not getattr(settings, "auto_approve_code", False):
                description = (description or "Apply a unified diff") + \
                              " (auto_approve_code is recommended for automated edits)"

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

    def route(self, prompt: str) -> IntentDecision:
        if not prompt.strip():
            raise IntentRoutingError("Empty prompt provided for intent routing.")

        # Require an LLM client; no keyword-based fallback
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
        session = self._session_manager.ensure_session(
            self._session_id,
            system_messages=system_messages,
            metadata={
                "mode": "intent-router",
                "project_profile": self.project_profile,
            },
        )
        with session.lock:
            session.metadata["last_prompt"] = prompt.strip()

        self._session_manager.add_user_message(self._session_id, prompt.strip())

        try:
            result: ToolCallResult = self.client.invoke_tools(
                self._session_manager.compose(self._session_id),
                tools=self.tools,
                temperature=0.1,
            )
        except LLMError as exc:
            self._session_manager.add_system_message(
                self._session_id,
                f"Intent routing error: {exc}",
            )
            raise IntentRoutingError(f"LLM tool call failed: {exc}") from exc
        except Exception as exc:
            self._session_manager.add_system_message(
                self._session_id,
                f"Intent routing failure: {exc}",
            )
            raise IntentRoutingError(f"Intent routing failed: {exc}") from exc

        if result.calls:
            call = result.calls[0]
            self._session_manager.add_assistant_message(
                self._session_id,
                result.message_content,
                tool_calls=result.raw_tool_calls,
            )
            return IntentDecision(
                tool=call.name,
                arguments=call.arguments,
                rationale=(result.message_content or "").strip() or None,
            )

        if result.message_content:
            self._session_manager.add_assistant_message(
                self._session_id,
                result.message_content,
            )
            # No tool used; surface the response directly without invoking a handler.
            return IntentDecision(
                tool=None,
                arguments={"text": result.message_content.strip()},
            )

        raise IntentRoutingError("Model response did not include a tool call or content.")

    @property
    def session_id(self) -> str:
        return self._session_id

    def _system_prompt(self) -> str:
        workspace = str(self.settings.workspace_root or ".")
        ctx = self._system_context
        available_tools = ", ".join(ctx["available_tools"]) if ctx["available_tools"] else "none detected"
        command_mappings = ", ".join(f"{key}={value}" for key, value in ctx["command_mappings"].items())

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

    def _project_context_lines(self, workspace: str) -> List[str]:
        """Build prompt lines with repository-specific context for better routing."""

        project = self.project_profile or {}
        if not project:
            return ["- Repository context not supplied; confirm assumptions when necessary."]

        lines: List[str] = []
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
            lines.append(f"- Structure summary: {flattened[:200]}{'…' if len(flattened) > 200 else ''}")

        workspace_hint = project.get("workspace_root")
        if workspace_hint and workspace_hint != workspace:
            lines.append(f"- Override workspace root: {workspace_hint}")

        return lines or ["- Repository context not supplied; confirm assumptions when necessary."]

    def _tool_performance_lines(self) -> List[str]:
        """Surface historical tool performance to steer selection."""

        history = self.tool_success_history or {}
        metrics: List[tuple[float, float, float, str]] = []

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
            avg_duration = float(raw.get("avg_duration", raw.get("total_duration", 0.0) / count if count else 0.0))
            metrics.append((count, success_rate, avg_duration, name))

        if not metrics:
            return ["- No historical tool metrics captured; treat all tools as neutral."]

        metrics.sort(reverse=True)
        lines = []
        for count, success_rate, avg_duration, name in metrics[:4]:
            duration_text = f", avg {avg_duration:.1f}s" if avg_duration else ""
            lines.append(f"- {name}: {success_rate:.0%} success over {int(count)} runs{duration_text}")

        if len(metrics) > 4:
            lines.append("- Additional tools tracked; use stored metrics when selecting fallbacks.")

        return lines

    def _augment_run_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
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
        available_tools = ", ".join(ctx["available_tools"]) if ctx["available_tools"] else "none detected"
        return (
            f"{base} Platform: {ctx['os_friendly']} {ctx['os_version']} ({ctx['os']}). "
            f"Shell: {ctx['shell']} ({ctx['shell_type']} syntax). Available tools: {available_tools}. "
            f"Examples: {ctx['platform_examples']}."
        )

    # No keyword-based fallback routing


__all__ = ["IntentDecision", "IntentRouter", "IntentRoutingError", "DEFAULT_TOOLS"]
