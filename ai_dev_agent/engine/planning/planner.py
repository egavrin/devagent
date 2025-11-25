"""Planning module for generating work breakdown structures."""

from __future__ import annotations

import json
import re
import time
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from uuid import uuid4

from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.prompts.loader import PromptLoader
from ai_dev_agent.providers.llm import LLMClient, LLMConnectionError, LLMError, LLMTimeoutError
from ai_dev_agent.session import SessionManager, build_system_messages

LOGGER = get_logger(__name__)

_PROMPT_LOADER = PromptLoader()
SYSTEM_PROMPT = _PROMPT_LOADER.load_system_prompt("planner_system")
USER_TEMPLATE = _PROMPT_LOADER.load_system_prompt("planner_user")

JSON_PATTERN = re.compile(r"```json\s*(?P<json>{.*?})\s*```", re.DOTALL)


@dataclass
class PlanningContext:
    """Supplemental signals used to enrich planner prompts."""

    project_structure: str | None = None
    repository_metrics: str | None = None
    dominant_language: str | None = None
    dependency_landscape: str | None = None
    code_conventions: str | None = None
    quality_metrics: str | None = None
    historical_success: str | None = None
    recent_failures: str | None = None
    risk_register: str | None = None
    related_components: str | None = None

    def as_prompt_block(self) -> str:
        """Render a compact multi-section context block for the planner prompt."""

        sections: list[str] = []

        def _add(label: str, value: str | None) -> None:
            normalized = (value or "Not available").strip()
            sections.append(f"{label}:\n{normalized}")

        _add("Repository Metrics", self.repository_metrics)
        _add("Primary Language", self.dominant_language)
        _add("Dependency Landscape", self.dependency_landscape)
        _add("Code Conventions", self.code_conventions)
        _add("Quality & Coverage Signals", self.quality_metrics)
        _add("Historical Success Patterns", self.historical_success)
        _add("Recent Failures or Regressions", self.recent_failures)
        _add("Risk Register", self.risk_register)
        _add("Related Components or Modules", self.related_components)

        if self.project_structure:
            structure = self.project_structure.strip()
            sections.append(f"Project Structure Outline:\n{structure}")

        return "\n\n".join(sections)


@dataclass
class PlanTask:
    step_number: int | None = None
    title: str = "Untitled"
    description: str = ""
    status: str = "pending"
    dependencies: list[int] = field(default_factory=list)
    category: str = "implementation"
    effort: int | None = None
    reach: int | None = None
    impact: int | None = None
    confidence: float | None = None
    risk_mitigation: str | None = None
    deliverables: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    identifier: InitVar[str | None] = None

    _identifier: str = field(init=False)

    def __post_init__(self, identifier: str | None) -> None:
        candidate = (identifier or "").strip()
        if self.step_number is None:
            if candidate.startswith("T") and candidate[1:].isdigit():
                self.step_number = int(candidate[1:])
            else:
                self.step_number = 1
        if not candidate:
            candidate = f"T{self.step_number}"
        self._identifier = candidate
        self.identifier = candidate

        normalized_deps: list[int] = []
        for dep in self.dependencies:
            try:
                normalized_deps.append(int(dep))
            except (TypeError, ValueError):
                continue
        self.dependencies = normalized_deps

        self.deliverables = [str(item) for item in (self.deliverables or [])]
        self.commands = [str(item) for item in (self.commands or [])]

    def to_dict(self) -> dict:
        data = {
            "id": self._identifier,
            "step_number": self.step_number,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "dependencies": self.dependencies,
        }
        optional_fields = {
            "category": self.category,
            "effort": self.effort,
            "reach": self.reach,
            "impact": self.impact,
            "confidence": self.confidence,
            "deliverables": self.deliverables,
            "commands": self.commands,
            "risk_mitigation": self.risk_mitigation,
        }
        for key, value in optional_fields.items():
            if value not in (None, [], {}):
                data[key] = value
        return data


@dataclass
class PlanResult:
    goal: str
    summary: str
    tasks: list[PlanTask]
    raw_response: str
    fallback_reason: str | None = None
    project_structure: str | None = None
    context_snapshot: str | None = None
    complexity: str | None = None
    success_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = {
            "goal": self.goal,
            "summary": self.summary,
            "tasks": [task.to_dict() for task in self.tasks],
            "raw_response": self.raw_response,
            "status": "planned",
            "fallback_reason": self.fallback_reason,
        }
        if self.project_structure:
            data["project_structure"] = self.project_structure
        if self.context_snapshot:
            data["context_snapshot"] = self.context_snapshot
        if self.complexity:
            data["complexity"] = self.complexity
        if self.success_criteria:
            data["success_criteria"] = self.success_criteria
        return data


class Planner:
    """Generates structured plans using an LLM provider."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self._session_manager = SessionManager.get_instance()

    def generate(
        self,
        goal: str,
        project_structure: str | None = None,
        context: PlanningContext | None = None,
        *,
        session_id: str | None = None,
    ) -> PlanResult:
        LOGGER.debug("Requesting plan from LLM for goal: %s", goal)
        plan_context = context or PlanningContext()
        if plan_context.project_structure is None and project_structure:
            plan_context.project_structure = project_structure
        if plan_context.dominant_language is None:
            plan_context.dominant_language = "Unknown"

        context_block = plan_context.as_prompt_block()
        user_prompt = USER_TEMPLATE.format(
            goal=goal.strip(),
            context_block=context_block,
        )

        session_key = session_id or f"planner-{uuid4()}"
        system_messages = build_system_messages(
            include_react_guidance=False,
            extra_messages=[SYSTEM_PROMPT],
            workspace_root=Path.cwd(),
        )
        session = self._session_manager.ensure_session(
            session_key,
            system_messages=system_messages,
            metadata={
                "mode": "planner",
                "goal": goal,
                "context_snapshot": context_block,
            },
        )

        with session.lock:
            session.metadata["history_anchor"] = len(session.history)

        self._session_manager.add_user_message(session_key, user_prompt)
        start_time = time.time()
        next_heartbeat = start_time + 10.0
        try:
            while True:
                try:
                    conversation = self._session_manager.compose(session_key)
                    # Use default temperature (0.0) for reproducible planning
                    response_text = self.client.complete(conversation)
                    break
                except LLMTimeoutError:
                    now = time.time()
                    LOGGER.warning("Planner request timed out, retrying for goal: %s", goal)
                    if now >= next_heartbeat:
                        elapsed = now - start_time
                        LOGGER.info("Still waiting for planner response (%.1fs elapsed)", elapsed)
                        next_heartbeat = now + 10.0
                except LLMConnectionError as exc:
                    now = time.time()
                    LOGGER.warning("Planner connection issue (%s). Retrying goal: %s", exc, goal)
                    if now >= next_heartbeat:
                        elapsed = now - start_time
                        LOGGER.info("Still waiting for planner response (%.1fs elapsed)", elapsed)
                        next_heartbeat = now + 10.0
        except LLMError as exc:
            # Fail fast - no fallback
            LOGGER.error("LLM planning failed: %s", exc)
            self._session_manager.add_system_message(
                session_key,
                f"Planner failed: {exc}",
            )
            raise LLMError(f"Failed to generate plan: {exc}") from exc
        self._session_manager.add_assistant_message(session_key, response_text)
        try:
            payload = self._extract_json(response_text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Planner response was not valid JSON: {exc}") from exc
        tasks = [
            self._task_from_dict(entry, idx)
            for idx, entry in enumerate(payload.get("tasks", []), 1)
        ]
        summary = payload.get("summary", goal)
        complexity = payload.get("complexity")
        criteria_raw = payload.get("success_criteria") or []
        if isinstance(criteria_raw, (str, bytes)):
            success_criteria = [str(criteria_raw)]
        elif isinstance(criteria_raw, list):
            success_criteria = [str(item) for item in criteria_raw if item]
        else:
            success_criteria = []
        return PlanResult(
            goal=goal,
            summary=summary,
            tasks=tasks,
            raw_response=response_text,
            project_structure=project_structure,
            context_snapshot=context_block,
            complexity=str(complexity) if complexity else None,
            success_criteria=success_criteria,
        )

    def _extract_json(self, text: str) -> dict:
        match = JSON_PATTERN.search(text)
        if match:
            candidate = match.group("json")
        else:
            # Fallback: attempt to locate first JSON object in text
            start = text.find("{")
            end = text.rfind("}")
            candidate = text[start : end + 1] if start != -1 and end != -1 else "{}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse planner response as JSON: %s", exc)
            raise

    def _task_from_dict(self, data: dict, index: int) -> PlanTask:
        step_number = data.get("step_number", index)
        deliverables = data.get("deliverables") or []
        if isinstance(deliverables, (str, bytes)):
            deliverables = [str(deliverables)]
        commands = data.get("commands") or []
        if isinstance(commands, (str, bytes)):
            commands = [str(commands)]
        return PlanTask(
            step_number=step_number,
            title=data.get("title", "Untitled"),
            description=data.get("description", ""),
            status=data.get("status", "pending"),
            dependencies=_normalize_int_list(data.get("dependencies")),
            category=data.get("category", "implementation"),
            effort=data.get("effort"),
            reach=data.get("reach"),
            impact=data.get("impact"),
            confidence=data.get("confidence"),
            deliverables=[str(item) for item in deliverables],
            commands=[str(item) for item in commands],
            identifier=data.get("id") or data.get("identifier"),
            risk_mitigation=data.get("risk_mitigation"),
        )


def _normalize_int_list(value: object) -> list[int]:
    """Convert a value to a list of integers for dependencies."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        # Try to parse as integer
        try:
            return [int(value)]
        except ValueError:
            return []
    try:
        result = []
        for item in value:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, str):
                try:
                    result.append(int(item))
                except ValueError:
                    # Skip non-integer strings
                    pass
        return result
    except (TypeError, ValueError):
        return []


__all__ = ["PlanResult", "PlanTask", "Planner"]
