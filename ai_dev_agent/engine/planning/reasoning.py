"""Reasoning helpers for multi-step task execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _now_ts() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.utcnow().strftime(ISO_FORMAT)


@dataclass
class ToolUse:
    """Describe a tool invocation that supports a reasoning step."""

    name: str
    command: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {"name": self.name}
        if self.command:
            payload["command"] = self.command
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass
class ReasoningStep:
    """Represent a single step in a multi-step reasoning trace."""

    identifier: str
    title: str
    detail: str
    status: str = "pending"
    result: str | None = None
    started_at: str = field(default_factory=_now_ts)
    completed_at: str | None = None
    tool: ToolUse | None = None

    def mark_in_progress(self) -> None:
        self.status = "in_progress"
        self.started_at = self.started_at or _now_ts()

    def complete(self, result: str | None = None) -> None:
        self.status = "completed"
        self.result = result
        self.completed_at = _now_ts()

    def fail(self, result: str | None = None) -> None:
        self.status = "failed"
        self.result = result
        self.completed_at = _now_ts()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.identifier,
            "title": self.title,
            "detail": self.detail,
            "status": self.status,
            "result": self.result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.tool:
            payload["tool"] = self.tool.to_dict()
        return payload


@dataclass
class PlanAdjustment:
    """Describe an on-the-fly adjustment proposed during execution."""

    summary: str
    detail: str
    created_at: str = field(default_factory=_now_ts)

    def to_dict(self) -> dict[str, str]:
        return {
            "summary": self.summary,
            "detail": self.detail,
            "created_at": self.created_at,
        }


@dataclass
class TaskReasoning:
    """Aggregate reasoning steps and plan adjustments for a task."""

    task_id: str
    goal: str | None = None
    task_title: str | None = None
    steps: list[ReasoningStep] = field(default_factory=list)
    adjustments: list[PlanAdjustment] = field(default_factory=list)

    def start_step(
        self,
        title: str,
        detail: str,
        tool: ToolUse | None = None,
    ) -> ReasoningStep:
        identifier = f"S{len(self.steps) + 1}"
        step = ReasoningStep(
            identifier=identifier,
            title=title,
            detail=detail,
            status="in_progress",
            tool=tool,
        )
        self.steps.append(step)
        return step

    def record_adjustment(self, summary: str, detail: str) -> PlanAdjustment:
        adjustment = PlanAdjustment(summary=summary, detail=detail)
        self.adjustments.append(adjustment)
        return adjustment

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "task_title": self.task_title,
            "steps": [step.to_dict() for step in self.steps],
            "adjustments": [adj.to_dict() for adj in self.adjustments],
        }

    def apply_to_task(self, task: dict[str, object]) -> None:
        task["reasoning_log"] = [step.to_dict() for step in self.steps]
        if self.adjustments:
            task["plan_adjustments"] = [adj.to_dict() for adj in self.adjustments]

    def merge_into_plan(self, plan: dict[str, object]) -> None:
        if self.adjustments:
            adjustments_obj = plan.setdefault("adjustments", [])
            # Ensure adjustments is a list (type narrowing)
            if not isinstance(adjustments_obj, list):
                adjustments_obj = []
                plan["adjustments"] = adjustments_obj
            adjustments = adjustments_obj

            seen = {
                (entry.get("summary"), entry.get("detail"))
                for entry in adjustments
                if isinstance(entry, dict)
            }
            for adj in self.adjustments:
                data = adj.to_dict()
                key = (data["summary"], data["detail"])
                if key not in seen:
                    adjustments.append(data)
                    seen.add(key)


__all__ = [
    "PlanAdjustment",
    "ReasoningStep",
    "TaskReasoning",
    "ToolUse",
]
