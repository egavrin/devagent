"""Shared data structures for the ReAct execution loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from collections.abc import Mapping


class ToolCall(BaseModel):
    """Single tool invocation within a batched request."""

    tool: str = Field(..., description="Registered tool identifier.")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the tool."
    )
    call_id: Optional[str] = Field(
        default=None, description="Unique identifier for this specific call."
    )

    model_config = ConfigDict(extra="allow")


class ToolResult(BaseModel):
    """Result from executing a single tool in a batch."""

    call_id: Optional[str] = Field(default=None, description="Identifier matching the tool call.")
    tool: str = Field(..., description="Tool that was executed.")
    success: bool = Field(..., description="Whether execution succeeded.")
    outcome: str = Field(default="", description="Result summary.")
    error: Optional[str] = Field(default=None, description="Error if failed.")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Per-call metrics.")
    wall_time: Optional[float] = Field(default=None, description="Execution time in seconds.")

    model_config = ConfigDict(extra="allow")


class ActionRequest(BaseModel):
    """Instruction for invoking a tool within the ReAct loop."""

    step_id: str
    thought: str = Field(..., description="Reasoning that led to the action.")
    tool: str = Field(..., description="Registered tool identifier (single-tool mode).")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the tool."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Auxiliary data for logging."
    )

    # Batch support
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Multiple tool calls for parallel execution."
    )

    model_config = ConfigDict(extra="allow")


class Observation(BaseModel):
    """Observation returned after executing a tool action."""

    success: bool
    outcome: str = Field(default="", description="Primary result summary.")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Structured metrics captured from the action."
    )
    artifacts: list[str] = Field(
        default_factory=list, description="Recorded artifact identifiers (paths, URIs)."
    )
    tool: Optional[str] = Field(default=None, description="Tool that produced the observation.")
    raw_output: Optional[str] = Field(
        default=None, description="Unstructured output for later inspection."
    )
    error: Optional[str] = Field(
        default=None, description="Error information if the action failed."
    )

    # Batch support
    results: list[ToolResult] = Field(
        default_factory=list, description="Per-call results for batched execution."
    )

    model_config = ConfigDict(extra="allow")


class CLIObservation(Observation):
    """Extended observation carrying UI-friendly metadata for the CLI."""

    formatted_output: Optional[str] = Field(
        default=None,
        description="Pre-formatted summary for immediate CLI display.",
    )
    artifact_path: Optional[str] = Field(
        default=None,
        description="Path to the full output artifact, if materialized.",
    )
    display_message: Optional[str] = Field(
        default=None,
        description="High-level message to show to the user.",
    )


class MetricsSnapshot(BaseModel):
    """Normalized metrics captured after each step."""

    tests_passed: Optional[bool] = None
    lint_errors: Optional[int] = None
    type_errors: Optional[int] = None
    format_errors: Optional[int] = None
    diff_lines: Optional[int] = None
    diff_files: Optional[int] = None
    diff_concentration: Optional[float] = None
    patch_coverage: Optional[float] = None
    secrets_found: Optional[int] = None
    sandbox_violations: Optional[int] = None
    flaky_tests: Optional[int] = None
    tokens_cost: Optional[float] = None
    tokens_used: Optional[float] = None
    time_elapsed: Optional[float] = None
    wall_time: Optional[float] = None
    compile_errors: Optional[int] = None
    gate_notes: dict[str, str] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_raw(cls, values: dict[str, Any]) -> dict[str, Any]:
        raw = dict(values)
        for key in list(values.keys()):
            if key in cls.model_fields:
                raw.pop(key, None)
        values.setdefault("raw", raw)
        return values


class GateConfig(BaseModel):
    """Gate thresholds, behavioural budgets, and required gate profile."""

    # Thresholds/budgets
    diff_limit_lines: int = 400
    diff_limit_files: int = 10
    patch_coverage_target: float = 0.8
    stuck_threshold: int = 3
    steps_budget: int = 25
    allow_flaky: bool = False

    # Required gates (profile toggles)
    require_tests: bool = True
    require_lint: bool = True
    require_types: bool = True
    require_format: bool = True
    require_compile: bool = True
    require_diff_limits: bool = True
    require_patch_coverage: bool = True
    require_secrets: bool = True
    require_sandbox: bool = True
    require_flaky: bool = True

    # Optional/extended gates
    require_design_doc: bool = False
    require_perf: bool = False

    @field_validator("patch_coverage_target")
    @classmethod
    def _clamp_patch_cov(cls, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    @field_validator("diff_limit_lines", "diff_limit_files", "stuck_threshold", "steps_budget")
    @classmethod
    def _ensure_positive(cls, value: int) -> int:
        return max(1, value)


class EvaluationResult(BaseModel):
    """Result of evaluating gate status after a step."""

    gates: dict[str, bool]
    required_gates: dict[str, bool] = Field(default_factory=dict)
    should_stop: bool
    stop_reason: Optional[str] = None
    next_action_hint: Optional[str] = None
    improved_metrics: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(
        "in_progress", description="Lifecycle indicator (in_progress|success|failure|blocked)"
    )


class StepRecord(BaseModel):
    """Record of a single ReAct step."""

    action: ActionRequest
    observation: Observation
    metrics: MetricsSnapshot
    evaluation: EvaluationResult
    step_index: int = 0


@dataclass
class TaskSpec:
    """Input description for executing a task with the ReAct loop."""

    identifier: str
    goal: str
    category: str = "implementation"
    instructions: str | None = None
    files: list[str] | None = None


class RunResult(BaseModel):
    """Final outcome for a ReAct run."""

    task_id: str
    status: str
    steps: list[StepRecord]
    gates: dict[str, bool]
    required_gates: dict[str, bool] = Field(default_factory=dict)
    stop_reason: Optional[str] = None
    runtime_seconds: Optional[float] = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    def gate_summary(self) -> Mapping[str, bool]:
        return self.gates
