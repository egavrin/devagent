"""Iteration and tool-budget helpers for the CLI ReAct executor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ai_dev_agent.tools import READ, RUN, WRITE

_DEFAULT_THRESHOLD_EXPLORATION = 30.0
_DEFAULT_THRESHOLD_INVESTIGATION = 60.0
_DEFAULT_THRESHOLD_CONSOLIDATION = 85.0


PHASE_PROMPTS: Dict[str, str] = {
    "exploration": (
        "You are beginning your investigation.\n"
        "Cast a wide net, explore the codebase structure, gather context, identify key components.\n\n"
        "💡 PARALLEL TOOL EXECUTION: You can call multiple tools in a single response.\n"
        "When operations are independent, use parallel execution for efficiency:\n"
        "  • Reading multiple files: call read() multiple times\n"
        "  • Searching different patterns: call grep() multiple times\n"
        "  • Finding various file types: call find() multiple times\n"
        "Example: To analyze 3 files, make 3 read() calls in one response."
    ),
    "investigation": (
        "You are investigating specific areas.\n"
        "Focus on the most promising leads, validate hypotheses, dive deeper.\n\n"
        "💡 Use parallel tool calls when examining multiple independent sources.\n"
        "Combine multiple read(), grep(), or find() calls in a single response."
    ),
    "consolidation": (
        "You are consolidating discoveries.\n"
        "Connect findings, validate conclusions, prepare to formulate your answer.\n\n"
        "💡 For final validations, you can still use parallel tool calls to verify multiple items at once."
    ),
    "preparation": (
        "⚠️ IMPORTANT: You are nearing completion.\n"
        "Focus only on essential validations. Begin drafting your comprehensive answer.\n\n"
        "💡 Last chance for parallel tool calls if you need to verify multiple items quickly."
    ),
    "final_warning": (
        "⚡ CRITICAL: After this response, you must provide final synthesis.\n"
        "Complete any essential work NOW. Your next response will be text-only."
    ),
    "synthesis": (
        "📋 SYNTHESIS REQUIRED\n"
        "Based on your investigation, provide your complete answer.\n"
        "Include: findings, file locations, recommendations, unknowns.\n"
        "This is your final response."
    ),
}


@dataclass(frozen=True)
class IterationContext:
    """Immutable snapshot for a single loop iteration."""

    number: int
    total: int
    remaining: int
    percent_complete: float
    phase: str
    is_final: bool
    is_penultimate: bool
    reflection_count: int = 0
    reflection_allowed: bool = False


class BudgetManager:
    """Track iteration state and determine behavioural phases.

    Enhanced with adaptive budget allocation based on model capacity.
    """

    def __init__(
        self,
        max_iterations: int,
        *,
        phase_thresholds: Mapping[str, Any] | None = None,
        warnings: Mapping[str, Any] | None = None,
        model_context_window: int | None = None,
        adaptive_scaling: bool = True,
    ) -> None:
        self.max_iterations = max(1, int(max_iterations or 1))
        self._current = 0
        self.model_context_window = model_context_window or 100000
        self.adaptive_scaling = adaptive_scaling

        thresholds = phase_thresholds or {}

        # Base thresholds
        base_exploration = _DEFAULT_THRESHOLD_EXPLORATION
        base_investigation = _DEFAULT_THRESHOLD_INVESTIGATION
        base_consolidation = _DEFAULT_THRESHOLD_CONSOLIDATION

        # Apply adaptive scaling if enabled
        if adaptive_scaling and model_context_window:
            # Scale phases based on context window size
            # Larger models can explore more, smaller models need to focus faster
            scaling_factor = min(model_context_window / 100000, 2.0)
            scaling_factor = max(scaling_factor, 0.5)  # Clamp between 0.5x and 2x

            base_exploration *= scaling_factor
            base_investigation = base_exploration + (base_investigation - _DEFAULT_THRESHOLD_EXPLORATION)
            base_consolidation = base_investigation + (base_consolidation - _DEFAULT_THRESHOLD_INVESTIGATION)

        self._exploration_end = _coerce_percentage(
            thresholds.get("exploration_end"),
            fallback=base_exploration,
        )
        self._investigation_end = _coerce_percentage(
            thresholds.get("investigation_end"),
            fallback=base_investigation,
        )
        self._consolidation_end = _coerce_percentage(
            thresholds.get("consolidation_end"),
            fallback=base_consolidation,
        )

        warn_cfg = warnings or {}
        self._warn_before_final = bool(warn_cfg.get("warn_before_final", True))
        self._final_warning_iterations = max(
            int(warn_cfg.get("final_warning_iterations", 1) or 0),
            0,
        )

    @property
    def current(self) -> int:
        return self._current

    def next_iteration(self) -> IterationContext | None:
        """Advance to the next iteration if within budget."""

        if self._current >= self.max_iterations:
            return None

        self._current += 1
        remaining = max(self.max_iterations - self._current, 0)
        percent_complete = (self._current / self.max_iterations) * 100

        is_final = remaining == 0
        is_penultimate = remaining == 1

        phase = self._determine_phase(percent_complete, remaining, is_final)

        return IterationContext(
            number=self._current,
            total=self.max_iterations,
            remaining=remaining,
            percent_complete=percent_complete,
            phase=phase,
            is_final=is_final,
            is_penultimate=is_penultimate,
        )

    def _determine_phase(
        self,
        percent_complete: float,
        remaining: int,
        is_final: bool,
    ) -> str:
        if is_final:
            return "synthesis"

        if (
            self._warn_before_final
            and self._final_warning_iterations > 0
            and remaining <= self._final_warning_iterations
        ):
            return "final_warning"

        if percent_complete < self._exploration_end:
            return "exploration"
        if percent_complete < self._investigation_end:
            return "investigation"
        if percent_complete < self._consolidation_end:
            return "consolidation"
        return "preparation"


def get_tools_for_iteration(
    context: IterationContext,
    available_tools: Sequence[Dict[str, Any]],
    *,
    tool_config: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Select tools based on iteration context and optional config."""

    if context.is_final:
        return [create_text_only_tool()]

    config = tool_config or {}

    if context.is_penultimate and config.get("essential_only_in_final", True):
        limited = filter_essential_tools(available_tools)
        if limited:
            return limited

    if context.phase == "preparation" and config.get("limit_in_preparation", True):
        limited = filter_non_exploratory_tools(available_tools)
        if limited:
            return limited

    if config.get("remove_exploratory_late", True) and context.phase in {"preparation", "final_warning"}:
        limited = filter_non_exploratory_tools(available_tools)
        if limited:
            return limited

    return list(available_tools)


def filter_non_exploratory_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exploratory_patterns = ("search", "plan", "outline", "index", "crawl", "explore", "symbol")
    filtered: List[Dict[str, Any]] = []
    for tool in tools:
        name = _extract_tool_name(tool)
        if not name:
            filtered.append(tool)
            continue
        lowered = name.lower()
        if any(pattern in lowered for pattern in exploratory_patterns):
            continue
        filtered.append(tool)
    return filtered


def filter_essential_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    essential_names = {
        READ,
        "fs_read_text",
        "fs.list",
        RUN,
        "exec.shell",
        "plan.summarize",
        "summarize",
    }
    selected: List[Dict[str, Any]] = []
    for tool in tools:
        name = _extract_tool_name(tool)
        if name and name.lower() in essential_names:
            selected.append(tool)
    return selected or list(tools)


def create_text_only_tool() -> Dict[str, Any]:
    """Return a tool schema that only accepts the final answer text."""

    return {
        "type": "function",
        "function": {
            "name": "submit_final_answer",
            "description": "Submit your complete final answer as plain text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your complete analysis and findings.",
                    }
                },
                "required": ["answer"],
            },
        },
    }


def extract_text_content(result: Any) -> str:
    """Pull any available text content from a tool invocation result."""

    text = getattr(result, "message_content", None)
    if isinstance(text, str):
        text = text.strip()
    else:
        text = ""
    return text or ""


def auto_generate_summary(
    conversation: Sequence[Any],
    *,
    files_examined: Iterable[str] | None = None,
    searches_performed: Iterable[str] | None = None,
) -> str:
    """Generate a comprehensive summary when the LLM doesn't provide submit_final_answer.

    This fallback should provide useful information about what was investigated,
    not just say "no answer found".
    """

    files = sorted({str(item) for item in files_examined or [] if item})
    searches = sorted({str(item) for item in searches_performed or [] if item})

    # Extract ALL substantive assistant messages (observations, findings, etc.)
    observations: List[str] = []
    for message in reversed(conversation):
        role = getattr(message, "role", "")
        content = getattr(message, "content", None)
        if role == "assistant" and isinstance(content, str):
            text = content.strip()
            # Skip empty, context summaries, and very short messages
            if not text or text.startswith('[') or len(text) < 15:
                continue
            # Skip pure action descriptions
            if text.lower().startswith(('let me', 'i will', 'i\'ll', 'searching for', 'looking for')):
                continue
            # Include messages ending with : if they're long enough (might be explanations)
            if text.endswith(':') and len(text) < 40:
                continue
            observations.append(text)

    # Build comprehensive summary
    summary_parts: List[str] = []

    summary_parts.append("Based on the investigation:")
    summary_parts.append("")

    # Include key observations
    if observations:
        # Take up to 3 most recent substantial observations
        for i, obs in enumerate(observations[:3]):
            # Truncate if very long
            if len(obs) > 300:
                obs = obs[:297] + "..."
            summary_parts.append(f"• {obs}")
        summary_parts.append("")

    # Add investigation scope
    if files or searches:
        summary_parts.append(f"Investigation scope: {len(files)} file(s) examined, {len(searches)} search(es) performed.")

    # If no observations, at least list what was checked
    if not observations and files:
        summary_parts.append("")
        summary_parts.append("Files examined:")
        for f in files[:10]:
            summary_parts.append(f"• {f}")
        if len(files) > 10:
            summary_parts.append(f"• ... and {len(files) - 10} more")

    return "\n".join(summary_parts)


def combine_partial_responses(*parts: str) -> str:
    """Merge partial text fragments into a cohesive response."""

    combined: List[str] = []
    for part in parts:
        if not part:
            continue
        stripped = part.strip()
        if not stripped:
            continue
        combined.append(stripped)
    return "\n\n".join(combined)


def _extract_tool_name(tool: Mapping[str, Any]) -> str | None:
    if not isinstance(tool, Mapping):
        return None
    name = tool.get("name")
    if isinstance(name, str) and name:
        return name
    function = tool.get("function")
    if isinstance(function, Mapping):
        value = function.get("name")
        if isinstance(value, str) and value:
            return value
    return None


def _coerce_percentage(value: Any, *, fallback: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return fallback
    if numeric <= 0:
        return fallback
    if numeric >= 100:
        return 99.9
    return numeric


@dataclass
class ReflectionContext:
    """Context for reflection mechanism."""

    max_reflections: int = 3
    current_reflection: int = 0
    last_error: str | None = None
    enabled: bool = True


class AdaptiveBudgetManager(BudgetManager):
    """Enhanced budget manager with reflection and dynamic adjustment.

    Key features:
    - Reflection mechanism for error recovery
    - Dynamic phase adjustment based on progress
    """

    def __init__(
        self,
        max_iterations: int,
        *,
        phase_thresholds: Mapping[str, Any] | None = None,
        warnings: Mapping[str, Any] | None = None,
        model_context_window: int | None = None,
        adaptive_scaling: bool = True,
        enable_reflection: bool = True,
        max_reflections: int = 3,
    ) -> None:
        super().__init__(
            max_iterations,
            phase_thresholds=phase_thresholds,
            warnings=warnings,
            model_context_window=model_context_window,
            adaptive_scaling=adaptive_scaling,
        )
        self.reflection = ReflectionContext(
            enabled=enable_reflection,
            max_reflections=max_reflections,
        )
        self._success_count = 0
        self._failure_count = 0
        self._phase_adjustments = {"exploration": 0, "investigation": 0, "consolidation": 0}

    def next_iteration(self) -> IterationContext | None:
        """Advance to next iteration with reflection support."""
        context = super().next_iteration()
        if context is None:
            return None

        # Add reflection information
        return IterationContext(
            number=context.number,
            total=context.total,
            remaining=context.remaining,
            percent_complete=context.percent_complete,
            phase=context.phase,
            is_final=context.is_final,
            is_penultimate=context.is_penultimate,
            reflection_count=self.reflection.current_reflection,
            reflection_allowed=self._can_reflect(),
        )

    def allow_reflection(self, error_msg: str) -> bool:
        """Check if reflection is allowed for error recovery.

        Args:
            error_msg: Error message that triggered reflection

        Returns:
            True if reflection allowed, False otherwise
        """
        if not self.reflection.enabled:
            return False

        if self.reflection.current_reflection >= self.reflection.max_reflections:
            return False

        self.reflection.current_reflection += 1
        self.reflection.last_error = error_msg
        self._failure_count += 1
        return True

    def reset_reflection(self):
        """Reset reflection counter after successful operation."""
        self.reflection.current_reflection = 0
        self.reflection.last_error = None
        self._success_count += 1
        self._failure_count = 0

    def _can_reflect(self) -> bool:
        """Check if reflection is currently possible."""
        return (
            self.reflection.enabled
            and self.reflection.current_reflection < self.reflection.max_reflections
        )

    def adjust_phases_for_progress(self, success_rate: float):
        """Dynamically adjust phase thresholds based on progress.

        Args:
            success_rate: Recent success rate (0.0 to 1.0)
        """
        if not self.adaptive_scaling:
            return

        current_phase = self._determine_phase(
            self.percent_complete,
            self.max_iterations - self._current,
            False,
        )

        # If making good progress in exploration, extend it
        if success_rate > 0.8 and current_phase == "exploration":
            adjustment = min(10.0, 100.0 - self._exploration_end)
            self._exploration_end += adjustment
            self._investigation_end += adjustment
            self._phase_adjustments["exploration"] += adjustment

        # If struggling in investigation, move to consolidation early
        elif success_rate < 0.3 and current_phase == "investigation":
            adjustment = min(10.0, self._consolidation_end - self._investigation_end)
            self._consolidation_end -= adjustment
            self._phase_adjustments["investigation"] -= adjustment

    @property
    def percent_complete(self) -> float:
        """Get current completion percentage."""
        return (self._current / self.max_iterations) * 100 if self.max_iterations > 0 else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get budget manager statistics."""
        return {
            "current_iteration": self._current,
            "max_iterations": self.max_iterations,
            "percent_complete": self.percent_complete,
            "reflection_count": self.reflection.current_reflection,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "phase_adjustments": self._phase_adjustments.copy(),
            "current_thresholds": {
                "exploration_end": self._exploration_end,
                "investigation_end": self._investigation_end,
                "consolidation_end": self._consolidation_end,
            },
        }


__all__ = [
    "PHASE_PROMPTS",
    "IterationContext",
    "BudgetManager",
    "AdaptiveBudgetManager",
    "ReflectionContext",
    "get_tools_for_iteration",
    "filter_non_exploratory_tools",
    "filter_essential_tools",
    "create_text_only_tool",
    "extract_text_content",
    "auto_generate_summary",
    "combine_partial_responses",
]
