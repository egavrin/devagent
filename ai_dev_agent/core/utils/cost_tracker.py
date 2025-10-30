"""Token usage and cost tracking for LLM interactions.

Provides granular accounting with micro-cent precision to track input/output/cache
tokens and calculate costs based on model pricing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Model pricing per million tokens (in dollars)
# Based on current pricing as of 2025
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic models
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-3-5-haiku-20241022": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "claude-3-sonnet-20240229": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25,
        "cache_write": 0.30,
        "cache_read": 0.03,
    },
    # DeepSeek models
    "deepseek-coder": {"input": 0.14, "output": 0.28, "cache_write": 0.14, "cache_read": 0.014},
    "deepseek-chat": {"input": 0.14, "output": 0.28, "cache_write": 0.14, "cache_read": 0.014},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19, "reasoning": 0.55},
    # Default pricing for unknown models
    "default": {"input": 1.00, "output": 2.00},
}


@dataclass
class TokenUsage:
    """Token usage for a single LLM request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0  # For models like o1 or DeepSeek reasoner
    total_tokens: int = 0

    def __post_init__(self):
        """Calculate total if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens + self.reasoning_tokens


@dataclass
class CostRecord:
    """Cost record for a single LLM request."""

    timestamp: float
    model: str
    usage: TokenUsage
    cost_usd: float
    operation: str = ""  # e.g., "completion", "tool_call", "summarization"
    iteration: int | None = None
    phase: str | None = None


@dataclass
class CostTracker:
    """Track token usage and costs across an entire session.

    Key features:
    - Granular per-request tracking
    - Cache-aware pricing (25% premium write, 90% discount read)
    - Model-specific pricing
    - Micro-cent precision (8 decimal places)
    - Session aggregation
    """

    records: list[CostRecord] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_reasoning_tokens: int = 0

    # Per-model breakdown
    model_costs: dict[str, float] = field(default_factory=dict)
    model_tokens: dict[str, TokenUsage] = field(default_factory=dict)

    # Per-phase breakdown (exploration, investigation, etc.)
    phase_costs: dict[str, float] = field(default_factory=dict)
    phase_tokens: dict[str, TokenUsage] = field(default_factory=dict)

    def track_request(
        self,
        model: str,
        usage: TokenUsage,
        operation: str = "",
        iteration: int | None = None,
        phase: str | None = None,
    ) -> CostRecord:
        """Track a single LLM request and calculate its cost.

        Args:
            model: Model name/identifier
            usage: Token usage for this request
            operation: Type of operation (completion, tool_call, etc.)
            iteration: Current iteration number if applicable
            phase: Current phase (exploration, investigation, etc.)

        Returns:
            CostRecord with calculated cost
        """
        # Calculate cost
        cost_usd = self._calculate_cost(model, usage)

        # Create record
        record = CostRecord(
            timestamp=time.time(),
            model=model,
            usage=usage,
            cost_usd=cost_usd,
            operation=operation,
            iteration=iteration,
            phase=phase,
        )

        # Update totals
        self.records.append(record)
        self.total_cost_usd += cost_usd
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_cache_read_tokens += usage.cache_read_tokens
        self.total_cache_write_tokens += usage.cache_write_tokens
        self.total_reasoning_tokens += usage.reasoning_tokens

        # Update per-model stats
        if model not in self.model_costs:
            self.model_costs[model] = 0.0
            self.model_tokens[model] = TokenUsage()
        self.model_costs[model] += cost_usd
        model_usage = self.model_tokens[model]
        model_usage.prompt_tokens += usage.prompt_tokens
        model_usage.completion_tokens += usage.completion_tokens
        model_usage.cache_read_tokens += usage.cache_read_tokens
        model_usage.cache_write_tokens += usage.cache_write_tokens
        model_usage.reasoning_tokens += usage.reasoning_tokens
        model_usage.total_tokens += usage.total_tokens

        # Update per-phase stats
        if phase:
            if phase not in self.phase_costs:
                self.phase_costs[phase] = 0.0
                self.phase_tokens[phase] = TokenUsage()
            self.phase_costs[phase] += cost_usd
            phase_usage = self.phase_tokens[phase]
            phase_usage.prompt_tokens += usage.prompt_tokens
            phase_usage.completion_tokens += usage.completion_tokens
            phase_usage.cache_read_tokens += usage.cache_read_tokens
            phase_usage.cache_write_tokens += usage.cache_write_tokens
            phase_usage.reasoning_tokens += usage.reasoning_tokens
            phase_usage.total_tokens += usage.total_tokens

        return record

    def _calculate_cost(self, model: str, usage: TokenUsage) -> float:
        """Calculate cost in USD for given token usage.

        Uses micro-cent precision (8 decimal places) to avoid rounding errors.
        """
        # Get model pricing or use default
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

        # Calculate base costs (price per million tokens)
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing.get("input", 1.0)
        output_cost = (usage.completion_tokens / 1_000_000) * pricing.get("output", 2.0)

        # Cache costs (Anthropic/DeepSeek style)
        cache_write_cost = 0.0
        cache_read_cost = 0.0
        if "cache_write" in pricing and usage.cache_write_tokens > 0:
            # Cache writes cost 25% more than regular input (Anthropic)
            cache_write_cost = (usage.cache_write_tokens / 1_000_000) * pricing["cache_write"]
        if "cache_read" in pricing and usage.cache_read_tokens > 0:
            # Cache reads cost 90% less than regular input (Anthropic)
            cache_read_cost = (usage.cache_read_tokens / 1_000_000) * pricing["cache_read"]

        # Reasoning token costs (for o1, DeepSeek reasoner)
        reasoning_cost = 0.0
        if "reasoning" in pricing and usage.reasoning_tokens > 0:
            reasoning_cost = (usage.reasoning_tokens / 1_000_000) * pricing["reasoning"]

        # Total with micro-cent precision
        total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost + reasoning_cost

        # Round to 8 decimal places for micro-cent precision
        return round(total_cost, 8)

    def get_average_tokens_per_iteration(self) -> tuple[int, int]:
        """Get average input/output tokens per iteration.

        Returns:
            Tuple of (avg_input_tokens, avg_output_tokens)
        """
        iterations_with_usage = [r for r in self.records if r.iteration is not None]
        if not iterations_with_usage:
            return (0, 0)

        # Filter out None values before max()
        iteration_values = [r.iteration for r in iterations_with_usage if r.iteration is not None]
        if not iteration_values:
            return (0, 0)

        num_iterations = max(iteration_values)
        if num_iterations == 0:
            return (0, 0)

        avg_input = self.total_prompt_tokens // num_iterations
        avg_output = self.total_completion_tokens // num_iterations
        return (avg_input, avg_output)

    def format_summary(self, detailed: bool = False) -> str:
        """Format a cost summary for display.

        Args:
            detailed: If True, include per-model and per-phase breakdowns

        Returns:
            Formatted string summary
        """
        lines = []

        # Overall summary
        lines.append(f"💰 Total Cost: ${self.total_cost_usd:.4f}")
        lines.append(
            f"📊 Tokens: {self.total_prompt_tokens:,} in, " f"{self.total_completion_tokens:,} out"
        )

        if self.total_cache_read_tokens > 0 or self.total_cache_write_tokens > 0:
            lines.append(
                f"💾 Cache: {self.total_cache_read_tokens:,} read "
                f"(${self._calculate_cache_savings():.4f} saved), "
                f"{self.total_cache_write_tokens:,} write"
            )

        if self.total_reasoning_tokens > 0:
            lines.append(f"🧠 Reasoning: {self.total_reasoning_tokens:,} tokens")

        if detailed:
            # Per-model breakdown
            if len(self.model_costs) > 1:
                lines.append("\n📈 Per Model:")
                for model, cost in sorted(self.model_costs.items(), key=lambda x: -x[1]):
                    usage = self.model_tokens[model]
                    lines.append(f"  • {model}: ${cost:.4f} " f"({usage.total_tokens:,} tokens)")

            # Per-phase breakdown
            if self.phase_costs:
                lines.append("\n📋 Per Phase:")
                for phase, cost in self.phase_costs.items():
                    usage = self.phase_tokens[phase]
                    percentage = (
                        (cost / self.total_cost_usd * 100) if self.total_cost_usd > 0 else 0
                    )
                    lines.append(
                        f"  • {phase}: ${cost:.4f} ({percentage:.1f}%) "
                        f"- {usage.total_tokens:,} tokens"
                    )

        return "\n".join(lines)

    def format_inline(self) -> str:
        """Format a compact inline cost display."""
        cache_info = ""
        if self.total_cache_read_tokens > 0:
            savings = self._calculate_cache_savings()
            cache_info = f" | Cache saved: ${savings:.4f}"

        return (
            f"Cost: ${self.total_cost_usd:.4f} | "
            f"Tokens: {self.total_prompt_tokens:,}→{self.total_completion_tokens:,}"
            f"{cache_info}"
        )

    def _calculate_cache_savings(self) -> float:
        """Calculate how much was saved by cache hits."""
        # Estimate savings (cache reads cost ~90% less than regular input)
        # This is a rough estimate since we don't track which model each cache hit was for
        avg_input_price = 3.0  # Average across common models
        regular_cost = (self.total_cache_read_tokens / 1_000_000) * avg_input_price
        actual_cost = (self.total_cache_read_tokens / 1_000_000) * (avg_input_price * 0.1)
        return round(regular_cost - actual_cost, 8)

    def get_forecast(self, remaining_iterations: int) -> float:
        """Forecast remaining cost based on current usage patterns.

        Args:
            remaining_iterations: Number of iterations remaining

        Returns:
            Estimated remaining cost in USD
        """
        if not self.records or remaining_iterations <= 0:
            return 0.0

        # Get average cost per iteration
        iterations_with_cost = [r for r in self.records if r.iteration is not None]
        if not iterations_with_cost:
            # Fall back to simple average
            avg_cost_per_request = self.total_cost_usd / len(self.records)
            # Assume ~5 requests per iteration
            return round(avg_cost_per_request * 5 * remaining_iterations, 4)

        # Calculate average cost per iteration
        iteration_values = [r.iteration for r in iterations_with_cost if r.iteration is not None]
        if not iteration_values:
            return 0.0

        max_iteration = max(iteration_values)
        if max_iteration == 0:
            return 0.0

        avg_cost_per_iteration = self.total_cost_usd / max_iteration
        return round(avg_cost_per_iteration * remaining_iterations, 4)

    def should_warn(self, threshold_usd: float) -> bool:
        """Check if cost has exceeded warning threshold.

        Args:
            threshold_usd: Warning threshold in USD

        Returns:
            True if current cost exceeds threshold
        """
        return self.total_cost_usd >= threshold_usd


def create_token_usage_from_response(response_data: dict) -> TokenUsage:
    """Create TokenUsage from LLM response data.

    Handles various response formats from different providers.
    """
    usage = TokenUsage()

    # Standard OpenAI format
    if "usage" in response_data:
        usage_data = response_data["usage"]
        usage.prompt_tokens = usage_data.get("prompt_tokens", 0)
        usage.completion_tokens = usage_data.get("completion_tokens", 0)
        usage.total_tokens = usage_data.get("total_tokens", 0)

        # Anthropic cache tokens (fixed: was reversed)
        usage.cache_write_tokens = usage_data.get("cache_creation_input_tokens", 0)
        usage.cache_read_tokens = usage_data.get("cache_read_input_tokens", 0)

        # Reasoning tokens (o1, DeepSeek)
        usage.reasoning_tokens = usage_data.get("reasoning_tokens", 0)

    return usage


__all__ = [
    "MODEL_PRICING",
    "CostRecord",
    "CostTracker",
    "TokenUsage",
    "create_token_usage_from_response",
]
