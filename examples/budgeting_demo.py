#!/usr/bin/env python3
"""Demo script showing enhanced budgeting features in action.

This example demonstrates:
1. Cost tracking across iterations
2. Adaptive phase management
3. Retry mechanism with exponential backoff
4. Two-tier context pruning
5. Reflection for error recovery
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_dev_agent.cli.react.budget_control import PHASE_PROMPTS, AdaptiveBudgetManager
from ai_dev_agent.core.utils.context_budget import ContextBudgetConfig, estimate_tokens
from ai_dev_agent.core.utils.cost_tracker import CostTracker, TokenUsage
from ai_dev_agent.core.utils.retry_handler import create_retry_handler
from ai_dev_agent.providers.llm.base import Message


def simulate_llm_call(messages, phase, cost_tracker, iteration):
    """Simulate an LLM call with cost tracking."""
    # Estimate tokens
    prompt_tokens = estimate_tokens(messages)
    completion_tokens = int(prompt_tokens * 0.3)  # Assume 30% output ratio

    # Track usage
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    # Track cost (using GPT-4 as example)
    cost_tracker.track_request(
        model="gpt-4o",
        usage=usage,
        operation="completion",
        iteration=iteration,
        phase=phase,
    )

    return f"Response for phase: {phase}"


def demo_adaptive_budgeting():
    """Demonstrate adaptive budget management."""
    print("=" * 60)
    print("ENHANCED BUDGETING DEMO")
    print("=" * 60)

    # Initialize components
    cost_tracker = CostTracker()
    budget_manager = AdaptiveBudgetManager(
        max_iterations=10,
        model_context_window=128000,  # GPT-4 context window
        enable_reflection=True,
    )

    # Simulate conversation
    messages = [
        Message(role="system", content="You are a helpful coding assistant."),
        Message(role="user", content="Help me refactor this complex codebase."),
    ]

    print("\n📊 Starting iterations with adaptive phases:\n")

    # Run iterations
    for _i in range(10):
        context = budget_manager.next_iteration()
        if context is None:
            break

        # Show phase info
        print(f"Iteration {context.number}/{context.total}:")
        print(f"  Phase: {context.phase.upper()}")
        print(
            f"  Progress: [{'█' * int(context.percent_complete / 5)}{'░' * (20 - int(context.percent_complete / 5))}] {context.percent_complete:.0f}%"
        )

        # Get phase-specific prompt
        if context.phase in PHASE_PROMPTS:
            print(f"  Guidance: {PHASE_PROMPTS[context.phase][:60]}...")

        # Simulate LLM call
        response = simulate_llm_call(messages, context.phase, cost_tracker, context.number)
        messages.append(Message(role="assistant", content=response))

        # Simulate tool use in exploration/investigation
        if context.phase in ["exploration", "investigation"]:
            tool_output = f"Tool output for iteration {context.number}" * 50
            messages.append(Message(role="tool", content=tool_output))

        # Show cost tracking
        print(f"  {cost_tracker.format_inline()}")

        # Simulate error and reflection
        if context.number == 5:
            print("\n  ⚠️ Error occurred! Attempting reflection...")
            if budget_manager.allow_reflection("Network timeout"):
                print("  ✓ Reflection allowed, retrying with adjusted approach")
            else:
                print("  ✗ Reflection limit reached")

        # Adjust phases based on progress
        if context.number % 3 == 0:
            success_rate = 0.9 if context.phase == "exploration" else 0.4
            budget_manager.adjust_phases_for_progress(success_rate)

        print()

    # Final summary
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(cost_tracker.format_summary(detailed=True))
    print("\nBudget Stats:")
    for key, value in budget_manager.get_stats().items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def demo_retry_mechanism():
    """Demonstrate retry with exponential backoff."""
    print("\n" + "=" * 60)
    print("RETRY MECHANISM DEMO")
    print("=" * 60)

    retry_handler = create_retry_handler(smart=True, max_retries=3)

    def flaky_function(attempt=None):
        """Function that fails first 2 times."""
        if attempt is None:
            attempt = [0]
        attempt[0] += 1
        if attempt[0] < 3:
            raise ConnectionError(f"Connection failed (attempt {attempt[0]})")
        return "Success!"

    def on_retry(attempt, error, delay):
        print(f"  Retry {attempt}: {error} (waiting {delay:.2f}s)")

    try:
        print("\nAttempting flaky operation...")
        result = retry_handler.execute_with_retry(flaky_function, on_retry=on_retry)
        print(f"✓ Result: {result}")
        print(f"  Stats: {retry_handler.get_retry_stats()}")
    except Exception as e:
        print(f"✗ Failed: {e}")


def demo_two_tier_pruning():
    """Demonstrate two-tier context pruning."""
    print("\n" + "=" * 60)
    print("TWO-TIER PRUNING DEMO")
    print("=" * 60)

    # Create messages with large tool outputs
    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="User request"),
    ]

    # Add many tool outputs
    for i in range(20):
        messages.append(
            Message(
                role="tool", content=f"Tool output {i}: " + "x" * 2000, tool_call_id=f"call_{i}"
            )
        )

    print(f"\nInitial messages: {len(messages)}")
    print(f"Initial tokens: ~{estimate_tokens(messages):,}")

    # Apply two-tier pruning
    from ai_dev_agent.core.utils.context_budget import prune_messages

    config = ContextBudgetConfig(
        max_tokens=10000,
        enable_two_tier=True,
        prune_protect_tokens=5000,
        prune_minimum_savings=2000,
    )

    pruned = prune_messages(messages, config)

    print("\nAfter pruning:")
    print(f"  Messages: {len(pruned)}")
    print(f"  Tokens: ~{estimate_tokens(pruned):,}")

    # Show what happened to tool outputs
    tool_messages = [m for m in pruned if m.role == "tool"]
    truncated = [m for m in tool_messages if "pruned for context" in (m.content or "")]
    print(f"  Tool outputs truncated: {len(truncated)}/{len(tool_messages)}")


def demo_cost_forecasting():
    """Demonstrate cost forecasting."""
    print("\n" + "=" * 60)
    print("COST FORECASTING DEMO")
    print("=" * 60)

    cost_tracker = CostTracker()

    # Simulate some iterations
    for i in range(1, 4):
        usage = TokenUsage(
            prompt_tokens=1000 * i,
            completion_tokens=300 * i,
        )
        cost_tracker.track_request(
            model="gpt-4o",
            usage=usage,
            iteration=i,
        )

    print(f"\nCurrent cost: ${cost_tracker.total_cost_usd:.4f}")
    print(
        f"Tokens used: {cost_tracker.total_prompt_tokens:,} in, {cost_tracker.total_completion_tokens:,} out"
    )

    # Forecast remaining cost
    remaining_iterations = 7
    forecast = cost_tracker.get_forecast(remaining_iterations)
    print(f"\nForecast for {remaining_iterations} more iterations: ${forecast:.4f}")
    print(f"Estimated total: ${cost_tracker.total_cost_usd + forecast:.4f}")

    # Check warning threshold
    if cost_tracker.should_warn(0.10):  # $0.10 threshold
        print("\n⚠️ WARNING: Cost exceeds threshold!")


if __name__ == "__main__":
    # Run all demos
    demo_adaptive_budgeting()
    demo_retry_mechanism()
    demo_two_tier_pruning()
    demo_cost_forecasting()

    print("\n" + "=" * 60)
    print("✅ All demos completed successfully!")
    print("=" * 60)
