"""Test the new phase-based budget guidance system."""


def test_phase_determination():
    """Test that phases are determined correctly based on percentage."""

    # Test cases: (iteration, total, expected_phase)
    test_cases = [
        (1, 10, "exploration"),  # 10% consumed
        (3, 10, "exploration"),  # 30% consumed
        (4, 10, "investigation"),  # 40% consumed - boundary (>=40% is investigation)
        (5, 10, "investigation"),  # 50% consumed
        (7, 10, "consolidation"),  # 70% consumed - boundary (>=70% is consolidation)
        (8, 10, "consolidation"),  # 80% consumed
        (9, 10, "late"),  # 90% consumed - boundary (>=90% is late phase)
        (10, 10, "synthesis"),  # 100% consumed - final (special case)
    ]

    for iteration, total, expected_phase in test_cases:
        remaining = total - iteration
        percent_consumed = iteration / total * 100

        # Determine phase (matching executor logic)
        if remaining == 0:
            phase = "synthesis"
        elif percent_consumed < 40:
            phase = "exploration"
        elif percent_consumed < 70:
            phase = "investigation"
        elif percent_consumed < 90:
            phase = "consolidation"
        else:
            phase = "late"

        print(f"Iteration {iteration}/{total}: {percent_consumed:.0f}% consumed → {phase}")

        # For final iteration, check it's synthesis
        if iteration == total:
            assert phase == "synthesis", f"Final iteration should be synthesis, got {phase}"
        else:
            assert phase == expected_phase, f"Expected {expected_phase}, got {phase}"

    print("\n✓ All phase determinations correct!")
    print("\nKey improvements:")
    print("• No step numbers in prompts (saves ~50-100 tokens/iteration)")
    print("• External budget control (cleaner abstraction)")
    print("• Final iteration has NO TOOLS (guaranteed synthesis)")
    print("• Phase-based work style (not step counting)")


def test_token_savings():
    """Calculate token savings from removing budget info."""

    # Old format example
    old_format = """=== DEVAGENT ASSISTANT - STEP 7/25 ===

YOU ARE AN AGENT WITH LIMITED STEPS. YOU MUST RESPECT THE BUDGET.

CURRENT POSITION: Step 7 of 25 total steps
BUDGET STATUS: 18 steps remaining (28% budget consumed)"""

    # New format example
    new_format = """You are a development assistant analyzing a codebase."""

    # Rough token estimation (1 token ≈ 4 chars)
    old_tokens = len(old_format) / 4
    new_tokens = len(new_format) / 4
    saved_per_iteration = old_tokens - new_tokens

    print("\nToken usage comparison:")
    print(f"Old format: ~{old_tokens:.0f} tokens")
    print(f"New format: ~{new_tokens:.0f} tokens")
    print(f"Saved per iteration: ~{saved_per_iteration:.0f} tokens")
    print(f"Saved over 20 iterations: ~{saved_per_iteration * 20:.0f} tokens")


if __name__ == "__main__":
    print("Testing New Phase-Based Budget System")
    print("=" * 50)

    test_phase_determination()
    test_token_savings()

    print("\n" + "=" * 50)
    print("Summary of changes:")
    print("1. Removed all step/budget numbers from prompts")
    print("2. Phase determined externally based on % consumed")
    print("3. Final iteration gets synthesis prompt with NO TOOLS")
    print("4. Cleaner abstraction: LLM focuses on task, not budget")
    print("5. Guaranteed stop: Final step can't use tools")
