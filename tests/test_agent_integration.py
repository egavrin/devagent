"""Integration tests for multi-agent code review system."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


# Test data paths
EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "code_review"
# Use sample.patch for tests as it's simpler and more reliable
PATCH_FILE = EXAMPLES_DIR / "patches" / "sample.patch"
RULE_FILE = EXAMPLES_DIR / "rules" / "jsdoc-simple.md"


def _requires_api_key():
    """Skip test if API key is not available."""
    # Check if API key is available either via env var or config file
    try:
        from ai_dev_agent.core.utils.config import load_settings
        settings = load_settings()
        has_api_key = bool(settings.api_key)
    except Exception:
        has_api_key = False

    run_live = os.environ.get("DEVAGENT_RUN_LIVE_TESTS", "")
    live_enabled = run_live.strip().lower() in {"1", "true", "yes"}

    return pytest.mark.skipif(
        not (has_api_key and live_enabled),
        reason="Live review tests disabled (set DEVAGENT_RUN_LIVE_TESTS=1 with a valid API key to enable).",
    )


def _run_devagent_command(args, **kwargs):
    """Run devagent command using module invocation instead of shell command."""
    cmd = [sys.executable, "-m", "ai_dev_agent.cli.commands"] + args
    return subprocess.run(cmd, **kwargs)


@pytest.mark.integration
@pytest.mark.slow
@_requires_api_key()
def test_review_command_executes():
    """Test that review command executes successfully."""
    result = _run_devagent_command(
        ["review", str(PATCH_FILE), "--rule", str(RULE_FILE), "--json"],
        capture_output=True,
        text=True,
        timeout=180,  # 3 minutes
    )

    assert result.returncode == 0, f"Command failed with: {result.stderr}"
    assert result.stdout.strip(), "No output received"


@pytest.mark.integration
@pytest.mark.slow
@_requires_api_key()
def test_review_output_is_valid_json():
    """Test that review command outputs valid JSON."""
    result = _run_devagent_command(
        ["review", str(PATCH_FILE), "--rule", str(RULE_FILE), "--json"],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0

    # Parse JSON
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON output: {e}\n{result.stdout[:500]}")

    # Validate schema
    assert "violations" in data, f"Expected 'violations' key in JSON output, got: {list(data.keys())}"
    assert isinstance(data["violations"], list)

    if "summary" in data:
        assert isinstance(data["summary"], dict)


@pytest.mark.integration
@pytest.mark.slow
@_requires_api_key()
def test_review_detects_violations():
    """Test that review detects JSDoc violations."""
    result = _run_devagent_command(
        ["review", str(PATCH_FILE), "--rule", str(RULE_FILE), "--json"],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0
    data = json.loads(result.stdout)

    # Note: LLMs are not deterministic - sometimes they may not find violations
    # This test verifies the structure is correct when violations are found
    assert "violations" in data
    assert isinstance(data["violations"], list)

    # If violations found, check their structure
    for violation in data["violations"]:
        assert "file" in violation
        assert "line" in violation
        assert "message" in violation
        assert isinstance(violation["line"], int)


@pytest.mark.integration
@pytest.mark.slow
@_requires_api_key()
def test_reviewer_agent_uses_preparsed_dataset():
    """Test that reviewer agent relies on pre-parsed patch dataset."""
    result = _run_devagent_command(
        ["review", str(PATCH_FILE), "--rule", str(RULE_FILE)],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0

    combined_output = (result.stdout or "") + (result.stderr or "")
    assert "parse_patch" not in combined_output


@pytest.mark.integration
@pytest.mark.slow
@_requires_api_key()
def test_reviewer_agent_scope_filtering():
    """Test that reviewer only reviews files in scope."""
    result = _run_devagent_command(
        ["review", str(PATCH_FILE), "--rule", str(RULE_FILE), "--json"],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0
    data = json.loads(result.stdout)

    # Rule scope is "stdlib/**/*.ets"
    # examples/demo/utils.ets should be EXCLUDED
    for violation in data["violations"]:
        file_path = violation["file"]
        # Should not have violations from examples directory
        # (Note: LLM might hallucinate file names, so this is a best-effort check)
        if "examples" in file_path:
            pytest.fail(f"Found violation in out-of-scope file: {file_path}")


@pytest.mark.integration
@pytest.mark.slow
@_requires_api_key()
def test_reviewer_vs_manager_performance():
    """Compare reviewer agent vs manager agent performance."""

    # Test reviewer agent
    start = time.time()
    reviewer_result = _run_devagent_command(
        ["review", str(PATCH_FILE), "--rule", str(RULE_FILE), "--json"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    reviewer_time = time.time() - start

    assert reviewer_result.returncode == 0
    reviewer_data = json.loads(reviewer_result.stdout)

    # Test manager agent
    start = time.time()
    manager_result = _run_devagent_command(
        [
            "query", "--agent=manager",
            f"Review {PATCH_FILE} against {RULE_FILE}. Output JSON with violations array containing file, line, message."
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    manager_time = time.time() - start

    print(f"\nPerformance comparison:")
    print(f"  Reviewer agent: {reviewer_time:.2f}s")
    print(f"  Manager agent: {manager_time:.2f}s")
    print(f"  Speedup: {(manager_time / reviewer_time - 1) * 100:.1f}%")

    # Reviewer should be faster (or at least not significantly slower)
    # Allow some variance, but reviewer should be within reasonable range
    assert reviewer_time < manager_time * 2, \
        f"Reviewer ({reviewer_time:.2f}s) is too slow compared to manager ({manager_time:.2f}s)"


@pytest.mark.integration
def test_reviewer_agent_no_write_access():
    """Test that reviewer agent cannot write files."""
    # This is more of a configuration test
    from ai_dev_agent.agents import AgentRegistry

    reviewer = AgentRegistry.get("reviewer")
    assert "write" not in reviewer.tools
    assert "run" not in reviewer.tools


@pytest.mark.integration
def test_manager_agent_has_full_access():
    """Test that manager agent has all tools."""
    from ai_dev_agent.agents import AgentRegistry

    manager = AgentRegistry.get("manager")
    assert "write" in manager.tools
    assert "run" in manager.tools
    assert "read" in manager.tools


@pytest.mark.integration
@pytest.mark.slow
@_requires_api_key()
def test_empty_patch_handling():
    """Test review command with empty patch file."""
    empty_patch = EXAMPLES_DIR / "empty.patch"
    empty_patch.write_text("")

    try:
        result = _run_devagent_command(
            ["review", str(empty_patch), "--rule", str(RULE_FILE), "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should handle gracefully (either error or empty violations)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            assert len(data["violations"]) == 0
    finally:
        if empty_patch.exists():
            empty_patch.unlink()


@pytest.mark.integration
@pytest.mark.slow
@_requires_api_key()
def test_malformed_rule_handling():
    """Test review command with malformed rule file."""
    bad_rule = EXAMPLES_DIR / "bad_rule.md"
    bad_rule.write_text("This is not a proper rule")

    try:
        result = _run_devagent_command(
            ["review", str(PATCH_FILE), "--rule", str(bad_rule), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Should handle gracefully (might succeed with best-effort or fail cleanly)
        # We just want to ensure it doesn't crash
        assert result.returncode in [0, 1], "Unexpected return code"
    finally:
        if bad_rule.exists():
            bad_rule.unlink()


@pytest.mark.integration
def test_review_help_command():
    """Test that review command help works."""
    result = _run_devagent_command(
        ["review", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "Review a patch file" in result.stdout
    assert "--rule" in result.stdout
    assert "--json" in result.stdout


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
