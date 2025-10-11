#!/usr/bin/env python3
"""Test to verify tool invoker sends file lists to LLM for find/grep."""

from pathlib import Path
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.tool_invoker import SessionAwareToolInvoker
from ai_dev_agent.engine.react.types import ActionRequest, Observation
from ai_dev_agent.session import SessionManager

def test_find_formatted_output():
    """Test that find results include file lists in formatted_output."""

    # Create test observation with artifacts (files found)
    test_artifacts = [
        "file1.cpp",
        "file2.cpp",
        "file3.cpp",
        "subdir/file4.cpp",
        "subdir/file5.cpp"
    ]

    obs = Observation(
        success=True,
        outcome=f"Found {len(test_artifacts)} files",
        tool="find",
        artifacts=test_artifacts,
        metrics={"files": len(test_artifacts)}
    )

    # Create invoker
    settings = Settings()
    workspace = Path.cwd()
    session_manager = SessionManager.get_instance()
    session_id = "test-session"
    session_manager.ensure_session(session_id)

    invoker = SessionAwareToolInvoker(
        workspace=workspace,
        settings=settings,
        session_manager=session_manager,
        session_id=session_id
    )

    # Create action request
    action = ActionRequest(
        tool="find",
        args={"pattern": "*.cpp"},
        metadata={},
        step_id="1",
        thought="Testing find"
    )

    # Convert to CLI observation (this is where formatting happens)
    cli_obs = invoker._to_cli_observation(action, obs)

    # Verify formatted_output contains file list
    print(f"Display message: {cli_obs.display_message}")
    print(f"\nFormatted output for LLM:\n{cli_obs.formatted_output}")

    assert cli_obs.formatted_output is not None, "formatted_output should not be None for find"
    assert "Files found:" in cli_obs.formatted_output, "Should have header"
    assert "file1.cpp" in cli_obs.formatted_output, "Should contain first file"
    assert "file2.cpp" in cli_obs.formatted_output, "Should contain second file"

    print("\n✅ Test passed: find results include file lists")
    return True

def test_grep_formatted_output():
    """Test that grep results include file lists with match counts in formatted_output."""

    test_artifacts = [
        "src/handler.cpp",
        "src/types.h",
        "tests/test_types.cpp"
    ]

    # Simulate match counts from grep tool
    match_counts = {
        "src/handler.cpp": 5,
        "src/types.h": 2,
        "tests/test_types.cpp": 1
    }

    obs = Observation(
        success=True,
        outcome=f"Found matches in {len(test_artifacts)} files",
        tool="grep",
        artifacts=test_artifacts,
        metrics={"files": len(test_artifacts), "match_counts": match_counts}
    )

    settings = Settings()
    workspace = Path.cwd()
    session_manager = SessionManager.get_instance()
    session_id = "test-session-2"
    session_manager.ensure_session(session_id)

    invoker = SessionAwareToolInvoker(
        workspace=workspace,
        settings=settings,
        session_manager=session_manager,
        session_id=session_id
    )

    action = ActionRequest(
        tool="grep",
        args={"pattern": "IsAnyType"},
        metadata={},
        step_id="1",
        thought="Testing grep"
    )

    cli_obs = invoker._to_cli_observation(action, obs)

    print(f"Display message: {cli_obs.display_message}")
    print(f"\nFormatted output for LLM:\n{cli_obs.formatted_output}")

    assert cli_obs.formatted_output is not None, "formatted_output should not be None for grep"
    assert "Files with matches:" in cli_obs.formatted_output, "Should have header"
    assert "handler.cpp" in cli_obs.formatted_output, "Should contain file"
    assert "(5 matches)" in cli_obs.formatted_output, "Should show match count for handler.cpp"
    assert "(2 matches)" in cli_obs.formatted_output, "Should show match count for types.h"

    print("\n✅ Test passed: grep results include file lists with match counts")
    return True

def test_dynamic_limits():
    """Test that file limits adjust based on result count."""

    # Test with large result set (>50 files)
    large_artifacts = [f"file{i}.cpp" for i in range(100)]

    obs = Observation(
        success=True,
        outcome=f"Found {len(large_artifacts)} files",
        tool="find",
        artifacts=large_artifacts,
        metrics={"files": len(large_artifacts)}
    )

    settings = Settings()
    workspace = Path.cwd()
    session_manager = SessionManager.get_instance()
    session_id = "test-session-3"
    session_manager.ensure_session(session_id)

    invoker = SessionAwareToolInvoker(
        workspace=workspace,
        settings=settings,
        session_manager=session_manager,
        session_id=session_id
    )

    action = ActionRequest(
        tool="find",
        args={"pattern": "*.cpp"},
        metadata={},
        step_id="1",
        thought="Testing dynamic limits"
    )

    cli_obs = invoker._to_cli_observation(action, obs)

    print(f"Display message: {cli_obs.display_message}")
    print(f"\nFormatted output for LLM (first 200 chars):\n{cli_obs.formatted_output[:200]}...")

    # Should show only 20 files for large sets (>50)
    assert "... and 80 more files" in cli_obs.formatted_output, "Should indicate remaining files"
    assert "(Tip: Use more specific pattern" in cli_obs.formatted_output, "Should suggest refinement"

    print("\n✅ Test passed: dynamic limits work correctly")
    return True

if __name__ == "__main__":
    print("Testing tool invoker Phase 2 enhancements...\n")
    print("="*60)
    test_find_formatted_output()
    print("\n" + "="*60)
    test_grep_formatted_output()
    print("\n" + "="*60)
    test_dynamic_limits()
    print("\n" + "="*60)
    print("\n🎉 All Phase 2 tests passed!")
