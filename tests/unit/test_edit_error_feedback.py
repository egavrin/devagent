"""Test that EDIT failures provide detailed error feedback to the LLM."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.tool_invoker import SessionAwareToolInvoker
from ai_dev_agent.engine.react.types import ActionRequest
from ai_dev_agent.session import SessionManager


@pytest.fixture
def test_workspace(tmp_path):
    """Create a test workspace with a sample file."""
    readme = tmp_path / "README.md"
    readme.write_text("# Test Project\n\nThis is a test.\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def session_aware_invoker(test_workspace):
    """Create a SessionAwareToolInvoker for testing."""
    settings = Settings(workspace_root=test_workspace)
    session_manager = SessionManager.get_instance()
    session_id = "test-edit-error-feedback"

    invoker = SessionAwareToolInvoker(
        workspace=test_workspace,
        settings=settings,
        session_manager=session_manager,
        session_id=session_id,
    )

    # Initialize session
    session_manager.ensure_session(session_id, system_messages=[], metadata={"test": True})

    yield invoker

    # Cleanup
    try:
        session_manager.delete_session(session_id)
    except Exception:
        pass


def test_edit_failure_includes_detailed_errors(session_aware_invoker, test_workspace):
    """Test that EDIT failures include detailed error information in tool messages."""
    # Create a patch that will fail due to context mismatch
    failing_patch = """*** Begin Patch
*** Update File: README.md
@@
-# Wrong Title
-This does not match
+# Test Project
+Fixed content
*** End Patch
"""

    # Execute the failing EDIT
    action = ActionRequest(
        step_id="1",
        thought="Testing EDIT error feedback",
        tool="edit",
        args={"patch": failing_patch},
        metadata={"tool_call_id": "test-edit-fail-1"},
    )

    observation = session_aware_invoker(action)

    # Verify the observation reflects failure
    assert not observation.success
    assert "failed" in observation.outcome.lower()

    # Get the session history to check what was sent to the LLM
    session_id = session_aware_invoker.session_id
    session = session_aware_invoker.session_manager.get_session(session_id)

    # Find the tool message that was recorded
    tool_messages = [msg for msg in session.history if msg.role == "tool"]
    assert len(tool_messages) > 0, "No tool message was recorded"

    latest_tool_msg = tool_messages[-1]

    # The tool message content should include detailed error information
    assert latest_tool_msg.content is not None
    content = latest_tool_msg.content

    # Should contain error details, not just "Edit failed"
    assert "error" in content.lower() or "Error" in content

    # Should provide specific information about what went wrong
    # The error should mention "context" since that's what failed to match
    assert "context" in content.lower() or "not found" in content.lower()

    # Should NOT be just the display message - must have details
    assert len(content) > 20, "Tool message is too short - missing detailed errors"


def test_edit_success_does_not_include_error_details(session_aware_invoker, test_workspace):
    """Test that successful EDITs don't incorrectly include error information."""
    # Create a patch that will succeed
    success_patch = """*** Begin Patch
*** Update File: README.md
@@
-This is a test.
+This is an updated test.
*** End Patch
"""

    action = ActionRequest(
        step_id="1",
        thought="Testing successful EDIT",
        tool="edit",
        args={"patch": success_patch},
        metadata={"tool_call_id": "test-edit-success-1"},
    )

    observation = session_aware_invoker(action)

    # Verify success
    assert observation.success

    # Get the tool message
    session_id = session_aware_invoker.session_id
    session = session_aware_invoker.session_manager.get_session(session_id)
    tool_messages = [msg for msg in session.history if msg.role == "tool"]

    latest_tool_msg = tool_messages[-1]
    content = latest_tool_msg.content

    # Should NOT contain error details for successful edits
    assert "Error details:" not in content
    assert "Raw error:" not in content


def test_edit_multiple_errors_all_reported(session_aware_invoker, test_workspace):
    """Test that when multiple errors occur, all are reported to the LLM."""
    # Create a patch with multiple problems
    multi_error_patch = """*** Begin Patch
*** Update File: README.md
@@
-Non-existent line 1
+Replacement 1
@@
-Non-existent line 2
+Replacement 2
*** End Patch
"""

    action = ActionRequest(
        step_id="1",
        thought="Testing multiple errors",
        tool="edit",
        args={"patch": multi_error_patch},
        metadata={"tool_call_id": "test-edit-multi-error"},
    )

    observation = session_aware_invoker(action)

    # Should fail
    assert not observation.success

    # Check tool message includes multiple errors
    session_id = session_aware_invoker.session_id
    session = session_aware_invoker.session_manager.get_session(session_id)
    tool_messages = [msg for msg in session.history if msg.role == "tool"]

    latest_tool_msg = tool_messages[-1]
    content = latest_tool_msg.content

    # Should mention multiple chunks failed (Chunk 1, Chunk 2)
    # Note: The actual error format depends on the patch applier implementation
    assert "Chunk" in content or "chunk" in content or "context" in content.lower()
