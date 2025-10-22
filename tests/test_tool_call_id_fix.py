"""Test to ensure tool messages always have tool_call_id for API compatibility."""

import pytest
from unittest.mock import MagicMock, patch
from ai_dev_agent.cli.react.action_provider import LLMActionProvider
from ai_dev_agent.providers.llm.base import Message, ToolCallResult
from ai_dev_agent.session import SessionManager


def test_dummy_tool_messages_have_tool_call_id():
    """Test that _record_dummy_tool_messages ensures all tool messages have tool_call_id."""
    # Setup
    mock_client = MagicMock()
    session_manager = SessionManager.get_instance()
    session_id = "test-session"
    session_manager.ensure_session(session_id)

    action_provider = LLMActionProvider(
        llm_client=mock_client,
        session_manager=session_manager,
        session_id=session_id,
        tools=[]
    )

    # Simulate normalized tool calls (as they would be after _normalize_tool_calls)
    raw_tool_calls = [
        {
            "id": "tool-0-abc123",  # ID should be present after normalization
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": '{"arg": "value"}'
            }
        },
        {
            # Edge case: missing ID (shouldn't happen after normalization but testing fallback)
            "type": "function",
            "function": {
                "name": "another_tool",
                "arguments": '{}'
            }
        }
    ]

    # Call the method
    action_provider._record_dummy_tool_messages(raw_tool_calls)

    # Get the session history
    conversation = session_manager.compose(session_id)

    # Find tool messages
    tool_messages = [msg for msg in conversation if msg.role == "tool"]

    # Verify all tool messages have tool_call_id
    assert len(tool_messages) == 2, f"Expected 2 tool messages, got {len(tool_messages)}"

    for i, msg in enumerate(tool_messages):
        payload = msg.to_payload()
        assert "tool_call_id" in payload, f"Tool message {i} missing tool_call_id field"
        assert payload["tool_call_id"], f"Tool message {i} has empty tool_call_id"

        # First message should have the original ID
        if i == 0:
            assert payload["tool_call_id"] == "tool-0-abc123"
        # Second message should have a generated ID
        else:
            assert payload["tool_call_id"].startswith("tool-"), \
                f"Generated tool_call_id should start with 'tool-', got {payload['tool_call_id']}"


def test_tool_message_to_payload_includes_tool_call_id():
    """Test that Message.to_payload() always includes tool_call_id for tool messages."""
    # Test with explicit tool_call_id
    msg1 = Message(role="tool", content="Response", tool_call_id="test-id-123")
    payload1 = msg1.to_payload()
    assert payload1["role"] == "tool"
    assert payload1["tool_call_id"] == "test-id-123"

    # Test with None tool_call_id (should not be in payload)
    msg2 = Message(role="tool", content="Response", tool_call_id=None)
    payload2 = msg2.to_payload()
    assert payload2["role"] == "tool"
    assert "tool_call_id" not in payload2  # None should not be included

    # Test non-tool messages don't get tool_call_id incorrectly
    msg3 = Message(role="user", content="Question")
    payload3 = msg3.to_payload()
    assert "tool_call_id" not in payload3


def test_deepseek_api_compatibility():
    """Test that conversation payloads are compatible with DeepSeek API requirements."""
    session_manager = SessionManager.get_instance()
    session_id = "deepseek-test"
    session_manager.ensure_session(session_id)

    # Build a conversation with tool calls
    session_manager.add_user_message(session_id, "Test query")

    # Assistant with tool calls
    session_manager.add_assistant_message(
        session_id,
        "Processing...",
        tool_calls=[
            {
                "id": "call-123",
                "type": "function",
                "function": {"name": "test", "arguments": "{}"}
            }
        ]
    )

    # Tool response with matching ID
    session_manager.add_tool_message(session_id, "call-123", "Tool output")

    # Get conversation for API
    conversation = session_manager.compose(session_id)
    payloads = [msg.to_payload() for msg in conversation]

    # Verify structure matches DeepSeek requirements
    assert len(payloads) == 3

    # Check assistant message has tool_calls
    assistant_msg = payloads[1]
    assert assistant_msg["role"] == "assistant"
    assert "tool_calls" in assistant_msg
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["id"] == "call-123"

    # Check tool message has matching tool_call_id
    tool_msg = payloads[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call-123"

    # Verify the IDs match (DeepSeek requirement)
    assert assistant_msg["tool_calls"][0]["id"] == tool_msg["tool_call_id"]


if __name__ == "__main__":
    # Run tests directly
    test_dummy_tool_messages_have_tool_call_id()
    print("✅ test_dummy_tool_messages_have_tool_call_id passed")

    test_tool_message_to_payload_includes_tool_call_id()
    print("✅ test_tool_message_to_payload_includes_tool_call_id passed")

    test_deepseek_api_compatibility()
    print("✅ test_deepseek_api_compatibility passed")

    print("\n✨ All tests passed!")