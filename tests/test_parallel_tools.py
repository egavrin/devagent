"""Test parallel tool execution capabilities."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.tool_invoker import RegistryToolInvoker
from ai_dev_agent.engine.react.types import ActionRequest, ToolCall
from ai_dev_agent.tools import READ
from ai_dev_agent.providers.llm.base import Message, ToolCallResult, ToolCall as LLMToolCall


@pytest.fixture
def tool_invoker(tmp_path):
    """Create a tool invoker for testing."""
    settings = Settings()
    return RegistryToolInvoker(
        workspace=tmp_path,
        settings=settings,
    )


def test_batch_tool_execution(tool_invoker, tmp_path):
    """Test that multiple tool calls can be executed in batch."""
    # Create test files
    file1 = tmp_path / "test1.txt"
    file2 = tmp_path / "test2.txt"
    file3 = tmp_path / "test3.txt"

    file1.write_text("Content of file 1")
    file2.write_text("Content of file 2")
    file3.write_text("Content of file 3")

    # Create an action with multiple tool calls
    action = ActionRequest(
        step_id="test",
        thought="Read multiple files in parallel",
        tool=READ,  # Backward compatibility field
        args={},
        tool_calls=[
            ToolCall(
                tool=READ,
                args={"paths": [str(file1)]},
                call_id="call_1",
            ),
            ToolCall(
                tool=READ,
                args={"paths": [str(file2)]},
                call_id="call_2",
            ),
            ToolCall(
                tool=READ,
                args={"paths": [str(file3)]},
                call_id="call_3",
            ),
        ],
    )

    # Execute the batch
    start = time.time()
    observation = tool_invoker(action)
    elapsed = time.time() - start

    # Verify batch execution
    assert observation.success is True
    assert "3 tool(s)" in observation.outcome
    assert len(observation.results) == 3

    # Verify each result
    for result in observation.results:
        assert result.success is True
        assert result.tool == READ
        assert result.call_id in ["call_1", "call_2", "call_3"]

    # Batch execution should be reasonably fast
    # (This is a weak assertion since actual parallel speedup depends on system)
    assert elapsed < 2.0  # Should complete quickly

    print(f"✓ Batch execution of 3 tool calls completed in {elapsed:.3f}s")


def test_single_tool_backward_compatibility(tool_invoker, tmp_path):
    """Test that single tool execution still works (backward compatibility)."""
    test_file = tmp_path / "single.txt"
    test_file.write_text("Single file content")

    # Old-style single tool action
    action = ActionRequest(
        step_id="test",
        thought="Read one file",
        tool=READ,
        args={"paths": [str(test_file)]},
    )

    observation = tool_invoker(action)

    assert observation.success is True
    assert "Read 1 file(s)" in observation.outcome
    assert len(observation.results) == 0  # Single-tool mode doesn't populate results


def test_batch_execution_with_failures(tool_invoker, tmp_path):
    """Test batch execution handles partial failures gracefully."""
    existing_file = tmp_path / "exists.txt"
    existing_file.write_text("This file exists")

    action = ActionRequest(
        step_id="test",
        thought="Read files, some missing",
        tool=READ,
        args={},
        tool_calls=[
            ToolCall(
                tool=READ,
                args={"paths": [str(existing_file)]},
                call_id="call_1",
            ),
            ToolCall(
                tool="unknown.tool",  # This will fail
                args={},
                call_id="call_2",
            ),
        ],
    )

    observation = tool_invoker(action)

    # Overall success should be False if any call failed
    assert observation.success is False
    assert len(observation.results) == 2

    # Check individual results
    assert observation.results[0].success is True
    assert observation.results[1].success is False
    assert "not registered" in observation.results[1].error.lower()


def test_empty_batch_falls_back_to_single_mode(tool_invoker, tmp_path):
    """Test that empty batch requests fall back to single-tool mode."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")

    action = ActionRequest(
        step_id="test",
        thought="Empty batch falls back",
        tool=READ,
        args={"paths": [str(test_file)]},
        tool_calls=[],  # Empty batch - should use single-tool mode
    )

    observation = tool_invoker(action)

    # Empty tool_calls list should fall back to single-tool mode
    assert observation.success is True
    assert len(observation.results) == 0  # Single-tool mode doesn't populate results


def test_llm_client_parallel_tool_calls_parameter():
    """Test that LLM client properly handles parallel_tool_calls parameter."""
    from ai_dev_agent.providers.llm.base import HTTPChatLLMClient

    # Create a mock LLM client subclass
    class MockLLMClient(HTTPChatLLMClient):
        def _prepare_payload(self, messages, temperature, max_tokens):
            return {
                "model": self.model,
                "messages": [msg.to_payload() for msg in messages],
                "temperature": temperature,
            }

    # Create client instance
    client = MockLLMClient(
        provider_name="test",
        api_key="test-key",
        model="test-model",
        base_url="https://api.test.com",
    )

    # Mock the _post method to capture the payload
    captured_payload = {}
    def mock_post(payload, extra_headers=None):
        captured_payload.update(payload)
        return {
            "choices": [{
                "message": {
                    "content": "test response",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "test_tool",
                                "arguments": "{}"
                            }
                        }
                    ]
                }
            }]
        }

    client._post = mock_post

    # Test with parallel_tool_calls=True (default)
    messages = [Message(role="user", content="test")]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    result = client.invoke_tools(messages, tools, parallel_tool_calls=True)

    # Verify parallel_tool_calls was included in payload
    assert "parallel_tool_calls" in captured_payload
    assert captured_payload["parallel_tool_calls"] is True
    assert result.calls[0].name == "test_tool"

    # Test with parallel_tool_calls=False
    captured_payload.clear()
    result = client.invoke_tools(messages, tools, parallel_tool_calls=False)

    assert "parallel_tool_calls" in captured_payload
    assert captured_payload["parallel_tool_calls"] is False

    print("✓ LLM client properly handles parallel_tool_calls parameter")


def test_action_provider_enables_parallel_tool_calls():
    """Test that ActionProvider enables parallel_tool_calls by default."""
    from ai_dev_agent.cli.react.action_provider import LLMActionProvider
    from ai_dev_agent.session import SessionManager
    from ai_dev_agent.engine.react.types import TaskSpec

    # Create mock LLM client
    mock_client = MagicMock()
    mock_client.invoke_tools.return_value = ToolCallResult(
        calls=[],
        message_content="test response",
    )

    # Create session manager
    session_manager = SessionManager.get_instance()
    session_id = "test-session-parallel"
    session_manager.ensure_session(
        session_id,
        system_messages=[Message(role="system", content="test")],
    )

    # Create action provider
    action_provider = LLMActionProvider(
        llm_client=mock_client,
        session_manager=session_manager,
        session_id=session_id,
        tools=[{"type": "function", "function": {"name": "test_tool"}}],
    )

    # Create a task
    task = TaskSpec(
        identifier="test-task",
        goal="test goal",
        category="test",
    )

    # Call the action provider (will trigger StopIteration since no tool calls)
    try:
        action_provider(task, [])
    except StopIteration:
        pass  # Expected when no tool calls returned

    # Verify invoke_tools was called with parallel_tool_calls=True
    assert mock_client.invoke_tools.called
    call_kwargs = mock_client.invoke_tools.call_args[1]
    assert "parallel_tool_calls" in call_kwargs
    assert call_kwargs["parallel_tool_calls"] is True

    print("✓ ActionProvider enables parallel_tool_calls by default")


def test_action_provider_handles_non_numeric_model_limit():
    """LLMActionProvider should ignore non-numeric model limits."""
    from ai_dev_agent.cli.react.action_provider import LLMActionProvider
    from ai_dev_agent.session import SessionManager
    from ai_dev_agent.engine.react.types import TaskSpec

    mock_client = MagicMock()
    mock_client.invoke_tools.return_value = ToolCallResult(
        calls=[LLMToolCall(name="noop", arguments={}, call_id="c1")],
        message_content=None,
    )
    mock_client._MAX_CONTEXT_TOKENS = MagicMock()  # Non-numeric limit

    session_manager = SessionManager.get_instance()
    session_id = "test-session-non-numeric"
    session_manager.ensure_session(
        session_id,
        system_messages=[Message(role="system", content="test")],
    )

    provider = LLMActionProvider(
        llm_client=mock_client,
        session_manager=session_manager,
        session_id=session_id,
        tools=[{"type": "function", "function": {"name": "noop"}}],
    )

    task = TaskSpec(identifier="task", goal="goal")

    provider(task, [])

    assert mock_client.invoke_tools.called


def test_deepseek_client_does_not_send_parallel_tool_calls():
    """Test that DeepSeekClient excludes parallel_tool_calls from API payload."""
    from ai_dev_agent.providers.llm.deepseek import DeepSeekClient
    from ai_dev_agent.providers.llm.base import Message

    # Create DeepSeek client
    client = DeepSeekClient(
        api_key="test-key",
        model="deepseek-chat",
    )

    # Mock the _post method to capture the payload
    captured_payload = {}
    def mock_post(payload, extra_headers=None):
        captured_payload.update(payload)
        return {
            "choices": [{
                "message": {
                    "content": "test response",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "test_tool",
                                "arguments": "{}"
                            }
                        }
                    ]
                }
            }]
        }

    client._post = mock_post

    # Test with parallel_tool_calls=True (should be excluded from payload)
    messages = [Message(role="user", content="test")]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    result = client.invoke_tools(messages, tools, parallel_tool_calls=True)

    # Verify parallel_tool_calls was NOT included in payload (DeepSeek doesn't support it)
    assert "parallel_tool_calls" not in captured_payload
    assert "tools" in captured_payload  # But tools should be present
    assert result.calls[0].name == "test_tool"

    print("✓ DeepSeek client correctly excludes parallel_tool_calls parameter")
