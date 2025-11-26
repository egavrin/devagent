from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from ai_dev_agent.cli.react.action_provider import LLMActionProvider
from ai_dev_agent.core.utils.constants import LLM_DEFAULT_TEMPERATURE
from ai_dev_agent.engine.react.types import TaskSpec
from ai_dev_agent.providers.llm.base import ToolCall, ToolCallResult
from ai_dev_agent.session import SessionManager


@dataclass
class StubLLMClient:
    model: str = "stub-model"
    last_tools: Optional[list[dict[str, Any]]] = None

    def invoke_tools(
        self, messages, *, tools, temperature: float = LLM_DEFAULT_TEMPERATURE, **kwargs
    ):
        self.last_tools = list(tools)
        call = ToolCall(name="run", arguments={"cmd": "echo hi"}, call_id="call-1")
        return ToolCallResult(
            calls=[call],
            message_content="Executing command",
            raw_tool_calls=[{"id": "call-1", "function": {"name": "run"}}],
            _raw_response={"usage": {"total_tokens": 12}},
        )


def test_llm_action_provider_returns_action():
    session_manager = SessionManager.get_instance()
    session_id = f"provider-{uuid4()}"
    session_manager.ensure_session(session_id, system_messages=[])
    client = StubLLMClient()
    tools = [{"type": "function", "function": {"name": "run"}}]
    provider = LLMActionProvider(
        llm_client=client,
        session_manager=session_manager,
        session_id=session_id,
        tools=tools,
    )

    task = TaskSpec(identifier="T-action", goal="Test")
    action = provider(task, history=[])

    assert action.tool == "run"
    assert action.metadata["iteration"] == 1
    assert action.metadata["phase"] == "exploration"
    assert action.metadata["tool_call_id"] == "call-1"
    assert client.last_tools == tools

    session = session_manager.get_session(session_id)
    with session.lock:
        assert session.history
        last_message = session.history[-1]
    assert last_message.role == "assistant"


def test_llm_action_provider_stop_iteration():
    class EmptyLLM(StubLLMClient):
        def invoke_tools(
            self, messages, *, tools, temperature: float = LLM_DEFAULT_TEMPERATURE, **kwargs
        ):
            return ToolCallResult(
                calls=[],
                message_content="Done",
                raw_tool_calls=[],
                _raw_response={"usage": {"total_tokens": 4}},
            )

    session_manager = SessionManager.get_instance()
    session_id = f"provider-empty-{uuid4()}"
    session_manager.ensure_session(session_id, system_messages=[])
    client = EmptyLLM()
    provider = LLMActionProvider(
        llm_client=client,
        session_manager=session_manager,
        session_id=session_id,
        tools=[],
    )
    provider.update_phase("synthesis", is_final=True)

    task = TaskSpec(identifier="T-stop", goal="Stop test")

    # In final iteration with message_content but no tool calls,
    # it now returns a submit_final_answer action instead of raising StopIteration
    action = provider(task, history=[])
    assert action.tool == "submit_final_answer"
    assert action.args["answer"] == "Done"
    assert action.metadata["synthesized_from_text"] is True


def test_inject_repomap_update_uses_correct_messages():
    """Test that _inject_repomap_update properly injects repomap messages."""
    from pathlib import Path
    from unittest.mock import Mock, patch

    from ai_dev_agent.cli.dynamic_context import DynamicContextTracker

    # Create session and provider
    session_manager = SessionManager.get_instance()
    session_id = f"provider-repomap-{uuid4()}"
    session_manager.ensure_session(session_id, system_messages=[])

    client = StubLLMClient()
    provider = LLMActionProvider(
        llm_client=client,
        session_manager=session_manager,
        session_id=session_id,
        tools=[],
    )

    # Mock the context objects
    dynamic_context = DynamicContextTracker(Path.cwd())
    dynamic_context.mentioned_files = {"/path/to/file.py"}
    dynamic_context.mentioned_symbols = {"TestClass", "test_function"}

    # Mock settings with repomap debug enabled
    mock_settings = Mock()
    mock_settings.repomap_debug_stdout = True

    # Set up the context object
    provider._ctx_obj = {
        "_dynamic_context": dynamic_context,
        "_user_prompt": "Test query",
        "settings": mock_settings,
    }

    # Mock the repomap messages that should be returned
    mock_repomap_messages = [
        {"role": "user", "content": "Here are the relevant files:\n• /path/to/file.py"},
        {"role": "assistant", "content": "I can see the relevant files. I'll read them directly."},
    ]

    # Patch the enhancer to return our mock messages
    with patch("ai_dev_agent.cli.context_enhancer.get_context_enhancer") as mock_get_enhancer:
        mock_enhancer = Mock()
        mock_enhancer.get_repomap_messages.return_value = ("Test query", mock_repomap_messages)
        mock_get_enhancer.return_value = mock_enhancer

        # Call the method we're testing
        provider._inject_repomap_update()

        # Verify the enhancer was called with correct parameters
        mock_enhancer.get_repomap_messages.assert_called_once_with(
            query="Test query",
            max_files=15,
            additional_files=dynamic_context.mentioned_files,
            additional_symbols=dynamic_context.mentioned_symbols,
        )

    # Verify messages were added to session
    session = session_manager.get_session(session_id)
    with session.lock:
        # Find the injected messages in the session history
        messages = [msg.content for msg in session.history]

    # Check that both user and assistant messages were added
    assert "Here are the relevant files:\n• /path/to/file.py" in messages
    assert "I can see the relevant files. I'll read them directly." in messages

    # Verify they were added in the correct order
    user_idx = messages.index("Here are the relevant files:\n• /path/to/file.py")
    asst_idx = messages.index("I can see the relevant files. I'll read them directly.")
    assert asst_idx == user_idx + 1, "Assistant message should immediately follow user message"


def test_inject_repomap_update_handles_empty_messages():
    """Test that _inject_repomap_update handles case when no repomap messages returned."""
    from pathlib import Path
    from unittest.mock import Mock, patch

    from ai_dev_agent.cli.dynamic_context import DynamicContextTracker

    # Create session and provider
    session_manager = SessionManager.get_instance()
    session_id = f"provider-empty-repomap-{uuid4()}"
    session_manager.ensure_session(session_id, system_messages=[])

    client = StubLLMClient()
    provider = LLMActionProvider(
        llm_client=client,
        session_manager=session_manager,
        session_id=session_id,
        tools=[],
    )

    # Set up context with no files/symbols
    dynamic_context = DynamicContextTracker(Path.cwd())
    provider._ctx_obj = {
        "_dynamic_context": dynamic_context,
        "_user_prompt": "Broad query with no symbols",
        "settings": Mock(repomap_debug_stdout=False),
    }

    # Patch to return no messages
    with patch("ai_dev_agent.cli.context_enhancer.get_context_enhancer") as mock_get_enhancer:
        mock_enhancer = Mock()
        mock_enhancer.get_repomap_messages.return_value = ("Broad query with no symbols", None)
        mock_get_enhancer.return_value = mock_enhancer

        # Call should not raise exception
        provider._inject_repomap_update()

    # Verify no messages were added to session
    session = session_manager.get_session(session_id)
    with session.lock:
        # Should have no messages since repomap returned None
        assert len(session.history) == 0
