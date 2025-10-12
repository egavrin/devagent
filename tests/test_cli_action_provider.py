from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

from ai_dev_agent.cli.react.action_provider import LLMActionProvider
from ai_dev_agent.engine.react.types import TaskSpec
from ai_dev_agent.providers.llm.base import ToolCall, ToolCallResult
from ai_dev_agent.session import SessionManager


@dataclass
class StubLLMClient:
    model: str = "stub-model"
    last_tools: Optional[List[Dict[str, Any]]] = None

    def invoke_tools(self, messages, *, tools, temperature: float = 0.1, **kwargs):
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
        def invoke_tools(self, messages, *, tools, temperature: float = 0.1, **kwargs):
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
