"""Tests for shell conversation history management in the interactive query flow."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace

import click

from ai_dev_agent.cli.handlers import registry_handlers
from ai_dev_agent.cli.react.executor import _execute_react_assistant
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.providers.llm.base import Message, ToolCall, ToolCallResult
from ai_dev_agent.session import SessionManager


class FakeClient:
    """Minimal LLM client stub that records messages passed to invoke_tools."""

    def __init__(self, responses: Iterable[ToolCallResult]) -> None:
        self._responses: list[ToolCallResult] = list(responses)
        self.last_messages: list[Message] | None = None
        self.invocations: list[list[Message]] = []

    def invoke_tools(
        self,
        messages: Iterable[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = "auto",
        extra_headers: dict | None = None,
        parallel_tool_calls: bool = True,
    ) -> ToolCallResult:
        captured = list(messages)
        self.last_messages = captured
        self.invocations.append(captured)
        if self._responses:
            return self._responses.pop(0)
        return ToolCallResult(calls=[], message_content="", raw_tool_calls=None)

    def complete(
        self,
        messages: Iterable[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        """Complete method for forced synthesis."""
        captured = list(messages)
        self.last_messages = captured
        self.invocations.append(captured)
        if self._responses:
            result = self._responses.pop(0)
            return result.message_content or ""
        return ""


def _make_context(settings: Settings) -> click.Context:
    ctx = click.Context(click.Command("devagent"))
    ctx.obj = {
        "settings": settings,
        "_shell_conversation_history": [],
        "_shell_session_manager": None,
        "_shell_session_id": None,
        "_structure_hints_state": {"symbols": set(), "files": {}, "project_summary": None},
        "_detected_language": "python",
        "_repo_file_count": 5,
        "_project_structure_summary": "Stub structure",
        "devagent_config": SimpleNamespace(react_iteration_global_cap=None),
    }
    return ctx


def test_shell_conversation_history_persists_between_queries(capsys) -> None:
    settings = Settings()
    settings.keep_last_assistant_messages = 4

    client = FakeClient(
        [
            # First query - LLM returns text without tool calls, triggering StopIteration
            # The executor now uses this response directly instead of forcing synthesis
            ToolCallResult(calls=[], message_content="There are 1,369 files.", raw_tool_calls=None),
            # Second query - same behavior
            ToolCallResult(
                calls=[],
                message_content="72,650 is higher than 1,369.",
                raw_tool_calls=None,
            ),
        ]
    )
    ctx = _make_context(settings)

    _execute_react_assistant(
        ctx, client, settings, "How many files are in this project?", use_planning=False
    )

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 2
    assert history[0].role == "user"
    assert "How many files" in history[0].content
    assert history[1].role == "assistant"
    assert "1,369" in (history[1].content or "")

    _execute_react_assistant(ctx, client, settings, "Which number is higher?", use_planning=False)

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 4
    assistant_responses = [msg.content for msg in history if msg.role == "assistant"]
    assert "72,650 is higher" in assistant_responses[-1]

    assert client.last_messages is not None
    user_contents = [msg.content for msg in client.last_messages if msg.role == "user"]
    assert any("How many files" in content for content in user_contents)
    assert any("Which number is higher" in content for content in user_contents)


def test_shell_conversation_history_respects_limit(capsys) -> None:
    settings = Settings()
    settings.keep_last_assistant_messages = 1

    client = FakeClient(
        [
            # First query - no forced synthesis needed with the fix
            ToolCallResult(calls=[], message_content="First answer", raw_tool_calls=None),
            # Second query
            ToolCallResult(calls=[], message_content="Second answer", raw_tool_calls=None),
            # Third query
            ToolCallResult(calls=[], message_content="Third answer", raw_tool_calls=None),
        ]
    )
    ctx = _make_context(settings)

    _execute_react_assistant(ctx, client, settings, "Question one?", use_planning=False)
    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 2

    _execute_react_assistant(ctx, client, settings, "Question two?", use_planning=False)
    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 2
    assert history[0].content == "Question two?"
    assert history[1].content == "Second answer"

    _execute_react_assistant(ctx, client, settings, "Question three?", use_planning=False)
    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 2
    assert history[0].content == "Question three?"
    assert history[1].content == "Third answer"


def test_shell_history_ignores_tool_intermediate_assistant(monkeypatch, capsys) -> None:
    settings = Settings()
    settings.keep_last_assistant_messages = 4

    client = FakeClient(
        [
            # First query - tool call
            ToolCallResult(
                calls=[ToolCall(name="fake_tool", arguments={}, call_id="call-1")],
                message_content="Calling tool",
                raw_tool_calls=[{"id": "call-1", "type": "function"}],
            ),
            # After tool execution, synthesis
            ToolCallResult(
                calls=[],
                message_content="Tool result summarised.",
                raw_tool_calls=None,
            ),
            # Post-loop synthesis (called via complete() since last tool wasn't submit_final_answer)
            ToolCallResult(
                calls=[],
                message_content="Tool result summarised.",
                raw_tool_calls=None,
            ),
            # Second query
            ToolCallResult(
                calls=[],
                message_content="Follow-up answer using earlier info.",
                raw_tool_calls=None,
            ),
            # Post-loop synthesis for second query
            ToolCallResult(
                calls=[],
                message_content="Follow-up answer using earlier info.",
                raw_tool_calls=None,
            ),
        ]
    )

    ctx = _make_context(settings)

    def fake_handler(_ctx, _arguments):
        print("tool executed")

    monkeypatch.setitem(registry_handlers.INTENT_HANDLERS, "fake_tool", fake_handler)

    _execute_react_assistant(ctx, client, settings, "Run fake tool", use_planning=False)

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assistant_messages = [msg.content for msg in history if msg.role == "assistant"]
    assert assistant_messages == ["Tool result summarised."]

    _execute_react_assistant(ctx, client, settings, "Summarise result", use_planning=False)

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assistant_messages = [msg.content for msg in history if msg.role == "assistant"]
    assert assistant_messages == [
        "Tool result summarised.",
        "Follow-up answer using earlier info.",
    ]

    # Ensure both user prompts are present in payload of final invocation
    user_prompts = [msg.content for msg in client.invocations[-1] if msg.role == "user"]
    assert any("Run fake tool" in prompt for prompt in user_prompts)
    assert any("Summarise result" in prompt for prompt in user_prompts)


def test_shell_history_records_user_without_response(capsys) -> None:
    settings = Settings()
    client = FakeClient([ToolCallResult(calls=[], message_content=None, raw_tool_calls=None)])
    ctx = _make_context(settings)

    _execute_react_assistant(ctx, client, settings, "No answer?", use_planning=False)

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "No answer?"


def test_adds_dummy_tool_message_for_raw_tool_calls(capsys) -> None:
    settings = Settings()
    raw_tool_calls = [
        {
            "id": "call-missing-func",
            "type": "function",
            "function": {"name": "non_existent", "arguments": "{}"},
        }
    ]
    client = FakeClient(
        [
            ToolCallResult(
                calls=[],
                message_content="Fallback response",
                raw_tool_calls=raw_tool_calls,
            )
        ]
    )

    ctx = _make_context(settings)

    _execute_react_assistant(ctx, client, settings, "Trigger tool fallback", use_planning=False)

    session_id = ctx.obj.get("_session_id")
    assert session_id, "Session ID should be recorded on context"

    session_manager = SessionManager.get_instance()
    session = session_manager.get_session(str(session_id))
    tool_messages = [msg for msg in session.history if msg.role == "tool"]

    assert any(
        msg.tool_call_id == "call-missing-func" and "not executed" in (msg.content or "")
        for msg in tool_messages
    ), "Expected a dummy tool response tied to the raw tool call identifier"


def test_generates_tool_call_id_when_provider_omits_id(monkeypatch, capsys) -> None:
    settings = Settings()
    raw_tool_calls = [
        {
            "type": "function",
            "function": {"name": "fake_tool_missing_id", "arguments": "{}"},
        }
    ]
    client = FakeClient(
        [
            ToolCallResult(
                calls=[ToolCall(name="fake_tool_missing_id", arguments={}, call_id=None)],
                message_content="Requesting tool",
                raw_tool_calls=raw_tool_calls,
            ),
            ToolCallResult(calls=[], message_content="Tool completed", raw_tool_calls=None),
        ]
    )

    ctx = _make_context(settings)

    def fake_handler(_ctx, _arguments):
        print("fake tool executed")

    monkeypatch.setitem(registry_handlers.INTENT_HANDLERS, "fake_tool_missing_id", fake_handler)

    _execute_react_assistant(ctx, client, settings, "Run missing-id tool", use_planning=False)

    session_id = ctx.obj.get("_session_id")
    assert session_id

    session_manager = SessionManager.get_instance()
    session = session_manager.get_session(str(session_id))

    tool_messages = [msg for msg in session.history if msg.role == "tool"]
    assert tool_messages, "Expected at least one tool message"
    call_id = tool_messages[0].tool_call_id
    assert call_id, "Generated tool call id should not be falsy"

    assistant_tool_messages = [
        msg for msg in session.history if msg.role == "assistant" and msg.tool_calls
    ]
    assert assistant_tool_messages, "Assistant should record tool call metadata"
    first_tool_call = assistant_tool_messages[0].tool_calls[0]
    assert first_tool_call.get("id") == call_id
