"""Tests for session data structures and manager utilities."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session.manager import SessionManager
from ai_dev_agent.session.models import Session


@pytest.fixture(autouse=True)
def reset_session_manager():
    manager = SessionManager.get_instance()
    manager._sessions.clear()
    manager._max_sessions = SessionManager.DEFAULT_MAX_SESSIONS
    yield
    manager._sessions.clear()
    manager._max_sessions = SessionManager.DEFAULT_MAX_SESSIONS


def test_session_compose_merges_system_and_history():
    system_messages = [
        Message(role="system", content="policy"),
        Message(role="system", content="context"),
    ]
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]

    session = Session(id="session-1", system_messages=system_messages, history=history)

    composed = session.compose()

    assert composed[:2] == system_messages
    assert composed[2:] == history


def test_session_manager_creates_and_updates_sessions(monkeypatch):
    manager = SessionManager.get_instance()

    session = manager.ensure_session("test-session", metadata={"topic": "demo"})
    assert session.metadata["topic"] == "demo"

    manager.add_user_message("test-session", "Explain the build")
    manager.add_assistant_message("test-session", "Build completed")

    composed = manager.compose("test-session")
    assert composed[-2].role == "user"
    assert composed[-1].role == "assistant"


def test_session_manager_tool_messages_generate_ids():
    manager = SessionManager.get_instance()
    manager.ensure_session("tools-session")

    tool_message = manager.add_tool_message("tools-session", None, "output")
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id is not None

    # Reusing generated id should not create duplicates
    tool_message_2 = manager.add_tool_message("tools-session", tool_message.tool_call_id, "more")
    assert tool_message_2.tool_call_id == tool_message.tool_call_id


def test_session_manager_prunes_old_sessions(monkeypatch):
    manager = SessionManager.get_instance()
    session = manager.ensure_session("ttl-session")

    old_time = datetime.now() - timedelta(hours=1)
    session.last_accessed = old_time

    # Force eviction logic by exceeding max sessions
    manager._max_sessions = 1
    manager.ensure_session("new-session")

    assert manager.has_session("new-session")
