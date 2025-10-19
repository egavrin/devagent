from uuid import uuid4

from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session import SessionManager, build_system_messages


def test_session_manager_composes_messages():
    manager = SessionManager.get_instance()
    session_id = f"test-session-{uuid4()}"
    session = manager.ensure_session(session_id, system_messages=build_system_messages(iteration_cap=10))
    manager.extend_history(session.id, [Message(role="user", content="Hello")])
    manager.add_system_message(session.id, "Note", location="system")
    manager.add_tool_message(session.id, "tool-1", "result")

    composed = manager.compose(session.id)
    assert composed[0].role == "system"
    assert composed[1].role == "system"
    assert any(message.role == "tool" for message in composed)


def test_remove_system_messages():
    manager = SessionManager.get_instance()
    session_id = f"test-session-remove-{uuid4()}"
    session = manager.ensure_session(session_id)
    manager.add_system_message(session.id, "Keep this", location="system")
    manager.add_system_message(session.id, "Tool performance this session:", location="system")

    manager.remove_system_messages(session.id, lambda msg: msg.content and msg.content.startswith("Tool performance"))
    remaining = [msg for msg in manager.compose(session.id) if msg.role == "system"]
    assert remaining and all(not msg.content.startswith("Tool performance") for msg in remaining)


def test_add_tool_message_generates_identifier_when_missing():
    manager = SessionManager.get_instance()
    session_id = f"test-session-tool-id-{uuid4()}"
    session = manager.ensure_session(session_id)

    message = manager.add_tool_message(session.id, None, "tool output placeholder")

    assert message.tool_call_id, "Expected tool_call_id to be generated when missing"
    composed = manager.compose(session.id)
    tool_entries = [msg for msg in composed if msg.role == "tool"]
    assert tool_entries and all(msg.tool_call_id for msg in tool_entries)
