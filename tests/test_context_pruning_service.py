from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session import SessionManager
from ai_dev_agent.session.context_service import ContextPruningConfig, ContextPruningService
from ai_dev_agent.session.models import Session


class _StubSummarizer:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def summarize(self, messages, *, max_chars: int):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.response[:max_chars]


def _make_heavy_messages(count: int) -> list[Message]:
    payload = "x" * 400  # 100 token-equivalent characters per message
    messages: list[Message] = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append(Message(role=role, content=f"{role}::{payload}"))
    return messages


def test_pruning_service_inserts_summary() -> None:
    session = Session(id="summarize-demo")
    with session.lock:
        session.history.extend(_make_heavy_messages(12))

    service = ContextPruningService(
        ContextPruningConfig(
            max_total_tokens=600,
            trigger_tokens=400,
            keep_recent_messages=4,
            summary_max_chars=200,
            max_event_history=4,
        )
    )

    service.update_session(session)

    with session.lock:
        history = list(session.history)
        assert history, "Expected conversation history after pruning"
        assert any(
            msg.content and msg.content.startswith("[Context summary]")
            for msg in history
        ), "Summarized message not found"
        metadata = session.metadata.get("context_service", {})
        assert metadata.get("events"), "Pruning event metadata missing"
        final_tokens = metadata.get("token_estimate")
        assert isinstance(final_tokens, int) and final_tokens <= 600


def test_session_manager_uses_pruning_service() -> None:
    manager = SessionManager.get_instance()
    config = ContextPruningConfig(
        max_total_tokens=600,
        trigger_tokens=400,
        keep_recent_messages=4,
        summary_max_chars=160,
        max_event_history=4,
    )

    manager.configure_context_service(config)
    try:
        session = manager.ensure_session()

        for message in _make_heavy_messages(10):
            if message.role == "user":
                manager.add_user_message(session.id, message.content or "")
            else:
                manager.add_assistant_message(session.id, message.content or "")

        with session.lock:
            metadata = session.metadata.get("context_service")
            assert metadata is not None, "Context service metadata should be populated"
            assert metadata.get("events"), "Expected pruning events recorded"
    finally:
        manager.configure_context_service()


def test_pruning_preserves_tool_parent_pairs() -> None:
    session = Session(id="tool-parent")
    with session.lock:
        session.history.extend(
            [
                Message(role="user", content="step one"),
                Message(role="assistant", content="Preparing", tool_calls=[{"id": "call-1"}]),
                Message(role="tool", content="result", tool_call_id="call-1"),
                Message(role="user", content="follow up"),
                Message(role="assistant", content="answer"),
            ]
        )

    service = ContextPruningService(
        ContextPruningConfig(
            max_total_tokens=300,
            trigger_tokens=200,
            keep_recent_messages=4,
            summary_max_chars=120,
        )
    )

    service.update_session(session)

    with session.lock:
        assistant_with_tools = None
        for msg in session.history:
            if msg.role == "assistant" and msg.tool_calls:
                assistant_with_tools = msg
            if msg.role == "tool":
                assert assistant_with_tools is not None


def test_orphan_tool_message_removed() -> None:
    service = ContextPruningService(
        ContextPruningConfig(max_total_tokens=200, trigger_tokens=50, keep_recent_messages=2)
    )
    orphan_messages = [Message(role="tool", content="data", tool_call_id="missing")]

    sanitized = service._sanitize_tool_sequences(orphan_messages)
    assert not sanitized


def test_custom_summarizer_is_used() -> None:
    session = Session(id="llm-summary")
    with session.lock:
        session.history.extend(_make_heavy_messages(8))

    stub = _StubSummarizer("Important summary from LLM")
    service = ContextPruningService(
        ContextPruningConfig(
            max_total_tokens=600,
            trigger_tokens=400,
            keep_recent_messages=4,
            summary_max_chars=120,
        ),
        summarizer=stub,
    )

    service.update_session(session)

    assert stub.calls >= 1
    with session.lock:
        summary_messages = [
            msg.content or ""
            for msg in session.history
            if msg.content and msg.content.startswith("[Context summary]")
        ]
        assert summary_messages, "Expected summary message"
        assert "Important summary from LLM" in summary_messages[0]
