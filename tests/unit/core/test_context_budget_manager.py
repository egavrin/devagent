from ai_dev_agent.core.utils.context_budget import (
    BudgetedLLMClient,
    ContextBudgetConfig,
    ensure_context_budget,
    summarize_text,
)
from ai_dev_agent.providers.llm.base import Message


def test_summarize_text_truncates_and_marks_omission():
    original = "x" * 50
    result = summarize_text(original, 10)
    assert result.startswith("x" * 10)
    assert "omitted" in result


def test_ensure_context_budget_retains_key_messages():
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="question"),
    ]
    for idx in range(5):
        messages.append(Message(role="tool", content="data" * 200))
    config = ContextBudgetConfig(
        max_tokens=120, headroom_tokens=0, max_tool_messages=2, max_tool_output_chars=50
    )

    pruned = ensure_context_budget(messages, config)

    # System and user messages should be present
    roles = [msg.role for msg in pruned]
    assert roles.count("system") >= 1
    assert roles.count("user") >= 1

    # Older tool outputs should be summarized
    tool_contents = [msg.content for msg in pruned if msg.role == "tool"]
    assert any("omitted" in content for content in tool_contents)


def test_budgeted_client_applies_pruning():
    captured = {}

    class DummyClient:
        def complete(self, messages, **kwargs):
            captured["messages"] = list(messages)
            return "ok"

    inner = DummyClient()
    config = ContextBudgetConfig(
        max_tokens=80, headroom_tokens=0, max_tool_messages=1, max_tool_output_chars=40
    )
    client = BudgetedLLMClient(inner, config=config)
    msgs = [
        Message(role="system", content="system"),
        Message(role="user", content="ask"),
        Message(role="tool", content="tool" * 200),
        Message(role="tool", content="other" * 200),
    ]

    client.complete(msgs)

    forwarded = captured["messages"]
    assert len(forwarded) <= len(msgs)
    assert any("omitted" in (msg.content or "") for msg in forwarded if msg.role == "tool")
