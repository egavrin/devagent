from ai_dev_agent.core.utils import context_budget
from ai_dev_agent.core.utils.context_budget import (
    BudgetedLLMClient,
    ContextBudgetConfig,
    ensure_context_budget,
    estimate_tokens,
    prune_messages,
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


def test_estimate_tokens_prefers_accurate_counter(monkeypatch):
    messages = [Message(role="user", content="hello world")]

    called = {}

    def fake_counter(msgs, model_name):
        called["args"] = (msgs, model_name)
        return 321

    monkeypatch.setattr(context_budget, "_accurate_token_count", fake_counter)

    result = context_budget.estimate_tokens(messages, model="gpt-4o")

    assert result == 321
    assert called["args"][0] == messages
    assert called["args"][1] == "gpt-4o"


def test_estimate_tokens_falls_back_to_heuristic(monkeypatch):
    def boom(*_, **__):
        raise RuntimeError("no counter")

    monkeypatch.setattr(context_budget, "_accurate_token_count", boom)

    message = Message(role="assistant", content="abcd" * 10)
    heuristic = context_budget.estimate_tokens([message], model="gpt-4")

    # 40 characters => 10 tokens plus base allowance
    assert heuristic == (len(message.content) // 4) + 8


def test_two_tier_prune_truncates_large_tool_outputs():
    messages = [
        Message(role="system", content="boot"),
        Message(role="tool", content="x" * 600),
        Message(role="tool", content="y" * 600),
        Message(role="assistant", content="ok"),
    ]
    config = ContextBudgetConfig(
        max_tokens=250,
        headroom_tokens=0,
        max_tool_messages=5,
        max_tool_output_chars=40,
        keep_last_assistant=1,
        enable_two_tier=True,
        prune_protect_tokens=0,
        prune_minimum_savings=1,
    )

    pruned = prune_messages(messages, config)

    tool_contents = [msg.content or "" for msg in pruned if msg.role == "tool"]
    assert any("[... tool output pruned for context ...]" in content for content in tool_contents)
    assert estimate_tokens(pruned) <= config.max_tokens


def test_prune_messages_respects_keep_last_assistant_zero():
    messages = [
        Message(role="user", content="ask"),
        Message(role="assistant", content="answer" * 100),
        Message(role="assistant", content="more" * 100),
    ]
    config = ContextBudgetConfig(
        max_tokens=120,
        headroom_tokens=0,
        max_tool_messages=0,
        max_tool_output_chars=10,
        keep_last_assistant=0,
        enable_two_tier=False,
    )

    pruned = prune_messages(messages, config)

    assistant_contents = [msg.content for msg in pruned if msg.role == "assistant"]
    assert assistant_contents  # assistants remain but should be truncated
    assert all(content == "[context truncated]" for content in assistant_contents)
