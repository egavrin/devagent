import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ai_dev_agent.core.utils import context_budget
from ai_dev_agent.core.utils.context_budget import (
    BudgetedLLMClient,
    ContextBudgetConfig,
    prune_messages,
)
from ai_dev_agent.providers.llm.base import Message


class DummyInnerClient:
    def __init__(self):
        self.retry_config = None
        self.timeout = None
        self.calls = []

    def configure_retry(self, retry_config):
        self.retry_config = retry_config

    def configure_timeout(self, timeout):
        self.timeout = timeout

    def complete(self, messages, **kwargs):
        self.calls.append(("complete", messages, kwargs))
        return "complete-result"

    def stream(self, messages, **kwargs):
        self.calls.append(("stream", messages, kwargs))
        return iter(["chunk"])

    def invoke_tools(self, messages, tools, **kwargs):
        self.calls.append(("invoke_tools", messages, tools, kwargs))
        return SimpleNamespace(status="ok")

    def extra_attribute(self):
        return "inner-value"


def make_message(role, content, **extra):
    return Message(role=role, content=content, **extra)


def test_accurate_token_count_uses_tiktoken(monkeypatch):
    class FakeEncoding:
        def encode(self, text):
            return list(text)

    fake_module = ModuleType("tiktoken")

    def fake_get_encoding(name):
        return FakeEncoding()

    fake_module.get_encoding = fake_get_encoding

    monkeypatch.setitem(sys.modules, "tiktoken", fake_module)

    messages = [make_message("user", "hello world")]

    result = context_budget._accurate_token_count(messages, model="gpt-4o")

    # 11 encoded tokens + 4 structural tokens
    assert result == 15


def test_accurate_token_count_uses_litellm_fallback(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("no tiktoken")
        return original_import(name, *args, **kwargs)

    class FakeLitellmModule:
        def token_counter(self, *, model, messages):
            assert model == "gpt-x"
            assert messages[0]["role"] == "assistant"
            assert "tool_calls" in messages[0]
            return 123

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "litellm", FakeLitellmModule())

    tool_message = make_message(
        "assistant",
        "result",
        tool_calls=[{"id": "tool-1", "type": "function", "function": {"name": "lookup"}}],
    )

    result = context_budget._accurate_token_count([tool_message], model="gpt-x")

    assert result == 123


def test_accurate_token_count_raises_when_backends_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"tiktoken", "litellm"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(Exception, match="Accurate token counting not available"):
        context_budget._accurate_token_count([make_message("user", "noop")], model="other")


def test_prune_messages_handles_empty_sequences():
    config = ContextBudgetConfig(max_tokens=100, headroom_tokens=0)
    assert prune_messages([], config) == []


def test_prune_messages_truncates_each_role_branch():
    messages = [
        make_message("system", "S1" * 50),
        make_message("system", "S2" * 50),
        make_message("tool", "TOOL" * 120, tool_call_id="call-1"),
        make_message("user", "U" * 200),
        make_message("assistant", "A" * 200, tool_calls=[{"id": "call-1"}]),
    ]
    config = ContextBudgetConfig(
        max_tokens=0,
        headroom_tokens=0,
        max_tool_messages=0,
        max_tool_output_chars=5,
        keep_last_assistant=0,
        enable_two_tier=False,
    )

    pruned = prune_messages(messages, config)

    assert pruned[0].content == messages[0].content
    assert pruned[1].content == "[context truncated]"
    assert pruned[2].content == "[tool output truncated]"
    assert pruned[3].content == messages[3].content
    assert pruned[4].content == "[context truncated]"
    # Ensure tool call IDs are preserved when truncating
    assert pruned[2].tool_call_id == "call-1"
    assert pruned[4].tool_calls == [{"id": "call-1"}]


def test_budgeted_client_passthrough_and_configuration(monkeypatch):
    inner = DummyInnerClient()
    config = ContextBudgetConfig(max_tokens=10, headroom_tokens=0)
    client = BudgetedLLMClient(inner, config=config)

    assert client.inner is inner

    client.configure_retry({"retries": 2})
    client.configure_timeout(9.5)
    assert inner.retry_config == {"retries": 2}
    assert inner.timeout == 9.5

    client.disable()
    original_messages = [make_message("user", "hello")]
    prepared_disabled = client._prepare_messages(original_messages)
    assert prepared_disabled[0] is original_messages[0]

    client.enable()
    client.set_config(ContextBudgetConfig(max_tokens=5, headroom_tokens=0))

    result = client.complete(
        original_messages,
        temperature=0.3,
        max_tokens=20,
        extra_headers={"x": "1"},
        response_format={"type": "json"},
    )
    assert result == "complete-result"

    stream_iter = client.stream(
        original_messages,
        temperature=0.4,
        max_tokens=25,
        extra_headers={"y": "2"},
        hooks={"on_token": lambda _: None},
    )
    assert list(stream_iter) == ["chunk"]

    tools = [{"type": "function", "function": {"name": "do"}}]
    call_result = client.invoke_tools(
        original_messages,
        tools,
        temperature=0.1,
        max_tokens=15,
        tool_choice="required",
        extra_headers={"z": "3"},
        response_format={"type": "structured"},
        parallel_tool_calls=False,
    )
    assert call_result.status == "ok"

    assert client.inner.extra_attribute() == "inner-value"
    assert inner.calls[0][0] == "complete"
    assert inner.calls[1][0] == "stream"
    assert inner.calls[2][0] == "invoke_tools"
