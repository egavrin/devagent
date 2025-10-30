import asyncio

import pytest

from ai_dev_agent.core.utils.summarizer import (
    ConversationSummarizer,
    SummarizationConfig,
    TwoTierSummarizer,
    create_summarizer,
)
from ai_dev_agent.providers.llm.base import Message


class DummyLLM:
    def __init__(self):
        self.calls = []
        self.summary_text = "summary"

    def complete(self, messages, temperature=0.2, max_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return f"{self.summary_text}-{len(self.calls)}"


def make_messages(count, prefix="msg"):
    return [Message(role="user", content=f"{prefix}-{i} " + "x" * 200) for i in range(count)]


def test_summarize_if_needed_recurses_and_caches(monkeypatch):
    llm = DummyLLM()
    config = SummarizationConfig(
        max_history_tokens=200,
        min_messages_to_summarize=2,
        max_recursion_depth=2,
        summary_max_tokens=50,
    )
    summarizer = ConversationSummarizer(llm, config=config)
    messages = make_messages(6, prefix="history")

    summarized = summarizer.summarize_if_needed(messages, target_tokens=50)
    assert summarized != messages
    assert summarized[0].content.startswith("[Summary") or summarized[0].content.startswith(
        "[Complete conversation summary]"
    )
    assert llm.calls  # ensure LLM invoked during summarization


def test_summarize_if_needed_returns_original_when_under_budget():
    llm = DummyLLM()
    summarizer = ConversationSummarizer(llm)
    messages = make_messages(3, prefix="short")

    result = summarizer.summarize_if_needed(messages, target_tokens=10**6)
    assert result == messages


def test_summarize_all_and_token_estimate_paths():
    llm = DummyLLM()
    config = SummarizationConfig(max_recursion_depth=0, min_messages_to_summarize=1)
    summarizer = ConversationSummarizer(llm, config=config)

    messages = make_messages(6)
    summarized = summarizer.summarize_if_needed(messages, target_tokens=1)
    assert summarized and "Complete conversation summary" in summarized[0].content

    manual_estimate = summarizer._simple_token_estimate(messages)
    assert manual_estimate > 0

    cache_key = summarizer._get_cache_key(messages)
    assert isinstance(cache_key, str) and cache_key

    optimized = summarizer.optimize_context(messages, target_tokens=1)
    assert optimized


def test_create_summary_caches_results():
    llm = DummyLLM()
    summarizer = ConversationSummarizer(llm)
    messages = make_messages(4)

    summarizer._create_summary(messages)
    first_calls = len(llm.calls)
    summarizer._create_summary(messages)
    assert len(llm.calls) == first_calls


@pytest.mark.asyncio
async def test_summarize_async(monkeypatch):
    llm = DummyLLM()
    summarizer = ConversationSummarizer(llm)
    messages = make_messages(5)

    result = await summarizer.summarize_async(messages, target_tokens=10)
    assert isinstance(result, list)
    assert llm.calls  # ensure LLM invoked


def test_two_tier_prunes_tool_outputs():
    llm = DummyLLM()
    config = SummarizationConfig(min_messages_to_summarize=2)
    summarizer = TwoTierSummarizer(llm, config=config, prune_threshold=10, protect_recent=20)

    tool_content = "tool output " + "y" * 600
    messages = [
        Message(role="tool", content=tool_content, tool_call_id="123"),
        Message(role="assistant", content="recent message"),
    ]

    pruned = summarizer.optimize_context(messages, target_tokens=100)
    assert "[... tool output truncated ...]" in pruned[0].content
    assert pruned[0].tool_call_id == "123"


def test_create_summarizer_factory():
    llm = DummyLLM()
    assert isinstance(create_summarizer(llm, two_tier=True), TwoTierSummarizer)
    assert isinstance(create_summarizer(llm, two_tier=False), ConversationSummarizer)
