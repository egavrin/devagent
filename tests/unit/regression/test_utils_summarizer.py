"""Tests for conversation summarization utilities."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from ai_dev_agent.core.utils.constants import LLM_DEFAULT_TEMPERATURE
from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session.summarizer import LLMConversationSummarizer, SummarizationConfig


class StubLLM:
    """Stub LLM client that records calls."""

    def __init__(self, response: str = "summary text") -> None:
        self.response = response
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(messages)
        return self.response


def _make_messages(count: int, *, content: str = "data") -> list[Message]:
    return [Message(role="user", content=f"{content} {i}") for i in range(count)]


def test_summarize_if_needed_below_threshold_returns_original() -> None:
    """Small conversations should bypass summarization."""
    llm = StubLLM()
    summarizer = LLMConversationSummarizer(
        llm, config=SummarizationConfig(min_messages_to_summarize=5)
    )

    messages = _make_messages(3)
    result = summarizer.summarize_if_needed(messages, target_tokens=10_000)

    assert result == messages
    assert llm.calls == []


def test_summarize_if_needed_invokes_llm_when_over_budget() -> None:
    """Long transcripts trigger recursive summarization via the LLM."""
    llm = StubLLM(response="short summary")
    # Low thresholds to force summarization
    config = SummarizationConfig(min_messages_to_summarize=2, summary_max_tokens=50)
    summarizer = LLMConversationSummarizer(llm, config=config)

    messages = _make_messages(6, content="long message" * 50)
    result = summarizer.summarize_if_needed(messages, target_tokens=20)

    assert result[0].role == "assistant"
    assert "summary" in result[0].content.lower()
    assert llm.calls, "LLM should be invoked during summarization"


def test_create_summary_is_cached_for_identical_inputs() -> None:
    """Internal cache should prevent duplicate LLM calls."""
    llm = StubLLM(response="cached summary")
    summarizer = LLMConversationSummarizer(llm, config=SummarizationConfig())

    messages = _make_messages(3, content="cache" * 10)
    first = summarizer._create_summary(messages)  # - exercising cache path
    second = summarizer._create_summary(messages)

    assert first == "cached summary"
    assert second == "cached summary"
    assert len(llm.calls) == 1


def test_two_tier_summarizer_truncates_tool_outputs() -> None:
    """The two-tier strategy should prune large tool outputs before summarizing."""
    llm = StubLLM(response="combined summary")
    config = SummarizationConfig(
        min_messages_to_summarize=2,
        prune_threshold=10,
        protect_recent=0,
    )
    summarizer = LLMConversationSummarizer(llm, config=config)

    tool_message = Message(role="tool", content="X" * 800, tool_call_id="call-1")
    messages = [tool_message, *_make_messages(3, content="payload" * 10)]

    pruned = summarizer._prune_old_tool_outputs(  # - targeted behavior check
        messages,
        summarizer._simple_token_estimate,
    )
    assert "[... tool output truncated ...]" in pruned[0].content


@pytest.mark.asyncio
async def test_summarize_async_delegates_to_sync_path() -> None:
    """Async wrapper should reuse the synchronous summarization path."""
    llm = StubLLM(response="async summary")
    summarizer = LLMConversationSummarizer(
        llm, config=SummarizationConfig(min_messages_to_summarize=2)
    )
    messages = _make_messages(4, content="async" * 30)

    result = await summarizer.summarize_async(messages, target_tokens=50)

    assert result != messages
    assert llm.calls  # ensure LLM was invoked
