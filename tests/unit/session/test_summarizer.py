"""Unit tests for conversation summarizers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.providers.llm.base import LLMError, Message
from ai_dev_agent.session.summarizer import (
    LLMConversationSummarizer,
    SummarizationConfig,
    create_summarizer,
)


class TestLLMConversationSummarizer:
    """Test LLMConversationSummarizer - no heuristic fallback."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return MagicMock()

    @pytest.fixture
    def summarizer(self, mock_llm_client):
        """Create a summarizer with mock client."""
        return LLMConversationSummarizer(mock_llm_client)

    def test_summarize_empty_messages_returns_empty_string(self, summarizer):
        """Test that summarizing empty messages returns empty string."""
        messages = []

        result = summarizer.summarize(messages, max_chars=100)

        assert result == ""

    def test_summarize_messages_without_content_returns_empty_string(self, summarizer):
        """Test that messages without content return empty string."""
        messages = [
            Message(role="assistant", content=""),
            Message(role="user", content=None),
        ]

        result = summarizer.summarize(messages, max_chars=100)

        assert result == ""

    def test_successful_llm_summarization(self, mock_llm_client):
        """Test successful LLM-based summarization."""
        mock_llm_client.complete.return_value = (
            "User asked about weather. Assistant provided forecast."
        )
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [
            Message(role="user", content="What's the weather?"),
            Message(role="assistant", content="It will be sunny."),
        ]

        result = summarizer.summarize(messages, max_chars=200)

        assert "weather" in result.lower()
        mock_llm_client.complete.assert_called_once()

        # Verify the prompt structure
        call_args = mock_llm_client.complete.call_args
        prompt_messages = call_args[0][0]
        assert len(prompt_messages) == 2  # system + user
        assert prompt_messages[0].role == "system"
        assert prompt_messages[1].role == "user"

    def test_llm_error_propagates_no_fallback(self, mock_llm_client):
        """Test that LLM errors propagate (fail fast, no fallback)."""
        mock_llm_client.complete.side_effect = LLMError("API error")
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [Message(role="user", content="Hello")]

        # Should raise the exception, not fall back
        with pytest.raises(LLMError, match="API error"):
            summarizer.summarize(messages, max_chars=100)

    def test_empty_llm_response_raises_error(self, mock_llm_client):
        """Test that empty LLM response raises error (no fallback)."""
        mock_llm_client.complete.return_value = ""
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [Message(role="user", content="Hello")]

        # Should raise ValueError, not fall back
        with pytest.raises(ValueError, match="empty summary"):
            summarizer.summarize(messages, max_chars=100)

    def test_none_llm_response_raises_error(self, mock_llm_client):
        """Test that None LLM response raises error (no fallback)."""
        mock_llm_client.complete.return_value = None
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [Message(role="user", content="Hello")]

        # Should raise ValueError, not fall back
        with pytest.raises(ValueError, match="empty summary"):
            summarizer.summarize(messages, max_chars=100)

    def test_truncates_long_llm_response(self, mock_llm_client):
        """Test that long LLM response is truncated to max_chars."""
        long_summary = "x" * 1000
        mock_llm_client.complete.return_value = long_summary
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [Message(role="user", content="Hello")]

        result = summarizer.summarize(messages, max_chars=50)

        # Should be truncated to exactly 50 chars (including ...)
        assert len(result) == 50
        assert result.endswith("...")

    def test_custom_system_prompt(self, mock_llm_client):
        """Test using custom system prompt."""
        mock_llm_client.complete.return_value = "Summary"
        custom_system = "Custom system instructions"
        summarizer = LLMConversationSummarizer(
            mock_llm_client,
            system_prompt=custom_system,
        )
        messages = [Message(role="user", content="Hello")]

        _ = summarizer.summarize(messages, max_chars=100)

        call_args = mock_llm_client.complete.call_args
        prompt_messages = call_args[0][0]
        assert prompt_messages[0].content == custom_system

    def test_custom_user_template(self, mock_llm_client):
        """Test using custom user template."""
        mock_llm_client.complete.return_value = "Summary"
        custom_template = "Summarize this: {conversation} (max {max_chars} chars)"
        summarizer = LLMConversationSummarizer(
            mock_llm_client,
            user_template=custom_template,
        )
        messages = [Message(role="user", content="Hello")]

        _ = summarizer.summarize(messages, max_chars=50)

        call_args = mock_llm_client.complete.call_args
        prompt_messages = call_args[0][0]
        assert "Summarize this:" in prompt_messages[1].content
        assert "50" in prompt_messages[1].content

    def test_custom_max_tokens(self, mock_llm_client):
        """Test using custom max_tokens."""
        mock_llm_client.complete.return_value = "Summary"
        custom_max_tokens = 512
        summarizer = LLMConversationSummarizer(
            mock_llm_client,
            max_tokens=custom_max_tokens,
        )
        messages = [Message(role="user", content="Hello")]

        _ = summarizer.summarize(messages, max_chars=100)

        call_args = mock_llm_client.complete.call_args
        assert call_args[1]["max_tokens"] == custom_max_tokens

    def test_uses_temperature_from_config(self, mock_llm_client):
        """Test that temperature is taken from config."""
        mock_llm_client.complete.return_value = "Summary"
        config = SummarizationConfig(summary_temperature=0.5)
        summarizer = LLMConversationSummarizer(mock_llm_client, config=config)
        messages = [Message(role="user", content="Hello")]

        _ = summarizer.summarize(messages, max_chars=100)

        call_args = mock_llm_client.complete.call_args
        assert call_args[1]["temperature"] == 0.5

    def test_config_property(self, summarizer):
        """Test that config property returns the configuration."""
        assert isinstance(summarizer.config, SummarizationConfig)
        assert summarizer.config.max_history_tokens == 8192


class TestSummarizeIfNeeded:
    """Test summarize_if_needed method with recursive splitting."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.complete.return_value = "summary text"
        return client

    def _make_messages(self, count, prefix="msg"):
        """Helper to create messages."""
        return [Message(role="user", content=f"{prefix}-{i} " + "x" * 200) for i in range(count)]

    def test_returns_original_when_under_budget(self, mock_llm_client):
        """Test that messages are returned unchanged when under budget."""
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = self._make_messages(3)

        result = summarizer.summarize_if_needed(messages, target_tokens=10**6)

        assert result == messages
        mock_llm_client.complete.assert_not_called()

    def test_returns_original_when_too_few_messages(self, mock_llm_client):
        """Test that too few messages skip summarization."""
        config = SummarizationConfig(min_messages_to_summarize=5)
        summarizer = LLMConversationSummarizer(mock_llm_client, config=config)
        messages = self._make_messages(3)

        result = summarizer.summarize_if_needed(messages, target_tokens=1)

        assert result == messages
        mock_llm_client.complete.assert_not_called()

    def test_recursive_summarization(self, mock_llm_client):
        """Test recursive split-and-summarize strategy."""
        config = SummarizationConfig(
            max_history_tokens=200,
            min_messages_to_summarize=2,
            max_recursion_depth=2,
            summary_max_tokens=50,
        )
        summarizer = LLMConversationSummarizer(mock_llm_client, config=config)
        messages = self._make_messages(6, prefix="history")

        result = summarizer.summarize_if_needed(messages, target_tokens=50)

        # Should create summary messages
        assert result != messages
        # Check for either type of summary message format
        has_summary = any(
            "[Summary" in (msg.content or "") or "[Complete" in (msg.content or "")
            for msg in result
        )
        assert has_summary
        assert mock_llm_client.complete.called

    def test_summarize_all_when_max_depth_reached(self, mock_llm_client):
        """Test that all messages are summarized when max depth reached."""
        config = SummarizationConfig(max_recursion_depth=0, min_messages_to_summarize=1)
        summarizer = LLMConversationSummarizer(mock_llm_client, config=config)
        messages = self._make_messages(6)

        result = summarizer.summarize_if_needed(messages, target_tokens=1)

        assert result
        assert "Complete conversation summary" in result[0].content
        assert mock_llm_client.complete.called

    def test_summary_caching(self, mock_llm_client):
        """Test that summaries are cached to avoid redundant LLM calls."""
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = self._make_messages(4)

        # First call
        summarizer._create_summary(messages)
        first_call_count = mock_llm_client.complete.call_count

        # Second call with same messages - should use cache
        summarizer._create_summary(messages)
        assert mock_llm_client.complete.call_count == first_call_count


class TestTwoTierPruning:
    """Test two-tier pruning: cheap tool truncation before expensive summarization."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.complete.return_value = "summary"
        return client

    def test_tool_output_pruning(self, mock_llm_client):
        """Test that old tool outputs are truncated."""
        config = SummarizationConfig(
            min_messages_to_summarize=2,
            prune_threshold=10,
            protect_recent=20,
        )
        summarizer = LLMConversationSummarizer(mock_llm_client, config=config)

        tool_content = "tool output " + "y" * 600
        messages = [
            Message(role="tool", content=tool_content, tool_call_id="123"),
            Message(role="assistant", content="recent message"),
        ]

        pruned = summarizer.optimize_context(messages, target_tokens=100)

        # Old tool output should be truncated
        assert "[... tool output truncated ...]" in pruned[0].content
        assert pruned[0].tool_call_id == "123"
        # Recent message should be preserved
        assert pruned[1].content == "recent message"

    def test_two_tier_falls_back_to_summarization(self, mock_llm_client):
        """Test that two-tier falls back to summarization when pruning insufficient."""
        config = SummarizationConfig(
            min_messages_to_summarize=2,
            prune_threshold=10000,  # High threshold - won't save enough
            protect_recent=20,
        )
        summarizer = LLMConversationSummarizer(mock_llm_client, config=config)

        # Create messages that exceed budget even after pruning
        messages = [
            Message(role="user", content="x" * 1000),
            Message(role="assistant", content="y" * 1000),
            Message(role="user", content="z" * 1000),
        ]

        _ = summarizer.optimize_context(messages, target_tokens=100)

        # Should have called LLM for summarization
        assert mock_llm_client.complete.called


class TestAsyncSummarization:
    """Test async summarization support."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.complete.return_value = "async summary"
        return client

    @pytest.mark.asyncio
    async def test_summarize_async(self, mock_llm_client):
        """Test async summarization."""
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [
            Message(role="user", content="x" * 200),
            Message(role="assistant", content="y" * 200),
            Message(role="user", content="z" * 200),
            Message(role="assistant", content="a" * 200),
            Message(role="user", content="b" * 200),
        ]

        result = await summarizer.summarize_async(messages, target_tokens=10)

        assert isinstance(result, list)
        assert mock_llm_client.complete.called


class TestFactoryFunction:
    """Test factory function for creating summarizers."""

    def test_create_summarizer_returns_llm_summarizer(self):
        """Test that factory returns LLM summarizer."""
        llm = MagicMock()
        summarizer = create_summarizer(llm)

        assert isinstance(summarizer, LLMConversationSummarizer)
        assert summarizer._client is llm

    def test_create_summarizer_with_config(self):
        """Test factory with custom config."""
        llm = MagicMock()
        config = SummarizationConfig(max_history_tokens=4096)

        summarizer = create_summarizer(llm, config=config)

        assert summarizer.config.max_history_tokens == 4096

    def test_create_summarizer_with_kwargs(self):
        """Test factory with additional kwargs."""
        llm = MagicMock()

        summarizer = create_summarizer(llm, max_tokens=512)

        assert summarizer._max_tokens == 512


class TestTokenEstimation:
    """Test simple token estimation."""

    @pytest.fixture
    def summarizer(self):
        """Create a summarizer."""
        return LLMConversationSummarizer(MagicMock())

    def test_simple_token_estimate(self, summarizer):
        """Test simple token estimation (4 chars = 1 token)."""
        messages = [
            Message(role="user", content="x" * 400),  # ~100 tokens + overhead
            Message(role="assistant", content="y" * 400),  # ~100 tokens + overhead
        ]

        estimate = summarizer._simple_token_estimate(messages)

        # Should be around 200 tokens + overhead (2 * 8)
        assert estimate >= 200
        assert estimate <= 220

    def test_token_estimate_with_empty_content(self, summarizer):
        """Test token estimation with empty content."""
        messages = [
            Message(role="user", content=""),
            Message(role="assistant", content=None),
        ]

        estimate = summarizer._simple_token_estimate(messages)

        # Should just count overhead
        assert estimate == 16  # 2 messages * 8 overhead each


class TestCacheKey:
    """Test cache key generation."""

    @pytest.fixture
    def summarizer(self):
        """Create a summarizer."""
        return LLMConversationSummarizer(MagicMock())

    def test_cache_key_generation(self, summarizer):
        """Test that cache keys are generated from message content."""
        messages = [
            Message(role="user", content="Hello world"),
            Message(role="assistant", content="Hi there"),
        ]

        key = summarizer._get_cache_key(messages)

        assert isinstance(key, str)
        assert "Hello world" in key
        assert "Hi there" in key

    def test_cache_key_limits_to_first_five_messages(self, summarizer):
        """Test that cache key only uses first 5 messages."""
        messages = [Message(role="user", content=f"msg{i}") for i in range(10)]

        key = summarizer._get_cache_key(messages)

        # Should only contain first 5
        assert "msg0" in key
        assert "msg4" in key
        assert "msg5" not in key

    def test_cache_key_truncates_long_content(self, summarizer):
        """Test that cache key truncates long message content."""
        long_content = "x" * 1000
        messages = [Message(role="user", content=long_content)]

        key = summarizer._get_cache_key(messages)

        # Should be truncated to 50 chars
        assert len(key) == 50
