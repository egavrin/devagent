"""Unit tests for conversation summarizers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai_dev_agent.providers.llm.base import LLMError, Message
from ai_dev_agent.session.summarizer import (
    HeuristicConversationSummarizer,
    LLMConversationSummarizer,
)


class TestHeuristicConversationSummarizer:
    """Test HeuristicConversationSummarizer."""

    def test_summarize_empty_messages(self):
        """Test that summarizing empty messages returns fallback."""
        summarizer = HeuristicConversationSummarizer()
        messages = []

        result = summarizer.summarize(messages, max_chars=100)

        assert result == "(no additional context retained)"

    def test_summarize_messages_without_content(self):
        """Test that messages without content are skipped."""
        summarizer = HeuristicConversationSummarizer()
        messages = [
            Message(role="assistant", content=""),
            Message(role="user", content=None),
        ]

        result = summarizer.summarize(messages, max_chars=100)

        assert result == "(no additional context retained)"

    def test_summarize_with_content(self):
        """Test that messages with content are summarized."""
        summarizer = HeuristicConversationSummarizer()
        messages = [
            Message(role="user", content="What is the answer?"),
            Message(role="assistant", content="The answer is 42."),
        ]

        result = summarizer.summarize(messages, max_chars=100)

        assert "User:" in result or "user" in result.lower()
        assert "answer" in result.lower()

    def test_summarize_respects_max_chars(self):
        """Test that summary respects max_chars limit."""
        summarizer = HeuristicConversationSummarizer()
        long_content = "x" * 1000
        messages = [Message(role="user", content=long_content)]

        result = summarizer.summarize(messages, max_chars=50)

        # Result should be truncated
        assert len(result) <= 100  # Allow some buffer for truncation marker


class TestLLMConversationSummarizer:
    """Test LLMConversationSummarizer."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return MagicMock()

    def test_uses_fallback_for_empty_messages(self, mock_llm_client):
        """Test that fallback is used when messages are empty."""
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = []

        result = summarizer.summarize(messages, max_chars=100)

        assert result == "(no additional context retained)"
        mock_llm_client.complete.assert_not_called()

    def test_uses_fallback_when_llm_fails(self, mock_llm_client):
        """Test that fallback is used when LLM raises error."""
        mock_llm_client.complete.side_effect = LLMError("API error")
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [Message(role="user", content="Hello")]

        result = summarizer.summarize(messages, max_chars=100)

        # Should get some result (from fallback)
        assert result != ""
        assert "hello" in result.lower() or "additional context" in result.lower()

    def test_uses_fallback_when_llm_returns_empty(self, mock_llm_client):
        """Test that fallback is used when LLM returns empty string."""
        mock_llm_client.complete.return_value = ""
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [Message(role="user", content="Hello")]

        result = summarizer.summarize(messages, max_chars=100)

        # Should get result from fallback
        assert result != ""

    def test_uses_fallback_when_llm_returns_none(self, mock_llm_client):
        """Test that fallback is used when LLM returns None."""
        mock_llm_client.complete.return_value = None
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [Message(role="user", content="Hello")]

        result = summarizer.summarize(messages, max_chars=100)

        # Should get result from fallback
        assert result != ""

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

    def test_uses_temperature_zero(self, mock_llm_client):
        """Test that temperature=0.0 is used for deterministic output."""
        mock_llm_client.complete.return_value = "Summary"
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [Message(role="user", content="Hello")]

        _ = summarizer.summarize(messages, max_chars=100)

        call_args = mock_llm_client.complete.call_args
        assert call_args[1]["temperature"] == 0.0

    def test_custom_fallback(self, mock_llm_client):
        """Test using custom fallback summarizer."""
        mock_llm_client.complete.side_effect = LLMError("Error")
        custom_fallback = MagicMock()
        custom_fallback.summarize.return_value = "Custom fallback result"

        summarizer = LLMConversationSummarizer(
            mock_llm_client,
            fallback=custom_fallback,
        )
        messages = [Message(role="user", content="Hello")]

        result = summarizer.summarize(messages, max_chars=100)

        assert result == "Custom fallback result"
        custom_fallback.summarize.assert_called_once_with(messages, max_chars=100)

    def test_truncates_long_llm_response(self, mock_llm_client):
        """Test that long LLM response is truncated to max_chars."""
        long_summary = "x" * 1000
        mock_llm_client.complete.return_value = long_summary
        summarizer = LLMConversationSummarizer(mock_llm_client)
        messages = [Message(role="user", content="Hello")]

        result = summarizer.summarize(messages, max_chars=50)

        # Should be truncated (with some buffer for truncation marker)
        assert len(result) <= 100
