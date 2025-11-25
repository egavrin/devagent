"""Conversation summarization for context management.

Implements LLM-based summarization with recursive split strategy, two-tier pruning,
async support, and caching. Follows fail-fast philosophy: no heuristic fallbacks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ai_dev_agent.core.utils.constants import LLM_DEFAULT_TEMPERATURE
from ai_dev_agent.prompts.loader import PromptLoader
from ai_dev_agent.providers.llm.base import Message

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_dev_agent.providers.llm.base import LLMClient

LOGGER = logging.getLogger(__name__)


class ConversationSummarizer(Protocol):
    """Protocol implemented by conversation summarizers."""

    def summarize(self, messages: Sequence[Message], *, max_chars: int) -> str:
        """Return a concise summary of ``messages`` within ``max_chars`` characters.

        Raises:
            LLMError: If summarization fails - no fallback provided.
        """


@dataclass
class SummarizationConfig:
    """Configuration for summarization behavior."""

    # Maximum tokens for history before summarization
    max_history_tokens: int = 8192

    # Minimum messages before attempting summarization
    min_messages_to_summarize: int = 4

    # Maximum recursion depth for recursive summarization
    max_recursion_depth: int = 3

    # Token limit for individual summaries
    summary_max_tokens: int = 500

    # Temperature for summarization requests (default: 0.0 for reproducibility)
    summary_temperature: float = LLM_DEFAULT_TEMPERATURE

    # Whether to use async summarization
    async_summarization: bool = False

    # Two-tier pruning thresholds
    prune_threshold: int = 20000  # Minimum tokens to save before pruning
    protect_recent: int = 40000  # Recent tokens to protect from pruning


_PROMPT_LOADER = PromptLoader()

DEFAULT_SYSTEM_PROMPT = _PROMPT_LOADER.load_system_prompt("conversation_summary_system")
DEFAULT_USER_TEMPLATE = _PROMPT_LOADER.load_system_prompt("conversation_summary_user")


class LLMConversationSummarizer:
    """LLM-based conversation summarizer with advanced strategies.

    Implements:
    - Recursive split-and-summarize for deep conversations
    - Two-tier pruning (cheap tool truncation before expensive summarization)
    - Async summarization for non-blocking operation
    - Summary caching to avoid redundant LLM calls
    - Fail-fast: raises exceptions on LLM errors instead of masking them
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        config: SummarizationConfig | None = None,
        system_prompt: str | None = None,
        user_template: str | None = None,
        max_tokens: int = 384,
    ) -> None:
        """Initialize the LLM conversation summarizer.

        Args:
            client: LLM client for generating summaries
            config: Summarization configuration
            system_prompt: Custom system prompt (optional)
            user_template: Custom user template with {conversation} and {max_chars} placeholders
            max_tokens: Maximum tokens for LLM response
        """
        self._client = client
        self._config = config or SummarizationConfig()
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._user_template = user_template or DEFAULT_USER_TEMPLATE
        self._max_tokens = max_tokens
        self._summary_cache: dict[str, str] = {}

    @property
    def config(self) -> SummarizationConfig:
        """Return the summarization configuration."""
        return self._config

    def summarize(self, messages: Sequence[Message], *, max_chars: int) -> str:
        """Summarize messages using LLM.

        Args:
            messages: Messages to summarize
            max_chars: Maximum characters in summary

        Returns:
            Summary text within max_chars limit

        Raises:
            LLMError: If LLM call fails (no fallback - fail fast)
        """
        if not messages:
            return ""

        formatted = self._format_messages(messages)
        if not formatted:
            return ""

        prompt = [
            Message(role="system", content=self._system_prompt),
            Message(
                role="user",
                content=self._user_template.format(
                    conversation=formatted,
                    max_chars=max_chars,
                ),
            ),
        ]

        # Call LLM - let exceptions propagate (fail fast)
        summary = self._client.complete(
            prompt, temperature=self._config.summary_temperature, max_tokens=self._max_tokens
        )

        summary = (summary or "").strip()
        if not summary:
            # Empty response is an error condition - raise
            raise ValueError("LLM returned empty summary")

        # Truncate to max_chars if needed
        if len(summary) > max_chars:
            summary = summary[: max_chars - 3] + "..."

        return summary

    def summarize_if_needed(
        self,
        messages: list[Message],
        target_tokens: int,
        estimate_tokens_func=None,
    ) -> list[Message]:
        """Summarize messages if they exceed token budget.

        Args:
            messages: Messages to potentially summarize
            target_tokens: Target token count
            estimate_tokens_func: Function to estimate tokens (defaults to simple estimation)

        Returns:
            Messages with summaries if needed

        Raises:
            LLMError: If summarization fails
        """
        if not messages or len(messages) < self._config.min_messages_to_summarize:
            return messages

        # Default token estimation
        if estimate_tokens_func is None:
            estimate_tokens_func = self._simple_token_estimate

        current_tokens = estimate_tokens_func(messages)
        if current_tokens <= target_tokens:
            return messages

        # Apply recursive summarization with a split strategy
        return self._recursive_summarize(
            messages,
            target_tokens,
            estimate_tokens_func,
            depth=0,
        )

    def optimize_context(
        self,
        messages: list[Message],
        target_tokens: int,
        estimate_tokens_func=None,
    ) -> list[Message]:
        """Optimize context with two-tier approach.

        First tries cheap pruning (tool output truncation), then falls back to
        expensive summarization if needed.

        Args:
            messages: Messages to optimize
            target_tokens: Target token count
            estimate_tokens_func: Token estimation function

        Returns:
            Optimized messages

        Raises:
            LLMError: If summarization fails
        """
        if estimate_tokens_func is None:
            estimate_tokens_func = self._simple_token_estimate

        current_tokens = estimate_tokens_func(messages)
        if current_tokens <= target_tokens:
            return messages

        # Tier 1: Try cheap pruning first
        pruned = self._prune_old_tool_outputs(messages, estimate_tokens_func)
        pruned_tokens = estimate_tokens_func(pruned)

        # Check if we saved enough
        tokens_saved = current_tokens - pruned_tokens
        if tokens_saved >= self._config.prune_threshold and pruned_tokens <= target_tokens:
            LOGGER.info(f"Pruned {tokens_saved} tokens via tool output truncation")
            return pruned

        # Tier 2: Fall back to summarization
        LOGGER.info("Insufficient pruning, falling back to summarization")
        return self.summarize_if_needed(pruned, target_tokens, estimate_tokens_func)

    async def summarize_async(
        self,
        messages: list[Message],
        target_tokens: int,
    ) -> list[Message]:
        """Asynchronously summarize messages in a background thread.

        Args:
            messages: Messages to summarize
            target_tokens: Target token count

        Returns:
            Summarized messages

        Raises:
            LLMError: If summarization fails
        """
        # Run summarization in background
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.summarize_if_needed,
            messages,
            target_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recursive_summarize(
        self,
        messages: list[Message],
        target_tokens: int,
        estimate_tokens_func,
        depth: int,
    ) -> list[Message]:
        """Recursively summarize messages using a split strategy.

        Args:
            messages: Messages to summarize
            target_tokens: Target token count
            estimate_tokens_func: Token estimation function
            depth: Current recursion depth

        Returns:
            Summarized messages

        Raises:
            LLMError: If summarization fails
        """
        # Prevent infinite recursion
        if depth >= self._config.max_recursion_depth:
            return self._summarize_all(messages)

        # Too few messages to split
        if len(messages) < 4:
            return self._summarize_all(messages)

        # Find split point (keep recent messages fresh)
        split_index = len(messages) // 2

        # Separate old and recent messages
        old_messages = messages[:split_index]
        recent_messages = messages[split_index:]

        # Summarize old messages
        summary = self._create_summary(old_messages)
        summary_message = Message(
            role="assistant",
            content=f"[Summary of previous conversation]:\n{summary}",
        )

        # Combine summary with recent messages
        combined = [summary_message, *recent_messages]

        # Check if we're within budget
        if estimate_tokens_func(combined) <= target_tokens:
            return combined

        # Need more summarization - recurse
        return self._recursive_summarize(
            combined,
            target_tokens,
            estimate_tokens_func,
            depth + 1,
        )

    def _summarize_all(self, messages: list[Message]) -> list[Message]:
        """Create a single summary of all messages.

        Args:
            messages: Messages to summarize

        Returns:
            List with single summary message

        Raises:
            LLMError: If summarization fails
        """
        summary = self._create_summary(messages)
        return [
            Message(
                role="assistant",
                content=f"[Complete conversation summary]:\n{summary}",
            )
        ]

    def _create_summary(self, messages: list[Message]) -> str:
        """Create a summary of messages using the LLM.

        Args:
            messages: Messages to summarize

        Returns:
            Summary text

        Raises:
            LLMError: If summarization fails
        """
        # Check cache
        cache_key = self._get_cache_key(messages)
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        # Prepare summarization prompt
        system_prompt = Message(
            role="system",
            content=(
                "You are a helpful assistant that creates concise summaries. "
                "Provide a detailed but concise summary of the conversation below. "
                "Focus on key decisions, findings, and important context. "
                "Preserve technical details and specific file/function names."
            ),
        )

        # Format messages for summarization
        conversation_text = self._format_messages_for_summary(messages)
        user_prompt = Message(
            role="user",
            content=f"Please summarize this conversation:\n\n{conversation_text}",
        )

        # Generate summary - let exceptions propagate (fail fast)
        summary_messages = [system_prompt, user_prompt]
        summary = self._client.complete(
            summary_messages,
            temperature=self._config.summary_temperature,
            max_tokens=self._config.summary_max_tokens,
        )

        if not summary:
            raise ValueError("LLM returned empty summary")

        # Cache the summary
        self._summary_cache[cache_key] = summary
        return summary

    def _prune_old_tool_outputs(
        self,
        messages: list[Message],
        estimate_tokens_func,
    ) -> list[Message]:
        """Prune old tool outputs while protecting recent messages.

        Args:
            messages: Messages to prune
            estimate_tokens_func: Token estimation function

        Returns:
            Pruned messages
        """
        if not messages:
            return messages

        # Find protection boundary
        total_tokens = 0
        protect_from_index = len(messages)

        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_tokens_func([messages[i]])
            total_tokens += msg_tokens
            if total_tokens > self._config.protect_recent:
                protect_from_index = i + 1
                break

        # Prune tool outputs before protection boundary
        pruned = []
        for i, msg in enumerate(messages):
            if i >= protect_from_index:
                # Protected - keep as is
                pruned.append(msg)
            elif msg.role == "tool" and msg.content and len(msg.content) > 500:
                # Old tool output - truncate
                truncated = Message(
                    role="tool",
                    content=msg.content[:200] + "\n[... tool output truncated ...]",
                    tool_call_id=msg.tool_call_id,
                )
                pruned.append(truncated)
            else:
                # Keep other messages
                pruned.append(msg)

        return pruned

    def _format_messages(self, messages: Sequence[Message]) -> str:
        """Return a deterministic plain-text representation of chat messages."""
        lines: list[str] = []
        for message in messages:
            content = (message.content or "").strip()
            if not content:
                continue
            role = message.role.capitalize()
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _format_messages_for_summary(self, messages: list[Message]) -> str:
        """Format messages for summarization prompt.

        Args:
            messages: Messages to format

        Returns:
            Formatted text
        """
        lines = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content or "[No content]"

            # Truncate very long messages
            if len(content) > 1000:
                content = content[:997] + "..."

            lines.append(f"{role}: {content}")

        return "\n\n".join(lines)

    def _simple_token_estimate(self, messages: list[Message]) -> int:
        """Simple token estimation (4 chars = 1 token).

        Args:
            messages: Messages to estimate

        Returns:
            Estimated token count
        """
        total = 0
        for msg in messages:
            content = msg.content or ""
            total += len(content) // 4 + 8  # content + overhead
        return total

    def _get_cache_key(self, messages: list[Message]) -> str:
        """Generate cache key for messages.

        Args:
            messages: Messages to cache

        Returns:
            Cache key string
        """
        # Simple hash based on message contents
        key_parts = []
        for msg in messages[:5]:  # Use first 5 messages for key
            if msg.content:
                key_parts.append(msg.content[:50])
        return "|".join(key_parts)


def create_summarizer(
    llm: LLMClient,
    *,
    config: SummarizationConfig | None = None,
    **kwargs,
) -> LLMConversationSummarizer:
    """Factory function to create LLM conversation summarizer.

    Args:
        llm: LLM client for generating summaries
        config: Summarization configuration
        **kwargs: Additional arguments passed to LLMConversationSummarizer

    Returns:
        Configured summarizer instance
    """
    return LLMConversationSummarizer(llm, config=config, **kwargs)


__all__ = [
    "ConversationSummarizer",
    "LLMConversationSummarizer",
    "SummarizationConfig",
    "create_summarizer",
]
