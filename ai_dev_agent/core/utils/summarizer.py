"""Conversation summarization utilities for context management.

Implements recursive summarization and asynchronous workflows to preserve context
when pruning.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ai_dev_agent.providers.llm.base import Message

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)


class LLMSummarizer(Protocol):
    """Protocol for LLM clients used for summarization."""

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Complete a prompt with the LLM."""
        ...


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

    # Temperature for summarization requests
    summary_temperature: float = 0.3

    # Whether to use async summarization
    async_summarization: bool = False

    # Model to use for summarization (can be cheaper than main model)
    summarization_model: str | None = None


class ConversationSummarizer:
    """Summarize conversations to fit within token budgets.

    Implements two strategies:
    1. Recursive split-and-summarize for deep conversations
    2. Async summarization for non-blocking operation
    """

    def __init__(
        self,
        llm: LLMSummarizer,
        config: SummarizationConfig | None = None,
    ):
        """Initialize the summarizer.

        Args:
            llm: LLM client for generating summaries
            config: Summarization configuration
        """
        self.llm = llm
        self.config = config or SummarizationConfig()
        self._summary_cache = {}

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
        """
        if not messages or len(messages) < self.config.min_messages_to_summarize:
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
        """Alias for summarize_if_needed to provide consistent interface.

        This allows both ConversationSummarizer and TwoTierSummarizer
        to be used interchangeably.

        Args:
            messages: Messages to optimize
            target_tokens: Target token count
            estimate_tokens_func: Token estimation function

        Returns:
            Optimized messages
        """
        return self.summarize_if_needed(messages, target_tokens, estimate_tokens_func)

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
        """
        # Prevent infinite recursion
        if depth >= self.config.max_recursion_depth:
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

        # Generate summary
        summary_messages = [system_prompt, user_prompt]

        try:
            summary = self.llm.complete(
                summary_messages,
                temperature=self.config.summary_temperature,
                max_tokens=self.config.summary_max_tokens,
            )

            # Cache the summary
            self._summary_cache[cache_key] = summary
            return summary

        except Exception as e:
            LOGGER.warning(f"Failed to generate summary: {e}")
            # Fallback to simple truncation
            return f"[Summary generation failed. Last {len(messages)} messages truncated.]"

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
        """
        # Run summarization in background
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.summarize_if_needed,
            messages,
            target_tokens,
        )


class TwoTierSummarizer(ConversationSummarizer):
    """Enhanced summarizer with two-tier strategy.

    First tries cheap pruning, then falls back to expensive summarization.
    """

    def __init__(
        self,
        llm: LLMSummarizer,
        config: SummarizationConfig | None = None,
        prune_threshold: int = 20000,
        protect_recent: int = 40000,
    ):
        """Initialize two-tier summarizer.

        Args:
            llm: LLM client for summaries
            config: Summarization config
            prune_threshold: Minimum tokens to save before pruning (default: 20k)
            protect_recent: Recent tokens to protect from pruning (default: 40k)
        """
        super().__init__(llm, config)
        self.prune_threshold = prune_threshold
        self.protect_recent = protect_recent

    def optimize_context(
        self,
        messages: list[Message],
        target_tokens: int,
        estimate_tokens_func=None,
    ) -> list[Message]:
        """Optimize context with two-tier approach.

        Args:
            messages: Messages to optimize
            target_tokens: Target token count
            estimate_tokens_func: Token estimation function

        Returns:
            Optimized messages
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
        if tokens_saved >= self.prune_threshold and pruned_tokens <= target_tokens:
            LOGGER.info(f"Pruned {tokens_saved} tokens via tool output truncation")
            return pruned

        # Tier 2: Fall back to summarization
        LOGGER.info("Insufficient pruning, falling back to summarization")
        return self.summarize_if_needed(pruned, target_tokens, estimate_tokens_func)

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
            if total_tokens > self.protect_recent:
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


def create_summarizer(
    llm: LLMSummarizer,
    two_tier: bool = True,
    **config_kwargs,
) -> ConversationSummarizer:
    """Factory function to create appropriate summarizer.

    Args:
        llm: LLM client for generating summaries
        two_tier: If True, use TwoTierSummarizer
        **config_kwargs: Configuration parameters

    Returns:
        Configured summarizer instance
    """
    config = SummarizationConfig(**config_kwargs)

    if two_tier:
        return TwoTierSummarizer(llm, config)
    else:
        return ConversationSummarizer(llm, config)


__all__ = [
    "ConversationSummarizer",
    "SummarizationConfig",
    "TwoTierSummarizer",
    "create_summarizer",
]
