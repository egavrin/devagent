"""Conversation summarization helpers used by the session context pruner."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from ai_dev_agent.core.utils.context_budget import summarize_text
from ai_dev_agent.providers.llm.base import LLMClient, LLMError, Message

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)


class ConversationSummarizer(Protocol):
    """Protocol implemented by conversation summarizers."""

    def summarize(self, messages: Sequence[Message], *, max_chars: int) -> str:
        """Return a concise summary of ``messages`` within ``max_chars`` characters."""


def _format_messages(messages: Sequence[Message]) -> str:
    """Return a deterministic plain-text representation of chat messages."""

    lines: list[str] = []
    for message in messages:
        content = (message.content or "").strip()
        if not content:
            continue
        role = message.role.capitalize()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class HeuristicConversationSummarizer:
    """Fallback summarizer that mirrors the previous deterministic behaviour."""

    def summarize(self, messages: Sequence[Message], *, max_chars: int) -> str:
        formatted = _format_messages(messages)
        if not formatted:
            return "(no additional context retained)"
        return summarize_text(formatted, max_chars)


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert senior engineer helping to condense a conversation between "
    "a developer and their coding assistant. Capture the key goals, approaches, "
    "decisions, and remaining follow-ups so another engineer can quickly get back "
    "up to speed. Be precise and avoid embellishment."
)

DEFAULT_USER_TEMPLATE = (
    "Summarize the following conversation. Focus on:\n"
    "- Current objective or task\n"
    "- Progress made and important commands\n"
    "- Decisions, constraints, or notable discoveries\n"
    "- Remaining follow-ups or open questions\n\n"
    "Write 3-5 bullet points using concise sentences. Keep the response within {max_chars} "
    "characters. Conversation:\n{conversation}"
)


class LLMConversationSummarizer:
    """Summarizer backed by an LLM client with heuristic fallback."""

    def __init__(
        self,
        client: LLMClient,
        *,
        system_prompt: str | None = None,
        user_template: str | None = None,
        max_tokens: int = 384,
        fallback: ConversationSummarizer | None = None,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._user_template = user_template or DEFAULT_USER_TEMPLATE
        self._max_tokens = max_tokens
        self._fallback = fallback or HeuristicConversationSummarizer()

    def summarize(self, messages: Sequence[Message], *, max_chars: int) -> str:
        formatted = _format_messages(messages)
        if not formatted:
            return self._fallback.summarize(messages, max_chars=max_chars)

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

        try:
            summary = self._client.complete(prompt, temperature=0.0, max_tokens=self._max_tokens)
        except LLMError as exc:
            LOGGER.warning("LLM summarizer failed, falling back to heuristic: %s", exc)
            return self._fallback.summarize(messages, max_chars=max_chars)

        summary = (summary or "").strip()
        if not summary:
            return self._fallback.summarize(messages, max_chars=max_chars)

        return summarize_text(summary, max_chars)


__all__ = [
    "ConversationSummarizer",
    "HeuristicConversationSummarizer",
    "LLMConversationSummarizer",
]
