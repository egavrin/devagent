"""Lightweight conversation summarization and pruning service."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Set

from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.core.utils.context_budget import (
    ContextBudgetConfig,
    ensure_context_budget,
    estimate_tokens,
    summarize_text,
)


SUMMARY_PREFIX = "[Context summary]"
RESUME_HINT = "Use the summary above to continue the conversation from where we left off."


@dataclass
class ContextPruningConfig:
    """Configurable knobs for summarizing and pruning session history."""

    max_total_tokens: int = 12_000
    trigger_tokens: int | None = None
    keep_recent_messages: int = 10
    summary_max_chars: int = 1_200
    max_event_history: int = 10

    def __post_init__(self) -> None:
        if self.trigger_tokens is None:
            # Default to pruning once we hit 80% of the hard limit.
            self.trigger_tokens = int(self.max_total_tokens * 0.8)
        if self.keep_recent_messages < 2:
            # Always keep at least the latest user/assistant turn.
            self.keep_recent_messages = 2


@dataclass
class ContextPruningEvent:
    """Metadata recorded each time pruning/summarization occurs."""

    timestamp: float
    token_estimate_before: int
    token_estimate_after: int
    summarized_messages: int
    summary_chars: int


class ContextPruningService:
    """Summarize and prune session history when token usage spikes."""

    def __init__(self, config: ContextPruningConfig | None = None) -> None:
        self._config = config or ContextPruningConfig()
        headroom = max(int(self._config.max_total_tokens * 0.1), 0)
        keep_last_assistant = max(self._config.keep_recent_messages // 2, 2)
        self._budget = ContextBudgetConfig(
            max_tokens=self._config.max_total_tokens,
            headroom_tokens=headroom,
            max_tool_messages=3,
            max_tool_output_chars=self._config.summary_max_chars,
            keep_last_assistant=keep_last_assistant,
        )
        self._last_summarized_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_session(self, session) -> None:  # type: ignore[no-untyped-def]
        """Run summarization/pruning for the provided session in-place."""

        with session.lock:
            messages = session.compose()
            token_estimate = estimate_tokens(messages)

            metadata = session.metadata.setdefault("context_service", {})
            metadata["token_estimate"] = token_estimate
            metadata.setdefault("events", [])

            if token_estimate <= self._config.trigger_tokens:
                return

            pruned_history = self._summarize_and_prune(session.history)
            session.history = pruned_history

            refreshed_messages = session.compose()
            final_messages = ensure_context_budget(refreshed_messages, self._budget)
            sanitized = self._sanitize_tool_sequences(final_messages)
            sanitized, redacted_count = self._redact_stale_tool_outputs(sanitized)
            metadata["redacted_tool_messages"] = redacted_count
            session.history = [msg for msg in sanitized if msg.role != "system"]

            final_estimate = estimate_tokens(session.compose())
            summary_text = next(
                (msg.content or "" for msg in session.history if self._is_summary_message(msg)),
                "",
            )

            event = ContextPruningEvent(
                timestamp=time.time(),
                token_estimate_before=token_estimate,
                token_estimate_after=final_estimate,
                summarized_messages=self._last_summarized_count,
                summary_chars=len(summary_text or ""),
            )
            self._record_event(metadata["events"], event)
            metadata["last_summary"] = summary_text
            metadata["token_estimate"] = final_estimate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summarize_and_prune(self, history: Sequence[Message]) -> List[Message]:
        """Return a pruned history with a lightweight summary message."""

        history_list = list(history)
        if not history_list:
            return []

        split_index = max(len(history_list) - self._config.keep_recent_messages, 0)
        split_index = self._adjust_split_index(history_list, split_index)

        recent_messages = history_list[split_index:]
        older_messages = history_list[:split_index]

        messages_to_summarize = [
            msg
            for msg in older_messages
            if not self._is_summary_message(msg) and (msg.content or "").strip()
        ]

        self._last_summarized_count = len(messages_to_summarize)
        if not messages_to_summarize:
            # Nothing to summarize, keep only the most recent messages.
            return list(recent_messages)

        summary_body = self._build_summary(messages_to_summarize)
        summary_message = Message(
            role="assistant",
            content=f"{SUMMARY_PREFIX} (last {len(messages_to_summarize)} messages)\n{summary_body}",
        )

        resume_message = Message(role="user", content=RESUME_HINT)

        return [summary_message, resume_message, *recent_messages]

    def _adjust_split_index(self, history_list: Sequence[Message], split_index: int) -> int:
        """Shift the recent window to avoid leaving tool responses orphaned."""

        if split_index <= 0 or split_index >= len(history_list):
            return split_index

        while split_index > 0 and history_list[split_index].role == "tool":
            split_index -= 1

        return split_index

    def _build_summary(self, messages: Iterable[Message]) -> str:
        lines: List[str] = []
        for msg in messages:
            if not msg.content:
                continue
            role = msg.role.capitalize()
            content = msg.content.strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")

        merged = "\n".join(lines)
        if not merged:
            return "(no additional context retained)"
        return summarize_text(merged, self._config.summary_max_chars)

    def _is_summary_message(self, message: Message) -> bool:
        if message.role != "assistant":
            return False
        content = message.content or ""
        return content.startswith(SUMMARY_PREFIX)

    def _sanitize_tool_sequences(self, messages: Sequence[Message]) -> List[Message]:
        """Drop tool responses whose initiating assistant turn was pruned."""

        result: List[Message] = []
        last_assistant_with_tools: Message | None = None

        for msg in messages:
            if msg.role == "assistant":
                result.append(msg)
                last_assistant_with_tools = msg if getattr(msg, "tool_calls", None) else None
                continue

            if msg.role == "tool":
                if last_assistant_with_tools is None:
                    continue
                result.append(msg)
                continue

            result.append(msg)
            last_assistant_with_tools = None

        return result

    def _redact_stale_tool_outputs(self, messages: Sequence[Message]) -> tuple[List[Message], int]:
        """Summarise or redact tool outputs that fall outside the active window."""

        if not messages:
            return [], 0

        tool_indices = [idx for idx, msg in enumerate(messages) if msg.role == "tool"]
        if not tool_indices:
            return list(messages), 0

        keep_count = max(self._budget.max_tool_messages, 0)
        keep_indices: Set[int] = set(tool_indices[-keep_count:]) if keep_count else set()

        redacted = 0
        limit = max(self._config.summary_max_chars // 2, 512)
        result: List[Message] = []

        for idx, msg in enumerate(messages):
            if msg.role != "tool" or idx in keep_indices:
                result.append(msg)
                continue

            summary = summarize_text((msg.content or "").strip(), limit)
            placeholder_lines = ["[tool output redacted]"]
            if summary:
                placeholder_lines.append(summary)
            redacted_content = "\n".join(placeholder_lines)
            result.append(
                Message(
                    role="tool",
                    content=redacted_content,
                    tool_call_id=msg.tool_call_id,
                    tool_calls=msg.tool_calls,
                )
            )
            redacted += 1

        return result, redacted

    def _record_event(self, events: List[Mapping[str, object]], event: ContextPruningEvent) -> None:
        payload = {
            "timestamp": event.timestamp,
            "token_estimate_before": event.token_estimate_before,
            "token_estimate_after": event.token_estimate_after,
            "summarized_messages": event.summarized_messages,
            "summary_chars": event.summary_chars,
        }
        events.append(payload)
        if len(events) > self._config.max_event_history:
            del events[: len(events) - self._config.max_event_history]


__all__ = [
    "ContextPruningConfig",
    "ContextPruningEvent",
    "ContextPruningService",
]
