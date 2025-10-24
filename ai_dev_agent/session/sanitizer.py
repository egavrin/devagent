"""Conversation sanitization utilities to ensure API contract compliance."""
from __future__ import annotations

import logging
from typing import List, Mapping, Sequence

from ai_dev_agent.providers.llm.base import Message

logger = logging.getLogger(__name__)


def sanitize_conversation(messages: Sequence[Message]) -> List[Message]:
    """Remove tool messages whose IDs are not referenced by assistant tool calls.

    This ensures the conversation satisfies LLM API requirements:
    - Every tool message must have a tool_call_id
    - Every tool_call_id must reference a tool call from a prior assistant message
    - No orphaned tool messages exist

    This is critical for strict API providers like DeepSeek which validate
    that every tool response has a corresponding tool call.

    Args:
        messages: Raw conversation messages that may contain orphans

    Returns:
        Sanitized conversation with orphaned tool messages removed
    """
    # First pass: collect all valid tool_call_ids from assistant messages
    tool_call_ids: set[str] = set()

    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for call in msg.tool_calls:
                if isinstance(call, Mapping):
                    call_id = call.get("id") or call.get("tool_call_id")
                    if isinstance(call_id, str) and call_id:
                        tool_call_ids.add(call_id)

    # Second pass: filter out tool messages with invalid/missing tool_call_ids
    sanitized: List[Message] = []
    for msg in messages:
        if msg.role == "tool":
            if msg.tool_call_id and msg.tool_call_id in tool_call_ids:
                sanitized.append(msg)
            else:
                logger.debug(
                    "Removing orphaned tool message with ID: %s (not in valid tool_call_ids: %s)",
                    msg.tool_call_id,
                    list(tool_call_ids)[:5],  # Show first 5 for debugging
                )
        else:
            sanitized.append(msg)

    if len(sanitized) < len(messages):
        logger.info(
            "Sanitized conversation: removed %d orphaned tool message(s)",
            len(messages) - len(sanitized),
        )

    return sanitized


__all__ = ["sanitize_conversation"]
