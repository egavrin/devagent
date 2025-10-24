"""Utilities for keeping LLM conversations within configured context budgets.

Enhanced with a two-tier pruning strategy:
1. Try cheap pruning of old tool outputs first
2. Fall back to LLM summarization if needed
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ai_dev_agent.providers.llm.base import LLMClient, Message, ToolCallResult
from .constants import (
    DEFAULT_KEEP_LAST_ASSISTANT,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_TOOL_MESSAGES,
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    DEFAULT_RESPONSE_HEADROOM,
)

LOGGER = logging.getLogger(__name__)

# Two-tier pruning thresholds
PRUNE_PROTECT_TOKENS = 40000  # Protect recent 40k tokens
PRUNE_MINIMUM_SAVINGS = 20000  # Only prune if saving 20k+ tokens

@dataclass
class ContextBudgetConfig:
    """Configuration for context pruning and truncation.

    Enhanced with two-tier pruning configuration.
    """

    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    headroom_tokens: int = DEFAULT_RESPONSE_HEADROOM
    max_tool_messages: int = DEFAULT_MAX_TOOL_MESSAGES
    max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS
    keep_last_assistant: int = DEFAULT_KEEP_LAST_ASSISTANT

    # Two-tier pruning settings
    enable_two_tier: bool = True
    prune_protect_tokens: int = PRUNE_PROTECT_TOKENS
    prune_minimum_savings: int = PRUNE_MINIMUM_SAVINGS
    enable_summarization: bool = True
    summarization_model: Optional[str] = None


def estimate_tokens(messages: Sequence[Message], model: Optional[str] = None) -> int:
    """Estimate token usage for a list of messages.

    Enhanced version with optional accurate counting via tiktoken/litellm.
    Falls back to character-based estimation if accurate counting unavailable.

    Args:
        messages: List of messages to estimate
        model: Optional model name for accurate counting

    Returns:
        Estimated token count
    """
    # Try accurate counting first if model specified
    if model:
        try:
            return _accurate_token_count(messages, model)
        except Exception:
            # Fall back to heuristic
            pass

    # Character-based heuristic (4 chars ≈ 1 token)
    # This is the original simple estimation
    total = 0
    for msg in messages:
        content = msg.content or ""
        total += len(content) // 4
        if msg.tool_calls:
            total += 16 * len(msg.tool_calls)
        if msg.tool_call_id:
            total += 4
        total += 8  # base allowance per message
    return total


def _accurate_token_count(messages: Sequence[Message], model: str) -> int:
    """Attempt accurate token counting using tiktoken or litellm.

    Args:
        messages: Messages to count
        model: Model name for encoding

    Returns:
        Accurate token count

    Raises:
        Exception: If accurate counting not available
    """
    # Try tiktoken first (for OpenAI models)
    try:
        import tiktoken

        # Map model to encoding
        encoding_map = {
            "gpt-4": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base",
        }

        encoding_name = None
        for model_prefix, enc_name in encoding_map.items():
            if model.startswith(model_prefix):
                encoding_name = enc_name
                break

        if encoding_name:
            encoding = tiktoken.get_encoding(encoding_name)
            total = 0
            for msg in messages:
                if msg.content:
                    total += len(encoding.encode(msg.content))
                # Add tokens for message structure
                total += 4  # role, content markers
                if msg.tool_calls:
                    total += 20 * len(msg.tool_calls)  # rough estimate for tool call structure
            return total
    except ImportError:
        pass
    except Exception:
        pass

    # Try litellm as fallback
    try:
        import litellm

        # Convert messages to litellm format
        litellm_messages = []
        for msg in messages:
            litellm_msg = {"role": msg.role}
            if msg.content:
                litellm_msg["content"] = msg.content
            if msg.tool_calls:
                litellm_msg["tool_calls"] = msg.tool_calls
            litellm_messages.append(litellm_msg)

        # Use litellm's token counter
        return litellm.token_counter(model=model, messages=litellm_messages)
    except ImportError:
        pass
    except Exception:
        pass

    # If we get here, accurate counting not available
    raise Exception("Accurate token counting not available")


def summarize_text(text: str, max_chars: int) -> str:
    """Return a truncated text summary that notes omitted content."""

    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    omitted = len(text) - len(truncated)
    return f"{truncated}\n[... {omitted} characters omitted ...]"


def _index_of_last_role(messages: Sequence[Message], role: str) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == role:
            return idx
    return -1


def prune_messages(messages: Sequence[Message], config: ContextBudgetConfig) -> List[Message]:
    """Produce a pruned copy of messages that fits within the budget.

    Enhanced with two-tier pruning strategy:
    1. Try cheap pruning first (tool output truncation)
    2. Fall back to aggressive truncation if needed
    """

    if not messages:
        return []

    total_allowed = max(config.max_tokens - config.headroom_tokens, 0)
    initial_tokens = estimate_tokens(messages)

    if initial_tokens <= total_allowed:
        return list(messages)

    # Two-tier pruning if enabled
    if config.enable_two_tier:
        pruned = _two_tier_prune(messages, config, total_allowed)
        if estimate_tokens(pruned) <= total_allowed:
            return pruned
    else:
        # Original pruning logic
        pruned = [Message(role=msg.role, content=msg.content, tool_call_id=msg.tool_call_id, tool_calls=msg.tool_calls) for msg in messages]

    # Always retain the first system message and the most recent user message
    keep_indices = set()
    for idx, msg in enumerate(pruned):
        if msg.role == "system":
            keep_indices.add(idx)
            break

    last_user_index = _index_of_last_role(pruned, "user")
    if last_user_index != -1:
        keep_indices.add(last_user_index)

    # Keep the most recent assistant messages
    assistant_indices = [idx for idx, msg in enumerate(pruned) if msg.role == "assistant"]
    for idx in assistant_indices[-config.keep_last_assistant:]:
        keep_indices.add(idx)

    # Summarize tool messages older than the most recent ones
    tool_indices = [idx for idx, msg in enumerate(pruned) if msg.role == "tool"]
    if len(tool_indices) > config.max_tool_messages:
        protected = set(tool_indices[-config.max_tool_messages:])
    else:
        protected = set(tool_indices)

    for idx in tool_indices:
        msg = pruned[idx]
        if idx not in protected or idx not in keep_indices:
            summary = summarize_text(msg.content or "", config.max_tool_output_chars)
            pruned[idx] = Message(role="tool", content=summary, tool_call_id=msg.tool_call_id)
            keep_indices.add(idx)

    # Recalculate and drop oldest non-protected messages while over budget
    # NOTE: We preserve message roles to maintain conversation structure.
    # Orphaned tool messages will be cleaned up by _remove_orphaned_tool_messages()
    # in context_service.py or by sanitization before LLM calls.
    ordered_indices = list(range(len(pruned)))
    for idx in ordered_indices:
        if estimate_tokens(pruned) <= total_allowed:
            break
        if idx in keep_indices:
            continue
        msg = pruned[idx]
        # Preserve message role and critical IDs - don't change message types!
        if msg.role == "tool":
            pruned[idx] = Message(
                role="tool",
                content="[content truncated]",
                tool_call_id=msg.tool_call_id,
            )
        elif msg.role == "user":
            pruned[idx] = Message(role="user", content="[content truncated]")
        elif msg.role == "assistant":
            # Preserve tool_calls structure even when truncating content
            # (orphaned tool calls will be cleaned up later)
            pruned[idx] = Message(
                role="assistant",
                content="[context truncated]",
                tool_calls=msg.tool_calls,
            )
        else:
            # system messages
            pruned[idx] = Message(role=msg.role, content="[context truncated]")
        keep_indices.add(idx)

    # Final pass: if still over budget, aggressively trim earliest tool summaries
    for idx in ordered_indices:
        if estimate_tokens(pruned) <= total_allowed:
            break
        msg = pruned[idx]
        if msg.role == "tool":
            preserved_id = msg.tool_call_id
            if not preserved_id:
                from uuid import uuid4
                preserved_id = f"tool-pruned-{uuid4().hex[:8]}"
            pruned[idx] = Message(
                role="tool",
                content="[tool output truncated]",
                tool_call_id=preserved_id,
            )

    return pruned


def _two_tier_prune(
    messages: Sequence[Message],
    config: ContextBudgetConfig,
    target_tokens: int,
) -> List[Message]:
    """Apply two-tier pruning strategy.

    Args:
        messages: Messages to prune
        config: Budget configuration
        target_tokens: Target token count

    Returns:
        Pruned messages
    """
    messages_list = list(messages)
    initial_tokens = estimate_tokens(messages_list)

    # Find protection boundary (keep recent N tokens)
    protect_from_idx = 0
    token_sum = 0
    for i in range(len(messages_list) - 1, -1, -1):
        msg_tokens = estimate_tokens([messages_list[i]])
        token_sum += msg_tokens
        if token_sum > config.prune_protect_tokens:
            protect_from_idx = i + 1
            break

    # Tier 1: Prune old tool outputs
    pruned = []
    tokens_saved = 0

    for i, msg in enumerate(messages_list):
        if i >= protect_from_idx:
            # Within protected zone - keep as is
            pruned.append(msg)
        elif msg.role == "tool" and msg.content and len(msg.content) > 500:
            # Old tool output - truncate aggressively
            original_tokens = estimate_tokens([msg])
            truncated_msg = Message(
                role="tool",
                content=msg.content[:200] + "\n[... tool output pruned for context ...]",
                tool_call_id=msg.tool_call_id,
            )
            pruned.append(truncated_msg)
            new_tokens = estimate_tokens([truncated_msg])
            tokens_saved += original_tokens - new_tokens
        else:
            # Keep other message types
            pruned.append(msg)

    # Check if we saved enough
    if tokens_saved >= config.prune_minimum_savings:
        LOGGER.info(f"Two-tier pruning saved {tokens_saved} tokens")
        return pruned

    # Not enough savings - return original for fallback processing
    LOGGER.debug(f"Two-tier pruning saved only {tokens_saved} tokens, need {config.prune_minimum_savings}")
    return messages_list


def ensure_context_budget(messages: Iterable[Message], config: ContextBudgetConfig | None = None) -> List[Message]:
    """Return a context-limited version of messages according to configuration."""

    config = config or ContextBudgetConfig()
    messages_list = list(messages)
    pruned = prune_messages(messages_list, config)
    return pruned


def config_from_settings(settings) -> ContextBudgetConfig:
    """Build a ContextBudgetConfig using settings with safe fallbacks.

    Enhanced to include new two-tier pruning and summarization settings.
    """

    return ContextBudgetConfig(
        max_tokens=getattr(settings, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS),
        headroom_tokens=getattr(settings, "response_headroom_tokens", DEFAULT_RESPONSE_HEADROOM),
        max_tool_messages=getattr(settings, "max_tool_messages_kept", DEFAULT_MAX_TOOL_MESSAGES),
        max_tool_output_chars=getattr(settings, "max_tool_output_chars", DEFAULT_MAX_TOOL_OUTPUT_CHARS),
        keep_last_assistant=getattr(settings, "keep_last_assistant_messages", DEFAULT_KEEP_LAST_ASSISTANT),
        # New two-tier pruning settings
        enable_two_tier=getattr(settings, "enable_two_tier_pruning", True),
        prune_protect_tokens=getattr(settings, "prune_protect_tokens", PRUNE_PROTECT_TOKENS),
        prune_minimum_savings=getattr(settings, "prune_minimum_savings", PRUNE_MINIMUM_SAVINGS),
        enable_summarization=getattr(settings, "enable_summarization", True),
        summarization_model=getattr(settings, "summarization_model", None),
    )


class BudgetedLLMClient:
    """LLM client wrapper that enforces context budgets before each call."""

    def __init__(self, inner: LLMClient, config: ContextBudgetConfig | None = None, *, disabled: bool = False):
        self._inner = inner
        self._config = config or ContextBudgetConfig()
        self._disabled = disabled

    @property
    def inner(self) -> LLMClient:
        return self._inner

    def configure_retry(self, retry_config):  # type: ignore[override]
        if hasattr(self._inner, "configure_retry"):
            self._inner.configure_retry(retry_config)

    def configure_timeout(self, timeout: float):  # type: ignore[override]
        if hasattr(self._inner, "configure_timeout"):
            self._inner.configure_timeout(timeout)

    def _prepare_messages(self, messages: Sequence[Message]) -> List[Message]:
        original = list(messages)
        if self._disabled:
            return original
        before_tokens = estimate_tokens(original)
        pruned = ensure_context_budget(original, self._config)
        after_tokens = estimate_tokens(pruned)
        if after_tokens < before_tokens or len(pruned) < len(original):
            LOGGER.debug(
                "Context pruned: %s→%s estimated tokens, %s messages kept",
                before_tokens,
                after_tokens,
                len(pruned),
            )
        return pruned

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        prepared = self._prepare_messages(messages)
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers
        if response_format is not None:
            kwargs["response_format"] = response_format
        return self._inner.complete(prepared, **kwargs)

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: dict | None = None,
        hooks=None,
    ):
        prepared = self._prepare_messages(messages)
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers
        if hooks is not None:
            kwargs["hooks"] = hooks
        return self._inner.stream(prepared, **kwargs)

    def invoke_tools(
        self,
        messages: Sequence[Message],
        tools: List[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = "auto",
        extra_headers: dict | None = None,
        response_format: dict | None = None,
        parallel_tool_calls: bool = True,
    ) -> ToolCallResult:
        prepared = self._prepare_messages(messages)
        kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers
        if response_format is not None:
            kwargs["response_format"] = response_format
        return self._inner.invoke_tools(prepared, tools, **kwargs)

    def set_config(self, config: ContextBudgetConfig):
        self._config = config

    def disable(self) -> None:
        self._disabled = True

    def enable(self) -> None:
        self._disabled = False

    def __getattr__(self, item):
        return getattr(self._inner, item)


__all__ = [
    "ContextBudgetConfig",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RESPONSE_HEADROOM",
    "DEFAULT_MAX_TOOL_MESSAGES",
    "DEFAULT_MAX_TOOL_OUTPUT_CHARS",
    "DEFAULT_KEEP_LAST_ASSISTANT",
    "BudgetedLLMClient",
    "config_from_settings",
    "estimate_tokens",
    "summarize_text",
    "prune_messages",
    "ensure_context_budget",
]
