"""Utilities for working with tool identifiers and signatures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, TypeVar

T = TypeVar("T")


@lru_cache(maxsize=1)
def _registry():
    from ai_dev_agent.tools.registry import registry as _registry_instance

    return _registry_instance


def canonical_tool_name(tool_name: str | None) -> str:
    """Return a stable canonical name for a tool alias."""
    if not tool_name:
        return "generic"
    return _registry().canonical_name(tool_name)


def tool_category(tool_name: str | None) -> str:
    """Return the logical category for a tool alias."""
    if not tool_name:
        return "generic"
    return _registry().category(tool_name)


def display_tool_name(tool_name: str | None) -> str:
    """Return the user-facing display name for a tool."""
    if not tool_name:
        return "generic"
    return _registry().display_name(tool_name)


def tool_aliases(tool_name: str | None, *, include_canonical: bool = True) -> tuple[str, ...]:
    """Return all known aliases for a tool, including the canonical name."""
    return _registry().aliases(tool_name, include_canonical=include_canonical)


def expand_tool_aliases(mapping: Mapping[str, T]) -> dict[str, T]:
    """Expand a mapping keyed by tool name to include all aliases."""

    expanded: dict[str, T] = {}
    for name, value in mapping.items():
        for alias in _registry().aliases(name):
            expanded[alias] = value
    return expanded


def build_tool_signature(tool_name: str, arguments: Mapping[str, Any] | None) -> str:
    """Build a deterministic signature for a tool call."""
    import json

    args = arguments or {}
    canonical = canonical_tool_name(tool_name)

    if canonical == "symbols":
        return f"{tool_name}:{args.get('name', '')}"

    if _registry().tool_in_category(tool_name, "file_read"):
        path_value = args.get("path")
        if path_value:
            return f"{tool_name}:{path_value}"
        paths = args.get("paths") or []
        if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
            try:
                key = ",".join(sorted(str(p) for p in paths if p))
            except Exception:
                key = str(list(paths))
        else:
            key = str(paths)
        return f"{tool_name}:{key}"

    if _registry().tool_in_category(tool_name, "search"):
        return f"{tool_name}:{args.get('query', '')}"

    try:
        args_str = json.dumps(args, sort_keys=True, default=str)
        hashed = hash(args_str)
    except (TypeError, ValueError):
        hashed = hash(str(args))
    return f"{tool_name}:{hashed}"


def tool_signature(tool_call: Any) -> str:
    """Build a signature from an object that looks like a tool call."""
    name = getattr(tool_call, "name", "unknown")
    arguments = getattr(tool_call, "arguments", {}) or {}
    return build_tool_signature(name, arguments)


__all__ = [
    "build_tool_signature",
    "canonical_tool_name",
    "display_tool_name",
    "expand_tool_aliases",
    "tool_aliases",
    "tool_category",
    "tool_signature",
]
