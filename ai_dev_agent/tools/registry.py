"""Tool registry and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

from jsonschema import Draft7Validator

from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

LOGGER = get_logger(__name__)


@dataclass
class ToolContext:
    """Runtime context passed to tool handlers."""

    repo_root: Path
    settings: Any
    sandbox: Any
    devagent_config: Any = None
    metrics_collector: Any = None
    extra: dict[str, Any] | None = None


@dataclass
class ToolSpec:
    """Metadata for a registered tool."""

    name: str
    handler: Callable[[Mapping[str, Any], ToolContext], Mapping[str, Any]]
    request_schema_path: Path | None
    response_schema_path: Path | None
    description: str = ""
    display_name: str | None = None
    category: str | None = None


class ToolRegistry:
    """Registry that manages tool specifications and validation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._display_names: dict[str, str] = {}
        self._categories: dict[str, str] = {}
        self._category_members: dict[str, set[str]] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            LOGGER.debug("Overwriting existing tool registration for %s", spec.name)
        self._tools[spec.name] = spec
        self._rebuild_indices()

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def available(self) -> Iterable[str]:
        return sorted(self._tools.keys())

    def invoke(
        self, name: str, payload: Mapping[str, Any], context: ToolContext
    ) -> Mapping[str, Any]:
        spec = self.get(name)

        # Validate request payload if schema is provided
        if spec.request_schema_path is not None:
            validator = _load_validator(spec.request_schema_path)
            errors = sorted(validator.iter_errors(payload), key=lambda exc: exc.path)
            if errors:
                first = errors[0]
                raise ValueError(f"Invalid input for {name}: {first.message}")

        result = spec.handler(payload, context)

        # Validate response if schema is provided
        if spec.response_schema_path is not None:
            validator_out = _load_validator(spec.response_schema_path)
            errors_out = sorted(validator_out.iter_errors(result), key=lambda exc: exc.path)
            if errors_out:
                first_out = errors_out[0]
                raise ValueError(f"Tool {name} returned invalid response: {first_out.message}")

        return result

    # Metadata helpers --------------------------------------------------

    def canonical_name(self, name: str | None) -> str:
        if not name:
            return "generic"
        if name in self._tools:
            return name
        return name

    def display_name(self, name: str | None) -> str:
        canonical = self.canonical_name(name)
        return self._display_names.get(canonical, canonical)

    def category(self, name: str | None) -> str:
        canonical = self.canonical_name(name)
        return self._categories.get(canonical, "generic")

    def aliases(self, name: str | None, *, include_canonical: bool = True) -> tuple[str, ...]:
        canonical = self.canonical_name(name)
        if canonical not in self._tools:
            return (canonical,) if include_canonical and canonical != "generic" else ()
        return (canonical,) if include_canonical else ()

    def aliases_by_category(self, category: str) -> tuple[str, ...]:
        members = self._category_members.get(category, set())
        return tuple(sorted(members))

    def tool_in_category(self, name: str | None, category: str) -> bool:
        canonical = self.canonical_name(name)
        return canonical in self._categories and self._categories[canonical] == category

    # Internal -----------------------------------------------------------

    def _rebuild_indices(self) -> None:
        self._display_names.clear()
        self._categories.clear()
        self._category_members.clear()

        for canonical, spec in self._tools.items():
            display_name = spec.display_name or canonical
            category = spec.category or "generic"

            self._display_names[canonical] = display_name
            self._categories[canonical] = category

            members = self._category_members.setdefault(category, set())
            members.add(canonical)


@lru_cache(maxsize=64)
def _load_validator(schema_path: Path) -> Draft7Validator:
    with schema_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Draft7Validator(data)


# Global registry instance -------------------------------------------------

registry = ToolRegistry()


__all__ = ["ToolContext", "ToolRegistry", "ToolSpec", "registry"]
