"""Context providers for the review command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

LOGGER = get_logger(__name__)


@dataclass
class ContextItem:
    """Represents an auxiliary context block supplied to the reviewer."""

    kind: str
    title: str
    path: str | None
    span: tuple[int, int] | None
    body: str

    def line_count(self) -> int:
        return self.body.count("\n") + 1 if self.body else 0


class ContextProvider(Protocol):
    """Protocol for providers that can enrich a review hunk with context."""

    def build_items(
        self,
        workspace_root: Path,
        file_entry: Mapping[str, Any],
    ) -> Sequence[ContextItem]: ...


def _format_context_items(items: Sequence[ContextItem]) -> str:
    lines: list[str] = ["Context:"]
    for idx, item in enumerate(items, start=1):
        header = f"#{idx} {item.kind.upper()}: {item.title}"
        lines.append(header)
        if item.body:
            lines.append(item.body.rstrip())
        lines.append("")
    return "\n".join(lines).strip()


class ContextOrchestrator:
    """Collects context from multiple providers and enforces size limits."""

    def __init__(
        self,
        providers: Sequence[ContextProvider],
        *,
        max_total_lines: int = 320,
    ) -> None:
        self._providers = list(providers)
        self._max_total_lines = max_total_lines

    def build_section(
        self,
        workspace_root: Path,
        file_entries: Sequence[Mapping[str, Any]],
    ) -> str:
        collected: list[ContextItem] = []
        seen_keys: set[tuple[str, str | None, tuple[int, int] | None, str]] = set()

        for entry in file_entries:
            for provider in self._providers:
                try:
                    items = provider.build_items(workspace_root, entry)
                except Exception as exc:  # pragma: no cover - defensive guard
                    LOGGER.debug("Context provider %s failed: %s", provider, exc)
                    continue
                for item in items:
                    key = (item.kind, item.path, item.span, item.body)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    collected.append(item)

        if not collected:
            return ""

        trimmed: list[ContextItem] = []
        total_lines = 0
        for item in collected:
            lines = item.line_count()
            if total_lines + lines > self._max_total_lines and trimmed:
                continue
            trimmed.append(item)
            total_lines += lines

        return _format_context_items(trimmed) if trimmed else ""


class SourceContextProvider:
    """Provides surrounding source code for changed hunks."""

    def __init__(
        self,
        *,
        pad_lines: int = 20,
        max_lines_per_item: int = 160,
    ) -> None:
        self._pad_lines = pad_lines
        self._max_lines_per_item = max_lines_per_item
        self._content_cache: dict[Path, tuple[float, int | None, tuple[str, ...]]] = {}

    def build_items(
        self,
        workspace_root: Path,
        file_entry: Mapping[str, Any],
    ) -> Sequence[ContextItem]:
        rel_path = file_entry.get("path")
        if not isinstance(rel_path, str):
            return []

        abs_path = (workspace_root / rel_path).resolve()
        if not abs_path.is_file():
            return []

        lines = self._get_file_lines(abs_path)
        if lines is None:
            return []
        if not lines:
            return []

        ranges = self._collect_ranges(file_entry.get("hunks", []) or [])
        if not ranges:
            return []

        merged = self._merge_ranges(ranges, len(lines))
        items: list[ContextItem] = []
        total_lines = 0

        for start, end in merged:
            if start < 1:
                start = 1
            if end > len(lines):
                end = len(lines)
            snippet_lines = []
            for idx in range(start, end + 1):
                snippet_lines.append(f"{idx:>5}: {lines[idx - 1]}")
            if not snippet_lines:
                continue
            total_lines += len(snippet_lines)
            if total_lines > self._max_lines_per_item and items:
                break
            items.append(
                ContextItem(
                    kind="code",
                    title=f"{rel_path}:{start}-{end}",
                    path=rel_path,
                    span=(start, end),
                    body="\n".join(snippet_lines),
                )
            )

        return items

    def _get_file_lines(self, abs_path: Path) -> tuple[str, ...] | None:
        try:
            stat_info = abs_path.stat()
        except OSError:
            return None

        cached = self._content_cache.get(abs_path)
        file_size = getattr(stat_info, "st_size", None)
        if cached and cached[0] == stat_info.st_mtime and cached[1] == file_size:
            return cached[2]

        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

        lines = tuple(content.splitlines())
        self._content_cache[abs_path] = (stat_info.st_mtime, file_size, lines)
        return lines

    def _collect_ranges(self, hunks: Iterable[Mapping[str, Any]]) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for hunk in hunks:
            new_start = hunk.get("new_start")
            new_count = hunk.get("new_count")
            old_start = hunk.get("old_start")
            old_count = hunk.get("old_count")

            if isinstance(new_start, int) and isinstance(new_count, int) and new_count > 0:
                begin = new_start
                end = new_start + max(new_count - 1, 0)
            elif isinstance(old_start, int) and isinstance(old_count, int) and old_count > 0:
                begin = old_start
                end = old_start + max(old_count - 1, 0)
            else:
                continue

            begin -= self._pad_lines
            end += self._pad_lines
            ranges.append((begin, end))
        return ranges

    @staticmethod
    def _merge_ranges(ranges: Sequence[tuple[int, int]], max_line: int) -> list[tuple[int, int]]:
        if not ranges:
            return []
        ordered = sorted(ranges, key=lambda item: item[0])
        merged: list[tuple[int, int]] = []
        current_start, current_end = ordered[0]

        for start, end in ordered[1:]:
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged.append((max(1, current_start), min(max_line, current_end)))
                current_start, current_end = start, end

        merged.append((max(1, current_start), min(max_line, current_end)))
        return merged


__all__ = [
    "ContextItem",
    "ContextOrchestrator",
    "ContextProvider",
    "SourceContextProvider",
]
