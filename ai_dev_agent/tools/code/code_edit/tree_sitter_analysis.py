"""Lightweight text-based project structure summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

LOGGER = get_logger(__name__)

# Patterns capture lightweight structure hints across common languages.
_OUTLINE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("class", re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("function", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("function", re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)")),
    (
        "function",
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\(\s*"
        ),
    ),
    ("interface", re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("type", re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_][A-Za-z0-9_]*)\s*=")),
    ("enum", re.compile(r"^\s*(?:export\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("struct", re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("fn", re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("func", re.compile(r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\(")),
)

_SUPPORTED_SUFFIXES = (
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".mjs",
    ".c",
    ".cpp",
    ".cc",
    ".h",
    ".hpp",
    ".hh",
    ".java",
    ".go",
    ".rs",
)


@dataclass
class ParsedFileSummary:
    """Structured summary extracted from a source file."""

    path: str
    outline: list[str]

    def to_markdown(self) -> str:
        header = f"### {self.path}"
        body = "\n".join(self.outline)
        return f"{header}\n{body}" if body else header


class SimpleProjectAnalyzer:
    """Generate structural summaries using text heuristics."""

    SUPPORTED_SUFFIXES = _SUPPORTED_SUFFIXES

    def __init__(self, repo_root: Path, max_files: int = 8, max_lines_per_file: int = 12) -> None:
        self.repo_root = Path(repo_root)
        self.max_files = max_files
        self.max_lines_per_file = max_lines_per_file

    @property
    def available(self) -> bool:
        return True

    @property
    def supported_suffixes(self) -> list[str]:
        return list(self.SUPPORTED_SUFFIXES)

    def summarize_content(self, rel_path: str, content: str) -> list[str]:
        """Return a structural outline for a single file."""
        return _outline_from_text(content, self.max_lines_per_file)

    def build_project_summary(self, file_entries: Iterable[tuple[str, str]]) -> str | None:
        """Return a markdown summary describing the provided files."""
        summaries: list[ParsedFileSummary] = []

        for rel_path, content in file_entries:
            outline = self.summarize_content(rel_path, content)
            if not outline:
                continue
            summaries.append(ParsedFileSummary(rel_path, outline[: self.max_lines_per_file]))
            if len(summaries) >= self.max_files:
                break

        if not summaries:
            return None

        header = [
            "# Project Structure",
            "",
            "Generated with lightweight text heuristics to orient the language model.",
            "",
        ]
        body = "\n\n".join(summary.to_markdown() for summary in summaries)
        return "\n".join(header) + body


def _outline_from_text(content: str, max_lines: int) -> list[str]:
    outline: list[str] = []
    seen_lines = set()

    for idx, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        for label, pattern in _OUTLINE_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            symbol = match.group(1)
            entry = f"{idx:>4}: {label} {symbol}"
            if entry not in seen_lines:
                outline.append(entry)
                seen_lines.add(entry)
            break
        if len(outline) >= max_lines:
            break

    if outline:
        return outline

    # Fallback: capture up to max_lines non-empty snippets from the top of the file.
    for idx, line in enumerate(content.splitlines(), 1):
        snippet = line.strip()
        if not snippet:
            continue
        outline.append(f"{idx:>4}: {snippet[:80]}")
        if len(outline) >= max_lines:
            break

    return outline


def extract_symbols_from_outline(outline: Sequence[str]) -> list[str]:
    """Extract identifier-like tokens from an outline."""
    symbols: list[str] = []
    seen = set()

    for line in outline:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line)
        for token in tokens:
            if token in {"class", "function", "interface", "type", "enum", "struct", "fn", "func"}:
                continue
            if token not in seen:
                seen.add(token)
                symbols.append(token)

    return symbols


TreeSitterProjectAnalyzer = SimpleProjectAnalyzer

__all__ = [
    "ParsedFileSummary",
    "SimpleProjectAnalyzer",
    "TreeSitterProjectAnalyzer",
    "extract_symbols_from_outline",
]
