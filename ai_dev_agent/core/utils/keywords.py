"""Utility functions for extracting task and query keywords."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_IDENTIFIER_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_SPECIAL_TERMS_PATTERN = re.compile(
    r"\b(?:test|tests|pytest|unittest|function|class|module|import|export|api|database|sql|json|http|rest|graphql)\b",
    re.IGNORECASE,
)
_SPECIAL_TERMS = {
    "test",
    "tests",
    "pytest",
    "unittest",
    "function",
    "class",
    "module",
    "import",
    "export",
    "api",
    "database",
    "sql",
    "json",
    "http",
    "rest",
    "graphql",
}
_BASE_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "add",
    "create",
    "make",
    "implement",
    "fix",
    "update",
    "change",
    "modify",
    "remove",
    "delete",
    "function",
    "class",
    "method",
    "variable",
    "file",
    "code",
    "should",
    "will",
    "can",
    "could",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
}


def extract_keywords(
    text: str,
    *,
    limit: int = 10,
    include_special_terms: bool = False,
    extra_stopwords: Iterable[str] | None = None,
) -> list[str]:
    """Return notable keywords from *text* suitable for repository searches."""
    if not text:
        return []

    stops: set[str] = {word.lower() for word in _BASE_STOPWORDS}
    if extra_stopwords:
        stops.update(word.lower() for word in extra_stopwords)

    keywords: list[str] = []
    seen: set[str] = set()

    for token in _IDENTIFIER_PATTERN.findall(text):
        lower = token.lower()
        if lower in stops or len(lower) <= 2 or lower in seen:
            continue
        if not include_special_terms and lower in _SPECIAL_TERMS:
            continue
        keywords.append(lower)
        seen.add(lower)
        if len(keywords) >= limit:
            return keywords

    if include_special_terms and len(keywords) < limit:
        for match in _SPECIAL_TERMS_PATTERN.finditer(text):
            lower = match.group(0).lower()
            if lower in seen:
                continue
            keywords.append(lower)
            seen.add(lower)
            if len(keywords) >= limit:
                break

    return keywords


__all__ = ["extract_keywords"]
