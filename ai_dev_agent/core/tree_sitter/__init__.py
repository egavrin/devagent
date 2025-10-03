"""Shared tree-sitter infrastructure used across the agent.

This module centralizes language detection, parser lifecycle management, and
utility helpers so that tools can share the same configuration instead of
re-implementing their own variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from ai_dev_agent.core.utils.logger import get_logger

from .queries import (
    AST_QUERY_TEMPLATES,
    SUMMARY_QUERY_TEMPLATES,
    build_capture_query,
    build_field_capture_query,
    get_ast_query,
    get_summary_queries,
    iter_ast_queries,
    normalise_language,
)

LOGGER = get_logger(__name__)

try:  # Optional dependency – keep imports lazy-friendly
    try:
        import distutils  # type: ignore  # noqa: F401 - ensure legacy module exists
    except ModuleNotFoundError:  # pragma: no cover - Python 3.12 removed distutils
        # tree-sitter still imports ``distutils`` in its public module. When running
        # on Python versions where it was removed we shim the module using the
        # compatibility package exposed by ``setuptools`` so downstream imports
        # continue to work without requiring users to install anything else.
        import importlib
        import sys

        try:
            _distutils = importlib.import_module("setuptools._distutils")
        except ModuleNotFoundError:  # pragma: no cover - setuptools missing
            raise RuntimeError(
                "setuptools is required to provide the legacy distutils shim used by "
                "tree-sitter. Install setuptools before importing the agent."
            ) from None
        else:
            sys.modules.setdefault("distutils", _distutils)
            # Expose the submodules that tree-sitter currently relies on.
            for _name in ("ccompiler", "errors", "sysconfig"):
                try:
                    _mod = importlib.import_module(f"setuptools._distutils.{_name}")
                except ModuleNotFoundError:  # pragma: no cover - partial shim
                    continue
                sys.modules.setdefault(f"distutils.{_name}", _mod)

    from tree_sitter import Parser  # type: ignore
    from tree_sitter_languages import get_language, get_parser  # type: ignore
except Exception:  # pragma: no cover - dependency is optional in production
    Parser = None  # type: ignore
    get_language = None  # type: ignore
    get_parser = None  # type: ignore


# ---------------------------------------------------------------------------
# Language metadata and detection
# ---------------------------------------------------------------------------

# Canonical language identifiers keyed by file suffix. Both tools previously
# kept their own copies – the new shared mapping prevents divergence.
EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    # Python / JS / TS
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    # C / C++
    ".c": "c",
    ".h": "c",  # will be refined with content heuristics
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    # Other common languages already supported by the tools
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cs": "c_sharp",
    ".php": "php",
}

# Heuristic indicators used for language detection when the extension is not
# conclusive (e.g. shared headers, plain `.h` files, or extensionless scripts).
LANGUAGE_INDICATORS: Dict[str, Iterable[str]] = {
    "cpp": [
        "class ",
        "namespace ",
        "template<",
        "std::",
        "public:",
        "private:",
        "virtual ",
        "override",
        "using namespace",
        "::",
    ],
    "c": [
        "typedef struct",
        "#include <stdio.h>",
        "FILE *",
        "malloc(",
        "free(",
        "printf(",
        "scanf(",
        "void main",
    ],
    "python": [
        "def ",
        "class ",
        "import ",
        "from ",
        "__init__",
        "self.",
        "@property",
        "async def",
        "if __name__",
    ],
    "javascript": [
        "function ",
        "const ",
        "let ",
        "var ",
        "=>",
        "async function",
        "export ",
        "import ",
        "document.",
        "window.",
    ],
    "typescript": [
        "interface ",
        "type ",
        "enum ",
        ": string",
        ": number",
        "export interface",
        "<T>",
        "as ",
        ": void",
    ],
    "java": [
        "public class",
        "private ",
        "protected ",
        "extends ",
        "implements ",
        "package ",
        "@Override",
        "public static void main",
    ],
    "go": [
        "func ",
        "package ",
        "import (",
        "type ",
        "struct {",
        "interface {",
        "defer ",
        "go func",
        "make(",
    ],
    "rust": [
        "fn ",
        "impl ",
        "trait ",
        "struct ",
        "enum ",
        "pub ",
        "mod ",
        "use ",
        "match ",
        "let mut",
    ],
    "ruby": [
        "def ",
        "class ",
        "module ",
        "require ",
        "attr_",
        "end\n",
        "elsif",
        "@",
        "puts ",
    ],
    "c_sharp": [
        "namespace ",
        "public class",
        "private ",
        "using ",
        "override ",
        "async Task",
        "public static void Main",
    ],
    "php": [
        "<?php",
        "function ",
        "class ",
        "$this->",
        "namespace ",
        "use ",
        "trait ",
        "<?=",
        "echo ",
    ],
}


def _normalise_content(content: Optional[str | bytes]) -> str:
    if content is None:
        return ""
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="ignore")
    return content


def _match_indicator_score(text: str, language: str) -> int:
    indicators = LANGUAGE_INDICATORS.get(language, [])
    lower = text.lower()
    return sum(1 for indicator in indicators if indicator.lower() in lower)


def detect_language(path: Path, content: Optional[str | bytes] = None) -> Optional[str]:
    """Detect the most likely language for the given file."""

    suffix = path.suffix.lower()
    if suffix in EXTENSION_LANGUAGE_MAP:
        lang = EXTENSION_LANGUAGE_MAP[suffix]
        if suffix != ".h":  # `.h` is ambiguous – consider heuristics below
            return lang

    text = _normalise_content(content)

    if suffix == ".h":
        cpp_score = _match_indicator_score(text, "cpp")
        c_score = _match_indicator_score(text, "c")
        if cpp_score > c_score:
            return "cpp"
        if c_score > 0:
            return "c"
        # fall through when inconclusive so further heuristics run

    if not text:
        return EXTENSION_LANGUAGE_MAP.get(suffix)

    best_lang: Optional[str] = None
    best_score = 0
    lower = text.lower()

    for language, indicators in LANGUAGE_INDICATORS.items():
        score = sum(1 for indicator in indicators if indicator.lower() in lower)
        if score > best_score:
            best_score = score
            best_lang = language

    if best_lang:
        return best_lang

    return EXTENSION_LANGUAGE_MAP.get(suffix)


@dataclass(frozen=True)
class ParserHandle:
    """Container describing a ready-to-use parser."""

    language: str
    parser: object


class TreeSitterManager:
    """Singleton manager that caches parsers per language."""

    _instance: Optional["TreeSitterManager"] = None

    def __new__(cls) -> "TreeSitterManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._parser_factory = get_parser
        self._language_factory = get_language
        self._parser_cls = Parser
        self._parsers: Dict[str, object] = {}
        self._languages: Dict[str, object] = {}
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        return any([
            callable(self._parser_factory),
            callable(self._language_factory) and self._parser_cls is not None,
        ])

    # Public API -----------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    def parser_for_language(self, language: str) -> Optional[ParserHandle]:
        """Return a cached parser for the given language if available."""
        language = language.lower()
        if not self.available:
            return None

        if language in self._parsers:
            return ParserHandle(language, self._parsers[language])

        parser = self._create_parser(language)
        if parser is None:
            return None

        self._parsers[language] = parser
        return ParserHandle(language, parser)

    def parser_for_path(
        self, path: Path, *, content: Optional[str | bytes] = None
    ) -> Optional[ParserHandle]:
        language = detect_language(path, content)
        if language is None:
            return None
        return self.parser_for_language(language)

    def language(self, language: str) -> Optional[object]:
        """Return a cached tree-sitter language object."""

        language = language.lower()
        if not self.available or self._language_factory is None:
            # If language factory unavailable we might be using prebuilt parsers.
            parser = self._parsers.get(language)
            return getattr(parser, "language", None)

        if language in self._languages:
            return self._languages[language]

        try:
            lang_obj = self._language_factory(language)
        except Exception as exc:  # pragma: no cover - defensive fall back
            LOGGER.debug("Failed to load language object for %s: %s", language, exc)
            return None

        if lang_obj is not None:
            self._languages[language] = lang_obj

        return lang_obj

    # Internal helpers ----------------------------------------------------

    def _create_parser(self, language: str) -> Optional[object]:
        if self._parser_factory is not None:
            try:
                return self._parser_factory(language)
            except Exception as exc:  # pragma: no cover - defensive fall back
                LOGGER.debug("Failed to create parser via factory for %s: %s", language, exc)

        if self._parser_cls is None or self._language_factory is None:
            return None

        try:
            lang_obj = self._language_factory(language)
            if lang_obj is None:
                return None
            parser = self._parser_cls()
            parser.set_language(lang_obj)
            return parser
        except Exception as exc:  # pragma: no cover - defensive fall back
            LOGGER.debug("Failed to configure parser via set_language for %s: %s", language, exc)
            return None


# Global manager instance – keeps parsers alive across module usages.
MANAGER = TreeSitterManager()


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def node_text(node, source_bytes: bytes) -> str:
    """Return the UTF-8 decoded span covered by *node*."""

    start, end = node.start_byte, node.end_byte
    return source_bytes[start:end].decode("utf-8", errors="ignore")


def slice_bytes(source: str | bytes) -> bytes:
    """Ensure the input is a bytes object suitable for tree-sitter parse calls."""

    if isinstance(source, bytes):
        return source
    return source.encode("utf-8", errors="ignore")


def ensure_parser(path: Path, *, content: Optional[str | bytes] = None) -> Optional[ParserHandle]:
    """Convenience wrapper returning a parser pre-selected for *path*."""

    return MANAGER.parser_for_path(path, content=content)


def ensure_language(path: Path, *, content: Optional[str | bytes] = None) -> Optional[str]:
    """Return the detected language if supported by the shared infrastructure."""

    language = detect_language(path, content)
    if language is None:
        return None

    handle = MANAGER.parser_for_language(language)
    if handle is None:
        return None
    return handle.language


def language_object(language: str) -> Optional[object]:
    """Expose the underlying tree-sitter language object when available."""

    return MANAGER.language(language)


__all__ = [
    "AST_QUERY_TEMPLATES",
    "EXTENSION_LANGUAGE_MAP",
    "LANGUAGE_INDICATORS",
    "ParserHandle",
    "TreeSitterManager",
    "MANAGER",
    "SUMMARY_QUERY_TEMPLATES",
    "build_capture_query",
    "build_field_capture_query",
    "detect_language",
    "ensure_language",
    "ensure_parser",
    "get_ast_query",
    "get_summary_queries",
    "iter_ast_queries",
    "node_text",
    "normalise_language",
    "language_object",
    "slice_bytes",
]
