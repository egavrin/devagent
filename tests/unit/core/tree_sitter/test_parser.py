"""Unit tests for tree-sitter parser error handling and large file behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_dev_agent.core.tree_sitter import parser as parser_module
from ai_dev_agent.core.tree_sitter.parser import TreeSitterParser


def _write_source(tmp_path: Path, filename: str, content: str) -> Path:
    path = tmp_path / filename
    path.write_text(content)
    return path


class _DummyTree:
    def __init__(self, has_error: bool) -> None:
        self.root_node = type("Root", (), {"has_error": has_error})()


class _ParseErrorParser:
    def parse(self, _: bytes):
        raise ValueError("simulated parse failure")


class _TreeParser:
    def __init__(self, tree: _DummyTree) -> None:
        self._tree = tree

    def parse(self, _: bytes) -> _DummyTree:
        return self._tree


@pytest.fixture()
def parser(monkeypatch: pytest.MonkeyPatch) -> TreeSitterParser:
    """Create a parser instance with tree-sitter availability forced on."""
    monkeypatch.setattr(parser_module, "TREE_SITTER_AVAILABLE", True, raising=False)
    return TreeSitterParser()


def test_parser_error_recovery(
    parser: TreeSitterParser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Tree-sitter failures should be handled gracefully without raising."""
    broken_file = _write_source(tmp_path, "broken.py", "def broken(")
    original_get_parser = parser.get_parser

    # Malformed syntax: tree reports errors and should short-circuit.
    monkeypatch.setattr(parser, "get_parser", lambda _: _TreeParser(_DummyTree(has_error=True)))
    malformed_result = parser.extract_symbols(broken_file, "python")
    assert malformed_result == {"symbols": [], "imports": [], "references": []}
    assert parser.stats["invalid_tree"] == 1

    # Incomplete files / parser exceptions should also fail gracefully.
    monkeypatch.setattr(parser, "get_parser", lambda _: _ParseErrorParser())
    parse_error_result = parser.extract_symbols(broken_file, "python")
    assert parse_error_result == {"symbols": [], "imports": [], "references": []}
    assert parser.stats["parse_errors"] == 1

    # Unsupported languages should return the empty structure without parsing.
    monkeypatch.setattr(parser, "get_parser", original_get_parser, raising=False)
    unsupported_result = parser.extract_symbols(broken_file, "unsupported_language")
    assert unsupported_result == {"symbols": [], "imports": [], "references": []}
    assert parser.stats["unsupported_language"] == 1


def test_parser_large_files(
    parser: TreeSitterParser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Very large files should be skipped to avoid expensive tree-sitter work."""
    large_file = _write_source(tmp_path, "large.py", "print('hello')\n" * 32)

    # Force a small size threshold to guarantee the file is treated as large.
    parser.max_file_size = 128
    call_tracker = {"parse_called": False}

    monkeypatch.setattr(
        parser,
        "_extract_python_symbols",
        lambda *args, **kwargs: ([], [], []),
    )

    class _SentinelParser:
        def parse(self, _: bytes):
            call_tracker["parse_called"] = True
            return _DummyTree(has_error=False)

    monkeypatch.setattr(parser, "get_parser", lambda _: _SentinelParser())

    result = parser.extract_symbols(large_file, "python")
    assert result == {"symbols": [], "imports": [], "references": []}
    assert call_tracker["parse_called"] is False
    assert parser.stats["skipped_large"] == 1
