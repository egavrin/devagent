"""Unit tests for tree-sitter parser error handling and large file behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


class _StubQuery:
    def __init__(self, captures: list[tuple[SimpleNamespace, str]]) -> None:
        self._captures = captures

    def captures(self, _root) -> list[tuple[SimpleNamespace, str]]:
        return self._captures


class _StubLanguage:
    def __init__(self, captures: list[tuple[SimpleNamespace, str]]) -> None:
        self._captures = captures

    def query(self, _query_text: str) -> _StubQuery:
        return _StubQuery(self._captures)


def _build_content_and_captures(
    specs: list[tuple[str, str]]
) -> tuple[bytes, list[tuple[SimpleNamespace, str]]]:
    """Create content bytes and matching capture nodes for stub queries."""
    buffer = bytearray()
    captures: list[tuple[SimpleNamespace, str]] = []
    for text, capture in specs:
        encoded = text.encode("utf-8")
        start = len(buffer)
        buffer.extend(encoded)
        node = SimpleNamespace(start_byte=start, end_byte=start + len(encoded))
        captures.append((node, capture))
        buffer.append(0x0A)  # newline
    if buffer:
        buffer.pop()  # remove trailing newline
    return bytes(buffer), captures


def _run_language_extraction(
    parser: TreeSitterParser,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    language_key: str,
    specs: list[tuple[str, str]],
    *extra_args,
) -> tuple[list[str], list[str], list[str]]:
    """Invoke a language extraction helper with stubbed tree-sitter bindings."""
    content, captures = _build_content_and_captures(specs)
    tree = SimpleNamespace(root_node=object())

    def fake_get_language(name: str) -> _StubLanguage:
        assert name == language_key
        return _StubLanguage(captures)

    monkeypatch.setattr(parser_module.tsl, "get_language", fake_get_language, raising=False)
    parser.compiled_queries.clear()
    method = getattr(parser, method_name)
    result = method(tree, content, *extra_args)
    # Second call exercises cached query path
    method(tree, content, *extra_args)
    return result


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


def test_get_parser_returns_none_without_tree_sitter(monkeypatch: pytest.MonkeyPatch):
    """When tree-sitter bindings are unavailable, get_parser should return None."""
    monkeypatch.setattr(parser_module, "TREE_SITTER_AVAILABLE", False, raising=False)
    parser = TreeSitterParser()

    python_parser = parser.get_parser("python")

    assert python_parser is None


def test_extract_symbols_returns_empty_when_tree_sitter_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """extract_symbols should short-circuit to empty results if tree-sitter is missing."""
    monkeypatch.setattr(parser_module, "TREE_SITTER_AVAILABLE", False, raising=False)
    parser = TreeSitterParser()
    source_file = _write_source(tmp_path, "simple.py", "print('hi')\n")

    result = parser.extract_symbols(source_file, "python")

    assert result == {"symbols": [], "imports": [], "references": []}
    assert parser.stats["unsupported_language"] == 0


def test_extract_symbols_handles_stat_errors(
    parser: TreeSitterParser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Filesystem stat errors should be counted and return the empty result."""
    source_file = _write_source(tmp_path, "missing.py", "print('missing')\n")

    def fail_stat(self):
        raise OSError("stat failed")

    monkeypatch.setattr(Path, "stat", fail_stat, raising=False)

    result = parser.extract_symbols(source_file, "python")

    assert result == {"symbols": [], "imports": [], "references": []}
    assert parser.stats["stat_errors"] == 1


@pytest.mark.parametrize(
    ("method_name", "language_key", "specs", "expected", "extra"),
    [
        (
            "_extract_python_symbols",
            "python",
            [
                ("Foo", "class"),
                ("bar", "function"),
                ("import os", "import"),
                ("call", "function_call"),
            ],
            (["Foo", "bar"], ["import os"], ["call"]),
            (),
        ),
        (
            "_extract_javascript_symbols",
            "typescript",
            [
                ("Widget", "class"),
                ("handle", "method"),
                ("URL", "variable"),
                ("import lib", "import"),
                ("runTask", "function_call"),
            ],
            (["Widget", "handle", "URL"], ["import lib"], ["runTask"]),
            ("typescript",),
        ),
        (
            "_extract_javascript_symbols",
            "javascript",
            [
                ("Widget", "class"),
                ("handler", "method"),
                ("foo", "variable"),
                ("import lib", "import"),
                ("callFn", "function_call"),
            ],
            (["Widget", "handler"], ["import lib"], ["callFn"]),
            ("javascript",),
        ),
        (
            "_extract_cpp_symbols",
            "cpp",
            [
                ("Widget", "class"),
                ("std", "namespace_name"),
                ("doThing", "function"),
                ("<iostream>", "include"),
                ("invoke", "function_call"),
            ],
            (["Widget", "std", "doThing"], ["iostream"], ["invoke"]),
            (),
        ),
        (
            "_extract_java_symbols",
            "java",
            [
                ("Controller", "class"),
                ("handle", "method"),
                ("import service", "import"),
                ("dispatch", "method_call"),
            ],
            (["Controller", "handle"], ["import service"], ["dispatch"]),
            (),
        ),
        (
            "_extract_go_symbols",
            "go",
            [
                ("Process", "function"),
                ("WithContext", "method"),
                ("Client", "type"),
                ("import fmt", "import"),
                ("Call", "function_call"),
            ],
            (["Process", "WithContext", "Client"], ["import fmt"], ["Call"]),
            (),
        ),
        (
            "_extract_rust_symbols",
            "rust",
            [
                ("process", "function"),
                ("Worker", "struct"),
                ("use std::io", "use"),
                ("call_now", "function_call"),
            ],
            (["process", "Worker"], ["use std::io"], ["call_now"]),
            (),
        ),
        (
            "_extract_php_symbols",
            "php",
            [
                ("Controller", "class"),
                ("namespace App", "namespace"),
                ("handleRequest", "method"),
                ("use Service", "use"),
                ("dispatch", "function_call"),
            ],
            (["Controller", "namespace App", "handleRequest"], ["use Service"], ["dispatch"]),
            (),
        ),
        (
            "_extract_csharp_symbols",
            "c_sharp",
            [
                ("Controller", "class"),
                ("Utilities", "namespace"),
                ("System", "using"),
                ("Invoke", "function_call"),
            ],
            (["Controller", "Utilities"], ["System"], ["Invoke"]),
            (),
        ),
        (
            "_extract_ruby_symbols",
            "ruby",
            [
                ("Service", "class"),
                ("Helpers", "module"),
                ("perform", "method"),
                ("singleton", "singleton_method"),
                ("run", "method_call"),
            ],
            (["Service", "Helpers", "perform", "singleton"], [], ["run"]),
            (),
        ),
        (
            "_extract_kotlin_symbols",
            "kotlin",
            [
                ("Widget", "class"),
                ("render", "function"),
                ("import foo.Bar", "import"),
                ("dispatch", "function_call"),
            ],
            (["Widget", "render"], ["import foo.Bar"], ["dispatch"]),
            (),
        ),
        (
            "_extract_scala_symbols",
            "scala",
            [
                ("Widget", "class"),
                ("Helpers", "object"),
                ("Formatter", "trait"),
                ("render", "function"),
                ("invoke", "function_decl"),
                ("import foo.bar", "import"),
                ("runNow", "function_call"),
            ],
            (
                ["Widget", "Helpers", "Formatter", "render", "invoke"],
                ["import foo.bar"],
                ["runNow"],
            ),
            (),
        ),
        (
            "_extract_bash_symbols",
            "bash",
            [
                ("deploy", "function"),
                ("custom-cli", "command"),
                ("echo", "command"),
            ],
            (["deploy"], [], ["custom-cli"]),
            (),
        ),
        (
            "_extract_lua_symbols",
            "lua",
            [
                ("do_work", "function"),
                ("helper", "local_function"),
                ("call_now", "function_call"),
            ],
            (["do_work", "helper"], [], ["call_now"]),
            (),
        ),
    ],
)
def test_language_specific_extraction(
    parser: TreeSitterParser,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    language_key: str,
    specs: list[tuple[str, str]],
    expected: tuple[list[str], list[str], list[str]],
    extra: tuple[str, ...],
) -> None:
    result = _run_language_extraction(parser, monkeypatch, method_name, language_key, specs, *extra)
    assert result == expected


def test_extract_generic_symbols_deduplicates(tmp_path: Path) -> None:
    """Generic extraction should traverse children and deduplicate identifiers."""
    content = b"foo\nBar\n_skip\nfoo\n"
    node_foo = SimpleNamespace(type="identifier", start_byte=0, end_byte=3, children=[])
    node_bar = SimpleNamespace(type="type_identifier", start_byte=4, end_byte=7, children=[])
    node_skip = SimpleNamespace(type="field_identifier", start_byte=8, end_byte=13, children=[])
    node_dup = SimpleNamespace(type="identifier", start_byte=14, end_byte=17, children=[])
    root = SimpleNamespace(
        type="root",
        start_byte=0,
        end_byte=len(content),
        children=[node_foo, node_bar, node_skip, node_dup],
    )
    parser = TreeSitterParser()

    symbols = parser._extract_generic_symbols(SimpleNamespace(root_node=root), content)

    assert symbols == ["foo", "Bar"]
