import textwrap
from pathlib import Path

import pytest

from ai_dev_agent.tools.code.code_edit.context import ContextGatherer, ContextGatheringOptions
from ai_dev_agent.tools.code.code_edit.tree_sitter_analysis import (
    TreeSitterProjectAnalyzer,
    extract_symbols_from_outline,
)


def test_gather_contexts_includes_keyword_matches(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)

    (repo / "src/module.py").write_text(
        textwrap.dedent(
            """
            def main():
                return helper()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (repo / "src/helpers.py").write_text(
        textwrap.dedent(
            """
            def helper():
                return 42
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    options = ContextGatheringOptions(max_files=3, include_structure_summary=False)
    gatherer = ContextGatherer(repo, options)

    contexts = gatherer.gather_contexts(
        ["src/module.py"],
        task_description="Update helper logic",
        keywords=["helper"],
    )

    rel_paths = {ctx.path.relative_to(repo).as_posix(): ctx.reason for ctx in contexts}

    assert "src/module.py" in rel_paths
    assert "src/helpers.py" in rel_paths
    assert any(reason.startswith("keyword_match") for reason in rel_paths.values())


def test_structure_summary_is_added_when_enabled(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)

    (repo / "pkg/sample.py").write_text(
        textwrap.dedent(
            """
            class Alpha:
                pass
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    options = ContextGatheringOptions(max_files=5, include_structure_summary=True)
    gatherer = ContextGatherer(repo, options)

    contexts = gatherer.gather_contexts(["pkg/sample.py"], keywords=["Alpha"])

    reasons = {ctx.reason for ctx in contexts}
    assert "project_structure_summary" in reasons


def test_context_contains_outline_and_symbols(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)

    (repo / "pkg/sample.py").write_text(
        textwrap.dedent(
            """
            class Foo:
                def bar(self):
                    return 1


            def baz():
                return 2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    gatherer = ContextGatherer(repo)

    contexts = gatherer.gather_contexts(["pkg/sample.py"], keywords=["Foo"])
    sample = next(
        ctx for ctx in contexts if ctx.path.relative_to(repo).as_posix() == "pkg/sample.py"
    )

    assert sample.structure_outline
    assert any("class Foo" in line for line in sample.structure_outline)
    assert {symbol.lower() for symbol in sample.symbols} >= {"foo", "bar", "baz"}


@pytest.mark.parametrize(
    ("extension", "snippet", "expected_fragment"),
    [
        (".py", "def alpha(value):\n    return value * 2\n", "function alpha"),
        (".ts", "export class Bravo {\n  render() {}\n}\n", "class Bravo"),
        (".go", "func delta() int {\n    return 0\n}\n", "func delta"),
        (".rs", "pub struct Echo {}\n", "struct Echo"),
    ],
)
def test_context_gatherer_extracts_outlines_for_multiple_languages(
    tmp_path, extension, snippet, expected_fragment
):
    repo = tmp_path / "repo"
    repo.mkdir()
    file_path = repo / f"sample{extension}"
    file_path.write_text(snippet, encoding="utf-8")

    gatherer = ContextGatherer(
        repo,
        ContextGatheringOptions(
            include_related_files=False,
            include_structure_summary=False,
            use_repo_map=False,
        ),
    )

    [ctx] = gatherer.gather_contexts([file_path.name])
    assert any(expected_fragment in line for line in ctx.structure_outline), ctx.structure_outline


def test_extract_symbols_from_outline_skips_duplicates_and_reserved_words():
    outline = [
        "   4: class Tango",
        "   8: function tango",
        "  12: fn helper_function",
        "  13: function helper_function",
        "  15: struct Example",
        "  20: enum Mode",
        "  25: type Alias = str",
        "  30: func compute",
        "  35: const VALUE = OTHER_VALUE",
    ]

    symbols = extract_symbols_from_outline(outline)

    # Should skip reserved labels and deduplicate by token text
    assert symbols == [
        "Tango",
        "tango",
        "helper_function",
        "Example",
        "Mode",
        "Alias",
        "str",
        "compute",
        "const",
        "VALUE",
        "OTHER_VALUE",
    ]
    assert symbols.count("helper_function") == 1  # identical tokens deduplicated


def test_repo_map_integration_handles_external_chat_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)

    (repo / "pkg/target.py").write_text("def useful():\n    return True\n", encoding="utf-8")
    (repo / "pkg/discovered.py").write_text("def helper():\n    return False\n", encoding="utf-8")

    gatherer = ContextGatherer(
        repo,
        ContextGatheringOptions(
            include_related_files=True,
            include_structure_summary=False,
            use_repo_map=True,
            max_files=3,
        ),
    )

    class FakeRepoMap:
        def __init__(self, items):
            self.items = items

        def get_ranked_files(self, mentioned_files, mentioned_symbols, limit):
            return self.items[:limit]

    gatherer.repo_map = FakeRepoMap([("pkg/discovered.py", 0.9)])

    external_file = Path("/tmp/outside.py")
    contexts = gatherer.gather_contexts(
        ["pkg/target.py"],
        task_description="Investigate helper usage",
        chat_files=[external_file],
    )

    rel_paths = {ctx.path.relative_to(repo).as_posix() for ctx in contexts}
    assert "pkg/target.py" in rel_paths
    assert "pkg/discovered.py" in rel_paths


def test_search_and_symbol_resolution_fall_back_to_python_logic(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    file_path = repo / "module.py"
    file_path.write_text(
        "class Sample:\n    def needle(self):\n        return 42\n", encoding="utf-8"
    )

    gatherer = ContextGatherer(
        repo,
        ContextGatheringOptions(
            include_related_files=False,
            include_structure_summary=False,
            use_repo_map=False,
        ),
    )

    gatherer._rg_available = False
    gatherer._git_available = False

    search_results = gatherer.search_files("needle")
    assert search_results == ["module.py"]

    hits = gatherer.find_symbol_references("needle")
    assert hits == [("module.py", 2)]


def test_tree_sitter_project_analyzer_supports_supported_suffixes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    analyzer = TreeSitterProjectAnalyzer(repo, max_lines_per_file=3)
    assert analyzer.available is True
    assert ".py" in analyzer.supported_suffixes

    outline = analyzer.summarize_content(
        "example.ts",
        "export interface Thing {\n  execute(): void\n}\n",
    )
    assert outline[:1] == ["   1: interface Thing"]
