import textwrap

from ai_dev_agent.tools.code.code_edit.context import ContextGatherer, ContextGatheringOptions


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
