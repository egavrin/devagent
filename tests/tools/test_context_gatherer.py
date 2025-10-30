from pathlib import Path

import pytest

from ai_dev_agent.tools.code.code_edit.context import (
    ContextGatherer,
    ContextGatheringOptions,
    FileContext,
)


class DummyAnalyzer:
    max_lines_per_file = 5

    def summarize_content(self, rel_path: str, content: str) -> list[str]:
        return [f"{rel_path}:{len(content)}"]


class DummyRepoMap:
    def __init__(self):
        self.calls = []

    def get_ranked_files(self, mentioned_files, mentioned_symbols, limit):
        self.calls.append((frozenset(mentioned_files), frozenset(mentioned_symbols), limit))
        return [("extra.py", 0.9)]


@pytest.fixture(autouse=True)
def disable_external_commands(monkeypatch):
    monkeypatch.setattr(ContextGatherer, "_check_command", lambda self, name: False)


def test_gather_contexts_with_structure_and_repo_map(monkeypatch, tmp_path):
    repo_map = DummyRepoMap()
    monkeypatch.setattr(
        "ai_dev_agent.tools.code.code_edit.context.RepoMapManager.get_instance",
        lambda root: repo_map,
    )

    project = tmp_path / "src"
    project.mkdir()
    main_file = project / "main.py"
    main_file.write_text("from .helpers import Helper\n\nclass Controller:\n    pass\n")
    extra_file = project / "extra.py"
    extra_file.write_text("def extra():\n    return 42\n")

    options = ContextGatheringOptions(max_files=5, include_related_files=True)
    gatherer = ContextGatherer(project, options)
    gatherer._structure_analyzer = DummyAnalyzer()

    contexts = gatherer.gather_contexts(
        ["main.py"],
        task_description="Investigate Helper usage",
        keywords=["Helper"],
        chat_files=[main_file],
    )

    paths = [ctx.path.name for ctx in contexts]
    assert "main.py" in paths
    assert "extra.py" in paths
    assert any(ctx.reason == "project_structure_summary" for ctx in contexts)
    assert repo_map.calls, "expected RepoMap to be consulted"


def test_search_and_symbol_fallback(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    target = root / "module.py"
    target.write_text("value = 1\n\ndef finder():\n    return value\n")

    gatherer = ContextGatherer(
        root,
        ContextGatheringOptions(
            include_related_files=False, include_structure_summary=False, use_repo_map=False
        ),
    )
    gatherer._structure_analyzer = DummyAnalyzer()

    matches = gatherer.search_files("finder")
    assert "module.py" in matches

    symbols = gatherer.find_symbol_references("finder")
    assert symbols == [("module.py", 3)]


def test_size_limits_and_exclude_patterns(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    include = root / "include.txt"
    include.write_text("a" * 50)
    excluded = root / "build"
    excluded.mkdir()
    (excluded / "artifact.log").write_text("log")

    gatherer = ContextGatherer(root, ContextGatheringOptions(max_files=1, max_total_size=30))
    gatherer._structure_analyzer = DummyAnalyzer()

    contexts = [
        FileContext(path=include, content="a" * 40, relevance_score=1.0),
        FileContext(path=include, content="b" * 40, relevance_score=0.5),
    ]
    limited = gatherer._apply_size_limits(contexts)
    assert len(limited) == 1

    assert gatherer._should_include_file(excluded / "artifact.log") is False
