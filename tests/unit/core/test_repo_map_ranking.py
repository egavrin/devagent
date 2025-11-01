"""Tests for RepoMap ranking and PageRank utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_dev_agent.core import repo_map as repo_map_module
from ai_dev_agent.core.repo_map import FileInfo, RepoMap


def create_repo_with_files(tmp_path: Path) -> RepoMap:
    repo = RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)

    def add_file(
        path: str,
        *,
        symbols: list[str],
        symbols_used: list[str] | None = None,
        imports: list[str] | None = None,
        size: int = 4000,
        language: str = "python",
    ) -> FileInfo:
        file_path = Path(path)
        full_path = tmp_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("# stub\n", encoding="utf-8")

        info = FileInfo(
            path=path,
            size=size,
            modified_time=1.0,
            language=language,
            symbols=list(symbols),
            imports=list(imports or []),
            exports=[],
            dependencies=set(),
            references={},
            symbols_used=list(symbols_used or []),
            file_name=file_path.name,
            file_stem=file_path.stem,
            path_parts=tuple(file_path.parts),
        )
        repo.context.files[path] = info
        return info

    add_file(
        "src/core/main.py",
        symbols=["ImportantClass", "CoreHelper"],
        symbols_used=["UtilFunc"],
        imports=["os", "sys", "collections", "itertools", "json", "typing"],
        size=8000,
    )
    add_file(
        "src/utils/helpers.py",
        symbols=["UtilFunc", "format_value"],
        symbols_used=["ImportantClass"],
        size=5000,
    )
    add_file(
        "tests/test_main.py",
        symbols=["test_flow"],
        symbols_used=["ImportantClass"],
        size=12000,
    )

    repo._rebuild_indices()
    return repo


@pytest.fixture
def ranking_repo(tmp_path: Path) -> RepoMap:
    return create_repo_with_files(tmp_path)


def test_normalize_mentions_trims_and_limits(
    ranking_repo: RepoMap, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ranking_repo, "MAX_FASTPATH_MENTIONS", 3, raising=False)
    monkeypatch.setattr(ranking_repo, "MAX_DIRECTORY_MATCHES", 1, raising=False)

    mentions = {
        "src/core/main.py",
        "src/utils/helpers.py",
        "docs/README.md",
        "src/core/",
        "tests/test_main.py",
    }

    trimmed, names, stems, directories = ranking_repo._normalize_mentions(mentions)

    assert len(trimmed) == 3  # enforcing the MAX_FASTPATH_MENTIONS cap
    assert "main.py" in names
    assert "main" in stems
    assert len(directories) == 1


def test_quick_rank_by_symbols_scores_matches(ranking_repo: RepoMap) -> None:
    mentioned_files = {"src/utils/helpers.py"}
    mentioned_symbols = {"ImportantClass", "UtilFunc"}
    directory_mentions = ("src/core",)

    results = ranking_repo._quick_rank_by_symbols(
        mentioned_files,
        mentioned_symbols,
        max_files=5,
        directory_mentions=directory_mentions,
    )

    assert results
    assert any(path == "src/core/main.py" and score >= 100.0 for path, score in results)
    assert any(path == "src/utils/helpers.py" for path, score in results)


def test_get_ranked_files_uses_fast_path(
    ranking_repo: RepoMap, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_compute(*args, **kwargs):
        raise AssertionError("compute_pagerank should not be called for fast-path queries")

    monkeypatch.setattr(ranking_repo, "compute_pagerank", fail_compute)

    results = ranking_repo.get_ranked_files(set(), {"ImportantClass"})

    assert results
    assert results[0][0] == "src/core/main.py"


def test_get_ranked_files_falls_back_to_pagerank(
    ranking_repo: RepoMap, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranking_repo.context.pagerank_scores = {}

    captured = {"called": 0}

    def fake_compute(pers=None, *, cache_results=True, use_edge_distribution=True):
        captured["called"] += 1
        ranking_repo.context.pagerank_scores = {
            "src/core/main.py": 0.7,
            "src/utils/helpers.py": 0.2,
            "tests/test_main.py": 0.1,
        }
        return ranking_repo.context.pagerank_scores

    monkeypatch.setattr(ranking_repo, "compute_pagerank", fake_compute)

    results = ranking_repo.get_ranked_files({"src"}, set())

    assert captured["called"] >= 1
    assert results[0][0] == "src/core/main.py"


def test_get_file_summary_and_dependencies(ranking_repo: RepoMap) -> None:
    # Create a verbose file to exercise summary truncation.
    ranking_repo.context.files["src/summary.py"] = FileInfo(
        path="src/summary.py",
        size=100,
        modified_time=1.0,
        language="python",
        symbols=[f"Symbol{i}" for i in range(12)],
        imports=[f"pkg{i}" for i in range(7)],
        exports=[],
        dependencies={"src/utils/helpers.py"},
        references={},
        symbols_used=[],
        file_name="summary.py",
        file_stem="summary",
        path_parts=("src", "summary.py"),
    )

    summary = ranking_repo.get_file_summary("src/summary.py")
    assert summary is not None
    assert "... and 2 more" in summary
    assert "... and 2 more" in summary.splitlines()[-1]

    deps = ranking_repo.get_dependencies("src/summary.py")
    assert deps == {"src/utils/helpers.py"}
    assert ranking_repo.get_dependencies("missing.py") == set()


def test_build_dependency_graph_sets_edges(ranking_repo: RepoMap) -> None:
    graph = ranking_repo.build_dependency_graph()

    assert graph.has_edge("src/core/main.py", "src/utils/helpers.py")
    assert ranking_repo.context.files["src/core/main.py"].dependencies == {"src/utils/helpers.py"}
    assert ranking_repo.context.dependency_graph is graph


def test_compute_pagerank_handles_convergence_failure(
    ranking_repo: RepoMap, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = ranking_repo.build_dependency_graph()
    calls = {"count": 0}

    def fake_pagerank(G, alpha=0.85, personalization=None, weight=None):
        calls["count"] += 1
        if calls["count"] == 1 and weight == "weight":
            raise repo_map_module.nx.PowerIterationFailedConvergence(10)
        # Return a simple uniform distribution
        return {node: 1.0 / max(1, G.number_of_nodes()) for node in G.nodes()}

    monkeypatch.setattr(repo_map_module.nx, "pagerank", fake_pagerank)

    scores = ranking_repo.compute_pagerank()

    assert set(scores.keys()) == set(graph.nodes)
    assert calls["count"] == 2  # first call raised, second succeeded


def test_get_file_rank_triggers_pagerank_when_missing(
    ranking_repo: RepoMap, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranking_repo.context.pagerank_scores = {}
    called = {"count": 0}

    def fake_compute(personalization=None, *, cache_results=True, use_edge_distribution=True):
        called["count"] += 1
        ranking_repo.context.pagerank_scores = {"src/core/main.py": 0.42}
        return ranking_repo.context.pagerank_scores

    monkeypatch.setattr(ranking_repo, "compute_pagerank", fake_compute)

    rank = ranking_repo.get_file_rank("src/core/main.py")

    assert called["count"] == 1
    assert rank == pytest.approx(0.42)


def test_get_ranked_files_pagerank_personalization(
    ranking_repo: RepoMap, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranking_repo.build_dependency_graph()
    base_scores = {
        "src/core/main.py": 0.6,
        "src/utils/helpers.py": 0.3,
        "tests/test_main.py": 0.1,
    }
    ranking_repo.context.pagerank_scores = base_scores.copy()

    captured = {"personalization": None, "calls": 0}

    def fake_compute(personalization=None, *, cache_results=True, use_edge_distribution=True):
        captured["calls"] += 1
        captured["personalization"] = personalization
        if personalization:
            return {key: value * 2 for key, value in base_scores.items()}
        return base_scores

    monkeypatch.setattr(ranking_repo, "compute_pagerank", fake_compute)

    results = ranking_repo.get_ranked_files_pagerank(
        mentioned_files={"src/utils/helpers.py"},
        mentioned_symbols={"UtilFunc"},
        conversation_files={"src/utils/helpers.py"},
        max_files=2,
    )

    assert captured["calls"] == 1
    assert captured["personalization"] == {"src/utils/helpers.py": 100.0}
    top_path, score, metadata = results[0]
    assert top_path == "src/utils/helpers.py"
    assert metadata["language"] == "python"
    assert "incoming_edges" in metadata and "outgoing_edges" in metadata
