import json
from pathlib import Path

import pytest

from ai_dev_agent.core.repo_map import FileInfo, RepoMap


def _create_sample_repo(root: Path) -> None:
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "util.py").write_text(
        "class Helper:\n"
        "    def greet(self):\n"
        "        return 'hi'\n"
        "\n"
        "def helper_function(value):\n"
        "    return Helper()\n"
    )
    (pkg / "app.py").write_text(
        "from pkg.util import Helper\n"
        "\n"
        "class App:\n"
        "    def run(self):\n"
        "        h = Helper()\n"
        "        return h.greet()\n"
    )


def test_repo_map_scans_and_loads_cache(tmp_path):
    _create_sample_repo(tmp_path)

    repo_map = RepoMap(root_path=tmp_path, cache_enabled=True, use_tree_sitter=False)
    repo_map.scan_repository(force=True)

    assert "pkg/app.py" in repo_map.context.files
    util_info = repo_map.context.files["pkg/util.py"]
    assert "Helper" in util_info.symbols
    assert util_info.language == "python"

    cache_dir = tmp_path / ".devagent_cache"
    cached_files = [path for path in cache_dir.glob("repo_map.*") if path.is_file()]
    assert cached_files, "expected repo map cache to be written"

    # Instantiate a new RepoMap to ensure cache load path is exercised
    repo_map_cached = RepoMap(root_path=tmp_path, cache_enabled=True, use_tree_sitter=False)
    assert "pkg/app.py" in repo_map_cached.context.files
    assert repo_map_cached.context.last_updated > 0


def test_repo_map_pagerank_and_dependency_edges(tmp_path):
    _create_sample_repo(tmp_path)
    repo_map = RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)
    repo_map.scan_repository(force=True)

    graph = repo_map.build_dependency_graph()
    assert graph.has_edge("pkg/app.py", "pkg/util.py")

    scores = repo_map.compute_pagerank()
    assert scores["pkg/util.py"] > 0

    ranked = repo_map.get_ranked_files({"pkg/app.py"}, {"Helper"})
    top_files = [path for path, _ in ranked[:2]]
    assert "pkg/util.py" in top_files

    summary = repo_map.get_file_summary("pkg/util.py")
    assert summary and "Helper" in summary

    assert repo_map.find_symbol("Helper") == ["pkg/util.py"]


def test_repo_map_helpers_and_fallback_ranking(tmp_path):
    repo_map = RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)

    generated = tmp_path / "build" / "generated_file.py"
    generated.parent.mkdir()
    generated.write_text("print('hi')")
    should_skip, reason = repo_map._should_skip_file(generated)
    assert should_skip and "generated" in reason

    # _is_well_named_symbol caching behaviour
    assert repo_map._is_well_named_symbol("CamelCase")
    assert repo_map._is_well_named_symbol("CamelCase")  # cached path
    assert not repo_map._is_well_named_symbol("xx")

    _create_sample_repo(tmp_path)
    repo_map.scan_repository(force=True)

    # Force pagerank fallback by using no explicit mentions
    repo_map.compute_pagerank()
    results = repo_map.get_ranked_files(set(), set(), max_files=3)
    assert results, "Expected pagerank-based results when no mentions provided"

    repo_map.invalidate_file("pkg/util.py")
    assert "pkg/util.py" not in repo_map.context.files


def test_repo_map_dependency_weighting(tmp_path):
    repo_map = RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)

    file_a = FileInfo(
        path="src/a.py",
        size=10,
        modified_time=1.0,
        language="python",
        symbols=[],
        imports=[],
        dependencies=set(),
        references={},
        symbols_used=["ImportantSymbol"] * 4,
        file_name="a.py",
        file_stem="a",
        path_parts=("src", "a.py"),
    )

    file_b = FileInfo(
        path="src/b.py",
        size=10,
        modified_time=1.0,
        language="python",
        symbols=["ImportantSymbol"],
        imports=[],
        dependencies=set(),
        references={},
        symbols_used=[],
        file_name="b.py",
        file_stem="b",
        path_parts=("src", "b.py"),
    )

    repo_map.context.files = {
        "src/a.py": file_a,
        "src/b.py": file_b,
    }

    repo_map.context.symbol_index["ImportantSymbol"].add("src/b.py")

    graph = repo_map.build_dependency_graph()
    assert graph.has_edge("src/a.py", "src/b.py")
    weight = graph["src/a.py"]["src/b.py"]["weight"]
    assert weight >= 40.0

    ranks = repo_map.compute_pagerank()
    assert pytest.approx(sum(ranks.values()), rel=1e-6) == 1.0
    assert ranks["src/b.py"] > ranks["src/a.py"]
