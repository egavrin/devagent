import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from unittest.mock import patch

import networkx as nx
import pytest

from ai_dev_agent.core.repo_map import FileInfo, RepoMap


def _make_file_info(
    path: str,
    *,
    size: int = 200,
    modified_time: float = 1.0,
    language: str = "python",
    symbols: Optional[List[str]] = None,
    imports: Optional[List[str]] = None,
    dependencies: Optional[Set[str]] = None,
    references: Optional[Dict[str, List[Tuple[str, int]]]] = None,
    symbols_used: Optional[List[str]] = None,
) -> FileInfo:
    path_obj = Path(path)
    return FileInfo(
        path=path,
        size=size,
        modified_time=modified_time,
        language=language,
        symbols=symbols or [],
        imports=imports or [],
        exports=[],
        dependencies=dependencies or set(),
        references=references or {},
        symbols_used=symbols_used or [],
        file_name=path_obj.name,
        file_stem=path_obj.stem,
        path_parts=tuple(path_obj.parts),
    )


def test_repo_map_load_cache_restores_indices(tmp_path):
    cache_dir = tmp_path / ".devagent_cache"
    cache_dir.mkdir()
    cache_payload = {
        "version": RepoMap.CACHE_VERSION,
        "last_updated": 123.0,
        "last_pagerank_update": 456.0,
        "pagerank_scores": {"src/app.py": 0.42},
        "files": [
            {
                "path": "src/app.py",
                "size": 128,
                "modified_time": 100.0,
                "language": "python",
                "symbols": ["AppClass"],
                "imports": ["src/util.py"],
                "exports": [],
                "dependencies": ["src/util.py"],
                "references": {"AppClass": [("src/util.py", 15)]},
                "symbols_used": ["UtilFunc"],
                "file_name": "app.py",
                "file_stem": "app",
                "path_parts": ["src", "app.py"],
            }
        ],
    }
    (cache_dir / "repo_map.json").write_text(json.dumps(cache_payload))

    repo_map = RepoMap(root_path=tmp_path, cache_enabled=True, use_tree_sitter=False)

    assert "src/app.py" in repo_map.context.files
    cached = repo_map.context.files["src/app.py"]
    assert cached.symbols == ["AppClass"]
    assert repo_map.context.symbol_index["AppClass"] == {"src/app.py"}
    assert repo_map.context.import_graph["src/app.py"] == {"src/util.py"}
    assert repo_map.context.pagerank_scores["src/app.py"] == pytest.approx(0.42)


def test_repo_map_normalize_mentions_trims(monkeypatch, tmp_path):
    repo_map = RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)
    monkeypatch.setattr(repo_map, "MAX_FASTPATH_MENTIONS", 3)
    monkeypatch.setattr(repo_map, "MAX_DIRECTORY_MATCHES", 2)

    mentioned = {
        " src/app.py ",
        "src/feature.py",
        "docs/guide.md",
        "pkg/module.py",
        "pkg/subdir/",
        "pkg/another/subdir/file.py",
    }

    trimmed, names, stems, directories = repo_map._normalize_mentions(mentioned)

    assert len(trimmed) == 3
    # Names should only contain file names extracted from trimmed mentions
    assert names <= {"app.py", "feature.py", "module.py", "file.py", "guide.md"}
    # Stems should be at least four characters long
    assert all(len(stem) > 3 for stem in stems)
    # Directory mentions capped at MAX_DIRECTORY_MATCHES and always include a path separator
    assert len(directories) == 2
    assert all(("/" in entry or "\\" in entry) for entry in directories)


def test_repo_map_symbol_match_score_counts_variants(tmp_path):
    repo_map = RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)
    info = _make_file_info(
        "src/core/feature.py",
        symbols=["ExactMatch", "ExactMatchHelper"],
        symbols_used=["ExactMatch", "OtherSymbol"],
    )

    score = repo_map._symbol_match_score(
        info,
        mentioned_symbols={"ExactMatch"},
        long_symbol_prefixes=("ExactMat",),
    )

    # Direct match (100) + symbols_used hit (3)
    assert score == pytest.approx(103.0)

    prefix_only = _make_file_info(
        "src/core/helper.py",
        symbols=["ExactMatchExtended"],
    )
    prefix_score = repo_map._symbol_match_score(
        prefix_only,
        mentioned_symbols={"ExactMatch"},
        long_symbol_prefixes=("ExactMatch",),
    )
    assert prefix_score == pytest.approx(50.0)


def test_repo_map_quick_rank_applies_priority(tmp_path):
    repo_root = tmp_path
    (repo_root / "src/core").mkdir(parents=True)
    (repo_root / "tests").mkdir(parents=True)
    (repo_root / "src/helpers").mkdir(parents=True)
    (repo_root / "src/core/feature.py").write_text("print('feature')")
    (repo_root / "tests/test_feature.py").write_text("print('test')")
    (repo_root / "src/helpers/util.py").write_text("print('util')")

    repo_map = RepoMap(root_path=repo_root, cache_enabled=False, use_tree_sitter=False)
    repo_map.context.files = {
        "src/core/feature.py": _make_file_info(
            "src/core/feature.py",
            symbols=["PrimarySymbol"],
            symbols_used=["PrimarySymbol"],
        ),
        "tests/test_feature.py": _make_file_info(
            "tests/test_feature.py",
            symbols=["PrimarySymbol"],
        ),
        "src/helpers/util.py": _make_file_info(
            "src/helpers/util.py",
            symbols=["HelperSymbol"],
        ),
    }

    results = repo_map._quick_rank_by_symbols(
        mentioned_files=set(),
        mentioned_symbols={"PrimarySymbol"},
        max_files=5,
        mentioned_names={"feature.py"},
        mentioned_stems={"feature"},
        directory_mentions=("src", "tests"),
        long_symbol_prefixes=("PrimarySymbol",),
    )

    scores = {path: score for path, score in results}
    assert scores["src/core/feature.py"] > scores["tests/test_feature.py"]
    # Directory mention keeps helper file visible but still lower than primary target
    assert "src/helpers/util.py" in scores
    assert scores["src/helpers/util.py"] < scores["src/core/feature.py"]
    assert scores["src/helpers/util.py"] > scores["tests/test_feature.py"]


def test_repo_map_get_ranked_files_fast_path_symbol_match(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    feature_path = repo_root / "src/app.py"
    feature_path.write_text("print('hello')")

    repo_map = RepoMap(root_path=repo_root, cache_enabled=False, use_tree_sitter=False)
    repo_map.context.files = {
        "src/app.py": _make_file_info(
            "src/app.py",
            symbols=["TargetSymbol"],
            symbols_used=["TargetSymbol"],
        )
    }
    repo_map.context.pagerank_scores = {"src/app.py": 0.5}
    repo_map.context.dependency_graph = nx.DiGraph()

    results = repo_map.get_ranked_files(mentioned_files=set(), mentioned_symbols={"TargetSymbol"})
    assert results and results[0][0] == "src/app.py"


def test_repo_map_get_ranked_files_falls_back_to_pagerank(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "src/core").mkdir(parents=True)
    (repo_root / "src/shared").mkdir(parents=True)
    (repo_root / "docs").mkdir(parents=True)

    (repo_root / "src/core/feature.py").write_text("# core feature")
    (repo_root / "src/shared/helper.py").write_text("# helper")
    (repo_root / "docs/guide.md").write_text("Guide")

    repo_map = RepoMap(root_path=repo_root, cache_enabled=False, use_tree_sitter=False)
    repo_map.context.files = {
        "src/core/feature.py": _make_file_info(
            "src/core/feature.py",
            symbols=["FeatureSymbol"],
            dependencies={"src/shared/helper.py"},
            size=150,
        ),
        "src/shared/helper.py": _make_file_info(
            "src/shared/helper.py",
            dependencies={"src/core/feature.py"},
            size=90,
        ),
        "docs/guide.md": _make_file_info(
            "docs/guide.md",
            language="markdown",
            size=500,
        ),
    }
    repo_map.context.pagerank_scores = {
        "src/core/feature.py": 0.2,
        "src/shared/helper.py": 0.1,
        "docs/guide.md": 0.05,
    }
    repo_map.context.dependency_graph = nx.DiGraph()

    with patch.object(
        repo_map, "_quick_rank_by_symbols", return_value=[("src/core/feature.py", 80.0)]
    ):
        results = repo_map.get_ranked_files(
            mentioned_files={"src/core/feature.py", "src/shared/helper.py", "docs/"},
            mentioned_symbols=set(),
            max_files=3,
        )

    scores = {path: score for path, score in results}
    assert "src/core/feature.py" in scores
    assert scores["src/core/feature.py"] > scores["src/shared/helper.py"]
    assert scores["docs/guide.md"] > 0  # directory mention keeps docs file in results


@pytest.mark.parametrize(
    ("method_name", "relative_path"),
    [
        ("_extract_typescript_info", "src/sample.ts"),
        ("_extract_cpp_info", "src/sample.cpp"),
        ("_extract_java_info", "src/Sample.java"),
        ("_extract_go_info", "src/sample.go"),
        ("_extract_rust_info", "src/sample.rs"),
        ("_extract_ruby_info", "src/sample.rb"),
        ("_extract_kotlin_info", "src/sample.kt"),
        ("_extract_swift_info", "src/sample.swift"),
        ("_extract_dart_info", "src/sample.dart"),
        ("_extract_lua_info", "src/sample.lua"),
        ("_extract_php_info", "src/sample.php"),
    ],
)
def test_repo_map_extractors_ignore_errors(tmp_path, monkeypatch, method_name, relative_path):
    repo_root = tmp_path / "repo"
    target_path = repo_root / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("content")

    repo_map = RepoMap(root_path=repo_root, cache_enabled=False, use_tree_sitter=False)
    info = _make_file_info(relative_path)

    original_open = Path.open

    def raising_open(self, *args, **kwargs):
        if self == target_path:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "boom")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raising_open)

    extractor = getattr(repo_map, method_name)
    extractor(target_path, info)
    assert info.symbols == []
    assert info.imports == []


def test_repo_map_extract_with_regex_dispatch(monkeypatch, tmp_path):
    repo_map = RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)
    path = tmp_path / "file.ext"
    path.write_text("content")
    info = _make_file_info("file.ext")

    called = []

    def mark(name):
        def _rec(*_):
            called.append(name)

        return _rec

    dispatch_map = {
        "python": "_extract_python_info",
        "typescript": "_extract_typescript_info",
        "c": "_extract_cpp_info",
        "java": "_extract_java_info",
        "kotlin": "_extract_kotlin_info",
        "scala": "_extract_scala_info",
        "go": "_extract_go_info",
        "rust": "_extract_rust_info",
        "ruby": "_extract_ruby_info",
        "swift": "_extract_swift_info",
        "dart": "_extract_dart_info",
        "lua": "_extract_lua_info",
        "php": "_extract_php_info",
        "generic": "_extract_generic_info",
    }

    for lang, attr in dispatch_map.items():
        monkeypatch.setattr(repo_map, attr, mark(lang))

    languages = [
        "python",
        "typescript",
        "c",
        "java",
        "kotlin",
        "scala",
        "go",
        "rust",
        "ruby",
        "swift",
        "dart",
        "lua",
        "php",
        None,
    ]
    for lang in languages:
        repo_map._extract_with_regex(path, info, lang)

    assert set(called) >= {
        "python",
        "typescript",
        "c",
        "java",
        "kotlin",
        "scala",
        "go",
        "rust",
        "ruby",
        "swift",
        "dart",
        "lua",
        "php",
        "generic",
    }


def test_repo_map_build_dependency_graph_skips_noisy_symbols(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "src").mkdir()
    (repo_root / "src" / "a.py").write_text("print('a')")
    (repo_root / "src" / "b.py").write_text("print('b')")

    repo_map = RepoMap(root_path=repo_root, cache_enabled=False, use_tree_sitter=False)
    repo_map.context.files = {
        "src/a.py": _make_file_info(
            "src/a.py",
            symbols_used=["a", "MeaningfulSymbol", "VeryLongSymbolName", "_privateSymbol"],
        ),
        "src/b.py": _make_file_info(
            "src/b.py",
            symbols=["MeaningfulSymbol", "VeryLongSymbolName", "_privateSymbol"],
        ),
    }
    repo_map.context.symbol_index = {
        "MeaningfulSymbol": {"src/b.py"},
        "VeryLongSymbolName": {"src/b.py"},
        "_privateSymbol": {"src/b.py"},
    }

    graph = repo_map.build_dependency_graph()
    assert ("src/a.py", "src/b.py") in graph.edges
    # Noisy symbol "a" should not create dependencies
    assert all("a" not in data["symbols"] for _, _, data in graph.edges(data=True))
    assert repo_map.context.files["src/a.py"].dependencies == {"src/b.py"}
    edge_data = graph.get_edge_data("src/a.py", "src/b.py")
    assert any(symbol == "_privateSymbol" for symbol in edge_data["symbols"])


def test_repo_map_get_ranked_files_pagerank_adjustments(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src/a.py").write_text("print('a')")
    (repo_root / "src/b.py").write_text("print('b')")

    repo_map = RepoMap(root_path=repo_root, cache_enabled=False, use_tree_sitter=False)
    repo_map.context.files = {
        "src/a.py": _make_file_info(
            "src/a.py",
            size=60000,
            symbols=["LongSymbolName", "SymbolMatch"],
            symbols_used=["SymbolMatch"],
        ),
        "src/b.py": _make_file_info(
            "src/b.py",
            size=5000,
            symbols=["OtherSymbol"],
        ),
    }

    base_scores = {"src/a.py": 0.1, "src/b.py": 0.2}
    personalized_scores = {"src/a.py": 0.5, "src/b.py": 0.1}
    calls = []

    def fake_pagerank(personalization=None, cache_results=True, use_edge_distribution=True):
        calls.append((personalization, cache_results))
        return personalized_scores if personalization else base_scores

    monkeypatch.setattr(repo_map, "compute_pagerank", fake_pagerank)

    results = repo_map.get_ranked_files_pagerank(
        mentioned_files={"src/a.py"},
        mentioned_symbols={"SymbolMatch"},
        conversation_files={"src/a.py"},
        max_files=2,
    )

    assert results[0][0] == "src/a.py"
    # ensure both baseline and personalized computations executed
    assert any(personalization is None for personalization, _ in calls)
    assert any(personalization for personalization, _ in calls)

    repo_map.context.pagerank_scores = {
        "src/a.py": 0.2,
        "src/b.py": 0.1,
        "src/missing.py": 0.05,
    }
    monkeypatch.setattr(
        repo_map, "compute_pagerank", lambda *args, **kwargs: repo_map.context.pagerank_scores
    )
    fallback_results = repo_map.get_ranked_files_pagerank(
        mentioned_files=set(), mentioned_symbols=set(), conversation_files=None, max_files=2
    )
    assert all(entry[0] in {"src/a.py", "src/b.py"} for entry in fallback_results)


def test_repo_map_quick_rank_directory_boost(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "docs/guide.md").write_text("# guide")

    repo_map = RepoMap(root_path=repo_root, cache_enabled=False, use_tree_sitter=False)
    repo_map.context.files = {
        "docs/guide.md": _make_file_info(
            "docs/guide.md",
            language="markdown",
            size=100,
        )
    }

    results = repo_map._quick_rank_by_symbols(
        mentioned_files=set(),
        mentioned_symbols=set(),
        max_files=5,
        mentioned_names=set(),
        mentioned_stems=set(),
        directory_mentions=("docs",),
        long_symbol_prefixes=(),
    )

    assert results and results[0][0] == "docs/guide.md"
    assert results[0][1] >= 50.0


def test_repo_map_get_ranked_files_full_pagerank_scoring(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "src/core").mkdir(parents=True)
    (repo_root / "src/neighbor").mkdir(parents=True)
    (repo_root / "src/lib").mkdir(parents=True)
    (repo_root / "src/core/feature.py").write_text("feature")
    (repo_root / "src/neighbor/util.py").write_text("util")
    (repo_root / "src/lib/helper.py").write_text("helper")

    repo_map = RepoMap(root_path=repo_root, cache_enabled=False, use_tree_sitter=False)
    repo_map.context.files = {
        "src/core/feature.py": _make_file_info(
            "src/core/feature.py",
            size=52000,
            symbols=["SymbolHit", "VeryLongIdentifier"],
            symbols_used=["SymbolHit"],
            dependencies={"src/lib/helper.py"},
        ),
        "src/neighbor/util.py": _make_file_info(
            "src/neighbor/util.py",
            size=8000,
            symbols=[],
        ),
        "src/lib/helper.py": _make_file_info(
            "src/lib/helper.py",
            symbols=[],
            dependencies=set(),
        ),
    }
    repo_map.context.files["src/lib/helper.py"].dependencies = {"src/core/feature.py"}
    repo_map.context.pagerank_scores = {
        "src/core/feature.py": 0.3,
        "src/neighbor/util.py": 0.05,
        "src/lib/helper.py": 0.1,
    }
    repo_map.context.dependency_graph = nx.DiGraph()

    with patch.object(repo_map, "_quick_rank_by_symbols", return_value=[]):
        results = repo_map.get_ranked_files(
            mentioned_files={"src/core/feature.py", "src/neighbor/", "src/lib/helper.py"},
            mentioned_symbols={"SymbolHit"},
            max_files=3,
        )

    ranking = {path: score for path, score in results}
    assert "src/core/feature.py" in ranking
    assert "src/neighbor/util.py" in ranking  # directory boost keeps neighbor entry
    assert ranking["src/core/feature.py"] > 1000  # filename/stem boost applied
    assert ranking["src/lib/helper.py"] > 0  # dependency mention contributes
