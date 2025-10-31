import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
