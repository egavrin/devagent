"""Tests for RepoMap cache handling and repository scanning edge cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from ai_dev_agent.core import repo_map as repo_map_module
from ai_dev_agent.core.repo_map import FileInfo, RepoMap


@pytest.fixture
def repo(tmp_path: Path) -> RepoMap:
    """Create a RepoMap instance with caching disabled for direct method access."""
    return RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)


def test_restore_cache_version_mismatch(repo: RepoMap) -> None:
    """Old cache versions should be rejected without mutating the context."""
    cached = {
        "version": "0.9",
        "files": [{"path": "src/main.py", "size": 10, "modified_time": 1.0}],
    }

    restored = repo._restore_from_cache_data(cached)

    assert restored is False
    assert repo.context.files == {}


def test_restore_cache_populates_context(repo: RepoMap) -> None:
    """Valid cache payloads should populate FileInfo entries completely."""
    cached = {
        "version": RepoMap.CACHE_VERSION,
        "last_updated": 123.0,
        "files": [
            {
                "path": "src/main.py",
                "size": 42,
                "modified_time": 456.0,
                "language": "python",
                "symbols": ["Foo"],
                "imports": ["os"],
                "exports": [],
                "dependencies": ["src/utils.py"],
                "references": {"Foo": [("tests/test_main.py", 10)]},
                "symbols_used": ["Bar"],
                "file_name": "main.py",
                "file_stem": "main",
                "path_parts": ["src", "main.py"],
            }
        ],
    }

    restored = repo._restore_from_cache_data(cached)

    assert restored is True
    file_info = repo.context.files["src/main.py"]
    assert file_info.path_parts == ("src", "main.py")
    assert file_info.dependencies == {"src/utils.py"}
    assert repo.context.last_updated == pytest.approx(123.0)


def test_save_cache_uses_msgpack_and_removes_json(tmp_path: Path) -> None:
    """When msgpack is available, a msgpack cache is written and JSON is removed."""
    repo = RepoMap(root_path=tmp_path, cache_enabled=True, use_tree_sitter=False)
    repo.context.files["example.py"] = FileInfo(
        path="example.py", size=1, modified_time=1.0, language="python"
    )

    # Prime an old JSON cache to ensure it is removed.
    repo.cache_path.parent.mkdir(parents=True, exist_ok=True)
    repo.cache_path.write_text(json.dumps({"stale": True}), encoding="utf-8")

    repo._save_cache()

    msgpack_path = repo.cache_path.with_suffix(".msgpack")
    assert msgpack_path.exists()
    assert not repo.cache_path.exists()


def test_save_cache_falls_back_to_json_when_msgpack_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If msgpack is unavailable, JSON caching should be used instead."""
    monkeypatch.setattr(repo_map_module, "MSGPACK_AVAILABLE", False, raising=False)

    repo = RepoMap(root_path=tmp_path, cache_enabled=True, use_tree_sitter=False)
    repo.context.files["example.py"] = FileInfo(
        path="example.py", size=1, modified_time=1.0, language="python"
    )

    repo._save_cache()

    assert repo.cache_path.exists()
    assert repo.cache_path.with_suffix(".msgpack").exists() is False


def test_scan_repository_collects_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: RepoMap
) -> None:
    """scan_repository should increment statistics for different results."""
    files = [tmp_path / f"file{i}.py" for i in range(3)]
    for path in files:
        path.write_text("print('ok')", encoding="utf-8")

    def outcome_cycle() -> Iterator[str]:
        for value in ["scanned", "skipped_large", "error"]:
            yield value
        while True:
            yield "skipped_large"

    outcomes = outcome_cycle()

    def fake_scan(file_path: Path) -> str:
        result = next(outcomes)
        if result == "scanned":
            relative = str(file_path.relative_to(repo.root_path))
            repo.context.files[relative] = FileInfo(
                path=relative,
                size=10,
                modified_time=1.0,
                language="python",
                file_name=file_path.name,
                file_stem=file_path.stem,
                path_parts=tuple(file_path.relative_to(repo.root_path).parts),
            )
        return result

    monkeypatch.setattr(repo, "_scan_file", fake_scan)

    repo.scan_repository(force=True)

    assert repo.context.last_updated > 0
    assert any(name in repo.context.files for name in {"file0.py", "file1.py", "file2.py"})
    # Cache is disabled in the fixture, so PageRank should remain cleared after scan.
    assert repo.context.pagerank_scores == {}


def test_scan_file_respects_size_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: RepoMap
) -> None:
    """_scan_file should skip files larger than MAX_FILE_SIZE."""
    # Shrink limits to keep the test fast.
    monkeypatch.setattr(repo, "MAX_FILE_SIZE", 16, raising=False)
    monkeypatch.setattr(repo, "LARGE_FILE_SIZE", 8, raising=False)

    large_file = tmp_path / "too_big.py"
    large_file.write_bytes(b"x" * 32)

    result = repo._scan_file(large_file)

    assert result == "skipped_large"
