"""Additional CLI-focused tests for ContextEnhancer behaviour."""

from __future__ import annotations

import itertools
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.cli.context_enhancer import ContextEnhancer
from ai_dev_agent.core.utils.config import Settings


@pytest.fixture
def temp_workspace() -> Path:
    """Create a temporary workspace for context enhancer tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("def main():\n    return 42\n", encoding="utf-8")
        (root / "README.md").write_text("# Sample project\n", encoding="utf-8")
        yield root


@pytest.fixture
def mock_settings() -> Settings:
    """Return a minimal settings object required by ContextEnhancer."""
    settings = MagicMock(spec=Settings)
    settings.enable_memory_bank = False
    settings.enable_repo_map = True
    settings.repo_map_style = "aider"
    settings.repomap_debug_stdout = False
    return settings


class StubRepoMap:
    """Minimal RepoMap stub for tests."""

    def __init__(self, files: dict[str, SimpleNamespace], ranked=None, last_updated: float = 1.0):
        self.context = SimpleNamespace(
            files=files,
            symbol_index={},
            import_graph={},
            last_updated=last_updated,
        )
        self._ranked = ranked or []
        self.calls = []

    def get_ranked_files(self, *, mentioned_files, mentioned_symbols, max_files):
        self.calls.append(
            {
                "files": set(mentioned_files),
                "symbols": set(mentioned_symbols),
                "max_files": max_files,
            }
        )
        return list(self._ranked)

    def scan_repository(self):
        return None


def test_context_enhancer_file_discovery(monkeypatch, temp_workspace, mock_settings):
    """Context enhancer should enumerate files via git, respect ignores, and trim large repos."""
    tracked = [
        "src/app.py",
        "src/__pycache__/ignored.pyc",
        "node_modules/ignore.js",
        "dist/bundle.js",
        "tests/test_app.py",
        "README.md",
    ]

    # Simulate a large repository by appending many tracked files
    large_tail = [f"pkg/module_{idx}.py" for idx in range(6000)]
    git_output = "\n".join(itertools.chain(tracked, large_tail))

    class DummyCompletedProcess:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    dummy_process = DummyCompletedProcess(git_output)
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.subprocess.run",
        lambda *args, **kwargs: dummy_process,
    )

    repo_map = StubRepoMap(files={}, last_updated=2.0)
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    files = enhancer.list_repository_files()

    assert "src/app.py" in files
    assert "README.md" in files
    assert "node_modules/ignore.js" not in files
    assert "dist/bundle.js" not in files
    assert "src/__pycache__/ignored.pyc" not in files
    assert len(files) <= ContextEnhancer.LARGE_REPO_FILE_LIMIT


def test_context_enhancer_content_extraction(monkeypatch, temp_workspace, mock_settings):
    """Content extraction should prioritize symbols, directories, and produce structured context."""
    files = {
        "src/auth/service.py": SimpleNamespace(
            file_name="service.py",
            file_stem="service",
            language="python",
            size=2000,
            path_parts=("src", "auth", "service.py"),
        ),
        "src/auth/utils.py": SimpleNamespace(
            file_name="utils.py",
            file_stem="utils",
            language="python",
            size=1500,
            path_parts=("src", "auth", "utils.py"),
        ),
        "runtime_engine/loop.py": SimpleNamespace(
            file_name="loop.py",
            file_stem="loop",
            language="python",
            size=1800,
            path_parts=("runtime_engine", "loop.py"),
        ),
    }

    ranked = [
        ("src/auth/service.py", 12.5),
        ("src/auth/utils.py", 6.4),
        ("runtime_engine/loop.py", 2.1),
    ]
    repo_map = StubRepoMap(files=files, ranked=ranked, last_updated=3.0)
    repo_map.context.symbol_index = {"AuthService": {"src/auth/service.py"}}

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    mock_settings.repomap_debug_stdout = True
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

    query = (
        "Investigate AuthService logic in src/auth/service.py, compare against runtime_engine "
        "modules and MissingSymbol. Avoid http://example.com/resource.js."
    )

    symbols, files_found = enhancer.extract_symbols_and_files(query)

    assert "AuthService" in symbols
    assert "src/auth/service.py" in files_found
    assert "runtime_engine" in files_found  # directory mention
    assert all(not entry.startswith("http") for entry in files_found)

    enhanced = enhancer.enhance_query_with_context(
        query + " Also review nonexistent.py.", max_files=5
    )

    assert "[Automatic Context from RepoMap]" in enhanced
    assert "High relevance:" in enhanced
    assert "Medium relevance:" in enhanced
    assert "Other relevant files:" in enhanced
    assert "No files matched:" in enhanced and "nonexistent.py" in enhanced
    assert "Unknown symbols:" in enhanced and "MissingSymbol" in enhanced


def test_context_enhancer_caching(monkeypatch, temp_workspace, mock_settings):
    """Repository listing uses cache and invalidates when RepoMap metadata changes."""
    files = {
        "src/core.py": SimpleNamespace(
            file_name="core.py",
            file_stem="core",
            language="python",
            size=1024,
            path_parts=("src", "core.py"),
        )
    }
    repo_map = StubRepoMap(files=files, last_updated=10.0)

    run_calls = []

    def fake_run(*args, **kwargs):
        run_calls.append(args)

        class DummyProcess:
            returncode = 0
            stdout = "src/core.py\n"
            stderr = ""

        return DummyProcess()

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    git_dir = temp_workspace / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (git_dir / "index").write_bytes(b"")
    (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text("commit-a", encoding="utf-8")
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

    first = enhancer.list_repository_files()
    second = enhancer.list_repository_files()

    assert first == second
    assert len(run_calls) == 1  # cache hit on second call

    repo_map.context.last_updated = 99.0
    third = enhancer.list_repository_files()

    assert third == first
    assert len(run_calls) == 2  # cache refreshed after timestamp change


def test_context_enhancer_git_signature_invalidation(monkeypatch, temp_workspace, mock_settings):
    """Cache should refresh when git index/head state changes."""
    repo_map = StubRepoMap(files={}, last_updated=5.0)
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    git_dir = temp_workspace / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    index_path = git_dir / "index"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(b"")
    (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text("commit-a", encoding="utf-8")

    outputs = {"text": "src/app.py\n"}

    class DummyProcess:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(*args, **kwargs):
        return DummyProcess(outputs["text"])

    monkeypatch.setattr("ai_dev_agent.cli.context_enhancer.subprocess.run", fake_run)

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    first = enhancer.list_repository_files()
    assert first == ["src/app.py"]

    outputs["text"] = "src/changed.py\n"
    stat_result = index_path.stat()
    os.utime(index_path, (stat_result.st_atime, stat_result.st_mtime + 1))
    second = enhancer.list_repository_files()

    assert second == ["src/changed.py"]
