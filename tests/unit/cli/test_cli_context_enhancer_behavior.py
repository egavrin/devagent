"""Additional CLI-focused tests for ContextEnhancer behaviour."""

from __future__ import annotations

import itertools
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.cli.context_enhancer import ContextEnhancer, enhance_query, get_context_enhancer
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


def test_context_enhancer_important_file_prioritization(temp_workspace, mock_settings):
    """Important files and path heuristics should be prioritised over generic entries."""
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    candidates = [
        "pkg/module.py",
        "README.md",
        "lib/main.py",
        "src/index.js",
        "scripts/deploy.sh",
        "tests/test_sample.py",
        "docs/guide.txt",
        "build/output.bundle.js",
    ]

    important = enhancer._get_important_files(candidates, max_files=4)
    returned = [path for path, _ in important]

    assert returned[0] == "README.md"
    assert "src/index.js" in returned
    assert "lib/main.py" in returned
    assert "scripts/deploy.sh" in returned
    assert "tests/test_sample.py" not in returned
    assert "build/output.bundle.js" not in returned


def test_get_important_files_breaks_on_first_pass(temp_workspace, mock_settings):
    """First-pass important patterns should short-circuit once the cap is reached."""
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    result = enhancer._get_important_files(["README.md", "setup.py"], max_files=1)
    assert result == [("README.md", 20.0)]


def test_get_important_files_hits_second_pass(temp_workspace, mock_settings):
    """Path-based heuristics should populate the list when exact matches are absent."""
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    result = enhancer._get_important_files(["src/main/server.py", "lib/app.py"], max_files=1)
    assert result[0][0] == "src/main/server.py"


def test_get_important_files_third_pass_includes_generic(temp_workspace, mock_settings):
    """Third pass should add implementation files after filtering test artefacts."""
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    result = enhancer._get_important_files(
        ["pkg/component.py", "tests/test_component.py"], max_files=1
    )
    assert result == [("pkg/component.py", 3.0)]


def test_context_enhancer_filtering_respects_custom_ignores(mock_settings, temp_workspace):
    """Custom ignore fragments and suffixes should be honoured."""
    mock_settings.context_ignore_patterns = ["docs", "experimental"]
    mock_settings.context_ignore_suffixes = [".lock", ".tmp"]
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

    raw_entries = [
        "",
        "   ",
        "src/app.py",
        "docs/readme.md",
        "experimental/mod.py",
        "package.lock",
        "notes.TMP",
    ]

    filtered = enhancer._filter_discoverable_files(raw_entries)
    assert filtered == ["src/app.py"]


def test_context_enhancer_load_tracked_files_falls_back_to_repomap(
    monkeypatch, temp_workspace, mock_settings
):
    """When git is unavailable, RepoMap inventory should be used."""

    class RepoMapStub:
        def __init__(self):
            self.context = SimpleNamespace(
                files={
                    "src/core.py": SimpleNamespace(),
                    "README.md": SimpleNamespace(),
                },
                symbol_index={},
                import_graph={},
                last_updated=1.0,
            )

        def get_ranked_files(self, *args, **kwargs):
            return []

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: RepoMapStub(),
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    tracked = enhancer._load_tracked_files()
    assert set(tracked) == {"src/core.py", "README.md"}


def test_context_enhancer_memory_provider_delegation(temp_workspace, mock_settings):
    """All memory helper methods should proxy to the provider."""

    class MemoryStub:
        has_store = True

        def __init__(self):
            self.calls = []

        def retrieve_relevant_memories(self, query, task_type, limit):
            self.calls.append(("retrieve", query, task_type, limit))
            return [{"id": "mem-1", "content": "finding"}]

        def format_memories_for_context(self, memories):
            self.calls.append(("format", tuple(memories)))
            return "memory block"

        def store_memory(self, query, response, task_type, success, metadata):
            self.calls.append(("store", query, response, task_type, success, metadata))
            return "stored-id"

        def distill_and_store_memory(self, session_id, messages, metadata):
            self.calls.append(("distill", session_id, tuple(messages), metadata))
            return "distilled-id"

        def track_memory_effectiveness(self, memory_ids, success, feedback):
            self.calls.append(("track", tuple(memory_ids), success, feedback))

        def record_query_outcome(
            self, *, session_id, success, tools_used, task_type, error_type, duration_seconds
        ):
            self.calls.append(
                (
                    "record",
                    session_id,
                    success,
                    tuple(tools_used),
                    task_type,
                    error_type,
                    duration_seconds,
                )
            )

        def collect_statistics(self):
            self.calls.append(("collect",))
            return {"memories": 1}

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    memory = MemoryStub()
    enhancer._memory_provider = memory

    messages, ids = enhancer.get_memory_context("optimize query", task_type="analysis", limit=3)
    assert ids == ["mem-1"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "assistant"

    stored_id = enhancer.store_memory("q", "r", task_type="analysis")
    assert stored_id == "stored-id"
    distilled_id = enhancer.distill_and_store_memory("session", ["msg"], {"k": "v"})
    assert distilled_id == "distilled-id"
    enhancer.track_memory_effectiveness(["mem-1"], True, feedback="good")
    enhancer.record_query_outcome(
        session_id="session",
        success=False,
        tools_used=["read"],
        task_type="investigation",
        error_type="timeout",
        duration_seconds=1.5,
    )
    stats = enhancer.collect_memory_statistics()
    assert stats == {"memories": 1}

    call_names = [entry[0] for entry in memory.calls]
    assert call_names == [
        "retrieve",
        "format",
        "store",
        "distill",
        "track",
        "record",
        "collect",
    ]


def test_context_enhancer_third_pass_includes_generic_files(temp_workspace, mock_settings):
    """Third pass should include non-test implementation files."""
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    candidates = ["pkg/module.py", "tests/test_module.py", "build/generated.py"]

    important = enhancer._get_important_files(candidates, max_files=2)
    assert ("pkg/module.py", 3.0) in important
    assert all("test" not in path for path, _ in important)


def test_context_enhancer_git_signature_detached_head(temp_workspace, mock_settings):
    """Detached HEAD state should still surface a signature."""
    git_dir = temp_workspace / ".git"
    git_dir.mkdir()
    (git_dir / "index").write_bytes(b"")
    commit_hash = "deadbeefcafebabe"
    (git_dir / "HEAD").write_text(commit_hash, encoding="utf-8")

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    _, head_sig = enhancer._get_git_signature()

    assert head_sig == commit_hash


def test_context_enhancer_git_signature_missing_files(monkeypatch, temp_workspace, mock_settings):
    """Stat and HEAD read failures should yield a neutral git signature."""
    git_dir = temp_workspace / ".git"
    git_dir.mkdir()
    index_path = git_dir / "index"
    index_path.write_bytes(b"")
    head_path = git_dir / "HEAD"
    head_path.write_text("ref: refs/heads/main", encoding="utf-8")

    original_stat = Path.stat
    original_read = Path.read_text

    def stub_stat(self):
        if self == index_path:
            raise OSError("stat failed")
        return original_stat(self)

    def stub_read(self, *args, **kwargs):
        if self == head_path:
            raise OSError("read failed")
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stub_stat)
    monkeypatch.setattr(Path, "read_text", stub_read)

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    signature = enhancer._get_git_signature()

    assert signature == (None, None)


def test_extract_symbols_handles_bare_filenames_and_directories(
    monkeypatch, temp_workspace, mock_settings
):
    """Bare filenames and directory keywords should map to repo files."""
    files = {
        "src/helpers.py": SimpleNamespace(
            file_name="helpers.py",
            file_stem="helpers",
            language="python",
            size=512,
            path_parts=("src", "helpers.py"),
        ),
        "services/api/client.py": SimpleNamespace(
            file_name="client.py",
            file_stem="client",
            language="python",
            size=256,
            path_parts=("services", "api", "client.py"),
        ),
    }
    repo_map = StubRepoMap(files=files, ranked=[("services/api/client.py", 7.2)])

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    query = "Inspect helpers.py within the Services package and examine APIClient usage."
    symbols, mentions = enhancer.extract_symbols_and_files(query)

    assert "helpers.py" in mentions
    assert "services" in mentions
    assert any(sym in symbols for sym in {"APIClient", "helpers"})


def test_extract_symbols_trims_file_mentions(monkeypatch, temp_workspace, mock_settings):
    """Extracted file mentions should respect the hard limit."""

    repo_map = StubRepoMap(files={}, ranked=[])
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    mention_list = [f"file_{idx}.py" for idx in range(ContextEnhancer.FILE_MENTION_LIMIT + 5)]
    query = "analyse " + " ".join(mention_list)

    symbols, files = enhancer.extract_symbols_and_files(query)
    assert not symbols
    assert len(files) == ContextEnhancer.FILE_MENTION_LIMIT


def test_enhance_query_with_context_reports_unmatched(monkeypatch, temp_workspace, mock_settings):
    """When no ranked files exist, unmatched notices should be appended."""

    class QuietRepoMap(StubRepoMap):
        def get_ranked_files(self, *_, **__):
            return []

    files = {
        "src/app.py": SimpleNamespace(
            file_name="app.py",
            file_stem="app",
            language="python",
            size=200,
            path_parts=("src", "app.py"),
        )
    }
    repo_map = QuietRepoMap(files=files, last_updated=4.0)

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    enhanced = enhancer.enhance_query_with_context(
        "Analyse unknown_file.py for GhostSymbol metrics."
    )

    assert "[RepoMap Notice]" in enhanced
    assert "unknown_file.py" in enhanced
    assert "GhostSymbol" in enhanced


def test_context_for_files_returns_related(monkeypatch, temp_workspace, mock_settings):
    """Additional context helper should omit already supplied files."""

    class RelatedRepoMap(StubRepoMap):
        def get_ranked_files(self, *, mentioned_files, mentioned_symbols, max_files):
            return [("src/extra.py", 0.9), ("src/app.py", 0.5)]

    files = {
        "src/app.py": SimpleNamespace(
            file_name="app.py",
            file_stem="app",
            language="python",
            size=100,
            path_parts=("src", "app.py"),
        ),
        "src/extra.py": SimpleNamespace(
            file_name="extra.py",
            file_stem="extra",
            language="python",
            size=120,
            path_parts=("src", "extra.py"),
        ),
    }
    repo_map = RelatedRepoMap(files=files, last_updated=5.0)

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    related = enhancer.get_context_for_files(["src/app.py"], symbols={"ExtraHelper"})
    assert related == ["src/extra.py"]


def test_context_for_files_handles_errors(monkeypatch, temp_workspace, mock_settings):
    """Errors while gathering related files should yield an empty list."""

    class FailingRepoMap(StubRepoMap):
        def get_ranked_files(self, *args, **kwargs):
            raise RuntimeError("simulated failure")

    repo_map = FailingRepoMap(files={}, last_updated=6.0)

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    assert enhancer.get_context_for_files(["src/app.py"]) == []


def test_get_repomap_messages_tier_three(monkeypatch, temp_workspace, mock_settings):
    """Tier three fallback should rely on important files when PageRank yields nothing."""

    class TieredRepoMap:
        def __init__(self):
            file_info = SimpleNamespace(
                file_name="core.py",
                file_stem="core",
                language="python",
                size=256,
                path_parts=("src", "core.py"),
            )
            self.context = SimpleNamespace(
                files={"src/core.py": file_info, "docs/readme.md": file_info},
                symbol_index={},
            )
            self.calls = []

        def get_ranked_files(self, *, mentioned_files, mentioned_symbols, max_files):
            self.calls.append((set(mentioned_files), set(mentioned_symbols), max_files))
            return []

    repo_map = TieredRepoMap()
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    monkeypatch.setattr(
        enhancer,
        "_get_important_files",
        lambda all_files, max_files: [("src/core.py", 6.0), ("docs/readme.md", 4.0)],
    )

    query, messages = enhancer.get_repomap_messages(
        "Explore repository overview",
        max_files=2,
        additional_files={"seed.txt"},
        additional_symbols={"SeedSymbol"},
    )

    assert messages is not None
    assert messages[0]["content"].startswith("Here")
    assert repo_map.calls[-1][0] == set()


def test_context_enhancer_git_ls_files_failure(monkeypatch, temp_workspace, mock_settings):
    """git ls-files errors should fall back to filesystem listing."""

    class DummyProcess:
        returncode = 1
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.subprocess.run",
        lambda *args, **kwargs: DummyProcess(),
    )
    repo_map = StubRepoMap(files={}, last_updated=2.0)
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    (temp_workspace / "module.py").write_text("print('hi')\n", encoding="utf-8")

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    tracked = enhancer._load_tracked_files()

    assert "module.py" in tracked


def test_context_enhancer_load_tracked_files_missing_workspace(
    monkeypatch, mock_settings, tmp_path
):
    """Non-existent workspaces should return an empty file list."""
    missing = tmp_path / "no_such_dir"
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: StubRepoMap(files={}, last_updated=0.0),
    )
    enhancer = ContextEnhancer(workspace=missing, settings=mock_settings)
    assert enhancer._load_tracked_files() == []


def test_list_repository_files_enforces_limit(monkeypatch, temp_workspace, mock_settings):
    """Explicit limits should trim filtered file listings."""
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    monkeypatch.setattr(enhancer, "_get_git_signature", lambda: (1, "sig"))
    monkeypatch.setattr(enhancer, "_load_tracked_files", lambda: ["a.py", "b.py", "c.py"])
    monkeypatch.setattr(enhancer, "_filter_discoverable_files", lambda files: files)

    subset = enhancer.list_repository_files(limit=1)
    assert subset == ["a.py"]


def test_list_repository_files_cache_disabled_without_git_metadata(
    monkeypatch, temp_workspace, mock_settings
):
    """When git metadata is unavailable, caching should be bypassed."""
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    monkeypatch.setattr(enhancer, "_get_git_signature", lambda: (None, None))
    monkeypatch.setattr(enhancer, "_load_tracked_files", lambda: ["a.py", "b.py"])
    monkeypatch.setattr(enhancer, "_filter_discoverable_files", lambda files: files)

    files = enhancer.list_repository_files(use_cache=True)
    assert files == ["a.py", "b.py"]
    assert enhancer._file_discovery_cache is None


def test_get_repomap_messages_requires_existing_workspace(mock_settings, tmp_path):
    """Non-existent workspaces should short-circuit message generation."""
    missing = tmp_path / "absent"
    enhancer = ContextEnhancer(workspace=missing, settings=mock_settings)

    query, messages = enhancer.get_repomap_messages("anything")

    assert messages is None
    assert query == "anything"


def test_enhance_query_with_context_missing_workspace(monkeypatch, mock_settings, tmp_path):
    """Enhancement should no-op when the workspace directory is missing."""

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: StubRepoMap(files={}, ranked=[]),
    )

    missing = tmp_path / "missing"
    enhancer = ContextEnhancer(workspace=missing, settings=mock_settings)

    assert enhancer.enhance_query_with_context("Review foo") == "Review foo"


def test_enhance_query_handles_repomap_failure(monkeypatch, temp_workspace, mock_settings):
    """Enhancer should gracefully handle RepoMap errors."""

    class FailingRepoMap(StubRepoMap):
        def get_ranked_files(self, *args, **kwargs):
            raise RuntimeError("boom")

    files = {
        "src/app.py": SimpleNamespace(
            file_name="app.py",
            file_stem="app",
            language="python",
            size=200,
            path_parts=("src", "app.py"),
        )
    }
    repo_map = FailingRepoMap(files=files, ranked=[])

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    assert enhancer.enhance_query_with_context("Check src/app.py") == "Check src/app.py"


def test_enhance_query_returns_query_when_no_mentions(monkeypatch, temp_workspace, mock_settings):
    """Enhancement should return the original query when no symbols or files are detected."""

    repo_map = StubRepoMap(files={}, ranked=[])
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    assert enhancer.enhance_query_with_context("the when where how") == "the when where how"


def test_get_context_for_files_handles_exception(monkeypatch, temp_workspace, mock_settings):
    """Context discovery should return an empty list when RepoMap raises."""

    class ErrorRepoMap(StubRepoMap):
        def get_ranked_files(self, *args, **kwargs):
            raise RuntimeError("cannot rank")

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: ErrorRepoMap(files={}, ranked=[]),
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    assert enhancer.get_context_for_files(["src/app.py"]) == []


def test_context_enhancer_singleton_helpers(monkeypatch, temp_workspace, mock_settings):
    """Singleton helpers should reuse cached instances and expose helper API."""

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: StubRepoMap(files={}, ranked=[]),
    )
    monkeypatch.setattr("ai_dev_agent.cli.context_enhancer._context_enhancer", None)

    first = get_context_enhancer(temp_workspace, mock_settings)
    second = get_context_enhancer(temp_workspace, mock_settings)
    assert first is second

    query, messages = enhance_query("List repo files", workspace=temp_workspace)
    assert query == "List repo files"
    assert messages is None or isinstance(messages, list)


def test_enhance_query_with_context_debug_failure(monkeypatch, temp_workspace, mock_settings):
    """Debug logging path should be exercised when enhancement fails."""

    class DebugRepoMap(StubRepoMap):
        def get_ranked_files(self, *args, **kwargs):
            raise RuntimeError("debug failure")

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: DebugRepoMap(files={}, ranked=[]),
    )

    mock_settings.repomap_debug_stdout = True
    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    assert (
        enhancer.enhance_query_with_context("Investigate RunLoop behaviour")
        == "Investigate RunLoop behaviour"
    )


def test_get_repomap_messages_tier_two_low_relevance(monkeypatch, temp_workspace, mock_settings):
    """Tier two should include low relevance entries when space permits."""

    class TierTwoRepoMap(StubRepoMap):
        def __init__(self):
            files = {
                "src/alpha.py": SimpleNamespace(
                    file_name="alpha.py",
                    file_stem="alpha",
                    language="python",
                    size=128,
                    path_parts=("src", "alpha.py"),
                )
            }
            super().__init__(files=files, ranked=[("src/alpha.py", 2.5)])
            self.calls = []

        def get_ranked_files(self, *, mentioned_files, mentioned_symbols, max_files):
            self.calls.append((set(mentioned_files), set(mentioned_symbols), max_files))
            if len(self.calls) == 1:
                return []  # Tier 1 -> no matches
            return [("src/alpha.py", 2.5)]

    repo_map = TierTwoRepoMap()
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    monkeypatch.setattr(
        enhancer,
        "_get_important_files",
        lambda all_files, max_files: [("src/alpha.py", 2.5)],
    )

    query, messages = enhancer.get_repomap_messages("Investigate alpha.py flow", max_files=2)

    assert "  • src/alpha.py" in messages[0]["content"]


def test_get_repomap_messages_includes_medium_and_low(monkeypatch, temp_workspace, mock_settings):
    """Tier 2 message formatting should include medium and low relevance entries."""

    ranked = [
        ("src/high_a.py", 12.0),
        ("src/high_b.py", 11.0),
        ("src/medium.py", 6.0),
        ("src/low.py", 3.1),
    ]
    files = {
        path: SimpleNamespace(
            file_name=Path(path).name,
            file_stem=Path(path).stem,
            language="python",
            size=128,
            path_parts=tuple(Path(path).parts),
        )
        for path, _ in ranked
    }
    repo_map = StubRepoMap(files=files, ranked=ranked)

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    query_text = "Investigate src/high_a.py alongside src/medium.py and src/low.py"
    _, messages = enhancer.get_repomap_messages(query_text, max_files=4)

    assert messages is not None
    content = messages[0]["content"]
    assert "high_a.py" in content
    assert "medium.py" in content
    assert "low.py" in content


def test_get_repomap_messages_high_relevance_assistant(monkeypatch, temp_workspace, mock_settings):
    """Assistant response should reflect the number of highly ranked files."""

    ranked = [(f"src/service_{idx}.py", 12.0) for idx in range(4)]
    files = {
        path: SimpleNamespace(
            file_name=Path(path).name,
            file_stem=Path(path).stem,
            language="python",
            size=128,
            path_parts=tuple(Path(path).parts),
        )
        for path, _ in ranked
    }
    repo_map = StubRepoMap(files=files, ranked=ranked)

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    _, messages = enhancer.get_repomap_messages("Review service modules", max_files=4)

    assert messages is not None
    assert messages[1]["content"].startswith("I can see 4 relevant files")


def test_get_repomap_messages_returns_none_when_all_tiers_fail(
    monkeypatch, temp_workspace, mock_settings
):
    """All fallback tiers should decline to emit messages when no files are available."""

    class EmptyRepoMap(StubRepoMap):
        def get_ranked_files(self, *, mentioned_files, mentioned_symbols, max_files):
            return []

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: EmptyRepoMap(files={}, ranked=[]),
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    query, messages = enhancer.get_repomap_messages(
        "Review components", max_files=2, additional_symbols={"Hint"}
    )

    assert messages is None
    assert query == "Review components"


def test_get_repomap_messages_tier_three_fallback(monkeypatch, temp_workspace, mock_settings):
    """Tier three should fall back to important files when PageRank provides no matches."""

    class TierThreeRepoMap(StubRepoMap):
        def __init__(self):
            files = {
                "src/seed.py": SimpleNamespace(
                    file_name="seed.py",
                    file_stem="seed",
                    language="python",
                    size=256,
                    path_parts=("src", "seed.py"),
                ),
                "docs/info.md": SimpleNamespace(
                    file_name="info.md",
                    file_stem="info",
                    language="markdown",
                    size=120,
                    path_parts=("docs", "info.md"),
                ),
            }
            super().__init__(files=files, ranked=[])
            self.calls = []

        def get_ranked_files(self, *, mentioned_files, mentioned_symbols, max_files):
            self.calls.append((set(mentioned_files), set(mentioned_symbols), max_files))
            return []

    repo_map = TierThreeRepoMap()
    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
    monkeypatch.setattr(
        enhancer,
        "_get_important_files",
        lambda all_files, max_files: [("src/seed.py", 4.0), ("docs/info.md", 2.0)],
    )

    query, messages = enhancer.get_repomap_messages(
        "Explore onboarding flow", max_files=2, additional_files={"seed.txt"}
    )

    assert messages is not None
    assert "seed.py" in messages[0]["content"]
    assert "I can see a high-level view" in messages[1]["content"]
    assert len(repo_map.calls) >= 3
