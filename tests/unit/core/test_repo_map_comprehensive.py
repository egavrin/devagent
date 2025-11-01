"""Comprehensive tests for repo_map module to improve coverage."""

from pathlib import Path

import networkx as nx
import pytest

from ai_dev_agent.core.repo_map import FileInfo, RepoContext, RepoMap, RepoMapManager


class TestRepoMapManager:
    """Test the RepoMapManager singleton."""

    def test_get_instance_singleton(self, tmp_path):
        """Test that get_instance returns singleton."""
        instance1 = RepoMapManager.get_instance(tmp_path)
        instance2 = RepoMapManager.get_instance(tmp_path)

        assert instance1 is instance2

    def test_clear_instance(self, tmp_path):
        """Test clearing singleton instance."""
        instance1 = RepoMapManager.get_instance(tmp_path)
        RepoMapManager.clear_instance(tmp_path)
        instance2 = RepoMapManager.get_instance(tmp_path)

        assert instance1 is not instance2


class TestFileInfo:
    """Test FileInfo dataclass."""

    def test_file_info_creation(self):
        """Test creating FileInfo instance."""
        info = FileInfo(
            path="test.py",
            size=100,
            modified_time=1234567890.0,
            language="python",
            file_name="test.py",
            file_stem="test",
        )

        assert info.path == "test.py"
        assert info.size == 100
        assert info.modified_time == 1234567890.0
        assert info.language == "python"
        assert info.file_name == "test.py"
        assert info.file_stem == "test"
        assert info.symbols == []
        assert info.imports == []


class TestRepoContext:
    """Test RepoContext dataclass."""

    def test_repo_context_creation(self):
        """Test creating RepoContext instance."""

        context = RepoContext(root_path=Path("/test"))

        assert context.root_path == Path("/test")
        assert context.files == {}
        assert isinstance(context.symbol_index, dict)
        assert isinstance(context.import_graph, dict)
        assert context.dependency_graph is None


class TestRepoMap:
    """Test RepoMap main functionality."""

    @pytest.fixture
    def repo_map(self, tmp_path):
        """Create a RepoMap instance for testing."""
        return RepoMap(root_path=tmp_path, cache_enabled=False)

    def test_init_basic(self, tmp_path):
        """Test basic initialization."""
        repo_map = RepoMap(root_path=tmp_path)

        assert repo_map.root_path == tmp_path
        assert repo_map.context is not None
        assert isinstance(repo_map.context, RepoContext)

    def test_init_with_cache(self, tmp_path):
        """Test initialization with cache."""
        repo_map = RepoMap(root_path=tmp_path, cache_enabled=True)

        assert repo_map.cache_enabled is True
        # Cache path is configured
        assert repo_map.cache_path.name == "repo_map.json"

    def test_should_skip_file(self, repo_map, tmp_path):
        """Test file skipping logic."""
        # Create test files in a generated directory
        (tmp_path / "build").mkdir()
        build_file = tmp_path / "build" / "output.js"
        build_file.touch()

        should_skip, reason = repo_map._should_skip_file(build_file)
        assert should_skip is True
        assert "generated directory" in reason.lower()

        # Test generated file pattern
        gen_file = tmp_path / "test_generated.py"
        gen_file.touch()

        should_skip, reason = repo_map._should_skip_file(gen_file)
        assert should_skip is True
        assert "generated file pattern" in reason.lower()

        # Test normal file
        normal_file = tmp_path / "test.py"
        normal_file.touch()

        should_skip, reason = repo_map._should_skip_file(normal_file)
        assert should_skip is False

    def test_detect_language(self, repo_map, tmp_path):
        """Test language detection."""
        # Test Python file
        py_file = tmp_path / "test.py"
        py_file.touch()
        assert repo_map._detect_language(py_file) == "python"

        # Test JavaScript file
        js_file = tmp_path / "test.js"
        js_file.touch()
        assert repo_map._detect_language(js_file) == "javascript"

        # Test TypeScript file
        ts_file = tmp_path / "test.ts"
        ts_file.touch()
        assert repo_map._detect_language(ts_file) == "typescript"

        # Test unknown file
        unknown_file = tmp_path / "test.xyz"
        unknown_file.touch()
        assert repo_map._detect_language(unknown_file) is None

    def test_scan_repository_empty(self, repo_map):
        """Test scanning empty repository."""
        repo_map.scan_repository()

        assert len(repo_map.context.files) == 0

    def test_scan_repository_with_files(self, repo_map, tmp_path):
        """Test scanning repository with files."""
        # Create test files
        (tmp_path / "test1.py").write_text("def hello(): pass")
        (tmp_path / "test2.js").write_text("function world() {}")

        repo_map.scan_repository()

        assert len(repo_map.context.files) == 2
        assert "test1.py" in repo_map.context.files
        assert "test2.js" in repo_map.context.files

    def test_scan_file_python(self, repo_map, tmp_path):
        """Test scanning Python file."""
        py_file = tmp_path / "test.py"
        py_file.write_text(
            """
class TestClass:
    def test_method(self):
        pass

def test_function():
    return 42
"""
        )

        hash_value = repo_map._scan_file(py_file)

        assert hash_value != ""
        assert "test.py" in repo_map.context.files
        file_info = repo_map.context.files["test.py"]
        assert "TestClass" in file_info.symbols
        assert "test_function" in file_info.symbols

    def test_dependency_sets_populated_after_graph_build(self, repo_map, tmp_path):
        """Dependency graph should update FileInfo dependencies."""
        (tmp_path / "service.py").write_text(
            "class Service:\n    pass\n",
            encoding="utf-8",
        )
        (tmp_path / "consumer.py").write_text(
            "def needs_service():\n    return Service()\n",
            encoding="utf-8",
        )

        repo_map.scan_repository(force=True)

        assert repo_map.context.files["consumer.py"].dependencies == set()

        repo_map.build_dependency_graph()

        assert "service.py" in repo_map.context.files["consumer.py"].dependencies

    def test_scan_repository_invalidates_cached_graph_and_pagerank(self, repo_map, tmp_path):
        """Rescans should invalidate cached dependency graph and PageRank scores."""
        (tmp_path / "alpha.py").write_text("class Alpha:\n    pass\n", encoding="utf-8")

        repo_map.scan_repository(force=True)
        repo_map.build_dependency_graph()
        repo_map.compute_pagerank()

        assert repo_map.context.dependency_graph is not None
        assert repo_map.context.pagerank_scores
        assert repo_map.context.last_pagerank_update > 0

        (tmp_path / "beta.py").write_text("print('beta')\n", encoding="utf-8")

        repo_map.scan_repository(force=True)

        assert repo_map.context.dependency_graph is None
        assert repo_map.context.pagerank_scores == {}
        assert repo_map.context.last_pagerank_update == 0.0

    def test_extract_python_info(self, repo_map, tmp_path):
        """Test extracting Python file information."""
        py_file = tmp_path / "module.py"
        py_file.write_text(
            """
import os
from pathlib import Path

class MyClass:
    '''A test class'''

    def __init__(self):
        self.value = 0

    def method(self, arg):
        return arg * 2

def standalone_function(x, y):
    '''A standalone function'''
    return x + y

async def async_func():
    pass
"""
        )

        file_info = FileInfo(
            path="module.py",
            size=0,
            modified_time=0,
            language="python",
            file_name="module.py",
            file_stem="module",
        )

        repo_map._extract_python_info(py_file, file_info)

        assert "MyClass" in file_info.symbols
        assert "standalone_function" in file_info.symbols
        # Note: async_func is not extracted (only ast.FunctionDef, not AsyncFunctionDef)
        assert "os" in file_info.imports
        # ImportFrom extracts module name, not fully qualified
        assert "pathlib" in file_info.imports

    def test_is_noisy_symbol(self, repo_map):
        """Test noisy symbol detection."""
        # Common noisy symbols - single letters
        assert repo_map._is_noisy_symbol("i") is True
        assert repo_map._is_noisy_symbol("e") is True
        # Generic names
        assert repo_map._is_noisy_symbol("data") is True
        assert repo_map._is_noisy_symbol("value") is True

        # Good symbols
        assert repo_map._is_noisy_symbol("UserManager") is False
        assert repo_map._is_noisy_symbol("calculate_total") is False

    def test_is_well_named_symbol(self, repo_map):
        """Test well-named symbol detection."""
        # Well-named symbols
        assert repo_map._is_well_named_symbol("UserManager") is True
        assert repo_map._is_well_named_symbol("calculate_total") is True
        assert repo_map._is_well_named_symbol("HTTPClient") is True

        # Poorly named symbols
        assert repo_map._is_well_named_symbol("x") is False
        assert repo_map._is_well_named_symbol("tmp") is False
        assert repo_map._is_well_named_symbol("_") is False

    def test_get_file_priority_multiplier(self, repo_map):
        """Test file priority calculation."""
        # Core/source files get higher priority
        assert repo_map._get_file_priority_multiplier("src/main.py") > 1.0
        assert repo_map._get_file_priority_multiplier("core/engine.py") > 1.0
        assert repo_map._get_file_priority_multiplier("lib/utils.py") > 1.0

        # Test files (lower priority)
        assert repo_map._get_file_priority_multiplier("test_something.py") < 1.0
        assert repo_map._get_file_priority_multiplier("tests/test_module.py") < 1.0

        # Regular files have default priority
        assert repo_map._get_file_priority_multiplier("main.py") == 1.0
        assert repo_map._get_file_priority_multiplier("config.json") == 1.0

    def test_find_symbol(self, repo_map, tmp_path):
        """Test finding symbols."""
        # Create test file with symbols
        py_file = tmp_path / "module.py"
        py_file.write_text(
            """
class UserManager:
    def get_user(self): pass

def process_data():
    pass
"""
        )

        repo_map.scan_repository()

        # Find class
        results = repo_map.find_symbol("UserManager")
        assert "module.py" in results

        # Find function
        results = repo_map.find_symbol("process_data")
        assert "module.py" in results

        # Non-existent symbol
        results = repo_map.find_symbol("NonExistent")
        assert len(results) == 0

    def test_get_file_summary(self, repo_map, tmp_path):
        """Test getting file summary."""
        py_file = tmp_path / "example.py"
        py_file.write_text(
            """
class Example:
    def method1(self): pass
    def method2(self): pass

def helper(): pass
"""
        )

        repo_map.scan_repository()

        summary = repo_map.get_file_summary("example.py")
        assert summary is not None
        assert "Example" in summary
        assert "helper" in summary

    def test_invalidate_file(self, repo_map, tmp_path):
        """Test invalidating file cache."""
        py_file = tmp_path / "cached.py"
        py_file.write_text("def original(): pass")

        repo_map.scan_repository()
        assert "cached.py" in repo_map.context.files

        # Invalidate
        repo_map.invalidate_file("cached.py")
        assert "cached.py" not in repo_map.context.files

    def test_cache_operations(self, tmp_path):
        """Test cache save and load."""
        # Create repo map and scan
        repo_map1 = RepoMap(root_path=tmp_path, cache_enabled=True)
        (tmp_path / "test.py").write_text("def cached_func(): pass")
        repo_map1.scan_repository()

        # Force save cache
        repo_map1._save_cache()

        # Create new instance and load cache
        repo_map2 = RepoMap(root_path=tmp_path, cache_enabled=True)

        # Cache should be loaded
        assert "test.py" in repo_map2.context.files
        assert "cached_func" in repo_map2.context.files["test.py"].symbols

    def test_get_ranked_files_basic(self, repo_map, tmp_path):
        """Test basic file ranking."""
        # Create test files
        (tmp_path / "main.py").write_text("class Main: pass")
        (tmp_path / "helper.py").write_text("def help(): pass")
        (tmp_path / "test.py").write_text("def test(): pass")

        repo_map.scan_repository()

        ranked = repo_map.get_ranked_files(["Main"], [], max_files=2)

        assert len(ranked) <= 2
        # Files with matching symbols should rank higher
        if ranked:
            assert any("main.py" in str(f[0]) for f in ranked)

    def test_extract_javascript_info(self, repo_map, tmp_path):
        """Test extracting JavaScript file information."""
        js_file = tmp_path / "module.js"
        js_file.write_text(
            """
class UserController {
    constructor() {}
    getUser() {}
}

function processData(data) {
    return data;
}

const helper = () => {};
export default UserController;
"""
        )

        file_info = FileInfo(
            path="module.js",
            size=0,
            modified_time=0,
            language="javascript",
            file_name="module.js",
            file_stem="module",
        )

        repo_map._extract_with_regex(js_file, file_info, "javascript")

        assert "UserController" in file_info.symbols
        assert "processData" in file_info.symbols

    def test_extract_typescript_info(self, repo_map, tmp_path):
        """Test extracting TypeScript file information."""
        ts_file = tmp_path / "module.ts"
        ts_file.write_text(
            """
interface User {
    id: number;
    name: string;
}

class UserService {
    getUser(id: number): User {
        return { id, name: "test" };
    }
}

export function createUser(name: string): User {
    return { id: 1, name };
}
"""
        )

        file_info = FileInfo(
            path="module.ts",
            size=0,
            modified_time=0,
            language="typescript",
            file_name="module.ts",
            file_stem="module",
        )

        repo_map._extract_typescript_info(ts_file, file_info)

        assert "UserService" in file_info.symbols
        assert "createUser" in file_info.symbols
        # Note: Interface detection depends on regex patterns

    def test_scan_file_io_error(self, repo_map, tmp_path):
        """Test scan_file with IO error."""
        # Create a file that can't be read by making it a directory
        error_file = tmp_path / "error.py"
        error_file.mkdir()  # Create as directory, not a file

        # This should handle the error gracefully
        hash_value = repo_map._scan_file(error_file)

        # Returns a dummy value on error
        assert hash_value == "scanned"

    # Note: Removed test_normalize_mentions and test_symbol_match_score as their
    # signatures don't match what tests expected (they work with FileInfo objects)

    def test_rebuild_indices(self, repo_map, tmp_path):
        """Test rebuilding symbol indices."""
        # Create test file
        py_file = tmp_path / "indexed.py"
        py_file.write_text(
            """
class IndexedClass:
    pass

def indexed_function():
    pass
"""
        )

        repo_map.scan_repository()

        # Indices should be built
        assert "IndexedClass" in repo_map.context.symbol_index
        assert "indexed_function" in repo_map.context.symbol_index
        assert "indexed.py" in repo_map.context.symbol_index["IndexedClass"]

    def test_get_dependencies(self, repo_map, tmp_path):
        """Test getting file dependencies."""
        # Create files with imports
        (tmp_path / "main.py").write_text("import os")

        repo_map.scan_repository()

        # Dependencies are stored but may be empty initially
        deps = repo_map.get_dependencies("main.py")

        # Should return a set (may be empty as dependencies aren't auto-populated from imports)
        assert isinstance(deps, set)

    def test_scan_repository_force(self, repo_map, tmp_path):
        """Test force scanning repository."""
        # Create initial file
        py_file = tmp_path / "initial.py"
        py_file.write_text("def initial(): pass")

        repo_map.scan_repository()
        assert "initial.py" in repo_map.context.files

        # Modify file
        py_file.write_text("def modified(): pass")

        # Scan without force (might use cache)
        repo_map.scan_repository(force=False)

        # Scan with force
        repo_map.scan_repository(force=True)
        file_info = repo_map.context.files["initial.py"]
        assert "modified" in file_info.symbols

    def test_extract_cpp_info_captures_symbols_and_imports(self, repo_map, tmp_path):
        """Ensure C++ extraction captures diverse symbol types."""
        cpp_file = tmp_path / "engine.cpp"
        cpp_file.write_text(
            """
#include "engine/base.h"
#define EXPERIMENTAL_FLAG
template <typename T>
class EngineCore : public BaseEngine<T> {
public:
    void ComputeValue();
};

void EngineCore<int>::ComputeValue() {
    HelperUtility();
}
"""
        )

        file_info = FileInfo(
            path="engine.cpp",
            size=0,
            modified_time=0,
            language="cpp",
            file_name="engine.cpp",
            file_stem="engine",
        )

        repo_map._extract_cpp_info(cpp_file, file_info)

        assert {"EngineCore", "ComputeValue", "EXPERIMENTAL_FLAG"} <= set(file_info.symbols)
        assert "engine/base.h" in file_info.imports
        assert "BaseEngine" in file_info.symbols_used

    def test_extract_java_and_rust_info(self, repo_map, tmp_path):
        """Verify Java and Rust extraction handle packages, imports, and symbols."""
        java_file = tmp_path / "Service.java"
        java_file.write_text(
            """
package com.example.project;

import com.example.lib.HelperUtil;

public class ProjectService {
    private HelperUtil helper;

    public ProjectService() {}

    public String process() {
        return helper.toString();
    }
}
"""
        )

        java_info = FileInfo(
            path="Service.java",
            size=0,
            modified_time=0,
            language="java",
            file_name="Service.java",
            file_stem="Service",
        )

        repo_map._extract_java_info(java_file, java_info)

        assert {"ProjectService", "process"} <= set(java_info.symbols)
        assert "com.example.lib.HelperUtil" in java_info.imports
        assert "package:com.example.project" in java_info.imports

        rust_file = tmp_path / "lib.rs"
        rust_file.write_text(
            """
use crate::prelude::Logger;

pub struct DataStore {
    value: i32,
}

impl DataStore {
    pub fn new() -> Self {
        Self { value: 0 }
    }
}

pub fn fetch_data() -> DataStore {
    DataStore::new()
}

mod tests {
    pub fn helper_test() {}
}
"""
        )

        rust_info = FileInfo(
            path="lib.rs",
            size=0,
            modified_time=0,
            language="rust",
            file_name="lib.rs",
            file_stem="lib",
        )

        repo_map._extract_rust_info(rust_file, rust_info)

        assert {"DataStore", "fetch_data", "tests"} <= set(rust_info.symbols)
        assert "crate::prelude::Logger" in rust_info.imports

    def test_normalize_mentions_limits_outputs_trimmed(self, repo_map):
        """Large mention sets should be trimmed deterministically."""
        mentions = {f"src/module_{idx}.py" for idx in range(repo_map.MAX_FASTPATH_MENTIONS + 10)}
        mentions.update({f"src/subdir_{idx}/" for idx in range(repo_map.MAX_DIRECTORY_MATCHES + 5)})

        trimmed, names, stems, directories = repo_map._normalize_mentions(mentions)

        assert len(trimmed) == repo_map.MAX_FASTPATH_MENTIONS
        assert all(name.endswith(".py") for name in names)
        assert len(directories) <= repo_map.MAX_DIRECTORY_MATCHES

    def test_quick_rank_by_symbols_combines_signals(self, repo_map, tmp_path):
        """Quick ranking should accumulate symbol, file, and directory boosts."""
        file_path = "src/app/main.py"
        on_disk = tmp_path / "src" / "app" / "main.py"
        on_disk.parent.mkdir(parents=True, exist_ok=True)
        on_disk.write_text("class VerySpecificClass:\n    pass\n", encoding="utf-8")

        info = FileInfo(
            path=file_path,
            size=on_disk.stat().st_size,
            modified_time=on_disk.stat().st_mtime,
            language="python",
            symbols=["VerySpecificClass"],
            symbols_used=["VerySpecificClass"],
            file_name="main.py",
            file_stem="main",
            path_parts=("src", "app", "main.py"),
        )
        repo_map.context.files[file_path] = info

        ranked = repo_map._quick_rank_by_symbols(
            {file_path},
            {"VerySpecificClass"},
            max_files=3,
            mentioned_names={"main.py"},
            mentioned_stems={"main"},
            directory_mentions=("src/app",),
            long_symbol_prefixes=("VerySpec",),
        )

        assert ranked
        score = dict(ranked)[file_path]
        assert score >= 1100.0

    def test_get_ranked_files_fast_path_skips_pagerank(self, repo_map, tmp_path, monkeypatch):
        """Direct symbol matches should avoid the expensive PageRank path."""
        target_file = tmp_path / "main.py"
        target_file.write_text("class VerySpecificClass:\n    pass\n", encoding="utf-8")

        info = FileInfo(
            path="main.py",
            size=target_file.stat().st_size,
            modified_time=target_file.stat().st_mtime,
            language="python",
            symbols=["VerySpecificClass"],
            file_name="main.py",
            file_stem="main",
            path_parts=("main.py",),
        )
        repo_map.context.files["main.py"] = info
        repo_map._rebuild_indices()

        def fail_compute(*args, **kwargs):
            raise AssertionError("PageRank should not be computed for fast-path matches")

        monkeypatch.setattr(repo_map, "compute_pagerank", fail_compute)

        ranked = repo_map.get_ranked_files({"main.py"}, {"VerySpecificClass"}, max_files=1)

        assert ranked and ranked[0][0] == "main.py"

    def test_get_ranked_files_directory_mentions_triggers_pagerank(
        self, repo_map, tmp_path, monkeypatch
    ):
        """Directory-only hints should fall back to PageRank ranking."""
        src_dir = tmp_path / "src" / "app"
        src_dir.mkdir(parents=True)
        (src_dir / "core.py").write_text("class Core:\n    pass\n", encoding="utf-8")
        (src_dir / "utils.py").write_text("class Util:\n    pass\n", encoding="utf-8")

        repo_map.context.files["src/app/core.py"] = FileInfo(
            path="src/app/core.py",
            size=1,
            modified_time=0,
            language="python",
            symbols=["Core"],
            file_name="core.py",
            file_stem="core",
            path_parts=("src", "app", "core.py"),
        )
        repo_map.context.files["src/app/utils.py"] = FileInfo(
            path="src/app/utils.py",
            size=1,
            modified_time=0,
            language="python",
            symbols=["Util"],
            file_name="utils.py",
            file_stem="utils",
            path_parts=("src", "app", "utils.py"),
        )
        repo_map._rebuild_indices()
        repo_map.context.pagerank_scores = {}
        repo_map.context.dependency_graph = None

        def fake_build():
            graph = nx.DiGraph()
            graph.add_edge("src/app/core.py", "src/app/utils.py", weight=2.0)
            repo_map.context.dependency_graph = graph
            return graph

        compute_calls = {}

        def fake_compute(*args, **kwargs):
            compute_calls["count"] = compute_calls.get("count", 0) + 1
            repo_map.context.pagerank_scores = {
                "src/app/core.py": 0.6,
                "src/app/utils.py": 0.4,
            }
            return repo_map.context.pagerank_scores

        monkeypatch.setattr(repo_map, "build_dependency_graph", fake_build)
        monkeypatch.setattr(repo_map, "compute_pagerank", fake_compute)

        ranked = repo_map.get_ranked_files({"src/app"}, set(), max_files=2)

        assert compute_calls.get("count", 0) == 1
        assert ranked and ranked[0][0] == "src/app/core.py"

    def test_compute_pagerank_personalization_preserves_cache(self, repo_map):
        """Personalized PageRank should not mutate the cached baseline."""
        graph = nx.DiGraph()
        graph.add_edge("a.py", "b.py", weight=2.0)
        graph.add_edge("b.py", "c.py", weight=1.0)
        repo_map.context.dependency_graph = graph

        base_scores = repo_map.compute_pagerank()
        assert repo_map.context.pagerank_scores == base_scores

        personalized = repo_map.compute_pagerank({"b.py": 1.0}, cache_results=False)

        assert repo_map.context.pagerank_scores == base_scores
        assert personalized != base_scores
        assert pytest.approx(sum(personalized.values()), rel=1e-6) == 1.0

    def test_scan_repository_large_repo_resets_caches(self, tmp_path, monkeypatch):
        """Large rescans should prune stale files and clear ranking caches."""
        repo_map = RepoMap(root_path=tmp_path, cache_enabled=False, use_tree_sitter=False)
        repo_map.context.files["stale.py"] = FileInfo(
            path="stale.py",
            size=1,
            modified_time=0,
            language="python",
            file_name="stale.py",
            file_stem="stale",
            path_parts=("stale.py",),
        )
        repo_map.context.pagerank_scores = {"stale.py": 0.5}
        repo_map.context.dependency_graph = nx.DiGraph()
        repo_map.context.last_pagerank_update = 123.0

        class DummyEntry:
            def __init__(self, root: Path, relative: str) -> None:
                self._root = root
                self._relative = relative
                self.name = Path(relative).name
                self.parts = (root / relative).parts

            def is_file(self) -> bool:
                return True

            @property
            def suffix(self) -> str:
                return Path(self._relative).suffix

            def relative_to(self, root: Path) -> Path:
                assert root == self._root
                return Path(self._relative)

        entries = [DummyEntry(tmp_path, f"src/file_{idx}.py") for idx in range(15050)]

        original_rglob = Path.rglob

        def fake_rglob(self, pattern):
            if self == tmp_path:
                yield from entries
            else:
                yield from original_rglob(self, pattern)

        monkeypatch.setattr(Path, "rglob", fake_rglob, raising=False)
        monkeypatch.setattr(repo_map, "_save_cache", lambda: None)

        def fake_scan(self, file_path):
            rel = str(file_path.relative_to(self.root_path))
            info = FileInfo(
                path=rel,
                size=64,
                modified_time=0,
                language="python",
                file_name=Path(rel).name,
                file_stem=Path(rel).stem,
                path_parts=tuple(Path(rel).parts),
            )
            self.context.files[rel] = info
            return "scanned"

        monkeypatch.setattr(repo_map, "_scan_file", fake_scan.__get__(repo_map, RepoMap))

        repo_map.scan_repository(force=True)

        assert len(repo_map.context.files) == len(entries)
        assert "stale.py" not in repo_map.context.files
        assert repo_map.context.dependency_graph is None
        assert repo_map.context.pagerank_scores == {}
        assert repo_map.context.last_pagerank_update == 0.0

    def test_msgpack_cache_roundtrip_rebuilds_indices(self, tmp_path):
        """Msgpack caches should restore symbols and ranking metadata."""
        repo_map = RepoMap(root_path=tmp_path, cache_enabled=True, use_tree_sitter=False)
        (tmp_path / "models.py").write_text("class CacheModel:\n    pass\n", encoding="utf-8")
        repo_map.scan_repository(force=True)
        repo_map.context.pagerank_scores = {"models.py": 0.42}
        repo_map.context.last_pagerank_update = 321.0
        repo_map._save_cache()

        cache_path = repo_map.cache_path.with_suffix(".msgpack")
        assert cache_path.exists()

        reloaded = RepoMap(root_path=tmp_path, cache_enabled=True, use_tree_sitter=False)

        assert "models.py" in reloaded.context.files
        assert "CacheModel" in reloaded.context.symbol_index
        assert reloaded.context.pagerank_scores.get("models.py") == pytest.approx(0.42)
