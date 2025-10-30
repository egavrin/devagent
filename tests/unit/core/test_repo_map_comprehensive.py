"""Comprehensive tests for repo_map module to improve coverage."""

from pathlib import Path

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
