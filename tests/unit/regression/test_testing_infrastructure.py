"""Tests for the testing infrastructure modules.

This module tests the testing utilities to ensure they work correctly.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ai_dev_agent.testing.coverage_gate import CoverageGate, CoverageResult, check_coverage
from ai_dev_agent.testing.coverage_report import CoverageReporter, CoverageTrend
from ai_dev_agent.testing.helpers import (
    compare_json_structures,
    create_mock_response,
    create_test_project,
    generate_test_data,
    run_with_timeout,
    temporary_env,
    validate_schema,
    wait_for_condition,
)
from ai_dev_agent.testing.mocks import (
    MockCache,
    MockDatabase,
    MockFileSystem,
    MockGitRepo,
    MockHTTPClient,
    MockLLM,
    create_mock_agent,
    create_mock_tool,
)


class TestCoverageGate:
    """Test coverage gate functionality."""

    def test_coverage_gate_initialization(self):
        """Test coverage gate initialization."""
        gate = CoverageGate(threshold=90.0)
        assert gate.threshold == 90.0
        assert gate.project_root == Path.cwd()

    def test_coverage_result_dataclass(self):
        """Test coverage result data structure."""
        result = CoverageResult(
            total_coverage=85.5,
            passed=False,
            threshold=90.0,
            uncovered_files=["file1.py"],
            report="Test report",
            details={"file1.py": 85.5},
        )
        assert result.total_coverage == 85.5
        assert not result.passed
        assert result.threshold == 90.0
        assert len(result.uncovered_files) == 1

    @patch("subprocess.run")
    def test_run_coverage_success(self, mock_run):
        """Test running coverage successfully."""
        # Mock subprocess call
        mock_run.return_value = Mock(stdout="Coverage: 95%", returncode=0)

        # Mock coverage.json
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            coverage_json = tmpdir_path / "coverage.json"
            coverage_json.write_text(
                json.dumps(
                    {
                        "totals": {
                            "percent_covered": 95.0,
                            "num_statements": 1000,
                            "covered_lines": 950,
                        },
                        "files": {},
                    }
                )
            )

            gate = CoverageGate(threshold=90.0)
            gate.project_root = tmpdir_path

            result = gate.run_coverage(parallel=False, html_report=False)

            assert result.total_coverage == 95.0
            assert result.passed is True

    def test_check_coverage_function(self):
        """Test quick check_coverage function."""
        with patch("ai_dev_agent.testing.coverage_gate.CoverageGate") as mock_gate_class:
            mock_gate = Mock()
            mock_result = CoverageResult(
                total_coverage=95.0,
                passed=True,
                threshold=90.0,
                uncovered_files=[],
                report="",
                details={},
            )
            mock_gate.run_coverage.return_value = mock_result
            mock_gate_class.return_value = mock_gate

            result = check_coverage(threshold=90.0)
            assert result is True

    @patch("subprocess.run")
    def test_get_incremental_coverage(self, mock_run):
        """Test incremental coverage calculation."""
        mock_run.return_value = Mock(stdout="src/new_file.py", returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            coverage_json = tmpdir_path / "coverage.json"
            coverage_json.write_text(
                json.dumps({"files": {"src/new_file.py": {"summary": {"percent_covered": 85.0}}}})
            )

            gate = CoverageGate()
            gate.project_root = tmpdir_path

            result = gate.get_incremental_coverage()

            assert "files" in result or "message" in result


class TestCoverageReporter:
    """Test coverage reporting functionality."""

    def test_reporter_initialization(self):
        """Test reporter initialization."""
        reporter = CoverageReporter()
        assert reporter.project_root == Path.cwd()

    def test_coverage_trend_dataclass(self):
        """Test coverage trend data structure."""
        trend = CoverageTrend(
            timestamp="2025-01-15T10:00:00",
            total_coverage=95.0,
            branch_coverage=90.0,
            files_covered=50,
            total_files=55,
            commit_hash="abc123",
        )
        assert trend.total_coverage == 95.0
        assert trend.commit_hash == "abc123"

    def test_get_coverage_summary_no_data(self):
        """Test getting summary when no coverage data exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = CoverageReporter(project_root=Path(tmpdir))
            summary = reporter.get_coverage_summary()

            assert "error" in summary
            assert summary["total_coverage"] == 0.0

    def test_get_coverage_summary_with_data(self):
        """Test getting summary with coverage data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            coverage_json = tmpdir_path / "coverage.json"
            coverage_json.write_text(
                json.dumps(
                    {
                        "totals": {
                            "percent_covered": 92.5,
                            "percent_covered_display": "92.5",
                            "num_statements": 1000,
                            "covered_lines": 925,
                            "missing_lines": 75,
                        },
                        "files": {"file1.py": {}, "file2.py": {}},
                    }
                )
            )

            reporter = CoverageReporter(project_root=tmpdir_path)
            summary = reporter.get_coverage_summary()

            assert summary["total_coverage"] == 92.5
            assert summary["files"] == 2

    def test_get_file_coverage_categorization(self):
        """Test file coverage categorization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            coverage_json = tmpdir_path / "coverage.json"
            coverage_json.write_text(
                json.dumps(
                    {
                        "files": {
                            "excellent.py": {
                                "summary": {"percent_covered": 98.0, "num_statements": 100}
                            },
                            "good.py": {
                                "summary": {"percent_covered": 85.0, "num_statements": 100}
                            },
                            "fair.py": {
                                "summary": {"percent_covered": 70.0, "num_statements": 100}
                            },
                            "poor.py": {
                                "summary": {"percent_covered": 45.0, "num_statements": 100}
                            },
                        }
                    }
                )
            )

            reporter = CoverageReporter(project_root=tmpdir_path)
            categorized = reporter.get_file_coverage(threshold=95.0)

            assert len(categorized["excellent"]) == 1
            assert len(categorized["good"]) == 1
            assert len(categorized["fair"]) == 1
            assert len(categorized["poor"]) == 1

    def test_save_and_load_coverage_trends(self):
        """Test saving and loading coverage trends."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            devagent_dir = tmpdir_path / ".devagent"
            devagent_dir.mkdir()

            coverage_json = tmpdir_path / "coverage.json"
            coverage_json.write_text(
                json.dumps(
                    {
                        "totals": {"percent_covered": 95.0, "percent_covered_display": "95.0"},
                        "files": {},
                    }
                )
            )

            reporter = CoverageReporter(project_root=tmpdir_path)
            reporter.save_coverage_trend()

            trends = reporter.load_coverage_trends()
            assert len(trends) == 1
            assert trends[0].total_coverage == 95.0

    def test_get_coverage_trend_analysis_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = CoverageReporter(project_root=Path(tmpdir))
            analysis = reporter.get_coverage_trend_analysis()

            assert "message" in analysis
            assert analysis["data_points"] == 0

    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            coverage_json = tmpdir_path / "coverage.json"
            coverage_json.write_text(
                json.dumps(
                    {
                        "totals": {
                            "percent_covered": 96.0,
                            "num_statements": 1000,
                            "covered_lines": 960,
                            "missing_lines": 40,
                        },
                        "files": {},
                    }
                )
            )

            reporter = CoverageReporter(project_root=tmpdir_path)
            markdown = reporter.generate_markdown_report()

            assert "# DevAgent Coverage Report" in markdown
            assert "96.00%" in markdown
            assert "Excellent" in markdown

    def test_generate_badge_data(self):
        """Test badge data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            coverage_json = tmpdir_path / "coverage.json"
            coverage_json.write_text(json.dumps({"totals": {"percent_covered": 97.0}, "files": {}}))

            reporter = CoverageReporter(project_root=tmpdir_path)
            badge = reporter.generate_badge_data()

            assert badge["label"] == "coverage"
            assert "97.0" in badge["message"]
            assert badge["color"] == "brightgreen"


class TestMockLLM:
    """Test MockLLM implementation."""

    def test_mock_llm_initialization(self):
        """Test MockLLM initialization."""
        llm = MockLLM(default_response="Test response")
        assert llm.default_response == "Test response"
        assert llm.call_count == 0

    def test_mock_llm_complete(self):
        """Test MockLLM complete method."""
        llm = MockLLM()
        response = llm.complete("Write a test")

        assert "content" in response
        assert "test" in response["content"].lower()
        assert llm.call_count == 1

    def test_mock_llm_contextual_response(self):
        """Test MockLLM contextual responses."""
        llm = MockLLM()

        test_response = llm.complete("Write a test for feature X")
        assert "test" in test_response["content"].lower()

        implement_response = llm.complete("Implement feature Y")
        assert "implement" in implement_response["content"].lower()

    def test_mock_llm_failure_after_n_calls(self):
        """Test MockLLM failure after N calls."""
        llm = MockLLM(fail_after=2)

        llm.complete("First call")
        llm.complete("Second call")

        with pytest.raises(Exception, match="Mock LLM failure"):
            llm.complete("Third call should fail")

    def test_mock_llm_stream_complete(self):
        """Test MockLLM streaming."""
        llm = MockLLM()
        chunks = list(llm.stream_complete("Test prompt"))

        assert len(chunks) > 0
        assert chunks[-1]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_mock_llm_additional_branches_and_async(self):
        """Test additional prompt branches and async API."""
        llm = MockLLM(default_response="fallback response")

        review = llm.complete("Please review the code")
        plan = llm.complete("Plan the roadmap")
        fix = llm.complete("Fix this regression")
        fallback = llm.complete("Say hello")
        async_result = await llm.aComplete("async test")

        assert "review" in review["content"].lower()
        assert "analyze requirements" in plan["content"].lower()
        assert "fixed the error" in fix["content"].lower()
        assert fallback["content"] == "fallback response"
        assert "def test_example" in async_result["content"]


class TestMockFileSystem:
    """Test MockFileSystem implementation."""

    def test_mock_filesystem_initialization(self):
        """Test filesystem initialization."""
        fs = MockFileSystem()
        assert len(fs.files) == 0
        assert "/" in fs.directories

    def test_mock_filesystem_write_and_read(self):
        """Test writing and reading files."""
        fs = MockFileSystem()
        fs.write("/test.txt", "Hello, World!")

        content = fs.read("/test.txt")
        assert content == "Hello, World!"

    def test_mock_filesystem_read_nonexistent_file(self):
        """Test reading nonexistent file."""
        fs = MockFileSystem()

        with pytest.raises(FileNotFoundError):
            fs.read("/nonexistent.txt")

    def test_mock_filesystem_exists(self):
        """Test file existence check."""
        fs = MockFileSystem()
        fs.write("/test.txt", "content")

        assert fs.exists("/test.txt")
        assert not fs.exists("/nonexistent.txt")

    def test_mock_filesystem_list_dir(self):
        """Test directory listing."""
        fs = MockFileSystem()
        fs.write("/src/main.py", "code")
        fs.write("/src/utils.py", "code")

        items = fs.list_dir("/src")
        assert "main.py" in items
        assert "utils.py" in items

    def test_mock_filesystem_delete(self):
        """Test file deletion."""
        fs = MockFileSystem()
        fs.write("/test.txt", "content")

        fs.delete("/test.txt")
        assert not fs.exists("/test.txt")

    def test_mock_filesystem_list_dir_invalid(self):
        """Test list_dir on invalid path."""
        fs = MockFileSystem()
        with pytest.raises(NotADirectoryError):
            fs.list_dir("/unknown")

    def test_mock_filesystem_delete_missing(self):
        """Test delete raises when file missing."""
        fs = MockFileSystem()
        with pytest.raises(FileNotFoundError):
            fs.delete("/missing.txt")


class TestMockGitRepo:
    """Test MockGitRepo implementation."""

    def test_mock_git_initialization(self):
        """Test git repo initialization."""
        repo = MockGitRepo()
        assert repo.current_branch == "main"
        assert len(repo.commits) == 0

    def test_mock_git_add_and_commit(self):
        """Test adding and committing files."""
        repo = MockGitRepo()
        repo.add(["file1.py", "file2.py"])

        commit_hash = repo.commit("Initial commit")

        assert len(repo.commits) == 1
        assert repo.commits[0]["message"] == "Initial commit"
        assert len(commit_hash) == 7

    def test_mock_git_status(self):
        """Test git status."""
        repo = MockGitRepo()
        repo.add("file.py")

        status = repo.status()

        assert status["branch"] == "main"
        assert "file.py" in status["staged"]

    def test_mock_git_checkout(self):
        """Test branch checkout."""
        repo = MockGitRepo()
        repo.checkout("feature", create=True)

        assert repo.current_branch == "feature"
        assert "feature" in repo.branches

    def test_mock_git_log(self):
        """Test commit log."""
        repo = MockGitRepo()
        repo.add("file.py")
        repo.commit("Commit 1")
        repo.add("file2.py")
        repo.commit("Commit 2")

        log = repo.log(n=10)
        assert len(log) == 2

    def test_mock_git_checkout_invalid_branch(self):
        """Test checkout invalid branch raises."""
        repo = MockGitRepo()
        with pytest.raises(ValueError):
            repo.checkout("unknown-branch")


class TestMockHTTPClient:
    """Test MockHTTPClient implementation."""

    def test_mock_http_client_get_and_post(self):
        client = MockHTTPClient()
        client.set_response("api", {"status": 200, "content": {"ok": True}})

        resp = client.get("https://service/api/resource", headers={"x": "1"})
        assert resp["status"] == 200

        post_resp = client.post("https://service/api/resource", data={"value": 1})
        assert post_resp["status"] == 200
        assert any(entry["method"] == "GET" for entry in client.history)
        assert any(entry["method"] == "POST" for entry in client.history)

    def test_mock_http_client_defaults(self):
        client = MockHTTPClient()
        resp = client.get("https://service/other")
        assert resp["status"] == 404
        assert client.history[-1]["url"] == "https://service/other"


class TestMockDatabase:
    """Test MockDatabase implementation."""

    def test_mock_database_initialization(self):
        """Test database initialization."""
        db = MockDatabase()
        assert len(db.tables) == 0

    def test_mock_database_create_table(self):
        """Test table creation."""
        db = MockDatabase()
        db.create_table("users", {"id": "int", "name": "string"})

        assert "users" in db.tables

    def test_mock_database_insert(self):
        """Test record insertion."""
        db = MockDatabase()
        db.create_table("users", {})

        record_id = db.insert("users", {"name": "Alice", "age": 30})

        assert record_id == 0
        assert db.query_count == 1

    def test_mock_database_select(self):
        """Test record selection."""
        db = MockDatabase()
        db.create_table("users", {})
        db.insert("users", {"name": "Alice", "age": 30})
        db.insert("users", {"name": "Bob", "age": 25})

        results = db.select("users", where={"name": "Alice"})

        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_mock_database_update(self):
        """Test record update."""
        db = MockDatabase()
        db.create_table("users", {})
        record_id = db.insert("users", {"name": "Alice"})

        updated = db.update("users", record_id, {"name": "Alice Smith"})

        assert updated is True
        results = db.select("users", where={"id": record_id})
        assert results[0]["name"] == "Alice Smith"

    def test_mock_database_delete(self):
        """Test record deletion."""
        db = MockDatabase()
        db.create_table("users", {})
        record_id = db.insert("users", {"name": "Alice"})

        deleted = db.delete("users", record_id)

        assert deleted is True
        results = db.select("users")
        assert len(results) == 0

    def test_mock_database_error_paths(self):
        """Test error cases for missing tables."""
        db = MockDatabase()
        with pytest.raises(ValueError):
            db.insert("missing", {"name": "Bob"})
        with pytest.raises(ValueError):
            db.select("missing")
        with pytest.raises(ValueError):
            db.update("missing", 1, {})
        with pytest.raises(ValueError):
            db.delete("missing", 1)

    def test_mock_database_update_and_delete_miss(self):
        """Test update and delete miss return values."""
        db = MockDatabase()
        db.create_table("users", {})
        db.insert("users", {"name": "Alice"})

        updated = db.update("users", 99, {"name": "ghost"})
        deleted = db.delete("users", 99)

        assert updated is False
        assert deleted is False


class TestMockCache:
    """Test MockCache implementation."""

    def test_mock_cache_initialization(self):
        """Test cache initialization."""
        cache = MockCache()
        assert len(cache.data) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_mock_cache_set_and_get(self):
        """Test cache set and get."""
        cache = MockCache()
        cache.set("key1", "value1")

        value = cache.get("key1")

        assert value == "value1"
        assert cache.hits == 1

    def test_mock_cache_get_miss(self):
        """Test cache miss."""
        cache = MockCache()
        value = cache.get("nonexistent")

        assert value is None
        assert cache.misses == 1

    def test_mock_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = MockCache()
        cache.set("key1", "value1")

        cache.get("key1")  # hit
        cache.get("key2")  # miss

        assert cache.hit_rate == 50.0

    def test_mock_cache_hit_rate_zero(self):
        """Test cache hit rate when untouched."""
        cache = MockCache()
        assert cache.hit_rate == 0.0

    def test_mock_cache_delete(self):
        """Test cache deletion."""
        cache = MockCache()
        cache.set("key1", "value1")

        deleted = cache.delete("key1")

        assert deleted is True
        assert cache.get("key1") is None

    def test_mock_cache_clear(self):
        """Test cache clear."""
        cache = MockCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert len(cache.data) == 0


class TestHelperFunctions:
    """Test helper utility functions."""

    def test_run_with_timeout_success(self):
        """Test running function with timeout."""

        def fast_function():
            return "success"

        completed, result = run_with_timeout(fast_function, timeout=1.0)

        assert completed is True
        assert result == "success"

    def test_run_with_timeout_timeout(self):
        """Test function timeout."""
        import time

        def slow_function():
            time.sleep(2)
            return "done"

        completed, _result = run_with_timeout(slow_function, timeout=0.5)

        assert completed is False

    def test_create_test_project(self):
        """Test test project creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            structure = {
                "src": {"main.py": "def hello(): pass", "__init__.py": ""},
                "README.md": "# Test Project",
            }

            create_test_project(tmpdir_path, structure)

            assert (tmpdir_path / "src" / "main.py").exists()
            assert (tmpdir_path / "README.md").exists()

    def test_wait_for_condition_success(self):
        """Test waiting for condition."""
        counter = [0]

        def condition():
            counter[0] += 1
            return counter[0] >= 3

        result = wait_for_condition(condition, timeout=1.0, interval=0.1)

        assert result is True

    def test_wait_for_condition_timeout(self):
        """Test condition timeout."""

        def never_true():
            return False

        result = wait_for_condition(never_true, timeout=0.2, interval=0.1)

        assert result is False

    def test_generate_test_data_strings(self):
        """Test string data generation."""
        data = generate_test_data("strings", count=5, length=10)

        assert len(data) == 5
        assert all(len(s) == 10 for s in data)

    def test_generate_test_data_numbers(self):
        """Test number data generation."""
        data = generate_test_data("numbers", count=5, min=1, max=10)

        assert len(data) == 5
        assert all(1 <= n <= 10 for n in data)

    def test_generate_test_data_dicts(self):
        """Test dict data generation."""
        data = generate_test_data("dicts", count=3, keys=["a", "b"])

        assert len(data) == 3
        assert all("a" in d and "b" in d for d in data)

    def test_compare_json_structures(self):
        """Test JSON structure comparison."""
        json1 = {"a": 1, "b": 2, "timestamp": "now"}
        json2 = {"a": 1, "b": 2, "timestamp": "later"}

        assert compare_json_structures(json1, json2, ignore_keys=["timestamp"])
        assert not compare_json_structures(json1, json2)

    def test_create_mock_response(self):
        """Test mock HTTP response creation."""
        response = create_mock_response(status_code=200, content={"data": "value"})

        assert response.status_code == 200
        assert response.ok is True
        assert response.json() == {"data": "value"}

    def test_validate_schema_success(self):
        """Test schema validation success."""
        data = {"name": "Alice", "age": 30}
        schema = {
            "name": {"type": "string", "required": True},
            "age": {"type": "number", "required": True},
        }

        is_valid, errors = validate_schema(data, schema)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_schema_failure(self):
        """Test schema validation failure."""
        data = {"name": 123}  # name should be string
        schema = {
            "name": {"type": "string", "required": True},
            "age": {"type": "number", "required": True},
        }

        is_valid, errors = validate_schema(data, schema)

        assert is_valid is False
        assert len(errors) > 0

    def test_temporary_env(self):
        """Test temporary environment variables."""
        import os

        original = os.environ.get("TEST_VAR")

        with temporary_env(TEST_VAR="test_value"):
            assert os.environ["TEST_VAR"] == "test_value"

        # Should be restored
        current = os.environ.get("TEST_VAR")
        assert current == original


class TestMockFactoryFunctions:
    """Test mock factory functions."""

    def test_create_mock_agent(self):
        """Test agent mock creation."""
        agent = create_mock_agent("TestAgent", "test")

        assert agent.name == "TestAgent"
        assert agent.type == "test"
        assert agent.status == "ready"

        result = agent.execute({"task": "test"})
        assert result["status"] == "success"

    def test_create_mock_tool(self):
        """Test tool mock creation."""
        tool = create_mock_tool("TestTool", "Test tool description")

        assert tool.name == "TestTool"
        assert tool.description == "Test tool description"

        result = tool.execute()
        assert "Tool TestTool executed" in result["output"]
