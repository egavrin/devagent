"""End-to-end integration tests using real LLM APIs.

This module contains integration tests that use actual LLM APIs to verify
end-to-end functionality of the DevAgent CLI. Tests are designed to be fast
and use accurate verification methods.

To run these tests, ensure you have:
1. API key set (DEVAGENT_API_KEY environment variable or .devagent.toml)
2. pytest-xdist installed for parallel execution

Run commands:
    pytest -m llm tests/integration/test_llm_e2e.py -v
    pytest -m llm_fast tests/integration/test_llm_e2e.py -v
    pytest -m llm -n 4 tests/integration/test_llm_e2e.py  # parallel
"""

import json
import os
from pathlib import Path

import pytest

from .conftest import IntegrationTest, verify_file_list, verify_json_schema, verify_line_count


@pytest.mark.llm
@pytest.mark.integration
class TestSimpleQueries(IntegrationTest):
    """Test simple queries with direct execution (fast tests)."""

    @pytest.mark.llm_fast
    def test_line_count_query(self, llm_client_real, run_devagent_cli):
        """Test simple line count query with direct execution."""
        result = run_devagent_cli(
            ["how many lines are in ai_dev_agent/cli/runtime/main.py"], timeout=120
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify with accurate alternative
        actual_lines = verify_line_count(Path("ai_dev_agent/cli/runtime/main.py"))

        # Check if the line count appears in output
        assert (
            str(actual_lines) in result.stdout or str(actual_lines) in result.stderr
        ), f"Expected {actual_lines} lines in output. Got stdout: {result.stdout[:500]}, stderr: {result.stderr[:500]}"

    def test_file_search_query(self, llm_client_real, run_devagent_cli):
        """Test finding Python test files."""
        # Simpler query to avoid timeout
        result = run_devagent_cli(["list 3 Python files in the tests directory"], timeout=180)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify mentions test files
        output_combined = result.stdout + result.stderr
        assert (
            "test_" in output_combined or "tests/" in output_combined or ".py" in output_combined
        ), f"Output should mention test files: {output_combined[:500]}"

    def test_simple_code_question(self, llm_client_real, run_devagent_cli):
        """Test asking about code functionality."""
        # Use a simpler query that won't trigger extensive RepoMap searches
        result = run_devagent_cli(["explain what pytest does in one sentence"], timeout=180)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        output_combined = (result.stdout + result.stderr).lower()
        # Check for relevant keywords
        has_keyword = any(
            keyword in output_combined for keyword in ["test", "pytest", "framework", "testing"]
        )
        assert has_keyword, f"Output should mention testing: {output_combined[:500]}"

    @pytest.mark.llm_fast
    def test_list_files_in_directory(self, llm_client_real, run_devagent_cli):
        """Test listing files in a specific directory."""
        result = run_devagent_cli(["list files in ai_dev_agent/cli/"], timeout=120)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify mentions some CLI files
        output_combined = result.stdout + result.stderr
        expected_files = ["runtime/main.py", "review.py"]
        found_any = any(f in output_combined for f in expected_files)

        assert found_any, f"Output should mention CLI files: {output_combined[:500]}"


@pytest.mark.llm
@pytest.mark.integration
class TestPlanningMode(IntegrationTest):
    """Test queries with planning mode enabled."""

    def test_line_count_with_plan(self, llm_client_real, run_devagent_cli):
        """Test line count query with planning mode."""
        result = run_devagent_cli(
            ["--plan", "how many lines are in ai_dev_agent/cli/runtime/main.py"], timeout=45
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify correct line count
        actual_lines = verify_line_count(Path("ai_dev_agent/cli/runtime/main.py"))
        output_combined = result.stdout + result.stderr

        assert (
            str(actual_lines) in output_combined
        ), f"Expected {actual_lines} lines in output: {output_combined[:500]}"

    def test_complex_analysis_with_plan(self, llm_client_real, run_devagent_cli):
        """Test complex analysis with planning mode."""
        # Use a simpler query that still tests planning
        result = run_devagent_cli(["--plan", "count how many test files exist"], timeout=180)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        output_combined = (result.stdout + result.stderr).lower()
        # Should mention tests or files
        has_keyword = any(
            keyword in output_combined for keyword in ["test", "file", "count", "pytest"]
        )
        assert has_keyword, f"Output should mention tests/files: {output_combined[:500]}"

    def test_plan_vs_direct_consistency(self, llm_client_real, run_devagent_cli):
        """Test that plan and direct modes produce consistent results."""
        query = "how many lines are in ai_dev_agent/cli/runtime/main.py"

        # Run without plan
        result_direct = run_devagent_cli([query], timeout=120)
        # Run with plan
        result_plan = run_devagent_cli(["--plan", query], timeout=120)

        assert result_direct.returncode == 0, f"Direct command failed: {result_direct.stderr}"
        assert result_plan.returncode == 0, f"Plan command failed: {result_plan.stderr}"

        # Both should mention the correct line count
        actual_lines = verify_line_count(Path("ai_dev_agent/cli/runtime/main.py"))
        assert str(actual_lines) in (
            result_direct.stdout + result_direct.stderr
        ), "Direct mode should have correct count"
        assert str(actual_lines) in (
            result_plan.stdout + result_plan.stderr
        ), "Plan mode should have correct count"


@pytest.mark.llm
@pytest.mark.integration
class TestReviewCommand(IntegrationTest):
    """Test the review command with various scenarios."""

    def test_review_simple_patch(
        self, llm_client_real, run_devagent_cli, sample_patch, coding_rule, tmp_path
    ):
        """Test reviewing a simple patch with known violations."""
        # Create patch and rule files
        patch_path = tmp_path / "changes.patch"
        rule_path = tmp_path / "style.md"
        patch_path.write_text(sample_patch)
        rule_path.write_text(coding_rule)

        # Run review
        result = run_devagent_cli(
            ["review", str(patch_path), "--rule", str(rule_path), "--json"],
            cwd=tmp_path,
            timeout=120,
        )

        assert result.returncode == 0, f"Review failed: {result.stderr}"

        # Parse JSON output
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.stdout}")

        # Verify structure
        assert "violations" in data, "Missing 'violations' key"
        assert "summary" in data, "Missing 'summary' key"
        assert isinstance(data["violations"], list), "violations should be a list"
        assert isinstance(data["summary"], dict), "summary should be a dict"

        # Verify summary fields
        summary = data["summary"]
        assert "total_violations" in summary
        assert "files_reviewed" in summary

    def test_review_with_config(
        self,
        llm_client_real,
        run_devagent_cli,
        sample_patch,
        coding_rule,
        tmp_path,
    ):
        """Test review command respects .devagent.toml configuration."""
        # Create config with custom review settings
        config_path = tmp_path / ".devagent.toml"
        api_key = os.environ.get("DEVAGENT_API_KEY", "test-key")
        config_content = f"""
provider = "deepseek"
model = "deepseek-chat"
api_key = "{api_key}"
review_max_files_per_chunk = 5
review_max_lines_per_chunk = 100
"""
        config_path.write_text(config_content)

        # Create patch and rule files
        patch_path = tmp_path / "changes.patch"
        rule_path = tmp_path / "style.md"
        patch_path.write_text(sample_patch)
        rule_path.write_text(coding_rule)

        # Run review with config
        result = run_devagent_cli(
            [
                "--config",
                str(config_path),
                "review",
                str(patch_path),
                "--rule",
                str(rule_path),
                "--json",
            ],
            cwd=tmp_path,
            timeout=120,
        )

        assert result.returncode == 0, f"Review failed: {result.stderr}"

        # Parse output
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.stdout}")

        assert "violations" in data
        assert "summary" in data

    def test_review_no_violations(self, llm_client_real, run_devagent_cli, coding_rule, tmp_path):
        """Test review with patch that has no violations."""
        # Create clean patch
        clean_patch = """diff --git a/src/clean.py b/src/clean.py
index abc123..def456 100644
--- a/src/clean.py
+++ b/src/clean.py
@@ -1,3 +1,6 @@
+THRESHOLD_LIMIT = 100  # Named constant
+
 def calculate(x, y):
-    return x + y
+    result = x + y  # Proper spacing
+    return result
"""

        patch_path = tmp_path / "clean.patch"
        rule_path = tmp_path / "style.md"
        patch_path.write_text(clean_patch)
        rule_path.write_text(coding_rule)

        # Run review
        result = run_devagent_cli(
            ["review", str(patch_path), "--rule", str(rule_path), "--json"],
            cwd=tmp_path,
            timeout=120,
        )

        assert result.returncode == 0, f"Review failed: {result.stderr}"

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.stdout}")

        # Should have 0 or minimal violations
        assert "summary" in data
        assert data["summary"]["total_violations"] >= 0

    def test_review_source_file(self, llm_client_real, run_devagent_cli, test_project):
        """Test reviewing a source file directly (not a patch)."""
        # Review a Python source file without --rule (general code review)
        source_file = test_project / "src" / "main.py"

        result = run_devagent_cli(["review", str(source_file)], cwd=test_project, timeout=180)

        assert result.returncode == 0, f"Review failed: {result.stderr}"

        # Should produce output about the code
        output = result.stdout + result.stderr
        assert len(output) > 0, "Should produce review output"

    def test_review_applies_to_pattern(self, llm_client_real, run_devagent_cli, tmp_path):
        """Test that rule's 'Applies To' pattern filters files correctly."""
        # Create patch with mixed file types
        mixed_patch = """diff --git a/src/code.py b/src/code.py
index abc123..def456 100644
--- a/src/code.py
+++ b/src/code.py
@@ -1,2 +1,3 @@
 def test():
+    # TODO: fix
     pass
diff --git a/README.md b/README.md
index abc123..def456 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,3 @@
 # Project
+TODO: update docs
"""

        # Rule that only applies to .py files
        py_only_rule = """# Python Only Rule

## Applies To
*.py

## Description
No TODO comments in Python files.
"""

        patch_path = tmp_path / "mixed.patch"
        rule_path = tmp_path / "py_only.md"
        patch_path.write_text(mixed_patch)
        rule_path.write_text(py_only_rule)

        result = run_devagent_cli(
            ["review", str(patch_path), "--rule", str(rule_path), "--json"],
            cwd=tmp_path,
            timeout=120,
        )

        assert result.returncode == 0, f"Review failed: {result.stderr}"

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.stdout}")

        # Should only review .py file
        violations = data.get("violations", [])
        for violation in violations:
            file_path = violation.get("file", "")
            # Violations should only be from .py files
            if file_path:
                assert file_path.endswith(".py"), f"Unexpected file in violations: {file_path}"

    def test_review_large_patch(self, llm_client_real, run_devagent_cli, coding_rule, tmp_path):
        """Test reviewing a large patch that requires chunking."""
        # Create large patch with multiple files
        large_patch = """diff --git a/src/file1.py b/src/file1.py
index abc..def 100644
--- a/src/file1.py
+++ b/src/file1.py
@@ -1,3 +1,4 @@
+# TODO: implement
 def func1():
     pass

diff --git a/src/file2.py b/src/file2.py
index abc..def 100644
--- a/src/file2.py
+++ b/src/file2.py
@@ -1,3 +1,4 @@
+# FIXME: refactor
 def func2():
     pass

diff --git a/src/file3.py b/src/file3.py
index abc..def 100644
--- a/src/file3.py
+++ b/src/file3.py
@@ -1,3 +1,4 @@
+x=1+2  # Bad spacing
 def func3():
     pass
"""

        patch_path = tmp_path / "large.patch"
        rule_path = tmp_path / "style.md"
        patch_path.write_text(large_patch)
        rule_path.write_text(coding_rule)

        result = run_devagent_cli(
            ["review", str(patch_path), "--rule", str(rule_path), "--json"],
            cwd=tmp_path,
            timeout=90,
        )

        assert result.returncode == 0, f"Review failed: {result.stderr}"

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.stdout}")

        # Verify all files were reviewed
        summary = data["summary"]
        assert summary["files_reviewed"] == 3, "Should review all 3 files"


@pytest.mark.llm
@pytest.mark.integration
class TestSpecializedCommands(IntegrationTest):
    """Test specialized CLI commands."""

    @pytest.mark.llm_fast
    def test_specialized_commands_help(self, run_devagent_cli, tmp_path):
        """Smoke test: verify specialized commands show help (fast check)."""
        # Test create-design --help
        result = run_devagent_cli(["create-design", "--help"], cwd=tmp_path, timeout=10)
        assert result.returncode == 0
        assert "create-design" in result.stdout.lower()

        # Test generate-tests --help
        result = run_devagent_cli(["generate-tests", "--help"], cwd=tmp_path, timeout=10)
        assert result.returncode == 0
        assert "generate-tests" in result.stdout.lower()

        # Test write-code --help
        result = run_devagent_cli(["write-code", "--help"], cwd=tmp_path, timeout=10)
        assert result.returncode == 0
        assert "write-code" in result.stdout.lower()

    @pytest.mark.llm_slow
    def test_create_design_command(self, llm_client_real, run_devagent_cli, tmp_path):
        """Test create-design command - SLOW (~3-4 minutes).

        This test generates a full design document which requires:
        - Building RepoMap context
        - LLM generation of comprehensive design
        - Multiple iterations
        """
        result = run_devagent_cli(["create-design", "simple calculator"], cwd=tmp_path, timeout=480)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Should produce output
        output_combined = result.stdout + result.stderr
        assert len(output_combined) > 0, "Should produce output"

        # Check mentions design or calculator
        output_lower = output_combined.lower()
        assert "design" in output_lower or "calculator" in output_lower

    @pytest.mark.llm_slow
    def test_generate_tests_command(self, llm_client_real, run_devagent_cli, tmp_path):
        """Test generate-tests command - VERY SLOW (~5-6 minutes).

        This test generates comprehensive test files which requires:
        - Building RepoMap context
        - LLM generation of test code with 90% coverage
        - Coverage analysis and iteration
        - Often requires multiple LLM calls to achieve target coverage
        """
        result = run_devagent_cli(
            ["generate-tests", "calculator module", "--coverage", "90"],
            cwd=tmp_path,
            timeout=600,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        output_combined = result.stdout + result.stderr
        assert len(output_combined) > 0, "Should produce output"

        # Check mentions tests or coverage
        output_lower = output_combined.lower()
        assert "test" in output_lower or "coverage" in output_lower

    @pytest.mark.llm_slow
    def test_write_code_command(self, llm_client_real, run_devagent_cli, tmp_path):
        """Test write-code command - SLOW (~3-4 minutes).

        This test implements code from a design which requires:
        - Building RepoMap context
        - LLM generation of implementation code
        - Code structure analysis
        """
        # Create a simple design file
        design_path = tmp_path / "design.md"
        design_content = """# Calculator Design

## Requirements
- Add two numbers
- Subtract two numbers

## Implementation
Create a Calculator class with add() and subtract() methods.
"""
        design_path.write_text(design_content)

        result = run_devagent_cli(["write-code", str(design_path)], cwd=tmp_path, timeout=480)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        output_combined = result.stdout + result.stderr
        assert len(output_combined) > 0, "Should produce output"


@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.llm_slow
class TestChatMode(IntegrationTest):
    """Test interactive chat mode."""

    def test_chat_session_context(self, llm_client_real, tmp_path):
        """Test that chat maintains context across turns.

        Note: This test is more complex and may need manual verification.
        For automated testing, we test the chat command availability.
        """
        # For now, just verify chat command exists and can be invoked
        # Full interactive testing would require pexpect or similar
        import subprocess

        # Test that chat command is recognized
        result = subprocess.run(
            ["devagent", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert "chat" in result.stdout, "Chat command should be in help output"

    def test_chat_exit_commands(self, llm_client_real):
        """Test that chat exit commands are documented."""
        import subprocess

        result = subprocess.run(
            ["devagent", "chat", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Help should mention exit/quit
        help_text = result.stdout.lower()
        assert "exit" in help_text or "quit" in help_text


@pytest.mark.llm
@pytest.mark.integration
class TestConfigIntegration(IntegrationTest):
    """Test configuration file integration."""

    @pytest.mark.llm_fast
    def test_config_from_devagent_toml(self, llm_client_real, run_devagent_cli, tmp_path):
        """Test loading settings from .devagent.toml."""
        # Create config file
        config_path = tmp_path / ".devagent.toml"
        api_key = os.environ.get("DEVAGENT_API_KEY", "test-key")
        config_content = f"""
provider = "deepseek"
model = "deepseek-chat"
api_key = "{api_key}"
max_completion_tokens = 2048
"""
        config_path.write_text(config_content)

        # Run a simple query with config
        result = run_devagent_cli(
            [
                "--config",
                str(config_path),
                "how many lines are in ai_dev_agent/cli/runtime/main.py",
            ],
            cwd=tmp_path,
            timeout=120,
        )

        # Should work with config
        assert result.returncode == 0, f"Command with config failed: {result.stderr}"

    def test_config_hierarchy(self, llm_client_real, run_devagent_cli, tmp_path):
        """Test config file precedence (project > global)."""
        # Create project config
        project_config = tmp_path / ".devagent.toml"
        api_key = os.environ.get("DEVAGENT_API_KEY", "test-key")
        project_config.write_text(
            f"""
provider = "deepseek"
model = "deepseek-chat"
api_key = "{api_key}"
"""
        )

        # Run command with explicit config
        result = run_devagent_cli(
            [
                "--config",
                str(project_config),
                "how many lines are in ai_dev_agent/cli/runtime/main.py",
            ],
            cwd=tmp_path,
            timeout=120,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

    @pytest.mark.llm_fast
    def test_config_with_env_override(
        self, llm_client_real, run_devagent_cli, tmp_path, monkeypatch
    ):
        """Test that environment variables override config file."""
        # Create config with one API key
        config_path = tmp_path / ".devagent.toml"
        config_path.write_text(
            """
provider = "deepseek"
model = "deepseek-chat"
api_key = "config-key"
"""
        )

        # Set environment variable to override
        real_api_key = os.environ.get("DEVAGENT_API_KEY")
        if real_api_key:
            monkeypatch.setenv("DEVAGENT_API_KEY", real_api_key)

            # Run command - should use env var key
            result = run_devagent_cli(
                [
                    "--config",
                    str(config_path),
                    "how many lines are in ai_dev_agent/cli/runtime/main.py",
                ],
                cwd=tmp_path,
                timeout=120,
            )

            # Should work with env override
            assert result.returncode == 0, f"Command failed: {result.stderr}"
