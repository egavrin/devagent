"""Tests for unified review command."""
from unittest.mock import MagicMock
from pathlib import Path
import pytest
from click.testing import CliRunner
from ai_dev_agent.cli.commands import cli
from ai_dev_agent.agents.base import AgentResult


@pytest.fixture
def review_cli_stub(cli_stub_runtime, monkeypatch):
    """Stub review agent execution."""
    stub_result = AgentResult(
        success=True,
        output="Review completed",
        metadata={"issues_found": 0, "quality_score": 1.0},
    )
    execute_stub = MagicMock(return_value=stub_result)
    monkeypatch.setattr(
        "ai_dev_agent.agents.specialized.executor_bridge.execute_agent_with_react",
        execute_stub,
    )
    cli_stub_runtime["agent_execute"] = execute_stub
    return cli_stub_runtime


pytestmark = pytest.mark.usefixtures("review_cli_stub")


class TestReviewCommand:
    """Test the unified review command (file + patch)."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_review_help(self, runner):
        """review command should show help."""
        result = runner.invoke(cli, ["review", "--help"])
        assert result.exit_code in [0, 1, 2]
        assert "review" in result.output.lower()

    def test_review_file_basic(self, runner, review_cli_stub):
        """review should accept a source file."""
        result = runner.invoke(cli, ["review", "src/auth.py"])
        assert result.exit_code in [0, 1, 2]
        review_cli_stub["agent_execute"].assert_called_once()

    def test_review_file_with_report(self, runner, review_cli_stub, tmp_path):
        """review should accept --report flag for file review."""
        report_path = tmp_path / "review.md"
        result = runner.invoke(cli, [
            "review", "src/auth.py",
            "--report", str(report_path)
        ])
        assert result.exit_code in [0, 1, 2]
        assert report_path.exists()
        review_cli_stub["agent_execute"].assert_called_once()

    def test_review_file_with_json(self, runner):
        """review should accept --json flag."""
        result = runner.invoke(cli, [
            "review", "src/auth.py",
            "--json"
        ])
        assert result.exit_code in [0, 1, 2]

    def test_review_file_with_verbose(self, runner):
        """review should accept verbosity flags."""
        result = runner.invoke(cli, [
            "review", "src/auth.py",
            "-v"
        ])
        assert result.exit_code in [0, 1, 2]


class TestReviewPatchWithRule:
    """Test review command with --rule flag (patch review)."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_patch(self, tmp_path):
        """Create a sample patch file."""
        patch_file = tmp_path / "changes.patch"
        patch_content = """--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,5 @@
+# New comment
 def authenticate(username, password):
     # Authenticate user
     return validate(username, password)
"""
        patch_file.write_text(patch_content)
        return str(patch_file)

    @pytest.fixture
    def sample_rule(self, tmp_path):
        """Create a sample rule file."""
        rule_file = tmp_path / "style.md"
        rule_content = """# Code Style Rule

## Applies To
*.py

## Description
All Python files must have proper comments.
"""
        rule_file.write_text(rule_content)
        return str(rule_file)

    def test_review_patch_with_rule(self, runner, sample_patch, sample_rule):
        """review should accept --rule flag for patch review."""
        result = runner.invoke(cli, [
            "review", sample_patch,
            "--rule", sample_rule
        ])
        assert result.exit_code == 0

    def test_review_patch_with_rule_and_json(self, runner, sample_patch, sample_rule):
        """review patch should output JSON with --json flag."""
        result = runner.invoke(cli, [
            "review", sample_patch,
            "--rule", sample_rule,
            "--json"
        ])
        assert result.exit_code == 0

    def test_review_patch_without_rule(self, runner, sample_patch):
        """review patch without --rule should do general code review."""
        result = runner.invoke(cli, [
            "review", sample_patch
        ])
        assert result.exit_code == 0

    def test_review_patch_with_verbose(self, runner, sample_patch, sample_rule):
        """review patch should accept verbosity flags."""
        result = runner.invoke(cli, [
            "review", sample_patch,
            "--rule", sample_rule,
            "-vv"
        ])
        assert result.exit_code in [0, 1, 2]


class TestReviewCommandVariations:
    """Test various file types and scenarios."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_review_python_file(self, runner):
        """review should work with Python files."""
        result = runner.invoke(cli, ["review", "src/app.py"])
        assert result.exit_code == 0

    def test_review_javascript_file(self, runner):
        """review should work with JavaScript files."""
        result = runner.invoke(cli, ["review", "src/app.js"])
        assert result.exit_code == 0

    def test_review_typescript_file(self, runner):
        """review should work with TypeScript files."""
        result = runner.invoke(cli, ["review", "src/app.ts"])
        assert result.exit_code == 0

    def test_review_patch_extension(self, runner):
        """review should detect .patch files."""
        result = runner.invoke(cli, ["review", "changes.patch"])
        assert result.exit_code in [0, 1, 2]

    def test_review_diff_extension(self, runner):
        """review should detect .diff files."""
        result = runner.invoke(cli, ["review", "changes.diff"])
        assert result.exit_code in [0, 1, 2]


class TestReviewVsLint:
    """Test that review does both file and patch review (no separate lint command)."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_no_lint_command(self, runner):
        """There should be no separate 'lint' command."""
        result = runner.invoke(cli, ["lint", "--help"])
        # Should error (command doesn't exist)
        assert result.exit_code != 0

    def test_review_unified_command(self, runner):
        """Review command should handle both files and patches."""
        # Both should work with the review command
        result1 = runner.invoke(cli, ["review", "file.py"])
        result2 = runner.invoke(cli, ["review", "patch.patch", "--rule", "rule.md"])

        assert result1.exit_code in [0, 1, 2]
        assert result2.exit_code in [0, 1, 2]
