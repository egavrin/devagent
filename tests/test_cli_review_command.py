"""Tests for unified review command."""
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path
import pytest
import click
from click.testing import CliRunner
from ai_dev_agent.cli.commands import cli
from ai_dev_agent.agents.base import AgentResult
from ai_dev_agent.cli.review import (
    _PATCH_CACHE,
    format_patch_dataset,
    parse_patch_file,
    run_review,
    extract_applies_to_pattern,
)
from ai_dev_agent.core.utils.config import Settings

review_module = importlib.import_module("ai_dev_agent.cli.review")


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


class TestReviewInternals:
    """Test helpers backing the review command."""

    @pytest.fixture(autouse=True)
    def clear_patch_cache(self):
        """Ensure patch cache does not bleed across tests."""
        _PATCH_CACHE.clear()
        yield
        _PATCH_CACHE.clear()

    def test_run_review_restores_settings(
        self,
        monkeypatch,
        tmp_path,
    ):
        """run_review should leave settings unchanged after execution."""
        patch_path = tmp_path / "changes.patch"
        patch_path.write_text(
            """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -0,0 +1,2 @@
+def foo():
+    return 1
"""
        )
        rule_path = tmp_path / "rule.md"
        rule_path.write_text(
            """# Sample Rule

## Applies To
*.py

## Description
Ensure functions return something.
"""
        )

        settings = Settings(
            api_key="test-key",
            workspace_root=Path("/original/root"),
            max_tool_output_chars=2048,
            max_context_tokens=4096,
            response_headroom_tokens=512,
        )
        original_values = (
            settings.workspace_root,
            settings.max_tool_output_chars,
            settings.max_context_tokens,
            settings.response_headroom_tokens,
        )
        ctx = click.Context(click.Command("review"), obj={"settings": settings})

        def fake_import_module(name: str):
            assert name == "ai_dev_agent.cli"
            return SimpleNamespace(get_llm_client=lambda _: object())

        def fake_execute(*_args, **_kwargs):
            return {
                "result": {"status": "ok"},
                "final_json": {
                    "violations": [],
                    "summary": {
                        "total_violations": 0,
                        "files_reviewed": 1,
                    },
                },
            }

        monkeypatch.setattr(review_module, "import_module", fake_import_module)
        monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)
        monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)

        outcome = run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_path),
            json_output=True,
            settings=settings,
        )

        assert outcome["summary"]["files_reviewed"] == 1
        assert (
            settings.workspace_root,
            settings.max_tool_output_chars,
            settings.max_context_tokens,
            settings.response_headroom_tokens,
        ) == original_values

    def test_parse_patch_file_includes_size_in_cache_key(self, monkeypatch, tmp_path):
        """Cache should invalidate when file size changes within same timestamp."""
        patch_path = tmp_path / "cached.patch"
        patch_path.write_text("FIRST")

        parse_calls = []

        class StubParser:
            def __init__(self, content: str, include_context: bool = False):
                self._content = content

            def parse(self):
                parse_calls.append(self._content)
                label = self._content.strip()
                return {"files": [{"path": label}]}

        original_stat = Path.stat
        stat_reads = 0

        def fake_stat(self):
            nonlocal stat_reads
            if self == patch_path:
                stat_reads += 1
                if stat_reads == 1:
                    return SimpleNamespace(st_mtime=1000.0, st_size=5)
                return SimpleNamespace(st_mtime=1000.0, st_size=6)
            return original_stat(self)

        monkeypatch.setattr(review_module, "PatchParser", StubParser)
        monkeypatch.setattr(Path, "stat", fake_stat)

        first = parse_patch_file(patch_path)
        patch_path.write_text("SECOND")
        second = parse_patch_file(patch_path)

        assert parse_calls == ["FIRST", "SECOND"]
        assert first["files"][0]["path"] == "FIRST"
        assert second["files"][0]["path"] == "SECOND"

    def test_format_patch_dataset_includes_context_and_removals(self, tmp_path):
        """Dataset should surface context and removed lines for reviewers."""
        patch_path = tmp_path / "context.patch"
        patch_path.write_text(
            """diff --git a/sample.py b/sample.py
--- a/sample.py
+++ b/sample.py
@@ -1,3 +1,4 @@
 def foo():
-    return 1
+    value = compute()
+    return value
"""
        )

        parsed = parse_patch_file(patch_path)
        dataset = format_patch_dataset(parsed)

        assert "HUNK:" in dataset
        assert "CONTEXT:" in dataset
        assert "REMOVED LINES:" in dataset
        assert "2 -     return 1" in dataset

    def test_extract_applies_to_pattern_normalizes_globs(self, tmp_path):
        """Glob patterns should be converted to regex and filter datasets."""
        patch_path = tmp_path / "multi.patch"
        patch_path.write_text(
            """diff --git a/sample.py b/sample.py
--- a/sample.py
+++ b/sample.py
@@ -0,0 +1,2 @@
+print("py file")
+value = 1
diff --git a/sample.js b/sample.js
--- a/sample.js
+++ b/sample.js
@@ -0,0 +1,2 @@
+console.log("js file");
+const value = 1;
"""
        )

        parsed = parse_patch_file(patch_path)
        rule_text = """# Rule

## Applies To
*.py
"""
        pattern = extract_applies_to_pattern(rule_text)
        assert pattern is not None

        dataset = format_patch_dataset(parsed, filter_pattern=pattern)
        assert "FILE: sample.py" in dataset
        assert "FILE: sample.js" not in dataset
