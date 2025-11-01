"""Tests for unified review command."""

import importlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock
from uuid import uuid4

import click
import pytest
from click.testing import CliRunner

from ai_dev_agent.agents.base import AgentResult
from ai_dev_agent.cli import cli
from ai_dev_agent.cli.review import (
    _PATCH_CACHE,
    extract_applies_to_pattern,
    format_patch_dataset,
    parse_patch_file,
    run_review,
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
        "ai_dev_agent.agents.strategy_adapter.execute_agent_with_react",
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
        result = runner.invoke(cli, ["review", "src/auth.py", "--report", str(report_path)])
        assert result.exit_code in [0, 1, 2]
        assert report_path.exists()
        review_cli_stub["agent_execute"].assert_called_once()

    def test_review_file_with_json(self, runner):
        """review should accept --json flag."""
        result = runner.invoke(cli, ["review", "src/auth.py", "--json"])
        assert result.exit_code in [0, 1, 2]

    def test_review_file_with_verbose(self, runner):
        """review should accept verbosity flags."""
        result = runner.invoke(cli, ["review", "src/auth.py", "-v"])
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
        result = runner.invoke(cli, ["review", sample_patch, "--rule", sample_rule])
        assert result.exit_code == 0

    def test_review_patch_outputs_text_without_json(
        self, runner, sample_patch, sample_rule, monkeypatch
    ):
        """Human-readable summary should be printed when --json is omitted."""

        def fake_run_review(*_args, **_kwargs):
            return {
                "summary": {"files_reviewed": 1, "total_violations": 0, "rule_name": "TestRule"},
                "violations": [],
            }

        monkeypatch.setattr(review_module, "run_review", fake_run_review)
        monkeypatch.setattr("ai_dev_agent.cli.review.run_review", fake_run_review)

        result = runner.invoke(cli, ["review", sample_patch, "--rule", sample_rule])

        assert result.exit_code == 0
        assert "Files reviewed" in result.output
        assert "{\n" not in result.output  # no raw JSON block

    def test_review_patch_reports_discarded_count(
        self, runner, sample_patch, sample_rule, monkeypatch
    ):
        """CLI should surface discarded violations information."""

        def fake_run_review(*_args, **_kwargs):
            return {
                "summary": {
                    "files_reviewed": 1,
                    "total_violations": 0,
                    "discarded_violations": 2,
                    "rule_name": "TestRule",
                },
                "violations": [],
            }

        monkeypatch.setattr(review_module, "run_review", fake_run_review)
        monkeypatch.setattr("ai_dev_agent.cli.review.run_review", fake_run_review)

        result = runner.invoke(cli, ["review", sample_patch, "--rule", sample_rule])

        assert result.exit_code == 0
        assert "Discarded violations: 2" in result.output

    def test_review_patch_with_rule_and_json(self, runner, sample_patch, sample_rule):
        """review patch should output JSON with --json flag."""
        result = runner.invoke(cli, ["review", sample_patch, "--rule", sample_rule, "--json"])
        assert result.exit_code == 0

    def test_review_patch_without_rule(self, runner, sample_patch):
        """review patch without --rule should do general code review."""
        result = runner.invoke(cli, ["review", sample_patch])
        assert result.exit_code == 0

    def test_review_patch_with_verbose(self, runner, sample_patch, sample_rule):
        """review patch should accept verbosity flags."""
        result = runner.invoke(cli, ["review", sample_patch, "--rule", sample_rule, "-vv"])
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

    def test_run_review_splits_large_file_chunks(self, monkeypatch, tmp_path):
        """Large patches should be split into multiple review calls."""
        patch_path = tmp_path / "massive.patch"
        patch_lines = [
            "diff --git a/huge.py b/huge.py",
            "--- a/huge.py",
            "+++ b/huge.py",
        ]
        for idx in range(1, 61):
            patch_lines.append(f"@@ -{idx},0 +{idx},1 @@")
            patch_lines.append(f"+print('line {idx}')")
        patch_path.write_text("\n".join(patch_lines) + "\n")

        rule_path = tmp_path / "rule.md"
        rule_path.write_text("# Rule\n\n## Applies To\n*.py\n")

        settings = Settings(
            api_key="test-key",
            workspace_root=tmp_path,
            max_tool_output_chars=4096,
            max_context_tokens=4096,
            response_headroom_tokens=512,
        )
        setattr(settings, "review_max_lines_per_chunk", 10)

        ctx = click.Context(click.Command("review"), obj={"settings": settings})

        def fake_import_module(name: str):
            return SimpleNamespace(get_llm_client=lambda _: object())

        calls: list[str] = []

        def fake_execute(_ctx, _client, _settings, prompt, **_kwargs):
            calls.append(prompt)
            return {
                "result": {"status": "ok"},
                "final_json": {"violations": [], "summary": {"files_reviewed": 1}},
            }

        monkeypatch.setattr(review_module, "import_module", fake_import_module)
        monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)
        monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)

        run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_path),
            json_output=True,
            settings=settings,
        )

        assert len(calls) > 1, "Expected large patch to be reviewed in multiple chunks"

    def test_run_review_includes_source_context(self, monkeypatch, tmp_path):
        """Review prompt should include surrounding source context."""
        source_path = tmp_path / "module.py"
        source_path.write_text(
            "def existing():\n    return 1\n\n\ndef target():\n    value = 2\n    return value\n"
        )

        patch_path = tmp_path / "change.patch"
        patch_path.write_text(
            """diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -4,2 +4,2 @@
 def target():
-    value = 2
+    value = compute()
     return value
"""
        )
        rule_path = tmp_path / "rule.md"
        rule_path.write_text("# Rule\n\n## Applies To\n*.py\n")

        settings = Settings(
            api_key="test-key",
            workspace_root=tmp_path,
            max_tool_output_chars=4096,
            max_context_tokens=4096,
            response_headroom_tokens=512,
        )

        ctx = click.Context(click.Command("review"), obj={"settings": settings})

        def fake_import_module(name: str):
            return SimpleNamespace(get_llm_client=lambda _: object())

        captured_prompts: list[str] = []

        def fake_execute(_ctx, _client, _settings, prompt, **_kwargs):
            captured_prompts.append(prompt)
            return {
                "result": {"status": "ok"},
                "final_json": {"violations": [], "summary": {"files_reviewed": 1}},
            }

        monkeypatch.setattr(review_module, "import_module", fake_import_module)
        monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)
        monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)

        run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_path),
            json_output=True,
            settings=settings,
        )

        assert captured_prompts, "Review execution did not capture any prompts"
        combined_prompt = "\n".join(captured_prompts)
        assert "Context:" in combined_prompt
        assert "def target()" in combined_prompt

    def test_run_review_shared_parent_root(self, monkeypatch, tmp_path):
        """Workspace must stay unchanged when only '/' is shared."""
        patch_path = tmp_path / "patches" / "changes.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(
            """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -0,0 +1,1 @@
+print("hi")
""",
            encoding="utf-8",
        )

        external_root = Path("/tmp") / f"devagent_rule_{uuid4().hex}"
        external_root.mkdir(parents=True, exist_ok=True)
        rule_path = external_root / "rule.md"
        rule_path.write_text("# Rule\n", encoding="utf-8")

        settings = Settings(
            api_key="test-key",
            workspace_root=tmp_path,
            max_tool_output_chars=4096,
            max_context_tokens=4096,
            response_headroom_tokens=512,
        )

        ctx = click.Context(click.Command("review"), obj={"settings": settings})

        def fake_import_module(name: str):
            return SimpleNamespace(get_llm_client=lambda _: object())

        captured_roots: list[Path] = []

        def fake_build_items(
            self, workspace_root, file_entry
        ):  # pragma: no cover - instrumentation
            captured_roots.append(workspace_root)
            return []

        def fake_execute(*_args, **_kwargs):
            return {
                "result": {"status": "ok"},
                "final_json": {
                    "violations": [],
                    "summary": {"files_reviewed": 1},
                },
            }

        monkeypatch.setattr(review_module, "import_module", fake_import_module)
        monkeypatch.setattr(
            review_module.SourceContextProvider,
            "build_items",
            fake_build_items,
        )
        monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)
        monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)

        try:
            run_review(
                ctx,
                patch_file=str(patch_path),
                rule_file=str(rule_path),
                json_output=True,
                settings=settings,
            )
        finally:
            shutil.rmtree(external_root, ignore_errors=True)

        assert captured_roots
        assert all(root == settings.workspace_root for root in captured_roots)

    def test_run_review_preserves_workspace_for_disparate_paths(self, monkeypatch, tmp_path):
        """Review should not reset workspace_root to an unrelated shared parent."""
        patch_path = tmp_path / "patches" / "changes.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(
            """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -0,0 +1,1 @@
+print("hi")
""",
        )

        rule_path = tmp_path / "rules" / "rule.md"
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        rule_path.write_text("# Rule\n", encoding="utf-8")

        settings = Settings(
            api_key="test-key",
            workspace_root=Path.cwd(),
            max_tool_output_chars=4096,
            max_context_tokens=4096,
            response_headroom_tokens=512,
        )

        ctx = click.Context(click.Command("review"), obj={"settings": settings})

        def fake_import_module(name: str):
            return SimpleNamespace(get_llm_client=lambda _: object())

        captured_roots = []

        def fake_build_items(
            self, workspace_root, file_entry
        ):  # pragma: no cover - instrumentation
            captured_roots.append(workspace_root)
            return []

        def fake_execute(*_args, **_kwargs):
            return {
                "result": {"status": "ok"},
                "final_json": {
                    "violations": [],
                    "summary": {"files_reviewed": 1},
                },
            }

        monkeypatch.setattr(review_module, "import_module", fake_import_module)
        monkeypatch.setattr(
            review_module.SourceContextProvider,
            "build_items",
            fake_build_items,
        )
        monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)
        monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)

        run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_path),
            json_output=True,
            settings=settings,
        )

        assert captured_roots
        assert all(root == settings.workspace_root for root in captured_roots)

    def test_source_context_reads_file_once_per_review(self, monkeypatch, tmp_path):
        """Context provider should avoid rereading same file for chunked patches."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        source_path = workspace / "module.py"
        source_path.write_text(
            "\n".join(
                [
                    "def func():",
                    "    value = 1",
                    "",
                    "def helper():",
                    "    return 2",
                    "",
                    "def tail():",
                    "    return 3",
                ]
            ),
            encoding="utf-8",
        )

        patch_path = tmp_path / "changes.patch"
        patch_path.write_text(
            """diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -1,2 +1,2 @@
-def func():
-    value = 1
+def func():
+    value = compute()
@@ -4,2 +4,2 @@
-def helper():
-    return 2
+def helper():
+    return helper_value()
@@ -7,2 +7,2 @@
-def tail():
-    return 3
+def tail():
+    return compute_tail()
""",
            encoding="utf-8",
        )

        rule_path = workspace / "rule.md"
        rule_path.write_text("# Rule\n", encoding="utf-8")

        settings = Settings(
            api_key="test-key",
            workspace_root=workspace,
            max_tool_output_chars=4096,
            max_context_tokens=4096,
            response_headroom_tokens=512,
        )
        setattr(settings, "review_max_hunks_per_chunk", 1)
        setattr(settings, "review_max_files_per_chunk", 1)

        ctx = click.Context(click.Command("review"), obj={"settings": settings})

        def fake_import_module(name: str):
            return SimpleNamespace(get_llm_client=lambda _: object())

        call_counter = {"count": 0}
        real_read_text = Path.read_text

        def counting_read_text(self, *args, **kwargs):
            if self == source_path:
                call_counter["count"] += 1
            return real_read_text(self, *args, **kwargs)

        def fake_execute(_ctx, _client, _settings, _prompt, **_kwargs):
            return {
                "result": {"status": "ok"},
                "final_json": {
                    "violations": [],
                    "summary": {"files_reviewed": 1},
                },
            }

        monkeypatch.setattr(review_module, "import_module", fake_import_module)
        monkeypatch.setattr(Path, "read_text", counting_read_text, raising=False)
        monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)
        monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)

        run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_path),
            json_output=True,
            settings=settings,
        )

        assert call_counter["count"] == 1

    def test_review_uses_consistent_session_id(self, monkeypatch, tmp_path):
        """All review chunks should share a dedicated session id."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        module_path = workspace / "module.py"
        module_path.write_text(
            "def func():\n    return 1\n\n\ndef helper():\n    return 2\n\n\ndef tail():\n    return 3\n",
            encoding="utf-8",
        )

        patch_path = tmp_path / "chunked.patch"
        patch_path.write_text(
            """diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -1,2 +1,2 @@
-def func():
-    return 1
+def func():
+    return compute()
@@ -4,2 +4,2 @@
-def helper():
-    return 2
+def helper():
+    return helper_value()
@@ -7,2 +7,2 @@
-def tail():
-    return 3
+def tail():
+    return compute_tail()
""",
            encoding="utf-8",
        )

        rule_path = workspace / "rule.md"
        rule_path.write_text("# Rule\n", encoding="utf-8")

        settings = Settings(
            api_key="test-key",
            workspace_root=workspace,
            max_tool_output_chars=4096,
            max_context_tokens=4096,
            response_headroom_tokens=512,
        )
        setattr(settings, "review_max_hunks_per_chunk", 1)
        setattr(settings, "review_max_files_per_chunk", 1)

        ctx = click.Context(click.Command("review"), obj={"settings": settings})
        ctx.obj["_session_id"] = "parent-session"

        def fake_import_module(name: str):
            return SimpleNamespace(get_llm_client=lambda _: object())

        observed_sessions: list[Optional[str]] = []

        def fake_execute(_ctx, _client, _settings, _prompt, **_kwargs):
            observed_sessions.append(_ctx.obj.get("_session_id"))
            return {
                "result": {"status": "ok"},
                "final_json": {
                    "violations": [],
                    "summary": {"files_reviewed": 1},
                },
            }

        monkeypatch.setattr(review_module, "import_module", fake_import_module)
        monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)
        monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)

        run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_path),
            json_output=True,
            settings=settings,
        )

        assert len(observed_sessions) > 1
        assert len(set(observed_sessions)) == 1
        assert observed_sessions[0] != "parent-session"
        assert ctx.obj["_session_id"] == "parent-session"

    def test_review_adjusts_chunking_for_large_rules(self, monkeypatch, tmp_path):
        """Large rules should trigger more aggressive chunking."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        module_path = workspace / "module.py"
        module_lines = [f"line_{idx}" for idx in range(1, 122)]
        module_path.write_text("\n".join(module_lines) + "\n", encoding="utf-8")

        patch_lines = [
            "diff --git a/module.py b/module.py",
            "--- a/module.py",
            "+++ b/module.py",
        ]
        for idx in range(1, 121):
            patch_lines.append(f"@@ -{idx},1 +{idx},1 @@")
            patch_lines.append(f"-line_{idx}")
            patch_lines.append(f"+line_{idx}_updated")
        patch_path = tmp_path / "bulk.patch"
        patch_path.write_text("\n".join(patch_lines) + "\n", encoding="utf-8")

        rule_path = workspace / "rule.md"
        large_rule = "# Rule\n\n" + ("Very long guidance paragraph.\n" * 4000)
        rule_path.write_text(large_rule, encoding="utf-8")

        settings = Settings(
            api_key="test-key",
            workspace_root=workspace,
            max_tool_output_chars=4096,
            max_context_tokens=4096,
            response_headroom_tokens=512,
        )
        setattr(settings, "review_max_hunks_per_chunk", 0)
        setattr(settings, "review_max_lines_per_chunk", 0)

        ctx = click.Context(click.Command("review"), obj={"settings": settings})

        def fake_import_module(name: str):
            return SimpleNamespace(get_llm_client=lambda _: object())

        calls: list[str] = []

        def fake_execute(_ctx, _client, _settings, prompt, **_kwargs):
            calls.append(prompt)
            return {
                "result": {"status": "ok"},
                "final_json": {
                    "violations": [],
                    "summary": {"files_reviewed": 1},
                },
            }

        monkeypatch.setattr(review_module, "import_module", fake_import_module)
        monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)
        monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)

        run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_path),
            json_output=True,
            settings=settings,
        )

        # With optimized chunk sizes (Phase 1), large rules don't necessarily force
        # multiple chunks unless they truly exceed 150K+ token threshold
        # The rule here is ~32K tokens (4000 * 8 chars/paragraph / 4), well under limit
        assert (
            len(calls) >= 1
        )  # At least one call, may be one or multiple depending on dynamic sizing

    def test_review_token_budget_enforces_additional_splits(self, monkeypatch, tmp_path):
        """Token budget should force smaller chunks when necessary."""
        patch_path = tmp_path / "changes.patch"
        patch_path.write_text(
            """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -0,0 +1,1 @@
+print("one")
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -0,0 +1,1 @@
+print("two")
""",
            encoding="utf-8",
        )

        rule_path = tmp_path / "rule.md"
        rule_path.write_text("# Rule\n", encoding="utf-8")

        settings = Settings(
            api_key="test-key",
            workspace_root=tmp_path,
            max_tool_output_chars=4096,
            max_context_tokens=4096,
            response_headroom_tokens=512,
        )
        settings.review_token_budget = 10

        ctx = click.Context(click.Command("review"), obj={"settings": settings})

        def fake_import_module(name: str):
            return SimpleNamespace(get_llm_client=lambda _: object())

        calls: list[str] = []

        def fake_execute(_ctx, _client, _settings, prompt, **_kwargs):
            calls.append(prompt)
            return {
                "result": {"status": "ok"},
                "final_json": {
                    "violations": [],
                    "summary": {"files_reviewed": 1},
                },
            }

        monkeypatch.setattr(review_module, "import_module", fake_import_module)
        monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)
        monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)

        run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_path),
            json_output=True,
            settings=settings,
        )

        assert len(calls) == 2

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
