"""Focused tests for CodeEditor context analysis helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.tools.code.code_edit.context import FileContext
from ai_dev_agent.tools.code.code_edit.editor import CodeEditor, FixAttempt, IterativeFixConfig


@pytest.fixture
def editor(tmp_path):
    """Create a CodeEditor instance with patched session manager."""
    with (
        patch("ai_dev_agent.tools.code.code_edit.editor.SessionManager.get_instance") as mock_mgr,
        patch(
            "ai_dev_agent.tools.code.code_edit.editor.build_system_messages",
            return_value=["system"],
        ),
    ):
        session = MagicMock()
        mock_mgr.return_value = session
        session.ensure_session.return_value = None

        approvals = MagicMock()
        approvals.require.return_value = True
        editor = CodeEditor(
            repo_root=tmp_path,
            llm_client=MagicMock(),
            approvals=approvals,
            fix_config=IterativeFixConfig(run_tests=False),
        )
        yield editor


def _make_context(tmp_path, name: str, content: str, reason: str = "matched task"):
    path = tmp_path / name
    return FileContext(path=path, content=content, relevance_score=0.8, reason=reason)


def test_filter_code_contexts_skips_structure_summary(editor, tmp_path):
    contexts = [
        _make_context(tmp_path, "__project_structure__.md", "summary", "project_structure_summary"),
        _make_context(tmp_path, "app.py", "print('hi')"),
        _make_context(tmp_path, "empty.py", "", "matched task"),
    ]

    filtered = editor._filter_code_contexts(contexts)

    assert len(filtered) == 1
    assert filtered[0].path.name == "app.py"


def test_build_style_profile_detects_conventions(editor, tmp_path):
    contexts = [
        _make_context(
            tmp_path,
            "service.py",
            "from dataclasses import dataclass\n\n@dataclass\nclass Service:\n    name: str\n",
        ),
        _make_context(
            tmp_path,
            "handler.py",
            'def handle(event: str) -> None:\n    print("Processing", event)\n',
        ),
    ]

    profile = editor._build_style_profile(contexts)

    assert "4-space indentation" in profile
    assert "Prefers double quotes" in profile
    assert "Dataclass-heavy modules" in profile
    assert "Type hints in function signatures" in profile


def test_build_dependency_summary_from_multiple_languages(editor, tmp_path):
    contexts = [
        _make_context(tmp_path, "module.py", "import json\nfrom pathlib import Path\n"),
        _make_context(
            tmp_path, "component.ts", "import React from 'react'\nimport util from './util'\n"
        ),
    ]

    summary = editor._build_dependency_summary(contexts)

    assert "module.py: json, pathlib" in summary
    assert "component.ts: react, ./util" in summary.lower()


def test_identify_common_patterns_across_contexts(editor, tmp_path):
    contexts = [
        _make_context(tmp_path, "models.py", "@dataclass\nclass Model:\n    ...\n"),
        _make_context(tmp_path, "routes.py", "import click\nasync def main():\n    pass\n"),
        _make_context(tmp_path, "tests/test_api.py", "import pytest\nassert True\n"),
    ]

    patterns = editor._identify_common_patterns(contexts)

    assert "dataclasses" in patterns
    assert "pytest-style tests" in patterns
    assert "async routines" in patterns


def test_build_quality_notes_highlights_tests_and_todos(editor, tmp_path):
    contexts = [
        _make_context(tmp_path, "tests/test_flow.py", "def test_example():\n    assert True\n"),
        _make_context(tmp_path, "module.py", "# TODO: refine logic\nprint('ok')\n"),
    ]
    latest_attempt = FixAttempt(
        attempt_number=1,
        diff="",
        test_result=SimpleNamespace(command=["pytest"], returncode=1, stdout="", stderr="failure"),
    )

    notes = editor._build_quality_notes(contexts, "Improve performance", latest_attempt)

    assert "Relevant tests" in notes
    assert "Outstanding TODO" in notes
    assert "Assess runtime impact" in notes
    assert "failed" in notes.lower()


def test_format_test_output_includes_exit_code(editor):
    result = SimpleNamespace(returncode=2, stdout="line1\nline2\n", stderr="boom\n")

    output = editor._format_test_output(result)

    assert "Exit code: 2" in output
    assert "STDERR" in output
    assert "STDOUT" in output


def test_extract_files_from_diff(editor):
    diff = """--- a/app.py
+++ b/app.py
@@
--- a/unused.py
+++ /dev/null
"""

    files = editor._extract_files_from_diff(diff)

    assert files == [Path("app.py")]


def test_build_fallback_guidance_lists_contexts(editor, tmp_path):
    contexts = [
        _make_context(
            tmp_path, "core/service.py", "class Service:\n    pass\n", reason="High PageRank"
        ),
        _make_context(tmp_path, "tests/test_service.py", "def test_service():\n    assert True\n"),
    ]

    guidance = editor._build_fallback_guidance("Improve service", contexts, reason="llm_error")

    assert "llm_error" in guidance
    assert "core/service.py" in guidance
    assert "tests/test_service.py" in guidance


def test_build_style_profile_handles_tabs_and_single_quotes(editor, tmp_path):
    contexts = [
        _make_context(tmp_path, "tabbed.py", "\tdef handler():\n\t\treturn 'value'\n"),
        _make_context(tmp_path, "plain.py", "\tif flag:\n\t\treturn 'fallback'\n"),
    ]

    profile = editor._build_style_profile(contexts)

    assert "Tab-indented codebase" in profile
    assert "Prefers single quotes" in profile


def test_build_style_profile_no_contexts_returns_default(editor):
    profile = editor._build_style_profile([])

    assert profile == "No dominant style detected from provided files; mirror local context."


def test_build_dependency_summary_without_dependencies(editor, tmp_path):
    contexts = [
        _make_context(tmp_path, "module.py", "value = 1\n"),
    ]

    summary = editor._build_dependency_summary(contexts)

    assert summary == "No external dependencies detected in the provided snippets."


def test_build_quality_notes_handles_external_paths_and_security(editor, tmp_path):
    external = FileContext(
        path=Path("/outside/tests/test_alpha.py"),
        content="def test_alpha():\n    pass\n",
        relevance_score=0.8,
        reason="external-discovery",
    )
    test_contexts = [
        _make_context(
            tmp_path, f"tests/test_module_{idx}.py", "def test_case():\n    assert True\n"
        )
        for idx in range(4)
    ]
    todo_context = _make_context(
        tmp_path, "module.py", "# TODO: tighten auth checks\nvalue = compute()\n"
    )
    contexts = [external, *test_contexts, todo_context]

    notes = editor._build_quality_notes(
        contexts,
        "Perform security hardening",
        latest_attempt=None,
    )

    assert "Outstanding TODO" in notes
    assert "Relevant tests" in notes
    assert "…" in notes
    assert "Validate security invariants" in notes
    assert "/outside/tests/test_alpha.py" in notes


def test_build_fallback_guidance_with_external_paths(editor):
    contexts = [
        FileContext(
            path=Path("/opt/project/service.py"),
            content="def run():\n    pass\n",
            relevance_score=0.9,
            reason="external-match",
        )
    ]

    guidance = editor._build_fallback_guidance(
        "Handle fallback guidance", contexts, reason="network error"
    )

    assert "- /opt/project/service.py" in guidance
    assert "network error" in guidance


def test_build_fallback_guidance_without_context(editor):
    guidance = editor._build_fallback_guidance("   ", [], reason="no context")

    assert "No task description provided" in guidance
    assert "no context" in guidance


def test_run_tests_requires_configured_runner(editor):
    with pytest.raises(RuntimeError):
        editor._run_tests()


def test_run_tests_with_command(editor):
    runner = MagicMock()
    editor.test_runner = runner

    result = MagicMock()
    runner.run.return_value = result

    command = ["pytest", "-k", "fast"]
    assert editor._run_tests(command) is result
    runner.run.assert_called_once_with(command)


def test_run_tests_with_default_pytest(editor):
    runner = MagicMock()
    editor.test_runner = runner

    result = MagicMock()
    runner.run_pytest.return_value = result

    assert editor._run_tests() is result
    runner.run_pytest.assert_called_once_with()
