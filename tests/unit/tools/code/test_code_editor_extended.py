"""Extended tests for CodeEditor covering complex edit flows and utilities."""

from __future__ import annotations

import textwrap
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.providers.llm import LLMError
from ai_dev_agent.tools.code.code_edit.context import FileContext
from ai_dev_agent.tools.code.code_edit.diff_utils import DiffError, DiffValidationResult
from ai_dev_agent.tools.code.code_edit.editor import (
    CodeEditor,
    DiffProposal,
    FixAttempt,
    IterativeFixConfig,
)
from ai_dev_agent.tools.execution.testing.local_tests import TestResult


def _build_session_manager(session: SimpleNamespace | None = None):
    """Return a stubbed session manager and backing session."""
    if session is None:
        session = SimpleNamespace(lock=nullcontext(), metadata={}, system_messages=[])
    manager = MagicMock()
    # __init__ call returns None, subsequent calls return the provided session
    first_call = {"done": False}

    def _ensure_session_side_effect(*_args, **_kwargs):
        if not first_call["done"]:
            first_call["done"] = True
            return None
        return session

    manager.ensure_session.side_effect = _ensure_session_side_effect
    manager.compose.return_value = [{"role": "system", "content": "system"}]
    manager.add_user_message = MagicMock()
    manager.add_assistant_message = MagicMock()
    manager.add_system_message = MagicMock()
    return manager, session


def _patch_editor(tmp_path: Path, approvals: MagicMock, llm_client: MagicMock):
    """Patch global collaborators and return a configured CodeEditor instance."""
    with (
        patch("ai_dev_agent.tools.code.code_edit.editor.SessionManager.get_instance") as mock_mgr,
        patch(
            "ai_dev_agent.tools.code.code_edit.editor.build_system_messages",
            return_value=["system"],
        ),
        patch(
            "ai_dev_agent.tools.code.code_edit.editor.ContextGatherer.gather_contexts"
        ) as gather_mock,
    ):
        session_manager, session = _build_session_manager()
        mock_mgr.return_value = session_manager
        gather_mock.return_value = []

        editor = CodeEditor(
            repo_root=tmp_path,
            llm_client=llm_client,
            approvals=approvals,
            fix_config=IterativeFixConfig(max_attempts=2, run_tests=False),
        )

        # Ensure gather_context returns at least the target file when requested
        def gather_context(files, task_description=None, keywords=None, chat_files=None):
            contexts = []
            for file_path in files:
                abs_path = tmp_path / file_path
                if abs_path.exists():
                    contexts.append(
                        FileContext(
                            path=abs_path,
                            content=abs_path.read_text(encoding="utf-8"),
                            relevance_score=0.9,
                            reason="matched task",
                        )
                    )
            return contexts

        gather_mock.side_effect = gather_context
        return editor, session_manager, session


def _make_test_result(returncode: int, command: list[str] | None = None) -> TestResult:
    """Create a lightweight test result object for runner simulations."""
    return TestResult(
        command=command or ["pytest"],
        returncode=returncode,
        stdout="",
        stderr="",
    )


def _make_proposal(
    *,
    diff: str = "--- a/app.py\n+++ b/app.py\n@@\n-1\n+2\n",
    files: list[Path] | None = None,
    raw_response: str | None = None,
    validation_errors: list[str] | None = None,
    fallback_reason: str | None = None,
    fallback_guidance: str | None = None,
) -> DiffProposal:
    """Construct a DiffProposal with sensible defaults for tests."""
    return DiffProposal(
        diff=diff,
        raw_response=raw_response or diff,
        files=files or [Path("app.py")],
        preview=None,
        validation_errors=list(validation_errors or []),
        fallback_reason=fallback_reason,
        fallback_guidance=fallback_guidance,
    )


def test_editor_complex_edits(tmp_path):
    """Test complex editing operations including multi-line replacements and insertions."""
    repo_root = tmp_path
    target = repo_root / "service.py"
    target.write_text(
        textwrap.dedent(
            """
            def format_payload(data):
                result = []
                for item in data:
                    result.append(item.strip())
                return ", ".join(result)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    diff_response = '''```diff
--- a/service.py
+++ b/service.py
@@ -1,5 +1,9 @@
-def format_payload(data):
-    result = []
-    for item in data:
-        result.append(item.strip())
-    return ", ".join(result)
+def format_payload(data, *, separator=", ", transform=str):
+    """Normalize, transform, and join incoming data."""
+    processed = []
+    for item in data:
+        cleaned = item.strip()
+        if callable(transform):
+            cleaned = transform(cleaned)
+        processed.append(cleaned)
+    return separator.join(processed)
```'''

    llm_client = MagicMock()
    llm_client.complete.return_value = diff_response
    approvals = MagicMock()
    approvals.require.return_value = True

    with (
        patch("ai_dev_agent.tools.code.code_edit.editor.SessionManager.get_instance") as mock_mgr,
        patch(
            "ai_dev_agent.tools.code.code_edit.editor.build_system_messages",
            return_value=["system"],
        ),
        patch(
            "ai_dev_agent.tools.code.code_edit.editor.ContextGatherer.gather_contexts"
        ) as gather_mock,
    ):
        session_manager, session = _build_session_manager()
        mock_mgr.return_value = session_manager
        gather_mock.side_effect = lambda files, **_: [
            FileContext(
                path=target,
                content=target.read_text(encoding="utf-8"),
                relevance_score=0.9,
                reason="matched task",
            )
        ]

        editor = CodeEditor(
            repo_root=repo_root,
            llm_client=llm_client,
            approvals=approvals,
            fix_config=IterativeFixConfig(max_attempts=1, run_tests=False),
        )

        proposals = []
        success, attempts = editor.apply_diff_with_fixes(
            "Enhance payload formatting",
            files=["service.py"],
            on_proposal=lambda proposal, num: proposals.append((num, proposal)),
        )

        assert success is True
        assert proposals and proposals[0][0] == 1
        proposal = proposals[0][1]
        assert proposal.preview is not None
        assert proposal.preview.validation_result.lines_added >= 4
        assert "separator" in proposal.diff
        assert target.read_text(encoding="utf-8").startswith(
            'def format_payload(data, *, separator=", ", transform=str):'
        )
        assert "processed.append(cleaned)" in target.read_text(encoding="utf-8")


def test_editor_search_replace(tmp_path):
    """Test search-and-replace helper with regex, case sensitivity, and whole-word support."""
    repo_root = tmp_path
    (repo_root / "src").mkdir()

    auth_path = repo_root / "src" / "auth.py"
    auth_path.write_text("token Token token\n", encoding="utf-8")

    profile_path = repo_root / "src" / "profile.py"
    profile_path.write_text("user user_id enduser\n", encoding="utf-8")

    audit_path = repo_root / "src" / "audit.py"
    audit_path.write_text("event=login user_id=42 reference=123-45-6789\n", encoding="utf-8")

    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()
    llm_client.complete.return_value = ""

    editor, session_manager, _ = _patch_editor(repo_root, approvals, llm_client)

    proposal = editor.search_and_replace(
        "src/auth.py",
        pattern="token",
        replacement="credential",
        regex=False,
        case_sensitive=True,
        whole_word=False,
    )
    editor.apply_diff(proposal)
    contents = auth_path.read_text(encoding="utf-8")
    assert contents.count("credential") == 2
    assert "Token" in contents  # Case-sensitive, so capitalized variant remains

    proposal = editor.search_and_replace(
        "src/profile.py",
        pattern="user",
        replacement="member",
        regex=False,
        case_sensitive=False,
        whole_word=True,
    )
    editor.apply_diff(proposal)
    profile_contents = profile_path.read_text(encoding="utf-8")
    assert profile_contents.startswith("member ")
    assert profile_contents.count("member") == 1
    assert "user_id" in profile_contents  # Whole-word prevents replacement inside identifiers
    assert "enduser" in profile_contents

    proposal = editor.search_and_replace(
        "src/audit.py",
        pattern=r"(\d{3})-(\d{2})-(\d{4})",
        replacement="***-**-****",
        regex=True,
        case_sensitive=False,
        whole_word=False,
    )
    editor.apply_diff(proposal)
    audit_contents = audit_path.read_text(encoding="utf-8")
    assert "***-**-****" in audit_contents
    assert proposal.preview is not None
    assert not proposal.validation_errors

    # Ensure session metadata was updated during operations
    assert session_manager.add_user_message.call_count >= 1


def test_editor_diff_generation_handles_conflicts(tmp_path):
    """Test diff generation and conflict detection paths."""
    repo_root = tmp_path
    target = repo_root / "module.py"
    target.write_text("value = compute()\nreturn value\n", encoding="utf-8")

    conflict_diff = """```diff
--- a/module.py
+++ b/module.py
@@
<<<<<<< HEAD
value = compute()
=======
value = compute_v2()
>>>>>>> branch
return value
```"""

    llm_client = MagicMock()
    llm_client.complete.return_value = conflict_diff
    approvals = MagicMock()
    approvals.require.return_value = True

    with (
        patch("ai_dev_agent.tools.code.code_edit.editor.SessionManager.get_instance") as mock_mgr,
        patch(
            "ai_dev_agent.tools.code.code_edit.editor.build_system_messages",
            return_value=["system"],
        ),
        patch(
            "ai_dev_agent.tools.code.code_edit.editor.ContextGatherer.gather_contexts"
        ) as gather_mock,
    ):
        session_manager, session = _build_session_manager()
        mock_mgr.return_value = session_manager
        gather_mock.side_effect = lambda files, **_: [
            FileContext(
                path=target,
                content=target.read_text(encoding="utf-8"),
                relevance_score=0.9,
                reason="matched task",
            )
        ]

        editor = CodeEditor(
            repo_root=repo_root,
            llm_client=llm_client,
            approvals=approvals,
            fix_config=IterativeFixConfig(max_attempts=1, run_tests=False),
        )

        success, attempts = editor.apply_diff_with_fixes(
            "Resolve merge conflicts", files=["module.py"]
        )

        assert success is False
        assert attempts
        attempt = attempts[0]
        assert attempt.validation_errors
        assert any("conflict" in err.lower() for err in attempt.validation_errors)
        assert "Diff validation failed" in (attempt.error_message or "")
        # The diff should not have been applied when conflicts are detected
        assert target.read_text(encoding="utf-8") == "value = compute()\nreturn value\n"


def test_propose_diff_handles_llm_error_fallback(tmp_path):
    """Ensure LLM errors trigger fallback guidance rather than hard failures."""
    repo_root = tmp_path
    target = repo_root / "package.py"
    target.write_text("value = 1\n", encoding="utf-8")

    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()
    llm_client.complete.side_effect = LLMError("timeout reached")

    editor, session_manager, _ = _patch_editor(repo_root, approvals, llm_client)

    proposal = editor.propose_diff("Adjust value constant", files=["package.py"])

    assert proposal.diff == ""
    assert proposal.fallback_reason == "timeout reached"
    assert "package.py" in (proposal.fallback_guidance or "")
    session_manager.add_system_message.assert_called()


def test_propose_diff_validation_error_returns_empty_diff(tmp_path):
    """Invalid diff responses should be converted into validation errors."""
    repo_root = tmp_path
    target = repo_root / "logic.py"
    target.write_text("value = 0\n", encoding="utf-8")

    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()
    llm_client.complete.return_value = """```diff
--- a/logic.py
+++ b/logic.py
@@
-value = 0
+value = 1
```"""

    editor, _, _ = _patch_editor(repo_root, approvals, llm_client)

    with (
        patch.object(
            editor.diff_processor, "extract_and_validate_diff", side_effect=DiffError("bad diff")
        ),
        patch.object(editor.diff_processor, "create_preview") as preview_mock,
    ):
        proposal = editor.propose_diff("Update logic constant", files=["logic.py"])

    assert proposal.diff == ""
    assert proposal.validation_errors == ["bad diff"]
    preview_mock.assert_not_called()


def test_propose_diff_uses_fix_template_for_follow_up_attempts(tmp_path):
    """Follow-up attempts should use the fix template and higher temperature."""
    repo_root = tmp_path
    target = repo_root / "feature.py"
    target.write_text("def compute():\n    return 1\n", encoding="utf-8")

    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()
    diff_body = """--- a/feature.py
+++ b/feature.py
@@
-    return 1
+    return 2
"""
    llm_client.complete.return_value = f"```diff\n{diff_body}```"

    editor, session_manager, _ = _patch_editor(repo_root, approvals, llm_client)
    validation = DiffValidationResult(
        is_valid=False,
        errors=["Needs review"],
        warnings=[],
        affected_files=["feature.py"],
        lines_added=1,
        lines_removed=1,
    )
    preview = MagicMock()
    preview.validation_result.errors = []
    preview.validation_result.warnings = []
    preview.summary = "summary"
    latest_attempt = FixAttempt(
        attempt_number=1,
        diff="--- a/feature.py\n+++ b/feature.py\n@@\n-    return 1\n+    return 3\n",
        validation_errors=["previous failure"],
        test_result=_make_test_result(1, command=["pytest", "-k", "compute"]),
    )

    with (
        patch.object(
            editor.diff_processor, "extract_and_validate_diff", return_value=(diff_body, validation)
        ),
        patch.object(editor.diff_processor, "create_preview", return_value=preview),
    ):
        proposal = editor.propose_diff(
            "Fix compute implementation", files=["feature.py"], previous_attempts=[latest_attempt]
        )

    # Ensure the fix template made it into the prompt
    prompt = session_manager.add_user_message.call_args_list[-1].args[1]
    assert "Previous Attempt Diff" in prompt
    assert "Test Results" in prompt
    assert proposal.diff.endswith("\n")
    # Temperature now uses centralized default (0.0) - not passed explicitly


def test_apply_diff_with_fixes_stops_on_fallback(tmp_path):
    """A fallback proposal should terminate the iterative fix loop."""
    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)
    editor.fix_config = IterativeFixConfig(max_attempts=2, run_tests=False)

    fallback = _make_proposal(
        fallback_reason="unavailable", fallback_guidance="Manual intervention required."
    )

    with patch.object(editor, "propose_diff", return_value=fallback):
        success, attempts = editor.apply_diff_with_fixes("Update code", files=["app.py"])

    assert success is False
    assert len(attempts) == 1
    assert attempts[0].error_message == "Manual intervention required."
    assert attempts[0].approved is None


def test_apply_diff_with_fixes_records_application_failure(tmp_path):
    """Diff application errors should be surfaced and stop the attempt."""
    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)
    editor.fix_config = IterativeFixConfig(max_attempts=1, run_tests=False)

    proposal = _make_proposal()

    with (
        patch.object(editor, "propose_diff", return_value=proposal),
        patch.object(
            editor, "_apply_diff_with_approval", side_effect=RuntimeError("conflict detected")
        ),
    ):
        success, attempts = editor.apply_diff_with_fixes("Apply diff", files=["module.py"])

    assert success is False
    assert len(attempts) == 1
    assert attempts[0].error_message == "Failed to apply diff: conflict detected"
    assert attempts[0].approved is None


def test_apply_diff_with_tests_failure_and_retry(tmp_path):
    """Ensure failing tests trigger retries up to the configured maximum."""
    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)
    editor.fix_config = IterativeFixConfig(max_attempts=2, run_tests=True)
    editor.test_runner = MagicMock()

    proposals = [
        _make_proposal(),
        _make_proposal(diff="--- a/app.py\n+++ b/app.py\n@@\n-1\n+3\n"),
    ]

    with (
        patch.object(editor, "propose_diff", side_effect=proposals),
        patch.object(editor, "_apply_diff_with_approval", return_value=True),
        patch.object(
            editor,
            "_run_tests",
            side_effect=[
                _make_test_result(1, ["pytest"]),
                _make_test_result(1, ["pytest", "-k", "retry"]),
            ],
        ),
    ):
        success, attempts = editor.apply_diff_with_fixes(
            "Refine implementation", files=["module.py"]
        )

    assert success is False
    assert len(attempts) == 2
    assert all(attempt.approved for attempt in attempts)
    assert all(attempt.error_message == "Tests failed." for attempt in attempts)


def test_apply_diff_with_tests_stops_after_success(tmp_path):
    """Successful test execution should exit the loop early."""
    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)
    editor.fix_config = IterativeFixConfig(max_attempts=2, run_tests=True)
    editor.test_runner = MagicMock()

    proposal = _make_proposal()

    with (
        patch.object(editor, "propose_diff", return_value=proposal),
        patch.object(editor, "_apply_diff_with_approval", return_value=True),
        patch.object(editor, "_run_tests", return_value=_make_test_result(0, ["pytest"])),
    ):
        success, attempts = editor.apply_diff_with_fixes("Implement feature", files=["module.py"])

    assert success is True
    assert len(attempts) == 1
    assert attempts[0].test_result.returncode == 0


def test_apply_diff_with_approval_wraps_diff_errors(tmp_path):
    """Diff application errors should be surfaced as LLMError instances."""
    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)
    proposal = _make_proposal()

    with patch.object(
        editor.diff_processor, "apply_diff_safely", side_effect=DiffError("conflict")
    ):
        with pytest.raises(LLMError) as excinfo:
            editor._apply_diff_with_approval(proposal)

    assert "conflict" in str(excinfo.value)


def test_apply_diff_with_approval_respects_user_decline(tmp_path):
    """Approval manager returning False should abort diff application."""
    approvals = MagicMock()
    approvals.require.return_value = False
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)
    proposal = _make_proposal()

    assert editor._apply_diff_with_approval(proposal) is False


def test_apply_diff_with_approval_succeeds(tmp_path):
    """Successful diff application should return True."""
    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)
    proposal = _make_proposal()

    with patch.object(editor.diff_processor, "apply_diff_safely", return_value=True):
        assert editor._apply_diff_with_approval(proposal) is True


def test_search_and_replace_supports_multiline_regex(tmp_path):
    """Complex regex replacements should be handled correctly."""
    repo_root = tmp_path
    target = repo_root / "data.txt"
    target.write_text("item: alpha\nitem: beta\nVALUE: gamma\n", encoding="utf-8")

    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()
    llm_client.complete.return_value = ""

    editor, _, _ = _patch_editor(repo_root, approvals, llm_client)

    proposal = editor.search_and_replace(
        "data.txt",
        pattern=r"(?im)^item: (\w+)$",
        replacement=r"item: \1-processed",
        regex=True,
        case_sensitive=False,
    )

    diff_text = proposal.diff
    assert "+item: alpha-processed" in diff_text
    assert "+item: beta-processed" in diff_text
    assert "VALUE: gamma" in target.read_text(encoding="utf-8")


def test_search_and_replace_requires_change(tmp_path):
    """Search and replace should raise when no changes are produced."""
    repo_root = tmp_path
    target = repo_root / "module.py"
    target.write_text("status = 'ok'\n", encoding="utf-8")

    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(repo_root, approvals, llm_client)

    with pytest.raises(ValueError, match="produced no changes"):
        editor.search_and_replace("module.py", pattern="missing", replacement="value")


def test_search_and_replace_missing_file(tmp_path):
    """Missing files should raise a FileNotFoundError before processing."""
    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)

    with pytest.raises(FileNotFoundError):
        editor.search_and_replace("does_not_exist.py", pattern="foo", replacement="bar")


def test_build_manual_diff_outputs_unified_diff(tmp_path):
    """Manual diff builder should produce a unified diff with headers."""
    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)
    diff_text = editor._build_manual_diff(Path("module.py"), "value = 1\n", "value = 2\n")

    assert diff_text.startswith("--- a/module.py")
    assert "+++ b/module.py" in diff_text
    assert "+value = 2" in diff_text


def test_build_manual_diff_requires_changes(tmp_path):
    """Manual diff builder should reject identical content."""
    approvals = MagicMock()
    approvals.require.return_value = True
    llm_client = MagicMock()

    editor, _, _ = _patch_editor(tmp_path, approvals, llm_client)

    with pytest.raises(ValueError):
        editor._build_manual_diff(Path("module.py"), "value = 1\n", "value = 1\n")
