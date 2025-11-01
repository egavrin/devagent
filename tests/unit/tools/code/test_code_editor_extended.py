"""Extended tests for CodeEditor covering complex edit flows and utilities."""

from __future__ import annotations

import textwrap
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.tools.code.code_edit.context import FileContext
from ai_dev_agent.tools.code.code_edit.editor import CodeEditor, DiffProposal, IterativeFixConfig


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
