"""Additional CodeEditor operation tests covering complex edit flows."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_dev_agent.tools.code.code_edit.context import FileContext
from ai_dev_agent.tools.code.code_edit.editor import CodeEditor, DiffProposal, IterativeFixConfig


def _build_session_manager(session: SimpleNamespace | None = None):
    """Return a stubbed session manager and backing session."""
    if session is None:
        session = SimpleNamespace(lock=nullcontext(), metadata={}, system_messages=[])
    manager = MagicMock()
    # __init__ call returns None, subsequent calls return the provided session
    manager.ensure_session.side_effect = [None, session]
    manager.compose.return_value = [{"role": "system", "content": "system"}]
    manager.add_user_message = MagicMock()
    manager.add_assistant_message = MagicMock()
    manager.add_system_message = MagicMock()
    return manager, session


def test_gather_context_includes_search_replace_keywords(tmp_path):
    llm_client = MagicMock()
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
        session_manager, _ = _build_session_manager()
        mock_mgr.return_value = session_manager
        editor = CodeEditor(
            repo_root=tmp_path,
            llm_client=llm_client,
            approvals=approvals,
            fix_config=IterativeFixConfig(run_tests=False),
        )

        gather_mock.return_value = []

        files = ["src/main.py"]
        task = "Handle search/replace logic for JSON tests in main flow."

        editor.gather_context(files, task)

        assert gather_mock.called
        kwargs = gather_mock.call_args.kwargs
        assert kwargs["chat_files"][0] == tmp_path / "src/main.py"
        assert {"search", "replace", "json"} <= set(kwargs["keywords"])


def test_propose_diff_handles_multi_file_search_replace(tmp_path):
    repo_root = tmp_path
    (repo_root / "src").mkdir()
    (repo_root / "src/app.py").write_text(
        'def render(data):\n    print("search result:", data["search_term"])\n', encoding="utf-8"
    )
    (repo_root / "src/utils.py").write_text(
        'def transform(item):\n    value = item["search"]\n    return value.upper()\n',
        encoding="utf-8",
    )

    llm_client = MagicMock()
    approvals = MagicMock()
    approvals.require.return_value = True

    diff_block = (
        "```diff\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,4 @@\n"
        " def render(data):\n"
        '-    print("search result:", data["search_term"])\n'
        '+    print("search result:", data.get("search_term"))\n'
        '+    updated = data["search_term"].replace("deprecated", "replacement")\n'
        "+    return updated\n"
        "--- a/src/utils.py\n"
        "+++ b/src/utils.py\n"
        "@@ -1,3 +1,4 @@\n"
        " def transform(item):\n"
        '-    value = item["search"]\n'
        "-    return value.upper()\n"
        '+    value = item.get("search")\n'
        '+    return value.upper() if value else ""\n'
        "```"
    )
    llm_client.complete.return_value = f"Here is the update:\n{diff_block}\n"

    contexts = [
        FileContext(
            path=repo_root / "src/app.py",
            content=(repo_root / "src/app.py").read_text(encoding="utf-8"),
            relevance_score=0.9,
        ),
        FileContext(
            path=repo_root / "src/utils.py",
            content=(repo_root / "src/utils.py").read_text(encoding="utf-8"),
            relevance_score=0.8,
        ),
    ]

    session = SimpleNamespace(lock=nullcontext(), metadata={}, system_messages=[])

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
        session_manager, session_obj = _build_session_manager(session)
        # __init__ consumes first ensure_session call, so provide another for propose_diff
        session_manager.ensure_session.side_effect = [None, session_obj]
        mock_mgr.return_value = session_manager
        gather_mock.return_value = contexts

        editor = CodeEditor(
            repo_root=repo_root,
            llm_client=llm_client,
            approvals=approvals,
            fix_config=IterativeFixConfig(run_tests=False),
        )

        proposal = editor.propose_diff(
            "Update search/replace handling", files=["src/app.py", "src/utils.py"]
        )

        assert 'replace("deprecated", "replacement")' in proposal.diff
        assert [Path("src/app.py"), Path("src/utils.py")] == proposal.files
        assert proposal.preview is not None
        assert {"src/app.py", "src/utils.py"} == set(proposal.preview.file_changes)
        assert proposal.validation_errors == []
        assert session.metadata["last_context_summary"]["style"]


def test_apply_diff_with_fixes_handles_validation_and_decline(tmp_path):
    llm_client = MagicMock()
    approvals = MagicMock()
    approvals.require.return_value = False
    repo_root = tmp_path
    (repo_root / "src").mkdir()
    (repo_root / "src/app.py").write_text("print('hello')\n", encoding="utf-8")

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
        gather_mock.return_value = []
        session_manager, session = _build_session_manager()
        session_manager.ensure_session.side_effect = [None, session, session]
        mock_mgr.return_value = session_manager

        editor = CodeEditor(
            repo_root=repo_root,
            llm_client=llm_client,
            approvals=approvals,
            fix_config=IterativeFixConfig(max_attempts=2, run_tests=False),
        )

        raw_diff = (
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1 +1 @@\n"
            "-print('hello')\n"
            "+print('goodbye')\n"
        )
        raw_diff_with_newline = raw_diff + "\n"
        preview = editor.diff_processor.create_preview(raw_diff_with_newline)

        invalid = DiffProposal(
            diff="",
            raw_response="invalid",
            files=[],
            preview=None,
            validation_errors=["missing diff"],
        )
        valid_diff_text = raw_diff_with_newline
        valid = DiffProposal(
            diff=valid_diff_text,
            raw_response="valid",
            files=[Path("src/app.py")],
            preview=preview,
        )

        editor.propose_diff = MagicMock(side_effect=[invalid, valid])
        editor.diff_processor.apply_diff_safely = MagicMock()

        proposals_triggered = []

        success, attempts = editor.apply_diff_with_fixes(
            "Adjust greeting text",
            ["src/app.py"],
            on_proposal=lambda proposal, attempt: proposals_triggered.append(
                (attempt, proposal.diff)
            ),
        )

        assert success is False
        assert len(attempts) == 2
        assert attempts[0].validation_errors == ["missing diff"]
        assert "Diff validation failed" in attempts[0].error_message
        assert attempts[1].approved is False
        assert attempts[1].error_message == "Diff application declined by user."
        assert proposals_triggered == [(2, valid_diff_text)]
        editor.diff_processor.apply_diff_safely.assert_not_called()
