import os
import subprocess

import pytest

from ai_dev_agent.core.approval.approvals import ApprovalManager
from ai_dev_agent.core.approval.policy import ApprovalPolicy
from ai_dev_agent.providers.llm.base import LLMError
from ai_dev_agent.tools.code.code_edit.editor import CodeEditor, IterativeFixConfig


class DiffClient:
    def __init__(self, diff_text: str) -> None:
        self.diff_text = diff_text

    def complete(self, messages, temperature=0.2, max_tokens=None):
        return self.diff_text


class FailingClient:
    def complete(self, messages, temperature=0.2, max_tokens=None):
        raise LLMError("LLM offline")


def test_code_editor_applies_diff(tmp_path, monkeypatch):
    repo_root = tmp_path
    (repo_root / "file.txt").write_text("hello\n", encoding="utf-8")
    # initialize git repository for git apply to work reliably
    import subprocess

    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    diff_text = (
        "```diff\n"
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1 +1 @@\n"
        "-hello\n"
        "+hello world\n"
        "```"
    )
    client = DiffClient(diff_text)
    approvals = ApprovalManager(ApprovalPolicy(auto_approve_code=True))
    editor = CodeEditor(repo_root, client, approvals)
    proposal = editor.propose_diff("Update greeting", ["file.txt"])
    assert "hello world" in proposal.diff
    editor.apply_diff(proposal)
    updated = (repo_root / "file.txt").read_text(encoding="utf-8")
    assert "hello world" in updated


def test_code_editor_provides_fallback(tmp_path):
    repo_root = tmp_path
    (repo_root / "sample.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    approvals = ApprovalManager(ApprovalPolicy(auto_approve_code=True))
    editor = CodeEditor(repo_root, FailingClient(), approvals)

    proposal = editor.propose_diff("Adjust foo", ["sample.py"])

    assert proposal.diff == ""
    assert proposal.fallback_reason == "LLM offline"
    assert proposal.fallback_guidance

    with pytest.raises(LLMError):
        editor.apply_diff(proposal)


def test_apply_diff_with_fixes_invokes_preview_callback(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    target = repo_root / "file.txt"
    target.write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        env=env,
    )

    diff_text = (
        "```diff\n"
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1 +1 @@\n"
        "-hello\n"
        "+hello world\n"
        "```"
    )

    client = DiffClient(diff_text)
    approvals = ApprovalManager(ApprovalPolicy(auto_approve_code=True))
    fix_config = IterativeFixConfig(run_tests=False)
    editor = CodeEditor(repo_root, client, approvals, fix_config=fix_config)

    previews: list[tuple[int, str]] = []

    success, attempts = editor.apply_diff_with_fixes(
        "Update greeting",
        ["file.txt"],
        on_proposal=lambda proposal, attempt: previews.append(
            (attempt, proposal.preview.summary if proposal.preview else "")
        ),
    )

    assert success is True
    assert previews and previews[0][0] == 1
    assert "file" in previews[0][1]
    assert attempts and attempts[0].preview is not None
    assert target.read_text(encoding="utf-8") == "hello world\n"
