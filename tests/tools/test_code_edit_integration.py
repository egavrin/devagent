import textwrap
from pathlib import Path

import pytest

from ai_dev_agent.core.utils.constants import LLM_DEFAULT_TEMPERATURE
from ai_dev_agent.tools.code.code_edit.context import ContextGatherer, ContextGatheringOptions
from ai_dev_agent.tools.code.code_edit.diff_utils import DiffProcessor
from ai_dev_agent.tools.code.code_edit.editor import CodeEditor, IterativeFixConfig


class DummyStructureAnalyzer:
    max_lines_per_file = 5

    def summarize_content(self, rel_path: str, content: str):
        return [f"{rel_path}:{len(content.splitlines())}"]


class AlwaysApprove:
    def __init__(self):
        self.calls: list[tuple[str, bool]] = []

    def require(self, key: str, default: bool = True) -> bool:
        self.calls.append((key, default))
        return True


class StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[list] = []

    def complete(self, messages, temperature: float = LLM_DEFAULT_TEMPERATURE):
        self.calls.append(messages)
        return self.response


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    sample = repo / "sample.py"
    sample.write_text(
        textwrap.dedent(
            """
            def greet():
                return "hi"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return repo


def test_diff_processor_extract_preview_apply(monkeypatch, sample_repo: Path):
    processor = DiffProcessor(sample_repo)
    diff_response = """```diff
--- a/sample.py
+++ b/sample.py
@@ -1,2 +1,2 @@
-def greet():
-    return "hi"
+def greet(name: str) -> str:
+    return f"hi {name}"
```"""

    diff_text, validation = processor.extract_and_validate_diff(diff_response)
    assert validation.errors == []
    assert validation.affected_files == ["sample.py"]

    preview = processor.create_preview(diff_text)
    assert "sample.py" in preview.file_changes
    assert preview.validation_result.lines_added > 0

    calls = []

    def fake_run(cmd, input=None, cwd=None, capture_output=False, timeout=None):
        calls.append(tuple(cmd))

        class Result:
            def __init__(self):
                self.returncode = 0
                self.stdout = b""
                self.stderr = b""

        return Result()

    monkeypatch.setattr("ai_dev_agent.tools.code.code_edit.diff_utils.subprocess.run", fake_run)

    assert processor.apply_diff_safely(diff_text) is True
    assert calls and calls[0][0:2] == ("git", "apply")


def test_code_editor_propose_and_apply_diff(monkeypatch, sample_repo: Path):
    llm_response = """```diff
--- a/sample.py
+++ b/sample.py
@@ -1,2 +1,2 @@
-def greet():
-    return "hi"
+def greet(name: str) -> str:
+    return f"hi {name}"
```"""
    llm = StubLLM(llm_response)
    approvals = AlwaysApprove()
    editor = CodeEditor(
        repo_root=sample_repo,
        llm_client=llm,
        approvals=approvals,
        fix_config=IterativeFixConfig(
            max_attempts=1, run_tests=False, enable_context_expansion=False
        ),
    )

    gatherer = ContextGatherer(
        sample_repo,
        ContextGatheringOptions(
            include_related_files=False,
            include_structure_summary=False,
            use_repo_map=False,
        ),
    )
    gatherer._structure_analyzer = DummyStructureAnalyzer()
    editor.context_gatherer = gatherer

    applied_diffs = []

    def fake_apply(diff_text: str):
        applied_diffs.append(diff_text)
        return True

    monkeypatch.setattr(editor.diff_processor, "apply_diff_safely", fake_apply)

    proposal = editor.propose_diff("Update greeting", ["sample.py"])
    assert proposal.validation_errors == []
    assert proposal.preview is not None
    assert "sample.py" in proposal.preview.file_changes

    editor.apply_diff(proposal)
    assert approvals.calls and approvals.calls[-1][0] == "code"
    assert applied_diffs and "--- a/sample.py" in applied_diffs[0]
