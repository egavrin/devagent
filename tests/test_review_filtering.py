from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import click
import pytest

from ai_dev_agent.cli.review import run_review
from ai_dev_agent.core.utils.config import Settings


class DummyCompleteResult:
    def __init__(self, message: str) -> None:
        self.message_content = message
        self.calls = []
        self.raw_tool_calls = []


class DummyClient:
    """Stubbed LLM client returning the supplied assistant message."""

    def __init__(self, message: Dict[str, any]) -> None:
        self._message = json.dumps(message)

    def invoke_tools(self, *args, **kwargs):  # pylint: disable=unused-argument
        return DummyCompleteResult(self._message)

    def complete(self, *args, **kwargs):  # pylint: disable=unused-argument
        return self._message


@pytest.fixture
def review_context(tmp_path):
    patch_text = """diff --git a/stdlib/foo.ets b/stdlib/foo.ets
--- a/stdlib/foo.ets
+++ b/stdlib/foo.ets
@@ -0,0 +1,2 @@
+export function Foo() {}
+const helper = 1
"""
    patch_file = tmp_path / "changes.patch"
    patch_file.write_text(patch_text, encoding="utf-8")

    rule_text = """# Rule

## Applies To
stdlib/.*\\.ets$
"""
    rule_file = tmp_path / "rule.md"
    rule_file.write_text(rule_text, encoding="utf-8")

    class StubCtx:
        command_path = "devagent review"
        invoked_subcommand = "review"
        def __init__(self, settings):
            self.obj = {"settings": settings, "silent_mode": True}
            self.params = {}
            self.meta = {}

    settings = Settings()
    settings.api_key = "dummy"
    ctx = StubCtx(settings)

    return ctx, patch_file, rule_file


def run_review_with_message(ctx, patch_file, rule_file, message):
    from ai_dev_agent.cli import review
    import ai_dev_agent.cli as cli_package

    dummy_client = DummyClient(message)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(review, "get_llm_client", lambda _ctx: dummy_client)
    monkeypatch.setattr(cli_package, "get_llm_client", lambda _ctx: dummy_client)

    try:
        return run_review(
            ctx,
            patch_file=str(patch_file),
            rule_file=str(rule_file),
            json_output=False,
            settings=ctx.obj["settings"],
        )
    finally:
        monkeypatch.undo()


def test_filter_rejects_out_of_scope(review_context):
    ctx, patch_file, rule_file = review_context

    message = {
        "violations": [
            {
                "file": "stdlib/foo.ets",
                "line": 1,
                "message": "Missing docs",
                "code_snippet": "export function Foo() {}",
            },
            {
                "file": "examples/utils.ets",
                "line": 1,
                "message": "Should be ignored",
                "code_snippet": "export function Bar() {}",
            },
        ],
        "summary": {
            "total_violations": 2,
            "files_reviewed": 2,
            "rule_name": "Test Rule",
        },
    }

    result = run_review_with_message(ctx, patch_file, rule_file, message)

    assert result["violations"] == []
    assert result["summary"]["total_violations"] == 0


def test_filter_allows_in_scope(review_context):
    ctx, patch_file, rule_file = review_context

    message = {
        "violations": [
            {
                "file": "stdlib/foo.ets",
                "line": 1,
                "message": "Missing docs",
                "code_snippet": "export function Foo() {}",
            }
        ],
        "summary": {
            "total_violations": 10,
            "files_reviewed": 10,
            "rule_name": "Test Rule",
        },
    }

    result = run_review_with_message(ctx, patch_file, rule_file, message)

    assert result["violations"] == message["violations"]
    assert result["summary"]["total_violations"] == 1
    assert result["summary"]["files_reviewed"] == 1


def test_review_returns_fallback_when_json_parse_fails(review_context, monkeypatch):
    ctx, patch_file, rule_file = review_context

    from ai_dev_agent.cli import review
    import ai_dev_agent.cli as cli_package

    # Patch LLM client factory to avoid real network calls
    monkeypatch.setattr(review, "get_llm_client", lambda _ctx: object())
    monkeypatch.setattr(cli_package, "get_llm_client", lambda _ctx: object())

    # Simulate executor raising the JSON enforcement error
    def fake_execute(*args, **kwargs):
        raise click.ClickException("Assistant response did not contain valid JSON matching the required schema.")

    monkeypatch.setattr(review, "_execute_react_assistant", fake_execute)

    result = run_review(
        ctx,
        patch_file=str(patch_file),
        rule_file=str(rule_file),
        json_output=True,
        settings=ctx.obj["settings"],
    )

    assert result["violations"] == []
    summary = result["summary"]
    assert summary["total_violations"] == 0
    assert summary["files_reviewed"] == 1
    assert summary["rule_name"] == rule_file.stem
