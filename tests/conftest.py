"""Shared pytest fixtures for CLI tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def cli_stub_runtime(monkeypatch):
    """Provide a stubbed CLI runtime that never calls real LLMs."""
    monkeypatch.setenv("DEVAGENT_API_KEY", "test-key")

    llm_client = MagicMock(name="llm_client")

    def fake_get_llm_client(ctx):
        ctx.obj["llm_client"] = llm_client
        return llm_client

    monkeypatch.setattr("ai_dev_agent.cli.utils.get_llm_client", fake_get_llm_client)

    executor = MagicMock(
        name="_execute_react_assistant",
        return_value={
            "result": MagicMock(name="RunResult"),
            "final_json": {
                "violations": [],
                "summary": {"rule_name": "stub-rule", "total_violations": 0, "files_reviewed": 0},
            },
            "printed_final": False,
        },
    )
    monkeypatch.setattr("ai_dev_agent.cli.react.executor._execute_react_assistant", executor)
    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.commands.query._execute_react_assistant",
        executor,
    )

    import importlib

    review_module = importlib.import_module("ai_dev_agent.cli.review")
    monkeypatch.setattr(review_module, "_execute_react_assistant", executor)

    return {
        "llm_client": llm_client,
        "executor": executor,
    }
