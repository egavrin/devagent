import os
from pathlib import Path

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.session.prompt_builder import build_system_messages


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def test_build_system_messages_with_provider_and_language(monkeypatch, tmp_workspace):
    instruction_file = tmp_workspace / "guidance.md"
    instruction_file.write_text("Follow zero-regression policy.", encoding="utf-8")
    monkeypatch.setenv("DEVAGENT_PROMPT_INSTRUCTIONS", str(instruction_file))

    settings = Settings()
    settings.provider = "anthropic"
    settings.model = "claude-3"
    settings.workspace_root = tmp_workspace

    messages = build_system_messages(
        iteration_cap=8,
        repository_language="python",
        include_react_guidance=True,
        extra_messages=["Remember to summarise findings."],
        provider=settings.provider,
        model=settings.model,
        workspace_root=tmp_workspace,
        settings=settings,
    )

    assert messages, "Expected at least one system message"
    combined = "\n".join(msg.content or "" for msg in messages)
    assert "Provider: anthropic" in combined or "Model: claude-3" in combined
    assert "Use symbols for Python code structure" in combined
    assert "Follow zero-regression policy." in combined
    assert "Remember to summarise findings." in combined


def test_build_system_messages_fallback_without_guidance(monkeypatch, tmp_workspace):
    monkeypatch.delenv("DEVAGENT_PROMPT_INSTRUCTIONS", raising=False)
    settings = Settings()
    settings.provider = ""
    messages = build_system_messages(
        include_react_guidance=False,
        provider=settings.provider,
        workspace_root=tmp_workspace,
        settings=settings,
    )

    assert len(messages) == 1
    assert "environment snapshot" in (messages[0].content or "").lower()
