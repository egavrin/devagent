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
    assert "provider:" not in combined.lower()
    assert "model:" not in combined.lower()
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

    assert len(messages) == 2
    contents = [msg.content or "" for msg in messages]
    assert any("DevAgent System Context" in entry for entry in contents)
    assert any("environment snapshot" in entry.lower() for entry in contents)


def test_build_system_messages_uses_markdown_templates(monkeypatch, tmp_workspace):
    settings = Settings()
    call_log: dict[str, list[tuple[str, dict[str, str] | None]]] = {}

    class DummyPromptLoader:
        def __init__(self, prompts_dir=None):
            call_log["init"] = [(str(prompts_dir), None)]

        def load_system_prompt(
            self, system_name: str = "base_context", context: dict | None = None
        ):
            call_log.setdefault("system", []).append((system_name, context))
            return "BASE_TEMPLATE"

        def render_prompt(self, prompt_path: str, context: dict | None = None) -> str:
            call_log.setdefault("render", []).append((prompt_path, context))
            if prompt_path == "system/react_loop.md":
                return "REACT_TEMPLATE"
            return f"PROMPT::{prompt_path}"

        def load_prompt(self, prompt_path: str) -> str:
            call_log.setdefault("load", []).append((prompt_path, None))
            return f"PROMPT::{prompt_path}"

    dummy_loader = DummyPromptLoader()
    monkeypatch.setattr(
        "ai_dev_agent.session.prompt_builder._PROMPT_LOADER",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        "ai_dev_agent.session.prompt_builder.PromptLoader",
        lambda prompts_dir=None: dummy_loader,
        raising=False,
    )

    messages = build_system_messages(
        iteration_cap=3,
        repository_language="python",
        include_react_guidance=True,
        workspace_root=tmp_workspace,
        settings=settings,
    )

    combined = "\n".join(msg.content or "" for msg in messages)
    assert "BASE_TEMPLATE" in combined
    assert "REACT_TEMPLATE" in combined
    assert "system" in call_log, "system prompt should be loaded via PromptLoader"
    system_name, context = call_log["system"][0]
    assert system_name == "base_context"
    assert context is not None
    assert context.get("iteration_cap") == "3"
    assert "python" in (context.get("language_hint") or "").lower()
