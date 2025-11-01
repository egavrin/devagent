"""Tests for provider-specific prompt helpers."""

from __future__ import annotations

from ai_dev_agent.prompts.provider_prompts import PromptContext, ProviderPrompts


def _make_context(**overrides):
    base = {
        "working_directory": "/workspace",
        "is_git_repo": True,
        "platform": "darwin",
        "language": "python",
        "framework": "fastapi",
        "project_structure": "- app/\n- tests/",
        "custom_instructions": "Follow PEP 8.",
        "phase": "exploration",
        "task_description": "Add request validation",
    }
    base.update(overrides)
    return PromptContext(**base)


def test_anthropic_prompt_includes_environment_and_language():
    context = _make_context()

    prompt = ProviderPrompts.get_system_prompt("anthropic", context)

    assert "You are Claude" in prompt
    assert "Working directory: /workspace" in prompt
    assert "Language: python" in prompt
    assert "Follow PEP 8." in prompt
    assert "Add request validation" in prompt


def test_openai_prompt_defaults_and_phase_guidance():
    context = _make_context(phase="synthesis", language=None, custom_instructions=None)

    prompt = ProviderPrompts.get_system_prompt("openai", context)

    assert "You are an expert software engineer assistant." in prompt
    assert "# Current Phase: SYNTHESIS" in prompt
    assert "# Language" not in prompt


def test_tool_reminder_investigation_phase():
    reminder = ProviderPrompts.get_tool_reminder("investigation", ["read", "search"])

    assert "# Available Tools for Investigation" in reminder
    assert "- read" in reminder
    assert "- search" in reminder
