"""Tests for the Settings loader."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from ai_dev_agent.core.utils.config import find_config_in_parents, load_settings


def write_config(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_load_settings_prefers_explicit_path(tmp_path, monkeypatch):
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "openrouter"
        log_level = "DEBUG"
        state_file = "state/dev.json"
        """,
    )
    monkeypatch.setenv("DEVAGENT_API_KEY", "token-123")
    monkeypatch.setenv("DEVAGENT_AUTO_APPROVE_PLAN", "true")

    settings = load_settings(config)

    assert settings.provider == "openrouter"
    assert settings.log_level == "DEBUG"
    assert settings.api_key == "token-123"
    assert settings.auto_approve_plan is True
    assert settings.state_file == Path("state/dev.json")


def test_load_settings_searches_parent_directories(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    nested = project_root / "src" / "app"
    nested.mkdir(parents=True)
    config = write_config(
        project_root / ".devagent.toml",
        """
        provider = "anthropic"
        steps_budget = 5
        """,
    )
    monkeypatch.chdir(nested)

    settings = load_settings()

    assert settings.provider == "anthropic"
    assert settings.steps_budget == 5
    expected = config.resolve()
    assert find_config_in_parents(Path.cwd(), (".devagent.toml",)) == expected


def test_load_settings_environment_json(monkeypatch, tmp_path):
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "openrouter"
        """,
    )
    monkeypatch.setenv("DEVAGENT_PROVIDER_CONFIG", '{"priority": ["deepseek"]}')
    monkeypatch.setenv("DEVAGENT_PROVIDER_ONLY", "deepseek,openai ")
    settings = load_settings(config)

    assert settings.provider_config == {"priority": ["deepseek"]}
    assert settings.provider_only == ("deepseek", "openai")


def test_load_settings_handles_missing_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Ensure no config file exists
    for name in (".devagent.toml", "devagent.toml"):
        path = tmp_path / name
        if path.exists():
            path.unlink()

    settings = load_settings()

    assert settings.provider == "deepseek"
    assert settings.workspace_root == Path.cwd().resolve()
