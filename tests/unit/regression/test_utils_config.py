"""Tests for the modern configuration utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_dev_agent.core.utils import config as config_module
from ai_dev_agent.core.utils.config import DEFAULT_MAX_ITERATIONS, Settings, load_settings


def test_find_config_in_parents_returns_closest(tmp_path: Path) -> None:
    """Walk parent directories until a config file is located."""
    (tmp_path / ".devagent.toml").write_text('provider = "deepseek"\n', encoding="utf-8")
    nested = tmp_path / "project" / "subdir"
    nested.mkdir(parents=True)

    found = config_module.find_config_in_parents(nested, ".devagent.toml")

    assert found is not None
    assert found.parent == tmp_path


def test_settings_ensure_state_dir_creates_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_state_dir should create the parent directory for the state file."""
    settings = Settings()
    state_path = tmp_path / "state" / "session.json"
    settings.state_file = state_path

    settings.ensure_state_dir()

    assert state_path.parent.is_dir()


def test_load_settings_from_explicit_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit configuration path should be loaded and normalized."""
    config_file = tmp_path / "devagent.toml"
    workspace_root = tmp_path / "workspace"
    state_file = tmp_path / "state" / "cache.json"
    config_file.write_text(
        f"""
        provider = "anthropic"
        model = "claude"
        workspace_root = "{workspace_root}"
        state_file = "{state_file}"
        auto_approve_plan = true
        """,
        encoding="utf-8",
    )

    settings = load_settings(config_file)

    assert settings.provider == "anthropic"
    assert settings.model == "claude"
    assert settings.auto_approve_plan is True
    assert settings.workspace_root == workspace_root.resolve()
    assert settings.state_file == state_file
    assert settings.state_file.parent.exists()


def test_load_settings_merges_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should override file configuration."""
    config_file = tmp_path / "devagent.toml"
    config_file.write_text('model = "default"\n', encoding="utf-8")

    monkeypatch.setenv("DEVAGENT_MODEL", "env-model")
    env_state = tmp_path / "env" / "state.json"
    monkeypatch.setenv("DEVAGENT_STATE_FILE", str(env_state))
    monkeypatch.setenv("DEVAGENT_SANDBOX_ALLOWLIST", "bash,python")

    settings = load_settings(config_file)

    assert settings.model == "env-model"
    assert settings.state_file == env_state
    assert settings.sandbox_allowlist == ("bash", "python")
    assert (tmp_path / "env").exists()


def test_default_max_iterations_matches_settings() -> None:
    """Ensure DEFAULT_MAX_ITERATIONS stays in sync with Settings default."""
    assert DEFAULT_MAX_ITERATIONS == Settings.__dataclass_fields__["max_iterations"].default
