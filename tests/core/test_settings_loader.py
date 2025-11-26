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


def test_load_settings_provider_section_api_key(tmp_path, monkeypatch):
    """Test that API key is loaded from [providers.<provider>] section."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "openai"
        model = "gpt-4o"

        [providers.openai]
        api_key = "sk-test-openai-key"
        """,
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.provider == "openai"
    assert settings.model == "gpt-4o"
    assert settings.api_key == "sk-test-openai-key"


def test_load_settings_provider_section_base_url(tmp_path, monkeypatch):
    """Test that base_url is loaded from provider section when not at top level."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "custom"
        model = "my-model"

        [providers.custom]
        api_key = "sk-custom-key"
        base_url = "https://custom.api.com/v1"
        """,
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.provider == "custom"
    assert settings.api_key == "sk-custom-key"
    assert settings.base_url == "https://custom.api.com/v1"


def test_load_settings_top_level_api_key_takes_precedence(tmp_path, monkeypatch):
    """Test that top-level api_key takes precedence over provider section."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "openai"
        api_key = "sk-top-level-key"

        [providers.openai]
        api_key = "sk-provider-section-key"
        """,
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.api_key == "sk-top-level-key"


def test_load_settings_multiple_providers(tmp_path, monkeypatch):
    """Test config with multiple provider sections uses the active provider."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "anthropic"
        model = "claude-3-5-sonnet-20241022"

        [providers.openai]
        api_key = "sk-openai-key"

        [providers.anthropic]
        api_key = "sk-anthropic-key"

        [providers.deepseek]
        api_key = "sk-deepseek-key"
        """,
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.provider == "anthropic"
    assert settings.api_key == "sk-anthropic-key"


def test_load_settings_default_provider_deepseek(tmp_path, monkeypatch):
    """Test that default provider is deepseek when not specified."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        [providers.deepseek]
        api_key = "sk-deepseek-default"
        """,
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.provider == "deepseek"
    assert settings.api_key == "sk-deepseek-default"


def test_load_settings_env_overrides_provider_section(tmp_path, monkeypatch):
    """Test that environment variable overrides provider section api_key."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "openai"

        [providers.openai]
        api_key = "sk-from-config"
        """,
    )
    monkeypatch.setenv("DEVAGENT_API_KEY", "sk-from-env")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.api_key == "sk-from-env"


def test_load_settings_integer_env_fields(tmp_path, monkeypatch):
    """Test that integer environment variables are parsed correctly."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "deepseek"
        """,
    )
    monkeypatch.setenv("DEVAGENT_STEPS_BUDGET", "25")
    monkeypatch.setenv("DEVAGENT_MAX_ITERATIONS", "50")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.steps_budget == 25
    assert settings.max_iterations == 50


def test_load_settings_json_env_decoding_error(tmp_path, monkeypatch):
    """Test that invalid JSON in env vars defaults to empty dict."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "deepseek"
        """,
    )
    monkeypatch.setenv("DEVAGENT_PROVIDER_CONFIG", "not-valid-json")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.provider_config == {}


def test_load_settings_shell_session_limits_from_env(tmp_path, monkeypatch):
    """Test shell session limit environment variables."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "deepseek"
        """,
    )
    monkeypatch.setenv("DEVAGENT_SHELL_SESSION_CPU_TIME_LIMIT", "120")
    monkeypatch.setenv("DEVAGENT_SHELL_SESSION_MEMORY_LIMIT_MB", "512")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.shell_session_cpu_time_limit == 120
    assert settings.shell_session_memory_limit_mb == 512


def test_load_settings_sandbox_allowlist_from_toml_list(tmp_path, monkeypatch):
    """Test that sandbox_allowlist loads from TOML list format."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "deepseek"
        sandbox_allowlist = ["cat", "ls", "grep"]
        """,
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.sandbox_allowlist == ("cat", "ls", "grep")


def test_load_settings_provider_only_from_toml_list(tmp_path, monkeypatch):
    """Test that provider_only loads from TOML list format."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "deepseek"
        provider_only = ["openai", "anthropic"]
        """,
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.provider_only == ("openai", "anthropic")


def test_find_config_file_path(tmp_path, monkeypatch):
    """Test find_config_in_parents with a file path input."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = write_config(
        project_root / ".devagent.toml",
        """
        provider = "anthropic"
        """,
    )
    # Create a file inside the project
    test_file = project_root / "test.py"
    test_file.write_text("# test")
    monkeypatch.chdir(project_root)

    # Pass a file path instead of a directory
    found = find_config_in_parents(test_file, (".devagent.toml",))

    assert found is not None
    assert found == config.resolve()


def test_load_settings_provider_config_from_toml_string(tmp_path, monkeypatch):
    """Test provider_config from TOML when it's a string (JSON encoded)."""
    config = write_config(
        tmp_path / ".devagent.toml",
        """
        provider = "deepseek"
        provider_config = '{"key": "value"}'
        """,
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config)

    assert settings.provider_config == {"key": "value"}
