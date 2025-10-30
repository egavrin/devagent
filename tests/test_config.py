from pathlib import Path

from ai_dev_agent.core.utils.config import load_settings


def test_env_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text("provider = 'deepseek'\nmodel = 'deepseek-lite'\n")
    monkeypatch.setenv("DEVAGENT_API_KEY", "abc123")
    monkeypatch.setenv("DEVAGENT_AUTO_APPROVE_PLAN", "true")
    settings = load_settings(config_path)
    assert settings.api_key == "abc123"
    assert settings.auto_approve_plan is True
    assert isinstance(settings.state_file, Path)


def test_provider_customization_from_config(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
provider = "openrouter"
provider_only = ["Cerebras"]
"""
        "[provider_config]\n"
        'priority = ["Cerebras"]\n'
        "[request_headers]\n"
        'HTTP-Referer = "https://example.com"\n'
    )

    settings = load_settings(config_path)

    assert settings.provider_only == ("Cerebras",)
    assert settings.provider_config == {"priority": ["Cerebras"]}
    assert settings.request_headers == {"HTTP-Referer": "https://example.com"}


def test_project_config_discovered_in_parent_directory(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    nested_dir = project_root / "src" / "module"
    nested_dir.mkdir(parents=True)

    config_path = project_root / ".devagent.toml"
    config_path.write_text("model = 'parent-tree-model'\n")

    monkeypatch.chdir(nested_dir)

    settings = load_settings()

    assert settings.model == "parent-tree-model"


def test_project_config_without_dot_discovered_in_parent_directory(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    nested_dir = project_root / "src" / "module"
    nested_dir.mkdir(parents=True)

    config_path = project_root / "devagent.toml"
    config_path.write_text("model = 'parent-tree-no-dot'\n")

    monkeypatch.chdir(nested_dir)

    settings = load_settings()

    assert settings.model == "parent-tree-no-dot"
