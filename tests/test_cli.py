from pathlib import Path

from click.testing import CliRunner

import ai_dev_agent.cli as cli_module
from ai_dev_agent.cli import infer_task_files, cli
from ai_dev_agent.cli.router import IntentDecision
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session import SessionManager, build_system_messages


def test_infer_task_files_from_commands(tmp_path: Path) -> None:
    target = tmp_path / "ai_dev_agent" / "core.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('hello')\n", encoding="utf-8")

    task = {
        "commands": [
            "devagent react run --files ai_dev_agent/core.py docs/guide.md",
            "pytest",
        ]
    }

    inferred = infer_task_files(task, tmp_path)
    assert inferred == ["ai_dev_agent/core.py"]


def test_infer_task_files_from_deliverables(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs" / "overview.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("overview", encoding="utf-8")

    task = {
        "commands": [],
        "deliverables": ["docs/overview.md", "docs/missing.md"],
    }

    inferred = infer_task_files(task, tmp_path)
    assert inferred == ["docs/overview.md"]


def test_infer_task_files_from_keywords(tmp_path: Path) -> None:
    file_path = tmp_path / "ai_dev_agent" / "core.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("calc = 1\n", encoding="utf-8")

    task = {
        "title": "Remove obsolete core module files",
        "description": "Delete unused core module functionality from the project",
    }

    inferred = infer_task_files(task, tmp_path)
    assert inferred == ["ai_dev_agent/core.py"]


def test_infer_task_files_from_path_hints(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs" / "guide.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("guide", encoding="utf-8")

    task = {
        "title": "Update guides",
        "description": "Ensure docs/guide.md reflects the new workflow",
    }

    inferred = infer_task_files(task, tmp_path)
    assert inferred == ["docs/guide.md"]


def test_cli_rejects_deprecated_list_directory_tool(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "sample.txt").write_text("hello", encoding="utf-8")

    settings = Settings()
    settings.workspace_root = repo_root
    settings.state_file = repo_root / ".devagent" / "state.json"
    settings.ensure_state_dir()
    settings.api_key = "test"
    monkeypatch.setattr(cli_module, "load_settings", lambda path=None: settings)

    mock_client = object()
    monkeypatch.setattr(cli_module, "get_llm_client", lambda ctx: mock_client)

    class DummyRouter:
        def __init__(self, client, settings_obj, **kwargs) -> None:
            self.client = client
            self.settings = settings_obj
            self.extra = kwargs
            self.session_id = "dummy-router"

        def route(self, prompt: str) -> IntentDecision:
            assert "дир" in prompt
            return IntentDecision(tool="list_directory", arguments={"path": "."}, rationale="List repo root")

    monkeypatch.setattr(cli_module, "IntentRouter", DummyRouter)
    monkeypatch.chdir(repo_root)

    runner = CliRunner()
    result = runner.invoke(cli, ["покажи", "содержимое", "директории"])

    assert result.exit_code == 1
    assert "Intent tool 'list_directory' is not supported yet." in result.output


def test_query_command_invokes_router(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    settings = Settings()
    settings.workspace_root = repo_root
    settings.state_file = repo_root / ".devagent" / "state.json"
    settings.ensure_state_dir()
    settings.api_key = "test"
    monkeypatch.setattr(cli_module, "load_settings", lambda path=None: settings)

    captures = {}

    class DummyRouter:
        def __init__(self, client, settings_obj, **kwargs) -> None:
            self.client = client
            self.settings = settings_obj
            self.extra = kwargs
            self.session_id = "dummy-router"

        def route(self, prompt: str) -> IntentDecision:
            captures.setdefault("prompt", prompt)
            return IntentDecision(tool=None, arguments={"text": "ok"})

    monkeypatch.setattr(cli_module, "IntentRouter", DummyRouter)
    monkeypatch.setattr(cli_module, "get_llm_client", lambda ctx: object())
    monkeypatch.chdir(repo_root)

    runner = CliRunner()
    result = runner.invoke(cli, ["query", "hello", "world"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "ok"
    assert captures["prompt"] == "hello world"
