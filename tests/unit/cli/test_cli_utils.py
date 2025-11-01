"""Tests for CLI utilities module."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import click
import pytest

from ai_dev_agent.cli.context_enhancer import ContextEnhancer
from ai_dev_agent.cli.utils import (
    _build_context,
    _build_context_pruning_config_from_settings,
    _collect_project_structure_outline,
    _detect_repository_language,
    _export_structure_hints_state,
    _get_repo_context_enhancer,
    _get_structure_hints_state,
    _invoke_registry_tool,
    _make_tool_context,
    _merge_structure_hints_state,
    _normalize_argument_list,
    _record_invocation,
    _resolve_repo_path,
    _update_files_discovered,
    build_system_context,
    get_llm_client,
    infer_task_files,
    resolve_prompt_input,
    update_task_state,
)
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.state import InMemoryStateStore
from ai_dev_agent.providers.llm import DEEPSEEK_DEFAULT_BASE_URL


def test_build_system_context():
    """Test build_system_context returns expected structure."""
    ctx = build_system_context()

    # Check required keys exist
    assert "os" in ctx
    assert "os_friendly" in ctx
    assert "os_version" in ctx
    assert "architecture" in ctx
    assert "shell" in ctx
    assert "cwd" in ctx
    assert "home_dir" in ctx
    assert "python_version" in ctx
    assert "shell_type" in ctx
    assert "path_separator" in ctx
    assert "command_separator" in ctx
    assert "null_device" in ctx
    assert "temp_dir" in ctx
    assert "available_tools" in ctx
    assert "command_mappings" in ctx
    assert "platform_examples" in ctx

    # Check types
    assert isinstance(ctx["os"], str)
    assert isinstance(ctx["os_friendly"], str)
    assert isinstance(ctx["cwd"], str)
    assert isinstance(ctx["available_tools"], list)
    assert isinstance(ctx["command_mappings"], dict)

    # Check platform-specific values
    if platform.system() == "Darwin":
        assert ctx["os"] == "Darwin"
        assert ctx["os_friendly"] == "macOS"
        assert ctx["shell_type"] == "unix"
        assert ctx["path_separator"] == "/"
        assert ctx["command_separator"] == "&&"
        assert ctx["null_device"] == "/dev/null"
    elif platform.system() == "Linux":
        assert ctx["os"] == "Linux"
        assert ctx["os_friendly"] == "Linux"
        assert ctx["shell_type"] == "unix"
    elif platform.system() == "Windows":
        assert ctx["os"] == "Windows"
        assert ctx["shell_type"] == "windows"
        assert ctx["path_separator"] == "\\"


def test_build_system_context_command_mappings():
    """Test command mappings are present."""
    ctx = build_system_context()
    mappings = ctx["command_mappings"]

    # Check common commands exist
    assert "list_files" in mappings
    assert "find_files" in mappings
    assert "copy" in mappings
    assert "move" in mappings
    assert "delete" in mappings
    assert "open_file" in mappings

    # Values should be strings
    for key, value in mappings.items():
        assert isinstance(value, str)
        assert len(value) > 0


def test_build_system_context_available_tools():
    """Test available tools detection."""
    ctx = build_system_context()
    tools = ctx["available_tools"]

    # Should be a list (may be empty if no tools installed)
    assert isinstance(tools, list)

    # If git is in the list, it should actually be available
    if "git" in tools:
        import shutil

        assert shutil.which("git") is not None


class TestResolveRepoPath:
    """Test repository path resolution."""

    def test_resolve_repo_path_none(self):
        """Test resolving None path returns current directory."""
        result = _resolve_repo_path(None)

        # Should return current working directory
        assert result == Path.cwd()

    def test_resolve_repo_path_relative(self):
        """Test resolving relative path."""
        result = _resolve_repo_path("./subdir")

        assert result.name == "subdir"
        assert result.is_absolute()

    def test_resolve_repo_path_absolute(self):
        """Test resolving absolute path raises error for paths outside repo."""
        # Absolute paths outside the repo should raise an error
        with pytest.raises(click.ClickException) as exc_info:
            _resolve_repo_path("/home/user/project")

        assert "escapes the repository root" in str(exc_info.value)


class TestDetectRepositoryLanguage:
    """Test repository language detection."""

    def test_detect_python_repository(self, tmp_path):
        """Test detecting Python repository."""
        (tmp_path / "setup.py").touch()
        (tmp_path / "requirements.txt").touch()
        (tmp_path / "main.py").touch()

        language, count = _detect_repository_language(tmp_path)

        assert language == "python"
        assert count is not None

    def test_detect_javascript_repository(self, tmp_path):
        """Test detecting JavaScript repository."""
        (tmp_path / "package.json").write_text('{"name": "test"}')
        (tmp_path / "index.js").touch()

        language, count = _detect_repository_language(tmp_path)

        assert language == "javascript"
        assert count is not None

    def test_detect_typescript_repository(self, tmp_path):
        """Test detecting TypeScript repository."""
        (tmp_path / "tsconfig.json").write_text("{}")
        (tmp_path / "package.json").write_text('{"name": "test"}')
        (tmp_path / "index.ts").touch()

        language, count = _detect_repository_language(tmp_path)

        assert language == "typescript"
        assert count is not None

    def test_detect_unknown_repository(self, tmp_path):
        """Test detecting unknown repository type."""
        (tmp_path / "random.txt").touch()

        language, _count = _detect_repository_language(tmp_path)

        # With only non-code files, it might return None or a generic type
        assert language in ["unknown", None, "text"]

    def test_detect_language_prefers_context_enhancer(self, monkeypatch, tmp_path):
        """Detection should fall back to ContextEnhancer listings when rglob is unavailable."""

        captured: dict[str, Any] = {}

        class StubEnhancer:
            def __init__(self, workspace, settings=None):
                captured["workspace"] = workspace
                captured["settings"] = settings

            def list_repository_files(self, limit=None, use_cache=True):
                captured["limit"] = limit
                captured["use_cache"] = use_cache
                return ["src/main.py", "docs/readme.md"]

        monkeypatch.setattr(
            "ai_dev_agent.cli.context_enhancer.ContextEnhancer",
            StubEnhancer,
        )
        monkeypatch.setattr(
            "ai_dev_agent.cli.utils.ContextEnhancer",
            StubEnhancer,
            raising=False,
        )
        monkeypatch.setattr(
            "ai_dev_agent.cli.utils.extract_keywords",
            lambda text: {"api"},
        )

        def fail_rglob(self, pattern="*"):
            raise OSError("Filesystem traversal disabled")

        monkeypatch.setattr(Path, "rglob", fail_rglob)

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "readme.md").write_text("hello\n", encoding="utf-8")

        settings = Settings()
        language, total = _detect_repository_language(tmp_path, settings=settings)

        assert language == "python"
        assert total == 2
        assert captured["limit"] == 400
        assert captured["workspace"] == tmp_path.resolve()

    def test_get_repo_context_enhancer_caches_instances(self, monkeypatch, tmp_path):
        """Enhancer retrieval should reuse cache per repo and settings."""
        monkeypatch.setattr(
            "ai_dev_agent.cli.utils._CLI_CONTEXT_ENHANCERS",
            {},
            raising=False,
        )

        first = _get_repo_context_enhancer(tmp_path)
        second = _get_repo_context_enhancer(tmp_path)
        assert first is second

        settings = Settings()
        third = _get_repo_context_enhancer(tmp_path, settings=settings)
        assert third is not first
        assert third.settings is settings

    def test_detect_language_handles_rglob_edge_cases(self, monkeypatch, tmp_path):
        """Fallback filesystem scan should skip directories and handle ValueError."""
        (tmp_path / "src").mkdir()
        main_file = tmp_path / "src" / "main.py"
        main_file.write_text("print('hi')\n", encoding="utf-8")
        external = tmp_path.parent / "external.txt"
        external.write_text("noop\n", encoding="utf-8")

        class StubEnhancer:
            _DEFAULT_IGNORE_SUBSTRINGS: tuple[str, ...] = tuple(
                ContextEnhancer._DEFAULT_IGNORE_SUBSTRINGS
            )

            def __init__(self, workspace, settings=None):
                pass

            def list_repository_files(self, limit=None, use_cache=True):
                return []

        monkeypatch.setattr(
            "ai_dev_agent.cli.utils.ContextEnhancer",
            StubEnhancer,
            raising=False,
        )

        def fake_rglob(self, pattern="*"):
            assert self == tmp_path.resolve()
            git_dir = tmp_path / ".git"
            git_dir.mkdir(exist_ok=True)
            config_path = git_dir / "config"
            config_path.write_text("config\n", encoding="utf-8")
            yield tmp_path / "src"
            yield main_file
            yield config_path
            yield external

        monkeypatch.setattr(Path, "rglob", fake_rglob, raising=False)

        language, total = _detect_repository_language(tmp_path, max_files=10)

        assert language == "python"
        assert total == 2

    def test_detect_language_respects_max_files(self, monkeypatch, tmp_path):
        """Max file limit should stop scanning early."""
        first = tmp_path / "first.py"
        first.write_text("print('first')\n", encoding="utf-8")
        second = tmp_path / "second.py"
        second.write_text("print('second')\n", encoding="utf-8")

        monkeypatch.setattr(
            "ai_dev_agent.cli.utils._get_repo_context_enhancer",
            lambda repo_root, settings=None: SimpleNamespace(
                list_repository_files=lambda limit=1: []
            ),
        )

        def fake_rglob(self, pattern="*"):
            yield first
            yield second
            yield second

        monkeypatch.setattr(Path, "rglob", fake_rglob, raising=False)

        language, total = _detect_repository_language(tmp_path, max_files=1)

        assert language == "python"
        assert total == 1


class TestInferTaskFiles:
    """Test task file inference."""

    def test_infer_task_files_with_explicit_paths(self, tmp_path):
        """Test inferring files from explicit paths in task."""
        # Create the files
        (tmp_path / "src").mkdir(exist_ok=True)
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "tests").mkdir(exist_ok=True)
        (tmp_path / "tests" / "test_main.py").touch()

        task = {"description": "Fix bug in src/main.py and tests/test_main.py"}

        files = infer_task_files(task, tmp_path)

        assert "src/main.py" in files
        assert "tests/test_main.py" in files

    def test_infer_task_files_with_keywords(self, tmp_path):
        """Test inferring files from keywords."""
        # Create test files
        (tmp_path / "auth.py").touch()
        (tmp_path / "authentication.py").touch()
        (tmp_path / "login.py").touch()
        (tmp_path / "other.py").touch()

        task = {"description": "Fix authentication issue in the login system"}

        files = infer_task_files(task, tmp_path)

        # Should find files matching authentication/login keywords
        assert any("auth" in f.lower() or "login" in f.lower() for f in files)

    def test_infer_task_files_no_matches(self, tmp_path):
        """Test inferring files with no matches."""
        (tmp_path / "random.txt").touch()

        task = {"description": "Do something abstract with no file mentions"}

        files = infer_task_files(task, tmp_path)

        assert files == []

    def test_infer_task_files_uses_context_enhancer_keywords(self, monkeypatch, tmp_path):
        """Keyword inference should consult ContextEnhancer listings when filesystem scan blocked."""

        captured: dict[str, Any] = {}

        class StubEnhancer:
            def __init__(self, workspace, settings=None):
                captured["workspace"] = workspace
                captured["settings"] = settings

            def list_repository_files(self, limit=None, use_cache=True):
                captured["limit"] = limit
                captured["use_cache"] = use_cache
                return ["src/api/routes.py", "README.md"]

        monkeypatch.setattr(
            "ai_dev_agent.cli.context_enhancer.ContextEnhancer",
            StubEnhancer,
        )
        monkeypatch.setattr(
            "ai_dev_agent.cli.utils.ContextEnhancer",
            StubEnhancer,
            raising=False,
        )

        def fail_rglob(self, pattern="*"):
            raise OSError("Traversal disabled")

        monkeypatch.setattr(Path, "rglob", fail_rglob)

        (tmp_path / "src" / "api").mkdir(parents=True)
        target = tmp_path / "src" / "api" / "routes.py"
        target.write_text("ROUTES = []\n", encoding="utf-8")

        task = {"description": "Refine API routes to support versioning"}

        files = infer_task_files(task, tmp_path)

        assert files == ["src/api/routes.py"]
        assert "limit" in captured
        assert captured["workspace"] == tmp_path.resolve()

    def test_infer_task_files_skips_ignored_fragments(self, tmp_path):
        """Repository scans should ignore paths containing default skip substrings."""
        (tmp_path / "src").mkdir()
        good = tmp_path / "src" / "deploy.py"
        good.write_text("print('deploy')\n", encoding="utf-8")
        git_hook = tmp_path / ".git" / "hooks"
        git_hook.mkdir(parents=True)
        (git_hook / "post_deploy.sh").write_text("#!/bin/bash\n", encoding="utf-8")

        task = {"description": "Review deploy hooks"}

        files = infer_task_files(task, tmp_path)

        assert "src/deploy.py" in files
        assert all(".git" not in path for path in files)

    def test_infer_task_files_ignores_outside_repo_paths(self, tmp_path):
        """Paths resolving outside of repo root should be ignored."""
        external = tmp_path.parent / "external.py"
        external.write_text("print('ext')\n", encoding="utf-8")

        task = {
            "commands": [f"devagent --files {external}"],
            "description": "Touch external file",
        }

        files = infer_task_files(task, tmp_path)

        assert files == []

    def test_infer_task_files_uses_keyword_candidates(self, monkeypatch, tmp_path):
        """Keyword-derived scans should consider repository candidates."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        target = reports_dir / "monthly.md"
        target.write_text("# report\n", encoding="utf-8")

        monkeypatch.setattr(
            "ai_dev_agent.cli.utils.extract_keywords",
            lambda text: {"reports"},
        )
        monkeypatch.setattr(
            "ai_dev_agent.cli.utils._get_repo_context_enhancer",
            lambda repo_root, settings=None: SimpleNamespace(
                list_repository_files=lambda limit=800: []
            ),
        )

        def fake_rglob(self, pattern="*"):
            yield reports_dir
            yield target

        monkeypatch.setattr(Path, "rglob", fake_rglob, raising=False)

        task = {"description": "Analyze quarterly reports"}

        files = infer_task_files(task, tmp_path)

        assert files == ["reports/monthly.md"]


class TestNormalizeArgumentList:
    """Test argument list normalization."""

    def test_normalize_argument_list_plural(self):
        """Test normalizing with plural key."""
        arguments = {"files": ["file1.txt", "file2.txt"]}
        result = _normalize_argument_list(arguments, plural_key="files")
        assert result == ["file1.txt", "file2.txt"]

    def test_normalize_argument_list_singular(self):
        """Test normalizing with singular key."""
        arguments = {"file": "file1.txt"}
        result = _normalize_argument_list(arguments, plural_key="files", singular_key="file")
        assert result == ["file1.txt"]

    def test_normalize_argument_list_both_keys(self):
        """Test normalizing with both singular and plural keys."""
        # When both keys exist, the function may only use the plural key
        arguments = {"file": "file1.txt", "files": ["file2.txt", "file3.txt"]}
        result = _normalize_argument_list(arguments, plural_key="files", singular_key="file")
        # Function prioritizes plural key when both are present
        assert "file2.txt" in result
        assert "file3.txt" in result
        assert len(result) >= 2

    def test_normalize_argument_list_empty(self):
        """Test normalizing empty arguments."""
        arguments = {}
        result = _normalize_argument_list(arguments, plural_key="files")
        assert result == []


class TestBuildContext:
    """Test context building."""

    def test_build_context_basic(self):
        """Test building basic context from settings."""
        settings = Settings()
        settings.max_context = 1000

        context = _build_context(settings)

        assert "settings" in context
        assert context["settings"] == settings
        assert "state" in context
        assert "llm_client" in context
        assert "approval_policy" in context
        assert context["llm_client"] is None


class TestProjectStructureOutline:
    """Test project structure outline collection."""

    def test_collect_project_structure_outline(self, tmp_path):
        """Test collecting project structure outline."""
        # Create test structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass")
        (tmp_path / "README.md").write_text("# Project")

        with patch("ai_dev_agent.cli.utils.generate_repo_outline") as mock_outline:
            mock_outline.return_value = "src/\n  main.py\ntests/\n  test_main.py\nREADME.md"

            outline = _collect_project_structure_outline(tmp_path, max_entries=100)

            assert "src/" in outline
            assert "tests/" in outline
            assert "README.md" in outline

    def test_collect_project_structure_outline_empty_returns_none(self, tmp_path):
        """Empty outlines should return None rather than an empty string."""
        with patch("ai_dev_agent.cli.utils.generate_repo_outline", return_value=""):
            result = _collect_project_structure_outline(tmp_path)

        assert result is None


class TestMakeToolContext:
    """Test tool context creation."""

    def test_make_tool_context_basic(self):
        """Test creating basic tool context."""
        ctx = click.Context(click.Command("test"))
        ctx.obj = {
            "cwd": Path.cwd(),  # Use actual current directory
            "settings": Settings(),
            "approval_policy": MagicMock(),
        }

        with (
            patch("ai_dev_agent.cli.utils.ApprovalManager"),
            patch("ai_dev_agent.cli.utils.load_devagent_yaml", return_value={"cfg": 1}),
        ):
            tool_ctx = _make_tool_context(ctx)

            # ToolContext uses the actual current directory
            assert tool_ctx.repo_root == Path.cwd()
            assert hasattr(tool_ctx, "settings")
            assert tool_ctx.settings == ctx.obj["settings"]

    def test_make_tool_context_with_sandbox_and_shell_manager(self, monkeypatch):
        """Sandbox flag and shell manager metadata should populate extra fields."""
        ctx = click.Context(click.Command("test"))
        settings = Settings()
        settings.workspace_root = Path.cwd()
        ctx.obj = {
            "settings": settings,
            "_shell_session_manager": "manager",
            "_shell_session_id": "session-xyz",
        }

        monkeypatch.setattr(
            "ai_dev_agent.cli.utils._create_sandbox",
            lambda s: "sandbox",
        )

        tool_ctx = _make_tool_context(ctx, with_sandbox=True)

        assert tool_ctx.sandbox == "sandbox"
        assert tool_ctx.extra["shell_session_manager"] == "manager"
        assert tool_ctx.extra["shell_session_id"] == "session-xyz"


class TestGetLLMClient:
    """Test LLM client creation."""

    @patch("ai_dev_agent.cli.utils.BudgetedLLMClient")
    @patch("ai_dev_agent.cli.utils.create_client")
    def test_get_llm_client_new(self, mock_create, mock_budgeted):
        """Test getting new LLM client."""
        ctx = click.Context(click.Command("test"))
        settings = Settings()
        # Need to set an API key
        settings.api_key = "test-api-key"
        ctx.obj = {"settings": settings}

        mock_raw_client = MagicMock()
        mock_create.return_value = mock_raw_client

        mock_wrapped_client = MagicMock()
        mock_budgeted.return_value = mock_wrapped_client

        client = get_llm_client(ctx)

        # Should return the BudgetedLLMClient wrapper
        assert client == mock_wrapped_client
        assert ctx.obj["llm_client"] == mock_wrapped_client
        mock_create.assert_called_once()
        mock_budgeted.assert_called_once()

    def test_get_llm_client_cached(self):
        """Test getting cached LLM client."""
        ctx = click.Context(click.Command("test"))
        mock_client = MagicMock()
        ctx.obj = {"llm_client": mock_client}

        client = get_llm_client(ctx)

        assert client == mock_client

    def test_get_llm_client_configures_provider_overrides(self, monkeypatch):
        ctx = click.Context(click.Command("test"))
        settings = Settings()
        settings.api_key = "key"
        settings.provider = "OpenRouter"
        settings.model = "gpt"
        settings.base_url = DEEPSEEK_DEFAULT_BASE_URL
        settings.provider_only = ["anthropic"]
        settings.provider_config = {"anthropic": {"model": "claude"}}
        settings.request_headers = {"X-Test": "1"}
        settings.context_pruner_max_total_tokens = 500
        settings.context_pruner_trigger_ratio = 0.6
        settings.context_pruner_keep_recent_messages = 4
        settings.context_pruner_summary_max_chars = 120
        settings.context_pruner_max_event_history = 8
        settings.max_tool_messages_kept = 12
        settings.disable_context_pruner = True

        ctx.obj = {"settings": settings}

        captured_client_kwargs: dict[str, Any] = {}

        class DummyClient:
            def __init__(self):
                self.timeout = None
                self.retry = None

            def configure_timeout(self, value):
                self.timeout = value

            def configure_retry(self, config):
                self.retry = config

        def fake_create_client(**kwargs):
            captured_client_kwargs.update(kwargs)
            return DummyClient()

        monkeypatch.setattr("ai_dev_agent.cli.utils.create_client", fake_create_client)

        captured_budget: dict[str, Any] = {}

        def fake_budgeted(client, budget_config, disabled):
            captured_budget["client"] = client
            captured_budget["budget"] = budget_config
            captured_budget["disabled"] = disabled
            return {"wrapped": client}

        monkeypatch.setattr("ai_dev_agent.cli.utils.BudgetedLLMClient", fake_budgeted)

        pruning_config = object()
        monkeypatch.setattr(
            "ai_dev_agent.cli.utils._build_context_pruning_config_from_settings",
            lambda s: pruning_config,
        )

        monkeypatch.setattr(
            "ai_dev_agent.cli.utils.config_from_settings",
            lambda s: "budget-config",
        )

        class StubSessionManager:
            def __init__(self):
                self.args = None

            def configure_context_service(self, config, summarizer):
                self.args = (config, summarizer)

        session_manager = StubSessionManager()
        monkeypatch.setattr(
            "ai_dev_agent.cli.utils.SessionManager.get_instance",
            lambda: session_manager,
        )

        monkeypatch.setattr(
            "ai_dev_agent.cli.utils.LLMConversationSummarizer",
            lambda client: {"summarizer": client},
        )

        monkeypatch.setitem(
            sys.modules,
            "ai_dev_agent.session.enhanced_summarizer_wrapper",
            SimpleNamespace(create_enhanced_summarizer=lambda client: None),
        )

        client = get_llm_client(ctx)

        assert captured_client_kwargs["provider"] == "OpenRouter"
        assert captured_client_kwargs["base_url"] is None
        assert captured_client_kwargs["provider_only"] == tuple(settings.provider_only)
        assert "provider_config" in captured_client_kwargs
        assert "default_headers" in captured_client_kwargs

        raw_client = captured_budget["client"]
        assert isinstance(raw_client, DummyClient)
        assert raw_client.timeout == 120.0
        assert raw_client.retry is not None
        assert captured_budget["budget"] == "budget-config"
        assert captured_budget["disabled"] is True
        assert ctx.obj["_raw_llm_client"] is raw_client
        assert ctx.obj["llm_client"] == {"wrapped": raw_client}

        assert session_manager.args == (
            pruning_config,
            {"summarizer": ctx.obj["llm_client"]},
        )
        assert client == {"wrapped": raw_client}


class TestBuildContextPruningConfig:
    """Test context pruning config building."""

    def test_build_context_pruning_config(self):
        """Test building context pruning config from settings."""
        settings = Settings()
        settings.context_pruner_max_total_tokens = 5000
        settings.context_pruner_trigger_ratio = 0.8
        settings.context_pruner_keep_recent_messages = 10

        config = _build_context_pruning_config_from_settings(settings)

        assert config.max_total_tokens == 5000
        # trigger_tokens should be 80% of max_total_tokens
        assert config.trigger_tokens == int(5000 * 0.8)
        assert config.keep_recent_messages == 10

    def test_build_context_pruning_config_with_explicit_trigger(self):
        settings = Settings()
        settings.context_pruner_max_total_tokens = 200
        settings.context_pruner_trigger_tokens = 50
        settings.context_pruner_keep_recent_messages = 1
        settings.context_pruner_summary_max_chars = 0
        settings.context_pruner_max_event_history = 0

        config = _build_context_pruning_config_from_settings(settings)

        assert config.trigger_tokens == 50
        assert config.keep_recent_messages >= 2


class TestResolvePromptInput:
    """Test prompt resolution helper."""

    def test_resolve_prompt_input_passes_through_multiline_text(self):
        """Inline prompt text containing newlines should be returned unchanged."""
        source = "Title\n* bullet"

        assert resolve_prompt_input(source) == source

    def test_resolve_prompt_input_none_returns_none(self):
        """None inputs should return None without error."""
        assert resolve_prompt_input(None) is None

    def test_resolve_prompt_input_blank_returns_none(self):
        """Whitespace-only prompt strings should return None."""
        assert resolve_prompt_input("   \t") is None

    def test_resolve_prompt_input_inline_literal_returns_value(self):
        """Inline single-line text should be returned unchanged."""
        assert resolve_prompt_input("Hello world") == "Hello world"

    def test_resolve_prompt_input_reads_file(self, tmp_path):
        """Prompt file paths should be read and returned as text."""
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("# Heading", encoding="utf-8")

        result = resolve_prompt_input(str(prompt_file))

        assert result == "# Heading"

    def test_resolve_prompt_input_missing_file_raises(self, tmp_path):
        """Missing prompt files should raise FileNotFoundError with guidance."""
        missing = tmp_path / "missing.md"

        with pytest.raises(FileNotFoundError) as exc:
            resolve_prompt_input(str(missing))

        assert "does not exist" in str(exc.value)

    def test_resolve_prompt_input_directory_raises(self, tmp_path):
        """Directories passed as prompt input should raise a clear error."""
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()

        with pytest.raises(IsADirectoryError) as exc:
            resolve_prompt_input(str(prompt_dir))

        assert "Expected prompt file" in str(exc.value)


class TestStructureHintsState:
    """Test helpers that capture project structure hints."""

    def test_get_structure_hints_state_initializes_defaults(self):
        """Initial call should populate state containers."""
        ctx = click.Context(click.Command("test"))
        ctx.obj = {}

        state = _get_structure_hints_state(ctx)

        assert isinstance(state["symbols"], set)
        assert isinstance(state["files"], dict)
        assert state["project_summary"] is None
        # Ensure subsequent calls reuse the same object
        assert _get_structure_hints_state(ctx) is state

    def test_merge_structure_hints_state_merges_payload(self):
        """Merging payload should accumulate symbols, file info, and summary."""
        state = {"symbols": {"Existing"}, "files": {}, "project_summary": None}
        payload = {
            "symbols": ["NewSymbol"],
            "files": {
                "src/app.py": {"outline": "class App:", "summary": "Updated app"},
                "src/other.py": {"symbols": ["DirectSymbol"]},
            },
            "project_summary": "Overall summary",
        }

        with patch(
            "ai_dev_agent.cli.utils.extract_symbols_from_outline",
            return_value=["OutlineSymbol"],
        ) as mock_extract:
            _merge_structure_hints_state(state, payload)

        mock_extract.assert_called_once_with("class App:")
        assert {"Existing", "NewSymbol"} == set(state["symbols"])
        assert state["files"]["src/app.py"]["symbols"] == ["OutlineSymbol"]
        assert state["files"]["src/app.py"]["summaries"] == ["Updated app"]
        assert state["files"]["src/other.py"]["symbols"] == ["DirectSymbol"]
        assert state["project_summary"] == "Overall summary"

    def test_get_structure_hints_state_normalizes_existing_state(self):
        """Existing state with incorrect container types should be normalized."""
        ctx = click.Context(click.Command("test"))
        ctx.obj = {
            "_structure_hints_state": {
                "symbols": ["Alpha"],
                "files": [],
                "project_summary": None,
            }
        }

        state = _get_structure_hints_state(ctx)

        assert isinstance(state["symbols"], set)
        assert isinstance(state["files"], dict)
        assert state["symbols"] == set()

    def test_export_structure_hints_state_formats_output(self):
        """Exported state should provide sorted symbols and shallow copies."""
        state = {
            "symbols": {"Beta", "Alpha"},
            "files": {"file.py": {"symbols": ["Zeta", "Eta"]}},
            "project_summary": "Summary",
        }

        exported = _export_structure_hints_state(state)

        assert exported["symbols"] == ["Alpha", "Beta"]
        assert exported["files"] == state["files"]
        assert exported["project_summary"] == "Summary"

    def test_merge_structure_hints_state_ignores_none_payload(self):
        """None payloads should leave the state unchanged."""
        state = {"symbols": {"Existing"}, "files": {"file.py": {}}, "project_summary": "Summary"}

        _merge_structure_hints_state(state, None)

        assert state == {
            "symbols": {"Existing"},
            "files": {"file.py": {}},
            "project_summary": "Summary",
        }

    def test_merge_structure_hints_state_ignores_non_mapping_files(self):
        """File entries without mapping metadata should be skipped."""
        state = {"symbols": set(), "files": {}, "project_summary": None}
        payload = {"files": {"bad": "value", "good": {"symbols": ["A"]}}}

        _merge_structure_hints_state(state, payload)

        assert "good" in state["files"]
        assert "bad" not in state["files"]

    def test_merge_structure_hints_state_handles_non_mapping_container(self):
        """Non-mapping file payloads should be ignored entirely."""
        state = {"symbols": set(), "files": {}, "project_summary": None}

        _merge_structure_hints_state(state, {"files": ["not", "mapping"]})

        assert state["files"] == {}


class TestUpdateFilesDiscovered:
    """Test helper gathering discovered file paths."""

    def test_update_files_discovered_collects_paths(self):
        """Payload entries from multiple sections should be normalized."""
        files_discovered: set[str] = set()
        payload = {
            "files": {
                "first": {"path": "src/app.py"},
                "second": {"path": Path("docs/README.md")},
            },
            "matches": [{"path": "tests/test_app.py"}, "lib/module.py"],
            "summaries": [{"path": "SUMMARY.md"}, {"no_path": "skip"}, "notes.txt"],
        }

        _update_files_discovered(files_discovered, payload)

        assert files_discovered == {
            "src/app.py",
            "docs/README.md",
            "tests/test_app.py",
            "lib/module.py",
            "SUMMARY.md",
            "notes.txt",
        }

    def test_update_files_discovered_handles_non_iterable_fields(self):
        """String and non-iterable payload sections should be ignored."""
        files_discovered: set[str] = set()
        payload = {
            "files": ["src/app.py", "", None],
            "matches": "not-a-list",
            "summaries": "summary.txt",
        }

        _update_files_discovered(files_discovered, payload)

        assert files_discovered == {"src/app.py"}

    def test_update_files_discovered_none_payload(self):
        """None payload should exit immediately."""
        files_discovered: set[str] = set()

        _update_files_discovered(files_discovered, None)

        assert files_discovered == set()


class TestInvokeRegistryTool:
    """Test registry tool invocation helper."""

    def test_invoke_registry_tool_uses_context(self):
        """Helper should build context and pass payload to the registry."""
        ctx = click.Context(click.Command("test"))
        ctx.obj = {}

        with (
            patch("ai_dev_agent.cli.utils._make_tool_context") as mock_context,
            patch("ai_dev_agent.cli.utils.tool_registry.invoke") as mock_invoke,
        ):
            sentinel_context = object()
            mock_context.return_value = sentinel_context
            mock_invoke.return_value = {"status": "ok"}

            result = _invoke_registry_tool(ctx, "tool-name", {"foo": "bar"}, with_sandbox=True)

        mock_context.assert_called_once_with(ctx, with_sandbox=True)
        mock_invoke.assert_called_once_with("tool-name", {"foo": "bar"}, sentinel_context)
        assert result == {"status": "ok"}


class TestRecordInvocation:
    """Test command invocation recording."""

    def test_record_invocation_appends_history(self):
        """Recording should capture trimmed command path and overrides."""
        root_cmd = click.Group("devagent")
        sub_cmd = click.Command("plan")
        root_cmd.add_command(sub_cmd)
        root_ctx = root_cmd.make_context("devagent", ["plan"])
        ctx = sub_cmd.make_context("plan", [], parent=root_ctx)
        store = InMemoryStateStore()
        ctx.obj = {"state": store}
        ctx.params = {"arg": "original"}

        _record_invocation(ctx, overrides={"arg": "override"})

        snapshot = store.load()
        history = snapshot["command_history"]
        assert len(history) == 1
        entry = history[0]
        assert entry["command_path"] == ["plan"]
        assert entry["params"]["arg"] == "override"
        assert "timestamp" in entry

    def test_record_invocation_skips_history_command(self):
        """History command invocations should not be recorded."""
        root_cmd = click.Group("devagent")
        history_cmd = click.Command("history")
        root_cmd.add_command(history_cmd)
        root_ctx = root_cmd.make_context("devagent", ["history"])  # root invocation
        ctx = history_cmd.make_context("history", [], parent=root_ctx)
        store = InMemoryStateStore()
        ctx.obj = {"state": store}

        _record_invocation(ctx)

        snapshot = store.load()
        assert snapshot["command_history"] == []


class TestUpdateTaskState:
    """Tests for plan/task persistence helper."""

    def test_update_task_state_applies_reasoning_hooks(self, monkeypatch):
        store = InMemoryStateStore()
        plan = {"tasks": [{"id": "task-1", "status": "pending"}]}
        task = {"id": "task-1", "status": "pending"}
        updates = {"status": "complete"}

        class DummyReasoning:
            def apply_to_task(self, task_obj):
                task_obj["status"] = "enriched"

            def merge_into_plan(self, plan_obj):
                plan_obj["notes"] = "merged"

        update_task_state(store, plan, task, updates, reasoning=DummyReasoning())

        snapshot = store.load()
        assert plan["status"] == "complete"
        assert plan["notes"] == "merged"
        assert task["status"] == "enriched"
        saved_task = snapshot["last_plan"]["tasks"][0]
        assert saved_task["status"] == "enriched"

    def test_update_task_state_appends_when_missing(self):
        store = InMemoryStateStore()
        plan = {"tasks": []}
        task = {"id": "task-2", "status": "todo"}

        update_task_state(store, plan, task, {"status": "in_progress"})

        assert plan["tasks"][0]["id"] == "task-2"
        snapshot = store.load()
        assert snapshot["last_plan"]["tasks"][0]["status"] == "in_progress"

    def test_update_task_state_handles_missing_identifier(self):
        store = InMemoryStateStore()
        plan = {"tasks": []}
        task = {"title": "Ad-hoc"}

        update_task_state(store, plan, task, {"status": "queued"})

        assert plan["tasks"] == []  # no identifier, nothing appended
        assert task["status"] == "queued"
