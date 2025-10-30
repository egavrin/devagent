"""Tests for CLI utilities module."""

from __future__ import annotations

import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest

from ai_dev_agent.cli.utils import (
    _build_context,
    _build_context_pruning_config_from_settings,
    _collect_project_structure_outline,
    _detect_repository_language,
    _make_tool_context,
    _normalize_argument_list,
    _resolve_repo_path,
    build_system_context,
    get_llm_client,
    infer_task_files,
)
from ai_dev_agent.core.utils.config import Settings


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

        with patch("ai_dev_agent.cli.utils.ApprovalManager"):
            tool_ctx = _make_tool_context(ctx)

            # ToolContext uses the actual current directory
            assert tool_ctx.repo_root == Path.cwd()
            assert hasattr(tool_ctx, "settings")
            assert tool_ctx.settings == ctx.obj["settings"]


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
