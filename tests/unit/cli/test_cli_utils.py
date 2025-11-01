"""Tests for CLI utilities module."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import click
import pytest

from ai_dev_agent.cli.utils import (
    _build_context,
    _build_context_pruning_config_from_settings,
    _collect_project_structure_outline,
    _detect_repository_language,
    _export_structure_hints_state,
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
)
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.state import InMemoryStateStore


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


class TestResolvePromptInput:
    """Test prompt resolution helper."""

    def test_resolve_prompt_input_passes_through_multiline_text(self):
        """Inline prompt text containing newlines should be returned unchanged."""
        source = "Title\n* bullet"

        assert resolve_prompt_input(source) == source

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
