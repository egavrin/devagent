"""Tests for cli/utils.py - utility helper functions."""

import platform
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ai_dev_agent.cli.utils import (
    _COMMAND_MAPPINGS,
    _OS_FRIENDLY_NAMES,
    _PLATFORM_EXAMPLES,
    build_system_context,
)


class TestBuildSystemContext:
    """Test build_system_context function."""

    def test_build_system_context_structure(self):
        """Test that build_system_context returns correct structure."""
        context = build_system_context()

        assert isinstance(context, dict)
        assert "os" in context
        assert "os_friendly" in context
        assert "command_mappings" in context
        assert "platform_examples" in context

    def test_build_system_context_os_name(self):
        """Test that OS name is included."""
        context = build_system_context()

        system = platform.system()
        expected_os = _OS_FRIENDLY_NAMES.get(system, system)

        assert context["os"] == system  # Raw OS name
        assert context["os_friendly"] == expected_os  # Friendly name

    def test_build_system_context_platform(self):
        """Test that platform info is included."""
        context = build_system_context()

        assert "os_version" in context
        assert isinstance(context["os_version"], str)
        assert "architecture" in context
        assert isinstance(context["architecture"], str)

    def test_build_system_context_commands(self):
        """Test that command mappings are included."""
        context = build_system_context()

        system = platform.system()
        expected_commands = _COMMAND_MAPPINGS.get(system, _COMMAND_MAPPINGS.get("Linux"))

        assert context["command_mappings"] == expected_commands

    def test_build_system_context_examples(self):
        """Test that platform examples are included."""
        context = build_system_context()

        system = platform.system()
        expected_examples = _PLATFORM_EXAMPLES.get(system, _PLATFORM_EXAMPLES.get("Linux"))

        assert context["platform_examples"] == expected_examples


class TestOSFriendlyNames:
    """Test OS friendly name mappings."""

    def test_os_friendly_names_darwin(self):
        """Test Darwin maps to macOS."""
        assert _OS_FRIENDLY_NAMES["Darwin"] == "macOS"

    def test_os_friendly_names_linux(self):
        """Test Linux maps to Linux."""
        assert _OS_FRIENDLY_NAMES["Linux"] == "Linux"

    def test_os_friendly_names_windows(self):
        """Test Windows maps to Windows."""
        assert _OS_FRIENDLY_NAMES["Windows"] == "Windows"

    def test_os_friendly_names_contains_major_platforms(self):
        """Test that all major platforms are covered."""
        assert len(_OS_FRIENDLY_NAMES) >= 3
        assert all(isinstance(v, str) for v in _OS_FRIENDLY_NAMES.values())


class TestCommandMappings:
    """Test platform command mappings."""

    def test_command_mappings_structure(self):
        """Test that command mappings have correct structure."""
        for platform_name, commands in _COMMAND_MAPPINGS.items():
            assert isinstance(platform_name, str)
            assert isinstance(commands, dict)
            assert "list_files" in commands
            assert "find_files" in commands
            assert "copy" in commands

    def test_command_mappings_darwin(self):
        """Test macOS command mappings."""
        darwin_commands = _COMMAND_MAPPINGS["Darwin"]

        assert darwin_commands["list_files"] == "ls -la"
        assert darwin_commands["find_files"] == "find"
        assert darwin_commands["open_file"] == "open"

    def test_command_mappings_linux(self):
        """Test Linux command mappings."""
        linux_commands = _COMMAND_MAPPINGS["Linux"]

        assert linux_commands["list_files"] == "ls -la"
        assert linux_commands["find_files"] == "find"
        assert linux_commands["open_file"] == "xdg-open"

    def test_command_mappings_windows(self):
        """Test Windows command mappings."""
        windows_commands = _COMMAND_MAPPINGS["Windows"]

        assert windows_commands["list_files"] == "dir"
        assert windows_commands["find_files"] == "where"
        assert windows_commands["open_file"] == "start"


class TestPlatformExamples:
    """Test platform example strings."""

    def test_platform_examples_structure(self):
        """Test that platform examples exist for major platforms."""
        assert "Darwin" in _PLATFORM_EXAMPLES
        assert "Linux" in _PLATFORM_EXAMPLES
        assert "Windows" in _PLATFORM_EXAMPLES

    def test_platform_examples_contain_examples(self):
        """Test that example strings contain actual examples."""
        for platform_name, example_str in _PLATFORM_EXAMPLES.items():
            assert isinstance(example_str, str)
            assert len(example_str) > 0
            assert (
                "e.g." in example_str.lower()
                or "example" in example_str.lower()
                or "ls" in example_str
                or "dir" in example_str
            )


class TestResolveRepoPath:
    """Test _resolve_repo_path function."""

    def test_resolve_repo_path_none(self):
        """Test that None returns current directory."""
        from ai_dev_agent.cli.utils import _resolve_repo_path

        result = _resolve_repo_path(None)

        assert isinstance(result, Path)
        assert result == Path.cwd()

    def test_resolve_repo_path_absolute(self):
        """Test that absolute paths outside repo raise error."""
        import click

        from ai_dev_agent.cli.utils import _resolve_repo_path

        # Absolute paths outside repo should raise ClickException
        with pytest.raises(click.ClickException, match="escapes the repository root"):
            _resolve_repo_path("/tmp/test")

    def test_resolve_repo_path_relative(self):
        """Test resolving relative path."""
        from ai_dev_agent.cli.utils import _resolve_repo_path

        result = _resolve_repo_path("./test")

        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_resolve_repo_path_tilde(self):
        """Test that tilde paths are treated as relative to repo."""
        from ai_dev_agent.cli.utils import _resolve_repo_path

        # Tilde is NOT expanded - it's treated as a literal directory name
        result = _resolve_repo_path("~/test")

        assert isinstance(result, Path)
        assert result.is_absolute()
        # The function treats "~/test" as a relative path within repo
        assert str(result).startswith(str(Path.cwd()))


class TestDetectRepositoryLanguage:
    """Test _detect_repository_language function."""

    def test_detect_language_python(self, tmp_path):
        """Test detecting Python repository."""
        from ai_dev_agent.cli.utils import _detect_repository_language

        # Create Python files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "test.py").write_text("assert True")

        language, total = _detect_repository_language(tmp_path)

        assert language == "python"
        assert total == 2

    def test_detect_language_javascript(self, tmp_path):
        """Test detecting JavaScript repository."""
        from ai_dev_agent.cli.utils import _detect_repository_language

        # Create JS files
        (tmp_path / "index.js").write_text("console.log('hello');")
        (tmp_path / "package.json").write_text("{}")

        language, total = _detect_repository_language(tmp_path)

        assert language == "javascript"
        assert total == 2

    def test_detect_language_mixed(self, tmp_path):
        """Test detecting mixed language repository."""
        from ai_dev_agent.cli.utils import _detect_repository_language

        # Create multiple language files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "index.js").write_text("console.log('hello');")
        (tmp_path / "README.md").write_text("# Test")

        language, total = _detect_repository_language(tmp_path)

        # Should detect one of the languages (whichever has more files, or first alphabetically if tied)
        assert language in ["python", "javascript"]
        assert total == 3

    def test_detect_language_empty(self, tmp_path):
        """Test detecting language in empty directory."""
        from ai_dev_agent.cli.utils import _detect_repository_language

        language, total = _detect_repository_language(tmp_path)

        # Should return None for language when no files detected
        assert language is None
        assert total is None


class TestNormalizeArgumentList:
    """Test _normalize_argument_list function."""

    def test_normalize_none(self):
        """Test normalizing None input."""
        from ai_dev_agent.cli.utils import _normalize_argument_list

        result = _normalize_argument_list({}, plural_key="files")

        assert result == []

    def test_normalize_empty_list(self):
        """Test normalizing empty list."""
        from ai_dev_agent.cli.utils import _normalize_argument_list

        result = _normalize_argument_list({"files": []}, plural_key="files")

        assert result == []

    def test_normalize_list_of_strings(self):
        """Test normalizing list of strings."""
        from ai_dev_agent.cli.utils import _normalize_argument_list

        result = _normalize_argument_list({"files": ["arg1", "arg2", "arg3"]}, plural_key="files")

        assert result == ["arg1", "arg2", "arg3"]

    def test_normalize_with_singular_fallback(self):
        """Test fallback to singular key."""
        from ai_dev_agent.cli.utils import _normalize_argument_list

        result = _normalize_argument_list(
            {"file": "single.txt"}, plural_key="files", singular_key="file"
        )

        assert result == ["single.txt"]


class TestPromptYesNo:
    """Test _prompt_yes_no function."""

    @patch("ai_dev_agent.core.approval.approvals.ApprovalManager.require")
    def test_prompt_yes_no_yes(self, mock_require):
        """Test prompting with yes response."""
        from ai_dev_agent.cli.utils import _prompt_yes_no
        from ai_dev_agent.core.approval.policy import ApprovalPolicy

        mock_require.return_value = True

        # Create a mock Click context
        ctx = Mock()
        ctx.obj = {"approval_policy": ApprovalPolicy()}

        result = _prompt_yes_no(ctx, "test_purpose", "Continue?")

        assert result is True
        assert mock_require.called

    @patch("ai_dev_agent.core.approval.approvals.ApprovalManager.require")
    def test_prompt_yes_no_no(self, mock_require):
        """Test prompting with no response."""
        from ai_dev_agent.cli.utils import _prompt_yes_no
        from ai_dev_agent.core.approval.policy import ApprovalPolicy

        mock_require.return_value = False

        ctx = Mock()
        ctx.obj = {"approval_policy": ApprovalPolicy()}

        result = _prompt_yes_no(ctx, "test_purpose", "Continue?")

        assert result is False

    @patch("ai_dev_agent.core.approval.approvals.ApprovalManager.require")
    def test_prompt_yes_no_default_yes(self, mock_require):
        """Test prompting with default yes."""
        from ai_dev_agent.cli.utils import _prompt_yes_no
        from ai_dev_agent.core.approval.policy import ApprovalPolicy

        mock_require.return_value = True

        ctx = Mock()
        ctx.obj = {"approval_policy": ApprovalPolicy()}

        result = _prompt_yes_no(ctx, "test_purpose", "Continue?", default=True)

        # Should use default
        assert result is True

    @patch("ai_dev_agent.core.approval.approvals.ApprovalManager.require")
    def test_prompt_yes_no_default_no(self, mock_require):
        """Test prompting with default no."""
        from ai_dev_agent.cli.utils import _prompt_yes_no
        from ai_dev_agent.core.approval.policy import ApprovalPolicy

        mock_require.return_value = False

        ctx = Mock()
        ctx.obj = {"approval_policy": ApprovalPolicy()}

        result = _prompt_yes_no(ctx, "test_purpose", "Continue?", default=False)

        # Should use default
        assert result is False


class TestToolCandidates:
    """Test _TOOL_CANDIDATES constant."""

    def test_tool_candidates_structure(self):
        """Test that tool candidates are properly defined."""
        from ai_dev_agent.cli.utils import _TOOL_CANDIDATES

        assert isinstance(_TOOL_CANDIDATES, tuple)
        assert len(_TOOL_CANDIDATES) > 0
        assert all(isinstance(tool, str) for tool in _TOOL_CANDIDATES)

    def test_tool_candidates_common_tools(self):
        """Test that common development tools are included."""
        from ai_dev_agent.cli.utils import _TOOL_CANDIDATES

        # Check for common tools
        assert "git" in _TOOL_CANDIDATES
        assert "python" in _TOOL_CANDIDATES or "node" in _TOOL_CANDIDATES
        assert "npm" in _TOOL_CANDIDATES or "pip" in _TOOL_CANDIDATES


class TestBuildContextFunction:
    """Test _build_context helper function."""

    def test_build_context_creates_dict(self):
        """Test that _build_context returns a dictionary."""
        from ai_dev_agent.cli.utils import _build_context
        from ai_dev_agent.core.utils.config import Settings

        settings = Settings()
        context = _build_context(settings)

        assert isinstance(context, dict)

    def test_build_context_includes_settings(self):
        """Test that context includes settings-derived values."""
        from ai_dev_agent.cli.utils import _build_context
        from ai_dev_agent.core.utils.config import Settings

        settings = Settings()
        context = _build_context(settings)

        # Should have some basic context keys
        assert isinstance(context, dict)
        assert len(context) > 0
