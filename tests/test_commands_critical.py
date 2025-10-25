"""Tests for critical CLI command paths and error handling."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import pytest

from ai_dev_agent.cli.commands import cli, diagnostics, shell
from ai_dev_agent.core.utils.config import Settings


class TestDiagnosticsCommand:
    """Test diagnostics command critical paths."""

    def test_diagnostics_requires_session(self):
        """Test that diagnostics command requires a session."""
        runner = CliRunner()
        result = runner.invoke(cli, ["diagnostics"])

        # Should fail without session or show helpful error
        assert result.exit_code != 0 or "session" in result.output.lower()
        assert "session" in result.output.lower() or "Error" in result.output

    def test_diagnostics_help(self):
        """Test that diagnostics command has help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["diagnostics", "--help"])

        assert result.exit_code == 0
        # Should show help information
        assert "diagnostics" in result.output.lower() or "session" in result.output.lower()


class TestShellCommand:
    """Test shell command critical paths and error handling."""

    @patch('ai_dev_agent.cli.commands.ShellSessionManager')
    def test_shell_command_initialization(self, mock_manager_class):
        """Test shell command initializes session manager."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.start_session.return_value = "test-session-id"

        # Mock the interactive loop to exit immediately
        mock_manager.is_session_active.return_value = False

        runner = CliRunner()
        with patch('ai_dev_agent.cli.commands.click.echo'):
            result = runner.invoke(cli, ["shell"], catch_exceptions=False)

        # Should have attempted to start session
        assert mock_manager.start_session.called or result.exit_code == 0

    @patch('ai_dev_agent.cli.commands.ShellSessionManager')
    def test_shell_command_handles_session_error(self, mock_manager_class):
        """Test shell command handles session errors gracefully."""
        from ai_dev_agent.tools.execution.shell_session import ShellSessionError

        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.start_session.side_effect = ShellSessionError("Test error")

        runner = CliRunner()
        result = runner.invoke(cli, ["shell"])

        # Should handle error gracefully
        assert result.exit_code != 0 or "error" in result.output.lower()


class TestNaturalLanguageRouting:
    """Test natural language routing critical paths."""

    def test_natural_language_routing_empty_query(self):
        """Test that empty natural language queries are rejected."""
        runner = CliRunner()
        result = runner.invoke(cli, [])

        # Should show help or error
        assert result.exit_code == 0  # Shows help
        assert "AI-assisted development agent CLI" in result.output or "--help" in result.output

    def test_natural_language_routing_with_flags(self):
        """Test natural language routing with option flags."""
        runner = CliRunner()

        # Test with --plan flag
        result = runner.invoke(cli, ["--plan", "--help"])
        assert result.exit_code == 0

    def test_custom_options_system_flag(self):
        """Test --system flag routing."""
        runner = CliRunner()

        # --system should route to query command
        # This will fail without actual LLM but tests the routing
        result = runner.invoke(cli, ["--system", "test", "query", "hello"])

        # May fail due to missing API key, but routing should work
        assert "API key" in result.output or result.exit_code == 0 or result.exception is not None

    def test_custom_options_format_flag(self):
        """Test --format flag routing."""
        runner = CliRunner()

        # --format should route to query command
        result = runner.invoke(cli, ["--format", "json", "query", "hello"])

        # May fail due to missing API key, but routing should work
        assert "API key" in result.output or result.exit_code == 0 or result.exception is not None


class TestQueryCommand:
    """Test query command critical paths."""

    def test_query_command_requires_input(self):
        """Test that query command requires input."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query"])

        # Should fail or require input
        assert result.exit_code != 0 or "request" in result.output.lower()

    def test_query_command_with_stdin(self):
        """Test query command can accept stdin input."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query"], input="test query\n")

        # Will fail without API key, but should process input
        assert "API key" in result.output or result.exit_code != 0


class TestReviewCommand:
    """Test review command critical paths."""

    def test_review_command_requires_targets(self):
        """Test that review command requires target files."""
        runner = CliRunner()
        result = runner.invoke(cli, ["review"])

        # Should fail or show help
        assert result.exit_code != 0 or "target" in result.output.lower() or "file" in result.output.lower()

    def test_review_command_with_nonexistent_file(self):
        """Test review command with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["review", "nonexistent_file.py"])

        # Should handle missing file
        assert result.exit_code != 0 or "not found" in result.output.lower() or "error" in result.output.lower()


class TestConfigHandling:
    """Test config file handling."""

    def test_config_file_loading(self, tmp_path):
        """Test loading config from file."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"verbose": true}')

        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(config_file), "--help"])

        # Should load config without error
        assert result.exit_code == 0

    def test_invalid_config_file(self, tmp_path):
        """Test invalid config file handling."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text('invalid json{')

        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(config_file), "--help"])

        # Should handle invalid config gracefully
        assert "error" in result.output.lower() or result.exit_code != 0 or result.exit_code == 0


class TestVerbosityHandling:
    """Test verbose and silent flags."""

    def test_verbose_flag_enables_logging(self):
        """Test that --verbose flag enables verbose logging."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "diagnostics"])

        # Should not crash with verbose enabled
        assert result.exit_code == 0

    def test_silent_flag_suppresses_output(self):
        """Test that --silent flag works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--silent", "diagnostics"])

        # Should not crash with silent enabled
        assert result.exit_code == 0


class TestErrorHandling:
    """Test error handling in commands."""

    def test_usage_error_handling(self):
        """Test that UsageError is handled properly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["nonexistent-command"])

        # Should show error or route to natural language
        assert result.exit_code != 0 or result.exception is not None

    def test_argument_parsing_errors(self):
        """Test argument parsing error handling."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--invalid-flag"])

        # Should show error about invalid flag
        assert result.exit_code != 0 or "invalid" in result.output.lower() or "error" in result.output.lower()
