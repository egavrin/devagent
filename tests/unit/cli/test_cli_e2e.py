"""End-to-end tests for CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from ai_dev_agent.cli import cli


class TestCLIEndToEnd:
    """Test CLI commands end-to-end."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace with git initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            # Initialize git repo
            import subprocess

            subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"], cwd=workspace, check=True
            )
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=workspace, check=True)

            # Create a sample file
            (workspace / "test.py").write_text("def hello():\n    return 'world'\n")
            subprocess.run(["git", "add", "."], cwd=workspace, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=workspace, check=True)

            yield workspace

    @pytest.mark.skip(reason="Requires complex mocking of LLM client")
    @pytest.mark.skip(reason="Requires complex mocking of LLM client")
    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_query_command_basic(self, mock_client, mock_execute, runner, temp_workspace):
        """Test basic query command."""
        # Mock the LLM client and execution
        mock_client.return_value = MagicMock()
        mock_execute.return_value = None  # _execute_react_assistant doesn't return a value

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            # Set environment variable to mock API key
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["query", "test the system"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        # Check the actual error
        if result.exit_code != 0:
            print(f"Command failed with output: {result.output}")
            print(f"Exception: {result.exception}")
        assert result.exit_code == 0
        assert mock_execute.called

    @pytest.mark.skip(reason="Requires complex mocking of LLM client")
    @pytest.mark.skip(reason="Requires complex mocking of LLM client")
    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_query_command_with_json_output(
        self, mock_client, mock_execute, runner, temp_workspace
    ):
        """Test query command with JSON output."""
        mock_client.return_value = MagicMock()

        # Mock json output written to stdout
        def mock_execute_json(*args, **kwargs):
            click.echo(json.dumps({"status": "success", "data": "test"}))

        mock_execute.side_effect = mock_execute_json

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["query", "--json", "analyze code"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0
        # Output should be valid JSON
        output = json.loads(result.output)
        assert "status" in output

    @patch("ai_dev_agent.agents.specialized.review_agent.ReviewAgent.execute")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_review_command_file(self, mock_client, mock_execute, runner, temp_workspace):
        """Test review command with a file."""
        mock_client.return_value = MagicMock()
        mock_execute.return_value = MagicMock(
            success=True,
            output="Code review complete",
            metadata={"findings": [], "issues_found": 0, "quality_score": 1.0},
        )

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            # Create a file to review
            test_file = temp_workspace / "review_me.py"
            test_file.write_text("def bad_function():\n    pass  # TODO: implement\n")

            result = runner.invoke(cli, ["review", str(test_file)])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0
        mock_execute.assert_called_once()

    @pytest.mark.skip(reason="Review command requires LLM client")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_review_command_nonexistent_file(self, mock_client, runner, temp_workspace):
        """Test review command with nonexistent file."""
        mock_client.return_value = MagicMock()

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["review", "nonexistent.py"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    @patch("ai_dev_agent.agents.specialized.design_agent.DesignAgent.execute")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_create_design_command(self, mock_client, mock_execute, runner, temp_workspace):
        """Test create-design command."""
        mock_client.return_value = MagicMock()
        mock_execute.return_value = MagicMock(
            success=True, output="# Design Document\n\n## Architecture\n...", metadata={}
        )

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["create-design", "authentication system"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0
        mock_execute.assert_called_once()
        # Check that the feature description was passed
        call_args = mock_execute.call_args
        assert "authentication system" in call_args[0][0]

    @patch("ai_dev_agent.agents.specialized.design_agent.DesignAgent.execute")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_create_design_with_output_file(
        self, mock_client, mock_execute, runner, temp_workspace
    ):
        """Test create-design command with output file."""
        mock_client.return_value = MagicMock()
        design_content = "# Design Document\n\n## Overview\nTest design"
        mock_execute.return_value = MagicMock(success=True, output=design_content, metadata={})

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            output_file = temp_workspace / "design.md"
            result = runner.invoke(cli, ["create-design", "feature", "--output", str(output_file)])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.read_text() == design_content

    @patch("ai_dev_agent.agents.specialized.testing_agent.TestingAgent.execute")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_generate_tests_command(self, mock_client, mock_execute, runner, temp_workspace):
        """Test generate-tests command."""
        mock_client.return_value = MagicMock()
        test_code = "def test_feature():\n    assert True"
        mock_execute.return_value = MagicMock(
            success=True, output=test_code, metadata={"coverage": 90}
        )

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["generate-tests", "authentication"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0
        mock_execute.assert_called_once()

    @patch("ai_dev_agent.agents.specialized.testing_agent.TestingAgent.execute")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_generate_tests_with_coverage_target(
        self, mock_client, mock_execute, runner, temp_workspace
    ):
        """Test generate-tests command with coverage target."""
        mock_client.return_value = MagicMock()
        mock_execute.return_value = MagicMock(success=True, output="test code", metadata={})

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["generate-tests", "feature", "--coverage", "95"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0
        # Check that coverage target was included in prompt
        call_args = mock_execute.call_args
        assert "95" in str(call_args[0][0])

    @patch("ai_dev_agent.agents.specialized.implementation_agent.ImplementationAgent.execute")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_write_code_command(self, mock_client, mock_execute, runner, temp_workspace):
        """Test write-code command."""
        mock_client.return_value = MagicMock()
        implementation = "class AuthSystem:\n    pass"
        mock_execute.return_value = MagicMock(success=True, output=implementation, metadata={})

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            # Create a design file
            design_file = temp_workspace / "design.md"
            design_file.write_text("# Design\nImplement auth system")

            result = runner.invoke(cli, ["write-code", str(design_file)])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0
        mock_execute.assert_called_once()

    @pytest.mark.skip(reason="Requires complex mocking of LLM client")
    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_query_with_verbose(self, mock_client, mock_execute, runner, temp_workspace):
        """Test query command with verbose output."""
        mock_client.return_value = MagicMock()
        mock_execute.return_value = None

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["-v", "query", "test"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0
        # Verbose flag should be set in context

    @pytest.mark.skip(reason="Requires complex mocking of LLM client")
    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_query_with_quiet(self, mock_client, mock_execute, runner, temp_workspace):
        """Test query command with quiet output."""
        mock_client.return_value = MagicMock()
        mock_execute.return_value = None

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["-q", "query", "test"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0

    def test_help_command(self, runner):
        """Test help command."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "Commands:" in result.output

    def test_command_help(self, runner):
        """Test individual command help."""
        commands = ["query", "review", "create-design", "generate-tests", "write-code"]

        for cmd in commands:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0
            assert "Usage:" in result.output

    @pytest.mark.skip(reason="ChatSession import issues")
    @patch("ai_dev_agent.cli.runtime.commands.chat.ShellSessionManager")
    def test_chat_command(self, mock_chat_session, runner, temp_workspace):
        """Test chat command starts a session."""
        mock_session = MagicMock()
        mock_chat_session.return_value = mock_session

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            # Simulate immediate exit
            mock_session.run.side_effect = KeyboardInterrupt()

            result = runner.invoke(cli, ["chat"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        # Chat should handle keyboard interrupt gracefully
        assert result.exit_code == 0 or result.exit_code == 1
        mock_session.run.assert_called_once()

    def test_invalid_command(self, runner):
        """Test invalid command shows help."""
        result = runner.invoke(cli, ["invalid-command"])

        assert result.exit_code != 0
        assert "Error" in result.output or "Usage" in result.output

    @pytest.mark.skip(reason="Requires complex mocking of LLM client")
    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_natural_language_fallback(self, mock_client, mock_execute, runner, temp_workspace):
        """Test natural language query as direct command."""
        mock_client.return_value = MagicMock()
        mock_execute.return_value = None

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            # This should be interpreted as a natural language query
            result = runner.invoke(cli, ["list all Python files in the project"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        # Should succeed and execute as a query
        assert result.exit_code == 0
        mock_execute.assert_called()

    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_review_with_json_output(self, mock_client, runner, temp_workspace):
        """Test review command with JSON output format."""
        mock_client.return_value = MagicMock()

        with patch(
            "ai_dev_agent.agents.specialized.review_agent.ReviewAgent.execute"
        ) as mock_execute:
            mock_execute.return_value = MagicMock(
                success=True,
                output="Review complete",
                metadata={
                    "findings": [{"severity": "high", "issue": "security vulnerability"}],
                    "issues_found": 1,
                    "quality_score": 0.7,
                },
            )

            original_cwd = str(Path.cwd())
            try:
                os.chdir(temp_workspace)
                os.environ["DEVAGENT_API_KEY"] = "test-key"
                test_file = temp_workspace / "code.py"
                test_file.write_text("pass")

                result = runner.invoke(cli, ["review", str(test_file), "--json"])
            finally:
                os.chdir(original_cwd)
                os.environ.pop("DEVAGENT_API_KEY", None)

            assert result.exit_code == 0
            # Should output valid JSON
            output = json.loads(result.output)
            assert "success" in output
            assert output["success"]
