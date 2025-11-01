"""End-to-end tests for CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from ai_dev_agent.agents import AgentSpec
from ai_dev_agent.cli import cli


@pytest.fixture(autouse=True)
def stub_cli_dependencies(monkeypatch):
    """Stub heavy CLI dependencies to keep tests fast and deterministic."""
    monkeypatch.setenv("DEVAGENT_API_KEY", "test-key")

    executor = MagicMock(name="_execute_react_assistant", return_value={})
    monkeypatch.setattr("ai_dev_agent.cli.react.executor._execute_react_assistant", executor)
    monkeypatch.setattr(
        "ai_dev_agent.cli.runtime.commands.query._execute_react_assistant",
        executor,
    )

    dummy_client = object()

    def fake_get_llm_client(ctx):
        ctx.obj["llm_client"] = dummy_client
        return dummy_client

    monkeypatch.setattr("ai_dev_agent.cli.utils.get_llm_client", fake_get_llm_client)

    class DummyRegistry:
        @staticmethod
        def has_agent(name):
            return name == "manager"

        @staticmethod
        def get(name):
            if name != "manager":
                raise KeyError(name)
            return AgentSpec(name="manager", tools=[], max_iterations=10)

        @staticmethod
        def list_agents():
            return ["manager"]

    monkeypatch.setattr("ai_dev_agent.cli.router.AgentRegistry", DummyRegistry)
    monkeypatch.setattr("ai_dev_agent.cli.runtime.commands.query.AgentRegistry", DummyRegistry)
    yield {"executor": executor, "llm_client": dummy_client}


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

    def test_query_command_basic(self, runner, temp_workspace, stub_cli_dependencies):
        """Test basic query command."""
        stub_cli_dependencies["executor"].return_value = None  # Already a MagicMock

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
        assert stub_cli_dependencies["executor"].called

    def test_query_command_with_json_output(self, runner, temp_workspace, stub_cli_dependencies):
        """Test query command with JSON output."""

        # Mock json output written to stdout
        def mock_execute_json(*args, **kwargs):
            click.echo(json.dumps({"status": "success", "data": "test"}))

        stub_cli_dependencies["executor"].side_effect = mock_execute_json

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["--json", "query", "analyze code"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code == 0
        # Output should be valid JSON
        lines = [line for line in result.output.splitlines() if line.strip()]
        output = json.loads(lines[-1])
        assert "status" in output

    @patch("ai_dev_agent.cli.runtime.commands.review.execute_strategy")
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

    @patch("ai_dev_agent.cli.runtime.commands.review.execute_strategy")
    def test_review_command_failure_surface(self, mock_execute, runner, temp_workspace):
        """Test review command surfaces failures from the strategy."""
        mock_execute.return_value = MagicMock(
            success=False,
            output="file not found",
            metadata={"issues_found": 1},
        )

        original_cwd = str(Path.cwd())
        try:
            os.chdir(temp_workspace)
            os.environ["DEVAGENT_API_KEY"] = "test-key"
            result = runner.invoke(cli, ["review", "nonexistent.py"])
        finally:
            os.chdir(original_cwd)
            os.environ.pop("DEVAGENT_API_KEY", None)

        assert result.exit_code != 0
        output = result.output.lower()
        assert "file not found" in output or "failed" in output
        mock_execute.assert_called_once()

    @patch("ai_dev_agent.cli.runtime.commands.design.execute_strategy")
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
        agent_type, prompt, *_ = call_args[0]
        assert agent_type == "design"
        assert "authentication system" in prompt

    @patch("ai_dev_agent.cli.runtime.commands.design.execute_strategy")
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

    @patch("ai_dev_agent.cli.runtime.commands.generate_tests.execute_strategy")
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

    @patch("ai_dev_agent.cli.runtime.commands.generate_tests.execute_strategy")
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
        agent_type, prompt, *_ = call_args[0]
        assert agent_type == "test"
        assert "95" in prompt

    @patch("ai_dev_agent.cli.runtime.commands.write_code.execute_strategy")
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

        with patch("ai_dev_agent.cli.runtime.commands.review.execute_strategy") as mock_execute:
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
            lines = [line for line in result.output.splitlines() if line.strip()]
            output = json.loads(lines[-1])
            assert "success" in output
            assert output["success"]

    def test_system_flag_inline_text(self, runner):
        """Test --system flag with inline text is passed to executor."""
        # Patch at the module level where it's called from
        with patch(
            "ai_dev_agent.cli.runtime.commands.query._execute_react_assistant"
        ) as mock_execute:
            # Mock execute to return simple answer
            mock_execute.return_value = {
                "final_message": "4",
                "final_answer": "4",
                "final_json": None,
                "result": MagicMock(status="success", stop_reason="Completed"),
                "printed_final": True,
            }

            os.environ["DEVAGENT_API_KEY"] = "test-key"
            try:
                result = runner.invoke(cli, ["--system", "Answer with one number", "query", "2+2"])
            finally:
                os.environ.pop("DEVAGENT_API_KEY", None)

            # Verify the command executed successfully
            assert result.exit_code == 0

            # Verify _execute_react_assistant was called
            assert mock_execute.called

            # Verify system_extension remains unset and global message stored in settings
            call_args = mock_execute.call_args[0]
            call_kwargs = mock_execute.call_args[1]
            system_extension = call_kwargs.get("system_extension")
            assert system_extension is None

            settings = call_args[2]
            assert settings.global_system_message == "Answer with one number"

    def test_context_flag_inline_text(self, runner):
        """Test --context flag with inline text is passed to executor."""
        # Patch at the module level where it's called from
        with patch(
            "ai_dev_agent.cli.runtime.commands.query._execute_react_assistant"
        ) as mock_execute:
            # Mock execute to return answer
            mock_execute.return_value = {
                "final_message": "The project uses JWT tokens for authentication.",
                "final_answer": "The project uses JWT tokens for authentication.",
                "final_json": None,
                "result": MagicMock(status="success", stop_reason="Completed"),
                "printed_final": True,
            }

            os.environ["DEVAGENT_API_KEY"] = "test-key"
            try:
                result = runner.invoke(
                    cli,
                    ["--context", "This project uses JWT for auth", "query", "how does auth work"],
                )
            finally:
                os.environ.pop("DEVAGENT_API_KEY", None)

            # Verify the command executed
            assert result.exit_code == 0

            # Verify _execute_react_assistant was called and context stored on settings
            assert mock_execute.called
            call_args = mock_execute.call_args[0]
            call_kwargs = mock_execute.call_args[1]

            settings = call_args[2]  # Third positional arg is settings
            assert settings.global_context_message == "This project uses JWT for auth"
            assert call_kwargs.get("system_extension") is None

    def test_system_and_context_flags_together(self, runner):
        """Test both --system and --context flags work together."""
        # Patch at the module level where it's called from
        with patch(
            "ai_dev_agent.cli.runtime.commands.query._execute_react_assistant"
        ) as mock_execute:
            # Mock execute to return answer
            mock_execute.return_value = {
                "final_message": "Optimized",
                "final_answer": "Optimized",
                "final_json": None,
                "result": MagicMock(status="success", stop_reason="Completed"),
                "printed_final": True,
            }

            os.environ["DEVAGENT_API_KEY"] = "test-key"
            try:
                result = runner.invoke(
                    cli,
                    [
                        "--system",
                        "Be concise",
                        "--context",
                        "High-traffic API",
                        "query",
                        "optimize cache",
                    ],
                )
            finally:
                os.environ.pop("DEVAGENT_API_KEY", None)

            # Verify the command executed
            assert result.exit_code == 0

            # Verify both messages are stored on settings without mutating system_extension
            call_args = mock_execute.call_args[0]
            call_kwargs = mock_execute.call_args[1]
            settings = call_args[2]
            assert call_kwargs.get("system_extension") is None
            assert settings.global_system_message == "Be concise"
            assert settings.global_context_message == "High-traffic API"
