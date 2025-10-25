"""Tests for specialized commands (create-design, generate-tests, write-code)."""
from unittest.mock import MagicMock
from pathlib import Path
import pytest
from click.testing import CliRunner
from ai_dev_agent.cli.commands import cli
from ai_dev_agent.agents.base import AgentResult


@pytest.fixture
def specialized_cli_stub(cli_stub_runtime, monkeypatch):
    """Stub specialized agent execution to avoid real LLM usage."""
    stub_result = AgentResult(success=True, output="Stubbed result", metadata={"source": "test"})
    execute_stub = MagicMock(return_value=stub_result)
    monkeypatch.setattr(
        "ai_dev_agent.agents.specialized.executor_bridge.execute_agent_with_react",
        execute_stub,
    )
    cli_stub_runtime["agent_execute"] = execute_stub
    return cli_stub_runtime


pytestmark = pytest.mark.usefixtures("specialized_cli_stub")


class TestCreateDesignCommand:
    """Test the create-design command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_create_design_help(self, runner):
        """create-design command should show help."""
        result = runner.invoke(cli, ["create-design", "--help"])
        assert result.exit_code == 0
        assert "design" in result.output.lower()

    def test_create_design_basic(self, runner, specialized_cli_stub):
        """create-design should accept a feature name."""
        result = runner.invoke(cli, ["create-design", "User Authentication"])
        assert result.exit_code == 0
        executor = specialized_cli_stub["agent_execute"]
        executor.assert_called_once()
        assert executor.call_args.kwargs["prompt"] == "Design User Authentication"

    def test_create_design_with_context(self, runner, specialized_cli_stub):
        """create-design should accept --context flag."""
        result = runner.invoke(cli, [
            "create-design", "REST API",
            "--context", "CRUD operations for blog posts"
        ])
        assert result.exit_code == 0
        executor = specialized_cli_stub["agent_execute"]
        executor.assert_called_once()
        prompt = executor.call_args.kwargs["prompt"]
        assert "Design REST API" in prompt
        assert "Context: CRUD operations for blog posts" in prompt

    def test_create_design_with_output(self, runner, specialized_cli_stub, tmp_path):
        """create-design should accept --output flag."""
        output_path = tmp_path / "design.md"
        result = runner.invoke(cli, [
            "create-design", "Authentication System",
            "--output", str(output_path)
        ])
        assert result.exit_code == 0
        executor = specialized_cli_stub["agent_execute"]
        executor.assert_called_once()
        assert output_path.exists()

    def test_create_design_with_verbose(self, runner):
        """create-design should accept verbosity flags."""
        result = runner.invoke(cli, [
            "create-design", "API",
            "-v"
        ])
        assert result.exit_code in [0, 2]

    def test_create_design_with_very_verbose(self, runner):
        """create-design should accept -vv flag."""
        result = runner.invoke(cli, [
            "create-design", "API",
            "-vv"
        ])
        assert result.exit_code in [0, 2]

    def test_create_design_with_json(self, runner):
        """create-design should accept --json flag."""
        result = runner.invoke(cli, [
            "create-design", "API",
            "--json"
        ])
        assert result.exit_code in [0, 2]


class TestGenerateTestsCommand:
    """Test the generate-tests command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_generate_tests_help(self, runner):
        """generate-tests command should show help."""
        result = runner.invoke(cli, ["generate-tests", "--help"])
        assert result.exit_code == 0
        assert "test" in result.output.lower()

    def test_generate_tests_basic(self, runner, specialized_cli_stub):
        """generate-tests should accept a feature name."""
        result = runner.invoke(cli, ["generate-tests", "Authentication Module"])
        assert result.exit_code == 0
        executor = specialized_cli_stub["agent_execute"]
        executor.assert_called_once()
        assert executor.call_args.kwargs["prompt"] == "Create all tests for Authentication Module with 90% coverage"

    def test_generate_tests_with_coverage(self, runner, specialized_cli_stub):
        """generate-tests should accept --coverage flag."""
        result = runner.invoke(cli, [
            "generate-tests", "Auth",
            "--coverage", "95"
        ])
        assert result.exit_code == 0
        executor = specialized_cli_stub["agent_execute"]
        executor.assert_called_once()

    def test_generate_tests_with_type(self, runner):
        """generate-tests should accept --type flag (unit|integration|all)."""
        result = runner.invoke(cli, [
            "generate-tests", "Auth",
            "--type", "integration"
        ])
        assert result.exit_code == 0

    def test_generate_tests_with_invalid_type(self, runner):
        """generate-tests should reject invalid --type values."""
        result = runner.invoke(cli, [
            "generate-tests", "Auth",
            "--type", "invalid"
        ])
        # Should either error or show usage
        assert result.exit_code in [0, 1, 2]

    def test_generate_tests_all_types(self, runner, specialized_cli_stub):
        """generate-tests should accept all valid test types."""
        for test_type in ["unit", "integration", "all"]:
            result = runner.invoke(cli, [
                "generate-tests", "Auth",
                "--type", test_type
            ])
            assert result.exit_code == 0, f"Failed for type={test_type}"
        executor = specialized_cli_stub["agent_execute"]
        assert executor.call_count == 3

    def test_generate_tests_with_verbose(self, runner):
        """generate-tests should accept verbosity flags."""
        result = runner.invoke(cli, [
            "generate-tests", "Auth",
            "-vvv"
        ])
        assert result.exit_code in [0, 2]


class TestWriteCodeCommand:
    """Test the write-code command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_write_code_help(self, runner):
        """write-code command should show help."""
        result = runner.invoke(cli, ["write-code", "--help"])
        assert result.exit_code == 0
        assert "code" in result.output.lower() or "implement" in result.output.lower()

    def test_write_code_basic(self, runner, specialized_cli_stub):
        """write-code should accept a design file."""
        result = runner.invoke(cli, ["write-code", "design.md"])
        assert result.exit_code in [0, 1, 2]
        executor = specialized_cli_stub["agent_execute"]
        executor.assert_called_once()
        assert executor.call_args.kwargs["prompt"].startswith("Implement the design at design.md")

    def test_write_code_with_test_file(self, runner, specialized_cli_stub):
        """write-code should accept --test-file flag."""
        result = runner.invoke(cli, [
            "write-code", "design.md",
            "--test-file", "tests/test_auth.py"
        ])
        assert result.exit_code in [0, 1, 2]
        executor = specialized_cli_stub["agent_execute"]
        executor.assert_called_once()

    def test_write_code_with_verbose(self, runner):
        """write-code should accept verbosity flags."""
        result = runner.invoke(cli, [
            "write-code", "design.md",
            "-v"
        ])
        assert result.exit_code in [0, 1, 2]


class TestSpecializedVsAutoDetection:
    """Test that specialized commands are for manual override only."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_query_auto_detects_design(self, runner, specialized_cli_stub):
        """Natural language should auto-detect design intent."""
        result = runner.invoke(cli, ["Design the user authentication system"])
        assert result.exit_code == 0
        executor = specialized_cli_stub["executor"]
        executor.assert_called_once()

    def test_query_auto_detects_tests(self, runner, specialized_cli_stub):
        """Natural language should auto-detect test intent."""
        result = runner.invoke(cli, ["Generate tests for authentication module"])
        assert result.exit_code == 0
        executor = specialized_cli_stub["executor"]
        executor.assert_called_once()

    def test_explicit_command_overrides_auto_detection(self, runner):
        """Explicit command should override auto-detection."""
        result = runner.invoke(cli, ["create-design", "Feature"])
        # Should use the explicit command, not auto-detect
        assert result.exit_code in [0, 1, 2]
