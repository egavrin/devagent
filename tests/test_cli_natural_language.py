"""Tests for natural language query and auto-detection."""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from ai_dev_agent.agents import AgentSpec
from ai_dev_agent.cli.commands import cli


@pytest.fixture(autouse=True)
def stub_cli_dependencies(monkeypatch):
    """Stub heavy CLI dependencies to keep natural language tests fast."""
    monkeypatch.setenv("DEVAGENT_API_KEY", "test-key")

    executor = MagicMock(name="_execute_react_assistant", return_value={})
    monkeypatch.setattr("ai_dev_agent.cli.commands._execute_react_assistant", executor)

    dummy_client = object()

    def fake_get_llm_client(ctx):
        ctx.obj["llm_client"] = dummy_client
        return dummy_client

    monkeypatch.setattr("ai_dev_agent.cli.commands.get_llm_client", fake_get_llm_client)

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

    monkeypatch.setattr("ai_dev_agent.cli.commands.AgentRegistry", DummyRegistry)
    yield {"executor": executor, "llm_client": dummy_client}


class TestNaturalLanguageQuery:
    """Test natural language queries with auto-detection."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_simple_question_direct_mode(self, runner, stub_cli_dependencies):
        """Simple questions should execute with direct mode."""
        prompt = "How do I use pytest?"
        result = runner.invoke(cli, [prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        call = executor.call_args
        assert call.args[3] == prompt
        assert call.kwargs["use_planning"] is False

    def test_complex_query_still_routes(self, runner, stub_cli_dependencies):
        """Complex queries should still route through the natural language fallback."""
        complex_query = (
            "Build a complete REST API with user authentication, tests, and security checks"
        )
        result = runner.invoke(cli, [complex_query])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == complex_query

    def test_review_keyword_routes(self, runner, stub_cli_dependencies):
        """Queries mentioning 'review' should still be handled through query fallback."""
        question = "Review src/auth.py for security issues"
        result = runner.invoke(cli, [question])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == question

    def test_design_keyword_routes(self, runner, stub_cli_dependencies):
        """Queries mentioning 'design' should still be handled by natural language fallback."""
        question = "Design the database schema for user authentication"
        result = runner.invoke(cli, [question])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == question

    def test_multiple_actions_routes(self, runner, stub_cli_dependencies):
        """Multi-step queries should route through the fallback path."""
        question = "Design the API and write tests for it"
        result = runner.invoke(cli, [question])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == question

    def test_long_query_routes(self, runner, stub_cli_dependencies):
        """Long queries should still route through the fallback path."""
        question = (
            "I need to build a complete feature that includes designing the architecture, "
            "writing comprehensive tests, implementing the code, and reviewing it"
        )
        result = runner.invoke(cli, [question])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == question


class TestNaturalLanguageWithFlags:
    """Test natural language queries with various flags."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_query_with_verbose_flag(self, runner, stub_cli_dependencies):
        """Test natural language with -v flag."""
        prompt = "How do I use pytest?"
        result = runner.invoke(cli, ["-v", prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == prompt

    def test_query_with_double_verbose_flag(self, runner, stub_cli_dependencies):
        """Test natural language with -vv flag."""
        prompt = "How do I use pytest?"
        result = runner.invoke(cli, ["-vv", prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()

    def test_query_with_triple_verbose_flag(self, runner, stub_cli_dependencies):
        """Test natural language with -vvv flag."""
        prompt = "How do I use pytest?"
        result = runner.invoke(cli, ["-vvv", prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()

    def test_query_with_quiet_flag(self, runner, stub_cli_dependencies):
        """Test natural language with -q flag."""
        prompt = "How do I use pytest?"
        result = runner.invoke(cli, ["-q", prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()

    def test_query_with_json_output(self, runner, stub_cli_dependencies):
        """Test natural language with --json flag."""
        prompt = "How do I use pytest?"
        result = runner.invoke(cli, ["--json", prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()

    def test_query_with_plan_flag(self, runner, stub_cli_dependencies):
        """Test natural language with --plan flag enabling planning."""
        prompt = "How do I use pytest?"
        result = runner.invoke(cli, ["--plan", prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.kwargs["use_planning"] is True

    def test_query_with_custom_config(self, runner, stub_cli_dependencies):
        """Test natural language with custom config file."""
        prompt = "How do I use pytest?"
        result = runner.invoke(cli, ["--config", "/dev/null", prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == prompt


class TestQueryCommand:
    """Test the deprecated 'query' command (should work with deprecation warning)."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_query_command_exists(self, runner, stub_cli_dependencies):
        """The query command should still exist for backward compatibility."""
        prompt = "How do I use pytest?"
        result = runner.invoke(cli, ["query", prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == prompt

    def test_query_command_plan_flag(self, runner, stub_cli_dependencies):
        """The query command should respect global --plan flag."""
        prompt = "test"
        result = runner.invoke(cli, ["--plan", "query", prompt])
        assert result.exit_code == 0
        executor = stub_cli_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == prompt
        assert executor.call_args.kwargs["use_planning"] is True
