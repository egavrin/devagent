"""Tests for the agent executor bridge module."""

from unittest.mock import MagicMock, Mock, patch

import click
import pytest

from ai_dev_agent.agents.base import AgentContext, AgentResult, BaseAgent
from ai_dev_agent.agents.executor import AgentExecutor, execute_agent_with_react
from ai_dev_agent.core.utils.config import Settings


class TestAgentExecutor:
    """Test AgentExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create an AgentExecutor instance."""
        return AgentExecutor()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock BaseAgent."""
        agent = MagicMock(spec=BaseAgent)
        agent.name = "test_agent"
        agent.description = "Test Agent for unit testing"
        agent.capabilities = ["analysis", "planning"]
        agent.permissions = {"write": "deny", "edit": "deny"}
        return agent

    @pytest.fixture
    def mock_context(self):
        """Create a mock AgentContext."""
        context = MagicMock(spec=AgentContext)
        context.session_id = "test_session_123"
        context.agent_id = "test_agent"
        context.metadata = {}
        return context

    def test_init(self, executor):
        """Test executor initialization."""
        assert executor._settings_cache is None

    def test_get_settings_creates_new(self, executor):
        """Test that _get_settings creates Settings on first call."""
        settings = executor._get_settings()
        assert isinstance(settings, Settings)
        assert executor._settings_cache is settings

    def test_get_settings_returns_cached(self, executor):
        """Test that _get_settings returns cached Settings."""
        settings1 = executor._get_settings()
        settings2 = executor._get_settings()
        assert settings1 is settings2

    def test_create_click_context(self, executor):
        """Test Click context creation using metadata."""
        settings = Settings()
        metadata = {
            "system_context": {"os": "TestOS"},
            "project_context": {"workspace": "repo"},
        }

        ctx = executor._create_click_context(settings, metadata)

        assert isinstance(ctx, click.Context)
        assert ctx.obj["settings"] is settings
        assert ctx.obj["system_context"]["os"] == "TestOS"
        assert ctx.obj["project_context"]["workspace"] == "repo"
        assert ctx.obj["silent_mode"] is True
        assert ctx.obj["_session_id"].startswith("delegate-")

    def test_build_agent_prompt_with_spec(self, executor, mock_agent):
        """Test building agent prompt with agent spec."""
        mock_spec = MagicMock()
        mock_spec.system_prompt_suffix = "You are a specialized test agent."

        prompt = executor._build_agent_prompt(
            mock_agent, "Test the authentication system", mock_spec
        )

        assert "You are a specialized test agent." in prompt
        assert "Test the authentication system" in prompt
        assert "# Task" in prompt

    def test_build_agent_prompt_without_spec(self, executor, mock_agent):
        """Test building agent prompt without agent spec."""
        prompt = executor._build_agent_prompt(mock_agent, "Test the authentication system", None)

        assert "Test Agent for unit testing" in prompt
        assert "Your capabilities: analysis, planning" in prompt
        assert "IMPORTANT: You CANNOT use these tools: write, edit" in prompt
        assert "You are read-only" in prompt
        assert "Test the authentication system" in prompt

    def test_build_agent_prompt_no_permissions(self, executor):
        """Test building prompt for agent with no permission restrictions."""
        agent = MagicMock(spec=BaseAgent)
        agent.name = "unrestricted"
        agent.description = "Unrestricted agent"
        agent.capabilities = ["all"]
        agent.permissions = {}

        prompt = executor._build_agent_prompt(agent, "Do something", None)

        assert "Unrestricted agent" in prompt
        assert "Your capabilities: all" in prompt
        assert "CANNOT use these tools" not in prompt
        assert "Do something" in prompt

    def test_convert_result_success(self, executor):
        """Test converting successful ReAct result."""
        mock_result = Mock()
        mock_result.stop_condition = "success"
        mock_result.steps = [1, 2, 3]
        # Don't set exception attribute - hasattr should return False
        del mock_result.exception

        react_result = {"final_message": "Task completed successfully", "result": mock_result}

        result = executor._convert_result(react_result, "test_agent")

        assert result.success is True
        assert result.output == "Task completed successfully"
        assert result.error is None
        assert result.metadata["agent"] == "test_agent"
        assert result.metadata["stop_condition"] == "success"
        assert result.metadata["steps_taken"] == 3

    def test_convert_result_with_json(self, executor):
        """Test converting result with JSON output."""
        mock_result = Mock()
        mock_result.stop_condition = "success"
        mock_result.steps = []
        # Don't set exception attribute
        del mock_result.exception

        react_result = {
            "final_message": "Analysis complete",
            "final_json": {"result": "data"},
            "result": mock_result,
        }

        result = executor._convert_result(react_result, "test_agent")

        assert result.success is True
        assert result.metadata["json_output"] == {"result": "data"}

    def test_convert_result_with_exception(self, executor):
        """Test converting result with exception."""
        mock_result = MagicMock()
        mock_result.exception = Exception("Test error")
        mock_result.stop_condition = "error"
        mock_result.steps = []

        react_result = {"final_message": "Error occurred", "result": mock_result}

        result = executor._convert_result(react_result, "test_agent")

        assert result.success is False
        assert result.error == "Test error"
        assert result.output == "Error occurred"

    def test_convert_result_budget_exhausted(self, executor):
        """Test converting result when budget is exhausted."""
        mock_result = Mock()
        mock_result.stop_condition = "budget"
        mock_result.steps = [1, 2]
        # Don't set exception attribute
        del mock_result.exception

        react_result = {
            "final_message": "",  # Empty message indicates incomplete
            "result": mock_result,
        }

        result = executor._convert_result(react_result, "test_agent")

        assert result.success is False
        assert result.error == "Execution reached iteration limit without completing task"

    def test_convert_result_budget_with_output(self, executor):
        """Test converting result when budget exhausted but has output."""
        mock_result = Mock()
        mock_result.stop_condition = "budget"
        mock_result.steps = [1, 2]
        # Don't set exception attribute
        del mock_result.exception

        react_result = {"final_message": "Partial results: found 3 issues", "result": mock_result}

        result = executor._convert_result(react_result, "test_agent")

        # Should be success since we got useful output
        assert result.success is True
        assert result.output == "Partial results: found 3 issues"

    def test_convert_result_no_run_result(self, executor):
        """Test converting result with no run_result."""
        react_result = {"final_message": "Completed", "result": None}

        result = executor._convert_result(react_result, "test_agent")

        assert result.success is True
        assert result.output == "Completed"
        assert result.metadata["steps_taken"] == 0
        assert result.metadata["stop_condition"] is None

    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    @patch("ai_dev_agent.agents.AgentRegistry")
    def test_execute_with_react_success(
        self, mock_registry, mock_get_client, mock_execute, executor, mock_agent, mock_context
    ):
        """Test successful execution with ReAct."""
        # Setup mocks
        mock_registry.has_agent.return_value = True
        mock_registry.get.return_value = MagicMock(system_prompt_suffix="Test prompt")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_result = Mock()
        mock_result.stop_condition = "success"
        mock_result.steps = [1]
        del mock_result.exception

        mock_execute.return_value = {"final_message": "Task completed", "result": mock_result}

        # Execute
        result = executor.execute_with_react(mock_agent, "Test prompt", mock_context)

        # Verify
        assert result.success is True
        assert result.output == "Task completed"
        mock_execute.assert_called_once()

    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    def test_execute_with_react_with_provided_context(
        self, mock_get_client, mock_execute, executor, mock_agent, mock_context
    ):
        """Test execution with provided Click context."""
        # Create a real Click context
        ctx = click.Context(click.Command("test"))
        ctx.obj = {"existing": "data"}

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_execute.return_value = {"final_message": "Done", "result": None}

        # Execute
        executor.execute_with_react(
            mock_agent, "Test", mock_context, ctx=ctx, cli_client=mock_client
        )

        # Verify context was modified
        assert "_session_id" in ctx.obj
        assert ctx.obj["silent_mode"] is True
        assert ctx.obj["existing"] == "data"  # Original data preserved

    @patch("ai_dev_agent.cli.utils.get_llm_client")
    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    def test_execute_with_react_exception(
        self, mock_execute, mock_get_client, executor, mock_agent, mock_context
    ):
        """Test execution when exception occurs."""
        # Mock the get_llm_client to avoid API key error
        mock_get_client.side_effect = Exception("Test error")

        result = executor.execute_with_react(mock_agent, "Test", mock_context)

        assert result.success is False
        assert "Agent execution failed" in result.output
        assert "Test error" in result.error

    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("ai_dev_agent.cli.utils.get_llm_client")
    @patch("ai_dev_agent.agents.AgentRegistry")
    def test_execute_with_react_unregistered_agent(
        self, mock_registry, mock_get_client, mock_execute, executor, mock_agent, mock_context
    ):
        """Test execution with unregistered agent."""
        # Agent not in registry
        mock_registry.has_agent.return_value = False

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_execute.return_value = {"final_message": "Done", "result": None}

        executor.execute_with_react(mock_agent, "Test", mock_context)

        # Should use agent name when not registered
        call_args = mock_execute.call_args
        # When agent is not registered, it uses the agent.name directly
        assert call_args[1]["agent_type"] == "test_agent"


class TestModuleFunctions:
    """Test module-level functions."""

    @patch("ai_dev_agent.agents.executor._executor.execute_with_react")
    def test_execute_agent_with_react(self, mock_execute):
        """Test the convenience function."""
        mock_agent = MagicMock(spec=BaseAgent)
        mock_context = MagicMock(spec=AgentContext)
        mock_ctx = MagicMock()
        mock_client = MagicMock()

        expected_result = AgentResult(success=True, output="Done")
        mock_execute.return_value = expected_result

        result = execute_agent_with_react(
            mock_agent, "Test prompt", mock_context, ctx=mock_ctx, cli_client=mock_client
        )

        assert result is expected_result
        mock_execute.assert_called_once_with(
            mock_agent, "Test prompt", mock_context, mock_ctx, mock_client
        )
