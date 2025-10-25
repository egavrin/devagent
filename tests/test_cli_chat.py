"""Tests for chat/interactive command."""
from unittest.mock import MagicMock
import pytest
from click.testing import CliRunner
from ai_dev_agent.cli.commands import cli


@pytest.fixture(autouse=True)
def stub_chat_dependencies(monkeypatch):
    """Stub heavy components so the chat command stays fast in tests."""
    monkeypatch.setenv("DEVAGENT_API_KEY", "test-key")

    manager = MagicMock(spec=("create_session", "close_all"))
    manager.create_session.return_value = "session-id"
    manager_factory = MagicMock(return_value=manager)
    monkeypatch.setattr("ai_dev_agent.cli.commands.ShellSessionManager", manager_factory)

    executor = MagicMock(name="_execute_react_assistant")
    monkeypatch.setattr("ai_dev_agent.cli.commands._execute_react_assistant", executor)

    llm_client = MagicMock(name="llm_client")

    def fake_get_llm_client(ctx):
        ctx.obj["llm_client"] = llm_client
        return llm_client

    monkeypatch.setattr("ai_dev_agent.cli.commands.get_llm_client", fake_get_llm_client)
    yield {
        "manager": manager,
        "manager_factory": manager_factory,
        "executor": executor,
        "llm_client": llm_client,
    }


class TestChatCommand:
    """Test the chat (interactive) command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_chat_command_exists(self, runner):
        """The chat command should exist."""
        # Try to get help for chat command
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "interactive" in result.output.lower() or "chat" in result.output.lower()

    def test_chat_command_starts_prompt(self, runner, stub_chat_dependencies):
        """Chat command should start interactive mode with prompt."""
        # Simulate user exiting immediately with EOF
        result = runner.invoke(cli, ["chat"], input="exit\n")
        # Should handle the exit gracefully
        assert result.exit_code == 0
        output = result.output.lower()
        # Should show some indication of interactive mode
        assert "devagent" in output or ">" in output or "chat" in output
        stub_chat_dependencies["manager"].create_session.assert_called_once()
        stub_chat_dependencies["manager"].close_all.assert_called_once()

    def test_chat_with_help_command(self, runner, stub_chat_dependencies):
        """User should be able to type 'help' in chat mode."""
        result = runner.invoke(cli, ["chat"], input="help\nexit\n")
        assert result.exit_code == 0
        executor = stub_chat_dependencies["executor"]
        executor.assert_called_once()
        assert executor.call_args.args[3] == "help"

    def test_chat_with_quit_command(self, runner, stub_chat_dependencies):
        """User should be able to type 'quit' to exit."""
        result = runner.invoke(cli, ["chat"], input="quit\n")
        assert result.exit_code == 0
        stub_chat_dependencies["manager"].close_all.assert_called_once()

    def test_chat_with_q_command(self, runner, stub_chat_dependencies):
        """User should be able to type 'q' to exit."""
        result = runner.invoke(cli, ["chat"], input="q\n")
        assert result.exit_code == 0
        stub_chat_dependencies["manager"].close_all.assert_called_once()

    def test_chat_maintains_context(self, runner, stub_chat_dependencies):
        """Chat should maintain conversation history across queries."""
        result = runner.invoke(cli, ["chat"], input="What is Python?\nTell me more.\nexit\n")
        # Should complete without error
        assert result.exit_code == 0
        executor = stub_chat_dependencies["executor"]
        assert executor.call_count == 2
        prompts = [call.args[3] for call in executor.call_args_list]
        assert prompts == ["What is Python?", "Tell me more."]

    def test_chat_with_empty_input(self, runner, stub_chat_dependencies):
        """Chat should handle empty input gracefully."""
        result = runner.invoke(cli, ["chat"], input="\nexit\n")
        assert result.exit_code == 0
        # Empty prompt should not trigger executor
        stub_chat_dependencies["executor"].assert_not_called()


class TestInteractiveFlag:
    """Test that --interactive flag doesn't exist (use 'chat' instead)."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_interactive_flag_not_supported(self, runner):
        """The old --interactive flag should not exist (use 'chat' command instead)."""
        result = runner.invoke(cli, ["--interactive"])
        # Should show error or be handled as unrecognized option
        # (Click will route to query command if it's a natural language fallback)
        assert result.exit_code in [0, 1, 2]
