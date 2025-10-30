"""Simple tests for tool_invoker module to improve coverage."""

from unittest.mock import MagicMock

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.metrics import MetricsCollector
from ai_dev_agent.engine.react.tool_invoker import RegistryToolInvoker
from ai_dev_agent.engine.react.types import ActionRequest, Observation


class TestRegistryToolInvokerSimple:
    """Simple tests for the RegistryToolInvoker class."""

    def test_init_minimal(self, tmp_path):
        """Test minimal initialization."""
        settings = Settings()
        tool_invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        assert tool_invoker.workspace == tmp_path
        assert tool_invoker.settings == settings

    def test_init_with_collector(self, tmp_path):
        """Test initialization with metrics collector."""
        settings = Settings()
        collector = MetricsCollector(repo_root=tmp_path)

        tool_invoker = RegistryToolInvoker(
            workspace=tmp_path, settings=settings, collector=collector
        )

        assert tool_invoker.collector == collector

    def test_call_with_simple_action(self, tmp_path):
        """Test calling the tool invoker with a simple action."""
        settings = Settings()
        tool_invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        # Create a simple action that should fail (no tool registered)
        action = ActionRequest(
            tool="nonexistent", parameters={}, step_id="test_step", thought="Testing"
        )

        result = tool_invoker(action)

        assert isinstance(result, Observation)
        # Should contain error message
        assert result.outcome != "" or result.error != ""

    def test_invoke_batch_empty(self, tmp_path):
        """Test invoke_batch with empty list."""
        settings = Settings()
        tool_invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        result = tool_invoker.invoke_batch([])

        assert isinstance(result, Observation)

    def test_invoke_batch_single_tool(self, tmp_path):
        """Test invoke_batch with single tool call."""
        from ai_dev_agent.engine.react.types import ToolCall

        settings = Settings()
        tool_invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        tool_call = ToolCall(tool="test_function", args={}, call_id="test_id")

        result = tool_invoker.invoke_batch([tool_call])

        assert isinstance(result, Observation)

    def test_with_code_editor(self, tmp_path):
        """Test with code editor set."""
        settings = Settings()
        code_editor = MagicMock()

        tool_invoker = RegistryToolInvoker(
            workspace=tmp_path, settings=settings, code_editor=code_editor
        )

        assert tool_invoker.code_editor == code_editor

    def test_with_test_runner(self, tmp_path):
        """Test with test runner set."""
        settings = Settings()
        test_runner = MagicMock()

        tool_invoker = RegistryToolInvoker(
            workspace=tmp_path, settings=settings, test_runner=test_runner
        )

        assert tool_invoker.test_runner == test_runner

    def test_with_sandbox(self, tmp_path):
        """Test with sandbox set."""
        settings = Settings()
        sandbox = MagicMock()

        tool_invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings, sandbox=sandbox)

        assert tool_invoker.sandbox == sandbox

    def test_with_shell_session(self, tmp_path):
        """Test with shell session manager."""
        settings = Settings()
        shell_manager = MagicMock()

        tool_invoker = RegistryToolInvoker(
            workspace=tmp_path,
            settings=settings,
            shell_session_manager=shell_manager,
            shell_session_id="test_session",
        )

        assert tool_invoker.shell_session_manager == shell_manager
        assert tool_invoker.shell_session_id == "test_session"

    def test_with_devagent_config(self, tmp_path):
        """Test with devagent config."""
        from ai_dev_agent.core.utils.devagent_config import DevAgentConfig

        settings = Settings()
        devagent_cfg = DevAgentConfig()

        tool_invoker = RegistryToolInvoker(
            workspace=tmp_path, settings=settings, devagent_cfg=devagent_cfg
        )

        assert tool_invoker.devagent_cfg == devagent_cfg

    def test_multiple_calls(self, tmp_path):
        """Test multiple calls to the tool invoker."""
        settings = Settings()
        tool_invoker = RegistryToolInvoker(workspace=tmp_path, settings=settings)

        # Create multiple actions
        actions = [
            ActionRequest(
                tool=f"tool_{i}",
                parameters={"param": i},
                step_id=f"step_{i}",
                thought=f"Thought {i}",
            )
            for i in range(3)
        ]

        # Call with each action
        results = []
        for action in actions:
            result = tool_invoker(action)
            results.append(result)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, Observation)
