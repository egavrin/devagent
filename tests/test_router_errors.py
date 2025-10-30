"""Tests for IntentRouter error handling and edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from ai_dev_agent.cli.router import IntentDecision, IntentRouter, IntentRoutingError
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.providers.llm.base import LLMError, Message


class TestIntentRouterErrorHandling:
    """Test IntentRouter error handling scenarios."""

    def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises IntentRoutingError."""
        settings = Settings()
        mock_client = Mock()
        router = IntentRouter(client=mock_client, settings=settings)

        with pytest.raises(IntentRoutingError, match="Empty prompt"):
            router.route("")

    def test_whitespace_only_prompt_raises_error(self):
        """Test that whitespace-only prompt raises IntentRoutingError."""
        settings = Settings()
        mock_client = Mock()
        router = IntentRouter(client=mock_client, settings=settings)

        with pytest.raises(IntentRoutingError, match="Empty prompt"):
            router.route("   \n\t  ")

    def test_no_client_raises_error(self):
        """Test that missing LLM client raises IntentRoutingError."""
        settings = Settings()
        router = IntentRouter(client=None, settings=settings)

        with pytest.raises(IntentRoutingError, match="No LLM client available"):
            router.route("test prompt")

    @patch("ai_dev_agent.cli.router.SessionManager")
    def test_llm_error_wrapped_as_routing_error(self, mock_session_manager_class):
        """Test that LLMError is wrapped as IntentRoutingError."""
        settings = Settings()
        mock_client = Mock()
        mock_client.invoke_tools.side_effect = LLMError("API error")
        mock_client.generate.side_effect = LLMError("API error")

        # Mock session manager
        mock_session_manager = MagicMock()
        mock_session_manager_class.get_instance.return_value = mock_session_manager
        mock_session = MagicMock()
        mock_session_manager.ensure_session.return_value = mock_session
        mock_session_manager.compose.return_value = [Message(role="user", content="test")]

        router = IntentRouter(client=mock_client, settings=settings)

        with pytest.raises(IntentRoutingError, match="Intent routing failed"):
            router.route("test prompt")

        # Should have logged the error to session
        assert mock_session_manager.add_system_message.called

    @patch("ai_dev_agent.cli.router.SessionManager")
    def test_unexpected_exception_wrapped(self, mock_session_manager_class):
        """Test that unexpected exceptions are wrapped as IntentRoutingError."""
        settings = Settings()
        mock_client = Mock()
        mock_client.invoke_tools.side_effect = RuntimeError("Unexpected error")

        # Mock session manager
        mock_session_manager = MagicMock()
        mock_session_manager_class.get_instance.return_value = mock_session_manager
        mock_session = MagicMock()
        mock_session_manager.ensure_session.return_value = mock_session
        mock_session_manager.compose.return_value = [Message(role="user", content="test")]

        router = IntentRouter(client=mock_client, settings=settings)

        with pytest.raises(IntentRoutingError, match="Unexpected routing error"):
            router.route("test prompt")

        # Should have logged the error to session
        assert mock_session_manager.add_system_message.called


class TestIntentRouterEdgeCases:
    """Test IntentRouter edge case scenarios."""

    @patch("ai_dev_agent.cli.router.SessionManager")
    def test_no_tool_calls_with_prefer_generate(self, mock_session_manager_class):
        """Test behavior when model returns no tool calls with prefer_generate=True."""
        settings = Settings()
        mock_client = Mock()

        # Need to properly mock the response structure
        from ai_dev_agent.providers.llm.base import ToolCallResult

        # Mock response with no tool calls
        mock_response = ToolCallResult(
            message_content="I need more information",
            calls=[],  # No tool calls
            raw_tool_calls=None,
        )
        mock_client.generate.return_value = mock_response

        # Mock session manager
        mock_session_manager = MagicMock()
        mock_session_manager_class.get_instance.return_value = mock_session_manager
        mock_session = MagicMock()
        mock_session_manager.ensure_session.return_value = mock_session
        mock_session_manager.compose.return_value = [Message(role="user", content="test")]

        router = IntentRouter(client=mock_client, settings=settings)

        # Router validates response structure first
        with pytest.raises(IntentRoutingError):
            router.route_prompt("test prompt")

    @patch("ai_dev_agent.cli.router.SessionManager")
    def test_no_tool_calls_with_prefer_invoke(self, mock_session_manager_class):
        """Test behavior when model returns no tool calls with prefer_generate=False."""
        settings = Settings()
        mock_client = Mock()

        # Need to properly mock the response structure
        from ai_dev_agent.providers.llm.base import ToolCallResult

        # Mock response with no tool calls but has message
        mock_response = ToolCallResult(
            message_content="Here's my response", calls=[], raw_tool_calls=None
        )
        mock_client.invoke_tools.return_value = mock_response

        # Mock session manager
        mock_session_manager = MagicMock()
        mock_session_manager_class.get_instance.return_value = mock_session_manager
        mock_session = MagicMock()
        mock_session_manager.ensure_session.return_value = mock_session
        mock_session_manager.compose.return_value = [Message(role="user", content="test")]

        router = IntentRouter(client=mock_client, settings=settings)

        # Should return IntentDecision with text instead of raising
        result = router.route("test prompt")

        assert result.tool is None
        assert result.arguments == {"text": "Here's my response"}

    @patch("ai_dev_agent.cli.router.SessionManager")
    def test_empty_message_no_tool_calls(self, mock_session_manager_class):
        """Test behavior when model returns empty message and no tool calls."""
        settings = Settings()
        mock_client = Mock()

        # Need to properly mock the response structure
        from ai_dev_agent.providers.llm.base import ToolCallResult

        # Mock response with empty message and no tool calls
        mock_response = ToolCallResult(message_content="", calls=[], raw_tool_calls=None)
        mock_client.invoke_tools.return_value = mock_response

        # Mock session manager
        mock_session_manager = MagicMock()
        mock_session_manager_class.get_instance.return_value = mock_session_manager
        mock_session = MagicMock()
        mock_session_manager.ensure_session.return_value = mock_session
        mock_session_manager.compose.return_value = [Message(role="user", content="test")]

        router = IntentRouter(client=mock_client, settings=settings)

        with pytest.raises(IntentRoutingError, match="Could not determine a tool"):
            router.route("test prompt")


class TestIntentRouterNormalization:
    """Test IntentRouter system context normalization."""

    def test_normalize_system_context_with_dict(self):
        """Test that dict context is normalized correctly."""
        settings = Settings()
        mock_client = Mock()

        context = {"os": "Linux", "shell": "/bin/bash", "available_tools": ["git", "npm"]}

        router = IntentRouter(client=mock_client, settings=settings)
        normalized = router._normalise_system_context(context)

        assert normalized["os"] == "Linux"
        assert normalized["shell"] == "/bin/bash"
        assert "git" in normalized["available_tools"]
        assert "npm" in normalized["available_tools"]

    def test_normalize_system_context_with_none(self):
        """Test that None context gets defaults."""
        settings = Settings()
        mock_client = Mock()

        router = IntentRouter(client=mock_client, settings=settings)
        normalized = router._normalise_system_context(None)

        # Should have default values
        assert "os" in normalized
        assert "shell" in normalized
        assert isinstance(normalized["available_tools"], list)

    def test_normalize_system_context_available_tools_as_string(self):
        """Test that single string tool is converted to list."""
        settings = Settings()
        mock_client = Mock()

        context = {"available_tools": "git"}

        router = IntentRouter(client=mock_client, settings=settings)
        normalized = router._normalise_system_context(context)

        assert normalized["available_tools"] == ["git"]

    def test_normalize_system_context_filters_empty_tools(self):
        """Test that empty tools are filtered out."""
        settings = Settings()
        mock_client = Mock()

        context = {"available_tools": ["git", "", None, "npm"]}

        router = IntentRouter(client=mock_client, settings=settings)
        normalized = router._normalise_system_context(context)

        assert "git" in normalized["available_tools"]
        assert "npm" in normalized["available_tools"]
        assert "" not in normalized["available_tools"]
        # None gets stringified but filtered because falsy


class TestIntentRouterInitialization:
    """Test IntentRouter initialization scenarios."""

    def test_init_with_default_tools(self):
        """Test router initializes with default tools when none provided."""
        settings = Settings()
        mock_client = Mock()

        router = IntentRouter(client=mock_client, settings=settings)

        assert router.tools is not None
        assert isinstance(router.tools, list)

    def test_init_with_custom_tools(self):
        """Test router accepts custom tools list."""
        settings = Settings()
        mock_client = Mock()
        custom_tools = [{"type": "function", "function": {"name": "custom_tool"}}]

        router = IntentRouter(client=mock_client, settings=settings, tools=custom_tools)

        assert router.tools == custom_tools

    def test_init_with_project_profile(self):
        """Test router accepts project profile."""
        settings = Settings()
        mock_client = Mock()
        profile = {"language": "python", "framework": "pytest"}

        router = IntentRouter(client=mock_client, settings=settings, project_profile=profile)

        assert router.project_profile == profile

    def test_init_with_tool_success_history(self):
        """Test router accepts tool success history."""
        settings = Settings()
        mock_client = Mock()
        history = {"read": 0.95, "write": 0.87}

        router = IntentRouter(client=mock_client, settings=settings, tool_success_history=history)

        assert router.tool_success_history == history

    @patch("ai_dev_agent.cli.router.build_system_context")
    def test_init_handles_build_context_exception(self, mock_build_context):
        """Test router handles exception from build_system_context."""
        settings = Settings()
        mock_client = Mock()
        mock_build_context.side_effect = RuntimeError("Context error")

        # Should not raise, should use defaults
        router = IntentRouter(client=mock_client, settings=settings)

        assert router._system_context is not None
        # Should have normalized the empty dict to defaults
        assert "os" in router._system_context


class TestIntentDecision:
    """Test IntentDecision dataclass."""

    def test_intent_decision_creation(self):
        """Test creating IntentDecision."""
        decision = IntentDecision(
            tool="read", arguments={"path": "test.py"}, rationale="Need to read the file"
        )

        assert decision.tool == "read"
        assert decision.arguments == {"path": "test.py"}
        assert decision.rationale == "Need to read the file"

    def test_intent_decision_optional_rationale(self):
        """Test IntentDecision with no rationale."""
        decision = IntentDecision(tool="write", arguments={"content": "test"})

        assert decision.tool == "write"
        assert decision.rationale is None

    def test_intent_decision_no_tool(self):
        """Test IntentDecision with no tool (text response)."""
        decision = IntentDecision(tool=None, arguments={"text": "I cannot help with that"})

        assert decision.tool is None
        assert "text" in decision.arguments
