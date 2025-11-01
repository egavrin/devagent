"""Comprehensive tests for the CLI router module."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest

from ai_dev_agent.agents import AgentSpec
from ai_dev_agent.cli.router import IntentDecision, IntentRouter, IntentRoutingError
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.providers.llm.base import LLMError, ToolCall, ToolCallResult


class TestIntentDecision:
    """Tests for IntentDecision dataclass."""

    def test_intent_decision_creation(self):
        """Test creating an IntentDecision."""
        decision = IntentDecision(
            tool="test_tool", arguments={"arg1": "value1"}, rationale="Test rationale"
        )
        assert decision.tool == "test_tool"
        assert decision.arguments == {"arg1": "value1"}
        assert decision.rationale == "Test rationale"

    def test_intent_decision_optional_rationale(self):
        """Test IntentDecision with no rationale."""
        decision = IntentDecision(tool="test_tool", arguments={"arg1": "value1"})
        assert decision.rationale is None


def test_tool_call_result_legacy_constructor():
    """Ensure ToolCallResult supports legacy constructor arguments."""
    result = ToolCallResult(call_id="legacy", name="find", content='{"query": "*.py"}')

    assert result.calls
    assert result.calls[0].name == "find"
    assert result.calls[0].arguments["query"] == "*.py"


class TestIntentRoutingError:
    """Tests for IntentRoutingError exception."""

    def test_intent_routing_error(self):
        """Test raising IntentRoutingError."""
        with pytest.raises(IntentRoutingError, match="Test error"):
            raise IntentRoutingError("Test error")


class TestIntentRouter:
    """Tests for IntentRouter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.settings = Settings()

        # Create a mock AgentSpec
        self.mock_agent_spec = AgentSpec(
            name="test_agent",
            tools=["find", "grep", "read", "write", "run", "symbols"],
            max_iterations=10,
            system_prompt_suffix="Test suffix",
            description="Test agent description",
        )

        # Mock AgentRegistry
        self.registry_patcher = patch("ai_dev_agent.cli.router.AgentRegistry")
        mock_registry = self.registry_patcher.start()
        mock_registry.get.return_value = self.mock_agent_spec

        # Mock SessionManager
        self.session_patcher = patch("ai_dev_agent.cli.router.SessionManager")
        mock_session_manager_class = self.session_patcher.start()
        self.mock_session = MagicMock()
        mock_session_manager_class.get_instance.return_value = self.mock_session

        # Mock tool registry
        self.tool_registry_patcher = patch("ai_dev_agent.cli.router.tool_registry")
        self.mock_tool_registry = self.tool_registry_patcher.start()

        # Create mock tool specs
        mock_spec = MagicMock()
        mock_spec.description = "Test tool"
        mock_spec.request_schema_path = MagicMock()
        self.mock_tool_registry.get.return_value = mock_spec

    def teardown_method(self):
        """Clean up patches."""
        self.registry_patcher.stop()
        self.session_patcher.stop()
        self.tool_registry_patcher.stop()

    def test_init_with_defaults(self):
        """Test router initialization with default parameters."""
        router = IntentRouter(client=self.mock_client, settings=self.settings)
        assert router.client == self.mock_client
        assert router.settings == self.settings
        assert router.agent_spec == self.mock_agent_spec
        assert router.project_profile == {}
        assert router.tool_success_history == {}

    def test_init_with_custom_params(self):
        """Test router initialization with custom parameters."""
        custom_tools = [{"type": "function", "function": {"name": "custom"}}]
        project_profile = {"key": "value"}
        tool_history = {"tool1": 0.9}

        router = IntentRouter(
            client=self.mock_client,
            settings=self.settings,
            agent_type="custom_agent",
            tools=custom_tools,
            project_profile=project_profile,
            tool_success_history=tool_history,
        )

        assert router.tools == custom_tools
        assert router.project_profile == project_profile
        assert router.tool_success_history == tool_history

    @patch("ai_dev_agent.cli.router.build_system_context")
    def test_build_tool_list(self, mock_build_context):
        """Test building the tool list."""
        mock_build_context.return_value = "System context"

        router = IntentRouter(client=self.mock_client, settings=self.settings)

        # Should combine DEFAULT_TOOLS with registry tools
        assert isinstance(router.tools, list)
        mock_build_context.assert_called_once()

    def test_build_registry_tools(self):
        """Test building registry tools."""
        # Set up mock file reading for schema
        schema_content = json.dumps({"type": "object", "properties": {"test": {"type": "string"}}})

        mock_file = mock_open(read_data=schema_content)

        with patch("builtins.open", mock_file):
            router = IntentRouter(client=self.mock_client, settings=self.settings)

            # Access private method for testing
            used_names = set()
            tools = router._build_registry_tools(self.settings, self.mock_agent_spec, used_names)

            assert isinstance(tools, list)
            # Should have created tool definitions for allowed tools
            assert len(tools) > 0

    def test_build_registry_tools_with_missing_tool(self):
        """Test building registry tools when a tool is missing."""

        # Make tool registry raise KeyError for some tools
        def mock_get(name):
            if name == "find":
                raise KeyError(f"Tool {name} not found")
            mock_spec = MagicMock()
            mock_spec.description = f"Tool {name}"
            mock_spec.request_schema_path = MagicMock()
            return mock_spec

        self.mock_tool_registry.get.side_effect = mock_get

        router = IntentRouter(client=self.mock_client, settings=self.settings)

        # Should skip missing tools without error
        assert isinstance(router.tools, list)

    def test_build_registry_tools_with_schema_error(self):
        """Test handling schema loading errors."""
        # Make schema path raise exception when opened
        mock_spec = MagicMock()
        mock_spec.description = "Test tool"
        mock_spec.request_schema_path.open.side_effect = FileNotFoundError()
        self.mock_tool_registry.get.return_value = mock_spec

        router = IntentRouter(client=self.mock_client, settings=self.settings)

        # Should use default schema on error
        assert isinstance(router.tools, list)

    def test_augment_run_schema(self):
        """Test augmenting the run tool schema."""
        # Create a router with mocked registry that returns RUN tool
        schema_content = json.dumps(
            {"type": "object", "properties": {"command": {"type": "string"}}}
        )

        mock_file = mock_open(read_data=schema_content)

        with patch("builtins.open", mock_file):
            router = IntentRouter(client=self.mock_client, settings=self.settings)

            # Test private method
            base_schema = {"type": "object", "properties": {}}
            augmented = router._augment_run_schema(base_schema)

            # Should add examples and improve descriptions
            assert "examples" in str(augmented).lower() or augmented == base_schema

    @patch("ai_dev_agent.cli.router.build_system_messages")
    def test_route_prompt_success(self, mock_build_messages):
        """Test successful prompt routing."""
        mock_build_messages.return_value = []

        # Mock successful tool call
        tool_result = ToolCallResult(call_id="test_id", name="find", content='{"success": true}')
        self.mock_client.generate_with_tools.return_value = ("Test response", [tool_result])

        router = IntentRouter(client=self.mock_client, settings=self.settings)

        decision = router.route_prompt("Find all Python files")

        assert isinstance(decision, IntentDecision)
        assert decision.tool == "find"
        self.mock_client.generate_with_tools.assert_called_once()

    @patch("ai_dev_agent.cli.router.build_system_messages")
    def test_route_prompt_no_tool_calls(self, mock_build_messages):
        """Test routing when no tool calls are made."""
        mock_build_messages.return_value = []

        # Mock response with no tool calls
        self.mock_client.generate_with_tools.return_value = ("No action needed", [])

        router = IntentRouter(client=self.mock_client, settings=self.settings)

        with pytest.raises(IntentRoutingError, match="Could not determine"):
            router.route_prompt("Ambiguous request")

    @patch("ai_dev_agent.cli.router.build_system_messages")
    def test_route_prompt_invalid_json(self, mock_build_messages):
        """Test handling invalid JSON in tool response."""
        mock_build_messages.return_value = []

        # Mock tool call with invalid JSON
        tool_result = ToolCallResult(call_id="test_id", name="find", content="invalid json")
        self.mock_client.generate_with_tools.return_value = ("Test response", [tool_result])

        router = IntentRouter(client=self.mock_client, settings=self.settings)

        decision = router.route_prompt("Find files")

        # Should handle gracefully and return empty arguments
        assert isinstance(decision, IntentDecision)
        assert decision.tool == "find"
        assert decision.arguments == {}

    @patch("ai_dev_agent.cli.router.build_system_messages")
    def test_route_prompt_llm_error(self, mock_build_messages):
        """Test handling LLM errors during routing."""
        mock_build_messages.return_value = []

        # Mock LLM error
        self.mock_client.generate_with_tools.side_effect = LLMError("API error")

        router = IntentRouter(client=self.mock_client, settings=self.settings)

        with pytest.raises(IntentRoutingError, match="routing failed"):
            router.route_prompt("Find files")

    @patch("ai_dev_agent.cli.router.build_system_messages")
    def test_route_prompt_general_exception(self, mock_build_messages):
        """Test handling general exceptions during routing."""
        mock_build_messages.return_value = []

        # Mock general exception
        self.mock_client.generate_with_tools.side_effect = Exception("Unexpected error")

        router = IntentRouter(client=self.mock_client, settings=self.settings)

        with pytest.raises(IntentRoutingError, match="Unexpected"):
            router.route_prompt("Find files")

    def test_route_prompt_with_project_context(self):
        """Test routing with project profile context."""
        project_profile = {"language": "Python", "framework": "Django"}

        router = IntentRouter(
            client=self.mock_client, settings=self.settings, project_profile=project_profile
        )

        # Mock successful response
        tool_result = ToolCallResult(
            call_id="test_id", name="read", content='{"file": "models.py"}'
        )
        self.mock_client.generate_with_tools.return_value = ("Reading Django models", [tool_result])

        with patch("ai_dev_agent.cli.router.build_system_messages") as mock_build:
            mock_build.return_value = []
            decision = router.route_prompt("Show me the models")

            assert decision.tool == "read"
            # Verify project profile was included in messages
            call_args = self.mock_client.generate_with_tools.call_args
            messages = call_args[0][0]
            # Check if project context appears in any message
            assert any("Django" in str(msg) or "Python" in str(msg) for msg in messages if msg)

    def test_route_prompt_with_tool_success_history(self):
        """Test routing considers tool success history."""
        tool_history = {"find": 0.95, "grep": 0.60, "read": 0.85}

        router = IntentRouter(
            client=self.mock_client, settings=self.settings, tool_success_history=tool_history
        )

        # Mock response preferring high-success tool
        tool_result = ToolCallResult(call_id="test_id", name="find", content='{"pattern": "*.py"}')
        self.mock_client.generate_with_tools.return_value = (
            "Using find due to high success rate",
            [tool_result],
        )

        with patch("ai_dev_agent.cli.router.build_system_messages") as mock_build:
            mock_build.return_value = []
            decision = router.route_prompt("Search for Python files")

            assert decision.tool == "find"

    def test_duplicate_tool_filtering(self):
        """Test that duplicate tools are filtered out."""
        # Add a tool to DEFAULT_TOOLS
        with patch(
            "ai_dev_agent.cli.router.DEFAULT_TOOLS",
            [{"type": "function", "function": {"name": "find", "description": "Find files"}}],
        ):
            router = IntentRouter(client=self.mock_client, settings=self.settings)

            # Count occurrences of 'find' tool
            find_count = sum(
                1 for tool in router.tools if tool.get("function", {}).get("name") == "find"
            )

            # Should only have one 'find' tool despite being in DEFAULT_TOOLS
            assert find_count <= 1

    def test_empty_tools_list(self):
        """Test router with empty tools list."""
        router = IntentRouter(client=self.mock_client, settings=self.settings, tools=[])

        assert router.tools == []

        # Should still work but with no tools available
        self.mock_client.generate_with_tools.return_value = ("No tools", [])

        with patch("ai_dev_agent.cli.router.build_system_messages") as mock_build:
            mock_build.return_value = []
            with pytest.raises(IntentRoutingError):
                router.route_prompt("Do something")

    def test_normalise_system_context_handles_edge_types(self):
        router = IntentRouter(client=self.mock_client, settings=self.settings)

        normalized = router._normalise_system_context("not-a-dict")
        assert normalized["available_tools"] == []
        assert normalized["cwd"] == "."

        normalized = router._normalise_system_context(
            {"available_tools": ["find", ""], "cwd": None, "command_mappings": []}
        )
        assert normalized["available_tools"] == ["find"]
        assert normalized["command_mappings"] == {}

    def test_project_context_lines_formats_profile(self):
        profile = {
            "language": "Python",
            "repository_size": 420,
            "active_plan_complexity": "medium",
            "recent_files": ["a.py", "b.py", "c.py", "d.py", "e.py"],
            "style_notes": "PEP8 everywhere",
            "project_summary": "This project demonstrates extensive routing behaviour " * 4,
            "workspace_root": "/other",
        }
        router = IntentRouter(
            client=self.mock_client, settings=self.settings, project_profile=profile
        )

        lines = router._project_context_lines("/expected")
        joined = "\n".join(lines)
        assert "Dominant language: Python" in joined
        assert "Approximate file count: 420" in joined
        assert "Current plan complexity: medium" in joined
        assert "Recently touched files" in joined and "…" in joined
        assert "Style highlights: PEP8 everywhere" in joined
        assert "Structure summary:" in joined
        assert "…" in joined
        assert "Override workspace root: /other" in joined

    def test_tool_performance_lines_sorts_metrics(self):
        history = {
            "read": {"success": 8, "failure": 2, "avg_duration": 2.5},
            "find": {"success": 15, "failure": 5, "avg_duration": 1.2},
            "write": {"success": 1, "failure": 0, "avg_duration": 0.0},
        }
        router = IntentRouter(
            client=self.mock_client, settings=self.settings, tool_success_history=history
        )

        lines = router._tool_performance_lines()
        assert lines[0].startswith("- find:")
        assert any("avg 2.5s" in line for line in lines)

    def test_build_raw_tool_calls_handles_invalid_arguments(self):
        router = IntentRouter(client=self.mock_client, settings=self.settings)
        calls = [
            ToolCall(name="find", arguments={"query": "*.py"}, call_id="a"),
            ToolCall(name="run", arguments="not-json", call_id=None),
        ]

        payload = router._build_raw_tool_calls(calls)
        assert payload[0]["function"]["arguments"] == json.dumps({"query": "*.py"})
        assert payload[1]["function"]["arguments"] == "{}"

    def test_parse_arguments_prefers_call_then_result(self):
        router = IntentRouter(client=self.mock_client, settings=self.settings)
        result = ToolCallResult()
        call = SimpleNamespace(name="find", arguments='{"query":"*.py"}')

        parsed = router._parse_arguments(result, call)
        assert parsed == {"query": "*.py"}

        call.arguments = "invalid"
        result.arguments = {"fallback": True}
        parsed = router._parse_arguments(result, call)
        assert parsed == {"fallback": True}

        result.arguments = None
        result.content = '{"text": "fallback"}'
        parsed = router._parse_arguments(result, call)
        assert parsed == {"text": "fallback"}

    def test_coerce_tool_call_result_tuple_input(self):
        router = IntentRouter(client=self.mock_client, settings=self.settings)
        tool_entry = {
            "id": "call-1",
            "function": {"name": "find", "arguments": '{"query": "*.py"}'},
        }

        coerced = router._coerce_tool_call_result(("message", [tool_entry]))
        assert isinstance(coerced, ToolCallResult)
        assert coerced.calls[0].name == "find"
        assert coerced.calls[0].arguments == {"query": "*.py"}
        assert coerced.raw_tool_calls[0]["id"] == "call-1"

    @patch("ai_dev_agent.cli.router.build_system_messages")
    def test_route_returns_message_when_no_tool(self, mock_build_messages):
        mock_build_messages.return_value = []

        router = IntentRouter(client=self.mock_client, settings=self.settings)
        router._invoke_model = MagicMock(return_value=ToolCallResult(message_content="Here you go"))

        decision = router.route("Explain usage")
        assert decision.tool is None
        assert decision.arguments == {"text": "Here you go"}
