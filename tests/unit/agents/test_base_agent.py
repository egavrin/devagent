"""Tests for base agent framework."""

import pytest

from ai_dev_agent.agents.base import (
    AgentCapability,
    AgentContext,
    AgentMessage,
    AgentResult,
    AgentSession,
    AgentStatus,
    BaseAgent,
)

pytest_plugins = ("pytest_asyncio",)


class TestAgentCapability:
    """Test agent capability registration and management."""

    def test_capability_creation(self):
        """Test creating a capability."""
        capability = AgentCapability(
            name="code_analysis",
            description="Analyze code quality",
            required_tools=["read", "grep"],
            optional_tools=["symbols"],
        )

        assert capability.name == "code_analysis"
        assert capability.description == "Analyze code quality"
        assert capability.required_tools == ["read", "grep"]
        assert capability.optional_tools == ["symbols"]

    def test_capability_validation(self):
        """Test capability validates tool availability."""
        capability = AgentCapability(
            name="test", required_tools=["read", "edit"], optional_tools=["run"]
        )

        # All required tools present
        assert capability.is_available(["read", "edit", "run"])
        assert capability.is_available(["read", "edit"])

        # Missing required tool
        assert not capability.is_available(["read"])
        assert not capability.is_available(["edit"])
        assert not capability.is_available([])


class TestAgentContext:
    """Test agent execution context."""

    def test_context_creation(self):
        """Test creating an agent context."""
        context = AgentContext(
            session_id="test-123",
            parent_id="parent-456",
            working_directory="/test/dir",
            environment={"TEST": "value"},
            metadata={"key": "value"},
        )

        assert context.session_id == "test-123"
        assert context.parent_id == "parent-456"
        assert context.working_directory == "/test/dir"
        assert context.environment["TEST"] == "value"
        assert context.metadata["key"] == "value"

    def test_context_defaults(self):
        """Test context with default values."""
        context = AgentContext(session_id="test")

        assert context.session_id == "test"
        assert context.parent_id is None
        assert context.working_directory is not None  # Should have a default
        assert context.environment == {}
        assert context.metadata == {}


class TestAgentResult:
    """Test agent execution results."""

    def test_success_result(self):
        """Test creating a success result."""
        result = AgentResult(
            success=True,
            output="Task completed",
            metadata={"lines_changed": 10},
            tool_calls=[{"tool": "edit", "file": "test.py"}],
        )

        assert result.success is True
        assert result.output == "Task completed"
        assert result.metadata["lines_changed"] == 10
        assert len(result.tool_calls) == 1
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = AgentResult(
            success=False, output="", error="Failed to compile", metadata={"error_code": 1}
        )

        assert result.success is False
        assert result.error == "Failed to compile"
        assert result.metadata["error_code"] == 1


class TestBaseAgent:
    """Test base agent functionality."""

    def test_agent_initialization(self):
        """Test creating a base agent."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent for unit tests",
            capabilities=["code_review", "test_generation"],
        )

        assert agent.name == "test_agent"
        assert agent.description == "Test agent for unit tests"
        assert "code_review" in agent.capabilities
        assert "test_generation" in agent.capabilities
        assert agent.status == AgentStatus.IDLE

    def test_agent_with_tools(self):
        """Test agent with tool permissions."""
        agent = BaseAgent(name="test", tools=["read", "edit", "grep"], max_iterations=10)

        assert agent.has_tool("read")
        assert agent.has_tool("edit")
        assert not agent.has_tool("run")
        assert agent.max_iterations == 10

    def test_agent_capability_registration(self):
        """Test registering capabilities."""
        agent = BaseAgent(name="test")

        capability = AgentCapability(name="analysis", required_tools=["read"])

        agent.register_capability(capability)
        assert agent.has_capability("analysis")
        assert not agent.has_capability("unknown")

    def test_agent_session_creation(self):
        """Test creating agent sessions."""
        agent = BaseAgent(name="test")
        context = AgentContext(session_id="test-123")

        session = agent.create_session(context)

        assert session.agent_name == "test"
        assert session.session_id == "test-123"
        assert session.status == AgentStatus.IDLE
        assert session.parent_id == context.parent_id

    @pytest.mark.asyncio
    async def test_agent_execute_async(self):
        """Test async agent execution."""
        agent = BaseAgent(name="test")

        # Mock the execution
        async def mock_execute(prompt, context):
            return AgentResult(success=True, output="Executed successfully")

        agent._execute_async = mock_execute

        context = AgentContext(session_id="test")
        result = await agent.execute_async("Do something", context)

        assert result.success is True
        assert result.output == "Executed successfully"

    def test_agent_execute_sync(self):
        """Test synchronous agent execution."""
        agent = BaseAgent(name="test")

        # Mock the execution
        def mock_execute(prompt, context):
            return AgentResult(success=True, output="Sync execution")

        agent._execute_sync = mock_execute

        context = AgentContext(session_id="test")
        result = agent.execute("Do something", context)

        assert result.success is True
        assert result.output == "Sync execution"

    def test_agent_status_transitions(self):
        """Test agent status transitions."""
        agent = BaseAgent(name="test")

        assert agent.status == AgentStatus.IDLE

        agent.set_status(AgentStatus.RUNNING)
        assert agent.status == AgentStatus.RUNNING

        agent.set_status(AgentStatus.COMPLETED)
        assert agent.status == AgentStatus.COMPLETED

        agent.set_status(AgentStatus.FAILED)
        assert agent.status == AgentStatus.FAILED

    def test_agent_message_handling(self):
        """Test agent message creation and handling."""
        agent = BaseAgent(name="test")

        message = agent.create_message(
            content="Processing task", message_type="info", metadata={"progress": 50}
        )

        assert message.agent_name == "test"
        assert message.content == "Processing task"
        assert message.message_type == "info"
        assert message.metadata["progress"] == 50

    def test_agent_error_handling(self):
        """Test agent error handling."""
        agent = BaseAgent(name="test")

        def failing_execute(prompt, context):
            raise ValueError("Invalid input")

        agent._execute_sync = failing_execute

        context = AgentContext(session_id="test")
        result = agent.execute("Bad input", context)

        assert result.success is False
        assert "Invalid input" in result.error

    def test_agent_permission_checking(self):
        """Test agent permission system."""
        agent = BaseAgent(
            name="test",
            tools=["read", "grep"],
            permissions={"write": "deny", "run": "ask", "read": "allow"},
        )

        assert agent.can_use_tool("read") is True
        assert agent.can_use_tool("write") is False
        assert agent.can_use_tool("run") == "ask"
        assert agent.can_use_tool("unknown") is False

    def test_agent_with_parent_child_relationship(self):
        """Test parent-child agent relationships."""
        parent_agent = BaseAgent(name="parent")
        child_agent = BaseAgent(name="child", parent_agent=parent_agent)

        assert child_agent.parent_agent == parent_agent
        assert child_agent.name == "child"

        # Child should inherit some properties from parent
        parent_agent.set_status(AgentStatus.RUNNING)
        child_context = child_agent.create_child_context(
            parent_context=AgentContext(session_id="parent-123")
        )
        assert child_context.parent_id == "parent-123"


class TestAgentSession:
    """Test agent session management."""

    def test_session_creation(self):
        """Test creating an agent session."""
        session = AgentSession(
            session_id="test-123", agent_name="test_agent", parent_id="parent-456"
        )

        assert session.session_id == "test-123"
        assert session.agent_name == "test_agent"
        assert session.parent_id == "parent-456"
        assert session.status == AgentStatus.IDLE
        assert session.history == []

    def test_session_message_history(self):
        """Test session message history."""
        session = AgentSession(session_id="test", agent_name="test")

        message1 = AgentMessage(agent_name="test", content="Starting task", message_type="info")
        message2 = AgentMessage(agent_name="test", content="Task completed", message_type="success")

        session.add_message(message1)
        session.add_message(message2)

        assert len(session.history) == 2
        assert session.history[0].content == "Starting task"
        assert session.history[1].content == "Task completed"

    def test_session_result_tracking(self):
        """Test session result tracking."""
        session = AgentSession(session_id="test", agent_name="test")

        result = AgentResult(success=True, output="Completed", metadata={"tasks": 5})

        session.set_result(result)
        assert session.result == result
        assert session.result.success is True
        assert session.result.metadata["tasks"] == 5
