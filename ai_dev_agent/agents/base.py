"""Base agent framework for multi-agent coordination system."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class AgentStatus(Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class AgentError(Exception):
    """Base exception for agent errors."""

    pass


@dataclass
class AgentCapability:
    """Defines a capability that an agent can have."""

    name: str
    description: str = ""
    required_tools: list[str] = field(default_factory=list)
    optional_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_available(self, available_tools: list[str]) -> bool:
        """Check if this capability is available given the tools."""
        return all(tool in available_tools for tool in self.required_tools)


@dataclass
class AgentContext:
    """Execution context for an agent."""

    session_id: str
    parent_id: str | None = None
    working_directory: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.working_directory is None:
            self.working_directory = str(Path.cwd())


@dataclass
class AgentMessage:
    """Message from an agent during execution."""

    agent_name: str
    content: str
    message_type: str = "info"  # info, warning, error, success
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result of agent execution."""

    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class AgentSession:
    """Represents an agent execution session."""

    session_id: str
    agent_name: str
    parent_id: str | None = None
    status: AgentStatus = AgentStatus.IDLE
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    history: list[AgentMessage] = field(default_factory=list)
    result: AgentResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the session history."""
        self.history.append(message)

    def set_result(self, result: AgentResult) -> None:
        """Set the session result."""
        self.result = result
        if result.success:
            self.status = AgentStatus.COMPLETED
        else:
            self.status = AgentStatus.FAILED
        self.completed_at = datetime.now()


class BaseAgent:
    """Base class for all agent types."""

    def __init__(
        self,
        name: str,
        description: str = "",
        capabilities: list[str] | None = None,
        tools: list[str] | None = None,
        max_iterations: int = 30,
        permissions: dict[str, str] | None = None,
        parent_agent: BaseAgent | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name/identifier
            description: Agent description
            capabilities: List of capability names this agent has
            tools: List of tools this agent can use
            max_iterations: Maximum iterations for execution
            permissions: Tool permissions (allow/deny/ask)
            parent_agent: Parent agent if this is a subagent
            metadata: Additional metadata
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.permissions = permissions or {}
        self.parent_agent = parent_agent
        self.metadata = metadata or {}
        self.status = AgentStatus.IDLE
        self._registered_capabilities: dict[str, AgentCapability] = {}
        self._current_session: AgentSession | None = None

        # Default execution methods (can be overridden)
        self._execute_sync: Callable | None = None
        self._execute_async: Callable | None = None

    def has_tool(self, tool_name: str) -> bool:
        """Check if agent has access to a tool."""
        return tool_name in self.tools

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a capability."""
        return capability_name in self._registered_capabilities

    def can_use_tool(self, tool_name: str) -> bool | str:
        """
        Check if agent can use a tool based on permissions.

        Returns:
            True if allowed, False if denied, "ask" if needs permission
        """
        if tool_name in self.permissions:
            permission = self.permissions[tool_name]
            if permission == "allow":
                return True
            elif permission == "deny":
                return False
            else:
                return permission  # "ask"

        # Default: allow if tool is in agent's tool list
        return tool_name in self.tools

    def register_capability(self, capability: AgentCapability) -> None:
        """Register a capability for this agent."""
        self._registered_capabilities[capability.name] = capability
        if capability.name not in self.capabilities:
            self.capabilities.append(capability.name)

    def create_session(self, context: AgentContext) -> AgentSession:
        """Create a new execution session."""
        session = AgentSession(
            session_id=context.session_id,
            agent_name=self.name,
            parent_id=context.parent_id,
            metadata=context.metadata.copy(),
        )
        self._current_session = session
        return session

    def create_child_context(self, parent_context: AgentContext) -> AgentContext:
        """Create a child context from parent context."""
        return AgentContext(
            session_id=f"{parent_context.session_id}-{self.name}",
            parent_id=parent_context.session_id,
            working_directory=parent_context.working_directory,
            environment=parent_context.environment.copy(),
            metadata=parent_context.metadata.copy(),
        )

    def set_status(self, status: AgentStatus) -> None:
        """Set the agent status."""
        self.status = status
        if self._current_session:
            self._current_session.status = status

    def create_message(
        self, content: str, message_type: str = "info", metadata: dict[str, Any] | None = None
    ) -> AgentMessage:
        """Create an agent message."""
        return AgentMessage(
            agent_name=self.name,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )

    async def execute_async(self, prompt: str, context: AgentContext) -> AgentResult:
        """
        Execute agent task asynchronously.

        Args:
            prompt: Task prompt/instruction
            context: Execution context

        Returns:
            AgentResult with execution outcome
        """
        if self._execute_async:
            try:
                self.set_status(AgentStatus.RUNNING)
                result = await self._execute_async(prompt, context)
                self.set_status(AgentStatus.COMPLETED if result.success else AgentStatus.FAILED)
                return result
            except Exception as e:
                self.set_status(AgentStatus.FAILED)
                return AgentResult(success=False, output="", error=str(e))
        else:
            # Fallback to sync execution in async context
            return await asyncio.get_event_loop().run_in_executor(
                None, self.execute, prompt, context
            )

    def execute(self, prompt: str, context: AgentContext) -> AgentResult:
        """
        Execute agent task synchronously.

        Args:
            prompt: Task prompt/instruction
            context: Execution context

        Returns:
            AgentResult with execution outcome
        """
        try:
            self.set_status(AgentStatus.RUNNING)

            if self._execute_sync:
                result = self._execute_sync(prompt, context)
            else:
                # Default implementation (should be overridden)
                result = self._default_execute(prompt, context)

            self.set_status(AgentStatus.COMPLETED if result.success else AgentStatus.FAILED)
            return result

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            return AgentResult(success=False, output="", error=str(e))

    def _default_execute(self, prompt: str, context: AgentContext) -> AgentResult:
        """
        Default execution implementation.
        Should be overridden by specific agent types.
        """
        return AgentResult(
            success=False, output="", error=f"Agent {self.name} has no execution implementation"
        )

    def validate_tools(self, required_tools: list[str]) -> bool:
        """Validate that agent has required tools."""
        return all(self.has_tool(tool) for tool in required_tools)

    def get_capability(self, name: str) -> AgentCapability | None:
        """Get a registered capability by name."""
        return self._registered_capabilities.get(name)

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"<{self.__class__.__name__}(name={self.name}, status={self.status})>"
