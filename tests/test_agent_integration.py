"""Integration tests for multi-agent workflows and coordination."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
import time

from ai_dev_agent.agents.base import AgentContext, AgentResult, AgentCapability
from ai_dev_agent.agents.communication.bus import AgentBus, AgentEvent, EventType
from ai_dev_agent.agents.integration.planning_integration import (
    TaskAgentMapper,
    PlanningIntegration,
)
from ai_dev_agent.agents.work_planner.models import WorkPlan, Task, TaskStatus, Priority


class TestAgentCommunicationBus:
    """Test agent communication bus workflows."""

    def test_bus_start_and_stop(self):
        """Test bus lifecycle management."""
        bus = AgentBus()

        assert not bus.is_running

        bus.start()
        assert bus.is_running

        bus.stop()
        assert not bus.is_running

    def test_publish_event(self):
        """Test event publishing."""
        bus = AgentBus()
        bus.start()

        event = AgentEvent(
            event_type=EventType.TASK_STARTED,
            source_agent="test_agent",
            data={"task_id": "test-123"}
        )

        # Should not raise error
        bus.publish(event)

        # Give bus time to process
        time.sleep(0.1)

        bus.stop()

        # Check metrics
        assert bus._metrics["events_published"] == 1

    def test_context_manager(self):
        """Test bus as context manager."""
        with AgentBus() as bus:
            assert bus.is_running

            event = AgentEvent(
                event_type=EventType.MESSAGE,
                source_agent="sender",
                data={"content": "hello"}
            )

            bus.publish(event)
            time.sleep(0.1)

        # Should be stopped after context exit
        assert not bus.is_running

    def test_event_types(self):
        """Test different event types can be published."""
        bus = AgentBus()
        bus.start()

        event_types = [
            EventType.TASK_STARTED,
            EventType.TASK_COMPLETED,
            EventType.PROGRESS_UPDATE,
            EventType.ERROR,
            EventType.MESSAGE
        ]

        for evt_type in event_types:
            bus.publish(AgentEvent(
                event_type=evt_type,
                source_agent="test",
                data={}
            ))

        time.sleep(0.2)
        bus.stop()

        # All events should be published
        assert bus._metrics["events_published"] == len(event_types)

    def test_targeted_message(self):
        """Test creating targeted message to specific agent."""
        event = AgentEvent(
            event_type=EventType.MESSAGE,
            source_agent="orchestrator",
            target_agent="agent1",
            data={"instruction": "do something"}
        )

        assert event.target_agent == "agent1"
        assert event.source_agent == "orchestrator"
        assert event.data["instruction"] == "do something"

    def test_subscribe_and_unsubscribe(self):
        """Test subscribing to and unsubscribing from bus events."""
        bus = AgentBus()
        bus.start()

        received: list[AgentEvent] = []

        subscription_id = bus.subscribe(EventType.ERROR, received.append)
        assert subscription_id

        bus.publish(AgentEvent(
            event_type=EventType.ERROR,
            source_agent="implementation_agent",
            data={"error": "failure"}
        ))

        time.sleep(0.1)
        assert len(received) == 1
        assert received[0].data["error"] == "failure"

        # Unsubscribe and ensure no more events are received
        bus.unsubscribe(subscription_id)

        bus.publish(AgentEvent(
            event_type=EventType.ERROR,
            source_agent="implementation_agent",
            data={"error": "again"}
        ))
        time.sleep(0.1)

        assert len(received) == 1

        bus.stop()


class TestTaskAgentMapper:
    """Test task to agent mapping logic."""

    def test_map_by_tag(self):
        """Test mapping task to agent based on tags."""
        mapper = TaskAgentMapper()

        task = Task(
            id="test-1",
            title="Some task",
            description="Do something",
            tags=["design"],
            status=TaskStatus.PENDING,
            priority=Priority.MEDIUM
        )

        agent = mapper.map_task_to_agent(task)
        assert agent == "design"

    def test_map_by_title_keyword(self):
        """Test mapping based on title keywords."""
        mapper = TaskAgentMapper()

        task = Task(
            id="test-2",
            title="Implement user authentication",
            description="Add auth",
            status=TaskStatus.PENDING,
            priority=Priority.HIGH
        )

        agent = mapper.map_task_to_agent(task)
        assert agent == "implement"

    def test_map_by_description_keyword(self):
        """Test mapping based on description keywords."""
        mapper = TaskAgentMapper()

        task = Task(
            id="test-3",
            title="User feature",
            description="Write tests for the new feature",
            status=TaskStatus.PENDING,
            priority=Priority.MEDIUM
        )

        agent = mapper.map_task_to_agent(task)
        assert agent == "test"

    def test_map_default_to_implement(self):
        """Test default mapping to implement agent."""
        mapper = TaskAgentMapper()

        task = Task(
            id="test-4",
            title="Generic task",
            description="Do something generic",
            status=TaskStatus.PENDING,
            priority=Priority.LOW
        )

        agent = mapper.map_task_to_agent(task)
        assert agent == "implement"

    def test_map_review_task(self):
        """Test mapping review task."""
        mapper = TaskAgentMapper()

        task = Task(
            id="test-5",
            title="Review PR #123",
            description="Analyze the changes carefully",
            status=TaskStatus.PENDING,
            priority=Priority.HIGH
        )

        agent = mapper.map_task_to_agent(task)
        assert agent == "review"

    def test_review_priority_over_implement_keywords(self):
        """Review keywords should take precedence when mixed with implement terms."""
        mapper = TaskAgentMapper()

        task = Task(
            id="test-6",
            title="Code review for implementation",
            description="Review the implemented feature",
            status=TaskStatus.PENDING,
            priority=Priority.MEDIUM
        )

        agent = mapper.map_task_to_agent(task)
        assert agent == "review"


class TestPlanningIntegration:
    """Test planning integration workflows."""

    @patch('ai_dev_agent.agents.integration.planning_integration.OrchestratorAgent')
    def test_initialization(self, mock_orchestrator_class):
        """Test planning integration initializes correctly."""
        integration = PlanningIntegration()

        assert integration.mapper is not None
        assert integration.orchestrator is not None
        assert integration.current_plan is None

    @patch('ai_dev_agent.agents.integration.planning_integration.OrchestratorAgent')
    def test_load_plan(self, mock_orchestrator_class):
        """Test loading a work plan."""
        integration = PlanningIntegration()

        plan = WorkPlan(
            id="plan-1",
            goal="Build feature",
            tasks=[]
        )

        integration.current_plan = plan

        assert integration.current_plan == plan
        assert integration.current_plan.id == "plan-1"


class TestAgentContext:
    """Test AgentContext for integration scenarios."""

    def test_context_creation(self):
        """Test creating agent context."""
        context = AgentContext(
            session_id="test-session",
            working_directory="/tmp/test",
            metadata={"key": "value"}
        )

        assert context.session_id == "test-session"
        assert context.working_directory == "/tmp/test"
        assert context.metadata["key"] == "value"

    def test_context_with_bus(self):
        """Test agent context with communication bus."""
        bus = AgentBus()

        context = AgentContext(
            session_id="test-session",
            metadata={"bus": bus}
        )

        # Context can store bus reference
        assert context.metadata["bus"] == bus


class TestAgentResult:
    """Test AgentResult for integration scenarios."""

    def test_success_result(self):
        """Test creating successful agent result."""
        result = AgentResult(
            success=True,
            output="Task completed successfully",
            metadata={"duration": 1.5}
        )

        assert result.success is True
        assert result.output == "Task completed successfully"
        assert result.error is None

    def test_failure_result(self):
        """Test creating failure agent result."""
        result = AgentResult(
            success=False,
            output="",
            error="Task failed due to error"
        )

        assert result.success is False
        assert result.error == "Task failed due to error"

    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = AgentResult(
            success=True,
            output="Generated code",
            metadata={"files": ["file1.py", "file2.py"], "changes": 5}
        )

        assert result.metadata["files"] == ["file1.py", "file2.py"]
        assert result.metadata["changes"] == 5


class TestEndToEndWorkflow:
    """Test end-to-end agent workflows."""

    def test_task_creation_to_assignment_flow(self):
        """Test flow from task creation to agent assignment."""
        # Create task
        task = Task(
            id="e2e-1",
            title="Design API endpoints",
            description="Design REST API",
            tags=["design"],
            status=TaskStatus.PENDING,
            priority=Priority.HIGH
        )

        # Map to agent
        mapper = TaskAgentMapper()
        agent_name = mapper.map_task_to_agent(task)

        assert agent_name == "design"
        assert task.status == TaskStatus.PENDING

    def test_event_flow_through_bus(self):
        """Test event flowing through bus from start to completion."""
        bus = AgentBus()
        bus.start()

        # Simulate workflow
        bus.publish(AgentEvent(
            event_type=EventType.TASK_STARTED,
            source_agent="design_agent",
            data={"task_id": "task-1"}
        ))

        bus.publish(AgentEvent(
            event_type=EventType.PROGRESS_UPDATE,
            source_agent="design_agent",
            data={"progress": 50}
        ))

        bus.publish(AgentEvent(
            event_type=EventType.TASK_COMPLETED,
            source_agent="design_agent",
            data={"task_id": "task-1", "result": "success"}
        ))

        time.sleep(0.2)
        bus.stop()

        # All events should be published and processed
        assert bus._metrics["events_published"] == 3
        assert bus._metrics["events_processed"] == 3

    @patch('ai_dev_agent.agents.integration.planning_integration.OrchestratorAgent')
    def test_plan_execution_coordination(self, mock_orchestrator_class):
        """Test coordinating plan execution with multiple tasks."""
        integration = PlanningIntegration()

        # Create plan with multiple tasks
        tasks = [
            Task(
                id="task-1",
                title="Design system",
                description="Create design",
                tags=["design"],
                status=TaskStatus.PENDING,
                priority=Priority.HIGH
            ),
            Task(
                id="task-2",
                title="Implement system",
                description="Write code",
                tags=["implement"],
                status=TaskStatus.PENDING,
                priority=Priority.MEDIUM,
                dependencies=["task-1"]
            ),
            Task(
                id="task-3",
                title="Test system",
                description="Write tests",
                tags=["test"],
                status=TaskStatus.PENDING,
                priority=Priority.MEDIUM,
                dependencies=["task-2"]
            )
        ]

        plan = WorkPlan(
            id="integration-plan",
            goal="Build complete system",
            tasks=tasks
        )

        integration.current_plan = plan

        # Verify tasks are mapped correctly
        mapper = TaskAgentMapper()
        assert mapper.map_task_to_agent(tasks[0]) == "design"
        assert mapper.map_task_to_agent(tasks[1]) == "implement"
        assert mapper.map_task_to_agent(tasks[2]) == "test"

        # Verify dependencies
        assert "task-1" in tasks[1].dependencies
        assert "task-2" in tasks[2].dependencies


class TestErrorPropagation:
    """Test error handling and propagation across agents."""

    def test_error_event_publishing(self):
        """Test that error events are published correctly."""
        bus = AgentBus()
        bus.start()

        # Simulate agent error
        bus.publish(AgentEvent(
            event_type=EventType.ERROR,
            source_agent="implementation_agent",
            data={
                "error": "Compilation failed",
                "details": "Syntax error in file.py"
            }
        ))

        time.sleep(0.1)
        bus.stop()

        # Event should be published and processed
        assert bus._metrics["events_published"] == 1
        assert bus._metrics["events_processed"] == 1

    def test_agent_result_error_handling(self):
        """Test agent result carries error information."""
        result = AgentResult(
            success=False,
            output="",
            error="Failed to execute task",
            metadata={"error_code": "EXEC_FAIL"}
        )

        assert result.success is False
        assert result.error is not None
        assert result.metadata["error_code"] == "EXEC_FAIL"
