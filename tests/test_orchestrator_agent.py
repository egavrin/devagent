"""Tests for Orchestrator Agent."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from ai_dev_agent.agents.specialized.orchestrator_agent import OrchestratorAgent
from ai_dev_agent.agents.specialized import DesignAgent, TestingAgent, ImplementationAgent, ReviewAgent
from ai_dev_agent.agents.base import AgentContext, AgentResult


class TestOrchestratorAgent:
    """Test Orchestrator Agent functionality."""

    def test_orchestrator_initialization(self):
        """Test creating an orchestrator agent."""
        agent = OrchestratorAgent()

        assert agent.name == "orchestrator_agent"
        assert "coordination" in agent.capabilities
        assert "task_delegation" in agent.capabilities
        assert "workflow_management" in agent.capabilities

    def test_register_subagent(self):
        """Test registering subagents."""
        orchestrator = OrchestratorAgent()
        design_agent = DesignAgent()

        orchestrator.register_subagent("design", design_agent)

        assert orchestrator.has_subagent("design")
        assert orchestrator.get_subagent("design") == design_agent

    def test_delegate_task(self):
        """Test delegating a task to a subagent."""
        orchestrator = OrchestratorAgent()
        context = AgentContext(session_id="test-delegate")

        # Register a mock agent
        mock_agent = Mock()
        mock_agent.execute.return_value = AgentResult(
            success=True,
            output="Task completed",
            metadata={"result": "success"}
        )

        orchestrator.register_subagent("mock", mock_agent)

        result = orchestrator.delegate_task(
            agent_name="mock",
            task="Do something",
            context=context
        )

        assert result.success is True
        assert mock_agent.execute.called

    def test_coordinate_workflow(self):
        """Test coordinating a multi-agent workflow."""
        orchestrator = OrchestratorAgent()
        context = AgentContext(session_id="test-workflow")

        # Define a workflow
        workflow = {
            "steps": [
                {"agent": "design", "task": "Create design"},
                {"agent": "test", "task": "Generate tests"},
                {"agent": "implement", "task": "Implement code"}
            ]
        }

        # Register mock agents
        for agent_name in ["design", "test", "implement"]:
            mock_agent = Mock()
            mock_agent.execute.return_value = AgentResult(
                success=True,
                output=f"{agent_name} completed"
            )
            orchestrator.register_subagent(agent_name, mock_agent)

        result = orchestrator.coordinate_workflow(workflow, context)

        assert result["success"] is True
        assert result["steps_completed"] == 3

    def test_parallel_execution(self):
        """Test executing multiple agents in parallel."""
        orchestrator = OrchestratorAgent()
        context = AgentContext(session_id="test-parallel")

        # Tasks that can run in parallel
        tasks = [
            {"agent": "agent1", "task": "Task 1"},
            {"agent": "agent2", "task": "Task 2"},
            {"agent": "agent3", "task": "Task 3"}
        ]

        # Register mock agents
        for i in range(1, 4):
            mock_agent = Mock()
            mock_agent.execute.return_value = AgentResult(
                success=True,
                output=f"Agent {i} done"
            )
            orchestrator.register_subagent(f"agent{i}", mock_agent)

        results = orchestrator.execute_parallel(tasks, context)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_sequential_execution(self):
        """Test executing agents sequentially with dependencies."""
        orchestrator = OrchestratorAgent()
        context = AgentContext(session_id="test-sequential")

        # Sequential tasks with dependencies
        tasks = [
            {"agent": "design", "task": "Create design", "depends_on": []},
            {"agent": "test", "task": "Write tests", "depends_on": ["design"]},
            {"agent": "implement", "task": "Code", "depends_on": ["test"]}
        ]

        # Register mock agents
        for agent_name in ["design", "test", "implement"]:
            mock_agent = Mock()
            mock_agent.execute.return_value = AgentResult(
                success=True,
                output=f"{agent_name} done"
            )
            orchestrator.register_subagent(agent_name, mock_agent)

        results = orchestrator.execute_sequential(tasks, context)

        assert len(results) == 3
        # Verify execution order
        assert all(r.success for r in results)

    def test_handle_agent_failure(self):
        """Test handling agent failures in workflow."""
        orchestrator = OrchestratorAgent()
        context = AgentContext(session_id="test-failure")

        # Agent that fails
        failing_agent = Mock()
        failing_agent.execute.return_value = AgentResult(
            success=False,
            output="",
            error="Agent failed"
        )

        orchestrator.register_subagent("failing", failing_agent)

        result = orchestrator.delegate_task(
            agent_name="failing",
            task="Will fail",
            context=context
        )

        assert result.success is False
        assert "fail" in result.error.lower()

    def test_retry_on_failure(self):
        """Test retrying failed tasks."""
        orchestrator = OrchestratorAgent()
        context = AgentContext(session_id="test-retry")

        # Agent that fails first, then succeeds
        mock_agent = Mock()
        mock_agent.execute.side_effect = [
            AgentResult(success=False, output="", error="First fail"),
            AgentResult(success=True, output="Success on retry")
        ]

        orchestrator.register_subagent("retry_agent", mock_agent)

        result = orchestrator.delegate_with_retry(
            agent_name="retry_agent",
            task="Task",
            context=context,
            max_retries=2
        )

        assert result.success is True
        assert mock_agent.execute.call_count == 2

    def test_select_best_agent(self):
        """Test selecting the best agent for a task."""
        orchestrator = OrchestratorAgent()

        # Register agents with different capabilities
        design_agent = DesignAgent()
        test_agent = TestingAgent()
        review_agent = ReviewAgent()

        orchestrator.register_subagent("design", design_agent)
        orchestrator.register_subagent("test", test_agent)
        orchestrator.register_subagent("review", review_agent)

        # Task requiring design capability
        selected = orchestrator.select_agent_for_task(
            task_type="create_design",
            required_capabilities=["technical_design"]
        )

        assert selected == "design"

    def test_aggregate_results(self):
        """Test aggregating results from multiple agents."""
        orchestrator = OrchestratorAgent()

        results = [
            AgentResult(success=True, output="Result 1", metadata={"score": 0.9}),
            AgentResult(success=True, output="Result 2", metadata={"score": 0.8}),
            AgentResult(success=False, output="", error="Failed", metadata={"score": 0.0})
        ]

        aggregated = orchestrator.aggregate_results(results)

        assert aggregated["total_tasks"] == 3
        assert aggregated["successful"] == 2
        assert aggregated["failed"] == 1
        assert aggregated["success_rate"] == pytest.approx(0.67, 0.01)

    def test_create_workflow_from_plan(self):
        """Test creating a workflow from a work plan."""
        orchestrator = OrchestratorAgent()

        plan = {
            "goal": "Build REST API",
            "tasks": [
                {"id": "1", "description": "Design API", "type": "design"},
                {"id": "2", "description": "Write tests", "type": "test", "depends_on": ["1"]},
                {"id": "3", "description": "Implement", "type": "implement", "depends_on": ["2"]}
            ]
        }

        workflow = orchestrator.create_workflow_from_plan(plan)

        assert "steps" in workflow
        assert len(workflow["steps"]) == 3
        assert workflow["steps"][0]["agent"] == "design"
        assert workflow["steps"][1]["depends_on"] == ["1"]

    def test_monitor_progress(self):
        """Test monitoring workflow progress."""
        orchestrator = OrchestratorAgent()
        context = AgentContext(session_id="test-monitor")

        workflow_id = "workflow-123"

        # Start tracking
        orchestrator.start_workflow(workflow_id, total_steps=5)

        # Update progress
        orchestrator.update_progress(workflow_id, completed=2)

        status = orchestrator.get_workflow_status(workflow_id)

        assert status["workflow_id"] == workflow_id
        assert status["total_steps"] == 5
        assert status["completed"] == 2
        assert status["progress_percentage"] == 40.0

    def test_orchestrator_execute(self):
        """Test full orchestrator execution."""
        orchestrator = OrchestratorAgent()
        context = AgentContext(session_id="test-execute")

        # Register all specialized agents
        orchestrator.register_subagent("design", DesignAgent())
        orchestrator.register_subagent("test", TestingAgent())
        orchestrator.register_subagent("implement", ImplementationAgent())
        orchestrator.register_subagent("review", ReviewAgent())

        prompt = """
        Coordinate implementation of User Authentication feature:
        1. Design the authentication system
        2. Generate tests
        3. Implement the code
        4. Review for security
        """

        with patch.object(DesignAgent, 'execute') as mock_design, \
             patch.object(TestingAgent, 'execute') as mock_test, \
             patch.object(ImplementationAgent, 'execute') as mock_impl, \
             patch.object(ReviewAgent, 'execute') as mock_review:

            # Mock all agent executions
            mock_design.return_value = AgentResult(success=True, output="Design done")
            mock_test.return_value = AgentResult(success=True, output="Tests done")
            mock_impl.return_value = AgentResult(success=True, output="Implementation done")
            mock_review.return_value = AgentResult(success=True, output="Review done")

            result = orchestrator.execute(prompt, context)

            assert result.success is True
            assert "workflow_completed" in result.metadata