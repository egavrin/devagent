"""Tests for Agent-Planning System Integration."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from ai_dev_agent.agents.integration.planning_integration import (
    PlanningIntegration,
    TaskAgentMapper,
    AutomatedWorkflow
)
from ai_dev_agent.agents.work_planner.models import WorkPlan, Task, TaskStatus, Priority
from ai_dev_agent.agents.specialized import (
    DesignAgent, TestingAgent, ImplementationAgent, ReviewAgent, OrchestratorAgent
)
from ai_dev_agent.agents.base import AgentContext, AgentResult


class TestTaskAgentMapper:
    """Test task-to-agent mapping."""

    def test_map_design_task(self):
        """Test mapping design tasks to Design Agent."""
        mapper = TaskAgentMapper()

        task = Task(title="Design REST API", description="Create API design")
        agent_name = mapper.map_task_to_agent(task)

        assert agent_name == "design"

    def test_map_test_task(self):
        """Test mapping test tasks to Test Agent."""
        mapper = TaskAgentMapper()

        task = Task(title="Write tests for API", tags=["test"])
        agent_name = mapper.map_task_to_agent(task)

        assert agent_name == "test"

    def test_map_implementation_task(self):
        """Test mapping implementation tasks."""
        mapper = TaskAgentMapper()

        task = Task(title="Implement user service", tags=["implement"])
        agent_name = mapper.map_task_to_agent(task)

        assert agent_name == "implement"

    def test_map_review_task(self):
        """Test mapping review tasks."""
        mapper = TaskAgentMapper()

        task = Task(title="Review code quality", tags=["review"])
        agent_name = mapper.map_task_to_agent(task)

        assert agent_name == "review"

    def test_map_by_keywords(self):
        """Test mapping by keywords in description."""
        mapper = TaskAgentMapper()

        # Design keyword
        task1 = Task(description="Design the authentication system")
        assert mapper.map_task_to_agent(task1) == "design"

        # Test keyword
        task2 = Task(description="Generate unit tests")
        assert mapper.map_task_to_agent(task2) == "test"

        # Implement keyword
        task3 = Task(description="Code the API endpoints")
        assert mapper.map_task_to_agent(task3) == "implement"


class TestPlanningIntegration:
    """Test Planning Integration."""

    def test_initialization(self):
        """Test creating planning integration."""
        integration = PlanningIntegration()

        assert integration is not None
        assert integration.orchestrator is not None

    def test_load_work_plan(self):
        """Test loading a work plan."""
        integration = PlanningIntegration()

        plan = WorkPlan(
            goal="Build REST API",
            tasks=[
                Task(title="Design API", tags=["design"]),
                Task(title="Write tests", tags=["test"]),
            ]
        )

        integration.load_plan(plan)

        assert integration.current_plan == plan

    def test_convert_plan_to_workflow(self):
        """Test converting work plan to agent workflow."""
        integration = PlanningIntegration()

        plan = WorkPlan(
            goal="Build feature",
            tasks=[
                Task(id="1", title="Design", tags=["design"]),
                Task(id="2", title="Test", tags=["test"], dependencies=["1"]),
                Task(id="3", title="Implement", tags=["implement"], dependencies=["2"]),
            ]
        )

        workflow = integration.convert_plan_to_workflow(plan)

        assert "steps" in workflow
        assert len(workflow["steps"]) == 3
        assert workflow["steps"][0]["agent"] == "design"
        assert workflow["steps"][1]["depends_on"] == ["1"]

    def test_execute_plan_with_agents(self):
        """Test executing a plan with agents."""
        integration = PlanningIntegration()
        context = AgentContext(session_id="test-exec")

        plan = WorkPlan(
            goal="Simple task",
            tasks=[Task(id="1", title="Design something", tags=["design"])]
        )

        # Mock the orchestrator
        with patch.object(integration.orchestrator, 'coordinate_workflow') as mock_coord:
            mock_coord.return_value = {
                "success": True,
                "steps_completed": 1,
                "total_steps": 1
            }

            result = integration.execute_plan(plan, context)

            assert result["success"] is True
            assert mock_coord.called

    def test_update_task_status_from_agent_result(self):
        """Test updating task status based on agent result."""
        integration = PlanningIntegration()

        task = Task(id="task-1", title="Test task", status=TaskStatus.PENDING)
        plan = WorkPlan(tasks=[task])
        integration.load_plan(plan)

        # Agent succeeds
        agent_result = AgentResult(success=True, output="Done")
        integration.update_task_from_result(task.id, agent_result)

        assert task.status == TaskStatus.COMPLETED

    def test_handle_agent_failure(self):
        """Test handling when agent fails a task."""
        integration = PlanningIntegration()

        task = Task(id="task-1", status=TaskStatus.IN_PROGRESS)
        plan = WorkPlan(tasks=[task])
        integration.load_plan(plan)

        # Agent fails
        agent_result = AgentResult(success=False, output="", error="Failed")
        integration.update_task_from_result(task.id, agent_result)

        assert task.status == TaskStatus.BLOCKED or task.status == TaskStatus.PENDING

    def test_get_next_task_for_execution(self):
        """Test getting next task respecting dependencies."""
        integration = PlanningIntegration()

        task1 = Task(id="1", title="First", status=TaskStatus.COMPLETED)
        task2 = Task(id="2", title="Second", dependencies=["1"], status=TaskStatus.PENDING)
        task3 = Task(id="3", title="Third", dependencies=["2"], status=TaskStatus.PENDING)

        plan = WorkPlan(tasks=[task1, task2, task3])
        integration.load_plan(plan)

        next_task = integration.get_next_task()

        assert next_task == task2  # task1 is complete, task2 dependencies met

    def test_track_progress(self):
        """Test tracking plan execution progress."""
        integration = PlanningIntegration()

        plan = WorkPlan(
            tasks=[
                Task(status=TaskStatus.COMPLETED),
                Task(status=TaskStatus.IN_PROGRESS),
                Task(status=TaskStatus.PENDING),
            ]
        )
        integration.load_plan(plan)

        progress = integration.get_progress()

        assert progress["total_tasks"] == 3
        assert progress["completed"] == 1
        assert progress["in_progress"] == 1
        assert progress["pending"] == 1
        assert progress["completion_percentage"] == pytest.approx(33.33, 0.1)


class TestAutomatedWorkflow:
    """Test automated workflow execution."""

    def test_create_automated_workflow(self):
        """Test creating automated workflow."""
        workflow = AutomatedWorkflow()

        assert workflow is not None

    def test_execute_plan_automatically(self):
        """Test executing entire plan automatically."""
        workflow = AutomatedWorkflow()
        context = AgentContext(session_id="auto-test")

        plan = WorkPlan(
            goal="Auto execution test",
            tasks=[
                Task(id="1", title="Task 1", tags=["design"]),
                Task(id="2", title="Task 2", tags=["test"], dependencies=["1"]),
            ]
        )

        # Mock agent executions
        with patch('ai_dev_agent.agents.specialized.design_agent.DesignAgent.execute') as mock_design, \
             patch('ai_dev_agent.agents.specialized.testing_agent.TestingAgent.execute') as mock_test:

            mock_design.return_value = AgentResult(success=True, output="Design done")
            mock_test.return_value = AgentResult(success=True, output="Tests done")

            result = workflow.execute_plan_automatically(plan, context)

            assert result["success"] is True
            assert result["tasks_completed"] == 2

    def test_stop_on_failure(self):
        """Test stopping workflow when task fails."""
        workflow = AutomatedWorkflow()
        context = AgentContext(session_id="fail-test")

        plan = WorkPlan(
            tasks=[
                Task(id="1", title="Will fail", tags=["design"]),
                Task(id="2", title="Won't run", tags=["test"], dependencies=["1"]),
            ]
        )

        with patch('ai_dev_agent.agents.specialized.design_agent.DesignAgent.execute') as mock_design:
            mock_design.return_value = AgentResult(success=False, output="", error="Failed")

            result = workflow.execute_plan_automatically(plan, context, stop_on_failure=True)

            assert result["success"] is False
            assert result["tasks_completed"] < 2

    def test_continue_on_failure(self):
        """Test continuing workflow despite failures."""
        workflow = AutomatedWorkflow()
        context = AgentContext(session_id="continue-test")

        plan = WorkPlan(
            tasks=[
                Task(id="1", title="Will fail", tags=["design"]),
                Task(id="2", title="Independent task", tags=["review"]),  # No dependency on task 1
            ]
        )

        with patch('ai_dev_agent.agents.specialized.design_agent.DesignAgent.execute') as mock_design, \
             patch('ai_dev_agent.agents.specialized.review_agent.ReviewAgent.execute') as mock_review:

            mock_design.return_value = AgentResult(success=False, output="", error="Failed")
            mock_review.return_value = AgentResult(success=True, output="Review done")

            result = workflow.execute_plan_automatically(plan, context, stop_on_failure=False)

            # Should complete independent tasks even if one fails
            assert result["tasks_completed"] >= 1

    def test_progress_callback(self):
        """Test progress callback during execution."""
        workflow = AutomatedWorkflow()
        context = AgentContext(session_id="callback-test")

        progress_updates = []

        def progress_callback(task_id, status, message):
            progress_updates.append((task_id, status, message))

        plan = WorkPlan(
            tasks=[Task(id="1", title="Task", tags=["design"])]
        )

        with patch('ai_dev_agent.agents.specialized.design_agent.DesignAgent.execute') as mock:
            mock.return_value = AgentResult(success=True, output="Done")

            workflow.execute_plan_automatically(
                plan,
                context,
                progress_callback=progress_callback
            )

            assert len(progress_updates) > 0