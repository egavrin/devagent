"""
Tests for Work Planning Agent

Following TDD: These tests are written BEFORE implementation.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ai_dev_agent.agents.work_planner import (
    Priority,
    Task,
    TaskStatus,
    WorkPlanningAgent,
    WorkPlanStorage,
)


class TestWorkPlanningAgent:
    """Test WorkPlanningAgent"""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def agent(self, temp_storage_dir):
        """Create agent instance with temp storage"""
        storage = WorkPlanStorage(storage_dir=temp_storage_dir)
        return WorkPlanningAgent(storage=storage)

    def test_agent_initialization(self, agent):
        """Test agent can be initialized"""
        assert agent is not None
        assert agent.storage is not None

    def test_create_plan_without_llm(self, agent):
        """Test creating a basic plan without LLM"""
        plan = agent.create_plan(
            goal="Implement authentication system",
            context={"description": "Add user auth with JWT"},
        )

        assert plan is not None
        assert plan.goal == "Implement authentication system"
        assert plan.context == "Add user auth with JWT"
        assert len(plan.tasks) >= 1  # Should create at least one placeholder task
        assert plan.version == 1
        assert plan.is_active is True

    def test_create_plan_is_persisted(self, agent):
        """Test that created plan is saved to storage"""
        plan = agent.create_plan(goal="Test goal")

        # Verify we can load it back
        loaded_plan = agent.storage.load_plan(plan.id)
        assert loaded_plan is not None
        assert loaded_plan.id == plan.id
        assert loaded_plan.goal == "Test goal"

    def test_get_next_task(self, agent):
        """Test getting next task from a plan"""
        # Create plan with tasks
        plan = agent.create_plan(goal="Test goal")

        task1 = Task(title="Task 1", priority=Priority.HIGH)
        task2 = Task(title="Task 2", priority=Priority.MEDIUM)
        plan.tasks = [task1, task2]
        agent.storage.save_plan(plan)

        # Get next task
        next_task = agent.get_next_task(plan.id)

        assert next_task is not None
        assert next_task.title == "Task 1"  # Higher priority
        assert next_task.status == TaskStatus.PENDING

    def test_get_next_task_nonexistent_plan(self, agent):
        """Test getting next task from nonexistent plan"""
        next_task = agent.get_next_task("nonexistent-id")
        assert next_task is None

    def test_mark_task_started(self, agent):
        """Test marking a task as started"""
        # Create plan with task
        plan = agent.create_plan(goal="Test goal")
        task1 = Task(title="Task 1")
        plan.tasks = [task1]
        agent.storage.save_plan(plan)

        # Mark as started
        agent.mark_task_started(plan.id, task1.id)

        # Verify
        loaded_plan = agent.storage.load_plan(plan.id)
        loaded_task = loaded_plan.get_task(task1.id)
        assert loaded_task.status == TaskStatus.IN_PROGRESS
        assert loaded_task.started_at is not None
        assert isinstance(loaded_task.started_at, datetime)

    def test_mark_task_complete(self, agent):
        """Test marking a task as complete"""
        # Create plan with task
        plan = agent.create_plan(goal="Test goal")
        task1 = Task(title="Task 1", status=TaskStatus.IN_PROGRESS)
        plan.tasks = [task1]
        agent.storage.save_plan(plan)

        # Mark as complete
        agent.mark_task_complete(plan.id, task1.id)

        # Verify
        loaded_plan = agent.storage.load_plan(plan.id)
        loaded_task = loaded_plan.get_task(task1.id)
        assert loaded_task.status == TaskStatus.COMPLETED
        assert loaded_task.completed_at is not None
        assert isinstance(loaded_task.completed_at, datetime)

    def test_mark_task_started_invalid_plan(self, agent):
        """Test marking task started with invalid plan ID"""
        with pytest.raises(ValueError, match="Plan not found"):
            agent.mark_task_started("nonexistent-plan", "task-id")

    def test_mark_task_started_invalid_task(self, agent):
        """Test marking task started with invalid task ID"""
        plan = agent.create_plan(goal="Test goal")

        with pytest.raises(ValueError, match="Task not found"):
            agent.mark_task_started(plan.id, "nonexistent-task")

    def test_mark_task_complete_invalid_plan(self, agent):
        """Test marking task complete with invalid plan ID"""
        with pytest.raises(ValueError, match="Plan not found"):
            agent.mark_task_complete("nonexistent-plan", "task-id")

    def test_mark_task_complete_invalid_task(self, agent):
        """Test marking task complete with invalid task ID"""
        plan = agent.create_plan(goal="Test goal")

        with pytest.raises(ValueError, match="Task not found"):
            agent.mark_task_complete(plan.id, "nonexistent-task")

    def test_get_plan_summary(self, agent):
        """Test generating markdown summary of plan"""
        # Create plan with tasks
        plan = agent.create_plan(goal="Build API")

        task1 = Task(
            title="Design schema",
            description="Design database schema",
            priority=Priority.HIGH,
            effort_estimate="2h",
            status=TaskStatus.COMPLETED,
        )

        task2 = Task(
            title="Implement endpoints",
            description="Create REST endpoints",
            priority=Priority.HIGH,
            effort_estimate="4h",
            status=TaskStatus.IN_PROGRESS,
            dependencies=[task1.id],
        )

        task3 = Task(
            title="Write tests",
            description="Add integration tests",
            priority=Priority.MEDIUM,
            effort_estimate="2h",
            status=TaskStatus.PENDING,
            dependencies=[task2.id],
        )

        plan.tasks = [task1, task2, task3]
        agent.storage.save_plan(plan)

        # Get summary
        summary = agent.get_plan_summary(plan.id)

        # Verify summary contains expected elements
        assert "Build API" in summary
        assert "Design schema" in summary
        assert "Implement endpoints" in summary
        assert "Write tests" in summary
        assert "✅" in summary  # Completed task icon
        assert "🔄" in summary  # In progress task icon
        assert "⏳" in summary  # Pending task icon
        assert "2h" in summary  # Effort estimate
        assert "4h" in summary

    def test_get_plan_summary_nonexistent_plan(self, agent):
        """Test getting summary of nonexistent plan"""
        summary = agent.get_plan_summary("nonexistent-id")
        assert summary == "Plan not found"

    def test_update_plan_increments_version(self, agent):
        """Test that updating plan increments version"""
        # Create initial plan
        plan = agent.create_plan(goal="Initial goal")
        initial_version = plan.version

        # Update plan
        updated_plan = agent.update_plan(plan.id, feedback="Add more detail")

        assert updated_plan.version == initial_version + 1

    def test_update_plan_invalid_plan(self, agent):
        """Test updating nonexistent plan raises error"""
        with pytest.raises(ValueError, match="Plan not found"):
            agent.update_plan("nonexistent-id", "feedback")

    def test_workflow_create_and_complete_tasks(self, agent):
        """Test complete workflow: create plan, start task, complete task"""
        # Create plan
        plan = agent.create_plan(goal="Implement feature")

        task1 = Task(title="Step 1", priority=Priority.HIGH)
        task2 = Task(title="Step 2", priority=Priority.HIGH, dependencies=[task1.id])
        plan.tasks = [task1, task2]
        agent.storage.save_plan(plan)

        # Get next task (should be task1)
        next_task = agent.get_next_task(plan.id)
        assert next_task.id == task1.id

        # Start task1
        agent.mark_task_started(plan.id, task1.id)
        loaded_plan = agent.storage.load_plan(plan.id)
        assert loaded_plan.get_task(task1.id).status == TaskStatus.IN_PROGRESS

        # Complete task1
        agent.mark_task_complete(plan.id, task1.id)
        loaded_plan = agent.storage.load_plan(plan.id)
        assert loaded_plan.get_task(task1.id).status == TaskStatus.COMPLETED

        # Get next task (should now be task2 since dependency is met)
        next_task = agent.get_next_task(plan.id)
        assert next_task.id == task2.id

        # Start and complete task2
        agent.mark_task_started(plan.id, task2.id)
        agent.mark_task_complete(plan.id, task2.id)

        # Verify plan is 100% complete
        loaded_plan = agent.storage.load_plan(plan.id)
        assert loaded_plan.get_completion_percentage() == 100.0

        # No more tasks available
        next_task = agent.get_next_task(plan.id)
        assert next_task is None
