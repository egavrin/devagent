"""
Tests for Work Planning Storage

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import tempfile
import json
from pathlib import Path
from ai_dev_agent.agents.work_planner import (
    Task,
    WorkPlan,
    WorkPlanStorage,
    TaskStatus,
    Priority,
)


class TestWorkPlanStorage:
    """Test WorkPlanStorage"""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """Create storage instance with temp directory"""
        return WorkPlanStorage(storage_dir=temp_storage_dir)

    def test_storage_directory_creation(self, temp_storage_dir):
        """Test that storage directory is created if it doesn't exist"""
        storage_path = temp_storage_dir / "plans"
        assert not storage_path.exists()

        WorkPlanStorage(storage_dir=storage_path)

        assert storage_path.exists()
        assert storage_path.is_dir()

    def test_save_plan(self, storage, temp_storage_dir):
        """Test saving a work plan"""
        plan = WorkPlan(name="Test Plan", goal="Test goal")
        task1 = Task(title="Task 1")
        plan.tasks = [task1]

        storage.save_plan(plan)

        # Verify file was created
        plan_file = temp_storage_dir / f"{plan.id}.json"
        assert plan_file.exists()

        # Verify file contents
        with open(plan_file, "r") as f:
            data = json.load(f)
            assert data["name"] == "Test Plan"
            assert data["goal"] == "Test goal"
            assert len(data["tasks"]) == 1

    def test_load_plan(self, storage, temp_storage_dir):
        """Test loading a work plan"""
        # Create and save a plan
        original_plan = WorkPlan(name="Test Plan", goal="Test goal")
        task1 = Task(title="Task 1", priority=Priority.HIGH)
        original_plan.tasks = [task1]
        storage.save_plan(original_plan)

        # Load the plan
        loaded_plan = storage.load_plan(original_plan.id)

        assert loaded_plan is not None
        assert loaded_plan.id == original_plan.id
        assert loaded_plan.name == "Test Plan"
        assert loaded_plan.goal == "Test goal"
        assert len(loaded_plan.tasks) == 1
        assert loaded_plan.tasks[0].title == "Task 1"
        assert loaded_plan.tasks[0].priority == Priority.HIGH

    def test_load_nonexistent_plan(self, storage):
        """Test loading a plan that doesn't exist"""
        loaded_plan = storage.load_plan("nonexistent-id")
        assert loaded_plan is None

    def test_list_plans_empty(self, storage):
        """Test listing plans when directory is empty"""
        plans = storage.list_plans()
        assert plans == []

    def test_list_plans(self, storage):
        """Test listing multiple plans"""
        # Create and save multiple plans
        plan1 = WorkPlan(name="Plan 1", goal="Goal 1")
        plan2 = WorkPlan(name="Plan 2", goal="Goal 2")
        plan3 = WorkPlan(name="Plan 3", goal="Goal 3")

        storage.save_plan(plan1)
        storage.save_plan(plan2)
        storage.save_plan(plan3)

        # List plans
        plans = storage.list_plans()

        assert len(plans) == 3
        plan_names = {p.name for p in plans}
        assert "Plan 1" in plan_names
        assert "Plan 2" in plan_names
        assert "Plan 3" in plan_names

    def test_delete_plan(self, storage, temp_storage_dir):
        """Test deleting a plan"""
        # Create and save a plan
        plan = WorkPlan(name="Test Plan", goal="Test goal")
        storage.save_plan(plan)

        # Verify file exists
        plan_file = temp_storage_dir / f"{plan.id}.json"
        assert plan_file.exists()

        # Delete the plan
        result = storage.delete_plan(plan.id)

        assert result is True
        assert not plan_file.exists()

        # Verify plan can't be loaded
        loaded_plan = storage.load_plan(plan.id)
        assert loaded_plan is None

    def test_delete_nonexistent_plan(self, storage):
        """Test deleting a plan that doesn't exist"""
        result = storage.delete_plan("nonexistent-id")
        assert result is False

    def test_update_plan(self, storage):
        """Test updating an existing plan"""
        # Create and save initial plan
        plan = WorkPlan(name="Original Name", goal="Original Goal")
        task1 = Task(title="Task 1")
        plan.tasks = [task1]
        storage.save_plan(plan)

        # Modify and save again
        plan.name = "Updated Name"
        plan.goal = "Updated Goal"
        task2 = Task(title="Task 2")
        plan.tasks.append(task2)
        storage.save_plan(plan)

        # Load and verify
        loaded_plan = storage.load_plan(plan.id)
        assert loaded_plan.name == "Updated Name"
        assert loaded_plan.goal == "Updated Goal"
        assert len(loaded_plan.tasks) == 2
        assert loaded_plan.tasks[1].title == "Task 2"

    def test_save_and_load_complex_plan(self, storage):
        """Test saving and loading a complex plan with dependencies"""
        # Create complex plan
        task1 = Task(
            title="Task 1",
            description="First task",
            priority=Priority.CRITICAL,
            effort_estimate="2h",
        )

        task2 = Task(
            title="Task 2",
            description="Second task",
            priority=Priority.HIGH,
            dependencies=[task1.id],
            effort_estimate="3h",
            tags=["backend", "api"],
        )

        task3 = Task(
            title="Task 3",
            description="Third task",
            priority=Priority.MEDIUM,
            dependencies=[task1.id, task2.id],
            status=TaskStatus.COMPLETED,
        )

        plan = WorkPlan(
            name="Complex Plan",
            goal="Implement feature X",
            context="This is a complex multi-task plan",
            tasks=[task1, task2, task3],
            version=1,
        )

        # Save
        storage.save_plan(plan)

        # Load
        loaded_plan = storage.load_plan(plan.id)

        # Verify
        assert loaded_plan is not None
        assert loaded_plan.name == "Complex Plan"
        assert len(loaded_plan.tasks) == 3

        # Verify task 1
        loaded_task1 = loaded_plan.get_task(task1.id)
        assert loaded_task1.title == "Task 1"
        assert loaded_task1.priority == Priority.CRITICAL
        assert loaded_task1.effort_estimate == "2h"

        # Verify task 2
        loaded_task2 = loaded_plan.get_task(task2.id)
        assert loaded_task2.title == "Task 2"
        assert loaded_task2.dependencies == [task1.id]
        assert loaded_task2.tags == ["backend", "api"]

        # Verify task 3
        loaded_task3 = loaded_plan.get_task(task3.id)
        assert loaded_task3.status == TaskStatus.COMPLETED
        assert set(loaded_task3.dependencies) == {task1.id, task2.id}
