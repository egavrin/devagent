"""
Tests for Work Planning Agent Models

Following TDD: These tests are written BEFORE implementation.
"""

from datetime import datetime

from ai_dev_agent.agents.work_planner import Priority, Task, TaskStatus, WorkPlan


class TestTaskStatus:
    """Test TaskStatus enum"""

    def test_task_status_values(self):
        """Test all task status values exist"""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.BLOCKED == "blocked"
        assert TaskStatus.CANCELLED == "cancelled"


class TestPriority:
    """Test Priority enum"""

    def test_priority_values(self):
        """Test all priority values exist"""
        assert Priority.CRITICAL == "critical"
        assert Priority.HIGH == "high"
        assert Priority.MEDIUM == "medium"
        assert Priority.LOW == "low"


class TestTask:
    """Test Task model"""

    def test_task_creation_with_defaults(self):
        """Test task creation with default values"""
        task = Task(title="Test task")

        assert task.title == "Test task"
        assert task.description == ""
        assert task.status == TaskStatus.PENDING
        assert task.priority == Priority.MEDIUM
        assert task.effort_estimate == "unknown"
        assert task.dependencies == []
        assert task.parent_id is None
        assert task.tags == []
        assert task.acceptance_criteria == []
        assert task.notes == []
        assert task.files_involved == []

        # Check auto-generated fields
        assert len(task.id) > 0
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)
        assert task.started_at is None
        assert task.completed_at is None

    def test_task_creation_with_all_fields(self):
        """Test task creation with all fields specified"""
        task = Task(
            title="Implement feature X",
            description="Detailed description here",
            status=TaskStatus.IN_PROGRESS,
            priority=Priority.HIGH,
            effort_estimate="2h",
            dependencies=["task-1", "task-2"],
            parent_id="parent-task",
            tags=["backend", "api"],
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            notes=["Note 1"],
            files_involved=["file1.py", "file2.py"],
        )

        assert task.title == "Implement feature X"
        assert task.description == "Detailed description here"
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.priority == Priority.HIGH
        assert task.effort_estimate == "2h"
        assert task.dependencies == ["task-1", "task-2"]
        assert task.parent_id == "parent-task"
        assert task.tags == ["backend", "api"]
        assert task.acceptance_criteria == ["Criterion 1", "Criterion 2"]
        assert task.notes == ["Note 1"]
        assert task.files_involved == ["file1.py", "file2.py"]

    def test_task_serialization(self):
        """Test task can be converted to dictionary"""
        task = Task(
            title="Test task",
            description="Test description",
            status=TaskStatus.COMPLETED,
            priority=Priority.HIGH,
            effort_estimate="1h",
        )

        data = task.to_dict()

        assert isinstance(data, dict)
        assert data["title"] == "Test task"
        assert data["description"] == "Test description"
        assert data["status"] == "completed"
        assert data["priority"] == "high"
        assert data["effort_estimate"] == "1h"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_task_deserialization(self):
        """Test task can be created from dictionary"""
        task_dict = {
            "id": "test-id-123",
            "title": "Test task",
            "description": "Test description",
            "acceptance_criteria": ["Criterion 1"],
            "status": "in_progress",
            "priority": "high",
            "effort_estimate": "1h",
            "dependencies": ["dep-1"],
            "parent_id": "parent-1",
            "tags": ["test"],
            "notes": ["Note 1"],
            "files_involved": ["file1.py"],
            "created_at": "2025-10-11T12:00:00",
            "updated_at": "2025-10-11T13:00:00",
            "started_at": "2025-10-11T12:30:00",
            "completed_at": None,
        }

        task = Task.from_dict(task_dict)

        assert task.id == "test-id-123"
        assert task.title == "Test task"
        assert task.description == "Test description"
        assert task.acceptance_criteria == ["Criterion 1"]
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.priority == Priority.HIGH
        assert task.effort_estimate == "1h"
        assert task.dependencies == ["dep-1"]
        assert task.parent_id == "parent-1"
        assert task.tags == ["test"]
        assert task.notes == ["Note 1"]
        assert task.files_involved == ["file1.py"]
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)
        assert isinstance(task.started_at, datetime)
        assert task.completed_at is None

    def test_task_round_trip_serialization(self):
        """Test task serialization and deserialization preserves data"""
        original_task = Task(
            title="Round trip test",
            description="Testing serialization",
            status=TaskStatus.IN_PROGRESS,
            priority=Priority.CRITICAL,
            effort_estimate="3h",
            dependencies=["dep-1", "dep-2"],
            tags=["test", "serialization"],
        )

        # Serialize
        data = original_task.to_dict()

        # Deserialize
        restored_task = Task.from_dict(data)

        # Verify
        assert restored_task.id == original_task.id
        assert restored_task.title == original_task.title
        assert restored_task.description == original_task.description
        assert restored_task.status == original_task.status
        assert restored_task.priority == original_task.priority
        assert restored_task.effort_estimate == original_task.effort_estimate
        assert restored_task.dependencies == original_task.dependencies
        assert restored_task.tags == original_task.tags


class TestWorkPlan:
    """Test WorkPlan model"""

    def test_workplan_creation_with_defaults(self):
        """Test work plan creation with default values"""
        plan = WorkPlan(name="Test Plan", goal="Test goal")

        assert plan.name == "Test Plan"
        assert plan.goal == "Test goal"
        assert plan.context == ""
        assert plan.tasks == []
        assert plan.version == 1
        assert plan.is_active is True
        assert len(plan.id) > 0
        assert isinstance(plan.created_at, datetime)
        assert isinstance(plan.updated_at, datetime)

    def test_workplan_creation_with_tasks(self):
        """Test work plan creation with tasks"""
        task1 = Task(title="Task 1")
        task2 = Task(title="Task 2")

        plan = WorkPlan(
            name="Test Plan",
            goal="Test goal",
            context="Test context",
            tasks=[task1, task2],
        )

        assert len(plan.tasks) == 2
        assert plan.tasks[0].title == "Task 1"
        assert plan.tasks[1].title == "Task 2"
        assert plan.context == "Test context"

    def test_workplan_get_task(self):
        """Test finding task by ID"""
        task1 = Task(title="Task 1")
        task2 = Task(title="Task 2")
        plan = WorkPlan(name="Test Plan", goal="Goal", tasks=[task1, task2])

        found_task = plan.get_task(task1.id)
        assert found_task is not None
        assert found_task.id == task1.id
        assert found_task.title == "Task 1"

        not_found = plan.get_task("non-existent-id")
        assert not_found is None

    def test_workplan_get_next_task_simple(self):
        """Test getting next task with no dependencies"""
        task1 = Task(title="Task 1", priority=Priority.HIGH)
        task2 = Task(title="Task 2", priority=Priority.CRITICAL)
        task3 = Task(title="Task 3", priority=Priority.LOW)

        plan = WorkPlan(name="Test Plan", goal="Goal", tasks=[task1, task2, task3])

        next_task = plan.get_next_task()
        assert next_task is not None
        # Should return critical priority first
        assert next_task.title == "Task 2"
        assert next_task.priority == Priority.CRITICAL

    def test_workplan_get_next_task_with_dependencies(self):
        """Test getting next task respecting dependencies"""
        task1 = Task(title="Task 1", priority=Priority.HIGH)
        task2 = Task(title="Task 2", priority=Priority.CRITICAL, dependencies=[task1.id])
        task3 = Task(title="Task 3", priority=Priority.MEDIUM)

        plan = WorkPlan(name="Test Plan", goal="Goal", tasks=[task1, task2, task3])

        next_task = plan.get_next_task()
        assert next_task is not None
        # Task 2 has higher priority but is blocked, so should get Task 1
        assert next_task.title == "Task 1"

    def test_workplan_get_next_task_skips_completed(self):
        """Test that completed tasks are not returned"""
        task1 = Task(title="Task 1", status=TaskStatus.COMPLETED)
        task2 = Task(title="Task 2", status=TaskStatus.PENDING)

        plan = WorkPlan(name="Test Plan", goal="Goal", tasks=[task1, task2])

        next_task = plan.get_next_task()
        assert next_task is not None
        assert next_task.title == "Task 2"

    def test_workplan_get_next_task_respects_completed_dependencies(self):
        """Test that tasks with completed dependencies are available"""
        task1 = Task(title="Task 1", status=TaskStatus.COMPLETED)
        task2 = Task(title="Task 2", dependencies=[task1.id], priority=Priority.HIGH)

        plan = WorkPlan(name="Test Plan", goal="Goal", tasks=[task1, task2])

        next_task = plan.get_next_task()
        assert next_task is not None
        assert next_task.title == "Task 2"

    def test_workplan_get_next_task_no_available_tasks(self):
        """Test when no tasks are available"""
        task1 = Task(title="Task 1", status=TaskStatus.COMPLETED)
        task2 = Task(title="Task 2", status=TaskStatus.IN_PROGRESS)

        plan = WorkPlan(name="Test Plan", goal="Goal", tasks=[task1, task2])

        next_task = plan.get_next_task()
        assert next_task == task2

    def test_workplan_completion_percentage_empty(self):
        """Test completion percentage with no tasks"""
        plan = WorkPlan(name="Test Plan", goal="Goal")
        assert plan.get_completion_percentage() == 0.0

    def test_workplan_completion_percentage_partial(self):
        """Test completion percentage with some completed tasks"""
        task1 = Task(title="Task 1", status=TaskStatus.COMPLETED)
        task2 = Task(title="Task 2", status=TaskStatus.PENDING)
        task3 = Task(title="Task 3", status=TaskStatus.COMPLETED)
        task4 = Task(title="Task 4", status=TaskStatus.IN_PROGRESS)

        plan = WorkPlan(name="Test Plan", goal="Goal", tasks=[task1, task2, task3, task4])

        # 2 out of 4 completed = 50%
        assert plan.get_completion_percentage() == 50.0

    def test_workplan_completion_percentage_complete(self):
        """Test completion percentage when all tasks completed"""
        task1 = Task(title="Task 1", status=TaskStatus.COMPLETED)
        task2 = Task(title="Task 2", status=TaskStatus.COMPLETED)

        plan = WorkPlan(name="Test Plan", goal="Goal", tasks=[task1, task2])

        assert plan.get_completion_percentage() == 100.0

    def test_workplan_serialization(self):
        """Test work plan can be converted to dictionary"""
        task1 = Task(title="Task 1")
        plan = WorkPlan(
            name="Test Plan",
            goal="Test goal",
            context="Test context",
            tasks=[task1],
            version=2,
        )

        data = plan.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "Test Plan"
        assert data["goal"] == "Test goal"
        assert data["context"] == "Test context"
        assert data["version"] == 2
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["title"] == "Task 1"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_workplan_deserialization(self):
        """Test work plan can be created from dictionary"""
        plan_dict = {
            "id": "plan-id-123",
            "name": "Test Plan",
            "goal": "Test goal",
            "context": "Test context",
            "tasks": [
                {
                    "id": "task-1",
                    "title": "Task 1",
                    "description": "",
                    "acceptance_criteria": [],
                    "status": "pending",
                    "priority": "medium",
                    "effort_estimate": "unknown",
                    "dependencies": [],
                    "parent_id": None,
                    "tags": [],
                    "notes": [],
                    "files_involved": [],
                    "created_at": "2025-10-11T12:00:00",
                    "updated_at": "2025-10-11T12:00:00",
                    "started_at": None,
                    "completed_at": None,
                }
            ],
            "version": 2,
            "created_at": "2025-10-11T12:00:00",
            "updated_at": "2025-10-11T13:00:00",
            "is_active": True,
        }

        plan = WorkPlan.from_dict(plan_dict)

        assert plan.id == "plan-id-123"
        assert plan.name == "Test Plan"
        assert plan.goal == "Test goal"
        assert plan.context == "Test context"
        assert plan.version == 2
        assert plan.is_active is True
        assert len(plan.tasks) == 1
        assert plan.tasks[0].title == "Task 1"

    def test_workplan_round_trip_serialization(self):
        """Test work plan serialization and deserialization preserves data"""
        task1 = Task(title="Task 1", priority=Priority.HIGH)
        task2 = Task(title="Task 2", dependencies=[task1.id])

        original_plan = WorkPlan(
            name="Round trip test",
            goal="Testing serialization",
            context="Test context",
            tasks=[task1, task2],
            version=3,
        )

        # Serialize
        data = original_plan.to_dict()

        # Deserialize
        restored_plan = WorkPlan.from_dict(data)

        # Verify
        assert restored_plan.id == original_plan.id
        assert restored_plan.name == original_plan.name
        assert restored_plan.goal == original_plan.goal
        assert restored_plan.context == original_plan.context
        assert restored_plan.version == original_plan.version
        assert len(restored_plan.tasks) == len(original_plan.tasks)
        assert restored_plan.tasks[0].id == original_plan.tasks[0].id
        assert restored_plan.tasks[1].dependencies == original_plan.tasks[1].dependencies

    def test_workplan_update_task_status_to_in_progress(self):
        """Test updating task status to IN_PROGRESS sets started_at."""
        task = Task(title="Task 1", status=TaskStatus.PENDING)
        plan = WorkPlan(name="Test", goal="Goal", tasks=[task])

        # Task should not have started_at initially
        assert task.started_at is None

        # Update to IN_PROGRESS
        result = plan.update_task_status(task.id, TaskStatus.IN_PROGRESS)

        assert result is True
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None
        assert isinstance(task.started_at, datetime)
        assert task.updated_at is not None

    def test_workplan_update_task_status_to_completed(self):
        """Test updating task status to COMPLETED sets completed_at."""
        task = Task(title="Task 1", status=TaskStatus.IN_PROGRESS)
        plan = WorkPlan(name="Test", goal="Goal", tasks=[task])

        # Task should not have completed_at initially
        assert task.completed_at is None

        # Update to COMPLETED
        result = plan.update_task_status(task.id, TaskStatus.COMPLETED)

        assert result is True
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert isinstance(task.completed_at, datetime)
        assert task.updated_at is not None

    def test_workplan_update_task_status_preserves_timestamps(self):
        """Test that updating status preserves existing timestamps."""
        task = Task(title="Task 1", status=TaskStatus.IN_PROGRESS)
        plan = WorkPlan(name="Test", goal="Goal", tasks=[task])

        # First update to IN_PROGRESS
        plan.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        started_at_first = task.started_at

        # Update again to BLOCKED - should not change started_at
        plan.update_task_status(task.id, TaskStatus.BLOCKED)

        assert task.started_at == started_at_first
        assert task.status == TaskStatus.BLOCKED

    def test_workplan_update_task_status_non_existent(self):
        """Test updating non-existent task returns False."""
        task = Task(title="Task 1")
        plan = WorkPlan(name="Test", goal="Goal", tasks=[task])

        result = plan.update_task_status("non-existent-id", TaskStatus.COMPLETED)

        assert result is False

    def test_workplan_update_task_status_multiple_transitions(self):
        """Test multiple status transitions update timestamps correctly."""
        task = Task(title="Task 1", status=TaskStatus.PENDING)
        plan = WorkPlan(name="Test", goal="Goal", tasks=[task])

        # PENDING -> IN_PROGRESS
        plan.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        assert task.started_at is not None
        assert task.completed_at is None
        started_at = task.started_at

        # IN_PROGRESS -> COMPLETED
        plan.update_task_status(task.id, TaskStatus.COMPLETED)
        assert task.completed_at is not None
        assert task.started_at == started_at  # Preserved
        assert isinstance(task.completed_at, datetime)

    def test_workplan_get_next_task_all_blocked(self):
        """Test when all tasks are blocked by dependencies."""
        task1 = Task(title="Task 1", dependencies=["nonexistent"])
        task2 = Task(title="Task 2", dependencies=["also-nonexistent"])

        plan = WorkPlan(name="Test", goal="Goal", tasks=[task1, task2])

        # All tasks are blocked, should return None
        next_task = plan.get_next_task()
        assert next_task is None

    def test_workplan_completion_percentage(self):
        """Test completion percentage calculation."""
        task1 = Task(title="Task 1", status=TaskStatus.COMPLETED)
        task2 = Task(title="Task 2", status=TaskStatus.COMPLETED)
        task3 = Task(title="Task 3", status=TaskStatus.PENDING)
        task4 = Task(title="Task 4", status=TaskStatus.IN_PROGRESS)

        plan = WorkPlan(name="Test", goal="Goal", tasks=[task1, task2, task3, task4])

        progress = plan.get_completion_percentage()
        assert progress == 50.0  # 2 out of 4 tasks completed

    def test_workplan_completion_percentage_no_tasks(self):
        """Test completion percentage when there are no tasks."""
        plan = WorkPlan(name="Test", goal="Goal", tasks=[])

        progress = plan.get_completion_percentage()
        assert progress == 0.0

    def test_workplan_with_mixed_task_statuses(self):
        """Test work plan with tasks in various states."""
        tasks = [
            Task(title="T1", status=TaskStatus.COMPLETED),
            Task(title="T2", status=TaskStatus.IN_PROGRESS),
            Task(title="T3", status=TaskStatus.PENDING),
            Task(title="T4", status=TaskStatus.BLOCKED),
            Task(title="T5", status=TaskStatus.CANCELLED),
        ]

        plan = WorkPlan(name="Mixed Plan", goal="Test", tasks=tasks)

        assert len(plan.tasks) == 5
        assert plan.get_completion_percentage() == 20.0  # Only 1 completed

    def test_task_with_complex_dependencies(self):
        """Test task with multiple dependencies."""
        task1 = Task(title="Foundation", priority=Priority.CRITICAL)
        task2 = Task(title="Module A", dependencies=[task1.id], priority=Priority.HIGH)
        task3 = Task(title="Module B", dependencies=[task1.id], priority=Priority.HIGH)
        task4 = Task(
            title="Integration", dependencies=[task2.id, task3.id], priority=Priority.MEDIUM
        )

        plan = WorkPlan(name="Complex", goal="Build", tasks=[task1, task2, task3, task4])

        # Task 1 should be next (no dependencies, highest priority)
        next_task = plan.get_next_task()
        assert next_task.id == task1.id

        # Mark task1 complete
        plan.update_task_status(task1.id, TaskStatus.COMPLETED)

        # Now either task2 or task3 should be available (both HIGH priority)
        next_task = plan.get_next_task()
        assert next_task.id in [task2.id, task3.id]
