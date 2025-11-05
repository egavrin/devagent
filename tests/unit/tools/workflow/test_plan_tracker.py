"""Tests for plan_tracker module."""

import pytest

from ai_dev_agent.tools.workflow.plan_tracker import (
    PlanTracker,
    clear_plan,
    get_tracker,
    start_plan_tracking,
    update_task_status,
)


class TestPlanTracker:
    """Test PlanTracker class."""

    def test_init(self):
        """Test tracker initialization."""
        tracker = PlanTracker()

        assert tracker._current_plan is None
        assert tracker._task_status == {}

    def test_start_plan(self, capsys):
        """Test starting a plan."""
        tracker = PlanTracker()
        plan = {
            "goal": "Test goal",
            "tasks": [
                {"id": "t1", "title": "Task 1", "agent": "design_agent", "description": "Design"},
                {"id": "t2", "title": "Task 2", "agent": "test_agent", "description": "Test"},
            ],
        }

        tracker.start_plan(plan)

        assert tracker._current_plan == plan
        assert tracker._task_status == {"t1": "pending", "t2": "pending"}

    def test_update_task_status(self, capsys):
        """Test updating task status."""
        tracker = PlanTracker()
        plan = {
            "goal": "Test",
            "tasks": [
                {"id": "t1", "title": "Task", "agent": "design_agent", "description": "Desc"}
            ],
        }
        tracker.start_plan(plan)

        tracker.update_task_status("t1", "in_progress")

        assert tracker._task_status["t1"] == "in_progress"

    def test_update_nonexistent_task(self, capsys):
        """Test updating nonexistent task does nothing."""
        tracker = PlanTracker()
        plan = {"goal": "Test", "tasks": []}
        tracker.start_plan(plan)

        # Should not raise
        tracker.update_task_status("nonexistent", "completed")

    def test_clear(self):
        """Test clearing plan."""
        tracker = PlanTracker()
        plan = {
            "goal": "Test",
            "tasks": [{"id": "t1", "title": "T", "agent": "design_agent", "description": "D"}],
        }
        tracker.start_plan(plan)

        tracker.clear()

        assert tracker._current_plan is None
        assert tracker._task_status == {}

    def test_display_plan_no_plan(self, capsys):
        """Test display with no plan does nothing."""
        tracker = PlanTracker()

        # Should not raise
        tracker._display_plan()

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_display_plan_with_tasks(self, capsys):
        """Test plan display shows tasks."""
        tracker = PlanTracker()
        plan = {
            "goal": "Build feature",
            "tasks": [
                {
                    "id": "t1",
                    "title": "Design",
                    "agent": "design_agent",
                    "description": "Create design",
                },
            ],
        }
        tracker.start_plan(plan)

        captured = capsys.readouterr()
        assert "Work Plan Progress" in captured.out
        assert "Build feature" in captured.out
        assert "Design" in captured.out

    def test_status_icons(self, capsys):
        """Test different status icons."""
        tracker = PlanTracker()
        plan = {
            "goal": "Test",
            "tasks": [
                {"id": "t1", "title": "T1", "agent": "design_agent", "description": "D1"},
                {"id": "t2", "title": "T2", "agent": "test_agent", "description": "D2"},
                {"id": "t3", "title": "T3", "agent": "review_agent", "description": "D3"},
                {"id": "t4", "title": "T4", "agent": "impl_agent", "description": "D4"},
            ],
        }
        tracker.start_plan(plan)

        # Update to different statuses
        tracker.update_task_status("t1", "pending")
        tracker.update_task_status("t2", "in_progress")
        tracker.update_task_status("t3", "completed")
        tracker.update_task_status("t4", "failed")

        captured = capsys.readouterr()
        assert "[ ]" in captured.out  # pending
        assert "[~]" in captured.out  # in_progress
        assert "[✓]" in captured.out  # completed
        assert "[✗]" in captured.out  # failed


class TestGlobalFunctions:
    """Test module-level functions."""

    def test_get_tracker_creates_singleton(self):
        """Test get_tracker creates and reuses singleton."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()

        assert tracker1 is tracker2
        assert isinstance(tracker1, PlanTracker)

    def test_start_plan_tracking(self, capsys):
        """Test start_plan_tracking function."""
        plan = {
            "goal": "Test goal",
            "tasks": [
                {"id": "t1", "title": "Task", "agent": "design_agent", "description": "Desc"}
            ],
        }

        start_plan_tracking(plan)

        tracker = get_tracker()
        assert tracker._current_plan == plan

    def test_update_task_status_function(self, capsys):
        """Test update_task_status function."""
        plan = {
            "goal": "Test",
            "tasks": [
                {"id": "t1", "title": "Task", "agent": "design_agent", "description": "Desc"}
            ],
        }
        start_plan_tracking(plan)

        update_task_status("t1", "completed")

        tracker = get_tracker()
        assert tracker._task_status["t1"] == "completed"

    def test_clear_plan_function(self, capsys):
        """Test clear_plan function."""
        plan = {
            "goal": "Test",
            "tasks": [{"id": "t1", "title": "T", "agent": "design_agent", "description": "D"}],
        }
        start_plan_tracking(plan)

        clear_plan()

        tracker = get_tracker()
        assert tracker._current_plan is None


class TestTaskStatusTransitions:
    """Test task status transitions."""

    def test_pending_to_in_progress(self, capsys):
        """Test transitioning task from pending to in_progress."""
        tracker = PlanTracker()
        plan = {
            "goal": "Test",
            "tasks": [
                {
                    "id": "t1",
                    "title": "Task",
                    "agent": "design_agent",
                    "description": "Design system",
                }
            ],
        }
        tracker.start_plan(plan)

        assert tracker._task_status["t1"] == "pending"

        tracker.update_task_status("t1", "in_progress")

        assert tracker._task_status["t1"] == "in_progress"

    def test_in_progress_to_completed(self, capsys):
        """Test transitioning task from in_progress to completed."""
        tracker = PlanTracker()
        plan = {
            "goal": "Test",
            "tasks": [
                {"id": "t1", "title": "Task", "agent": "test_agent", "description": "Write tests"}
            ],
        }
        tracker.start_plan(plan)
        tracker.update_task_status("t1", "in_progress")

        tracker.update_task_status("t1", "completed")

        assert tracker._task_status["t1"] == "completed"

    def test_in_progress_to_failed(self, capsys):
        """Test transitioning task from in_progress to failed."""
        tracker = PlanTracker()
        plan = {
            "goal": "Test",
            "tasks": [
                {"id": "t1", "title": "Task", "agent": "impl_agent", "description": "Implement"}
            ],
        }
        tracker.start_plan(plan)
        tracker.update_task_status("t1", "in_progress")

        tracker.update_task_status("t1", "failed")

        assert tracker._task_status["t1"] == "failed"


class TestMultipleTasks:
    """Test tracking multiple tasks."""

    def test_multiple_tasks_independent(self, capsys):
        """Test multiple tasks can have independent statuses."""
        tracker = PlanTracker()
        plan = {
            "goal": "Complex project",
            "tasks": [
                {"id": "t1", "title": "Design", "agent": "design_agent", "description": "D1"},
                {"id": "t2", "title": "Implement", "agent": "impl_agent", "description": "D2"},
                {"id": "t3", "title": "Test", "agent": "test_agent", "description": "D3"},
            ],
        }
        tracker.start_plan(plan)

        tracker.update_task_status("t1", "completed")
        tracker.update_task_status("t2", "in_progress")
        # t3 stays pending

        assert tracker._task_status["t1"] == "completed"
        assert tracker._task_status["t2"] == "in_progress"
        assert tracker._task_status["t3"] == "pending"

    def test_sequential_updates(self, capsys):
        """Test sequential task updates."""
        tracker = PlanTracker()
        plan = {
            "goal": "Sequential work",
            "tasks": [
                {"id": "t1", "title": "First", "agent": "design_agent", "description": "D1"},
                {"id": "t2", "title": "Second", "agent": "impl_agent", "description": "D2"},
            ],
        }
        tracker.start_plan(plan)

        # Complete first task
        tracker.update_task_status("t1", "in_progress")
        tracker.update_task_status("t1", "completed")

        # Start second task
        tracker.update_task_status("t2", "in_progress")

        assert tracker._task_status["t1"] == "completed"
        assert tracker._task_status["t2"] == "in_progress"


class TestAgentLabelFormatting:
    """Test agent label formatting in display."""

    def test_agent_name_formatting(self, capsys):
        """Test agent names are formatted (removing _agent suffix)."""
        tracker = PlanTracker()
        plan = {
            "goal": "Test",
            "tasks": [
                {"id": "t1", "title": "Task 1", "agent": "design_agent", "description": "D"},
                {
                    "id": "t2",
                    "title": "Task 2",
                    "agent": "implementation_agent",
                    "description": "D",
                },
            ],
        }
        tracker.start_plan(plan)

        captured = capsys.readouterr()
        assert "design)" in captured.out
        assert "implementation)" in captured.out


class TestEmptyPlan:
    """Test handling of empty plans."""

    def test_empty_task_list(self, capsys):
        """Test plan with no tasks."""
        tracker = PlanTracker()
        plan = {"goal": "Empty plan", "tasks": []}

        tracker.start_plan(plan)

        assert tracker._current_plan == plan
        assert tracker._task_status == {}

        captured = capsys.readouterr()
        assert "Empty plan" in captured.out
