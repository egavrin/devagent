"""Test plan execution with status tracking, early stopping, and error handling."""

import json
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from ai_dev_agent.engine.react.types import RunResult
from ai_dev_agent.tools.workflow.plan_tracker import PlanTracker


class TestStatusUpdatesDuringExecution:
    """Test real-time status tracking during plan execution."""

    def test_status_updates_during_task_execution(self):
        """Test that task status transitions properly during execution."""
        tasks = [
            {"id": "task-1", "title": "Analyze", "description": "Analyze code", "dependencies": []},
            {
                "id": "task-2",
                "title": "Implement",
                "description": "Write code",
                "dependencies": ["task-1"],
            },
        ]

        with patch("ai_dev_agent.cli.react.plan_executor.create_plan") as mock_create:
            mock_create.return_value = {
                "success": True,
                "plan": {"goal": "test", "tasks": tasks},
            }

            with patch("ai_dev_agent.cli.react.executor._execute_react_assistant") as mock_exec:
                mock_exec.return_value = {"final_message": "Done", "result": {}}

                # Track calls to update_task_status - patch where it's imported
                with patch(
                    "ai_dev_agent.cli.react.plan_executor.update_task_status"
                ) as mock_update:
                    from ai_dev_agent.cli.react.plan_executor import execute_with_planning
                    from ai_dev_agent.core.utils.config import Settings

                    settings = Settings()
                    settings.always_use_planning = True  # Force planning
                    ctx = Mock()
                    ctx.echo = Mock()
                    client = Mock()

                    result = execute_with_planning(ctx, client, settings, "implement feature")

                    # Should have called update_task_status for each task
                    # 2 tasks × 2 status updates (in_progress, completed) = 4 calls
                    assert mock_update.call_count == 4
                    assert result is not None  # Use result to avoid linting warning

                    # Verify the specific calls
                    expected_calls = [
                        call("task-1", "in_progress"),
                        call("task-1", "completed"),
                        call("task-2", "in_progress"),
                        call("task-2", "completed"),
                    ]
                    mock_update.assert_has_calls(expected_calls)
                    assert result["result"]["tasks_completed"] == len(tasks)
                    assert result["result"]["error_occurred"] is False

    def test_progress_calculation_accuracy(self):
        """Test that progress percentage reflects actual completion."""
        # This test verifies progress reporting during execution
        # Currently progress is printed but not returned in structured format
        tasks = [
            {
                "id": f"task-{i}",
                "title": f"Task {i}",
                "description": f"Do task {i}",
                "dependencies": [],
            }
            for i in range(1, 5)
        ]

        with patch("ai_dev_agent.cli.react.plan_executor.create_plan") as mock_create:
            mock_create.return_value = {
                "success": True,
                "plan": {"goal": "test", "tasks": tasks},
            }

            with patch("ai_dev_agent.cli.react.executor._execute_react_assistant") as mock_exec:
                mock_exec.return_value = {"final_message": "Done", "result": {}}

                with patch("click.echo") as mock_echo:
                    from ai_dev_agent.cli.react.plan_executor import execute_with_planning
                    from ai_dev_agent.core.utils.config import Settings

                    settings = Settings()
                    settings.always_use_planning = True
                    ctx = Mock()
                    client = Mock()

                    result = execute_with_planning(ctx, client, settings, "complex task")

                    # Check that progress was reported correctly (25%, 50%, 75%, 100%)
                    assert result is not None  # Use result to avoid linting warning
                    progress_calls = [
                        call for call in mock_echo.call_args_list if "Progress:" in str(call)
                    ]

                    # Should have 4 progress updates for 4 tasks
                    assert len(progress_calls) == 4
                    assert "25%" in str(progress_calls[0])
                    assert "50%" in str(progress_calls[1])
                    assert "75%" in str(progress_calls[2])
                    assert "100%" in str(progress_calls[3])
                    assert result["result"]["tasks_completed"] == len(tasks)
                    assert result["result"]["stopped_early"] is False

    def test_failed_task_status_propagates(self):
        """Test that failed task status is visible in tracker."""
        tasks = [
            {
                "id": "task-1",
                "title": "Failing task",
                "description": "This will fail",
                "dependencies": [],
            },
        ]

        with patch("ai_dev_agent.cli.react.plan_executor.create_plan") as mock_create:
            mock_create.return_value = {
                "success": True,
                "plan": {"goal": "test", "tasks": tasks},
            }

            with patch("ai_dev_agent.cli.react.executor._execute_react_assistant") as mock_exec:
                # Simulate a task failure with proper RunResult object
                failed_result = RunResult(
                    task_id="task-1",
                    status="failure",
                    steps=[],
                    gates={},
                    stop_reason="Task failed",
                )
                mock_exec.return_value = {
                    "final_message": "Error occurred",
                    "result": failed_result,
                }

                with patch(
                    "ai_dev_agent.cli.react.plan_executor.update_task_status"
                ) as mock_update:
                    from ai_dev_agent.cli.react.plan_executor import execute_with_planning
                    from ai_dev_agent.core.utils.config import Settings

                    settings = Settings()
                    settings.always_use_planning = True
                    ctx = Mock()
                    ctx.echo = Mock()
                    client = Mock()

                    result = execute_with_planning(ctx, client, settings, "failing task")

                    # Verify result before other assertions
                    assert result is not None
                    # Now properly detects failures and updates task status
                    # Should call update_task_status twice: "in_progress" then "failed"
                    assert mock_update.call_count == 2
                    mock_update.assert_any_call("task-1", "in_progress")
                    mock_update.assert_any_call("task-1", "failed")
                    assert result["result"]["error_occurred"] is True
                    assert result["result"]["tasks_completed"] == 0


class TestEarlyStoppingMechanism:
    """Test early stopping with done_when conditions."""

    def test_early_stop_on_condition_met(self):
        """Stop execution when done_when condition is satisfied."""
        plan = {
            "goal": "Build feature",
            "done_when": "tests pass",  # Not currently supported
            "tasks": [
                {"id": "t1", "title": "Write tests", "description": "Write tests"},
                {"id": "t2", "title": "Implement", "description": "Implement feature"},
                {
                    "id": "t3",
                    "title": "Refactor",
                    "description": "Refactor code",
                },  # Should be skipped
            ],
        }

        with patch("ai_dev_agent.cli.react.plan_executor.create_plan") as mock_create:
            mock_create.return_value = {"success": True, "plan": plan}

            with patch("ai_dev_agent.cli.react.executor._execute_react_assistant") as mock_exec:
                # First two tasks succeed, third should be skipped
                mock_exec.side_effect = [
                    {"final_message": "Tests written", "result": {}},
                    {"final_message": "Tests pass!", "result": {"tests_pass": True}},
                    {"final_message": "Refactored", "result": {}},  # Shouldn't be called
                ]

                from ai_dev_agent.cli.react.plan_executor import execute_with_planning
                from ai_dev_agent.core.utils.config import Settings

                settings = Settings()
                settings.always_use_planning = True
                ctx = Mock()
                client = Mock()

                result = execute_with_planning(ctx, client, settings, "build feature")

                # Should only execute first two tasks (early stopping after "tests pass!")
                assert mock_exec.call_count == 2  # Early stopping is working!

                # Verify the result indicates early stopping
                assert result["result"]["stopped_early"] is True
                assert result["result"]["tasks_completed"] == 2
                assert result["result"]["tasks_total"] == 3

    def test_no_early_stop_without_condition(self):
        """All tasks execute when no done_when specified."""
        tasks = [
            {"id": f"t{i}", "title": f"Task {i}", "description": f"Do task {i}", "dependencies": []}
            for i in range(1, 4)
        ]

        with patch("ai_dev_agent.cli.react.plan_executor.create_plan") as mock_create:
            mock_create.return_value = {
                "success": True,
                "plan": {"goal": "test", "tasks": tasks},  # No done_when
            }

            with patch("ai_dev_agent.cli.react.executor._execute_react_assistant") as mock_exec:
                mock_exec.return_value = {"final_message": "Done", "result": {}}

                from ai_dev_agent.cli.react.plan_executor import execute_with_planning
                from ai_dev_agent.core.utils.config import Settings

                settings = Settings()
                settings.always_use_planning = True
                ctx = Mock()
                client = Mock()

                result = execute_with_planning(ctx, client, settings, "do all tasks")

                # Should execute all 3 tasks
                assert mock_exec.call_count == 3
                assert result["result"]["tasks_completed"] == len(tasks)
                assert result["result"]["stopped_early"] is False


class TestTaskDependencies:
    """Test dependency resolution and execution order."""

    def test_dependencies_executed_in_order(self):
        """Tasks with dependencies wait for prerequisites."""
        tasks = [
            {"id": "t1", "title": "Setup", "description": "Setup env", "dependencies": []},
            {"id": "t2", "title": "Build", "description": "Build app", "dependencies": ["t1"]},
            {"id": "t3", "title": "Test", "description": "Run tests", "dependencies": ["t1", "t2"]},
        ]

        with patch("ai_dev_agent.cli.react.plan_executor.create_plan") as mock_create:
            mock_create.return_value = {"success": True, "plan": {"goal": "test", "tasks": tasks}}

            execution_order = []

            def track_execution(ctx, client, settings, prompt, **kwargs):
                # Extract task ID from prompt
                for task in tasks:
                    if task["title"] in prompt:
                        execution_order.append(task["id"])
                        break
                return {"final_message": "Done", "result": {}}

            with patch(
                "ai_dev_agent.cli.react.executor._execute_react_assistant",
                side_effect=track_execution,
            ):
                from ai_dev_agent.cli.react.plan_executor import execute_with_planning
                from ai_dev_agent.core.utils.config import Settings

                settings = Settings()
                settings.always_use_planning = True
                ctx = Mock()
                client = Mock()

                result = execute_with_planning(ctx, client, settings, "build and test")

                # Verify result
                assert result is not None
                # Currently executes in order listed (ignores dependencies)
                # Should respect dependencies when implemented
                assert execution_order == ["t1", "t2", "t3"]

    def test_failed_dependency_blocks_downstream(self):
        """Task failure prevents dependent tasks from executing."""
        tasks = [
            {"id": "t1", "title": "Setup", "description": "Setup", "dependencies": []},
            {"id": "t2", "title": "Build", "description": "Build", "dependencies": ["t1"]},
        ]

        with patch("ai_dev_agent.cli.react.plan_executor.create_plan") as mock_create:
            mock_create.return_value = {"success": True, "plan": {"goal": "test", "tasks": tasks}}

            with patch("ai_dev_agent.cli.react.executor._execute_react_assistant") as mock_exec:
                # First task fails
                failed_result = RunResult(
                    task_id="t1", status="failure", steps=[], gates={}, stop_reason="Setup failed"
                )
                success_result = RunResult(task_id="t2", status="success", steps=[], gates={})
                mock_exec.side_effect = [
                    {"final_message": "Failed", "result": failed_result},
                    {"final_message": "Built", "result": success_result},  # Shouldn't be called
                ]

                from ai_dev_agent.cli.react.plan_executor import execute_with_planning
                from ai_dev_agent.core.utils.config import Settings

                settings = Settings()
                settings.always_use_planning = True
                ctx = Mock()
                client = Mock()

                result = execute_with_planning(ctx, client, settings, "build project")

                # Verify result
                assert result is not None
                # Should stop after first task fails (fail-fast)
                assert mock_exec.call_count == 1  # Fail-fast stops after first failure
                assert result["result"]["error_occurred"] is True
                assert result["result"]["tasks_completed"] == 0


class TestErrorHandling:
    """Test error handling and fail-fast behavior."""

    def test_task_failure_stops_execution(self):
        """Plan stops when a critical task fails."""
        tasks = [
            {"id": "t1", "title": "Task 1", "description": "First", "dependencies": []},
            {"id": "t2", "title": "Task 2", "description": "Second", "dependencies": []},
            {"id": "t3", "title": "Task 3", "description": "Third", "dependencies": []},
        ]

        with patch("ai_dev_agent.cli.react.plan_executor.create_plan") as mock_create:
            mock_create.return_value = {"success": True, "plan": {"goal": "test", "tasks": tasks}}

            with patch("ai_dev_agent.cli.react.executor._execute_react_assistant") as mock_exec:
                # Second task fails
                success_result_1 = RunResult(task_id="t1", status="success", steps=[], gates={})
                failed_result = RunResult(
                    task_id="t2",
                    status="failure",
                    steps=[],
                    gates={},
                    stop_reason="Critical failure",
                )
                success_result_3 = RunResult(task_id="t3", status="success", steps=[], gates={})
                mock_exec.side_effect = [
                    {"final_message": "Success", "result": success_result_1},
                    {"final_message": "Error", "result": failed_result},
                    {"final_message": "Success", "result": success_result_3},  # Shouldn't be called
                ]

                from ai_dev_agent.cli.react.plan_executor import execute_with_planning
                from ai_dev_agent.core.utils.config import Settings

                settings = Settings()
                settings.always_use_planning = True
                ctx = Mock()
                client = Mock()

                result = execute_with_planning(ctx, client, settings, "execute tasks")

                # Should stop after task 2 fails (fail-fast)
                assert mock_exec.call_count == 2  # Fail-fast is working!

                # Verify the result indicates failure
                assert result["result"]["error_occurred"] is True
                assert result["result"]["tasks_completed"] == 1  # Only first task succeeded
                assert result["result"]["tasks_total"] == 3

    def test_error_message_reported_to_user(self):
        """Error messages are properly reported when tasks fail."""
        tasks = [
            {"id": "t1", "title": "Failing task", "description": "Will fail", "dependencies": []},
        ]

        with patch("ai_dev_agent.cli.react.plan_executor.create_plan") as mock_create:
            mock_create.return_value = {"success": True, "plan": {"goal": "test", "tasks": tasks}}

            with patch("ai_dev_agent.cli.react.executor._execute_react_assistant") as mock_exec:
                failed_result = RunResult(
                    task_id="t1",
                    status="failure",
                    steps=[],
                    gates={},
                    stop_reason="Connection timeout",
                )
                mock_exec.return_value = {
                    "final_message": "Task failed with error",
                    "result": failed_result,
                }

                with patch("click.echo") as mock_echo:
                    from ai_dev_agent.cli.react.plan_executor import execute_with_planning
                    from ai_dev_agent.core.utils.config import Settings

                    settings = Settings()
                    settings.always_use_planning = True
                    ctx = Mock()
                    client = Mock()

                    result = execute_with_planning(ctx, client, settings, "failing task")

                    # Verify result
                    assert result is not None
                    # Error should be reported to user
                    error_reported = any(
                        "failed" in str(call).lower() or "stopping" in str(call).lower()
                        for call in mock_echo.call_args_list
                    )

                    # Should show error message
                    assert error_reported  # Error handling is now working!
                    assert result["result"]["error_occurred"] is True
                    assert result["result"]["tasks_completed"] == 0
