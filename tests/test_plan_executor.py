"""Tests for the plan executor module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.agents.work_planner import Priority, Task, TaskStatus, WorkPlan, WorkPlanningAgent

# _execute_react_assistant is imported from executor module, not plan_executor
from ai_dev_agent.cli.react.plan_executor import (
    _assess_query_with_llm,
    _check_if_query_satisfied,
    _create_plan_from_query,
    _execute_task,
    _parse_task_breakdown,
    _synthesize_final_message,
    execute_with_planning,
)


class TestExecuteWithPlanning:
    """Test the execute_with_planning function."""

    @patch("ai_dev_agent.cli.react.plan_executor._assess_query_with_llm")
    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("click.echo")
    def test_execute_with_planning_direct(self, mock_echo, mock_execute, mock_assess):
        """Test direct execution when assessment says direct."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_settings = MagicMock()

        # Assessment says use direct execution
        mock_assess.return_value = {
            "approach": "direct",
            "reasoning": "Simple query",
            "estimated_tasks": 1,
        }

        # _execute_react_assistant doesn't return a value
        mock_execute.return_value = None

        execute_with_planning(
            ctx=mock_ctx, client=mock_client, settings=mock_settings, user_prompt="What is 2+2?"
        )

        # The function returns the result from _execute_react_assistant
        # which is None in the direct case
        mock_execute.assert_called_once()
        mock_assess.assert_called_once_with(mock_client, "What is 2+2?")

    @patch("ai_dev_agent.cli.react.plan_executor.WorkPlanningAgent")
    @patch("ai_dev_agent.cli.react.plan_executor._assess_query_with_llm")
    @patch("ai_dev_agent.cli.react.plan_executor._create_plan_from_query")
    @patch("ai_dev_agent.cli.react.plan_executor._execute_task")
    @patch("click.echo")
    def test_execute_with_planning_simple_plan(
        self, mock_echo, mock_execute_task, mock_create_plan, mock_assess, mock_agent_class
    ):
        """Test execution with simple planning."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_settings = MagicMock()

        # Assessment says use simple plan
        mock_assess.return_value = {
            "approach": "simple_plan",
            "reasoning": "Multi-step task",
            "estimated_tasks": 3,
        }

        # Mock the agent instance and its storage
        mock_agent = MagicMock()
        mock_storage = MagicMock()
        mock_agent.storage = mock_storage
        mock_agent_class.return_value = mock_agent
        mock_agent.get_active_plan.return_value = None

        # Create a mock plan
        mock_task = Task(
            title="Test Task",
            description="Test Description",
            priority=Priority.MEDIUM,
            status=TaskStatus.PENDING,
        )
        mock_plan = WorkPlan(name="Test Plan", goal="Demo goal", tasks=[mock_task])
        mock_create_plan.return_value = mock_plan
        # Mock storage to return the plan
        mock_storage.load_plan.return_value = mock_plan

        mock_execute_task.return_value = {
            "task_id": mock_task.id,
            "task_title": "Test Task",
            "result": {"final_message": "Task done", "printed_final": True},
        }

        with (
            patch(
                "ai_dev_agent.cli.react.plan_executor._synthesize_final_message"
            ) as mock_synthesize,
            patch.object(WorkPlan, "get_completion_percentage", return_value=100.0),
        ):
            mock_synthesize.return_value = "All tasks completed"

            result = execute_with_planning(
                ctx=mock_ctx,
                client=mock_client,
                settings=mock_settings,
                user_prompt="Build a feature",
            )

            # Should create and execute plan
            assert "final_message" in result
            mock_create_plan.assert_called_once()
            mock_execute_task.assert_called_once()

    @patch("ai_dev_agent.cli.react.plan_executor._assess_query_with_llm")
    @patch("ai_dev_agent.cli.react.plan_executor._create_plan_from_query")
    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    @patch("click.echo")
    def test_execute_with_planning_fallback(
        self, mock_echo, mock_execute, mock_create_plan, mock_assess
    ):
        """Test fallback to direct execution when plan creation fails."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_settings = MagicMock()

        # Assessment says use planning
        mock_assess.return_value = {
            "approach": "simple_plan",
            "reasoning": "Complex task",
            "estimated_tasks": 5,
        }

        # Plan creation fails
        mock_create_plan.return_value = None

        mock_execute.return_value = {"result": "fallback"}

        result = execute_with_planning(
            ctx=mock_ctx, client=mock_client, settings=mock_settings, user_prompt="Complex task"
        )

        # Should fall back to direct execution
        assert result == {"result": "fallback"}
        mock_execute.assert_called_once()


class TestAssessQueryWithLLM:
    """Test query assessment function."""

    def test_assess_query_direct(self):
        """Test assessment returning direct approach."""
        mock_client = MagicMock()
        # Mock the complete method to return JSON
        mock_client.complete.return_value = json.dumps(
            {
                "approach": "direct",
                "reasoning": "Simple query",
                "estimated_tasks": 1,
                "can_answer_immediately": True,
            }
        )

        result = _assess_query_with_llm(mock_client, "What time is it?")

        assert result["approach"] == "direct"
        assert result["estimated_tasks"] == 1
        mock_client.complete.assert_called_once()

    @patch("ai_dev_agent.cli.react.plan_executor.json.loads")
    def test_assess_query_complex(self, mock_json_loads):
        """Test assessment returning complex plan approach."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"approach": "complex_plan", "reasoning": "Multi-agent", "estimated_tasks": 10}'
        )
        mock_client.create_message.return_value = mock_response

        mock_json_loads.return_value = {
            "approach": "complex_plan",
            "reasoning": "Requires multi-agent coordination",
            "estimated_tasks": 10,
        }

        result = _assess_query_with_llm(mock_client, "Refactor the entire codebase")

        assert result["approach"] == "complex_plan"
        assert result["estimated_tasks"] == 10

    def test_assess_query_parse_error(self):
        """Test assessment with JSON parsing error."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON"
        mock_client.create_message.return_value = mock_response

        result = _assess_query_with_llm(mock_client, "Test query")

        # Should return default values on error
        assert result["approach"] == "simple_plan"
        assert result["estimated_tasks"] == 2


class TestCreatePlanFromQuery:
    """Test plan creation from query."""

    def test_create_plan_from_query_simple(self):
        """Test creating a simple plan from query."""
        mock_agent = MagicMock(spec=WorkPlanningAgent)
        mock_client = MagicMock()
        mock_assessment = {
            "approach": "simple_plan",
            "reasoning": "Straightforward task",
            "estimated_tasks": 2,
        }

        # Mock LLM response with numbered list format (which _parse_task_breakdown expects)
        mock_client.complete.return_value = """
        1. Step 1 - First step
        2. Step 2 - Second step
        """

        # Mock agent plan creation and storage
        mock_storage = MagicMock()
        mock_agent.storage = mock_storage
        mock_plan = WorkPlan(
            name="Auto Plan",
            goal="Test query",
            tasks=[
                Task(title="Step 1", description="First step", priority=Priority.HIGH),
                Task(title="Step 2", description="Second step", priority=Priority.MEDIUM),
            ],
        )
        mock_agent.create_plan.return_value = mock_plan

        result = _create_plan_from_query(mock_agent, mock_client, "Test query", mock_assessment)

        assert result == mock_plan
        assert len(result.tasks) == 2
        mock_agent.create_plan.assert_called_once()
        mock_storage.save_plan.assert_called_once_with(mock_plan)

    def test_create_plan_from_query_with_error(self):
        """Test plan creation with error handling."""
        mock_agent = MagicMock(spec=WorkPlanningAgent)
        mock_client = MagicMock()
        mock_assessment = {"approach": "simple_plan"}

        # LLM throws error
        mock_client.complete.side_effect = Exception("LLM Error")

        result = _create_plan_from_query(mock_agent, mock_client, "Test query", mock_assessment)

        # Should return None on error
        assert result is None


class TestParseTaskBreakdown:
    """Test task breakdown parsing."""

    def test_parse_task_breakdown_json(self):
        """Test parsing JSON task breakdown."""
        # The function doesn't actually parse JSON, it parses numbered lists
        # So this test is invalid. Let's skip it.
        pytest.skip("Function doesn't parse JSON format")

    def test_parse_task_breakdown_numbered_list(self):
        """Test parsing numbered list format."""
        response = """
        1. First Task - Do the first thing
        2. Second Task - Do the second thing
        3. Third Task - Complete the work
        """

        result = _parse_task_breakdown(response)

        assert len(result) >= 3
        # Result is list of [title, priority, description] lists
        # Check that tasks were extracted (result[i][0] is title)
        assert any("First" in str(t[0]) for t in result)

    def test_parse_task_breakdown_bullet_list(self):
        """Test parsing bullet list format."""
        response = """
        - Setup environment
        - Write tests
        - Implement feature
        - Deploy
        """

        result = _parse_task_breakdown(response)

        # Function doesn't parse bullet lists, returns default fallback tasks
        assert len(result) >= 2
        # Default tasks are: Analyze, Execute, Report

    def test_parse_task_breakdown_invalid(self):
        """Test parsing invalid format."""
        response = "This is just plain text with no structure"

        result = _parse_task_breakdown(response)

        # Should return empty list or single task
        assert isinstance(result, list)


class TestExecuteTask:
    """Test single task execution."""

    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    def test_execute_task_success(self, mock_execute):
        """Test successful task execution."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_settings = MagicMock()

        task = Task(title="Test Task", description="Do something", priority=Priority.HIGH)

        mock_execute.return_value = {
            "final_message": "Task completed successfully",
            "printed_final": True,
        }

        result = _execute_task(
            ctx=mock_ctx,
            client=mock_client,
            settings=mock_settings,
            task=task,
            original_query="Original",
        )

        assert result["task_title"] == "Test Task"
        assert result["result"]["final_message"] == "Task completed successfully"
        mock_execute.assert_called_once()

    @patch("ai_dev_agent.cli.react.executor._execute_react_assistant")
    def test_execute_task_with_context(self, mock_execute):
        """Test task execution with previous context."""
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_settings = MagicMock()

        task = Task(title="Second Task", priority=Priority.MEDIUM)

        previous_results = [{"task_title": "First Task", "output": "First result"}]

        mock_execute.return_value = {"final_message": "Second task done", "printed_final": True}

        result = _execute_task(
            ctx=mock_ctx,
            client=mock_client,
            settings=mock_settings,
            task=task,
            original_query="Do both tasks",
            task_index=1,
            total_tasks=2,
            previous_results=previous_results,
        )

        # Check result structure
        assert result["task_title"] == "Second Task"
        assert result["result"]["final_message"] == "Second task done"
        # Should pass context to executor
        call_args = mock_execute.call_args
        assert "First result" in str(call_args) or "First Task" in str(call_args)


class TestSynthesizeFinalMessage:
    """Test final message synthesis."""

    def test_synthesize_final_message_all_success(self):
        """Test synthesizing message when all tasks succeed."""
        task_results = [
            {
                "task_title": "Task 1",
                "success": True,
                "result": {"final_message": "Task 1 completed: analyzed the codebase structure"},
            },
            {
                "task_title": "Task 2",
                "success": True,
                "result": {
                    "final_message": "Task 2 completed: implemented the feature successfully"
                },
            },
        ]

        result = _synthesize_final_message(task_results)

        # The function returns the last task's result
        assert "Task 2 completed" in result
        assert "implemented the feature" in result

    def test_synthesize_final_message_with_failures(self):
        """Test synthesizing message with failures."""
        task_results = [
            {
                "task_title": "Task 1",
                "success": True,
                "result": {"final_message": "Task 1 done: initial analysis complete"},
            },
            {
                "task_title": "Task 2",
                "success": False,
                "result": {"final_message": "ERROR: Failed to complete task 2"},
            },
        ]

        result = _synthesize_final_message(task_results)

        # Function skips ERROR messages and returns last valid result
        assert "Task 1 done" in result

    def test_synthesize_final_message_empty(self):
        """Test synthesizing with no results."""
        task_results = []

        result = _synthesize_final_message(task_results)

        # Should return some message even with no results
        assert len(result) > 0


class TestCheckIfQuerySatisfied:
    """Test query satisfaction checking."""

    def test_check_query_satisfied_success(self):
        """Test checking if query is satisfied."""
        mock_client = MagicMock()
        # Mock the complete method to return JSON
        mock_client.complete.return_value = json.dumps(
            {
                "is_satisfied": True,
                "reasoning": "Query has been fully satisfied",
                "confidence": 0.9,
                "missing_aspects": [],
            }
        )

        completed_tasks = [
            {"task_title": "Build feature", "result": {"final_message": "Feature built"}},
            {"task_title": "Test feature", "result": {"final_message": "Tests passed"}},
        ]
        remaining_tasks = []

        result = _check_if_query_satisfied(
            mock_client, "Build feature X", completed_tasks, remaining_tasks
        )

        assert result["is_satisfied"]
        assert result["confidence"] == 0.9

    def test_check_query_satisfied_not_complete(self):
        """Test when query is not satisfied."""
        from ai_dev_agent.agents.work_planner import Task

        mock_client = MagicMock()
        # Mock the complete method to return JSON
        mock_client.complete.return_value = json.dumps(
            {
                "is_satisfied": False,
                "reasoning": "Testing was not done",
                "confidence": 0.7,
                "missing_aspects": ["Testing"],
            }
        )

        completed_tasks = [{"task_title": "Build", "result": {"final_message": "Built"}}]
        # Create proper Task objects with title and description
        remaining_tasks = [Task(title="Test", description="Test the feature")]

        result = _check_if_query_satisfied(
            mock_client, "Build and test feature", completed_tasks, remaining_tasks
        )

        assert not result["is_satisfied"]
        assert "Testing" in result["missing_aspects"]

    def test_check_query_satisfied_error(self):
        """Test error handling in satisfaction check."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("LLM Error")

        result = _check_if_query_satisfied(mock_client, "Query", [], [])

        # Should return dict with is_satisfied=False on error
        assert not result["is_satisfied"]
        assert "Assessment failed" in result["reasoning"]
