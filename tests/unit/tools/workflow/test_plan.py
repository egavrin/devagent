"""Tests for plan tool."""

import json
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import function for testing, but get module reference for patching
from ai_dev_agent.tools.workflow.plan import plan

# Get actual module object (not the function) for proper mocking
plan_module = sys.modules["ai_dev_agent.tools.workflow.plan"]


@pytest.fixture
def mock_context():
    """Create mock ToolContext."""
    context = Mock()
    context.repo_root = "/test"
    context.settings = {}
    context.extra = {"llm_client": MagicMock()}
    return context


class TestValidation:
    """Test input validation."""

    def test_missing_goal(self, mock_context):
        """Missing goal should return error."""
        result = plan({}, mock_context)
        assert result["success"] is False
        assert "Missing required parameter: goal" in result["error"]

    def test_empty_goal(self, mock_context):
        """Empty goal should return error."""
        result = plan({"goal": ""}, mock_context)
        assert result["success"] is False
        assert "Missing required parameter: goal" in result["error"]

    def test_none_goal(self, mock_context):
        """None goal should return error."""
        result = plan({"goal": None}, mock_context)
        assert result["success"] is False
        assert "Missing required parameter: goal" in result["error"]


class TestPlanCreation:
    """Test plan creation scenarios."""

    def test_simple_plan(self, mock_context):
        """Create simple plan."""
        mock_tasks = [{"id": "task-1", "title": "Design", "agent": "design_agent"}]

        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = mock_tasks
                result = plan({"goal": "Build feature"}, mock_context)

        assert result["success"] is True
        assert "plan" in result
        assert result["plan"]["goal"] == "Build feature"
        assert len(result["plan"]["tasks"]) == 1
        assert result["plan"]["complexity"] == "medium"  # default

    def test_plan_with_complexity(self, mock_context):
        """Create plan with specified complexity."""
        mock_tasks = [{"id": "t1"}]

        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = mock_tasks
                result = plan({"goal": "Refactor", "complexity": "simple"}, mock_context)

        assert result["success"] is True
        assert result["plan"]["complexity"] == "simple"

    def test_plan_with_context(self, mock_context):
        """Create plan with additional context."""
        mock_tasks = [{"id": "t1"}]

        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = mock_tasks
                result = plan(
                    {"goal": "Optimize", "context": "Focus on database queries"}, mock_context
                )

        assert result["success"] is True
        # Verify context was passed to generator
        mock_generate.assert_called_once()
        assert mock_generate.call_args[0][1] == "Focus on database queries"

    def test_complex_plan_multiple_tasks(self, mock_context):
        """Create complex plan with multiple tasks."""
        mock_tasks = [
            {"id": "t1", "title": "Design"},
            {"id": "t2", "title": "Implement"},
            {"id": "t3", "title": "Test"},
            {"id": "t4", "title": "Review"},
        ]

        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = mock_tasks
                result = plan({"goal": "Build microservice", "complexity": "complex"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 4
        assert "complex plan with 4 tasks" in result["result"]
        assert "delegate tool" in result["result"]

    def test_tracking_called(self, mock_context):
        """Plan tracking should be initiated."""
        mock_tasks = [{"id": "t1"}]

        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as mock_tracking:
                mock_generate.return_value = mock_tasks
                result = plan({"goal": "Test"}, mock_context)

        assert result["success"] is True
        mock_tracking.assert_called_once()
        # Verify plan structure passed to tracking
        call_args = mock_tracking.call_args[0][0]
        assert call_args["goal"] == "Test"
        assert "tasks" in call_args

    def test_default_complexity_medium(self, mock_context):
        """Default complexity should be medium."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = []
                result = plan({"goal": "Test"}, mock_context)

        assert result["success"] is True
        assert result["plan"]["complexity"] == "medium"

    def test_result_message_format(self, mock_context):
        """Result message should contain task count and instructions."""
        mock_tasks = [{"id": "1"}, {"id": "2"}]

        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = mock_tasks
                result = plan({"goal": "Test"}, mock_context)

        assert result["success"] is True
        assert "2 tasks" in result["result"]
        assert "delegate tool" in result["result"]

    def test_empty_task_list(self, mock_context):
        """Handle empty task list gracefully."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = []
                result = plan({"goal": "Test"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 0
        assert "0 tasks" in result["result"]


class TestExceptionHandling:
    """Test exception scenarios."""

    def test_generation_exception(self, mock_context):
        """Exception in task generation is handled."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            mock_generate.side_effect = RuntimeError("LLM error")
            result = plan({"goal": "Test"}, mock_context)

        assert result["success"] is False
        assert "Plan creation failed" in result["error"]
        assert "LLM error" in result["error"]

    def test_tracking_exception(self, mock_context):
        """Exception in plan tracking is handled."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as mock_tracking:
                mock_generate.return_value = [{"id": "t1"}]
                mock_tracking.side_effect = RuntimeError("Tracking error")
                result = plan({"goal": "Test"}, mock_context)

        assert result["success"] is False
        assert "Plan creation failed" in result["error"]


class TestComplexityLevels:
    """Test different complexity levels."""

    def test_simple_complexity(self, mock_context):
        """Test simple complexity level."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = [{"id": "t1"}]
                result = plan({"goal": "Test", "complexity": "simple"}, mock_context)

        assert result["success"] is True
        assert result["plan"]["complexity"] == "simple"
        assert "simple plan" in result["result"]

    def test_medium_complexity(self, mock_context):
        """Test medium complexity level."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = [{"id": "t1"}]
                result = plan({"goal": "Test", "complexity": "medium"}, mock_context)

        assert result["success"] is True
        assert result["plan"]["complexity"] == "medium"
        assert "medium plan" in result["result"]

    def test_complex_complexity(self, mock_context):
        """Test complex complexity level."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = [{"id": "t1"}]
                result = plan({"goal": "Test", "complexity": "complex"}, mock_context)

        assert result["success"] is True
        assert result["plan"]["complexity"] == "complex"
        assert "complex plan" in result["result"]


class TestContextParameter:
    """Test context parameter handling."""

    def test_with_context(self, mock_context):
        """Context parameter is passed through."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = []
                result = plan({"goal": "Test", "context": "Important context"}, mock_context)

        assert result["success"] is True
        # Verify context was passed
        assert mock_generate.call_args[0][1] == "Important context"

    def test_without_context(self, mock_context):
        """Missing context defaults to empty string."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            with patch.object(plan_module, "start_plan_tracking") as _:
                mock_generate.return_value = []
                result = plan({"goal": "Test"}, mock_context)

        assert result["success"] is True
        # Verify empty string passed
        assert mock_generate.call_args[0][1] == ""


class TestTaskGeneration:
    """Test actual task generation with mocked LLM."""

    def test_task_generation_with_llm_client(self, mock_context):
        """Task generation uses LLM client from context."""
        # Create mock LLM client that returns valid JSON (simple complexity = 2 tasks)
        mock_llm = MagicMock()
        mock_llm.complete.return_value = """{
            "tasks": [
                {"title": "Design API", "agent": "design_agent", "description": "Design REST API"},
                {"title": "Implement API", "agent": "implementation_agent", "description": "Code API", "dependencies": ["task-1"]}
            ]
        }"""
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            result = plan({"goal": "Build API", "complexity": "simple"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 2  # simple = 2 tasks
        assert result["plan"]["tasks"][0]["title"] == "Design API"
        # Verify LLM was called
        mock_llm.complete.assert_called_once()

    def test_task_generation_with_complex_llm_response(self, mock_context):
        """Task generation handles complex LLM response."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = """{
            "tasks": [
                {"title": "Design", "agent": "design_agent", "description": "Design system"},
                {"title": "Code", "agent": "implementation_agent", "description": "Write code"},
                {"title": "Test", "agent": "test_agent", "description": "Add tests"},
                {"title": "Review", "agent": "review_agent", "description": "Review code"}
            ]
        }"""
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            result = plan({"goal": "Build feature", "complexity": "medium"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 4  # medium = 4 tasks
        assert result["plan"]["tasks"][1]["agent"] == "implementation_agent"

    def test_task_generation_invalid_json_uses_fallback(self, mock_context):
        """Invalid LLM JSON response falls back to simple tasks."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "invalid json {"
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            result = plan({"goal": "Build API"}, mock_context)

        # Should still succeed with fallback tasks
        assert result["success"] is True
        assert len(result["plan"]["tasks"]) > 0

    def test_task_generation_llm_error_uses_fallback(self, mock_context):
        """LLM error falls back to simple tasks."""
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("API error")
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            result = plan({"goal": "Build feature"}, mock_context)

        # Should succeed with fallback
        assert result["success"] is True
        assert len(result["plan"]["tasks"]) > 0

    def test_task_generation_complexity_affects_count(self, mock_context):
        """Different complexity levels request different task counts."""
        mock_llm = MagicMock()

        # Track the prompts sent to LLM
        prompts = []

        def capture_prompt(messages, **kwargs):
            prompts.append(messages[0].content)
            # Return appropriate number of tasks based on prompt
            count = 2 if "2 tasks" in messages[0].content else 6
            tasks = [
                {"title": f"T{i}", "agent": "design_agent", "description": f"D{i}"}
                for i in range(1, count + 1)
            ]
            return json.dumps({"tasks": tasks})

        mock_llm.complete.side_effect = capture_prompt
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            # Test simple complexity (should request 2 tasks)
            plan({"goal": "Goal", "complexity": "simple"}, mock_context)
            assert "2 tasks" in prompts[0]

            # Test complex complexity (should request 6 tasks)
            plan({"goal": "Goal", "complexity": "complex"}, mock_context)
            assert "6 tasks" in prompts[1]


class TestFallbackGeneration:
    """Test fallback task generation."""

    def test_fallback_simple_complexity(self):
        """Fallback with simple complexity returns 2 tasks."""
        from ai_dev_agent.tools.workflow.plan import _generate_tasks_fallback

        tasks = _generate_tasks_fallback("Build API", "", "simple")

        assert len(tasks) == 2
        assert tasks[0]["agent"] == "design_agent"
        assert tasks[1]["agent"] == "test_agent"
        assert "Design" in tasks[0]["title"]
        assert "Build API" in tasks[0]["description"]

    def test_fallback_medium_complexity(self):
        """Fallback with medium complexity returns 4 tasks."""
        from ai_dev_agent.tools.workflow.plan import _generate_tasks_fallback

        tasks = _generate_tasks_fallback("Add feature", "", "medium")

        assert len(tasks) == 4
        assert tasks[0]["agent"] == "design_agent"
        assert tasks[1]["agent"] == "test_agent"
        assert tasks[2]["agent"] == "implementation_agent"
        assert tasks[3]["agent"] == "review_agent"

    def test_fallback_complex_complexity(self):
        """Fallback with complex complexity returns 6 tasks with extra phases."""
        from ai_dev_agent.tools.workflow.plan import _generate_tasks_fallback

        tasks = _generate_tasks_fallback("Complete solution", "", "complex")

        assert len(tasks) == 6
        # Check extra tasks for complex plans
        assert tasks[4]["title"].startswith("Integration")
        assert tasks[5]["title"].startswith("Documentation")
        assert tasks[4]["dependencies"] == ["task-4"]
        assert tasks[5]["dependencies"] == ["task-5"]

    def test_fallback_long_goal_truncated(self):
        """Fallback truncates long goals in task titles."""
        from ai_dev_agent.tools.workflow.plan import _generate_tasks_fallback

        long_goal = "A" * 100  # 100 character goal
        tasks = _generate_tasks_fallback(long_goal, "", "simple")

        # Title should contain truncated goal (60 chars)
        assert len(tasks[0]["title"]) < len(long_goal) + 20
        # But description should have full goal
        assert long_goal in tasks[0]["description"]


class TestMarkdownParsing:
    """Test markdown code block extraction."""

    def test_llm_response_with_json_code_block(self, mock_context):
        """LLM response wrapped in ```json code block."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = """```json
{
    "tasks": [
        {"title": "Design", "agent": "design_agent", "description": "Design"},
        {"title": "Code", "agent": "implementation_agent", "description": "Code"}
    ]
}
```"""
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            result = plan({"goal": "Test", "complexity": "simple"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 2

    def test_llm_response_with_plain_code_block(self, mock_context):
        """LLM response wrapped in ``` code block without json marker."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = """```
{
    "tasks": [
        {"title": "Design", "agent": "design_agent", "description": "Design"},
        {"title": "Code", "agent": "implementation_agent", "description": "Code"}
    ]
}
```"""
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            result = plan({"goal": "Test", "complexity": "simple"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 2

    def test_llm_response_wrong_task_count_uses_fallback(self, mock_context):
        """LLM returns wrong number of tasks, falls back."""
        mock_llm = MagicMock()
        # Simple complexity expects 2 tasks, but we return 3
        mock_llm.complete.return_value = """{
            "tasks": [
                {"title": "T1", "agent": "design_agent", "description": "D1"},
                {"title": "T2", "agent": "test_agent", "description": "D2"},
                {"title": "T3", "agent": "implementation_agent", "description": "D3"}
            ]
        }"""
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            result = plan({"goal": "Test", "complexity": "simple"}, mock_context)

        # Should succeed with fallback (2 tasks for simple)
        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 2  # Fallback returns correct count

    def test_llm_response_as_non_string_object(self, mock_context):
        """LLM returns non-string response object."""
        mock_llm = MagicMock()
        # Return an object that needs str() conversion
        response_obj = MagicMock()
        response_obj.__str__ = (
            lambda self: '{"tasks": [{"title": "T", "agent": "design_agent", "description": "D"}, {"title": "T2", "agent": "test_agent", "description": "D2"}]}'
        )
        mock_llm.complete.return_value = response_obj
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            result = plan({"goal": "Test", "complexity": "simple"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 2

    def test_plan_context_included_in_prompt(self, mock_context):
        """Context parameter is included in LLM prompt."""
        mock_llm = MagicMock()

        captured_prompt = []

        def capture(messages, **kwargs):
            captured_prompt.append(messages[0].content)
            return '{"tasks": [{"title": "T", "agent": "design_agent", "description": "D"}, {"title": "T2", "agent": "test_agent", "description": "D2"}]}'

        mock_llm.complete.side_effect = capture
        mock_context.extra = {"llm_client": mock_llm}

        with patch.object(plan_module, "start_plan_tracking") as _:
            result = plan(
                {"goal": "Build API", "context": "REST endpoints with auth"}, mock_context
            )

        assert result["success"] is True
        # Verify context was in prompt
        assert "REST endpoints with auth" in captured_prompt[0]
        assert "**Context**:" in captured_prompt[0]
