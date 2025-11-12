"""Tests for simplified plan tool without fallbacks."""

import json
import sys
from pathlib import Path
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
    context.repo_root = Path("/test")
    context.settings = Mock()
    context.sandbox = None
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

    def test_simple_task_skips_planning(self, mock_context):
        """Simple tasks should skip planning entirely."""
        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)

            result = plan({"goal": "fix typo in readme"}, mock_context)

        assert result["success"] is True
        assert "plan" in result
        assert result["plan"]["simple"] is True
        assert result["plan"]["tasks"] == []
        assert "execute directly" in result["result"].lower()

    def test_complex_task_generates_plan(self, mock_context):
        """Complex tasks should generate dynamic plan."""
        mock_tasks = [
            {
                "id": "task-1",
                "title": "Design",
                "description": "Design the system",
                "dependencies": [],
            },
            {
                "id": "task-2",
                "title": "Build",
                "description": "Build it",
                "dependencies": ["task-1"],
            },
        ]

        mock_context.extra["llm_client"].complete.return_value = json.dumps({"tasks": mock_tasks})

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)

            result = plan({"goal": "implement authentication system"}, mock_context)

        assert result["success"] is True
        assert result["plan"]["simple"] is False
        assert len(result["plan"]["tasks"]) == 2

    def test_plan_with_context(self, mock_context):
        """Create plan with additional context."""
        mock_tasks = [
            {"id": "task-1", "title": "Task", "description": "Do task", "dependencies": []}
        ]
        mock_context.extra["llm_client"].complete.return_value = json.dumps({"tasks": mock_tasks})

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)
            with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
                mock_generate.return_value = mock_tasks
                result = plan(
                    {"goal": "optimize database", "context": "Focus on queries"}, mock_context
                )

        assert result["success"] is True
        # Verify context was passed to generator
        mock_generate.assert_called_once()
        assert mock_generate.call_args[0][1] == "Focus on queries"

    def test_planning_disabled_returns_empty(self, mock_context):
        """When planning is disabled, should return disabled plan."""
        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=False, plan_max_tasks=20)

            result = plan({"goal": "complex task"}, mock_context)

        assert result["success"] is True
        assert result["plan"]["disabled"] is True
        assert result["plan"]["tasks"] == []
        assert "disabled" in result["result"].lower()

    def test_dynamic_task_count(self, mock_context):
        """Plans can have any number of tasks, not fixed counts."""
        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)

            # Test with 1 task
            mock_context.extra["llm_client"].complete.return_value = json.dumps(
                {"tasks": [{"title": "Single", "description": "One task", "dependencies": []}]}
            )
            result = plan({"goal": "refactor module"}, mock_context)
            assert result["success"] is True
            assert len(result["plan"]["tasks"]) == 1

            # Test with 5 tasks - need fresh mock
            five_tasks = [
                {"title": f"Task {i}", "description": f"Do {i}", "dependencies": []}
                for i in range(1, 6)
            ]
            mock_context.extra["llm_client"].complete.return_value = json.dumps(
                {"tasks": five_tasks}
            )

            result = plan({"goal": "build microservice"}, mock_context)
            assert result["success"] is True
            assert len(result["plan"]["tasks"]) == 5


class TestExceptionHandling:
    """Test exception scenarios without fallbacks."""

    def test_generation_fails_fast(self, mock_context):
        """Exception in task generation fails immediately."""
        with patch.object(plan_module, "_generate_tasks_from_goal") as mock_generate:
            mock_generate.side_effect = RuntimeError("LLM error")
            result = plan({"goal": "implement authentication system"}, mock_context)

        assert result["success"] is False
        assert "Plan creation failed" in result["error"]
        assert "LLM error" in result["error"]

    def test_no_llm_client_fails_fast(self, mock_context):
        """Missing LLM client fails fast."""
        mock_context.extra = {}  # No LLM client

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)

            result = plan({"goal": "implement feature"}, mock_context)

        # Should fail because there's no LLM client
        assert result["success"] is False
        assert (
            "Cannot create LLM client" in result["error"]
            or "Plan creation failed" in result["error"]
        )

    def test_invalid_json_fails_after_retries(self, mock_context):
        """Invalid JSON response fails after retries, no fallback."""
        mock_context.extra["llm_client"].complete.side_effect = ["invalid json", "still invalid"]

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)

            result = plan({"goal": "implement complex feature"}, mock_context)

        # Should fail without falling back to template
        assert result["success"] is False
        assert "Invalid JSON" in result["error"] or "Plan creation failed" in result["error"]


class TestTaskGeneration:
    """Test actual task generation with mocked LLM."""

    def test_task_generation_uses_llm(self, mock_context):
        """Task generation uses LLM client from context."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = """{
            "tasks": [
                {"title": "Design API", "description": "Design REST API", "dependencies": []},
                {"title": "Implement API", "description": "Code API", "dependencies": ["task-1"]}
            ]
        }"""
        mock_context.extra = {"llm_client": mock_llm}

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)

            result = plan({"goal": "Build API"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 2
        assert result["plan"]["tasks"][0]["title"] == "Design API"
        # Verify LLM was called
        mock_llm.complete.assert_called_once()

    def test_excessive_tasks_truncated(self, mock_context):
        """Tasks exceeding max limit are truncated."""
        many_tasks = [
            {"title": f"Task {i}", "description": f"Do {i}", "dependencies": []} for i in range(30)
        ]
        mock_context.extra["llm_client"].complete.return_value = json.dumps({"tasks": many_tasks})

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=10)

            result = plan({"goal": "implement massive project"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 10  # Truncated to max

    def test_empty_task_list_fails(self, mock_context):
        """Empty task list from LLM causes failure."""
        mock_context.extra["llm_client"].complete.return_value = json.dumps({"tasks": []})

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)

            result = plan({"goal": "implement authentication"}, mock_context)

        # Should fail with empty task list
        assert result["success"] is False
        assert "empty task list" in result["error"] or "Plan creation failed" in result["error"]


class TestMarkdownParsing:
    """Test markdown code block extraction."""

    def test_llm_response_with_json_code_block(self, mock_context):
        """LLM response wrapped in ```json code block."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = """```json
{
    "tasks": [
        {"title": "Design", "description": "Design it", "dependencies": []},
        {"title": "Code", "description": "Code it", "dependencies": ["task-1"]}
    ]
}
```"""
        mock_context.extra = {"llm_client": mock_llm}

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)

            result = plan({"goal": "build feature"}, mock_context)

        assert result["success"] is True
        assert len(result["plan"]["tasks"]) == 2

    def test_plan_context_included_in_prompt(self, mock_context):
        """Context parameter is included in LLM prompt."""
        mock_llm = MagicMock()

        captured_prompt = []

        def capture(messages, **kwargs):
            captured_prompt.append(messages[0].content)
            return '{"tasks": [{"title": "T", "description": "D", "dependencies": []}]}'

        mock_llm.complete.side_effect = capture
        mock_context.extra = {"llm_client": mock_llm}

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Mock(planning_enabled=True, plan_max_tasks=20)

            result = plan(
                {"goal": "Build API", "context": "REST endpoints with auth"}, mock_context
            )

        assert result["success"] is True
        # Verify context was in prompt
        assert "REST endpoints with auth" in captured_prompt[0]
