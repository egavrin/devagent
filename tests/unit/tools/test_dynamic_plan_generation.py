"""Tests for simplified plan generation without fallbacks or templates."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.tools.registry import ToolContext
from ai_dev_agent.tools.workflow.plan import _generate_tasks_from_goal, _needs_plan, plan


class TestComplexityDetection:
    """Test the complexity detection for optional planning."""

    def test_simple_tasks_skip_planning(self):
        """Test that simple tasks are detected correctly."""
        simple_tasks = [
            "fix typo in readme",
            "rename variable foo to bar",
            "update comment in main.py",
            "fix syntax error",
            "correct spelling mistake",
            "add import statement",
        ]

        for task in simple_tasks:
            assert not _needs_plan(task), f"'{task}' should not need planning"

    def test_complex_tasks_need_planning(self):
        """Test that complex tasks are detected correctly."""
        complex_tasks = [
            "implement authentication system",
            "refactor database layer",
            "design API architecture",
            "integrate payment gateway",
            "migrate to new framework",
            "optimize system performance",
        ]

        for task in complex_tasks:
            assert _needs_plan(task), f"'{task}' should need planning"

    def test_multi_part_tasks_need_planning(self):
        """Test that tasks with multiple parts need planning."""
        tasks_with_multiple_parts = [
            "add login and logout functionality",
            "fix bug, update tests, and document changes",
            "implement feature A, B, and C",
        ]

        for task in tasks_with_multiple_parts:
            assert _needs_plan(task), f"Multi-part task '{task}' should need planning"

    def test_short_vs_long_goals(self):
        """Test that goal length influences planning decision."""
        # Very short goal without complexity indicators
        short_simple = "fix bug"
        assert not _needs_plan(short_simple)

        # Long goal (> 200 chars)
        long_goal = "a" * 201
        assert _needs_plan(long_goal)


class TestFailFastBehavior:
    """Test that the system fails fast without fallbacks."""

    def test_no_llm_client_fails_fast(self):
        """Test that missing LLM client causes immediate failure."""
        mock_context = Mock(spec=ToolContext)
        mock_context.extra = {}

        with patch("ai_dev_agent.providers.llm.create_client") as mock_create:
            mock_create.side_effect = Exception("Cannot connect to LLM")

            with pytest.raises(RuntimeError, match="Cannot create LLM client"):
                _generate_tasks_from_goal("test goal", "", mock_context)

    def test_llm_failure_raises_error(self):
        """Test that LLM failures raise errors instead of falling back."""
        mock_context = Mock(spec=ToolContext)
        mock_client = Mock()
        mock_context.extra = {"llm_client": mock_client}

        # Make LLM fail
        mock_client.complete.side_effect = Exception("LLM error")

        with pytest.raises(RuntimeError, match="Failed to generate plan after .* attempts"):
            _generate_tasks_from_goal("test goal", "", mock_context)

    def test_invalid_json_retries_then_fails(self):
        """Test retry logic for invalid JSON then fail."""
        mock_context = Mock(spec=ToolContext)
        mock_client = Mock()
        mock_context.extra = {"llm_client": mock_client}

        # Return invalid JSON twice
        mock_client.complete.side_effect = ["not valid json", "still not json"]

        with pytest.raises(RuntimeError, match="Invalid JSON response"):
            _generate_tasks_from_goal("test goal", "", mock_context)

        # Should have tried twice
        assert mock_client.complete.call_count == 2

    def test_empty_task_list_fails(self):
        """Test that empty task list from LLM causes failure."""
        mock_context = Mock(spec=ToolContext)
        mock_client = Mock()
        mock_context.extra = {"llm_client": mock_client}

        # Return empty tasks
        mock_client.complete.return_value = json.dumps({"tasks": []})

        with pytest.raises(RuntimeError, match="empty task list"):
            _generate_tasks_from_goal("test goal", "", mock_context)


class TestSimplifiedPlanning:
    """Test the simplified planning logic."""

    def test_simple_task_returns_no_plan(self):
        """Test that simple tasks skip planning."""
        mock_context = Mock(spec=ToolContext)
        mock_context.repo_root = Path.cwd()
        mock_context.settings = Mock()
        mock_context.sandbox = None
        mock_context.extra = {}

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Settings(planning_enabled=True, plan_max_tasks=20)

            result = plan({"goal": "fix typo in readme"}, mock_context)

            assert result["success"]
            assert result["plan"]["simple"] is True
            assert result["plan"]["tasks"] == []
            assert "execute directly" in result["result"].lower()

    def test_planning_disabled_returns_empty(self):
        """Test that disabled planning returns empty plan."""
        mock_context = Mock(spec=ToolContext)
        mock_context.repo_root = Path.cwd()
        mock_context.settings = Mock()
        mock_context.sandbox = None

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Settings(
                planning_enabled=False, plan_max_tasks=20  # Disabled
            )

            result = plan({"goal": "implement complex feature"}, mock_context)

            assert result["success"]
            assert result["plan"]["disabled"] is True
            assert result["plan"]["tasks"] == []
            assert "disabled" in result["result"].lower()

    def test_complex_task_generates_plan(self):
        """Test that complex tasks generate appropriate plans."""
        mock_context = Mock(spec=ToolContext)
        mock_context.repo_root = Path.cwd()
        mock_context.settings = Mock()
        mock_context.sandbox = None
        mock_client = Mock()
        mock_context.extra = {"llm_client": mock_client}

        # Mock LLM to return a simple plan
        mock_client.complete.return_value = json.dumps(
            {
                "tasks": [
                    {
                        "title": "Design system",
                        "description": "Design the approach",
                        "dependencies": [],
                    },
                    {
                        "title": "Implement feature",
                        "description": "Build it",
                        "dependencies": ["task-1"],
                    },
                ]
            }
        )

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Settings(planning_enabled=True, plan_max_tasks=20)

            result = plan({"goal": "implement authentication system"}, mock_context)

            assert result["success"]
            assert result["plan"]["simple"] is False
            assert len(result["plan"]["tasks"]) == 2
            assert "task-1" in result["plan"]["tasks"][0]["id"]
            assert result["plan"]["tasks"][1]["dependencies"] == ["task-1"]

    def test_excessive_tasks_truncated(self):
        """Test that excessive task counts are truncated to limit."""
        mock_context = Mock(spec=ToolContext)
        mock_context.repo_root = Path.cwd()
        mock_context.settings = Mock()
        mock_context.sandbox = None
        mock_client = Mock()
        mock_context.extra = {"llm_client": mock_client}

        # Mock LLM to return 25 tasks
        many_tasks = [
            {"title": f"Task {i}", "description": f"Do {i}", "dependencies": []} for i in range(25)
        ]
        mock_client.complete.return_value = json.dumps({"tasks": many_tasks})

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Settings(
                planning_enabled=True, plan_max_tasks=10  # Set lower limit
            )

            tasks = _generate_tasks_from_goal("complex goal", "", mock_context)

            # Should be truncated to 10
            assert len(tasks) == 10
            assert tasks[-1]["id"] == "task-10"

    def test_dynamic_task_counts(self):
        """Test that plans can have varying task counts."""
        mock_context = Mock(spec=ToolContext)
        mock_context.repo_root = Path.cwd()
        mock_context.settings = Mock()
        mock_context.sandbox = None
        mock_client = Mock()
        mock_context.extra = {"llm_client": mock_client}

        test_cases = [
            (1, [{"title": "Single step", "description": "Just one thing", "dependencies": []}]),
            (
                3,
                [
                    {"title": "Step 1", "description": "First", "dependencies": []},
                    {"title": "Step 2", "description": "Second", "dependencies": ["task-1"]},
                    {"title": "Step 3", "description": "Third", "dependencies": ["task-2"]},
                ],
            ),
            (
                7,
                [
                    {"title": f"Step {i}", "description": f"Task {i}", "dependencies": []}
                    for i in range(1, 8)
                ],
            ),
        ]

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Settings(planning_enabled=True, plan_max_tasks=20)

            for expected_count, task_list in test_cases:
                mock_client.complete.return_value = json.dumps({"tasks": task_list})
                tasks = _generate_tasks_from_goal("goal", "", mock_context)
                assert len(tasks) == expected_count

    def test_simplified_prompt_content(self):
        """Test that prompts are simplified and clear."""
        mock_context = Mock(spec=ToolContext)
        mock_context.repo_root = Path.cwd()
        mock_context.settings = Mock()
        mock_context.sandbox = None
        mock_client = Mock()
        mock_context.extra = {"llm_client": mock_client}

        # Capture the prompt
        mock_client.complete.return_value = json.dumps(
            {"tasks": [{"title": "Task", "description": "Do it", "dependencies": []}]}
        )

        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Settings(planning_enabled=True, plan_max_tasks=20)

            _generate_tasks_from_goal("test goal", "context", mock_context)

            # Check the prompt sent to LLM
            call_args = mock_client.complete.call_args
            if call_args.args:
                messages = call_args.args[0]
            else:
                messages = call_args.kwargs.get("messages", [])
            prompt = messages[0].content if messages else ""

            # Verify simplified prompt elements
            assert "RIGHT number of steps" in prompt
            assert "Don't force Design→Test→Implement→Review" in prompt
            assert "as few or as many as actually needed" in prompt
            assert "No padding" not in prompt or "Don't add unnecessary" in prompt


class TestPromptSimplification:
    """Test that prompts are minimal and effective."""

    def test_planner_system_prompt_is_minimal(self):
        """Test that system prompt is simplified."""
        from pathlib import Path

        system_prompt_path = (
            Path(__file__).parent.parent.parent.parent
            / "ai_dev_agent"
            / "prompts"
            / "system"
            / "planner_system.md"
        )

        if system_prompt_path.exists():
            content = system_prompt_path.read_text()

            # Check for minimal guidance
            assert len(content) < 500  # Should be very short
            assert "no more, no less" in content.lower() or "right number" in content.lower()
            assert (
                "never force templates" in content.lower() or "never add filler" in content.lower()
            )

    def test_planner_user_prompt_is_direct(self):
        """Test that user prompt is direct and simple."""
        from pathlib import Path

        user_prompt_path = (
            Path(__file__).parent.parent.parent.parent
            / "ai_dev_agent"
            / "prompts"
            / "system"
            / "planner_user.md"
        )

        if user_prompt_path.exists():
            content = user_prompt_path.read_text()

            # Check for directness
            assert len(content) < 200  # Very concise
            assert "no padding" in content.lower() or "actual work" in content.lower()


class TestConfigurationSimplification:
    """Test simplified configuration."""

    def test_minimal_config_options(self):
        """Test that configuration is simplified."""
        settings = Settings()

        # Should only have essential options
        assert hasattr(settings, "planning_enabled")
        assert hasattr(settings, "plan_max_tasks")

        # Should NOT have complex options removed in simplification
        assert not hasattr(settings, "plan_use_fallback")
        assert not hasattr(settings, "plan_min_tasks")
        assert not hasattr(settings, "plan_force_dynamic")
        assert not hasattr(settings, "plan_max_retries")

    def test_planning_can_be_disabled(self):
        """Test that planning can be completely disabled."""
        with patch("ai_dev_agent.core.utils.config.load_settings") as mock_settings:
            mock_settings.return_value = Settings(planning_enabled=False)

            settings = mock_settings()
            assert settings.planning_enabled is False

            # When used in plan function, should return disabled plan
            mock_context = Mock(spec=ToolContext)
            mock_context.repo_root = Path.cwd()
            mock_context.settings = Mock()
            mock_context.sandbox = None
            result = plan({"goal": "any goal"}, mock_context)
            assert result["plan"].get("disabled") is True
