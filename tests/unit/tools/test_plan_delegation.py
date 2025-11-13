"""Tests for plan tool delegation safeguards."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ai_dev_agent.tools.registry import ToolContext
from ai_dev_agent.tools.workflow.plan import plan


class TestPlanDelegationSafeguards:
    """Test plan tool rejects nested planning attempts."""

    def test_plan_rejects_nested_planning_in_delegated_context(self):
        """Test that plan tool rejects calls when is_delegated=True."""
        # Create a tool context with delegation flag
        mock_settings = Mock()
        mock_settings.planning_enabled = True

        context = ToolContext(
            repo_root=Path.cwd(), settings=mock_settings, sandbox=None, extra={"is_delegated": True}
        )

        payload = {"goal": "Implement a new feature", "context": "Additional context"}

        result = plan(payload, context)

        # Should return error indicating nested planning is not allowed
        assert result["success"] is False
        assert "nested planning" in result["error"].lower()
        assert "not allowed" in result["error"].lower()

    def test_plan_allows_planning_in_top_level_context(self):
        """Test that plan tool works normally when not delegated."""
        # Create a tool context without delegation flag
        mock_settings = Mock()
        mock_settings.planning_enabled = True
        mock_settings.plan_max_tasks = 10

        # Mock LLM client
        mock_client = Mock()
        mock_client.complete.return_value = """
        {
            "tasks": [
                {
                    "title": "Task 1",
                    "description": "Do something",
                    "dependencies": []
                }
            ]
        }
        """

        context = ToolContext(
            repo_root=Path.cwd(),
            settings=mock_settings,
            sandbox=None,
            extra={"is_delegated": False, "llm_client": mock_client},
        )

        payload = {
            "goal": "Implement a complex feature with multiple components",
            "context": "Additional context",
        }

        result = plan(payload, context)

        # Should succeed and return a plan
        assert result["success"] is True
        assert "plan" in result
        assert len(result["plan"]["tasks"]) > 0

    def test_plan_allows_planning_when_extra_is_none(self):
        """Test that plan tool works when extra dict is None (no delegation context)."""
        # Create a tool context with None extra
        mock_settings = Mock()
        mock_settings.planning_enabled = True
        mock_settings.plan_max_tasks = 10

        # Mock LLM client
        mock_client = Mock()
        mock_client.complete.return_value = """
        {
            "tasks": [
                {
                    "title": "Task 1",
                    "description": "Do something",
                    "dependencies": []
                }
            ]
        }
        """

        context = ToolContext(
            repo_root=Path.cwd(),
            settings=mock_settings,
            sandbox=None,
            extra={"llm_client": mock_client},  # No is_delegated key
        )

        payload = {"goal": "Implement a complex feature", "context": ""}

        result = plan(payload, context)

        # Should succeed (no delegation flag means top-level)
        assert result["success"] is True
        assert "plan" in result

    def test_plan_missing_goal_parameter(self):
        """Test that plan tool returns error when goal is missing."""
        mock_settings = Mock()
        context = ToolContext(repo_root=Path.cwd(), settings=mock_settings, sandbox=None, extra={})

        payload = {}  # Missing goal

        result = plan(payload, context)

        assert result["success"] is False
        assert "missing required parameter" in result["error"].lower()
        assert "goal" in result["error"].lower()

    @patch("ai_dev_agent.core.utils.config.load_settings")
    def test_plan_respects_planning_disabled_setting(self, mock_load_settings):
        """Test that plan tool respects planning_enabled=False setting."""
        mock_settings = Mock()
        mock_settings.planning_enabled = False
        mock_load_settings.return_value = mock_settings

        context = ToolContext(
            repo_root=Path.cwd(),
            settings=mock_settings,
            sandbox=None,
            extra={"is_delegated": False},
        )

        payload = {"goal": "Some task"}

        result = plan(payload, context)

        # Should return success but with disabled flag
        assert result["success"] is True
        assert "plan" in result
        assert result["plan"]["disabled"] is True
        assert (
            "disabled" in result["result"].lower() or "execute directly" in result["result"].lower()
        )
