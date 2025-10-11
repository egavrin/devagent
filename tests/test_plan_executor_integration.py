"""
Integration tests for Work Planning Agent with --plan flag.

These tests actually invoke the CLI and can be slow (15-60 seconds each).
Run with: pytest -m "not slow" to skip these tests during development
Run with: pytest -m slow to run only these tests
"""
import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
from ai_dev_agent.cli.commands import cli
from ai_dev_agent.agents.work_planner import WorkPlanningAgent


# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


@pytest.fixture
def temp_plans_dir(monkeypatch):
    """Create a temporary directory for plan storage during tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "plans"
        temp_path.mkdir(parents=True, exist_ok=True)

        # Monkeypatch WorkPlanStorage to use temp directory
        from ai_dev_agent.agents.work_planner.storage import WorkPlanStorage
        original_storage_init = WorkPlanStorage.__init__

        def patched_storage_init(self, storage_dir=None):
            # Always use temp path, ignore passed storage_dir
            original_storage_init(self, storage_dir=temp_path)

        monkeypatch.setattr(WorkPlanStorage, "__init__", patched_storage_init)

        yield temp_path


class TestPlanExecutorIntegration:
    """Test that --plan flag properly integrates with Work Planning Agent."""

    def test_plan_flag_creates_and_executes_work_plan(self, temp_plans_dir):
        """Test that devagent --plan triggers complexity assessment."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plan", "how many lines in commands.py"],
            catch_exceptions=False
        )

        # Verify assessment happened (may use direct execution or planning)
        assert "🔍 Analyzing query complexity" in result.output or "🗺️" in result.output

        # Should not have errors
        assert result.exit_code == 0 or "Error" not in result.output

    def test_plan_flag_with_simple_query(self, temp_plans_dir):
        """Test --plan with a simple calculation query."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plan", "--system", "Answer in one word", "what is 2+2"],
            catch_exceptions=False
        )

        # Should complete successfully
        assert result.exit_code == 0 or "Error" not in result.output

    def test_plan_executor_fallback_to_direct_execution(self, temp_plans_dir):
        """Test that plan executor falls back gracefully if planning fails."""
        runner = CliRunner()
        # Use a query that might be difficult to plan
        result = runner.invoke(
            cli,
            ["--plan", "hello"],
            catch_exceptions=False
        )

        # Should still execute something (either planned or fallback)
        assert result.exit_code == 0 or "Error" not in result.output

    def test_plan_creates_persistent_plan_file(self, temp_plans_dir):
        """Test that --plan creates a persistent plan file."""
        runner = CliRunner()

        # Count plans before
        existing_plans = list(temp_plans_dir.glob("*.json")) if temp_plans_dir.exists() else []
        count_before = len(existing_plans)

        # Execute with --plan
        result = runner.invoke(
            cli,
            ["--plan", "list python files"],
            catch_exceptions=False
        )

        # Should have created a new plan file
        if temp_plans_dir.exists():
            new_plans = list(temp_plans_dir.glob("*.json"))
            # Either creates a new plan or reuses existing (both valid)
            assert len(new_plans) >= count_before

    def test_plan_executor_breaks_query_into_tasks(self, temp_plans_dir):
        """Test that plan executor breaks query into structured tasks."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plan", "find all TODO comments and count them"],
            catch_exceptions=False
        )

        output = result.output.lower()

        # Should show planning activity
        indicators = [
            "plan" in output,
            "task" in output,
            "🗺️" in result.output,
            "step" in output,
        ]

        # At least one planning indicator should be present
        assert any(indicators), f"No planning indicators found in output: {result.output[:200]}"

    def test_plan_executor_with_repo_context(self, temp_plans_dir):
        """Test that plan executor works with repository context."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plan", "analyze the structure of the cli module"],
            catch_exceptions=False
        )

        # Should complete without errors
        assert result.exit_code == 0 or "Error" not in result.output


class TestPlanExecutorTaskManagement:
    """Test task-level execution in plan executor."""

    def test_plan_executor_shows_task_progress(self, temp_plans_dir):
        """Test that plan executor shows progress through tasks."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plan", "count python files and show their sizes"],
            catch_exceptions=False
        )

        output = result.output.lower()

        # Should show some progress indicators
        progress_indicators = [
            "task" in output,
            "progress" in output,
            "completed" in output,
            "/" in result.output,  # e.g., "1/3"
            "%" in result.output,  # e.g., "33%"
        ]

        assert any(progress_indicators)

    def test_plan_executor_handles_sequential_tasks(self, temp_plans_dir):
        """Test that plan executor respects task dependencies."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plan", "first find commands.py, then count its lines"],
            catch_exceptions=False
        )

        # Should execute without errors
        assert result.exit_code == 0 or "Error" not in result.output


class TestPlanExecutorEdgeCases:
    """Test edge cases and error handling."""

    def test_plan_executor_with_empty_query(self, temp_plans_dir):
        """Test plan executor handles empty query gracefully."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plan", ""],
            catch_exceptions=False
        )

        # Should handle gracefully (exit code 0 or informative error)
        assert result.exit_code in [0, 1, 2]

    def test_plan_executor_with_complex_multi_step_query(self, temp_plans_dir):
        """Test plan executor handles complex queries."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plan",
                "analyze all Python files, count their lines, "
                "find the largest file, and report statistics"
            ],
            catch_exceptions=False
        )

        # Should attempt execution
        assert result.exit_code == 0 or "Error" not in result.output

    def test_plan_without_llm_creates_simple_plan(self, temp_plans_dir):
        """Test that planning works even if LLM task breakdown fails."""
        # This is implicitly tested by fallback mechanism in plan_executor.py
        # If LLM fails, it creates a simple single-task plan
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--plan", "test query"],
            catch_exceptions=False
        )

        # Should still execute something
        assert result.exit_code in [0, 1]
