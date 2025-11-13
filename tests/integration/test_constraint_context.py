"""
Integration tests verifying that user constraints are preserved through prompts.

These tests verify the actual prompt content and context propagation without
needing to mock complex execution paths.
"""

import pytest


class TestConstraintInPrompts:
    """Test that constraints are included in generated prompts."""

    def test_planning_prompt_includes_constraint_instructions(self):
        """Verify that the planning system prompt includes constraint preservation instructions."""
        from pathlib import Path

        planner_prompt_path = Path("ai_dev_agent/prompts/system/planner_system.md")
        assert planner_prompt_path.exists(), "Planner system prompt file should exist"

        content = planner_prompt_path.read_text()

        # Verify constraint preservation instructions are present
        assert (
            "CRITICAL" in content or "Preserve" in content
        ), "Planning prompt should include critical instructions about constraints"
        assert "constraint" in content.lower(), "Planning prompt should mention constraints"
        assert (
            "don't write code" in content or "read-only" in content
        ), "Planning prompt should give examples of constraints"

    def test_react_loop_prompt_includes_constraint_instructions(self):
        """Verify that the ReAct loop prompt includes constraint handling instructions."""
        from pathlib import Path

        react_prompt_path = Path("ai_dev_agent/prompts/system/react_loop.md")
        assert react_prompt_path.exists(), "ReAct loop prompt file should exist"

        content = react_prompt_path.read_text()

        # Verify constraint instructions are present
        assert "constraint" in content.lower(), "ReAct prompt should mention constraints"
        assert (
            "Honor" in content or "respect" in content.lower()
        ), "ReAct prompt should instruct to honor constraints"

    def test_plan_generation_format_includes_constraint_fields(self):
        """Verify that plan generation prompt mentions constraint preservation."""
        # Check the function to see if it has the right prompt structure
        import inspect

        from ai_dev_agent.tools.workflow.plan import _generate_tasks_from_goal

        source = inspect.getsource(_generate_tasks_from_goal)

        # Verify the prompt includes constraint instructions
        assert (
            "CRITICAL" in source or "constraint" in source.lower()
        ), "Plan generation should include constraint handling"

    def test_task_execution_adds_original_context(self):
        """Verify that task execution code adds original user request to task prompt."""
        from pathlib import Path

        executor_path = Path("ai_dev_agent/cli/react/plan_executor.py")
        assert executor_path.exists()

        content = executor_path.read_text()

        # Verify original context is included in task prompts
        assert (
            "Original User Request" in content
        ), "Task execution should include original user request"
        assert (
            "original_user_prompt" in content or "original_user_request" in content
        ), "Task execution should track original user prompt"

    def test_delegation_propagates_original_context(self):
        """Verify that delegation includes original context."""
        from pathlib import Path

        delegate_path = Path("ai_dev_agent/tools/workflow/delegate.py")
        assert delegate_path.exists()

        content = delegate_path.read_text()

        # Verify original context propagation
        assert (
            "original_request" in content or "original_user_request" in content
        ), "Delegation should handle original user request"
        assert (
            "Original User Request" in content
        ), "Delegation should include original request in prompts"

    def test_executor_accepts_original_user_request_parameter(self):
        """Verify that executor function signature includes original_user_request."""
        import inspect

        from ai_dev_agent.cli.react.executor import _execute_react_assistant

        sig = inspect.signature(_execute_react_assistant)
        params = list(sig.parameters.keys())

        assert (
            "original_user_request" in params
        ), "Executor should accept original_user_request parameter"

    def test_session_metadata_stores_original_request(self):
        """Verify that session metadata includes original user request."""
        from pathlib import Path

        executor_path = Path("ai_dev_agent/cli/react/executor.py")
        content = executor_path.read_text()

        # Check that session metadata includes original request
        assert (
            '"original_user_request"' in content
        ), "Session metadata should store original user request"


class TestConstraintPreservationLogic:
    """Test the logic that preserves constraints."""

    def test_plan_executor_stores_original_prompt(self):
        """Verify plan executor preserves original prompt for tasks."""
        from pathlib import Path

        plan_exec_path = Path("ai_dev_agent/cli/react/plan_executor.py")
        content = plan_exec_path.read_text()

        # Verify original prompt is stored
        assert (
            "original_user_prompt = user_prompt" in content or "original_user_prompt = " in content
        ), "Plan executor should store original user prompt"

    def test_delegate_retrieves_from_session_if_not_provided(self):
        """Verify delegate can retrieve original request from session metadata."""
        from pathlib import Path

        delegate_path = Path("ai_dev_agent/tools/workflow/delegate.py")
        content = delegate_path.read_text()

        # Verify session metadata fallback
        assert "session.metadata" in content, "Delegate should access session metadata"
        assert "SessionManager" in content, "Delegate should use SessionManager to get session"


class TestPromptExamples:
    """Test that prompt examples demonstrate constraint handling."""

    def test_planning_prompt_has_good_example(self):
        """Verify planning prompt includes a good example of constraint preservation."""
        from pathlib import Path

        planner_prompt_path = Path("ai_dev_agent/prompts/system/planner_system.md")
        content = planner_prompt_path.read_text()

        # Check for example showing constraint preservation
        assert (
            "Example" in content or "example" in content
        ), "Planning prompt should include examples"

        # Check that examples mention constraints
        lines = content.lower().split("\n")
        has_constraint_example = any(
            "read-only" in line or "no code" in line or "analyze" in line for line in lines
        )
        assert has_constraint_example, "Planning prompt should have examples with constraints"

    def test_planning_tool_has_updated_example(self):
        """Verify plan.py has updated example format."""
        from pathlib import Path

        plan_path = Path("ai_dev_agent/tools/workflow/plan.py")
        content = plan_path.read_text()

        # Check for constraint-aware examples
        assert (
            "read-only" in content or "no code changes" in content
        ), "Plan tool should have constraint-aware examples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
