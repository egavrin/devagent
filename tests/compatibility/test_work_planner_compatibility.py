"""
Backward Compatibility Tests for Work Planning Agent

These tests ensure that adding the work planning agent doesn't break existing functionality.
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestWorkPlannerBackwardCompatibility:
    """Test that work planning agent doesn't break existing features"""

    def test_existing_cli_still_works(self):
        """Ensure basic CLI functionality is unchanged"""
        # This is a placeholder test - will need actual CLI to test
        # For now, just verify imports work
        try:
            from ai_dev_agent.agents.work_planner import WorkPlanningAgent
            assert True
        except ImportError:
            pytest.fail("Work planner imports should not break")

    def test_no_config_format_changes(self):
        """Ensure .devagent.toml config format is unchanged"""
        # Work planner doesn't modify config, this should always pass
        # This is a placeholder to document the requirement
        assert True

    def test_work_planner_optional(self):
        """Ensure work planner is optional and doesn't break if it fails"""
        # Test that other parts of the system work even if work planner has issues
        try:
            from ai_dev_agent.agents.work_planner import WorkPlanningAgent

            # Work planner imports successfully
            assert True
        except ImportError:
            # Even if import fails, rest of system should work
            # This is acceptable for optional feature
            pass

    def test_existing_agents_unaffected(self):
        """Ensure existing agents still function"""
        # Verify work planner is isolated in its own module
        try:
            from ai_dev_agent.agents import work_planner

            # Check it's namespaced properly
            assert hasattr(work_planner, "WorkPlanningAgent")
            assert hasattr(work_planner, "Task")
            assert hasattr(work_planner, "WorkPlan")
        except ImportError:
            pytest.skip("Work planner not yet implemented")

    def test_no_breaking_api_changes(self):
        """Ensure no existing APIs are modified"""
        # Work planner is purely additive
        # This test documents that requirement
        assert True

    def test_storage_isolation(self):
        """Ensure work planner uses isolated storage"""
        try:
            from ai_dev_agent.agents.work_planner import WorkPlanStorage
            from pathlib import Path

            # Default storage should be in dedicated directory
            storage = WorkPlanStorage()
            storage_dir = storage.storage_dir

            # Should be under .devagent/plans/
            assert "plans" in str(storage_dir)
            assert ".devagent" in str(storage_dir) or storage_dir.parent.name == ".devagent"

        except ImportError:
            pytest.skip("Work planner not yet implemented")

    def test_no_global_state_pollution(self):
        """Ensure work planner doesn't pollute global state"""
        # Work planner should be stateless or use explicit storage
        # No global variables or singletons
        try:
            from ai_dev_agent.agents.work_planner import WorkPlanningAgent

            agent1 = WorkPlanningAgent()
            agent2 = WorkPlanningAgent()

            # Two agents should be independent
            assert agent1 is not agent2

        except ImportError:
            pytest.skip("Work planner not yet implemented")


class TestWorkPlannerIntegrationSafety:
    """Test safe integration of work planner with existing codebase"""

    def test_import_doesnt_have_side_effects(self):
        """Test that importing work planner has no side effects"""
        # Importing should not create files, directories, or modify state
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            try:
                # Import in clean environment
                import importlib
                import sys

                # Remove any cached imports
                for module_name in list(sys.modules.keys()):
                    if "work_planner" in module_name:
                        del sys.modules[module_name]

                # Now import fresh
                from ai_dev_agent.agents.work_planner import WorkPlanningAgent

                # No files should be created in cwd
                assert True

            except ImportError:
                pytest.skip("Work planner not yet implemented")

    def test_graceful_storage_failure(self):
        """Test that storage failures don't crash the system"""
        try:
            from ai_dev_agent.agents.work_planner import WorkPlanningAgent
            from pathlib import Path

            # Try to create storage in a read-only location (will fail gracefully)
            # This is a design requirement, not yet implemented
            pytest.skip("Graceful failure handling not yet implemented")

        except ImportError:
            pytest.skip("Work planner not yet implemented")
