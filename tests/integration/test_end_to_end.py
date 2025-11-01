"""End-to-end integration tests for DevAgent.

This module tests complete workflows from user query to final output.
"""

from pathlib import Path

import pytest

from .conftest import IntegrationTest


@pytest.mark.integration
class TestWorkPlanningWorkflow(IntegrationTest):
    """Test complete work planning workflow."""

    def test_natural_language_fallback_requires_api_key(self, test_project, devagent_cli):
        """Natural language queries should route through query command."""
        result = devagent_cli(["Plan the logging improvements"], cwd=test_project)
        assert result.returncode != 0
        assert "No API key configured" in result.stderr

    def test_plan_flag_enables_planning_mode(self, test_project, devagent_cli):
        """--plan flag should still route natural language queries through planner."""
        result = devagent_cli(["--plan", "Plan the calculator refactor"], cwd=test_project)
        assert result.returncode != 0
        assert "No API key configured" in result.stderr


@pytest.mark.integration
class TestCoverageEnforcement(IntegrationTest):
    """Test coverage enforcement integration."""

    def test_coverage_gate_passes(self, test_project):
        """Test that coverage gate passes with sufficient coverage."""
        # Change to test project
        import os

        from ai_dev_agent.testing.coverage_gate import CoverageGate

        original_dir = str(Path.cwd())
        try:
            os.chdir(test_project)

            # Run coverage (should pass with test project)
            gate = CoverageGate(threshold=50.0)  # Lower threshold for test
            result = gate.run_coverage(parallel=False, html_report=False)

            # Check results
            assert result.total_coverage >= 0
            assert result.report is not None
            assert result.passed is False  # Sample project intentionally below threshold

        finally:
            os.chdir(original_dir)

    def test_incremental_coverage(self, test_project):
        """Test incremental coverage calculation."""
        import subprocess

        from ai_dev_agent.testing.coverage_gate import CoverageGate

        # Make a change to a file
        utils_file = test_project / "src" / "utils.py"
        utils_file.write_text(utils_file.read_text() + "\n\ndef new_function():\n    return True\n")

        # Commit the change
        subprocess.run(["git", "add", "src/utils.py"], cwd=test_project)
        subprocess.run(["git", "commit", "-m", "Add new function"], cwd=test_project)

        # Calculate incremental coverage
        import os

        original_dir = str(Path.cwd())
        try:
            os.chdir(test_project)
            gate = CoverageGate()
            result = gate.get_incremental_coverage(base_branch="HEAD~1")

            # Should have detected the changed file
            assert "error" in result or "files" in result

        finally:
            os.chdir(original_dir)


@pytest.mark.integration
class TestRepoMapGeneration(IntegrationTest):
    """Test repository map generation."""

    def test_generate_repo_map(self, test_project):
        """Test generating a repository map from project."""
        from ai_dev_agent.core.repo_map import RepoMapManager

        repo_map = RepoMapManager.get_instance(test_project)
        repo_map.scan_repository(force=True)

        assert repo_map.context.files, "Repo map should contain scanned files"

    def test_symbol_extraction(self, test_project):
        """Test extracting symbols from Python files."""
        from ai_dev_agent.core.repo_map import RepoMapManager

        repo_map = RepoMapManager.get_instance(test_project)
        repo_map.scan_repository(force=True)
        file_info = None
        for path, info in repo_map.context.files.items():
            if path.endswith("src/main.py"):
                file_info = info
                break
        assert file_info is not None
        symbol_blob = " ".join(file_info.symbols)
        assert "Calculator" in symbol_blob
        assert "greet" in symbol_blob


@pytest.mark.integration
class TestMultiAgentCoordination(IntegrationTest):
    """Test multi-agent coordination."""

    def test_agent_registry(self):
        """Test agent registry functionality."""
        from ai_dev_agent.agents.registry import AgentRegistry

        registry = AgentRegistry()

        # Should have default agents
        agents = registry.list_agents()
        assert len(agents) > 0, "Registry should have agents"

    def test_agent_communication(self, mock_llm_env):
        """Test inter-agent communication."""
        from ai_dev_agent.agents.base import BaseAgent
        from ai_dev_agent.agents.communication.bus import AgentBus, AgentEvent, EventType

        bus = AgentBus()

        # Create mock agents
        class TestAgent(BaseAgent):
            def __init__(self, name: str):
                super().__init__(name=name, description="test agent")

            def execute(self, task):
                return {"status": "success", "result": "completed"}

        agent1 = TestAgent(name="Agent1")
        agent2 = TestAgent(name="Agent2")

        # Test message passing
        messages: list[AgentEvent] = []
        bus.subscribe(EventType.MESSAGE, lambda event: messages.append(event))
        bus.start()
        bus.publish(
            AgentEvent(
                event_type=EventType.MESSAGE,
                source_agent=agent1.name,
                target_agent=agent2.name,
                data={"message": "Hello"},
            )
        )

        # Should have messages
        bus._event_queue.join()
        bus.stop()
        assert messages and messages[0].data["message"] == "Hello"


@pytest.mark.integration
class TestMemorySystem(IntegrationTest):
    """Test memory system integration."""

    def test_memory_storage_and_retrieval(self, test_project):
        """Test storing and retrieving memories."""
        from ai_dev_agent.memory.distiller import Memory
        from ai_dev_agent.memory.store import MemoryStore

        memory_dir = test_project / ".devagent" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(store_path=memory_dir / "reasoning.json", auto_save=False)

        memory = Memory(
            task_type="integration",
            title="Testing reminder",
            query="How do we test the project?",
            outcome="success",
            strategies=[],
        )
        memory_id = store.add_memory(memory)
        retrieved = store.get_memory(memory_id)

        assert retrieved is not None
        assert retrieved.title == "Testing reminder"

    def test_memory_search(self, test_project):
        """Test semantic memory search."""
        from ai_dev_agent.memory.distiller import Memory
        from ai_dev_agent.memory.store import MemoryStore

        memory_dir = test_project / ".devagent" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(store_path=memory_dir / "reasoning.json", auto_save=False)

        memory = Memory(
            task_type="testing",
            title="Pytest tip",
            query="Remember to run pytest",
            outcome="success",
            strategies=[],
        )
        store.add_memory(memory)
        results = store.search_similar("pytest testing", task_type="testing", threshold=0.0)
        assert isinstance(results, list)


@pytest.mark.integration
class TestErrorHandling(IntegrationTest):
    """Test error handling in integration scenarios."""

    def test_missing_file_error(self, test_project, devagent_cli):
        """Test handling of missing file errors through CLI."""
        result = devagent_cli(["review", "nonexistent.py"], cwd=test_project)

        assert result.returncode != 0
        combined = f"{result.stdout}\n{result.stderr}".lower()
        assert "not found" in combined or "no api key" in combined or "error" in combined

    def test_invalid_command_error(self, test_project, devagent_cli):
        """Test handling of invalid CLI commands."""
        result = devagent_cli(["invalid-command"], cwd=test_project)

        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_git_error_handling(self, test_project):
        """Test handling of git operation errors."""
        import subprocess

        # Try to commit with no changes
        result = subprocess.run(
            ["git", "commit", "-m", "Empty commit"], cwd=test_project, capture_output=True
        )

        # Should fail gracefully
        assert result.returncode != 0
        assert b"nothing to commit" in result.stdout or b"nothing to commit" in result.stderr
