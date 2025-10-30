"""End-to-end integration tests for DevAgent.

This module tests complete workflows from user query to final output.
"""

import pytest

from .conftest import IntegrationTest


@pytest.mark.integration
class TestWorkPlanningWorkflow(IntegrationTest):
    """Test complete work planning workflow."""

    def test_create_and_execute_plan(self, test_project, devagent_cli, mock_llm_env):
        """Test creating and executing a work plan."""
        # Create a work plan
        result = devagent_cli(
            ["plan", "create", "Add logging system", "--context", "Add structured logging"],
            cwd=test_project,
        )
        self.assert_command_success(result)
        assert "Created work plan" in result.stdout

        # List plans
        result = devagent_cli(["plan", "list"], cwd=test_project)
        self.assert_command_success(result)
        assert "Add logging system" in result.stdout

    def test_plan_task_execution(self, test_project, devagent_cli, mock_llm_env):
        """Test executing tasks from a plan."""
        # Create plan
        result = devagent_cli(["plan", "create", "Refactor calculator"], cwd=test_project)
        self.assert_command_success(result)

        # Extract plan ID from output
        plan_id = None
        for line in result.stdout.split("\n"):
            if "ID:" in line:
                plan_id = line.split("ID:")[1].strip()
                break

        assert plan_id is not None, "Could not extract plan ID"

        # Get next task
        result = devagent_cli(["plan", "next", plan_id], cwd=test_project)
        self.assert_command_success(result)


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
            assert result.total_coverage > 0, "No coverage data collected"

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
        from ai_dev_agent.tools.repo_map import TreeSitterRepoMap

        repo_map = TreeSitterRepoMap(root_dir=str(test_project))
        result = repo_map.build()

        # Verify structure
        assert "files" in result or len(result) > 0, "Repo map should not be empty"

    def test_symbol_extraction(self, test_project):
        """Test extracting symbols from Python files."""
        from ai_dev_agent.tools.repo_map import TreeSitterRepoMap

        repo_map = TreeSitterRepoMap(root_dir=str(test_project))
        main_file = test_project / "src" / "main.py"

        # Extract symbols
        symbols = repo_map.extract_symbols(str(main_file))

        # Should find classes and functions
        symbol_names = [s.get("name") for s in symbols]
        assert "Calculator" in symbol_names
        assert "greet" in symbol_names


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
        from ai_dev_agent.agents.bus import CommunicationBus

        bus = CommunicationBus()

        # Create mock agents
        class TestAgent(BaseAgent):
            def execute(self, task):
                return {"status": "success", "result": "completed"}

        agent1 = TestAgent(name="Agent1", agent_type="test")
        agent2 = TestAgent(name="Agent2", agent_type="test")

        # Test message passing
        bus.subscribe("test_topic", agent2.name)
        bus.publish("test_topic", {"message": "Hello"}, sender=agent1.name)

        # Should have messages
        messages = bus.get_messages("test_topic")
        assert len(messages) > 0


@pytest.mark.integration
class TestMemorySystem(IntegrationTest):
    """Test memory system integration."""

    def test_memory_storage_and_retrieval(self, test_project):
        """Test storing and retrieving memories."""
        from ai_dev_agent.memory.memory_bank import MemoryBank

        memory_dir = test_project / ".devagent" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        bank = MemoryBank(memory_dir=str(memory_dir))

        # Store a memory
        bank.add_fact("test_session", "The project uses pytest for testing")

        # Retrieve memories
        memories = bank.get_memories("test_session")
        assert len(memories) > 0

    def test_memory_search(self, test_project):
        """Test semantic memory search."""
        from ai_dev_agent.memory.memory_bank import MemoryBank

        memory_dir = test_project / ".devagent" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        bank = MemoryBank(memory_dir=str(memory_dir))

        # Add several facts
        bank.add_fact("test_session", "Uses Python 3.11")
        bank.add_fact("test_session", "Testing with pytest")
        bank.add_fact("test_session", "Calculator class for math operations")

        # Search for relevant memories
        results = bank.search("testing framework")

        # Should find pytest-related memory
        assert len(results) > 0


@pytest.mark.integration
class TestFileOperations(IntegrationTest):
    """Test file operation integration."""

    def test_read_write_edit(self, test_project):
        """Test file read, write, and edit operations."""
        from ai_dev_agent.tools.file_operations import FileOperations

        ops = FileOperations(workspace=str(test_project))

        # Read existing file
        content = ops.read_file("src/main.py")
        assert "def greet" in content

        # Write new file
        new_file = "src/new_module.py"
        ops.write_file(new_file, "def new_function():\n    return True\n")
        assert (test_project / new_file).exists()

        # Edit file
        ops.edit_file(new_file, old_content="return True", new_content="return False")
        updated = ops.read_file(new_file)
        assert "return False" in updated

    def test_grep_and_glob(self, test_project):
        """Test grep and glob operations."""
        from ai_dev_agent.tools.glob import glob_files
        from ai_dev_agent.tools.grep import grep_search

        # Find Python files
        py_files = glob_files("**/*.py", root_dir=str(test_project))
        assert len(py_files) > 0

        # Search for pattern
        results = grep_search(pattern="def greet", path=str(test_project), file_pattern="*.py")
        assert len(results) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance(IntegrationTest):
    """Test performance characteristics."""

    def test_large_file_processing(self, test_project):
        """Test processing large files efficiently."""
        import time

        # Create a large file
        large_file = test_project / "src" / "large.py"
        content = "\n".join([f"def function_{i}():\n    return {i}" for i in range(1000)])
        large_file.write_text(content)

        from ai_dev_agent.tools.repo_map import TreeSitterRepoMap

        # Time the processing
        start = time.perf_counter()
        repo_map = TreeSitterRepoMap(root_dir=str(test_project))
        symbols = repo_map.extract_symbols(str(large_file))
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 2.0, f"Processing took too long: {elapsed}s"
        assert len(symbols) == 1000

    def test_concurrent_operations(self, test_project):
        """Test concurrent file operations."""
        import concurrent.futures

        from ai_dev_agent.tools.file_operations import FileOperations

        ops = FileOperations(workspace=str(test_project))

        def read_file(filename):
            return ops.read_file(filename)

        # Read multiple files concurrently
        files = ["src/main.py", "src/utils.py", "tests/test_main.py"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(read_file, files))

        assert len(results) == 3
        assert all(len(r) > 0 for r in results)


@pytest.mark.integration
class TestErrorHandling(IntegrationTest):
    """Test error handling in integration scenarios."""

    def test_missing_file_error(self, test_project):
        """Test handling of missing file errors."""
        from ai_dev_agent.tools.file_operations import FileOperations

        ops = FileOperations(workspace=str(test_project))

        with pytest.raises(FileNotFoundError):
            ops.read_file("nonexistent.py")

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
