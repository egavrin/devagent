"""Shared test fixtures for DevAgent testing.

This module provides reusable fixtures for testing various components
of the DevAgent system.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory for testing.

    Yields:
        Path to temporary directory that is cleaned up after test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Create basic project structure
        (workspace / "src").mkdir()
        (workspace / "tests").mkdir()
        (workspace / "docs").mkdir()
        (workspace / ".devagent").mkdir()

        # Create sample files
        (workspace / "README.md").write_text("# Test Project\n")
        (workspace / "pyproject.toml").write_text("[tool.poetry]\nname = 'test-project'\n")
        (workspace / "src" / "__init__.py").touch()
        (workspace / "src" / "main.py").write_text("def hello():\n    return 'Hello, World!'\n")

        yield workspace


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without API calls.

    Returns:
        MagicMock configured to simulate LLM responses.
    """
    mock = MagicMock()
    mock.complete.return_value = {
        "content": "This is a mock LLM response",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "model": "mock-model",
    }
    mock.stream_complete.return_value = iter(
        [{"delta": "This "}, {"delta": "is "}, {"delta": "streaming"}]
    )
    return mock


@pytest.fixture
def sample_repo_map() -> dict:
    """Provide a sample repository map for testing.

    Returns:
        Dictionary representing a repository structure.
    """
    return {
        "files": [
            {
                "path": "src/main.py",
                "language": "python",
                "size": 1024,
                "symbols": [
                    {"name": "hello", "type": "function", "line": 1},
                    {"name": "Calculator", "type": "class", "line": 10},
                ],
            },
            {
                "path": "src/utils.py",
                "language": "python",
                "size": 512,
                "symbols": [
                    {"name": "format_date", "type": "function", "line": 3},
                    {"name": "parse_config", "type": "function", "line": 15},
                ],
            },
            {
                "path": "tests/test_main.py",
                "language": "python",
                "size": 2048,
                "symbols": [
                    {"name": "test_hello", "type": "function", "line": 5},
                    {"name": "TestCalculator", "type": "class", "line": 12},
                ],
            },
        ],
        "summary": {"total_files": 3, "total_symbols": 6, "languages": {"python": 3}},
    }


@pytest.fixture
def mock_git_repo(temp_workspace):
    """Create a mock git repository for testing.

    Args:
        temp_workspace: Temporary workspace fixture

    Returns:
        Path to git repository
    """
    import subprocess

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=temp_workspace, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_workspace)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_workspace)

    # Create initial commit
    subprocess.run(["git", "add", "."], cwd=temp_workspace)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_workspace)

    return temp_workspace


@pytest.fixture
def sample_work_plan() -> dict:
    """Provide a sample work plan for testing.

    Returns:
        Dictionary representing a work plan.
    """
    return {
        "id": "test-plan-123",
        "goal": "Implement new feature",
        "context": "Add user authentication",
        "tasks": [
            {
                "id": "task-1",
                "description": "Create user model",
                "priority": "high",
                "status": "pending",
                "dependencies": [],
            },
            {
                "id": "task-2",
                "description": "Implement login endpoint",
                "priority": "high",
                "status": "pending",
                "dependencies": ["task-1"],
            },
            {
                "id": "task-3",
                "description": "Add authentication middleware",
                "priority": "medium",
                "status": "pending",
                "dependencies": ["task-2"],
            },
        ],
        "created_at": "2025-01-15T10:00:00Z",
        "status": "active",
    }


@pytest.fixture
def mock_session_state() -> dict:
    """Provide mock session state for testing.

    Returns:
        Dictionary representing session state.
    """
    return {
        "session_id": "test-session-123",
        "start_time": "2025-01-15T10:00:00Z",
        "context": {
            "working_directory": "/test/project",
            "active_files": ["src/main.py", "tests/test_main.py"],
            "recent_commands": ["pytest", "git status"],
        },
        "memory": {
            "facts": ["Project uses Python 3.11", "Testing framework is pytest"],
            "instructions": [],
        },
        "metrics": {"queries_processed": 5, "files_modified": 2, "tests_run": 10},
    }


@pytest.fixture
def mock_agent_registry():
    """Mock agent registry for multi-agent testing.

    Returns:
        MagicMock configured as agent registry.
    """
    registry = MagicMock()
    registry.list_agents.return_value = [
        {"name": "DesignAgent", "type": "design", "status": "ready"},
        {"name": "TestAgent", "type": "test", "status": "ready"},
        {"name": "ImplementationAgent", "type": "implementation", "status": "ready"},
    ]
    registry.get_agent.return_value = MagicMock(
        execute=MagicMock(return_value={"status": "success", "result": "Task completed"})
    )
    return registry


@pytest.fixture
def sample_code_file() -> str:
    """Provide sample Python code for testing.

    Returns:
        String containing Python code.
    """
    return '''"""Sample module for testing."""

import os
from typing import List, Optional


class Calculator:
    """A simple calculator class."""

    def __init__(self, precision: int = 2):
        """Initialize calculator with precision."""
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        return round(result, self.precision)

    def divide(self, a: float, b: float) -> Optional[float]:
        """Divide two numbers, handling division by zero."""
        if b == 0:
            return None
        result = a / b
        return round(result, self.precision)


def process_data(items: List[str]) -> dict:
    """Process a list of items into a summary."""
    summary = {
        "count": len(items),
        "unique": len(set(items)),
        "items": items[:10]  # First 10 items
    }
    return summary


def main():
    """Main entry point."""
    calc = Calculator()
    result = calc.add(10.5, 20.3)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
'''


@pytest.fixture
def mock_file_system(temp_workspace):
    """Mock file system operations for testing.

    Args:
        temp_workspace: Temporary workspace fixture

    Returns:
        Dictionary with mock file operations.
    """

    def read_file(path: str) -> str:
        file_path = temp_workspace / path
        if file_path.exists():
            return file_path.read_text()
        raise FileNotFoundError(f"File not found: {path}")

    def write_file(path: str, content: str) -> None:
        file_path = temp_workspace / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    def list_files(pattern: str = "**/*") -> list[str]:
        return [
            str(p.relative_to(temp_workspace)) for p in temp_workspace.glob(pattern) if p.is_file()
        ]

    return {"read": read_file, "write": write_file, "list": list_files, "workspace": temp_workspace}


@pytest.fixture
def mock_config():
    """Provide mock configuration for testing.

    Returns:
        Dictionary with test configuration.
    """
    return {
        "llm": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 2000},
        "testing": {"coverage_threshold": 95.0, "parallel": True, "timeout": 60},
        "paths": {"workspace": "/test/workspace", "cache": "/test/.cache", "logs": "/test/logs"},
        "features": {"auto_test": True, "memory_system": True, "multi_agent": True},
    }


# Parametrized fixtures for different test scenarios
@pytest.fixture(params=["small", "medium", "large"])
def project_size(request):
    """Parametrized fixture for different project sizes.

    Args:
        request: Pytest request object

    Returns:
        Dictionary with project size configuration.
    """
    sizes = {
        "small": {"files": 10, "lines": 1000, "tests": 20},
        "medium": {"files": 100, "lines": 10000, "tests": 200},
        "large": {"files": 1000, "lines": 100000, "tests": 2000},
    }
    return sizes[request.param]


@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance testing.

    Returns:
        Timer class for measuring execution time.
    """
    import time

    class Timer:
        def __init__(self):
            self.times = []

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *exc):
            elapsed = time.perf_counter() - self.start
            self.times.append(elapsed)

        @property
        def elapsed(self):
            return self.times[-1] if self.times else 0

        @property
        def average(self):
            return sum(self.times) / len(self.times) if self.times else 0

    return Timer()
