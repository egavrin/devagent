"""Pytest configuration for integration tests.

This module provides fixtures and configuration specific to integration testing.
"""

import os
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Optional

import pytest


@pytest.fixture(scope="session")
def integration_test_dir() -> Generator[Path, None, None]:
    """Create a session-scoped test directory for integration tests.

    This directory persists across all integration tests in a session
    but is cleaned up at the end.

    Yields:
        Path to integration test directory
    """
    with tempfile.TemporaryDirectory(prefix="devagent_integration_") as tmpdir:
        test_dir = Path(tmpdir)
        yield test_dir


@pytest.fixture
def test_project(integration_test_dir) -> Generator[Path, None, None]:
    """Create a complete test project for integration testing.

    Yields:
        Path to test project root
    """
    project_dir = integration_test_dir / f"project_{os.getpid()}"
    project_dir.mkdir(exist_ok=True)

    # Create project structure
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()
    (project_dir / "docs").mkdir()
    (project_dir / ".devagent").mkdir()

    # Create project files
    (project_dir / "README.md").write_text(
        """# Integration Test Project

This is a test project for DevAgent integration testing.
"""
    )

    (project_dir / "pyproject.toml").write_text(
        """[tool.poetry]
name = "integration-test-project"
version = "0.1.0"
description = "Test project for integration testing"

[tool.poetry.dependencies]
python = "^3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
    )

    (project_dir / "src" / "__init__.py").touch()
    (project_dir / "src" / "main.py").write_text(
        """\"\"\"Main module for test project.\"\"\"


def greet(name: str) -> str:
    \"\"\"Return a greeting message.\"\"\"
    return f"Hello, {name}!"


def calculate(a: int, b: int) -> int:
    \"\"\"Calculate sum of two numbers.\"\"\"
    return a + b


class Calculator:
    \"\"\"A simple calculator class.\"\"\"

    def __init__(self):
        self.history = []

    def add(self, a: float, b: float) -> float:
        \"\"\"Add two numbers.\"\"\"
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        \"\"\"Multiply two numbers.\"\"\"
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
"""
    )

    (project_dir / "src" / "utils.py").write_text(
        """\"\"\"Utility functions.\"\"\"

from typing import List, Dict, Any
import json


def parse_json(text: str) -> Dict[str, Any]:
    \"\"\"Parse JSON string to dictionary.\"\"\"
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": str(e)}


def format_list(items: List[str]) -> str:
    \"\"\"Format list as bullet points.\"\"\"
    if not items:
        return "No items"
    return "\\n".join(f"- {item}" for item in items)


def validate_email(email: str) -> bool:
    \"\"\"Simple email validation.\"\"\"
    return "@" in email and "." in email.split("@")[1]
"""
    )

    (project_dir / "tests" / "test_main.py").write_text(
        """\"\"\"Tests for main module.\"\"\"

import pytest
from src.main import greet, calculate, Calculator


def test_greet():
    \"\"\"Test greet function.\"\"\"
    assert greet("World") == "Hello, World!"
    assert greet("Alice") == "Hello, Alice!"


def test_calculate():
    \"\"\"Test calculate function.\"\"\"
    assert calculate(2, 3) == 5
    assert calculate(-1, 1) == 0
    assert calculate(0, 0) == 0


class TestCalculator:
    \"\"\"Test Calculator class.\"\"\"

    def test_add(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5
        assert calc.add(-1, 1) == 0

    def test_multiply(self):
        calc = Calculator()
        assert calc.multiply(2, 3) == 6
        assert calc.multiply(-2, 3) == -6

    def test_history(self):
        calc = Calculator()
        calc.add(1, 2)
        calc.multiply(3, 4)
        assert len(calc.history) == 2
        assert "1 + 2 = 3" in calc.history
        assert "3 * 4 = 12" in calc.history
"""
    )

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=project_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_dir)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=project_dir)
    subprocess.run(["git", "add", "."], cwd=project_dir)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=project_dir)

    yield project_dir

    # Cleanup is handled by the context manager


@pytest.fixture
def devagent_cli():
    """Fixture for testing DevAgent CLI commands.

    Returns:
        Function to run CLI commands
    """

    def run_command(args: list, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a DevAgent CLI command.

        Args:
            args: Command arguments
            cwd: Working directory

        Returns:
            Completed process result
        """
        cmd = ["python", "-m", "ai_dev_agent.cli.main", *args]
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        return result

    return run_command


@pytest.fixture
def mock_llm_env(monkeypatch):
    """Mock LLM environment variables for testing.

    This prevents actual API calls during integration tests.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-456")
    monkeypatch.setenv("DEVAGENT_TEST_MODE", "true")
    monkeypatch.setenv("DEVAGENT_MOCK_LLM", "true")


@pytest.mark.integration
class IntegrationTest:
    """Base class for integration tests.

    Provides common functionality for integration testing.
    """

    @staticmethod
    def assert_command_success(result: subprocess.CompletedProcess):
        """Assert that a command executed successfully.

        Args:
            result: Command result

        Raises:
            AssertionError: If command failed
        """
        assert result.returncode == 0, f"Command failed: {result.stderr}"

    @staticmethod
    def assert_file_contains(file_path: Path, content: str):
        """Assert that a file contains specific content.

        Args:
            file_path: Path to file
            content: Expected content

        Raises:
            AssertionError: If content not found
        """
        actual_content = file_path.read_text()
        assert content in actual_content, f"Content '{content}' not found in {file_path}"

    @staticmethod
    def wait_for_file(file_path: Path, timeout: float = 5.0) -> bool:
        """Wait for a file to be created.

        Args:
            file_path: Path to file
            timeout: Maximum wait time

        Returns:
            True if file exists, False if timeout
        """
        import time

        start = time.time()
        while time.time() - start < timeout:
            if file_path.exists():
                return True
            time.sleep(0.1)
        return False
