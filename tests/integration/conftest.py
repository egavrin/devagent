"""Pytest configuration for integration tests.

This module provides fixtures and configuration specific to integration testing.
"""

import os
import subprocess
import sys
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
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "tests").mkdir(exist_ok=True)
    (project_dir / "docs").mkdir(exist_ok=True)
    (project_dir / ".devagent").mkdir(exist_ok=True)

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
        cmd = [sys.executable, "-m", "ai_dev_agent.cli", *args]
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


@pytest.fixture
def llm_client_real():
    """Fixture for real LLM testing with API keys.

    Checks for API keys in environment or .devagent.toml.
    Skips tests if no API key is available.

    Returns:
        bool: True if LLM client is available for testing
    """
    # Check environment variables
    api_key = os.environ.get("DEVAGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        # Try to load from .devagent.toml
        config_path = Path.cwd() / ".devagent.toml"
        if config_path.exists():
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib

            try:
                with open(config_path, "rb") as f:
                    config = tomllib.load(f)
                    api_key = config.get("api_key")
            except Exception:
                pass

    if not api_key:
        pytest.skip(
            "No API key available for LLM testing (set DEVAGENT_API_KEY or configure .devagent.toml)"
        )

    return True


@pytest.fixture
def test_devagent_config(tmp_path):
    """Create a temporary .devagent.toml config for testing.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to the created config file
    """
    # Get API key from environment or existing config
    api_key = os.environ.get("DEVAGENT_API_KEY")

    if not api_key:
        # Try to load from project .devagent.toml
        project_config = Path.cwd() / ".devagent.toml"
        if project_config.exists():
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib

            try:
                with open(project_config, "rb") as f:
                    config_data = tomllib.load(f)
                    api_key = config_data.get("api_key")
            except Exception:
                pass

    if not api_key:
        api_key = "test-key-placeholder"

    config_content = f"""
provider = "deepseek"
model = "deepseek-chat"
api_key = "{api_key}"
base_url = "https://api.deepseek.com/v1"
max_completion_tokens = 4096
auto_approve_code = true
"""

    config_path = tmp_path / ".devagent.toml"
    config_path.write_text(config_content)

    return config_path


@pytest.fixture
def run_devagent_cli():
    """Enhanced fixture for running DevAgent CLI commands.

    Returns:
        Function to run CLI commands with config support
    """

    def run_command(
        args: list, cwd: Optional[Path] = None, timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """Run a DevAgent CLI command.

        Args:
            args: Command arguments
            cwd: Working directory (defaults to current)
            timeout: Command timeout in seconds

        Returns:
            Completed process result
        """
        cmd = ["devagent", *args]
        if cwd is None:
            cwd = Path.cwd()

        # Ensure API key is available in environment
        env = os.environ.copy()
        if "DEVAGENT_API_KEY" not in env:
            # Try to load from project .devagent.toml (use original cwd, not the test cwd)
            # This ensures we find the project config even when running from tmp_path
            original_cwd = Path(__file__).parent.parent.parent  # Go to project root
            project_config = original_cwd / ".devagent.toml"
            if project_config.exists():
                try:
                    import tomllib
                except ModuleNotFoundError:
                    import tomli as tomllib

                try:
                    with open(project_config, "rb") as f:
                        config_data = tomllib.load(f)
                        api_key = config_data.get("api_key")
                        if api_key:
                            env["DEVAGENT_API_KEY"] = api_key
                except Exception:
                    pass

        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env
        )
        return result

    return run_command


@pytest.fixture
def sample_patch():
    """Create a sample patch for review testing.

    Returns:
        str: Sample patch content with style violations
    """
    return """diff --git a/src/example.py b/src/example.py
index abc123..def456 100644
--- a/src/example.py
+++ b/src/example.py
@@ -1,5 +1,8 @@
 def calculate(x, y):
-    return x + y
+    # TODO: fix this
+    result=x+y  # Missing spaces around operator
+    return result

 def process_data(data):
+    # FIXME: hardcoded value
+    threshold = 100
     return [x for x in data if x > threshold]
"""


@pytest.fixture
def coding_rule():
    """Create a sample coding rule for review testing.

    Returns:
        str: Sample coding rule content
    """
    return """# Python Style Guide

## Applies To
*.py

## Description
Enforce Python code style best practices.

## Rules

1. **No TODO/FIXME comments**: Remove TODO and FIXME comments before commit
2. **Proper spacing**: Use spaces around operators (PEP 8)
3. **No hardcoded values**: Avoid magic numbers

## Examples

### Violation
```python
result=x+y  # Missing spaces
# TODO: fix this
threshold = 100  # Magic number
```

### Compliant
```python
result = x + y
# Properly documented
THRESHOLD_LIMIT = 100  # Named constant
```
"""


def verify_line_count(file_path: Path) -> int:
    """Verify line count in a file.

    Args:
        file_path: Path to file

    Returns:
        Number of lines in file
    """
    if not file_path.exists():
        return 0

    with open(file_path, "r") as f:
        return len(f.readlines())


def verify_file_list(directory: Path, pattern: str = "*") -> list[str]:
    """Verify list of files in a directory.

    Args:
        directory: Path to directory
        pattern: Glob pattern for matching files

    Returns:
        List of file names
    """
    if not directory.exists():
        return []

    return sorted([f.name for f in directory.glob(pattern) if f.is_file()])


def verify_json_schema(data: dict, required_keys: list[str]) -> bool:
    """Verify JSON data has required keys.

    Args:
        data: JSON data dictionary
        required_keys: List of required keys

    Returns:
        True if all keys present
    """
    return all(key in data for key in required_keys)


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
