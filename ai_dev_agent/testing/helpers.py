"""Helper functions for DevAgent testing.

This module provides utility functions to simplify common testing tasks.
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import MagicMock


def run_with_timeout(func, timeout: float = 5.0, *args, **kwargs) -> tuple[bool, Any]:
    """Run a function with timeout.

    Args:
        func: Function to run
        timeout: Timeout in seconds
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (completed, result)
    """
    import threading

    result = [None]
    exception = [None]
    completed = [False]

    def target():
        try:
            result[0] = func(*args, **kwargs)
            completed[0] = True
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if exception[0]:
        raise exception[0]

    return completed[0], result[0]


def assert_files_equal(file1: Path, file2: Path, ignore_whitespace: bool = False) -> None:
    """Assert two files have identical content.

    Args:
        file1: First file path
        file2: Second file path
        ignore_whitespace: Ignore whitespace differences

    Raises:
        AssertionError: If files differ
    """
    content1 = file1.read_text()
    content2 = file2.read_text()

    if ignore_whitespace:
        content1 = " ".join(content1.split())
        content2 = " ".join(content2.split())

    assert content1 == content2, f"Files {file1} and {file2} have different content"


def create_test_project(root_dir: Path, structure: dict[str, Union[str, dict]]) -> None:
    """Create a test project with given structure.

    Args:
        root_dir: Root directory for project
        structure: Dictionary defining project structure
                  Keys are file/dir names, values are content or nested dicts

    Example:
        create_test_project(Path("/tmp/test"), {
            "src": {
                "main.py": "print('hello')",
                "__init__.py": ""
            },
            "README.md": "# Test Project"
        })
    """
    for name, content in structure.items():
        path = root_dir / name

        if isinstance(content, dict):
            # It's a directory
            path.mkdir(parents=True, exist_ok=True)
            create_test_project(path, content)
        else:
            # It's a file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


@contextmanager
def capture_logs(logger_name: Optional[str] = None) -> list[dict]:
    """Capture log messages during test.

    Args:
        logger_name: Specific logger to capture (captures all if None)

    Returns:
        List of log records
    """
    import logging

    class LogCapture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(
                {
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name,
                    "timestamp": record.created,
                }
            )

    handler = LogCapture()

    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()

    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


@contextmanager
def temporary_env(**env_vars):
    """Temporarily set environment variables.

    Args:
        **env_vars: Environment variables to set

    Example:
        with temporary_env(API_KEY="test", DEBUG="true"):
            # Code runs with temporary env vars
    """
    original = {}
    for key, value in env_vars.items():
        original[key] = os.environ.get(key)
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1) -> bool:
    """Wait for a condition to become true.

    Args:
        condition_func: Function that returns True when condition is met
        timeout: Maximum time to wait
        interval: Check interval

    Returns:
        True if condition was met, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    return False


def mock_subprocess_run(commands: dict[str, dict]) -> MagicMock:
    """Create mock for subprocess.run with predefined responses.

    Args:
        commands: Dict mapping command strings to response dicts

    Returns:
        Configured mock

    Example:
        mock = mock_subprocess_run({
            "git status": {"stdout": "On branch main", "returncode": 0},
            "pytest": {"stdout": "All tests passed", "returncode": 0}
        })
    """

    def side_effect(cmd, **kwargs):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd

        for pattern, response in commands.items():
            if pattern in cmd_str:
                result = MagicMock()
                result.stdout = response.get("stdout", "")
                result.stderr = response.get("stderr", "")
                result.returncode = response.get("returncode", 0)
                return result

        # Default response
        result = MagicMock()
        result.stdout = ""
        result.stderr = f"Command not found: {cmd_str}"
        result.returncode = 1
        return result

    mock = MagicMock(side_effect=side_effect)
    return mock


@dataclass
class TestMetrics:
    """Container for test execution metrics."""

    test_name: str
    execution_time: float
    memory_usage: float
    assertions: int
    passed: bool
    coverage: Optional[float] = None


def measure_test_performance(func) -> TestMetrics:
    """Decorator to measure test performance.

    Args:
        func: Test function to measure

    Returns:
        TestMetrics object with measurements
    """
    import time
    import tracemalloc

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start measurements
        tracemalloc.start()
        start_time = time.perf_counter()
        assertion_count = [0]

        # Monkey-patch assert to count assertions
        original_assert = __builtins__.get("assert", None)

        def counting_assert(*args, **kwargs):
            assertion_count[0] += 1
            if original_assert:
                return original_assert(*args, **kwargs)

        # Run test
        passed = False
        try:
            result = func(*args, **kwargs)
            passed = True
        except Exception:
            passed = False
            raise
        finally:
            # Stop measurements
            end_time = time.perf_counter()
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Create metrics
            metrics = TestMetrics(
                test_name=func.__name__,
                execution_time=end_time - start_time,
                memory_usage=peak / 1024 / 1024,  # Convert to MB
                assertions=assertion_count[0],
                passed=passed,
            )

            # Store metrics for retrieval
            wrapper._test_metrics.append(metrics)

        return result

    wrapper._test_metrics = []
    return wrapper


def generate_test_data(data_type: str, count: int = 10, **kwargs) -> list[Any]:
    """Generate test data of various types.

    Args:
        data_type: Type of data to generate
        count: Number of items to generate
        **kwargs: Additional parameters for data generation

    Returns:
        List of generated test data
    """
    import random
    import string
    from datetime import datetime, timedelta

    if data_type == "strings":
        length = kwargs.get("length", 10)
        return ["".join(random.choices(string.ascii_letters, k=length)) for _ in range(count)]

    elif data_type == "numbers":
        min_val = kwargs.get("min", 0)
        max_val = kwargs.get("max", 100)
        return [random.randint(min_val, max_val) for _ in range(count)]

    elif data_type == "floats":
        min_val = kwargs.get("min", 0.0)
        max_val = kwargs.get("max", 100.0)
        return [random.uniform(min_val, max_val) for _ in range(count)]

    elif data_type == "dates":
        start = datetime.now()
        return [start + timedelta(days=i) for i in range(count)]

    elif data_type == "dicts":
        keys = kwargs.get("keys", ["key1", "key2", "key3"])
        return [{k: random.randint(0, 100) for k in keys} for _ in range(count)]

    elif data_type == "files":
        return [f"file_{i}.{kwargs.get('extension', 'txt')}" for i in range(count)]

    else:
        raise ValueError(f"Unknown data type: {data_type}")


def compare_json_structures(
    json1: Union[str, dict], json2: Union[str, dict], ignore_keys: Optional[list[str]] = None
) -> bool:
    """Compare two JSON structures ignoring specific keys.

    Args:
        json1: First JSON (string or dict)
        json2: Second JSON (string or dict)
        ignore_keys: Keys to ignore in comparison

    Returns:
        True if structures are equivalent
    """
    if isinstance(json1, str):
        json1 = json.loads(json1)
    if isinstance(json2, str):
        json2 = json.loads(json2)

    ignore_keys = ignore_keys or []

    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items() if k not in ignore_keys}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        else:
            return d

    return clean_dict(json1) == clean_dict(json2)


def create_mock_response(
    status_code: int = 200, content: Any = None, headers: Optional[dict] = None
) -> MagicMock:
    """Create a mock HTTP response.

    Args:
        status_code: HTTP status code
        content: Response content
        headers: Response headers

    Returns:
        Mock response object
    """
    response = MagicMock()
    response.status_code = status_code
    response.content = content or {"message": "success"}
    response.headers = headers or {"content-type": "application/json"}
    response.json.return_value = content if isinstance(content, dict) else {}
    response.text = str(content)
    response.ok = 200 <= status_code < 300
    return response


def validate_schema(data: dict, schema: dict) -> tuple[bool, list[str]]:
    """Validate data against a simple schema.

    Args:
        data: Data to validate
        schema: Schema definition

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    def validate_field(value, field_schema, path=""):
        field_type = field_schema.get("type")
        required = field_schema.get("required", False)

        if value is None:
            if required:
                errors.append(f"{path} is required")
            return

        if field_type == "string" and not isinstance(value, str):
            errors.append(f"{path} must be a string")
        elif field_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"{path} must be a number")
        elif field_type == "boolean" and not isinstance(value, bool):
            errors.append(f"{path} must be a boolean")
        elif field_type == "array" and not isinstance(value, list):
            errors.append(f"{path} must be an array")
        elif field_type == "object" and not isinstance(value, dict):
            errors.append(f"{path} must be an object")

        # Validate nested objects
        if field_type == "object" and isinstance(value, dict):
            properties = field_schema.get("properties", {})
            for key, prop_schema in properties.items():
                validate_field(value.get(key), prop_schema, f"{path}.{key}" if path else key)

    # Validate top-level properties
    for key, field_schema in schema.items():
        validate_field(data.get(key), field_schema, key)

    return len(errors) == 0, errors


def cleanup_test_artifacts(*paths: Path) -> None:
    """Clean up test artifacts after test completion.

    Args:
        *paths: Paths to clean up
    """
    import shutil

    for path in paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


# Test data fixtures
SAMPLE_PYTHON_CODE = '''
def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of numbers."""
    return sum(numbers)


class DataProcessor:
    """Process various types of data."""

    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)

    def process(self):
        return len(self.data)
'''

SAMPLE_TEST_CODE = '''
import pytest


def test_example():
    """Example test case."""
    assert 1 + 1 == 2


class TestClass:
    """Test class with multiple test methods."""

    def test_method1(self):
        assert True

    def test_method2(self):
        assert "hello".upper() == "HELLO"
'''
