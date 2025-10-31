"""Benchmark test cases for DevAgent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class TestCase:
    """A single benchmark test case."""

    __test__ = False

    name: str
    query: str
    validator: Callable[[str], bool]
    description: str = ""
    timeout: int = 120
    max_iterations: int = 20

    def validate_answer(self, answer: str) -> bool:
        """Validate if the answer is correct."""
        return self.validator(answer)


# Validation helpers
def _validate_line_count(answer: str) -> bool:
    """Validate that answer contains correct line count for runtime/main.py."""
    numbers = re.findall(r"\b(\d+)\b", answer)
    return any(200 <= int(n) <= 210 for n in numbers)


def _validate_python_files_count(answer: str) -> bool:
    """Validate count of Python files in the project."""
    numbers = re.findall(r"\b(\d+)\b", answer)
    return any(135 <= int(n) <= 155 for n in numbers)


def _validate_contains_keyword(keyword: str):
    """Create a validator that checks if answer contains a keyword."""

    def validator(answer: str) -> bool:
        return keyword.lower() in answer.lower()

    return validator


def _validate_number_range(min_val: int, max_val: int):
    """Create validator for numeric range."""

    def validator(answer: str) -> bool:
        numbers = re.findall(r"\b(\d+)\b", answer)
        return any(min_val <= int(n) <= max_val for n in numbers)

    return validator


def _validate_contains_all(keywords: list[str]):
    """Create validator that checks for ALL keywords."""

    def validator(answer: str) -> bool:
        lower_answer = answer.lower()
        return all(kw.lower() in lower_answer for kw in keywords)

    return validator


def _validate_contains_any(keywords: list[str]):
    """Create validator that checks for ANY keyword."""

    def validator(answer: str) -> bool:
        lower_answer = answer.lower()
        return any(kw.lower() in lower_answer for kw in keywords)

    return validator


# All test cases
TEST_CASES = [
    # Basic tests
    TestCase(
        name="line_count_runtime_main",
        query="how many lines are in ai_dev_agent/cli/runtime/main.py",
        validator=_validate_line_count,
        description="Count lines in ai_dev_agent/cli/runtime/main.py",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="python_files_count",
        query="how many python files in the project",
        validator=_validate_python_files_count,
        description="Count all .py files in the repository",
        timeout=90,
        max_iterations=15,
    ),
    TestCase(
        name="find_main_entry_point",
        query="what is the main entry point for the CLI",
        validator=_validate_contains_keyword("runtime/main.py"),
        description="Identify the CLI entry point",
        timeout=90,
        max_iterations=15,
    ),
    # Extended tests
    TestCase(
        name="find_all_test_files",
        query="how many test files are in the tests directory?",
        validator=_validate_number_range(35, 50),
        description="Test directory navigation and file counting",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="explain_configuration",
        query="what provider is configured in .devagent.toml?",
        validator=_validate_contains_any(["openrouter", "anthropic", "openai"]),
        description="Test config file reading and parsing",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="check_git_status",
        query="what branch am I on?",
        validator=_validate_contains_keyword("main"),
        description="Test git command execution",
        timeout=30,
        max_iterations=5,
    ),
    TestCase(
        name="find_function_definition",
        query="where is the function '_execute_react_assistant' defined?",
        validator=_validate_contains_all(["executor.py", "react"]),
        description="Test semantic search and code navigation",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="identify_test_framework",
        query="what testing framework does this project use?",
        validator=_validate_contains_keyword("pytest"),
        description="Test framework identification",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="count_function_definitions",
        query="how many functions are defined in benchmark_runner.py?",
        validator=_validate_number_range(8, 15),
        description="Test AST parsing and function counting",
        timeout=90,
        max_iterations=15,
    ),
    TestCase(
        name="find_dependencies",
        query="what are the main dependencies in pyproject.toml?",
        validator=_validate_contains_any(["click", "anthropic", "openai", "httpx"]),
        description="Test dependency analysis",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="find_todos",
        query="how many TODO comments are in the codebase?",
        validator=_validate_number_range(0, 100),
        description="Test code search for patterns",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="quick_file_count",
        query="how many files total in the project?",
        validator=_validate_number_range(150, 300),
        description="Test efficient counting",
        timeout=30,
        max_iterations=5,
    ),
    TestCase(
        name="find_imports_of_module",
        query="which files import the ToolContext class?",
        validator=_validate_contains_any(["tool_invoker", "registry"]),
        description="Test import analysis and search",
        timeout=90,
        max_iterations=15,
    ),
    TestCase(
        name="find_symbol_definitions",
        query="where is the ProviderConfig class defined?",
        validator=_validate_contains_keyword("benchmark_runner.py"),
        description="Test symbol lookup",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="identify_project_type",
        query="what type of project is this? (library, CLI tool, web app, etc.)",
        validator=_validate_contains_any(["cli", "command line", "tool", "agent"]),
        description="Test high-level project understanding",
        timeout=90,
        max_iterations=15,
    ),
    TestCase(
        name="find_documentation",
        query="where is the user documentation located?",
        validator=_validate_contains_any(["readme", "docs", ".md", "markdown"]),
        description="Test documentation discovery",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="check_code_style",
        query="does this project use a code formatter? which one?",
        validator=_validate_contains_any(["black", "ruff", "format", "style"]),
        description="Test tooling discovery",
        timeout=60,
        max_iterations=10,
    ),
    TestCase(
        name="count_recent_commits",
        query="how many commits in the last 7 days?",
        validator=_validate_number_range(0, 100),
        description="Test git log parsing",
        timeout=60,
        max_iterations=10,
    ),
]


__all__ = ["TEST_CASES", "TestCase"]
