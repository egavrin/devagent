"""Benchmarking system for context engineering validation.

This module provides tools to measure DevAgent's performance improvements
from context engineering features (memory, playbook, dynamic instructions).
"""

from .framework import (
    BenchmarkFramework,
    BenchmarkTask,
    BenchmarkRun,
    TaskResult,
    TaskCategory,
    TaskDifficulty,
    ContextMode
)

from .task_suite import (
    get_standard_task_suite,
    get_task_suite_by_category,
    get_task_suite_by_difficulty,
    get_quick_test_suite
)

__all__ = [
    # Framework
    "BenchmarkFramework",
    "BenchmarkTask",
    "BenchmarkRun",
    "TaskResult",
    "TaskCategory",
    "TaskDifficulty",
    "ContextMode",

    # Task Suite
    "get_standard_task_suite",
    "get_task_suite_by_category",
    "get_task_suite_by_difficulty",
    "get_quick_test_suite",
]
