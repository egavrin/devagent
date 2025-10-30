"""Benchmarking system for context engineering validation.

This module provides tools to measure DevAgent's performance improvements
from context engineering features (memory, playbook, dynamic instructions).
"""

from .framework import (
    BenchmarkFramework,
    BenchmarkRun,
    BenchmarkTask,
    ContextMode,
    TaskCategory,
    TaskDifficulty,
    TaskResult,
)
from .task_suite import (
    get_quick_test_suite,
    get_standard_task_suite,
    get_task_suite_by_category,
    get_task_suite_by_difficulty,
)

__all__ = [
    # Framework
    "BenchmarkFramework",
    "BenchmarkRun",
    "BenchmarkTask",
    "ContextMode",
    "TaskCategory",
    "TaskDifficulty",
    "TaskResult",
    "get_quick_test_suite",
    # Task Suite
    "get_standard_task_suite",
    "get_task_suite_by_category",
    "get_task_suite_by_difficulty",
]
