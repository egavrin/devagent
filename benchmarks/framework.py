"""Benchmarking framework for context engineering validation.

This module provides tools to measure and compare DevAgent's performance
with and without context engineering features (memory, playbook, dynamic instructions).
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


class TaskCategory(str, Enum):
    """Categories for benchmark tasks."""
    DEBUGGING = "debugging"
    FEATURE_IMPLEMENTATION = "feature_implementation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    SECURITY = "security"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"
    BUG_FIX = "bug_fix"
    MIXED = "mixed"


class TaskDifficulty(str, Enum):
    """Difficulty levels for tasks."""
    EASY = "easy"           # Simple, 1-2 steps
    MEDIUM = "medium"       # Moderate, 3-5 steps
    HARD = "hard"           # Complex, 6+ steps
    EXPERT = "expert"       # Very complex, requires deep reasoning


class ContextMode(str, Enum):
    """Context engineering modes for benchmarking."""
    NONE = "none"                       # No context engineering
    REPOMAP_ONLY = "repomap_only"      # Only RepoMap (baseline)
    MEMORY_ONLY = "memory_only"        # RepoMap + Memory
    PLAYBOOK_ONLY = "playbook_only"    # RepoMap + Playbook
    MEMORY_PLAYBOOK = "memory_playbook"  # RepoMap + Memory + Playbook
    FULL = "full"                      # All features (Memory + Playbook + Dynamic)


@dataclass
class BenchmarkTask:
    """A single benchmark task."""

    task_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    category: TaskCategory = TaskCategory.MIXED
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM

    # Task definition
    prompt: str = ""                    # The task prompt
    expected_outcome: str = ""          # What success looks like
    validation_fn: Optional[Callable] = None  # Optional validation function

    # Context
    files_needed: List[str] = field(default_factory=list)
    setup_commands: List[str] = field(default_factory=list)

    # Metadata
    estimated_time_seconds: int = 300   # Estimated completion time
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding validation_fn)."""
        data = asdict(self)
        data["category"] = self.category.value
        data["difficulty"] = self.difficulty.value
        data.pop("validation_fn", None)  # Can't serialize functions
        return data


@dataclass
class TaskResult:
    """Results from executing a benchmark task."""

    result_id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str = ""
    task_name: str = ""
    context_mode: ContextMode = ContextMode.NONE

    # Execution metrics
    success: bool = False
    execution_time_seconds: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0

    # Detailed metrics
    tokens_prompt: int = 0
    tokens_completion: int = 0
    context_tokens: int = 0           # Tokens from context engineering

    # Quality metrics
    validation_passed: bool = False
    validation_message: str = ""
    manual_review_score: Optional[float] = None  # 0-1 scale if manually reviewed

    # Context engineering usage
    memories_used: int = 0
    playbook_instructions_used: int = 0
    dynamic_updates_applied: int = 0
    ab_tests_active: int = 0

    # Error tracking
    errors_encountered: int = 0
    error_messages: List[str] = field(default_factory=list)

    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["context_mode"] = self.context_mode.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskResult:
        """Create from dictionary."""
        if "context_mode" in data and isinstance(data["context_mode"], str):
            data["context_mode"] = ContextMode(data["context_mode"])
        return cls(**data)


@dataclass
class BenchmarkRun:
    """A complete benchmark run with multiple tasks."""

    run_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    context_mode: ContextMode = ContextMode.NONE

    # Tasks and results
    tasks: List[BenchmarkTask] = field(default_factory=list)
    results: List[TaskResult] = field(default_factory=list)

    # Run metadata
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # System info
    python_version: str = ""
    devagent_version: str = ""
    llm_model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["context_mode"] = self.context_mode.value
        data["tasks"] = [t.to_dict() for t in self.tasks]
        data["results"] = [r.to_dict() for r in self.results]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkRun:
        """Create from dictionary."""
        if "context_mode" in data and isinstance(data["context_mode"], str):
            data["context_mode"] = ContextMode(data["context_mode"])
        # Note: tasks and results reconstruction omitted for brevity
        return cls(**data)

    @property
    def total_success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.results:
            return 0.0
        successes = sum(1 for r in self.results if r.success)
        return successes / len(self.results)

    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time."""
        if not self.results:
            return 0.0
        return sum(r.execution_time_seconds for r in self.results) / len(self.results)

    @property
    def total_tokens_used(self) -> int:
        """Calculate total tokens used."""
        return sum(r.tokens_used for r in self.results)

    @property
    def avg_tokens_per_task(self) -> float:
        """Calculate average tokens per task."""
        if not self.results:
            return 0.0
        return self.total_tokens_used / len(self.results)


class BenchmarkFramework:
    """Framework for running and comparing benchmarks."""

    DEFAULT_RESULTS_PATH = Path.home() / ".devagent" / "benchmarks"

    def __init__(self, results_path: Optional[Path] = None):
        """Initialize the benchmark framework.

        Args:
            results_path: Path to store benchmark results
        """
        self.results_path = results_path or self.DEFAULT_RESULTS_PATH
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Benchmark runs storage
        self._runs: Dict[str, BenchmarkRun] = {}

        # Load existing runs
        self._load_runs()

    def _load_runs(self) -> None:
        """Load existing benchmark runs from disk."""
        runs_file = self.results_path / "runs.json"
        if runs_file.exists():
            try:
                with open(runs_file, "r") as f:
                    data = json.load(f)
                    for run_data in data.get("runs", []):
                        run = BenchmarkRun.from_dict(run_data)
                        self._runs[run.run_id] = run
                logger.info(f"Loaded {len(self._runs)} benchmark runs")
            except Exception as e:
                logger.error(f"Failed to load benchmark runs: {e}")

    def _save_runs(self) -> None:
        """Save benchmark runs to disk."""
        runs_file = self.results_path / "runs.json"
        try:
            with open(runs_file, "w") as f:
                json.dump({
                    "runs": [run.to_dict() for run in self._runs.values()],
                    "saved_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save benchmark runs: {e}")

    def create_run(
        self,
        name: str,
        tasks: List[BenchmarkTask],
        context_mode: ContextMode,
        description: str = ""
    ) -> BenchmarkRun:
        """Create a new benchmark run.

        Args:
            name: Run name
            tasks: List of tasks to execute
            context_mode: Context engineering mode to use
            description: Run description

        Returns:
            Created BenchmarkRun
        """
        run = BenchmarkRun(
            name=name,
            description=description,
            context_mode=context_mode,
            tasks=tasks
        )

        self._runs[run.run_id] = run
        self._save_runs()

        logger.info(f"Created benchmark run: {name} ({len(tasks)} tasks)")
        return run

    def execute_task(
        self,
        task: BenchmarkTask,
        context_mode: ContextMode,
        executor_fn: Callable[[BenchmarkTask, ContextMode], TaskResult]
    ) -> TaskResult:
        """Execute a single benchmark task.

        Args:
            task: The task to execute
            context_mode: Context mode to use
            executor_fn: Function that executes the task and returns results

        Returns:
            TaskResult with execution metrics
        """
        logger.info(f"Executing task: {task.name} (mode: {context_mode.value})")

        start_time = time.time()

        try:
            # Execute the task using provided executor function
            result = executor_fn(task, context_mode)
            result.execution_time_seconds = time.time() - start_time
            result.completed_at = datetime.now().isoformat()

        except Exception as e:
            # Create error result
            result = TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                context_mode=context_mode,
                success=False,
                execution_time_seconds=time.time() - start_time,
                errors_encountered=1,
                error_messages=[str(e)],
                completed_at=datetime.now().isoformat()
            )
            logger.error(f"Task execution failed: {e}")

        return result

    def execute_run(
        self,
        run: BenchmarkRun,
        executor_fn: Callable[[BenchmarkTask, ContextMode], TaskResult],
        max_tasks: Optional[int] = None
    ) -> BenchmarkRun:
        """Execute all tasks in a benchmark run.

        Args:
            run: The benchmark run to execute
            executor_fn: Function to execute each task
            max_tasks: Maximum tasks to execute (for testing)

        Returns:
            Updated BenchmarkRun with results
        """
        logger.info(f"Starting benchmark run: {run.name}")

        tasks_to_run = run.tasks[:max_tasks] if max_tasks else run.tasks

        for i, task in enumerate(tasks_to_run, 1):
            logger.info(f"Task {i}/{len(tasks_to_run)}: {task.name}")

            result = self.execute_task(task, run.context_mode, executor_fn)
            run.results.append(result)

            # Save progress after each task
            self._save_runs()

        run.completed_at = datetime.now().isoformat()
        self._save_runs()

        logger.info(f"Completed benchmark run: {run.name}")
        logger.info(f"Success rate: {run.total_success_rate:.1%}")
        logger.info(f"Avg time: {run.avg_execution_time:.1f}s")
        logger.info(f"Total tokens: {run.total_tokens_used}")

        return run

    def compare_runs(
        self,
        baseline_run_id: str,
        comparison_run_id: str
    ) -> Dict[str, Any]:
        """Compare two benchmark runs.

        Args:
            baseline_run_id: ID of baseline run
            comparison_run_id: ID of comparison run

        Returns:
            Dictionary with comparison metrics
        """
        baseline = self._runs.get(baseline_run_id)
        comparison = self._runs.get(comparison_run_id)

        if not baseline or not comparison:
            raise ValueError("Invalid run IDs")

        # Calculate improvements
        success_improvement = (
            (comparison.total_success_rate - baseline.total_success_rate) /
            baseline.total_success_rate * 100 if baseline.total_success_rate > 0 else 0
        )

        time_improvement = (
            (baseline.avg_execution_time - comparison.avg_execution_time) /
            baseline.avg_execution_time * 100 if baseline.avg_execution_time > 0 else 0
        )

        token_improvement = (
            (baseline.avg_tokens_per_task - comparison.avg_tokens_per_task) /
            baseline.avg_tokens_per_task * 100 if baseline.avg_tokens_per_task > 0 else 0
        )

        return {
            "baseline": {
                "name": baseline.name,
                "context_mode": baseline.context_mode.value,
                "success_rate": baseline.total_success_rate,
                "avg_time": baseline.avg_execution_time,
                "avg_tokens": baseline.avg_tokens_per_task
            },
            "comparison": {
                "name": comparison.name,
                "context_mode": comparison.context_mode.value,
                "success_rate": comparison.total_success_rate,
                "avg_time": comparison.avg_execution_time,
                "avg_tokens": comparison.avg_tokens_per_task
            },
            "improvements": {
                "success_rate_pct": success_improvement,
                "time_reduction_pct": time_improvement,
                "token_reduction_pct": token_improvement
            },
            "meets_targets": {
                "success_rate": success_improvement >= 30,  # Target: ≥30%
                "token_reduction": token_improvement >= 30,  # Target: ≥30%
                "time_reduction": time_improvement >= 20    # Target: ≥20%
            }
        }

    def get_run(self, run_id: str) -> Optional[BenchmarkRun]:
        """Get a benchmark run by ID.

        Args:
            run_id: Run ID

        Returns:
            BenchmarkRun or None
        """
        return self._runs.get(run_id)

    def get_all_runs(self) -> List[BenchmarkRun]:
        """Get all benchmark runs.

        Returns:
            List of all BenchmarkRun objects
        """
        return list(self._runs.values())

    def generate_report(
        self,
        run_id: str,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a detailed report for a benchmark run.

        Args:
            run_id: Run ID
            output_path: Optional path to save report

        Returns:
            Report as markdown string
        """
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        # Build report
        report = f"# Benchmark Report: {run.name}\n\n"
        report += f"**Context Mode**: {run.context_mode.value}\n\n"
        report += f"**Started**: {run.started_at}\n\n"
        report += f"**Completed**: {run.completed_at}\n\n"

        report += "## Overall Metrics\n\n"
        report += f"- **Success Rate**: {run.total_success_rate:.1%}\n"
        report += f"- **Average Time**: {run.avg_execution_time:.1f}s\n"
        report += f"- **Total Tokens**: {run.total_tokens_used:,}\n"
        report += f"- **Avg Tokens/Task**: {run.avg_tokens_per_task:.0f}\n\n"

        # By category
        report += "## Results by Category\n\n"
        by_category: Dict[str, List[TaskResult]] = {}
        for result in run.results:
            # Find task to get category
            task = next((t for t in run.tasks if t.task_id == result.task_id), None)
            if task:
                cat = task.category.value
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(result)

        for category, results in sorted(by_category.items()):
            successes = sum(1 for r in results if r.success)
            success_rate = successes / len(results) if results else 0
            report += f"### {category.title()}\n"
            report += f"- Tasks: {len(results)}\n"
            report += f"- Success Rate: {success_rate:.1%}\n\n"

        # Context engineering usage
        if run.context_mode != ContextMode.NONE:
            report += "## Context Engineering Usage\n\n"
            total_memories = sum(r.memories_used for r in run.results)
            total_instructions = sum(r.playbook_instructions_used for r in run.results)
            total_updates = sum(r.dynamic_updates_applied for r in run.results)

            report += f"- **Memories Used**: {total_memories}\n"
            report += f"- **Playbook Instructions**: {total_instructions}\n"
            report += f"- **Dynamic Updates**: {total_updates}\n\n"

        # Save if requested
        if output_path:
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report
