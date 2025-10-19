"""Coverage enforcement gate for DevAgent.

This module provides utilities to enforce code coverage requirements
and generate detailed coverage reports.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoverageResult:
    """Represents coverage test results."""
    total_coverage: float
    passed: bool
    threshold: float
    uncovered_files: List[str]
    report: str
    details: Dict[str, float]


class CoverageGate:
    """Enforces code coverage requirements for the project."""

    def __init__(self, threshold: float = 95.0, config_file: Optional[Path] = None):
        """Initialize the coverage gate.

        Args:
            threshold: Minimum coverage percentage required (default 95%)
            config_file: Path to .coveragerc file (uses project default if None)
        """
        self.threshold = threshold
        self.config_file = config_file or Path.cwd() / ".coveragerc"
        self.project_root = Path.cwd()

    def run_coverage(self,
                    test_path: Optional[str] = None,
                    parallel: bool = True,
                    html_report: bool = True) -> CoverageResult:
        """Run tests with coverage and return results.

        Args:
            test_path: Specific test path to run (runs all if None)
            parallel: Enable parallel test execution
            html_report: Generate HTML coverage report

        Returns:
            CoverageResult with coverage metrics and pass/fail status
        """
        try:
            # Build pytest command
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=ai_dev_agent",
                f"--cov-config={self.config_file}",
                "--cov-report=term-missing",
                "--cov-report=json",
                f"--cov-fail-under={self.threshold}",
            ]

            if html_report:
                cmd.append("--cov-report=html")

            if parallel:
                # Add parallel execution if pytest-xdist is available
                try:
                    import pytest_xdist
                    cmd.extend(["-n", "auto"])
                except ImportError:
                    logger.info("pytest-xdist not installed, running tests sequentially")

            if test_path:
                cmd.append(test_path)
            else:
                cmd.append("tests/")

            # Run tests with coverage
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            # Parse coverage results
            coverage_data = self._parse_coverage_json()

            # Extract total coverage
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)

            # Check if coverage meets threshold
            passed = total_coverage >= self.threshold

            # Get uncovered files
            uncovered_files = self._get_uncovered_files(coverage_data)

            # Build detailed report
            details = self._get_file_coverage(coverage_data)

            return CoverageResult(
                total_coverage=total_coverage,
                passed=passed,
                threshold=self.threshold,
                uncovered_files=uncovered_files,
                report=result.stdout,
                details=details
            )

        except Exception as e:
            logger.error(f"Coverage run failed: {e}")
            return CoverageResult(
                total_coverage=0.0,
                passed=False,
                threshold=self.threshold,
                uncovered_files=[],
                report=str(e),
                details={}
            )

    def _parse_coverage_json(self) -> Dict:
        """Parse the coverage.json file."""
        coverage_file = self.project_root / "coverage.json"
        if not coverage_file.exists():
            return {}

        with open(coverage_file, 'r') as f:
            return json.load(f)

    def _get_uncovered_files(self, coverage_data: Dict) -> List[str]:
        """Extract list of files with insufficient coverage."""
        uncovered = []
        files = coverage_data.get("files", {})

        for file_path, file_data in files.items():
            file_coverage = file_data.get("summary", {}).get("percent_covered", 0)
            if file_coverage < self.threshold:
                uncovered.append(f"{file_path}: {file_coverage:.1f}%")

        return uncovered

    def _get_file_coverage(self, coverage_data: Dict) -> Dict[str, float]:
        """Get coverage percentage for each file."""
        details = {}
        files = coverage_data.get("files", {})

        for file_path, file_data in files.items():
            # Simplify file path for readability
            relative_path = file_path.replace(str(self.project_root) + "/", "")
            coverage = file_data.get("summary", {}).get("percent_covered", 0)
            details[relative_path] = coverage

        return details

    def enforce(self, exit_on_fail: bool = True) -> bool:
        """Enforce coverage requirements.

        Args:
            exit_on_fail: Exit process if coverage fails

        Returns:
            True if coverage meets threshold, False otherwise
        """
        result = self.run_coverage()

        if result.passed:
            print(f"✅ Coverage check passed: {result.total_coverage:.1f}% >= {result.threshold}%")
            return True
        else:
            print(f"❌ Coverage check failed: {result.total_coverage:.1f}% < {result.threshold}%")

            if result.uncovered_files:
                print("\nFiles with insufficient coverage:")
                for file in result.uncovered_files[:10]:  # Show top 10
                    print(f"  - {file}")
                if len(result.uncovered_files) > 10:
                    print(f"  ... and {len(result.uncovered_files) - 10} more files")

            if exit_on_fail:
                sys.exit(1)

            return False

    def get_incremental_coverage(self,
                                 base_branch: str = "main",
                                 current_branch: Optional[str] = None) -> Dict:
        """Calculate coverage for only changed files.

        Args:
            base_branch: Base branch to compare against
            current_branch: Current branch (uses current if None)

        Returns:
            Dictionary with incremental coverage metrics
        """
        try:
            # Get list of changed files
            cmd = ["git", "diff", f"{base_branch}...HEAD", "--name-only"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            changed_files = result.stdout.strip().split('\n')

            # Filter for Python files
            py_files = [f for f in changed_files if f.endswith('.py')]

            if not py_files:
                return {"message": "No Python files changed", "coverage": 100.0}

            # Run coverage on changed files only
            coverage_data = self._parse_coverage_json()
            incremental = {}

            for file in py_files:
                if file in coverage_data.get("files", {}):
                    file_coverage = coverage_data["files"][file].get("summary", {}).get("percent_covered", 0)
                    incremental[file] = file_coverage

            # Calculate average incremental coverage
            if incremental:
                avg_coverage = sum(incremental.values()) / len(incremental)
            else:
                avg_coverage = 0.0

            return {
                "files": incremental,
                "average": avg_coverage,
                "passed": avg_coverage >= self.threshold
            }

        except Exception as e:
            logger.error(f"Failed to calculate incremental coverage: {e}")
            return {"error": str(e)}


def check_coverage(threshold: float = 95.0) -> bool:
    """Quick function to check if coverage meets threshold.

    Args:
        threshold: Minimum coverage percentage required

    Returns:
        True if coverage meets threshold, False otherwise
    """
    gate = CoverageGate(threshold=threshold)
    result = gate.run_coverage()
    return result.passed


def enforce_coverage(threshold: float = 95.0, exit_on_fail: bool = True) -> bool:
    """Enforce coverage requirements with optional exit.

    Args:
        threshold: Minimum coverage percentage required
        exit_on_fail: Exit process if coverage fails

    Returns:
        True if coverage meets threshold, False otherwise
    """
    gate = CoverageGate(threshold=threshold)
    return gate.enforce(exit_on_fail=exit_on_fail)


if __name__ == "__main__":
    # CLI usage
    import argparse

    parser = argparse.ArgumentParser(description="Enforce code coverage requirements")
    parser.add_argument("--threshold", type=float, default=95.0,
                       help="Minimum coverage percentage (default: 95)")
    parser.add_argument("--no-exit", action="store_true",
                       help="Don't exit on coverage failure")
    parser.add_argument("--incremental", action="store_true",
                       help="Check only changed files")
    parser.add_argument("--base", default="main",
                       help="Base branch for incremental coverage")

    args = parser.parse_args()

    gate = CoverageGate(threshold=args.threshold)

    if args.incremental:
        result = gate.get_incremental_coverage(base_branch=args.base)
        print(f"Incremental coverage: {result.get('average', 0):.1f}%")
        if not result.get('passed', False):
            sys.exit(1 if not args.no_exit else 0)
    else:
        gate.enforce(exit_on_fail=not args.no_exit)