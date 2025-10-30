"""Benchmark runner for evaluating LLM provider performance."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_dev_agent.core.utils.config import load_settings
from ai_dev_agent.tools import FIND, GREP, READ, RUN, SYMBOLS, WRITE

if TYPE_CHECKING:
    from benchmarks.test_cases import TestCase


@dataclass
class CodeQualityMetrics:
    """Code quality metrics for task-based benchmarks."""

    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    syntax_errors: int = 0
    lint_errors: int = 0
    type_errors: int = 0
    malformed_response: bool = False
    has_expected_keywords: bool = False
    code_generated_lines: int = 0


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    provider_name: str
    provider: str
    model: str
    test_name: str
    query: str
    success: bool
    correct_answer: bool
    execution_time: float
    iterations: int
    tools_used: int
    answer: str
    error: str | None = None
    timeout: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Enhanced quality metrics
    quality_metrics: CodeQualityMetrics | None = None
    pass_rate: float | None = None  # Percentage of tests passed (0-100)
    task_type: str | None = None  # Type of task (query, code_generation, bug_fix, etc.)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Flatten quality_metrics for easier CSV export
        if self.quality_metrics:
            result["quality_metrics"] = asdict(self.quality_metrics)
        return result


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""

    name: str
    provider: str
    model: str
    base_url: str
    api_key: str
    max_completion_tokens: int | None = None


class BenchmarkRunner:
    """Orchestrates benchmark execution across providers."""

    def __init__(self, config_path: Path | None = None):
        """Initialize benchmark runner.

        Args:
            config_path: Path to providers_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "providers_config.yaml"

        self.config_path = config_path
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.providers = self._load_providers()
        self.default_settings = self.config.get("default_settings", {})

    def _load_providers(self) -> list[ProviderConfig]:
        """Load provider configurations from config file."""
        providers = []

        # Load settings from .devagent.toml
        settings = load_settings()

        # Get API key from settings
        api_key = settings.api_key or os.getenv("DEVAGENT_API_KEY")

        for prov_conf in self.config.get("providers", []):
            # Use the same API key for all providers (from .devagent.toml)
            current_api_key = api_key

            if not current_api_key:
                print(f"Warning: No API key found for {prov_conf['name']}, skipping")
                continue

            providers.append(
                ProviderConfig(
                    name=prov_conf["name"],
                    provider=prov_conf["provider"],
                    model=prov_conf["model"],
                    base_url=prov_conf["base_url"],
                    api_key=current_api_key,
                    max_completion_tokens=prov_conf.get("max_completion_tokens"),
                )
            )

        return providers

    def _extract_iterations(self, output: str) -> int:
        """Extract iteration count from output."""
        # Direct mode has no iterations, but is effectively 1 iteration
        if "Direct execution mode" in output or "direct)" in output:
            return 1

        # Look for patterns like "iteration 3" or "Iteration 5"
        matches = re.findall(r"iteration[:\s]+(\d+)", output, re.IGNORECASE)
        if matches:
            return max(int(m) for m in matches)

        # If completed, assume at least 1 iteration
        if "Completed in" in output or "✅" in output:
            return 1

        return 0

    def _extract_tools_used(self, output: str) -> int:
        f"""Extract number of tools used from output.

        Based on DevAgent's tool categories and status indicators:
        - ⚡ = command execution ({RUN}, sandbox.exec)
        - 🔍 = search operations ({FIND}/{GREP})
        - 📖 = file reading ({READ})
        - 📝 = file writing/patching ({WRITE})
        - 🔣 = symbol operations ({SYMBOLS})
        - 📁 = directory listing
        - ❌ = failed tool (still counts as tool usage)
        """
        tools = 0

        # IMPORTANT: Use specific patterns with quote delimiters to avoid double-counting
        # Each tool invocation format: "emoji tool_name "argument" → result"

        # Command execution (⚡ exec, ⚡ sandbox.exec)
        tools += len(re.findall(r'⚡\s+run\s+"', output))
        tools += len(re.findall(r'⚡\s+sandbox\.exec\s+"', output))

        # Search operations (🔍 find/grep "query")
        tools += len(re.findall(r'🔍\s+code\.search\s+"', output))

        # File reading (📖 read "path")
        tools += len(re.findall(r'📖\s+read\s+"', output))

        # File writing/patching (📝 write)
        tools += len(re.findall(r"📝\s+write", output))

        # Symbol operations (🔣 symbols)
        tools += len(re.findall(r"🔣\s+symbols\.(?:find|index)\s+", output))

        # Directory/file listing (📁 list or 📁 ls)
        tools += len(re.findall(r"📁\s+(?:list|ls)\s+", output))

        # Failed tools (❌ tool_name "arg")
        # Only count if followed by a tool name pattern and quote
        failed_tools = re.findall(
            r'❌\s+(?:exec|code\.search|(?:fs\.)?(?:read|write)|symbols\.(?:find|index))\s+"',
            output,
        )
        tools += len(failed_tools)

        # Fallback: if no tools found but task completed, check completion message
        if tools == 0 and ("Completed" in output or "✅" in output):
            # Parse from completion message if available
            tool_count_match = re.search(r"(\d+)\s+tools?\s+", output)
            if tool_count_match:
                tools = int(tool_count_match.group(1))

        return tools

    def run_single_test(
        self,
        provider: ProviderConfig,
        test_case: TestCase,
    ) -> BenchmarkResult:
        """Run a single test case with a specific provider.

        Args:
            provider: Provider configuration
            test_case: Test case to run

        Returns:
            BenchmarkResult with execution metrics
        """
        print(f"\n{'='*80}")
        print(f"Running: {test_case.name}")
        print(f"Provider: {provider.name}")
        print(f"Query: {test_case.query}")
        print(f"{'='*80}")

        # Create temporary config file for this provider
        temp_config = self._create_temp_config(provider)

        start_time = time.time()
        success = False
        correct_answer = False
        iterations = 0
        tools_used = 0
        answer = ""
        error_msg = None
        timeout = False

        try:
            # Execute devagent via subprocess to avoid conflicts
            cmd = [
                sys.executable,
                "-m",
                "ai_dev_agent.cli.commands",
                "--config",
                str(temp_config),
                "query",
                test_case.query,
            ]

            result_proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=test_case.timeout,
                cwd=Path.cwd(),
            )

            answer = result_proc.stdout + result_proc.stderr
            success = result_proc.returncode == 0

            if not success:
                error_msg = f"Process exited with code {result_proc.returncode}"

            # Extract metrics from output
            iterations = self._extract_iterations(answer)
            tools_used = self._extract_tools_used(answer)

            # Validate answer
            if success and answer:
                correct_answer = test_case.validate_answer(answer)

        except subprocess.TimeoutExpired:
            timeout = True
            error_msg = "Execution timeout"
            answer = "Timeout"
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
        finally:
            # Cleanup temp config
            try:
                if temp_config.exists():
                    temp_config.unlink()
            except Exception:
                pass  # Ignore cleanup errors

        execution_time = time.time() - start_time

        result = BenchmarkResult(
            provider_name=provider.name,
            provider=provider.provider,
            model=provider.model,
            test_name=test_case.name,
            query=test_case.query,
            success=success,
            correct_answer=correct_answer,
            execution_time=round(execution_time, 2),
            iterations=iterations,
            tools_used=tools_used,
            answer=answer[:500] if answer else "",  # Truncate for storage
            error=error_msg,
            timeout=timeout,
        )

        # Print summary
        status = "✓ PASS" if correct_answer else "✗ FAIL"
        print(f"\n{status} - {execution_time:.2f}s - {iterations} iterations - {tools_used} tools")

        return result

    def _create_temp_config(self, provider: ProviderConfig) -> Path:
        """Create a temporary .devagent.toml config file.

        Args:
            provider: Provider configuration

        Returns:
            Path to temporary config file
        """
        temp_config = Path(
            f".devagent.benchmark.{provider.provider}.{provider.model.replace('/', '-')}.toml"
        )

        config_content = f"""# Temporary benchmark config
provider = "{provider.provider}"
model = "{provider.model}"
api_key = "{provider.api_key}"
base_url = "{provider.base_url}"
auto_approve_code = true
max_iterations = 20
"""
        if provider.max_completion_tokens:
            config_content += f"max_completion_tokens = {provider.max_completion_tokens}\n"

        temp_config.write_text(config_content)
        return temp_config

    def run_benchmarks(
        self,
        test_cases: list[TestCase] | None = None,
        providers: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """Run benchmarks across all providers and test cases.

        Args:
            test_cases: List of test cases to run (default: SIMPLE_QUERIES)
            providers: List of provider names to test (default: all)

        Returns:
            List of all benchmark results
        """
        if test_cases is None:
            test_cases = SIMPLE_QUERIES

        # Filter providers if specified
        active_providers = self.providers
        if providers:
            active_providers = [p for p in self.providers if p.name in providers]

        results = []

        for provider in active_providers:
            for test_case in test_cases:
                result = self.run_single_test(provider, test_case)
                results.append(result)

        return results

    def save_results(self, results: list[BenchmarkResult], output_name: str = "benchmark"):
        """Save benchmark results in multiple formats.

        Args:
            results: List of benchmark results
            output_name: Base name for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.results_dir / f"{output_name}_{timestamp}"

        # Save JSON
        json_path = base_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to: {json_path}")

        # Save CSV
        csv_path = base_path.with_suffix(".csv")
        self._save_csv(results, csv_path)
        print(f"CSV saved to: {csv_path}")

        # Save Markdown report
        md_path = base_path.with_suffix(".md")
        self._save_markdown(results, md_path)
        print(f"Report saved to: {md_path}")

    def _save_csv(self, results: list[BenchmarkResult], path: Path):
        """Save results as CSV."""
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "provider_name",
                    "model",
                    "test_name",
                    "correct_answer",
                    "execution_time",
                    "iterations",
                    "tools_used",
                    "error",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "provider_name": result.provider_name,
                        "model": result.model,
                        "test_name": result.test_name,
                        "correct_answer": result.correct_answer,
                        "execution_time": result.execution_time,
                        "iterations": result.iterations,
                        "tools_used": result.tools_used,
                        "error": result.error or "",
                    }
                )

    def _save_markdown(self, results: list[BenchmarkResult], path: Path):
        """Save results as Markdown report."""
        lines = [
            "# DevAgent Benchmark Results",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Total Tests:** {len(results)}",
            "\n## Summary\n",
        ]

        # Group by test
        from collections import defaultdict

        by_test = defaultdict(list)
        for r in results:
            by_test[r.test_name].append(r)

        for test_name, test_results in by_test.items():
            lines.append(f"\n### {test_name}\n")
            lines.append("| Provider | Model | Time (s) | Iterations | Tools | Correct | Status |")
            lines.append("|----------|-------|----------|------------|-------|---------|--------|")

            for r in test_results:
                status = "✓" if r.correct_answer else "✗"
                if r.error:
                    status = f"ERROR: {r.error[:30]}"

                lines.append(
                    f"| {r.provider_name} | {r.model} | {r.execution_time:.2f} | "
                    f"{r.iterations} | {r.tools_used} | {r.correct_answer} | {status} |"
                )

        # Add rankings
        lines.append("\n## Rankings (by execution time)\n")
        lines.append("| Rank | Provider | Avg Time (s) | Success Rate |")
        lines.append("|------|----------|--------------|--------------|")

        # Calculate averages by provider
        provider_stats = defaultdict(lambda: {"times": [], "correct": 0, "total": 0})
        for r in results:
            provider_stats[r.provider_name]["times"].append(r.execution_time)
            provider_stats[r.provider_name]["total"] += 1
            if r.correct_answer:
                provider_stats[r.provider_name]["correct"] += 1

        # Sort by average time
        ranked = sorted(
            provider_stats.items(),
            key=lambda x: sum(x[1]["times"]) / len(x[1]["times"]),
        )

        for rank, (prov_name, stats) in enumerate(ranked, 1):
            avg_time = sum(stats["times"]) / len(stats["times"])
            success_rate = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            lines.append(f"| {rank} | {prov_name} | {avg_time:.2f} | {success_rate:.1f}% |")

        path.write_text("\n".join(lines))


def main():
    """Main entry point for benchmark runner."""
    runner = BenchmarkRunner()

    print("=" * 80)
    print("DevAgent Provider Benchmark")
    print("=" * 80)
    print(f"\nLoaded {len(runner.providers)} providers:")
    for p in runner.providers:
        print(f"  - {p.name} ({p.model})")

    print(f"\nTest cases: {len(SIMPLE_QUERIES)}")
    for tc in SIMPLE_QUERIES:
        print(f"  - {tc.name}: {tc.query}")

    # Run benchmarks
    results = runner.run_benchmarks()

    # Save results
    runner.save_results(results)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    correct = sum(1 for r in results if r.correct_answer)
    print(f"Total runs: {len(results)}")
    print(f"Correct answers: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
