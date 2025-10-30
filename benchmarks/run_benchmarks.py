#!/usr/bin/env python3
"""Run DevAgent benchmarks.

Usage:
    python3 benchmarks/run_benchmarks.py                    # All providers, all tests
    python3 benchmarks/run_benchmarks.py --limit 3          # First 3 providers
    python3 benchmarks/run_benchmarks.py --provider grok    # Single provider (fuzzy match)
    python3 benchmarks/run_benchmarks.py --tests 5          # First 5 tests
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_runner import BenchmarkRunner
from benchmarks.test_cases import TEST_CASES


def main():
    """Run DevAgent benchmarks."""
    parser = argparse.ArgumentParser(description="Run DevAgent benchmarks")
    parser.add_argument(
        "--provider", "-p", help="Provider name or partial match (e.g. 'grok', 'claude')"
    )
    parser.add_argument("--limit", "-l", type=int, help="Limit number of providers")
    parser.add_argument("--tests", "-t", type=int, help="Limit number of tests")
    parser.add_argument("--skip", help="Comma-separated provider names to skip")
    args = parser.parse_args()

    runner = BenchmarkRunner()

    # Filter providers
    providers_to_test = runner.providers

    if args.provider:
        # Fuzzy match provider name
        search = args.provider.lower()
        matched = [p for p in providers_to_test if search in p.name.lower()]
        if not matched:
            print(f"Error: No provider found matching '{args.provider}'")
            print("\nAvailable providers:")
            for p in providers_to_test:
                print(f"  - {p.name}")
            return 1
        providers_to_test = matched

    if args.skip:
        skip_list = [s.strip() for s in args.skip.split(",")]
        providers_to_test = [p for p in providers_to_test if p.name not in skip_list]

    if args.limit:
        providers_to_test = providers_to_test[: args.limit]

    # Filter tests
    tests_to_run = TEST_CASES
    if args.tests:
        tests_to_run = TEST_CASES[: args.tests]

    # Print summary
    print("=" * 80)
    print("DevAgent Benchmarks")
    print("=" * 80)
    print(f"\nProviders: {len(providers_to_test)}")
    for i, p in enumerate(providers_to_test, 1):
        print(f"  {i:2d}. {p.name}")

    print(f"\nTests: {len(tests_to_run)}")
    for i, t in enumerate(tests_to_run, 1):
        print(f"  {i:2d}. {t.name}")

    total_runs = len(providers_to_test) * len(tests_to_run)
    est_time_min = total_runs * 25 / 60

    print(f"\nTotal runs: {total_runs}")
    print(f"Estimated time: ~{est_time_min:.0f} minutes")
    print("=" * 80 + "\n")

    # Run benchmarks
    results = []
    for provider in providers_to_test:
        for test_case in tests_to_run:
            print(f"\n[{len(results)+1}/{total_runs}] {provider.name} - {test_case.name}")
            try:
                result = runner.run_single_test(provider, test_case)
                results.append(result)

                # Show result
                status = "✓" if result.correct_answer else ("✗" if result.success else "ERROR")
                print(
                    f"  {status} {result.execution_time:.2f}s - {result.iterations} iterations - {result.tools_used} tools"
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    # Save results
    print("\nSaving results...")
    output_name = args.provider.replace(" ", "_") if args.provider else "benchmarks"
    runner.save_results(results, output_name=output_name)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.success]
    correct = [r for r in results if r.correct_answer]

    print(f"\nTotal runs: {len(results)}")
    if results:
        print(
            f"Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)"
        )
        print(f"Correct: {len(correct)}/{len(results)} ({len(correct)/len(results)*100:.1f}%)")

        if correct:
            avg_time = sum(r.execution_time for r in correct) / len(correct)
            avg_iterations = sum(r.iterations for r in correct) / len(correct)
            avg_tools = sum(r.tools_used for r in correct) / len(correct)
            print("\nAverages (correct answers):")
            print(f"  Time: {avg_time:.2f}s")
            print(f"  Iterations: {avg_iterations:.1f}")
            print(f"  Tools: {avg_tools:.1f}")

    print("\n" + "=" * 80)
    print(f"Results saved to benchmarks/results/{output_name}_*")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
