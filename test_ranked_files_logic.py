#!/usr/bin/env python3
"""Test get_ranked_files with the fixed fast-path logic."""

import tempfile
from pathlib import Path
from ai_dev_agent.core.repo_map import RepoMap
import logging

# Enable debug logging to see fast-path decisions
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def test_cases():
    """Test various query types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test structure
        runtime_dir = tmpdir / "runtime"
        runtime_dir.mkdir()

        (runtime_dir / "important.py").write_text("""
class ImportantClass:
    def critical_method(self):
        pass
""")

        (runtime_dir / "trivial.py").write_text("""
x = 1
""")

        compiler_dir = tmpdir / "compiler"
        compiler_dir.mkdir()

        (compiler_dir / "optimizer.py").write_text("""
class Optimizer:
    pass
""")

        rm = RepoMap(root_path=tmpdir, cache_enabled=False)
        rm.scan_repository(force=True)

        print("\n" + "="*60)
        print("TEST 1: Symbol match (should use fast-path)")
        print("="*60)
        results1 = rm.get_ranked_files(set(), {"ImportantClass"}, max_files=5)
        print(f"Results: {results1}\n")

        print("="*60)
        print("TEST 2: Exact file match (should use fast-path)")
        print("="*60)
        results2 = rm.get_ranked_files({"important.py"}, set(), max_files=5)
        print(f"Results: {results2}\n")

        print("="*60)
        print("TEST 3: Directory match only (should use PageRank)")
        print("="*60)
        results3 = rm.get_ranked_files({"runtime"}, set(), max_files=5)
        print(f"Results: {results3}\n")

        print("="*60)
        print("TEST 4: No matches (should use PageRank)")
        print("="*60)
        results4 = rm.get_ranked_files(set(), {"NonExistent"}, max_files=5)
        print(f"Results: {results4}\n")


if __name__ == "__main__":
    test_cases()
