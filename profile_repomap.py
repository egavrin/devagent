#!/usr/bin/env python3
"""Profile the RepoMap scanning and ranking operations on a large repository."""

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from ai_dev_agent.cli.context_enhancer import ContextEnhancer
from ai_dev_agent.core.repo_map import RepoMapManager
from ai_dev_agent.core.utils.config import Settings


def profile_repomap_scan(repo_path: Path):
    """Profile the repository scanning phase."""
    print(f"\n{'='*80}")
    print(f"Profiling RepoMap scan on: {repo_path}")
    print(f"{'='*80}\n")

    # Clear any existing instance
    RepoMapManager.clear_instance(repo_path)

    # Profile the scan
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    rm = RepoMapManager.get_instance(repo_path)
    # Force a fresh scan
    rm.scan_repository(force=True)
    scan_time = time.time() - start_time

    pr.disable()

    print(f"\nScan completed in {scan_time:.2f} seconds")
    print(f"Files indexed: {len(rm.context.files)}")
    print(f"Symbols indexed: {len(rm.context.symbol_index)}")

    # Print profiling stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(50)  # Top 50 functions by cumulative time

    print("\n" + "=" * 80)
    print("TOP 50 FUNCTIONS BY CUMULATIVE TIME (scan phase)")
    print("=" * 80)
    print(s.getvalue())

    return rm


def profile_pagerank_computation(rm):
    """Profile the PageRank computation phase."""
    print(f"\n{'='*80}")
    print("Profiling PageRank computation")
    print(f"{'='*80}\n")

    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    rm.build_dependency_graph()
    graph_time = time.time() - start_time

    start_time = time.time()
    rm.compute_pagerank()
    pagerank_time = time.time() - start_time

    pr.disable()

    print(f"\nGraph building: {graph_time:.2f} seconds")
    print(f"PageRank computation: {pagerank_time:.2f} seconds")

    if rm.context.dependency_graph:
        print(f"Nodes: {rm.context.dependency_graph.number_of_nodes()}")
        print(f"Edges: {rm.context.dependency_graph.number_of_edges()}")

    # Print profiling stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(50)

    print("\n" + "=" * 80)
    print("TOP 50 FUNCTIONS BY CUMULATIVE TIME (PageRank phase)")
    print("=" * 80)
    print(s.getvalue())


def profile_context_enhancement(repo_path: Path, query: str):
    """Profile the context enhancement phase."""
    print(f"\n{'='*80}")
    print("Profiling context enhancement for query")
    print(f"{'='*80}\n")
    print(f"Query: {query[:100]}...")

    settings = Settings()
    settings.repomap_debug_stdout = True

    enhancer = ContextEnhancer(workspace=repo_path, settings=settings)

    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    _original_query, repomap_messages = enhancer.get_repomap_messages(query, max_files=15)
    enhancement_time = time.time() - start_time

    pr.disable()

    print(f"\nContext enhancement: {enhancement_time:.2f} seconds")
    if repomap_messages:
        print(f"Messages generated: {len(repomap_messages)}")
        print(f"Total message chars: {sum(len(m['content']) for m in repomap_messages)}")

    # Print profiling stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(50)

    print("\n" + "=" * 80)
    print("TOP 50 FUNCTIONS BY CUMULATIVE TIME (context enhancement)")
    print("=" * 80)
    print(s.getvalue())


def profile_file_ranking(rm, symbols: set[str], mentioned_files: set[str]):
    """Profile the file ranking phase."""
    print(f"\n{'='*80}")
    print("Profiling get_ranked_files")
    print(f"{'='*80}\n")
    print(f"Symbols: {list(symbols)[:10]}...")
    print(f"Mentioned files: {list(mentioned_files)[:5]}...")

    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    ranked_files = rm.get_ranked_files(
        mentioned_files=mentioned_files, mentioned_symbols=symbols, max_files=20
    )
    ranking_time = time.time() - start_time

    pr.disable()

    print(f"\nFile ranking: {ranking_time:.2f} seconds")
    print(f"Files returned: {len(ranked_files)}")
    if ranked_files:
        print("\nTop 5 ranked files:")
        for file_path, score in ranked_files[:5]:
            print(f"  {file_path}: {score:.2f}")

    # Print profiling stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(50)

    print("\n" + "=" * 80)
    print("TOP 50 FUNCTIONS BY CUMULATIVE TIME (file ranking)")
    print("=" * 80)
    print(s.getvalue())


def main():
    if len(sys.argv) < 3:
        print("Usage: python profile_repomap.py <repo_path> <query>")
        print("\nExample:")
        print('  python profile_repomap.py /path/to/repo "How to check if type is Any"')
        sys.exit(1)

    repo_path = Path(sys.argv[1]).expanduser().resolve()
    query = " ".join(sys.argv[2:])

    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    print(f"Repository: {repo_path}")
    print(f"Query: {query}")

    # 1. Profile scanning
    rm = profile_repomap_scan(repo_path)

    # 2. Profile PageRank
    profile_pagerank_computation(rm)

    # 3. Profile context enhancement (full pipeline)
    profile_context_enhancement(repo_path, query)

    # 4. Profile just the ranking step
    symbols = {"IsETSAnyType", "ETSGen", "type", "emitted", "Any", "generics", "union"}
    mentioned_files = set()
    profile_file_ranking(rm, symbols, mentioned_files)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
