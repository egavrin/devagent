"""Tests for caching utilities (ai_dev_agent.core.cache)."""
import json
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from ai_dev_agent.core.cache import RepoMapCache, SymbolCache


class TestSymbolCache:
    """Test SymbolCache functionality."""

    def test_init_with_default_cache_dir(self, tmp_path):
        """Test cache initialization with default directory."""
        cache = SymbolCache(tmp_path)
        assert cache.root == tmp_path
        assert (tmp_path / ".devagent" / "symbol_cache").exists()

    def test_init_with_custom_cache_dir(self, tmp_path):
        """Test cache initialization with custom directory."""
        custom_dir = tmp_path / "custom_cache"
        cache = SymbolCache(tmp_path, cache_dir=custom_dir)
        assert cache.root == tmp_path
        assert custom_dir.exists()

    def test_init_with_disk_cache_failure(self, tmp_path):
        """Test graceful fallback when disk cache fails."""
        with patch('ai_dev_agent.core.cache.DiskCache', side_effect=Exception("Disk error")):
            cache = SymbolCache(tmp_path)
            assert cache.cache is None
            assert hasattr(cache, '_fallback_cache')

    def test_get_symbols_file_not_exists(self, tmp_path):
        """Test getting symbols for non-existent file."""
        cache = SymbolCache(tmp_path)
        non_existent = tmp_path / "nonexistent.py"
        assert cache.get_symbols(non_existent) is None

    def test_get_symbols_cache_miss(self, tmp_path):
        """Test getting symbols when not in cache."""
        cache = SymbolCache(tmp_path)
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        assert cache.get_symbols(test_file) is None

    def test_set_and_get_symbols_disk_cache(self, tmp_path):
        """Test setting and getting symbols with disk cache."""
        cache = SymbolCache(tmp_path)
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        # Ensure mtime is stable before caching
        time.sleep(0.01)

        symbols = ["foo", "bar"]
        cache.set_symbols(test_file, symbols)

        # Should retrieve the same symbols
        result = cache.get_symbols(test_file)
        assert result == symbols

    def test_set_and_get_symbols_fallback_cache(self, tmp_path):
        """Test setting and getting symbols with fallback cache."""
        with patch('ai_dev_agent.core.cache.DiskCache', side_effect=Exception("Disk error")):
            cache = SymbolCache(tmp_path)
            test_file = tmp_path / "test.py"
            test_file.write_text("def bar(): pass")

            symbols = ["bar", "baz"]
            cache.set_symbols(test_file, symbols)

            result = cache.get_symbols(test_file)
            assert result == symbols

    def test_get_symbols_mtime_changed(self, tmp_path):
        """Test that symbols are invalidated when file is modified."""
        cache = SymbolCache(tmp_path)
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        # Ensure mtime is stable before caching
        time.sleep(0.01)

        symbols = ["foo"]
        cache.set_symbols(test_file, symbols)

        # Verify symbols are cached
        assert cache.get_symbols(test_file) == symbols

        # Wait and modify the file (ensure mtime changes)
        time.sleep(0.1)
        test_file.write_text("def foo(): pass\ndef bar(): pass")

        # Cache should be invalidated
        result = cache.get_symbols(test_file)
        assert result is None

    def test_set_symbols_file_not_exists(self, tmp_path):
        """Test setting symbols for non-existent file."""
        cache = SymbolCache(tmp_path)
        non_existent = tmp_path / "nonexistent.py"

        # Should not raise, just return early
        cache.set_symbols(non_existent, ["foo"])

    def test_invalidate_disk_cache(self, tmp_path):
        """Test invalidating a specific file's cache entry."""
        cache = SymbolCache(tmp_path)
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        # Ensure mtime is stable before caching
        time.sleep(0.01)

        symbols = ["foo"]
        cache.set_symbols(test_file, symbols)
        assert cache.get_symbols(test_file) == symbols

        # Invalidate the cache
        cache.invalidate(test_file)
        assert cache.get_symbols(test_file) is None

    def test_invalidate_fallback_cache(self, tmp_path):
        """Test invalidating a specific file's cache entry in fallback cache."""
        with patch('ai_dev_agent.core.cache.DiskCache', side_effect=Exception("Disk error")):
            cache = SymbolCache(tmp_path)
            test_file = tmp_path / "test.py"
            test_file.write_text("def foo(): pass")

            symbols = ["foo"]
            cache.set_symbols(test_file, symbols)
            assert cache.get_symbols(test_file) == symbols

            cache.invalidate(test_file)
            assert cache.get_symbols(test_file) is None

    def test_invalidate_nonexistent_key(self, tmp_path):
        """Test invalidating a cache entry that doesn't exist."""
        cache = SymbolCache(tmp_path)
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        # Should not raise
        cache.invalidate(test_file)

    def test_clear_disk_cache(self, tmp_path):
        """Test clearing all cache entries."""
        cache = SymbolCache(tmp_path)

        # Add multiple entries
        for i in range(3):
            test_file = tmp_path / f"test{i}.py"
            test_file.write_text(f"def foo{i}(): pass")
            time.sleep(0.01)  # Ensure mtime is stable
            cache.set_symbols(test_file, [f"foo{i}"])

        # Verify all are cached
        for i in range(3):
            test_file = tmp_path / f"test{i}.py"
            assert cache.get_symbols(test_file) is not None

        # Clear cache
        cache.clear()

        # All should be gone
        for i in range(3):
            test_file = tmp_path / f"test{i}.py"
            assert cache.get_symbols(test_file) is None

    def test_clear_fallback_cache(self, tmp_path):
        """Test clearing all entries in fallback cache."""
        with patch('ai_dev_agent.core.cache.DiskCache', side_effect=Exception("Disk error")):
            cache = SymbolCache(tmp_path)

            test_file = tmp_path / "test.py"
            test_file.write_text("def foo(): pass")
            cache.set_symbols(test_file, ["foo"])

            cache.clear()
            assert cache.get_symbols(test_file) is None

    def test_get_stats_disk_cache(self, tmp_path):
        """Test getting cache statistics for disk cache."""
        cache = SymbolCache(tmp_path)
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")
        time.sleep(0.01)  # Ensure mtime is stable
        cache.set_symbols(test_file, ["foo"])

        stats = cache.get_stats()
        # Verify we get stats (type depends on if DiskCache initialized successfully)
        assert stats["type"] in ["disk", "memory"]
        assert "size" in stats
        assert stats["size"] >= 1

    def test_get_stats_fallback_cache(self, tmp_path):
        """Test getting cache statistics for fallback cache."""
        with patch('ai_dev_agent.core.cache.DiskCache', side_effect=Exception("Disk error")):
            cache = SymbolCache(tmp_path)
            test_file = tmp_path / "test.py"
            test_file.write_text("def foo(): pass")
            cache.set_symbols(test_file, ["foo"])

            stats = cache.get_stats()
            assert stats["type"] == "memory"
            assert stats["size"] == 1

    def test_cache_read_exception_handling(self, tmp_path):
        """Test that exceptions during cache read are handled gracefully."""
        cache = SymbolCache(tmp_path)
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        # Mock the cache to raise an exception
        with patch.object(cache, 'cache') as mock_cache:
            mock_cache.get.side_effect = Exception("Read error")
            result = cache.get_symbols(test_file)
            assert result is None

    def test_cache_write_exception_handling(self, tmp_path):
        """Test that exceptions during cache write are handled gracefully."""
        cache = SymbolCache(tmp_path)
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        # Mock the cache to raise an exception on write
        with patch.object(cache, 'cache') as mock_cache:
            mock_cache.__setitem__.side_effect = Exception("Write error")
            # Should not raise
            cache.set_symbols(test_file, ["foo"])


class TestRepoMapCache:
    """Test RepoMapCache functionality."""

    def test_init(self, tmp_path):
        """Test repo map cache initialization."""
        cache = RepoMapCache(tmp_path)
        assert cache.root == tmp_path
        assert cache.cache_dir.exists()
        assert cache.graph_cache_file == cache.cache_dir / "graph.pkl"
        assert cache.pagerank_cache_file == cache.cache_dir / "pagerank.json"

    def test_save_and_load_graph(self, tmp_path):
        """Test saving and loading NetworkX graph."""
        cache = RepoMapCache(tmp_path)

        # Create a simple graph
        graph = nx.DiGraph()
        graph.add_edge("fileA.py", "fileB.py")
        graph.add_edge("fileB.py", "fileC.py")

        # Save the graph
        cache.save_graph(graph)
        assert cache.graph_cache_file.exists()

        # Load the graph
        loaded_graph = cache.load_graph()
        assert loaded_graph is not None
        assert len(loaded_graph.nodes()) == 3
        assert len(loaded_graph.edges()) == 2
        assert loaded_graph.has_edge("fileA.py", "fileB.py")

    def test_load_graph_not_exists(self, tmp_path):
        """Test loading graph when cache file doesn't exist."""
        cache = RepoMapCache(tmp_path)
        result = cache.load_graph()
        assert result is None

    def test_save_graph_exception_handling(self, tmp_path):
        """Test that exceptions during graph save are handled gracefully."""
        cache = RepoMapCache(tmp_path)

        # Make the cache file read-only to cause write error
        cache.cache_dir.chmod(0o444)

        # Should not raise
        graph = nx.DiGraph()
        cache.save_graph(graph)

        # Restore permissions
        cache.cache_dir.chmod(0o755)

    def test_load_graph_corrupted_file(self, tmp_path):
        """Test loading graph when cache file is corrupted."""
        cache = RepoMapCache(tmp_path)

        # Write corrupted data to the cache file
        cache.graph_cache_file.write_text("corrupted data")

        result = cache.load_graph()
        assert result is None

    def test_save_and_load_pagerank(self, tmp_path):
        """Test saving and loading PageRank scores."""
        cache = RepoMapCache(tmp_path)

        scores = {
            "fileA.py": 0.5,
            "fileB.py": 0.3,
            "fileC.py": 0.2
        }

        # Save scores
        cache.save_pagerank(scores)
        assert cache.pagerank_cache_file.exists()

        # Load scores
        loaded_scores = cache.load_pagerank()
        assert loaded_scores == scores

    def test_load_pagerank_not_exists(self, tmp_path):
        """Test loading PageRank when cache file doesn't exist."""
        cache = RepoMapCache(tmp_path)
        result = cache.load_pagerank()
        assert result is None

    def test_save_pagerank_exception_handling(self, tmp_path):
        """Test that exceptions during PageRank save are handled gracefully."""
        cache = RepoMapCache(tmp_path)

        # Make the cache file read-only
        cache.cache_dir.chmod(0o444)

        scores = {"fileA.py": 0.5}
        # Should not raise
        cache.save_pagerank(scores)

        # Restore permissions
        cache.cache_dir.chmod(0o755)

    def test_load_pagerank_corrupted_file(self, tmp_path):
        """Test loading PageRank when cache file is corrupted."""
        cache = RepoMapCache(tmp_path)

        # Write corrupted JSON
        cache.pagerank_cache_file.write_text("{ corrupted json")

        result = cache.load_pagerank()
        assert result is None

    def test_invalidate_all_caches(self, tmp_path):
        """Test invalidating all repo map caches."""
        cache = RepoMapCache(tmp_path)

        # Create both cache files
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        cache.save_graph(graph)

        scores = {"A": 0.5, "B": 0.5}
        cache.save_pagerank(scores)

        assert cache.graph_cache_file.exists()
        assert cache.pagerank_cache_file.exists()

        # Invalidate
        cache.invalidate()

        assert not cache.graph_cache_file.exists()
        assert not cache.pagerank_cache_file.exists()

    def test_invalidate_when_files_not_exist(self, tmp_path):
        """Test invalidating when cache files don't exist."""
        cache = RepoMapCache(tmp_path)

        # Should not raise
        cache.invalidate()

    def test_invalidate_exception_handling(self, tmp_path):
        """Test that exceptions during invalidation are handled gracefully."""
        cache = RepoMapCache(tmp_path)

        # Create a cache file
        cache.graph_cache_file.write_text("data")

        # Make parent directory read-only to prevent deletion
        cache.cache_dir.chmod(0o444)

        # Should not raise
        cache.invalidate()

        # Restore permissions
        cache.cache_dir.chmod(0o755)

    def test_multiple_cache_instances(self, tmp_path):
        """Test that multiple cache instances can share the same directory."""
        cache1 = RepoMapCache(tmp_path)
        cache2 = RepoMapCache(tmp_path)

        # Write with cache1
        scores = {"fileA.py": 0.8}
        cache1.save_pagerank(scores)

        # Read with cache2
        loaded_scores = cache2.load_pagerank()
        assert loaded_scores == scores

    def test_cache_persistence_across_instances(self, tmp_path):
        """Test that cache persists when creating new instances."""
        # First instance
        cache1 = RepoMapCache(tmp_path)
        graph = nx.DiGraph()
        graph.add_edge("X", "Y")
        cache1.save_graph(graph)

        # Second instance (simulates restart)
        cache2 = RepoMapCache(tmp_path)
        loaded_graph = cache2.load_graph()

        assert loaded_graph is not None
        assert loaded_graph.has_edge("X", "Y")
