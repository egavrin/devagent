"""Caching utilities for symbol extraction and repository mapping."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from diskcache import Cache as DiskCache

LOGGER = logging.getLogger(__name__)


class SymbolCache:
    """Persistent cache for tree-sitter symbol extraction with mtime tracking."""

    def __init__(self, root: Path, cache_dir: Optional[Path] = None):
        """Initialize symbol cache.

        Args:
            root: Repository root directory
            cache_dir: Custom cache directory (default: root/.devagent/symbol_cache)
        """
        self.root = Path(root)
        self._fallback_cache: Dict[str, Any] = {}

        if cache_dir is None:
            cache_dir = self.root / ".devagent" / "symbol_cache"

        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use diskcache for persistent storage
            self.cache = DiskCache(str(cache_dir), size_limit=100 * 1024 * 1024)  # 100MB limit
        except Exception as e:
            LOGGER.warning("Failed to initialize disk cache, using in-memory cache: %s", e)
            self.cache = None

    def get_symbols(self, file_path: Path) -> Optional[List[Any]]:
        """Get cached symbols for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of symbols if cached and valid, None otherwise
        """
        if not file_path.exists():
            return None

        try:
            mtime = file_path.stat().st_mtime
            cache_key = str(file_path)

            if self.cache:
                cached = self.cache.get(cache_key)
            else:
                cached = self._fallback_cache.get(cache_key)

            if cached and cached.get("mtime") == mtime:
                return cached.get("symbols")

        except Exception as e:
            LOGGER.debug("Cache read failed for %s: %s", file_path, e)

        return None

    def set_symbols(self, file_path: Path, symbols: List[Any]) -> None:
        """Cache symbols for a file.

        Args:
            file_path: Path to the file
            symbols: List of symbols to cache
        """
        if not file_path.exists():
            return

        try:
            mtime = file_path.stat().st_mtime
            cache_key = str(file_path)

            cache_value = {
                "mtime": mtime,
                "symbols": symbols
            }

            if self.cache:
                self.cache[cache_key] = cache_value
            else:
                self._fallback_cache[cache_key] = cache_value

        except Exception as e:
            LOGGER.debug("Cache write failed for %s: %s", file_path, e)

    def invalidate(self, file_path: Path) -> None:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to the file
        """
        cache_key = str(file_path)

        try:
            if self.cache:
                del self.cache[cache_key]
            elif cache_key in self._fallback_cache:
                del self._fallback_cache[cache_key]
        except KeyError:
            pass  # Already not in cache
        except Exception as e:
            LOGGER.debug("Cache invalidation failed for %s: %s", file_path, e)

    def clear(self) -> None:
        """Clear all cached data."""
        try:
            if self.cache:
                self.cache.clear()
            else:
                self._fallback_cache.clear()
        except Exception as e:
            LOGGER.warning("Failed to clear cache: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if self.cache:
            return {
                "type": "disk",
                "size": len(self.cache),
                "hits": getattr(self.cache, "hits", 0),
                "misses": getattr(self.cache, "misses", 0)
            }
        else:
            return {
                "type": "memory",
                "size": len(self._fallback_cache)
            }


class RepoMapCache:
    """Cache for repository mapping and PageRank results."""

    def __init__(self, root: Path):
        """Initialize repo map cache.

        Args:
            root: Repository root directory
        """
        self.root = Path(root)
        self.cache_dir = self.root / ".devagent" / "repo_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.graph_cache_file = self.cache_dir / "graph.pkl"
        self.pagerank_cache_file = self.cache_dir / "pagerank.json"

    def save_graph(self, graph: Any) -> None:
        """Save NetworkX graph to cache.

        Args:
            graph: NetworkX graph to save
        """
        try:
            with open(self.graph_cache_file, "wb") as f:
                pickle.dump(graph, f)
        except Exception as e:
            LOGGER.warning("Failed to save graph cache: %s", e)

    def load_graph(self) -> Optional[Any]:
        """Load NetworkX graph from cache.

        Returns:
            Cached graph or None if not available
        """
        if not self.graph_cache_file.exists():
            return None

        try:
            with open(self.graph_cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            LOGGER.warning("Failed to load graph cache: %s", e)
            return None

    def save_pagerank(self, scores: Dict[str, float]) -> None:
        """Save PageRank scores to cache.

        Args:
            scores: Dictionary of file paths to PageRank scores
        """
        try:
            with open(self.pagerank_cache_file, "w") as f:
                json.dump(scores, f, indent=2)
        except Exception as e:
            LOGGER.warning("Failed to save PageRank cache: %s", e)

    def load_pagerank(self) -> Optional[Dict[str, float]]:
        """Load PageRank scores from cache.

        Returns:
            Cached scores or None if not available
        """
        if not self.pagerank_cache_file.exists():
            return None

        try:
            with open(self.pagerank_cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            LOGGER.warning("Failed to load PageRank cache: %s", e)
            return None

    def invalidate(self) -> None:
        """Invalidate all repo map caches."""
        try:
            if self.graph_cache_file.exists():
                self.graph_cache_file.unlink()
            if self.pagerank_cache_file.exists():
                self.pagerank_cache_file.unlink()
        except Exception as e:
            LOGGER.warning("Failed to invalidate repo map cache: %s", e)