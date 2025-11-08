"""Comprehensive tests for ShortTermMemory."""

import threading
import time
from datetime import timedelta

import pytest

from ai_dev_agent.core.storage.short_term_memory import ShortTermMemory


class TestBasicOperations:
    """Test basic CRUD operations."""

    def test_get_set(self):
        """Test basic get/set operations."""
        memory = ShortTermMemory[str]()

        # Set and retrieve
        memory.set("key1", "value1")
        assert memory.get("key1") == "value1"

        # Get non-existent key
        assert memory.get("missing") is None
        assert memory.get("missing", "default") == "default"

    def test_delete(self):
        """Test delete operation."""
        memory = ShortTermMemory[str]()

        memory.set("key1", "value1")
        assert memory.delete("key1") is True
        assert memory.get("key1") is None

        # Delete non-existent key
        assert memory.delete("missing") is False

    def test_pop(self):
        """Test pop operation."""
        memory = ShortTermMemory[str]()

        memory.set("key1", "value1")
        assert memory.pop("key1") == "value1"
        assert memory.get("key1") is None

        # Pop non-existent key
        assert memory.pop("missing") is None
        assert memory.pop("missing", "default") == "default"

    def test_keys(self):
        """Test keys listing."""
        memory = ShortTermMemory[str]()

        memory.set("a", "1")
        memory.set("b", "2")
        memory.set("c", "3")

        keys = memory.keys()
        assert set(keys) == {"a", "b", "c"}

    def test_clear(self):
        """Test clearing all entries."""
        memory = ShortTermMemory[str]()

        memory.set("a", "1")
        memory.set("b", "2")

        memory.clear()

        assert memory.keys() == []
        assert memory.get("a") is None

    def test_update_existing_key(self):
        """Test updating an existing key."""
        memory = ShortTermMemory[dict]()

        memory.set("data", {"count": 0})

        result = memory.update("data", lambda d: {**d, "count": d["count"] + 1})

        assert result == {"count": 1}
        assert memory.get("data") == {"count": 1}

    def test_update_missing_key(self):
        """Test updating a non-existent key."""
        memory = ShortTermMemory[dict]()

        result = memory.update("missing", lambda d: {"updated": True})

        assert result is None
        assert memory.get("missing") is None


class TestLRUEviction:
    """Test LRU eviction when max_entries exceeded."""

    def test_lru_eviction_basic(self):
        """Test LRU eviction with max_entries=3."""
        memory = ShortTermMemory[str](max_entries=3)

        memory.set("a", "first")
        memory.set("b", "second")
        memory.set("c", "third")

        # All three should exist
        assert memory.get("a") == "first"
        assert memory.get("b") == "second"
        assert memory.get("c") == "third"

        # Add fourth entry - should evict "a" (LRU)
        memory.set("d", "fourth")

        assert memory.get("a") is None  # Evicted
        assert memory.get("b") == "second"
        assert memory.get("c") == "third"
        assert memory.get("d") == "fourth"

    def test_lru_get_updates_access(self):
        """Test that get() marks entry as recently used."""
        memory = ShortTermMemory[str](max_entries=3)

        memory.set("a", "first")
        memory.set("b", "second")
        memory.set("c", "third")

        # Access "a" to make it most recently used
        memory.get("a")

        # Add "d" - should evict "b" (now LRU)
        memory.set("d", "fourth")

        assert memory.get("a") == "first"  # Not evicted
        assert memory.get("b") is None  # Evicted
        assert memory.get("c") == "third"
        assert memory.get("d") == "fourth"

    def test_lru_set_existing_updates_access(self):
        """Test that setting existing key marks it as recently used."""
        memory = ShortTermMemory[str](max_entries=3)

        memory.set("a", "first")
        memory.set("b", "second")
        memory.set("c", "third")

        # Update "a" (makes it most recently used)
        memory.set("a", "updated_first")

        # Add "d" - should evict "b" (LRU)
        memory.set("d", "fourth")

        assert memory.get("a") == "updated_first"  # Not evicted
        assert memory.get("b") is None  # Evicted
        assert memory.get("c") == "third"
        assert memory.get("d") == "fourth"

    def test_lru_with_no_max(self):
        """Test that no eviction happens without max_entries."""
        memory = ShortTermMemory[str]()  # No max

        for i in range(100):
            memory.set(f"key{i}", f"value{i}")

        # All 100 should still exist
        assert len(memory.keys()) == 100


class TestTTLExpiration:
    """Test TTL-based expiration."""

    def test_ttl_cleanup(self):
        """Test TTL cleanup removes expired entries."""
        memory = ShortTermMemory[str](ttl=timedelta(seconds=0.5))

        memory.set("key1", "value1")
        memory.set("key2", "value2")

        # Both should exist initially
        assert memory.get("key1") == "value1"
        assert memory.get("key2") == "value2"

        # Wait for expiration
        time.sleep(0.6)

        # Cleanup should remove both
        removed = memory.cleanup_expired()
        assert removed == 2

        assert memory.get("key1") is None
        assert memory.get("key2") is None

    def test_ttl_get_extends_lifetime(self):
        """Test that get() updates access time."""
        memory = ShortTermMemory[str](ttl=timedelta(seconds=0.5))

        memory.set("key1", "value1")
        memory.set("key2", "value2")

        # Sleep for 0.3s
        time.sleep(0.3)

        # Access key1 (refreshes its access time)
        memory.get("key1")

        # Sleep another 0.3s (total 0.6s for key2, 0.3s for key1)
        time.sleep(0.3)

        # Cleanup should only remove key2
        removed = memory.cleanup_expired()
        assert removed == 1

        assert memory.get("key1") == "value1"  # Still alive
        assert memory.get("key2") is None  # Expired

    def test_no_ttl(self):
        """Test that cleanup does nothing without TTL."""
        memory = ShortTermMemory[str]()  # No TTL

        memory.set("key1", "value1")

        # Sleep and cleanup
        time.sleep(0.1)
        removed = memory.cleanup_expired()

        assert removed == 0
        assert memory.get("key1") == "value1"


class TestThreadSafety:
    """Test thread-safe concurrent access."""

    def test_concurrent_set_get(self):
        """Test concurrent set/get from multiple threads."""
        memory = ShortTermMemory[int]()
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(50):
                    key = f"key{worker_id}_{i}"
                    memory.set(key, worker_id * 1000 + i)

                    # Verify immediately
                    value = memory.get(key)
                    if value != worker_id * 1000 + i:
                        errors.append(
                            f"Worker {worker_id}: Expected {worker_id * 1000 + i}, got {value}"
                        )
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"

    def test_concurrent_update(self):
        """Test atomic update with concurrent access."""
        memory = ShortTermMemory[dict]()
        memory.set("counter", {"value": 0})

        def increment():
            for _ in range(100):
                memory.update("counter", lambda d: {"value": d["value"] + 1})

        threads = [threading.Thread(target=increment) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should be 1000 (100 * 10) if atomic
        assert memory.get("counter")["value"] == 1000

    def test_concurrent_eviction(self):
        """Test LRU eviction with concurrent access."""
        memory = ShortTermMemory[int](max_entries=50)
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(100):
                    key = f"w{worker_id}_k{i}"
                    memory.set(key, i)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Eviction errors: {errors}"

        # Should have exactly max_entries
        assert len(memory.keys()) == 50


class TestStatistics:
    """Test statistics tracking."""

    def test_usage_counts(self):
        """Test that usage counts are tracked."""
        memory = ShortTermMemory[str]()

        memory.set("key1", "value1")  # +1 access
        memory.get("key1")  # +1 access
        memory.get("key1")  # +1 access

        stats = memory.get_stats()

        assert stats["total_entries"] == 1
        assert stats["total_accesses"] == 3
        assert stats["avg_usage_per_key"] == 3.0

    def test_stats_structure(self):
        """Test statistics structure."""
        memory = ShortTermMemory[str](max_entries=100, ttl=timedelta(minutes=30))

        memory.set("a", "1")
        memory.set("b", "2")

        stats = memory.get_stats()

        assert stats["total_entries"] == 2
        assert stats["max_entries"] == 100
        assert stats["ttl_seconds"] == 1800.0
        assert "total_accesses" in stats
        assert "avg_usage_per_key" in stats

    def test_stats_empty(self):
        """Test statistics on empty memory."""
        memory = ShortTermMemory[str]()

        stats = memory.get_stats()

        assert stats["total_entries"] == 0
        assert stats["total_accesses"] == 0
        assert stats["avg_usage_per_key"] == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_key(self):
        """Test operations with empty key."""
        memory = ShortTermMemory[str]()

        memory.set("", "empty_key_value")
        assert memory.get("") == "empty_key_value"
        assert memory.delete("") is True

    def test_none_value(self):
        """Test storing None as a value."""
        from typing import Optional

        memory = ShortTermMemory[Optional[str]]()

        memory.set("key", None)
        # get() returns None for missing keys too, so check with keys()
        assert "key" in memory.keys()

    def test_large_values(self):
        """Test storing large values."""
        memory = ShortTermMemory[list]()

        large_list = list(range(10000))
        memory.set("large", large_list)

        retrieved = memory.get("large")
        assert retrieved == large_list

    def test_delete_during_iteration(self):
        """Test that deletion during cleanup works."""
        memory = ShortTermMemory[str](ttl=timedelta(seconds=0.1))

        for i in range(10):
            memory.set(f"key{i}", f"value{i}")

        time.sleep(0.15)

        # Should delete all 10
        removed = memory.cleanup_expired()
        assert removed == 10
        assert len(memory.keys()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
