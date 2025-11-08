"""Short-term memory: process-scoped storage with LRU eviction.

This module provides ephemeral storage for data that only needs to live
during the current devagent process (sessions, plans, tasks).

For persistent cross-session memory, see ai_dev_agent.memory.store.MemoryStore.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class ShortTermMemory(Generic[T]):
    """Thread-safe short-term storage with LRU eviction.

    Designed for process-scoped data that should be cleared when devagent exits:
    - Conversation sessions (SessionManager)
    - Plan execution state (StateStore)
    - Background task queue (TaskQueue)

    Features:
    - All operations are thread-safe
    - LRU eviction when max_entries exceeded
    - Optional TTL-based expiration
    - No file persistence (use MemoryStore for that)

    Example:
        >>> # Store conversation sessions (max 100, expire after 30min)
        >>> sessions = ShortTermMemory[Session](
        ...     max_entries=100,
        ...     ttl=timedelta(minutes=30)
        ... )
        >>> sessions.set("session-1", session_obj)
        >>> sessions.get("session-1")  # Marks as recently used
    """

    def __init__(
        self,
        max_entries: int | None = None,
        ttl: timedelta | None = None,
    ):
        """Initialize short-term memory.

        Args:
            max_entries: Max entries before LRU eviction (None = unlimited)
            ttl: Time-to-live for entries (None = no expiration)
        """
        self._data: OrderedDict[str, T] = OrderedDict()  # LRU ordering
        self._lock = RLock()
        self._max_entries = max_entries
        self._ttl = ttl
        self._access_times: dict[str, datetime] = {}
        self._usage_counts: dict[str, int] = {}

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get value, marking as recently used.

        Args:
            key: Key to retrieve
            default: Default if key not found

        Returns:
            Value if found, else default
        """
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)  # Mark as recently used
                self._access_times[key] = datetime.now()
                self._usage_counts[key] = self._usage_counts.get(key, 0) + 1
                return self._data[key]
            return default

    def set(self, key: str, value: T) -> None:
        """Set value, evicting LRU if at capacity.

        Args:
            key: Key to set
            value: Value to store
        """
        with self._lock:
            if key in self._data:
                # Update existing
                self._data[key] = value
                self._data.move_to_end(key)
            else:
                # Check capacity before adding new key
                if self._max_entries and len(self._data) >= self._max_entries:
                    # Evict least recently used (first in OrderedDict)
                    lru_key = next(iter(self._data))
                    self.delete(lru_key)

                self._data[key] = value

            self._access_times[key] = datetime.now()
            self._usage_counts[key] = self._usage_counts.get(key, 0) + 1

    def delete(self, key: str) -> bool:
        """Delete key and metadata.

        Args:
            key: Key to delete

        Returns:
            True if key existed
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._access_times.pop(key, None)
                self._usage_counts.pop(key, None)
                return True
            return False

    def pop(self, key: str, default: T | None = None) -> T | None:
        """Remove key and return its value.

        Args:
            key: Key to remove
            default: Default if key not found

        Returns:
            Value if found, else default
        """
        with self._lock:
            if key in self._data:
                value = self._data[key]
                self.delete(key)
                return value
            return default

    def keys(self) -> list[str]:
        """Return all keys (ordered by LRU)."""
        with self._lock:
            return list(self._data.keys())

    def __len__(self) -> int:
        """Return number of entries."""
        with self._lock:
            return len(self._data)

    def __iter__(self):
        """Return iterator over keys."""
        with self._lock:
            return iter(list(self._data.keys()))

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._data.clear()
            self._access_times.clear()
            self._usage_counts.clear()

    def update(self, key: str, updater: Callable[[T], T]) -> T | None:
        """Atomic read-modify-write.

        Args:
            key: Key to update
            updater: Function to transform value

        Returns:
            Updated value if key exists, None otherwise
        """
        with self._lock:
            if key in self._data:
                self._data[key] = updater(self._data[key])
                self._data.move_to_end(key)
                self._access_times[key] = datetime.now()
                return self._data[key]
            return None

    def cleanup_expired(self) -> int:
        """Remove entries past TTL.

        Returns:
            Number of entries removed
        """
        if not self._ttl:
            return 0

        now = datetime.now()
        expired = []

        with self._lock:
            for key, access_time in self._access_times.items():
                if now - access_time > self._ttl:
                    expired.append(key)

            for key in expired:
                self.delete(key)

        return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            return {
                "total_entries": len(self._data),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl.total_seconds() if self._ttl else None,
                "total_accesses": sum(self._usage_counts.values()),
                "avg_usage_per_key": (
                    sum(self._usage_counts.values()) / len(self._usage_counts)
                    if self._usage_counts
                    else 0
                ),
            }


__all__ = ["ShortTermMemory"]
