"""Short-term memory storage for process-scoped data.

This module provides ephemeral storage for data that only needs to live
during the current devagent process (sessions, plans, tasks).

For persistent cross-session memory, see ai_dev_agent.memory.store.MemoryStore.
"""

from .short_term_memory import ShortTermMemory

__all__ = ["ShortTermMemory"]
