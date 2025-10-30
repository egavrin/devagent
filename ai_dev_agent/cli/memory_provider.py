"""Memory context provider - extracted from ContextEnhancer."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import memory system if available
try:
    from ai_dev_agent.memory import MemoryDistiller, MemoryStore

    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    MEMORY_SYSTEM_AVAILABLE = False
    logger.debug("Memory system not available")


class MemoryProvider:
    """Provides memory-related context functionality."""

    def __init__(self, workspace: Path, enable_memory: bool = True):
        self.workspace = workspace
        self._memory_store = None

        if MEMORY_SYSTEM_AVAILABLE and enable_memory:
            try:
                # Use project-scoped storage: <workspace>/.devagent/memory/
                memory_path = self.workspace / ".devagent" / "memory" / "reasoning_bank.json"
                self._memory_store = MemoryStore(store_path=memory_path)
                logger.debug(
                    "Memory bank initialized with %d memories from %s",
                    len(self._memory_store._memories),
                    memory_path,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize memory store: {e}")
                self._memory_store = None

    def retrieve_relevant_memories(
        self, query: str, task_type: Optional[str] = None, limit: int = 5
    ) -> Optional[list[dict[str, Any]]]:
        """Retrieve relevant memories for a query.

        Args:
            query: The query to find relevant memories for
            task_type: Optional task type to filter memories
            limit: Maximum number of memories to retrieve

        Returns:
            List of memory dictionaries or None if memory system unavailable
        """
        if not self._memory_store:
            return None

        try:
            # Use search_similar method (correct API)
            results = self._memory_store.search_similar(
                query=query, task_type=task_type, limit=limit
            )

            if not results:
                return None

            # Convert memory objects to dictionaries
            memories = []
            for memory, similarity in results:
                memory_dict = {
                    "id": memory.memory_id,
                    "content": memory.content if hasattr(memory, "content") else memory.title,
                    "metadata": {
                        "effectiveness": memory.effectiveness_score,
                        "task_type": memory.task_type,
                        "similarity": similarity,
                    },
                }
                memories.append(memory_dict)

            return memories

        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
            return None

    def store_memory(
        self,
        query: str,
        response: str,
        task_type: Optional[str] = None,
        success: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store a new memory from query-response interaction.

        Args:
            query: The original query
            response: The response to the query
            task_type: Type of task for categorization
            success: Whether the interaction was successful
            metadata: Additional metadata to store

        Returns:
            Memory ID if stored successfully, None otherwise
        """
        if not self._memory_store or not MEMORY_SYSTEM_AVAILABLE:
            return None

        try:
            # Use MemoryDistiller to extract key reasoning
            distiller = MemoryDistiller()

            # Create a simple session format for distillation
            session_id = f"session_{hash(query)}"
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]

            # Use distill_from_session which is the correct API
            memory = distiller.distill_from_session(
                session_id=session_id, messages=messages, metadata=metadata
            )

            if memory:
                # Add the memory to store
                memory_id = self._memory_store.add_memory(memory)
                logger.debug(f"Stored memory {memory_id}: {memory.title}")
                return memory_id

        except Exception as e:
            logger.warning(f"Failed to distill/store memory: {e}")

        return None

    def format_memories_for_context(self, memories: list[dict[str, Any]]) -> str:
        """Format memories into a context string.

        Args:
            memories: List of memory dictionaries

        Returns:
            Formatted string for context inclusion
        """
        if not memories:
            return ""

        lines = ["## Relevant Past Experiences", ""]

        for i, memory in enumerate(memories, 1):
            content = memory.get("content", "")
            metadata = memory.get("metadata", {})
            effectiveness = metadata.get("effectiveness", 0.5)

            lines.append(f"{i}. {content}")
            if effectiveness > 0.7:
                lines.append("   (Highly effective approach)")
            elif effectiveness < 0.3:
                lines.append("   (Approach had limited success)")
            lines.append("")

        return "\n".join(lines)
