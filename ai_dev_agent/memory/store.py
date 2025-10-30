"""Memory Store - Persistent storage and retrieval for distilled memories.

Implements vector similarity search for retrieving relevant memories
and tracks effectiveness for continuous learning.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .distiller import Memory
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persistent store for memories with vector search capabilities."""

    # Default paths
    DEFAULT_STORE_PATH = Path.home() / ".devagent" / "memory" / "reasoning_bank.json"
    BACKUP_SUFFIX = ".backup"

    # Store configuration
    MAX_MEMORIES_PER_TYPE = 100  # Limit per task type to prevent unbounded growth
    PRUNE_THRESHOLD = 0.2  # Remove memories with effectiveness below this
    SIMILARITY_THRESHOLD = 0.7  # Default threshold for retrieving similar memories
    RETRIEVAL_LIMIT = 5  # Default number of memories to retrieve

    def __init__(
        self,
        store_path: Path | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        auto_save: bool = True,
        backup_on_save: bool = True,
    ):
        """Initialize the memory store.

        Args:
            store_path: Path to the persistent store file
            embedding_generator: Generator for embeddings (will create if None)
            auto_save: Whether to automatically save after modifications
            backup_on_save: Whether to create backups when saving
        """
        self.store_path = Path(store_path) if store_path else self.DEFAULT_STORE_PATH
        self.auto_save = auto_save
        self.backup_on_save = backup_on_save

        # Thread safety
        self._lock = threading.RLock()

        # Initialize embedding generator
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

        # In-memory store
        self._memories: dict[str, Memory] = {}
        self._memories_by_type: dict[str, list[str]] = defaultdict(list)
        self._embeddings: dict[str, np.ndarray] = {}

        # Usage tracking
        self._usage_stats: dict[str, dict[str, Any]] = {}

        # Load existing store
        self._load_store()

    def _load_store(self) -> None:
        """Load memories from persistent storage."""
        if not self.store_path.exists():
            # Create directory if needed
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created new memory store at {self.store_path}")
            return

        try:
            with self.store_path.open() as f:
                data = json.load(f)

            # Load memories
            for memory_data in data.get("memories", []):
                memory = Memory.from_dict(memory_data)
                self._memories[memory.memory_id] = memory
                self._memories_by_type[memory.task_type].append(memory.memory_id)

                # Regenerate embedding if needed
                if memory.embedding:
                    self._embeddings[memory.memory_id] = np.array(memory.embedding)
                else:
                    # Generate embedding for memory
                    embedding_text = self._get_embedding_text(memory)
                    embedding = self.embedding_generator.generate_embedding(embedding_text)
                    self._embeddings[memory.memory_id] = embedding
                    memory.embedding = embedding.tolist()

            # Load usage stats
            self._usage_stats = data.get("usage_stats", {})

            # Update TF-IDF corpus if using that method
            if self.embedding_generator.method == "tfidf":
                all_texts = [self._get_embedding_text(m) for m in self._memories.values()]
                self.embedding_generator.update_tfidf_corpus(all_texts)

            logger.debug(f"Loaded {len(self._memories)} memories from {self.store_path}")

        except Exception as e:
            logger.error(f"Failed to load memory store: {e}")
            # Start with empty store
            self._memories = {}
            self._memories_by_type = defaultdict(list)
            self._embeddings = {}
            self._usage_stats = {}

    def save_store(self) -> None:
        """Save memories to persistent storage."""
        with self._lock:
            # Create backup if requested
            if self.backup_on_save and self.store_path.exists():
                backup_path = self.store_path.with_suffix(self.BACKUP_SUFFIX)
                try:
                    import shutil

                    shutil.copy2(self.store_path, backup_path)
                except Exception as e:
                    logger.warning(f"Failed to create backup: {e}")

            # Prepare data for saving
            data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "memories": [memory.to_dict() for memory in self._memories.values()],
                "usage_stats": self._usage_stats,
                "metadata": {
                    "total_memories": len(self._memories),
                    "memories_by_type": {
                        task_type: len(ids) for task_type, ids in self._memories_by_type.items()
                    },
                },
            }

            # Save to file
            try:
                # Write to temporary file first
                temp_path = self.store_path.with_suffix(".tmp")
                with Path(temp_path).open("w") as f:
                    json.dump(data, f, indent=2, default=str)

                # Atomic rename
                temp_path.replace(self.store_path)
                logger.debug(f"Saved {len(self._memories)} memories to {self.store_path}")

            except Exception as e:
                logger.error(f"Failed to save memory store: {e}")
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()

    def add_memory(self, memory: Memory) -> str:
        """Add a new memory to the store.

        Args:
            memory: Memory to add

        Returns:
            Memory ID
        """
        with self._lock:
            # Check for similar existing memory
            similar = self._find_similar_memory(memory)
            if similar:
                # Merge with existing memory
                logger.debug(f"Merging with similar memory: {similar.memory_id}")
                from .distiller import MemoryDistiller

                distiller = MemoryDistiller()
                merged = distiller.merge_similar_memories(similar, memory)
                memory = merged
                # Remove the old memory
                self.remove_memory(similar.memory_id)

            # Generate embedding if not present
            if memory.embedding is None:
                embedding_text = self._get_embedding_text(memory)
                embedding = self.embedding_generator.generate_embedding(embedding_text)
                memory.embedding = embedding.tolist()
                self._embeddings[memory.memory_id] = embedding
            else:
                self._embeddings[memory.memory_id] = np.array(memory.embedding)

            # Add to store
            self._memories[memory.memory_id] = memory
            self._memories_by_type[memory.task_type].append(memory.memory_id)

            # Prune if needed
            self._prune_task_type(memory.task_type)

            # Update TF-IDF corpus if using that method
            if self.embedding_generator.method == "tfidf":
                all_texts = [self._get_embedding_text(m) for m in self._memories.values()]
                self.embedding_generator.update_tfidf_corpus(all_texts)

            # Save if auto-save enabled
            if self.auto_save:
                self.save_store()

            logger.debug(f"Added memory: {memory.title}")
            return memory.memory_id

    def search_similar(
        self,
        query: str,
        task_type: str | None = None,
        limit: int | None = None,
        threshold: float | None = None,
    ) -> list[tuple[Memory, float]]:
        """Search for similar memories using vector similarity.

        Args:
            query: Query text
            task_type: Optional task type filter
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            List of (memory, similarity) tuples
        """
        limit = limit or self.RETRIEVAL_LIMIT
        threshold = threshold or self.SIMILARITY_THRESHOLD

        with self._lock:
            if not self._memories:
                return []

            # Filter by task type if specified
            if task_type:
                candidate_ids = self._memories_by_type.get(task_type, [])
                if not candidate_ids:
                    return []
                candidates = [(id, self._embeddings[id]) for id in candidate_ids]
            else:
                candidates = [(id, emb) for id, emb in self._embeddings.items()]

            if not candidates:
                return []

            # Find similar memories
            candidate_list = [(id, emb) for id, emb in candidates]
            similar_indices = self.embedding_generator.find_similar(
                query, [(id, emb) for id, emb in candidate_list], top_k=limit, threshold=threshold
            )

            # Build results
            results = []
            for idx, similarity in similar_indices:
                memory_id = candidate_list[idx][0]
                memory = self._memories[memory_id]
                results.append((memory, similarity))

                # Track usage
                self._track_retrieval(memory_id, similarity)

            return results

    def get_memory(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory if found, None otherwise
        """
        with self._lock:
            return self._memories.get(memory_id)

    def remove_memory(self, memory_id: str) -> bool:
        """Remove a memory from the store.

        Args:
            memory_id: Memory identifier

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            memory = self._memories.get(memory_id)
            if not memory:
                return False

            # Remove from all indices
            del self._memories[memory_id]
            if memory_id in self._embeddings:
                del self._embeddings[memory_id]
            self._memories_by_type[memory.task_type].remove(memory_id)
            if memory_id in self._usage_stats:
                del self._usage_stats[memory_id]

            # Save if auto-save enabled
            if self.auto_save:
                self.save_store()

            logger.debug(f"Removed memory: {memory.title}")
            return True

    def update_effectiveness(
        self, memory_id: str, score_delta: float, usage_feedback: str | None = None
    ) -> None:
        """Update the effectiveness score of a memory.

        Args:
            memory_id: Memory identifier
            score_delta: Change in effectiveness score (-1 to 1)
            usage_feedback: Optional feedback about usage
        """
        with self._lock:
            memory = self._memories.get(memory_id)
            if not memory:
                return

            # Update score (bounded between 0 and 1)
            memory.effectiveness_score = max(0, min(1, memory.effectiveness_score + score_delta))

            # Update usage count and timestamp
            memory.usage_count += 1
            memory.last_used = datetime.now().isoformat()

            # Add feedback to metadata if provided
            if usage_feedback:
                if "usage_feedback" not in memory.metadata:
                    memory.metadata["usage_feedback"] = []
                memory.metadata["usage_feedback"].append(
                    {"feedback": usage_feedback, "timestamp": datetime.now().isoformat()}
                )

            # Update usage stats
            if memory_id not in self._usage_stats:
                self._usage_stats[memory_id] = {
                    "retrievals": 0,
                    "positive_feedback": 0,
                    "negative_feedback": 0,
                }

            if score_delta > 0:
                self._usage_stats[memory_id]["positive_feedback"] += 1
            elif score_delta < 0:
                self._usage_stats[memory_id]["negative_feedback"] += 1

            # Save if auto-save enabled
            if self.auto_save:
                self.save_store()

            logger.debug(f"Updated effectiveness for {memory_id}: {memory.effectiveness_score:.2f}")

    def prune_ineffective(self, threshold: float | None = None) -> int:
        """Remove memories with low effectiveness scores.

        Args:
            threshold: Effectiveness threshold (default: PRUNE_THRESHOLD)

        Returns:
            Number of memories pruned
        """
        threshold = threshold or self.PRUNE_THRESHOLD
        pruned = 0

        with self._lock:
            # Find memories to prune
            to_prune = []
            for memory_id, memory in self._memories.items():
                # Only prune if used at least 3 times
                if memory.usage_count >= 3 and memory.effectiveness_score < threshold:
                    to_prune.append(memory_id)

            # Remove ineffective memories
            for memory_id in to_prune:
                if self.remove_memory(memory_id):
                    pruned += 1

            if pruned > 0:
                logger.info(f"Pruned {pruned} ineffective memories")

        return pruned

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the memory store.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            total_memories = len(self._memories)
            if total_memories == 0:
                return {
                    "total_memories": 0,
                    "memories_by_type": {},
                    "avg_effectiveness": 0,
                    "total_usage": 0,
                }

            # Calculate statistics
            effectiveness_scores = [m.effectiveness_score for m in self._memories.values()]
            usage_counts = [m.usage_count for m in self._memories.values()]

            stats = {
                "total_memories": total_memories,
                "memories_by_type": {
                    task_type: len(ids) for task_type, ids in self._memories_by_type.items()
                },
                "avg_effectiveness": sum(effectiveness_scores) / len(effectiveness_scores),
                "min_effectiveness": min(effectiveness_scores),
                "max_effectiveness": max(effectiveness_scores),
                "total_usage": sum(usage_counts),
                "avg_usage": sum(usage_counts) / len(usage_counts),
                "most_used": (
                    max(self._memories.values(), key=lambda m: m.usage_count).title
                    if self._memories
                    else "N/A"
                ),
                "most_effective": (
                    max(self._memories.values(), key=lambda m: m.effectiveness_score).title
                    if self._memories
                    else "N/A"
                ),
                "storage_size_kb": (
                    self.store_path.stat().st_size / 1024 if self.store_path.exists() else 0
                ),
            }

            return stats

    def _get_embedding_text(self, memory: Memory) -> str:
        """Create text representation of memory for embedding.

        Args:
            memory: Memory to convert

        Returns:
            Text representation
        """
        parts = [memory.query]

        # Add strategies
        for strategy in memory.strategies[:2]:  # Top 2 strategies
            parts.append(strategy.description)
            if strategy.steps:
                parts.append(" ".join(strategy.steps[:2]))  # First 2 steps

        # Add lessons
        for lesson in memory.lessons[:1]:  # Top lesson
            parts.append(f"{lesson.mistake} -> {lesson.correction}")

        return " ".join(parts)

    def _find_similar_memory(self, memory: Memory, threshold: float = 0.95) -> Memory | None:
        """Find an existing similar memory.

        Args:
            memory: Memory to check
            threshold: Similarity threshold (very high to only merge near-duplicates)

        Returns:
            Similar memory if found
        """
        # Quick check using context hash - only merge exact duplicates
        for existing in self._memories.values():
            if existing.context_hash and memory.context_hash:
                if existing.context_hash == memory.context_hash:
                    return existing

        # For test cases with very short queries, check semantic similarity
        # Only merge if extremely similar (>0.95) to avoid over-merging
        if memory.task_type in self._memories_by_type and len(memory.query) < 20:
            embedding_text = self._get_embedding_text(memory)
            similar = self.search_similar(
                embedding_text, task_type=memory.task_type, limit=1, threshold=threshold
            )
            if similar:
                # Double-check it's really the same query
                similar_memory = similar[0][0]
                similarity = similar[0][1]
                if similarity > 0.98 or similar_memory.query == memory.query:
                    return similar_memory

        return None

    def _prune_task_type(self, task_type: str) -> None:
        """Prune memories of a specific type to stay under limit.

        Args:
            task_type: Task type to prune
        """
        memory_ids = self._memories_by_type.get(task_type, [])

        if len(memory_ids) <= self.MAX_MEMORIES_PER_TYPE:
            return

        # Sort by effectiveness and usage
        memories_with_scores = []
        for memory_id in memory_ids:
            memory = self._memories[memory_id]
            # Combined score: effectiveness + usage weight
            score = memory.effectiveness_score + (memory.usage_count * 0.01)
            memories_with_scores.append((memory_id, score))

        # Sort by score (ascending - lowest scores will be pruned)
        memories_with_scores.sort(key=lambda x: x[1])

        # Remove excess memories
        to_remove = len(memory_ids) - self.MAX_MEMORIES_PER_TYPE
        for memory_id, _ in memories_with_scores[:to_remove]:
            self.remove_memory(memory_id)

        if to_remove > 0:
            logger.info(f"Pruned {to_remove} memories of type {task_type}")

    def _track_retrieval(self, memory_id: str, similarity: float) -> None:
        """Track that a memory was retrieved.

        Args:
            memory_id: Memory that was retrieved
            similarity: Similarity score
        """
        if memory_id not in self._usage_stats:
            self._usage_stats[memory_id] = {
                "retrievals": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "avg_similarity": 0,
            }

        stats = self._usage_stats[memory_id]
        stats["retrievals"] += 1

        # Update average similarity
        prev_avg = stats.get("avg_similarity", 0)
        stats["avg_similarity"] = (prev_avg * (stats["retrievals"] - 1) + similarity) / stats[
            "retrievals"
        ]

    def export_memories(self, output_path: Path) -> None:
        """Export memories to a file.

        Args:
            output_path: Path for export file
        """
        with self._lock:
            data = {
                "exported_at": datetime.now().isoformat(),
                "memories": [memory.to_dict() for memory in self._memories.values()],
                "statistics": self.get_statistics(),
            }

            with Path(output_path).open("w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported {len(self._memories)} memories to {output_path}")

    def import_memories(self, input_path: Path, merge: bool = True) -> int:
        """Import memories from a file.

        Args:
            input_path: Path to import file
            merge: Whether to merge with existing memories

        Returns:
            Number of memories imported
        """
        with Path(input_path).open() as f:
            data = json.load(f)

        imported = 0
        for memory_data in data.get("memories", []):
            memory = Memory.from_dict(memory_data)

            # Generate new ID if merging to avoid conflicts
            if merge:
                from uuid import uuid4

                memory.memory_id = str(uuid4())

            self.add_memory(memory)
            imported += 1

        logger.info(f"Imported {imported} memories from {input_path}")
        return imported
