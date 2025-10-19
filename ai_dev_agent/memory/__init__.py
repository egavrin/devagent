"""Memory system for DevAgent - ReasoningBank pattern implementation.

This module implements a sophisticated memory system that:
1. Distills strategies and lessons from completed tasks
2. Stores them with embeddings for similarity search
3. Retrieves relevant memories for new tasks
4. Tracks effectiveness and learns continuously
"""

from .distiller import MemoryDistiller, Memory, Strategy, Lesson
from .store import MemoryStore
from .embeddings import EmbeddingGenerator

__all__ = [
    "MemoryDistiller",
    "Memory",
    "Strategy",
    "Lesson",
    "MemoryStore",
    "EmbeddingGenerator",
]