"""Tests for the memory system (ReasoningBank pattern)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_dev_agent.memory import Memory, Strategy, Lesson, MemoryDistiller, MemoryStore, EmbeddingGenerator
from ai_dev_agent.providers.llm.base import Message


class TestMemoryDistiller:
    """Tests for the MemoryDistiller class."""

    def test_memory_creation(self):
        """Test basic memory creation."""
        memory = Memory(
            task_type="debugging",
            title="Fix authentication error",
            query="The login is failing with a TypeError",
            outcome="success"
        )

        assert memory.task_type == "debugging"
        assert memory.title == "Fix authentication error"
        assert memory.outcome == "success"
        assert memory.memory_id  # Should have auto-generated ID

    def test_strategy_creation(self):
        """Test strategy creation."""
        strategy = Strategy(
            description="Check for null values in auth handler",
            context="When dealing with TypeErrors in authentication",
            steps=["Locate auth handler", "Add null check", "Test login flow"],
            tools_used=["read", "write", "run"]
        )

        assert strategy.description == "Check for null values in auth handler"
        assert len(strategy.steps) == 3
        assert "read" in strategy.tools_used

    def test_lesson_creation(self):
        """Test lesson creation."""
        lesson = Lesson(
            mistake="Assumed user object always exists",
            context="In authentication middleware",
            correction="Always check if user is None before accessing properties",
            severity="major"
        )

        assert lesson.mistake == "Assumed user object always exists"
        assert lesson.severity == "major"

    def test_distill_from_session(self):
        """Test distilling memory from conversation messages."""
        distiller = MemoryDistiller()

        messages = [
            Message(role="user", content="Fix the bug in the authentication system"),
            Message(role="assistant", content="I'll fix the authentication bug. Let me first check the auth handler."),
            Message(role="assistant", content="Found the issue - there's a missing null check. Fixed it successfully."),
        ]

        memory = distiller.distill_from_session(
            session_id="test-session",
            messages=messages
        )

        assert memory.task_type == "debugging"
        assert "authentication" in memory.query.lower()
        assert memory.outcome in ["success", "partial", "unknown"]
        assert isinstance(memory.strategies, list)
        assert isinstance(memory.lessons, list)

    def test_task_type_identification(self):
        """Test identification of different task types."""
        distiller = MemoryDistiller()

        assert distiller._identify_task_type("fix the bug") == "debugging"
        assert distiller._identify_task_type("add a new feature") == "feature"
        assert distiller._identify_task_type("refactor this code") == "refactoring"
        assert distiller._identify_task_type("write tests for this") == "testing"
        assert distiller._identify_task_type("review the changes") == "review"
        assert distiller._identify_task_type("help me understand") == "general"

    def test_outcome_determination(self):
        """Test determining task outcome from messages."""
        distiller = MemoryDistiller()

        success_messages = [
            Message(role="assistant", content="Successfully fixed the issue"),
            Message(role="assistant", content="Tests are passing now"),
        ]
        assert distiller._determine_outcome(success_messages) == "success"

        failure_messages = [
            Message(role="assistant", content="Failed to resolve the error"),
            Message(role="assistant", content="The issue persists"),
        ]
        assert distiller._determine_outcome(failure_messages) == "failure"

        partial_messages = [
            Message(role="assistant", content="Fixed part of it but there's still an issue"),
        ]
        outcome = distiller._determine_outcome(partial_messages)
        assert outcome in ["partial", "failure"]

    def test_merge_similar_memories(self):
        """Test merging two similar memories."""
        distiller = MemoryDistiller()

        memory1 = Memory(
            task_type="debugging",
            title="Fix auth bug",
            query="Authentication fails",
            strategies=[Strategy("Check nulls", "Auth context", [], [])],
            lessons=[],
            outcome="success",
            usage_count=5,
            effectiveness_score=0.8
        )

        memory2 = Memory(
            task_type="debugging",
            title="Fix login error",
            query="Login throws error",
            strategies=[Strategy("Validate input", "Login flow", [], [])],
            lessons=[Lesson("Check user exists", "Auth", "Add null check", "major")],
            outcome="partial",
            usage_count=2  # Set usage count for proper testing
        )

        merged = distiller.merge_similar_memories(memory1, memory2)

        assert merged.task_type == "debugging"
        assert len(merged.strategies) == 2  # Both strategies kept
        assert len(merged.lessons) == 1  # Lesson from memory2
        assert merged.outcome == "success"  # Better outcome
        assert merged.usage_count == 7  # Combined (5 + 2)
        assert merged.effectiveness_score == 0.8  # Max score


class TestEmbeddingGenerator:
    """Tests for the EmbeddingGenerator class."""

    def test_tfidf_embeddings(self):
        """Test TF-IDF embedding generation."""
        generator = EmbeddingGenerator(method="tfidf")

        text1 = "Fix authentication error in login system"
        text2 = "Authentication bug in user login"
        text3 = "Refactor database connection pooling"

        # Generate embeddings
        emb1 = generator.generate_embedding(text1)
        emb2 = generator.generate_embedding(text2)
        emb3 = generator.generate_embedding(text3)

        # Check dimensions
        assert emb1.shape[0] == 384  # Default TF-IDF dimension
        assert emb2.shape[0] == 384
        assert emb3.shape[0] == 384

        # Similar texts should have higher similarity
        sim12 = generator.compute_similarity(emb1, emb2.reshape(1, -1))[0]
        sim13 = generator.compute_similarity(emb1, emb3.reshape(1, -1))[0]

        # text1 and text2 are more similar than text1 and text3
        assert sim12 > sim13

    def test_batch_embedding_generation(self):
        """Test batch embedding generation."""
        generator = EmbeddingGenerator(method="tfidf")

        texts = [
            "Fix authentication error",
            "Debug login system",
            "Refactor code structure"
        ]

        embeddings = generator.generate_embeddings(texts)

        assert embeddings.shape == (3, 384)

    def test_find_similar(self):
        """Test finding similar texts."""
        generator = EmbeddingGenerator(method="tfidf")

        candidates = [
            ("Fix auth bug", generator.generate_embedding("Fix auth bug")),
            ("Login error", generator.generate_embedding("Login error")),
            ("Database issue", generator.generate_embedding("Database issue")),
            ("Auth problem", generator.generate_embedding("Auth problem")),
        ]

        query = "Authentication failure"
        results = generator.find_similar(query, candidates, top_k=2, threshold=0.1)

        assert len(results) <= 2
        # Results should be sorted by similarity
        if len(results) > 1:
            assert results[0][1] >= results[1][1]

    def test_embedding_cache(self):
        """Test that embeddings are cached."""
        generator = EmbeddingGenerator(method="tfidf", cache_embeddings=True)

        text = "Test caching behavior"

        # Generate embedding twice
        emb1 = generator.generate_embedding(text)
        emb2 = generator.generate_embedding(text)

        # Should be the exact same object due to caching
        assert np.array_equal(emb1, emb2)

        # Clear cache
        generator.clear_cache()
        assert len(generator._embedding_cache) == 0


class TestMemoryStore:
    """Tests for the MemoryStore class."""

    def test_store_initialization(self):
        """Test memory store initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_memory.json"
            store = MemoryStore(store_path=store_path, auto_save=False)

            assert store.store_path == store_path
            assert len(store._memories) == 0

    def test_add_and_retrieve_memory(self):
        """Test adding and retrieving memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_memory.json"
            store = MemoryStore(store_path=store_path, auto_save=False)

            memory = Memory(
                task_type="debugging",
                title="Fix auth bug",
                query="Authentication fails",
                strategies=[],
                lessons=[],
                outcome="success"
            )

            # Add memory
            memory_id = store.add_memory(memory)
            assert memory_id == memory.memory_id

            # Retrieve by ID
            retrieved = store.get_memory(memory_id)
            assert retrieved.title == "Fix auth bug"

            # Search similar
            results = store.search_similar("authentication error", limit=1)
            assert len(results) > 0
            assert results[0][0].memory_id == memory_id

    def test_memory_effectiveness_tracking(self):
        """Test tracking memory effectiveness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(store_path=Path(tmpdir) / "test.json", auto_save=False)

            memory = Memory(
                task_type="debugging",
                title="Test memory",
                query="Test query",
                effectiveness_score=0.5
            )

            memory_id = store.add_memory(memory)

            # Update effectiveness positively
            store.update_effectiveness(memory_id, 0.2, "Very helpful")
            updated = store.get_memory(memory_id)
            assert updated.effectiveness_score == 0.7
            assert updated.usage_count == 1

            # Update negatively
            store.update_effectiveness(memory_id, -0.3, "Not helpful")
            updated = store.get_memory(memory_id)
            assert abs(updated.effectiveness_score - 0.4) < 0.01  # Handle floating point precision
            assert updated.usage_count == 2

    def test_prune_ineffective_memories(self):
        """Test pruning ineffective memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(store_path=Path(tmpdir) / "test.json", auto_save=False)

            # Add memories with different effectiveness
            good_memory = Memory(
                task_type="debugging",
                title="Good memory",
                query="Good",
                effectiveness_score=0.8,
                usage_count=5
            )
            bad_memory = Memory(
                task_type="debugging",
                title="Bad memory",
                query="Bad",
                effectiveness_score=0.1,
                usage_count=5
            )

            good_id = store.add_memory(good_memory)
            bad_id = store.add_memory(bad_memory)

            # Prune with threshold 0.3
            pruned = store.prune_ineffective(threshold=0.3)
            assert pruned == 1

            # Good memory should remain
            assert store.get_memory(good_id) is not None
            assert store.get_memory(bad_id) is None

    def test_memory_persistence(self):
        """Test saving and loading memories from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_memory.json"

            # Create and save memories
            store1 = MemoryStore(store_path=store_path, auto_save=False)
            memory = Memory(
                task_type="feature",
                title="Test feature",
                query="Add new feature",
                strategies=[Strategy("Plan first", "Always", ["Step 1"], [])],
                lessons=[Lesson("Don't rush", "Development", "Take time", "minor")],
                outcome="success"
            )
            memory_id = store1.add_memory(memory)
            store1.save_store()

            # Load in new store instance
            store2 = MemoryStore(store_path=store_path, auto_save=False)

            # Memory should be loaded
            loaded = store2.get_memory(memory_id)
            assert loaded is not None
            assert loaded.title == "Test feature"
            assert len(loaded.strategies) == 1
            assert len(loaded.lessons) == 1

    def test_memory_statistics(self):
        """Test getting memory store statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(store_path=Path(tmpdir) / "test.json", auto_save=False)

            # Add some memories
            for i in range(5):
                memory = Memory(
                    task_type="debugging" if i < 3 else "feature",
                    title=f"Memory {i}",
                    query=f"Query {i}",
                    effectiveness_score=0.5 + i * 0.1,
                    usage_count=i
                )
                store.add_memory(memory)

            stats = store.get_statistics()

            assert stats["total_memories"] == 5
            assert stats["memories_by_type"]["debugging"] == 3
            assert stats["memories_by_type"]["feature"] == 2
            assert 0.5 <= stats["avg_effectiveness"] <= 0.9
            assert stats["total_usage"] == sum(range(5))

    def test_similar_memory_merging(self):
        """Test that similar memories are merged when added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(store_path=Path(tmpdir) / "test.json", auto_save=False)

            # Add first memory
            memory1 = Memory(
                task_type="debugging",
                title="Fix auth",
                query="Authentication error",
                context_hash="abc123"
            )
            id1 = store.add_memory(memory1)

            # Add similar memory with same context hash
            memory2 = Memory(
                task_type="debugging",
                title="Fix auth again",
                query="Authentication error",
                context_hash="abc123"
            )
            id2 = store.add_memory(memory2)

            # Should have merged
            assert len(store._memories) == 1
            # The merged memory should have a different ID
            assert id1 != id2

    def test_max_memories_per_type(self):
        """Test that memories per type are limited."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(store_path=Path(tmpdir) / "test.json", auto_save=False)
            # Set a low limit for testing
            store.MAX_MEMORIES_PER_TYPE = 3

            # Add more than limit
            for i in range(5):
                memory = Memory(
                    task_type="debugging",
                    title=f"Memory {i}",
                    query=f"Query {i}",
                    effectiveness_score=i * 0.2  # Later ones are better
                )
                store.add_memory(memory)

            # Should only have 3 memories
            debug_memories = store._memories_by_type["debugging"]
            assert len(debug_memories) == 3

            # Should have kept the better ones
            memories = [store.get_memory(mid) for mid in debug_memories]
            effectiveness_scores = [m.effectiveness_score for m in memories]
            # The worst memory (0.0 or 0.2) should have been pruned
            assert min(effectiveness_scores) >= 0.4


class TestMemoryIntegration:
    """Integration tests for the memory system."""

    def test_full_memory_workflow(self):
        """Test the complete memory workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_memory.json"
            store = MemoryStore(store_path=store_path)
            distiller = MemoryDistiller()

            # Simulate a debugging session
            messages = [
                Message(role="user", content="Fix the TypeError in user authentication"),
                Message(role="assistant", content="I'll fix the TypeError. Let me check the auth.py file first."),
                Message(role="assistant", content="Found it - user object can be None. Adding null check."),
                Message(role="assistant", content="Fixed! The authentication now handles null users correctly."),
            ]

            # Distill memory
            memory = distiller.distill_from_session("session-1", messages)
            assert memory.task_type == "debugging"
            assert "TypeError" in memory.query

            # Store memory
            memory_id = store.add_memory(memory)

            # Search for similar issue
            similar_query = "Authentication throwing errors"
            results = store.search_similar(similar_query, limit=1, threshold=0.1)

            assert len(results) > 0
            retrieved_memory, similarity = results[0]
            assert retrieved_memory.memory_id == memory_id
            assert similarity > 0.1  # Should have some similarity

            # Track effectiveness
            store.update_effectiveness(memory_id, 0.3, "Helpful for similar issue")

            # Verify persistence
            store.save_store()
            assert store_path.exists()

            # Load in new instance
            new_store = MemoryStore(store_path=store_path)
            loaded = new_store.get_memory(memory_id)
            assert loaded.effectiveness_score > 0
            assert loaded.usage_count == 1

    @patch('ai_dev_agent.cli.context_enhancer.MEMORY_SYSTEM_AVAILABLE', True)
    def test_context_enhancer_integration(self):
        """Test memory integration with context enhancer."""
        from ai_dev_agent.cli.context_enhancer import ContextEnhancer
        from ai_dev_agent.core.utils.config import Settings

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test memory store
            store_path = Path(tmpdir) / "test_memory.json"

            # Create settings with memory enabled
            settings = Settings()
            settings.enable_memory_bank = True
            settings.repomap_debug_stdout = False

            # Mock the MemoryStore initialization to use our test path
            with patch('ai_dev_agent.cli.context_enhancer.MemoryStore') as MockStore:
                mock_store = MemoryStore(store_path=store_path, auto_save=False)
                MockStore.return_value = mock_store

                # Create context enhancer
                enhancer = ContextEnhancer(workspace=Path(tmpdir), settings=settings)

                # Add a test memory
                memory = Memory(
                    task_type="debugging",
                    title="✓ Fix auth TypeError",
                    query="Fix TypeError in authentication",
                    strategies=[Strategy(
                        description="Check for null user objects",
                        context="Authentication middleware",
                        steps=["Check user exists", "Add null guard"],
                        tools_used=["read", "write"]
                    )],
                    lessons=[Lesson(
                        mistake="Assumed user always exists",
                        context="Auth handler",
                        correction="Always check user before accessing",
                        severity="major"
                    )],
                    outcome="success"
                )
                mock_store.add_memory(memory)

                # Test memory retrieval
                memory_messages, memory_ids = enhancer.get_memory_context(
                    query="Authentication error TypeError",
                    limit=5
                )

                assert memory_messages is not None
                assert len(memory_messages) == 2  # System message + assistant ack
                assert memory_ids is not None
                assert len(memory_ids) == 1

                # Check message content
                system_msg = memory_messages[0]
                assert system_msg["role"] == "system"
                assert "Memory Bank" in system_msg["content"]
                assert "Fix auth TypeError" in system_msg["content"]
                assert "Check for null user objects" in system_msg["content"]

                # Test effectiveness tracking
                enhancer.track_memory_effectiveness(
                    memory_ids=memory_ids,
                    success=True,
                    feedback="Very helpful"
                )

                updated = mock_store.get_memory(memory_ids[0])
                assert updated.effectiveness_score > 0
                assert updated.usage_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])