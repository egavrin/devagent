"""Additional tests for EmbeddingGenerator covering fallbacks and caching."""

from __future__ import annotations

import numpy as np
import pytest

from ai_dev_agent.memory.embeddings import EmbeddingGenerator


def test_embedding_generator_auto_selects_tfidf_when_transformers_unavailable(monkeypatch):
    """Auto mode should fall back to TF-IDF when sentence-transformers is missing."""
    monkeypatch.setattr(
        "ai_dev_agent.memory.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE", False, raising=True
    )

    generator = EmbeddingGenerator(method="auto", cache_embeddings=False)

    assert generator.method == "tfidf"
    assert generator.get_embedding_dimension() == 384


def test_embedding_generator_raises_for_unknown_method(monkeypatch):
    """Requesting an unsupported embedding method should raise an error."""
    generator = EmbeddingGenerator(method="tfidf", cache_embeddings=False)
    generator.method = "unsupported"  # simulate misconfiguration

    with pytest.raises(ValueError):
        generator.generate_embedding("test text")


def test_embedding_generator_disk_cache_round_trip(tmp_path, monkeypatch):
    """Embeddings should round-trip through the on-disk cache when enabled."""
    monkeypatch.setattr("ai_dev_agent.memory.embeddings.EmbeddingGenerator.CACHE_DIR", tmp_path)
    generator = EmbeddingGenerator(method="tfidf", cache_embeddings=True)

    text = "Investigate intermittent CI failure"
    first = generator.generate_embedding(text)

    # Ensure cache file exists on disk
    cache_files = list(tmp_path.glob("*.pkl"))
    assert cache_files, "Expected embedding cache file to be created"

    # Clear in-memory cache and load again, forcing disk read
    generator._embedding_cache.clear()
    second = generator.generate_embedding(text)

    assert np.array_equal(first, second)
    # After load the in-memory cache should be restored
    assert generator._embedding_cache


def test_embedding_generator_clear_cache_removes_files(tmp_path, monkeypatch):
    """clear_cache should wipe both memory and disk caches."""
    monkeypatch.setattr("ai_dev_agent.memory.embeddings.EmbeddingGenerator.CACHE_DIR", tmp_path)
    generator = EmbeddingGenerator(method="tfidf", cache_embeddings=True)

    generator.generate_embedding("Capture cache artefact")
    assert list(tmp_path.glob("*.pkl"))
    assert generator._embedding_cache

    generator.clear_cache()

    assert not generator._embedding_cache
    assert not list(tmp_path.glob("*.pkl"))


def test_embedding_generator_find_similar_respects_threshold(monkeypatch):
    """find_similar should filter candidates below the requested threshold."""
    generator = EmbeddingGenerator(method="tfidf", cache_embeddings=False)

    base = generator.generate_embedding("Investigate memory leak")
    candidates = [
        ("memory leak analysis", base),
        ("update documentation", generator.generate_embedding("update documentation")),
    ]

    results = generator.find_similar("memory leak", candidates, top_k=5, threshold=0.6)

    assert results
    assert results[0][0] == 0

    strict_results = generator.find_similar("memory leak", candidates, top_k=5, threshold=0.99)
    assert strict_results == []


def test_embedding_generator_compute_similarity_handles_vector_shapes():
    """compute_similarity should accept both 1-D and 2-D candidate arrays."""
    generator = EmbeddingGenerator(method="tfidf", cache_embeddings=False)
    query = generator.generate_embedding("Assess CLI reliability")
    candidate = generator.generate_embedding("Assess CLI reliability")
    stacked = np.vstack([candidate, candidate])

    single = generator.compute_similarity(query, candidate)
    stacked_scores = generator.compute_similarity(query, stacked)

    assert pytest.approx(single[0]) == stacked_scores[0]
