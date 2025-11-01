"""Additional tests for EmbeddingGenerator covering fallbacks and caching."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ai_dev_agent.memory.embeddings as embeddings_module
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


def test_embedding_generator_transformer_batch_cache_cooperation(monkeypatch, tmp_path):
    """Batch transformer embeddings should mix in-memory, disk, and fresh computations."""

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts, show_progress_bar: bool = False, convert_to_numpy: bool = True):
            def embed_for(text: str) -> np.ndarray:
                seed = float(len(text) % 7 + 1)
                return np.array([seed, seed + 0.25, seed + 0.5], dtype=float)

            if isinstance(texts, str):
                return embed_for(texts)
            rows = [embed_for(text) for text in texts]
            return np.vstack(rows)

    monkeypatch.setattr(embeddings_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        embeddings_module, "SentenceTransformer", FakeSentenceTransformer, raising=False
    )
    monkeypatch.setattr(EmbeddingGenerator, "CACHE_DIR", tmp_path, raising=False)

    generator = EmbeddingGenerator(method="transformer", cache_embeddings=True)
    generator.clear_cache()

    cached_text = "cached transformer text"
    disk_text = "read from disk cache"
    fresh_text = "fresh encode request"

    cached_vector = generator.generate_embedding(cached_text)
    disk_key = generator._get_cache_key(disk_text)
    disk_vector = np.array([9.0, 9.5, 10.0], dtype=float)
    generator._save_to_cache(disk_key, disk_vector)
    generator._embedding_cache.pop(disk_key, None)

    matrix = generator.generate_embeddings([cached_text, disk_text, fresh_text])
    expected_fresh = generator.transformer_model.encode(fresh_text)

    assert matrix.shape == (3, generator.get_embedding_dimension())
    assert np.array_equal(matrix[0], cached_vector)
    assert np.array_equal(matrix[1], disk_vector)
    assert np.array_equal(matrix[2], expected_fresh)
    assert generator._get_cache_key(cached_text) in generator._embedding_cache
    assert disk_key in generator._embedding_cache


@pytest.mark.parametrize(
    ("vector", "expect_tail_zero"),
    [
        (np.arange(10, dtype=float), True),
        (np.arange(400, dtype=float), False),
    ],
)
def test_embedding_generator_tfidf_padding_and_truncation(monkeypatch, vector, expect_tail_zero):
    """TF-IDF embeddings should pad short vectors and truncate long ones deterministically."""

    import sklearn.feature_extraction.text as sklearn_text

    class DummyVectorizer:
        def __init__(self, *args, **kwargs):
            self._output = vector

        def transform(self, texts):
            return SimpleNamespace(toarray=lambda: np.array([self._output]))

    monkeypatch.setattr(sklearn_text, "HashingVectorizer", DummyVectorizer, raising=False)
    generator = EmbeddingGenerator(method="tfidf", cache_embeddings=False)

    embedding = generator._generate_tfidf_embedding("exercise tfidf normalization")
    assert embedding.shape == (generator.embedding_dim,)

    if expect_tail_zero:
        assert np.allclose(embedding[: len(vector)], vector)
        assert np.count_nonzero(embedding[len(vector) :]) == 0
    else:
        assert np.array_equal(embedding, vector[: generator.embedding_dim])


def test_embedding_generator_find_similar_handles_empty_candidates():
    """find_similar should return an empty list when no candidates are provided."""
    generator = EmbeddingGenerator(method="tfidf", cache_embeddings=False)

    assert generator.find_similar("anything", [], top_k=3) == []


def test_embedding_generator_load_and_save_noops_when_caching_disabled(tmp_path, monkeypatch):
    """load/save helpers should short-circuit when caching is disabled."""
    monkeypatch.setattr(EmbeddingGenerator, "CACHE_DIR", tmp_path, raising=False)
    generator = EmbeddingGenerator(method="tfidf", cache_embeddings=False)

    assert generator._load_from_cache("never-stored") is None
    generator._save_to_cache("no-op", np.array([1.0, 2.0]))
    assert not list(tmp_path.glob("*.pkl"))


def test_embedding_generator_clear_cache_swallows_unlink_errors(tmp_path, monkeypatch):
    """clear_cache should ignore unlink errors while still clearing memory state."""
    monkeypatch.setattr(EmbeddingGenerator, "CACHE_DIR", tmp_path, raising=False)
    generator = EmbeddingGenerator(method="tfidf", cache_embeddings=True)
    generator.clear_cache()

    key = generator._get_cache_key("raise on unlink")
    generator._save_to_cache(key, np.ones(generator.embedding_dim))
    cache_file = Path(tmp_path, f"{key}.pkl")
    assert cache_file.exists()
    generator._embedding_cache[key] = np.ones(generator.embedding_dim)

    original_unlink = Path.unlink

    def flaky_unlink(self, *args, **kwargs):
        if self == cache_file:
            raise OSError("permission denied")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    generator.clear_cache()

    assert generator._embedding_cache == {}
    assert cache_file.exists()
