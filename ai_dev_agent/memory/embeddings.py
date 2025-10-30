"""Embedding generator for vector-based memory search.

Provides multiple embedding strategies:
1. TF-IDF based (no external dependencies)
2. Sentence transformers (if installed)
3. OpenAI embeddings (if API available)
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.debug("sentence-transformers not available, using TF-IDF embeddings")


class EmbeddingGenerator:
    """Generates and manages embeddings for memory search."""

    # Cache directory for embeddings
    CACHE_DIR = Path.home() / ".devagent" / "embeddings_cache"

    def __init__(
        self, method: str = "auto", model_name: str | None = None, cache_embeddings: bool = True
    ):
        """Initialize the embedding generator.

        Args:
            method: Embedding method - "tfidf", "transformer", "openai", or "auto"
            model_name: Model name for transformer method
            cache_embeddings: Whether to cache computed embeddings
        """
        self.method = self._select_method(method)
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: dict[str, np.ndarray] = {}

        # Initialize based on method
        if self.method == "transformer":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("Sentence transformers not available, falling back to TF-IDF")
                self.method = "tfidf"
            else:
                # Use a small, fast model by default
                model_name = model_name or "all-MiniLM-L6-v2"
                try:
                    self.transformer_model = SentenceTransformer(model_name)
                    self.embedding_dim = self.transformer_model.get_sentence_embedding_dimension()
                    logger.info(f"Using transformer model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load transformer model: {e}, falling back to TF-IDF")
                    self.method = "tfidf"

        if self.method == "tfidf":
            # Use a simpler approach: fixed vocabulary size
            self.embedding_dim = 384
            self._tfidf_fitted = False
            self._corpus: list[str] = []
            self._vocabulary = None
            logger.debug("Using TF-IDF embeddings")

        # Create cache directory if needed
        if cache_embeddings:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _select_method(self, method: str) -> str:
        """Select the best available embedding method."""
        if method == "auto":
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                return "transformer"
            else:
                return "tfidf"
        return method

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache first
        if self.cache_embeddings:
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

            # Check disk cache
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                self._embedding_cache[cache_key] = cached
                return cached

        # Generate embedding based on method
        if self.method == "transformer":
            embedding = self._generate_transformer_embedding(text)
        elif self.method == "tfidf":
            embedding = self._generate_tfidf_embedding(text)
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")

        # Cache the result
        if self.cache_embeddings:
            self._embedding_cache[cache_key] = embedding
            self._save_to_cache(cache_key, embedding)

        return embedding

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embedding vectors
        """
        embeddings = []

        # For transformer, batch processing is more efficient
        if self.method == "transformer" and len(texts) > 1:
            # Check cache for all texts
            uncached_texts = []
            uncached_indices = []
            cached_embeddings = {}

            for i, text in enumerate(texts):
                if self.cache_embeddings:
                    cache_key = self._get_cache_key(text)
                    if cache_key in self._embedding_cache:
                        cached_embeddings[i] = self._embedding_cache[cache_key]
                    else:
                        cached = self._load_from_cache(cache_key)
                        if cached is not None:
                            cached_embeddings[i] = cached
                            self._embedding_cache[cache_key] = cached
                        else:
                            uncached_texts.append(text)
                            uncached_indices.append(i)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.transformer_model.encode(
                    uncached_texts, show_progress_bar=False, convert_to_numpy=True
                )

                # Cache the new embeddings
                for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                    if self.cache_embeddings:
                        cache_key = self._get_cache_key(text)
                        self._embedding_cache[cache_key] = embedding
                        self._save_to_cache(cache_key, embedding)
                    cached_embeddings[idx] = embedding

            # Combine in correct order
            embeddings = [cached_embeddings[i] for i in range(len(texts))]
            return np.vstack(embeddings)

        else:
            # Generate one by one
            for text in texts:
                embeddings.append(self.generate_embedding(text))
            return np.vstack(embeddings)

    def compute_similarity(
        self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and candidates.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings

        Returns:
            Similarity scores
        """
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)

        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        return similarities

    def find_similar(
        self,
        query: str,
        candidates: list[tuple[str, np.ndarray]],
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[tuple[int, float]]:
        """Find most similar candidates to query.

        Args:
            query: Query text
            candidates: List of (text, embedding) tuples
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity) tuples
        """
        if not candidates:
            return []

        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Extract candidate embeddings
        candidate_embeddings = np.vstack([emb for _, emb in candidates])

        # Compute similarities
        similarities = self.compute_similarity(query_embedding, candidate_embeddings)

        # Filter by threshold and sort
        results = []
        for idx, sim in enumerate(similarities):
            if sim >= threshold:
                results.append((idx, float(sim)))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _generate_transformer_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using sentence transformer."""
        embedding = self.transformer_model.encode(
            text, show_progress_bar=False, convert_to_numpy=True
        )
        return embedding

    def _generate_tfidf_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using TF-IDF with fixed dimensions."""
        # Simple but effective: use character n-grams with hashing
        from sklearn.feature_extraction.text import HashingVectorizer

        # Use hashing vectorizer for consistent dimensions
        if not hasattr(self, "_hashing_vectorizer"):
            self._hashing_vectorizer = HashingVectorizer(
                n_features=self.embedding_dim,
                ngram_range=(2, 4),  # Character n-grams
                analyzer="char",
                norm="l2",
                alternate_sign=False,  # All positive values
            )

        # Generate embedding
        embedding = self._hashing_vectorizer.transform([text]).toarray()[0]

        # Ensure it has the correct shape
        if embedding.shape[0] != self.embedding_dim:
            # Pad or truncate if necessary
            if embedding.shape[0] < self.embedding_dim:
                # Pad with zeros
                padded = np.zeros(self.embedding_dim)
                padded[: embedding.shape[0]] = embedding
                embedding = padded
            else:
                # Truncate
                embedding = embedding[: self.embedding_dim]

        return embedding

    def update_tfidf_corpus(self, texts: list[str]) -> None:
        """Update TF-IDF corpus with new texts.

        This is kept for compatibility but not needed with HashingVectorizer.

        Args:
            texts: All texts in the memory store
        """
        # HashingVectorizer doesn't need corpus fitting
        # This method is kept for API compatibility
        if self.method == "tfidf" and texts:
            logger.debug(f"HashingVectorizer doesn't require corpus updates ({len(texts)} texts)")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Use first 100 chars + hash for key
        text_preview = text[:100]
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{self.method}_{text_preview}_{text_hash}"

    def _load_from_cache(self, cache_key: str) -> np.ndarray | None:
        """Load embedding from disk cache."""
        if not self.cache_embeddings:
            return None

        cache_file = self.CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with Path(cache_file).open("rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.debug(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Save embedding to disk cache."""
        if not self.cache_embeddings:
            return

        cache_file = self.CACHE_DIR / f"{cache_key}.pkl"
        try:
            with Path(cache_file).open("wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.debug(f"Failed to save cache {cache_key}: {e}")

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        if self.cache_embeddings and self.CACHE_DIR.exists():
            for cache_file in self.CACHE_DIR.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            logger.info("Cleared embedding cache")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim
