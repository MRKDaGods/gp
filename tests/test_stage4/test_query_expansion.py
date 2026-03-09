"""Tests for Average Query Expansion (AQE)."""

import numpy as np
import pytest

from src.stage4_association.query_expansion import (
    average_query_expansion,
    average_query_expansion_batched,
)


def _make_embeddings(n=10, d=64, seed=42):
    """Create random L2-normalised embeddings + FAISS-style indices."""
    rng = np.random.RandomState(seed)
    raw = rng.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    emb = raw / norms

    # Simulate FAISS top-k indices (sorted by sim) — approximate by dot product
    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)  # (N, N)
    return emb, indices


# ── Basic smoke tests ───────────────────────────────────────────────────────

class TestAverageQueryExpansion:
    def test_output_shape(self):
        emb, indices = _make_embeddings(n=20, d=32)
        result = average_query_expansion(emb, indices, k=5)
        assert result.shape == emb.shape

    def test_l2_normalised(self):
        emb, indices = _make_embeddings(n=15, d=64)
        result = average_query_expansion(emb, indices, k=3)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_k_zero_returns_copy(self):
        emb, indices = _make_embeddings(n=10, d=16)
        result = average_query_expansion(emb, indices, k=0)
        np.testing.assert_array_equal(result, emb)

    def test_empty_embeddings(self):
        emb = np.zeros((0, 32), dtype=np.float32)
        indices = np.zeros((0, 5), dtype=np.int64)
        result = average_query_expansion(emb, indices, k=5)
        assert result.shape == (0, 32)

    def test_single_embedding(self):
        emb, indices = _make_embeddings(n=1, d=16)
        result = average_query_expansion(emb, indices[:, :5], k=5)
        # With only 1 embedding, its only valid neighbour is itself
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_alpha_weighting(self):
        """Higher alpha should keep result closer to original."""
        emb, indices = _make_embeddings(n=10, d=32)
        result_a1 = average_query_expansion(emb, indices, k=5, alpha=1.0)
        result_a10 = average_query_expansion(emb, indices, k=5, alpha=10.0)

        # With alpha=10, result should be much closer to original than alpha=1
        diff_a1 = np.linalg.norm(result_a1 - emb, axis=1).mean()
        diff_a10 = np.linalg.norm(result_a10 - emb, axis=1).mean()
        assert diff_a10 < diff_a1

    def test_invalid_indices_handled(self):
        """FAISS returns -1 for invalid entries."""
        emb, _ = _make_embeddings(n=5, d=16)
        indices = np.full((5, 10), -1, dtype=np.int64)
        # All invalid indices → should return original embeddings
        result = average_query_expansion(emb, indices, k=5)
        np.testing.assert_allclose(
            np.linalg.norm(result, axis=1), 1.0, atol=1e-6
        )

    def test_expansion_changes_embeddings(self):
        """QE with k>0 should actually modify the embeddings."""
        emb, indices = _make_embeddings(n=20, d=64)
        result = average_query_expansion(emb, indices, k=5)
        # At least some embeddings should change
        assert not np.allclose(result, emb, atol=1e-6)

    def test_self_similarity_increases(self):
        """After QE, nearby embeddings should become more similar."""
        emb, indices = _make_embeddings(n=50, d=64)
        # Make two clusters separated
        emb[:25] += 2.0
        emb[25:] -= 2.0
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb /= norms

        sims = emb @ emb.T
        indices = np.argsort(-sims, axis=1)

        result = average_query_expansion(emb, indices, k=5)

        # Intra-cluster similarity should increase for at least one cluster
        before_intra = np.mean(sims[:25, :25])
        after_sims = result @ result.T
        after_intra = np.mean(after_sims[:25, :25])
        assert after_intra >= before_intra - 0.01  # allow small tolerance


# ── Batched variant ─────────────────────────────────────────────────────────

class TestAverageQueryExpansionBatched:
    def test_output_shape(self):
        emb, indices = _make_embeddings(n=20, d=32)
        result = average_query_expansion_batched(emb, indices, k=5)
        assert result.shape == emb.shape

    def test_l2_normalised(self):
        emb, indices = _make_embeddings(n=15, d=64)
        result = average_query_expansion_batched(emb, indices, k=3)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_k_zero_returns_copy(self):
        emb, indices = _make_embeddings(n=10, d=16)
        result = average_query_expansion_batched(emb, indices, k=0)
        np.testing.assert_array_equal(result, emb)

    def test_empty_embeddings(self):
        emb = np.zeros((0, 32), dtype=np.float32)
        indices = np.zeros((0, 5), dtype=np.int64)
        result = average_query_expansion_batched(emb, indices, k=5)
        assert result.shape == (0, 32)

    def test_matches_loop_variant(self):
        """Batched and loop variants should produce the same result."""
        emb, indices = _make_embeddings(n=30, d=32)
        result_loop = average_query_expansion(emb, indices, k=5, alpha=1.0)
        result_batch = average_query_expansion_batched(emb, indices, k=5, alpha=1.0)
        np.testing.assert_allclose(result_loop, result_batch, atol=1e-5)

    def test_alpha_weighting(self):
        emb, indices = _make_embeddings(n=10, d=32)
        result_a1 = average_query_expansion_batched(emb, indices, k=5, alpha=1.0)
        result_a10 = average_query_expansion_batched(emb, indices, k=5, alpha=10.0)
        diff_a1 = np.linalg.norm(result_a1 - emb, axis=1).mean()
        diff_a10 = np.linalg.norm(result_a10 - emb, axis=1).mean()
        assert diff_a10 < diff_a1

    def test_invalid_indices_handled(self):
        """Negative indices (FAISS sentinel) should be masked out."""
        emb, _ = _make_embeddings(n=5, d=16)
        indices = np.full((5, 10), -1, dtype=np.int64)
        result = average_query_expansion_batched(emb, indices, k=5)
        # With all invalid indices, result should be original (L2-normed)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_expansion_changes_embeddings(self):
        emb, indices = _make_embeddings(n=20, d=64)
        result = average_query_expansion_batched(emb, indices, k=5)
        assert not np.allclose(result, emb, atol=1e-6)


# ── Integration test ────────────────────────────────────────────────────────

class TestQEIntegration:
    def test_pipeline_flow(self):
        """Simulate the full QE flow as used in pipeline.py."""
        n, d = 50, 128
        rng = np.random.RandomState(0)
        raw = rng.randn(n, d).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        embeddings = raw / norms

        # Simulate FAISS search (inner product → cosine sim for L2-normed)
        sims = embeddings @ embeddings.T
        top_k = 20
        indices = np.argsort(-sims, axis=1)[:, :top_k]

        # Apply QE
        expanded = average_query_expansion_batched(
            embeddings, indices, k=5, alpha=1.0
        )

        # Verify properties
        assert expanded.shape == embeddings.shape
        norms = np.linalg.norm(expanded, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

        # Re-compute similarity with expanded embeddings
        new_sims = expanded @ expanded.T
        # Diagonal should be ~1.0
        np.testing.assert_allclose(np.diag(new_sims), 1.0, atol=1e-6)
