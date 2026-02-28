"""Embedding normalization utilities."""

from __future__ import annotations

import numpy as np


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row of an embedding matrix.

    Args:
        embeddings: (N, D) float array.

    Returns:
        (N, D) L2-normalized array (each row has unit norm).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return (embeddings / norms).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of L2-normed embeddings.

    Args:
        a: (N, D) L2-normalized embeddings.
        b: (M, D) L2-normalized embeddings.

    Returns:
        (N, M) similarity matrix with values in [-1, 1].
    """
    return np.dot(a, b.T).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine distance (1 - similarity) between two sets of embeddings.

    Args:
        a: (N, D) L2-normalized embeddings.
        b: (M, D) L2-normalized embeddings.

    Returns:
        (N, M) distance matrix with values in [0, 2].
    """
    return (1.0 - cosine_similarity(a, b)).astype(np.float32)
