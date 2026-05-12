"""Average Query Expansion (AQE) for ReID feature refinement.

Implements the query expansion technique that averages each embedding
with its top-K nearest neighbours from the gallery, then re-normalises
to unit length.  This smooths noise and makes embeddings more robust
to viewpoint / lighting changes.

Reference:
    Chum et al., "Total Recall: Automatic Query Expansion with a
    Generative Feature Model for Object Retrieval", ICCV 2007.

Used in gp-stage-2 (FastReID baseline) with k=5.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger


def average_query_expansion(
    embeddings: np.ndarray,
    indices: np.ndarray,
    k: int = 5,
    alpha: float = 1.0,
    faiss_index=None,
) -> np.ndarray:
    """Apply Average Query Expansion (AQE) to embeddings.

    For each embedding, averages it with its top-*k* nearest neighbours
    (weighted equally), then L2-normalises the result.  This 'smooths'
    embeddings toward their local neighbourhood, amplifying genuine
    similarity and suppressing noise.

    Args:
        embeddings: (N, D) L2-normalised embedding matrix.
        indices: (N, K) FAISS nearest-neighbour indices from initial retrieval.
            Each row contains the top-K neighbour IDs for the corresponding
            embedding.  Negative values (FAISS sentinel) are ignored.
        k: Number of neighbours to include in expansion.  Must be
            <= indices.shape[1].
        alpha: Weight for the original embedding vs neighbours.
            With alpha=1.0, the original query counts as 1 vote among
            k+1 total (uniform averaging).  Higher alpha increases the
            original query's weight relative to neighbours.
        faiss_index: Optional FAISS index for a second-pass retrieval
            when *indices* doesn't have enough valid columns.  Unused
            if indices already has >= k valid neighbours per row.

    Returns:
        (N, D) L2-normalised expanded embedding matrix.
    """
    n, d = embeddings.shape
    k_available = indices.shape[1] if indices.ndim == 2 else 0

    if k <= 0 or n == 0:
        return embeddings.copy()

    k_use = min(k, k_available)
    if k_use <= 0:
        logger.warning("QE: no valid neighbour indices available, returning original")
        return embeddings.copy()

    expanded = np.empty_like(embeddings)

    for i in range(n):
        nn_idx = indices[i, :k_use]
        # Filter out invalid FAISS indices (-1 sentinel or out of range)
        valid = nn_idx[(nn_idx >= 0) & (nn_idx < n)]

        if len(valid) == 0:
            expanded[i] = embeddings[i]
            continue

        # Neighbour embeddings
        nn_feats = embeddings[valid]  # (k', D)

        # Weighted average: original gets weight=alpha, each neighbour gets weight=1
        total_weight = alpha + len(valid)
        expanded[i] = (alpha * embeddings[i] + nn_feats.sum(axis=0)) / total_weight

        # L2 normalise
        norm = np.linalg.norm(expanded[i])
        if norm > 0:
            expanded[i] /= norm

    logger.info(f"Query Expansion applied: k={k_use}, alpha={alpha}, N={n}")
    return expanded


def average_query_expansion_batched(
    embeddings: np.ndarray,
    indices: np.ndarray,
    k: int = 5,
    alpha: float = 1.0,
) -> np.ndarray:
    """Vectorised variant of AQE — faster for large N.

    Semantics match :func:`average_query_expansion` (same weighting, same
    handling of invalid indices and of rows with no valid neighbours).

    Args:
        embeddings: (N, D) L2-normalised embedding matrix.
        indices: (N, K) neighbour indices, first *k* columns used (after k_use).
        k: Neighbours per query.
        alpha: Original-query weight.

    Returns:
        (N, D) L2-normalised expanded embedding matrix.
    """
    n, d = embeddings.shape
    k_available = indices.shape[1] if indices.ndim == 2 else 0

    if k <= 0 or n == 0:
        return embeddings.copy()

    k_use = min(k, k_available)
    if k_use <= 0:
        logger.warning("QE (batched): no valid neighbour indices available, returning original")
        return embeddings.copy()

    nn_idx = indices[:, :k_use]
    valid_mask = (nn_idx >= 0) & (nn_idx < n)
    safe_idx = np.where(valid_mask, nn_idx, 0)
    nn_feats = embeddings[safe_idx]
    nn_feats[~valid_mask] = 0.0

    valid_counts = valid_mask.sum(axis=1, keepdims=True).astype(np.float32)
    has_nn = valid_counts.squeeze(axis=1) >= 1.0

    nn_sum = nn_feats.sum(axis=1)
    numer = alpha * embeddings + nn_sum
    denom = alpha + valid_counts
    expanded = np.empty_like(embeddings)
    expanded[has_nn] = (numer / denom)[has_nn]
    expanded[~has_nn] = embeddings[~has_nn]

    norms = np.linalg.norm(expanded[has_nn], axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    expanded[has_nn] /= norms

    logger.info(f"Query Expansion (batched): k={k_use}, alpha={alpha}, N={n}")
    return expanded
