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

    Uses advanced indexing to avoid the per-row Python loop.
    Requires that *indices* has no negative (invalid) entries in
    the first *k* columns; caller should ensure this or use the
    loop-based :func:`average_query_expansion` instead.

    Args:
        embeddings: (N, D) L2-normalised embedding matrix.
        indices: (N, K) neighbour indices, first *k* columns used.
        k: Neighbours per query.
        alpha: Original-query weight.

    Returns:
        (N, D) L2-normalised expanded embedding matrix.
    """
    n, d = embeddings.shape
    k_use = min(k, indices.shape[1])
    if k_use <= 0 or n == 0:
        return embeddings.copy()

    nn_idx = indices[:, :k_use]  # (N, k)

    # Clamp invalid indices to 0 (will be masked out later)
    valid_mask = (nn_idx >= 0) & (nn_idx < n)
    safe_idx = np.where(valid_mask, nn_idx, 0)  # (N, k)

    # Gather neighbour embeddings: (N, k, D)
    nn_feats = embeddings[safe_idx]

    # Zero out invalid neighbours
    nn_feats[~valid_mask] = 0.0

    # Count valid neighbours per query
    valid_counts = valid_mask.sum(axis=1, keepdims=True).astype(np.float32)  # (N, 1)
    valid_counts = np.maximum(valid_counts, 1e-8)

    # Weighted sum: alpha * query + sum(valid neighbours)
    nn_sum = nn_feats.sum(axis=1)  # (N, D)
    expanded = (alpha * embeddings + nn_sum) / (alpha + valid_counts)

    # L2 normalise
    norms = np.linalg.norm(expanded, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    expanded /= norms

    logger.info(f"Query Expansion (batched): k={k_use}, alpha={alpha}, N={n}")
    return expanded
