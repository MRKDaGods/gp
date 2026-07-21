"""Average Query Expansion (AQE) for ReID feature refinement.
Reference:; Chum et al., "Total Recall: Automatic Query Expansion with a; Generative Feature Model for Object Retrieval", ICCV 2007.
"""

from __future__ import annotations


import numpy as np
import logging

logger = logging.getLogger(__name__)


def average_query_expansion(
    embeddings: np.ndarray,
    indices: np.ndarray,
    k: int = 5,
    alpha: float = 1.0,
    faiss_index=None,
) -> np.ndarray:
    """Apply Average Query Expansion (AQE) to embeddings."""
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
    """Vectorised variant of AQE - faster for large N."""
    n, d = embeddings.shape
    k_available = indices.shape[1] if indices.ndim == 2 else 0

    if k <= 0 or n == 0:
        return embeddings.copy()

    k_use = min(k + 1, k_available)
    if k_use <= 0:
        logger.warning("QE (batched): no valid neighbour indices available, returning original")
        return embeddings.copy()

    nn_idx = indices[:, :k_use]
    self_mask = nn_idx == np.arange(n).reshape(-1, 1)
    valid_mask = (nn_idx >= 0) & (nn_idx < n) & ~self_mask
    valid_mask &= np.cumsum(valid_mask, axis=1) <= k
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

    effective_k = min(k, max(k_available - 1, 0))
    logger.info(f"Query Expansion (batched): k={effective_k}, alpha={alpha}, N={n}")
    return expanded
