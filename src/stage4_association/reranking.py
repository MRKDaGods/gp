"""k-Reciprocal re-ranking for ReID (Zhong et al., 2017).

Refines initial similarity scores using local neighborhood information.
Applied as a post-processing step on FAISS retrieval results.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from loguru import logger


def k_reciprocal_rerank(
    embeddings: np.ndarray,
    candidate_pairs: List[Tuple[int, int, float]],
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> Dict[Tuple[int, int], float]:
    """Apply k-reciprocal re-ranking to refine appearance similarities.

    Simplified implementation: computes Jaccard distance based on
    k-reciprocal nearest neighbors, then blends with original distance.

    Args:
        embeddings: (N, D) L2-normalized embedding matrix.
        candidate_pairs: List of (i, j, initial_similarity) tuples.
        k1: Number of nearest neighbors for k-reciprocal set.
        k2: Number of nearest neighbors for expanded set.
        lambda_value: Blending factor (0 = only Jaccard, 1 = only original).

    Returns:
        Dict[(i, j)] -> re-ranked similarity score.
    """
    n = embeddings.shape[0]

    # Compute full similarity matrix (cosine sim since L2-normed)
    sim_matrix = np.dot(embeddings, embeddings.T)

    # Get top-k1 neighbors for each element
    topk_indices = np.argsort(-sim_matrix, axis=1)[:, :k1]

    # Build k-reciprocal sets
    k_reciprocal_sets = []
    for i in range(n):
        forward = set(topk_indices[i].tolist())
        reciprocal = set()
        for j in forward:
            backward = set(topk_indices[j].tolist())
            if i in backward:
                reciprocal.add(j)
        k_reciprocal_sets.append(reciprocal)

    # Expand reciprocal sets with k2
    expanded_sets = []
    for i in range(n):
        expanded = set(k_reciprocal_sets[i])
        for j in list(k_reciprocal_sets[i]):
            rj = k_reciprocal_sets[j]
            # If more than 2/3 of rj is in ri, add all of rj
            overlap = len(rj & k_reciprocal_sets[i])
            if overlap > len(rj) * 2 / 3:
                expanded |= rj
        expanded_sets.append(expanded)

    # Compute Jaccard distance for candidate pairs
    result = {}
    for i, j, original_sim in candidate_pairs:
        set_i = expanded_sets[i]
        set_j = expanded_sets[j]

        intersection = len(set_i & set_j)
        union = len(set_i | set_j)
        jaccard_sim = intersection / max(union, 1)

        # Blend original similarity with Jaccard
        reranked_sim = (1 - lambda_value) * jaccard_sim + lambda_value * original_sim
        result[(i, j)] = reranked_sim

    logger.debug(f"Re-ranked {len(result)} pairs (k1={k1}, k2={k2}, λ={lambda_value})")
    return result
