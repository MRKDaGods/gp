"""k-Reciprocal re-ranking for ReID (Zhong et al., 2017).

Refines initial similarity scores using local neighbourhood information.
Applied as a post-processing step on FAISS retrieval results.

Improvements over baseline
--------------------------
* **Sparse computation** — instead of an O(N²) full similarity matrix we use
  the FAISS index that already exists upstream to compute per-node
  neighbourhoods.  When no FAISS index is available we fall back to a
  *local* similarity matrix built only from the nodes that appear in the
  candidate set, which is typically ≪ N.
* **Weighted Jaccard** — the k-reciprocal overlap is weighted by the actual
  cosine similarity of the shared neighbours rather than treating every
  neighbour as binary.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def k_reciprocal_rerank(
    embeddings: np.ndarray,
    candidate_pairs: List[Tuple[int, int, float]],
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    faiss_index=None,
) -> Dict[Tuple[int, int], float]:
    """Apply k-reciprocal re-ranking to refine appearance similarities.

    Args:
        embeddings: (N, D) L2-normalized embedding matrix.
        candidate_pairs: List of (i, j, initial_similarity) tuples.
        k1: Number of nearest neighbours for k-reciprocal set.
        k2: Number of nearest neighbours for expanded set (unused in
            weighted variant but controls expansion breadth).
        lambda_value: Blending factor (0 = only Jaccard, 1 = only original).
        faiss_index: Optional FAISS index over *embeddings* — if supplied
            the expensive argsort is replaced with a fast ANN lookup.

    Returns:
        Dict[(i, j)] -> re-ranked similarity score.
    """
    if not candidate_pairs:
        return {}

    n = embeddings.shape[0]

    # Collect the set of node indices that actually participate in any pair
    active_nodes: Set[int] = set()
    for i, j, _ in candidate_pairs:
        active_nodes.add(i)
        active_nodes.add(j)

    # ------------------------------------------------------------------
    # Build per-node top-k1 neighbour lists  (sparse path)
    # ------------------------------------------------------------------
    topk_map: Dict[int, np.ndarray] = {}        # node → array of k1 nn ids
    sim_cache: Dict[int, np.ndarray] = {}        # node → corresponding sims

    if faiss_index is not None:
        # FAISS batch query — only for active nodes
        active_list = sorted(active_nodes)
        query_emb = embeddings[active_list].astype(np.float32)
        sims, indices = faiss_index.search(query_emb, k1)
        for idx, node in enumerate(active_list):
            topk_map[node] = indices[idx]
            sim_cache[node] = sims[idx]
    else:
        # Fallback: build a *local* similarity matrix over active nodes only
        # plus their potential neighbours, which is much cheaper than NxN.
        _build_local_topk(
            embeddings, active_nodes, k1, topk_map, sim_cache
        )

    # ------------------------------------------------------------------
    # Build k-reciprocal sets
    # ------------------------------------------------------------------
    k_reciprocal_sets: Dict[int, Set[int]] = {}
    for node in active_nodes:
        forward = set(topk_map[node].tolist()) - {-1}
        reciprocal: Set[int] = set()
        for j in forward:
            if j in topk_map:
                backward = set(topk_map[j].tolist()) - {-1}
            else:
                # j is not an active query — compute its nns on the fly
                jvec = embeddings[j : j + 1].astype(np.float32)
                if faiss_index is not None:
                    _, jnn = faiss_index.search(jvec, k1)
                    backward = set(jnn[0].tolist()) - {-1}
                else:
                    backward = _fast_topk_for_single(embeddings, j, k1)
                topk_map[j] = np.array(sorted(backward), dtype=np.int64)
            if node in backward:
                reciprocal.add(j)
        k_reciprocal_sets[node] = reciprocal

    # Expand reciprocal sets (step from original paper)
    expanded_sets: Dict[int, Set[int]] = {}
    for node in active_nodes:
        expanded = set(k_reciprocal_sets[node])
        for j in list(k_reciprocal_sets[node]):
            rj = k_reciprocal_sets.get(j, set())
            overlap = len(rj & k_reciprocal_sets[node])
            if overlap > len(rj) * 2 / 3:
                expanded |= rj
        expanded_sets[node] = expanded

    # ------------------------------------------------------------------
    # Compute weighted Jaccard for each candidate pair
    # ------------------------------------------------------------------
    result: Dict[Tuple[int, int], float] = {}
    for i, j, original_sim in candidate_pairs:
        set_i = expanded_sets.get(i, set())
        set_j = expanded_sets.get(j, set())

        intersection = set_i & set_j
        union = set_i | set_j

        if not union:
            jaccard_sim = 0.0
        else:
            # Weighted Jaccard: weight each shared neighbour by the minimum
            # of its similarity to both i and j. Using min (not mean) better
            # captures discriminative neighbours — a node close to i but far
            # from j gets low weight, matching k-reciprocal encoding intent.
            w_inter = 0.0
            w_union = 0.0
            for k in union:
                si = _pairwise_sim(embeddings, i, k)
                sj = _pairwise_sim(embeddings, j, k)
                w = min(si, sj)
                w_union += w
                if k in intersection:
                    w_inter += w
            jaccard_sim = w_inter / max(w_union, 1e-8)

        reranked_sim = (1 - lambda_value) * jaccard_sim + lambda_value * original_sim
        result[(i, j)] = reranked_sim

    logger.debug(
        f"Re-ranked {len(result)} pairs "
        f"(k1={k1}, k2={k2}, λ={lambda_value}, active_nodes={len(active_nodes)})"
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pairwise_sim(embeddings: np.ndarray, i: int, j: int) -> float:
    """Cosine similarity between two L2-normed vectors (dot product)."""
    return float(np.dot(embeddings[i], embeddings[j]))


def _build_local_topk(
    embeddings: np.ndarray,
    active_nodes: Set[int],
    k: int,
    topk_map: Dict[int, np.ndarray],
    sim_cache: Dict[int, np.ndarray],
) -> None:
    """Build top-k neighbours using a *local* similarity matrix.

    We only compute similarities among the active node set + a 1-hop expansion
    (all nodes that appear in any candidate pair). This is O(M²) where
    M = |active_nodes| ≪ N.
    """
    active_list = sorted(active_nodes)
    n_active = len(active_list)

    if n_active == 0:
        return

    # Build local similarity matrix
    active_emb = embeddings[active_list]  # (M, D)
    local_sim = np.dot(active_emb, active_emb.T)  # (M, M)

    # Map global index → local index
    g2l = {g: l for l, g in enumerate(active_list)}

    k_eff = min(k, n_active)
    topk_local = np.argsort(-local_sim, axis=1)[:, :k_eff]  # (M, k_eff)

    for l_idx, g_idx in enumerate(active_list):
        nn_global = np.array([active_list[li] for li in topk_local[l_idx]], dtype=np.int64)
        nn_sims = local_sim[l_idx, topk_local[l_idx]]
        topk_map[g_idx] = nn_global
        sim_cache[g_idx] = nn_sims


def _fast_topk_for_single(
    embeddings: np.ndarray,
    node: int,
    k: int,
) -> Set[int]:
    """Compute top-k neighbours for a single node via brute-force dot product."""
    sims = embeddings @ embeddings[node]  # (N,)
    k_eff = min(k, len(sims))
    topk = np.argpartition(-sims, k_eff)[:k_eff]
    return set(topk.tolist())
