"""ReID evaluation: mAP, CMC, and optional re-ranking.

Standard evaluation protocol for person/vehicle re-identification
on Market-1501, VeRi-776, MSMT17, etc.

Implements:
  - Feature extraction from query and gallery sets
  - Euclidean / cosine distance computation
  - mAP and CMC (Rank-1, Rank-5, Rank-10) metrics
  - k-reciprocal re-ranking (Zhong et al., CVPR 2017)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger


@torch.no_grad()
def extract_features(
    model,
    dataloader,
    device: str = "cuda:0",
    flip: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features, pids, and camids from a dataloader.

    Args:
        model: ReID model (eval mode).
        dataloader: DataLoader yielding (imgs, pids, cams, paths).
        device: Torch device.
        flip: Whether to use flip augmentation.

    Returns:
        (features [N, D], pids [N], camids [N])
    """
    model.eval()
    all_feats = []
    all_pids = []
    all_cams = []

    for imgs, pids, cams, _ in dataloader:
        imgs = imgs.to(device)
        feats = model(imgs)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]  # Use BN feature

        if flip:
            imgs_flip = torch.flip(imgs, dims=[3])  # horizontal flip
            feats_flip = model(imgs_flip)
            if isinstance(feats_flip, (tuple, list)):
                feats_flip = feats_flip[-1]
            feats = (feats + feats_flip) / 2.0

        # L2 normalize
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)

        all_feats.append(feats.cpu().numpy())
        all_pids.append(pids.numpy())
        all_cams.append(cams.numpy())

    features = np.concatenate(all_feats, axis=0)
    pids = np.concatenate(all_pids, axis=0)
    camids = np.concatenate(all_cams, axis=0)

    return features, pids, camids


def compute_distance_matrix(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute pairwise distance matrix.

    Args:
        query_features: (Nq, D) query features.
        gallery_features: (Ng, D) gallery features.
        metric: 'cosine' or 'euclidean'.

    Returns:
        (Nq, Ng) distance matrix.
    """
    if metric == "cosine":
        # Cosine distance = 1 - cosine similarity
        # Features should already be L2-normalized
        sim = query_features @ gallery_features.T
        dist = 1.0 - sim
    elif metric == "euclidean":
        # Squared Euclidean distance
        m, n = query_features.shape[0], gallery_features.shape[0]
        dist = (
            np.sum(query_features ** 2, axis=1, keepdims=True)
            + np.sum(gallery_features ** 2, axis=1, keepdims=True).T
            - 2 * query_features @ gallery_features.T
        )
        dist = np.clip(dist, 0, None)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return dist


def eval_market1501(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
) -> Tuple[float, np.ndarray]:
    """Evaluate with Market-1501 protocol (exclude same-pid-same-cam).

    Returns:
        (mAP, cmc_curve)
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)  # ascending distance

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Remove gallery samples with same pid AND same camid
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove

        if not np.any(matches[q_idx][keep]):
            continue  # No valid gallery match

        raw_cmc = matches[q_idx][keep]
        num_valid += 1

        # CMC
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        # AP (Average Precision)
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        tmp_cmc_mask = raw_cmc.astype(bool)
        AP = (precision * tmp_cmc_mask).sum() / num_rel if num_rel > 0 else 0
        all_AP.append(AP)

    if num_valid == 0:
        logger.warning("No valid query found!")
        return 0.0, np.zeros(max_rank)

    all_cmc = np.array(all_cmc, dtype=np.float32)
    cmc = all_cmc.mean(axis=0)
    mAP = float(np.mean(all_AP))

    return mAP, cmc


def compute_reranking(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """k-reciprocal re-ranking (Zhong et al., CVPR 2017).

    Reference: "Re-ranking Person Re-identification with k-reciprocal Encoding"

    Args:
        query_features: (Nq, D) L2-normalized query features.
        gallery_features: (Ng, D) L2-normalized gallery features.
        k1, k2, lambda_value: Re-ranking hyper-parameters.

    Returns:
        (Nq, Ng) re-ranked distance matrix.
    """
    # Combine query and gallery for unified distance computation
    all_features = np.concatenate([query_features, gallery_features], axis=0)
    N = all_features.shape[0]
    nq = query_features.shape[0]

    # Original distance (Euclidean on L2-normalized = related to cosine)
    original_dist = 2.0 - 2.0 * (all_features @ all_features.T)
    original_dist = np.clip(original_dist, 0, None)

    # k-NN
    initial_rank = np.argsort(original_dist, axis=1)

    # Jaccard distance via k-reciprocal expansion
    V = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        # Forward k-NN
        forward_k_neigh = initial_rank[i, :k1 + 1]
        # k-reciprocal neighbors
        k_reciprocal = []
        for candidate in forward_k_neigh:
            backward_k_neigh = initial_rank[candidate, :k1 + 1]
            if i in backward_k_neigh:
                k_reciprocal.append(candidate)
        k_reciprocal = np.array(k_reciprocal)

        # k-reciprocal expansion
        k_reciprocal_exp = k_reciprocal.copy()
        for candidate in k_reciprocal:
            candidate_forward = initial_rank[candidate, :int(np.round(k1 / 2)) + 1]
            candidate_reciprocal = []
            for cc in candidate_forward:
                cc_backward = initial_rank[cc, :int(np.round(k1 / 2)) + 1]
                if candidate in cc_backward:
                    candidate_reciprocal.append(cc)
            candidate_reciprocal = np.array(candidate_reciprocal)

            if len(candidate_reciprocal) > 2 / 3 * len(candidate_forward):
                k_reciprocal_exp = np.union1d(k_reciprocal_exp, candidate_reciprocal)

        # Gaussian kernel weight
        weight = np.exp(-original_dist[i, k_reciprocal_exp])
        V[i, k_reciprocal_exp] = weight / (np.sum(weight) + 1e-12)

    # Local query expansion
    if k2 > 0:
        V_qe = np.zeros_like(V)
        for i in range(N):
            V_qe[i] = np.mean(V[initial_rank[i, :k2 + 1]], axis=0)
        V = V_qe

    # Jaccard distance
    # Only compute query-gallery pairs
    jaccard_dist = np.zeros((nq, N - nq), dtype=np.float32)
    for i in range(nq):
        temp_min = np.minimum(V[i], V[nq:])
        temp_max = np.maximum(V[i], V[nq:])
        jaccard_dist[i] = 1 - np.sum(temp_min, axis=1) / (np.sum(temp_max, axis=1) + 1e-12)

    # Final distance: weighted combination
    final_dist = (
        jaccard_dist * (1 - lambda_value)
        + original_dist[:nq, nq:] * lambda_value
    )

    return final_dist


def evaluate_reid(
    model,
    query_loader,
    gallery_loader,
    device: str = "cuda:0",
    rerank: bool = False,
    rerank_k1: int = 20,
    rerank_k2: int = 6,
    rerank_lambda: float = 0.3,
) -> Tuple[float, np.ndarray, Optional[float], Optional[np.ndarray]]:
    """Full ReID evaluation pipeline.

    Args:
        model: Trained ReID model.
        query_loader: Query DataLoader.
        gallery_loader: Gallery DataLoader.
        device: Torch device.
        rerank: Whether to also compute re-ranked metrics.

    Returns:
        (mAP, cmc, mAP_rerank, cmc_rerank)
    """
    logger.info("Extracting query features...")
    q_feats, q_pids, q_camids = extract_features(model, query_loader, device)
    logger.info(f"  Query: {q_feats.shape[0]} images, {len(set(q_pids))} IDs")

    logger.info("Extracting gallery features...")
    g_feats, g_pids, g_camids = extract_features(model, gallery_loader, device)
    logger.info(f"  Gallery: {g_feats.shape[0]} images, {len(set(g_pids))} IDs")

    # Standard evaluation (cosine distance)
    logger.info("Computing distance matrix...")
    distmat = compute_distance_matrix(q_feats, g_feats, metric="cosine")
    mAP, cmc = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)

    # Re-ranking
    mAP_rr, cmc_rr = None, None
    if rerank:
        logger.info("Computing re-ranked distance matrix...")
        distmat_rr = compute_reranking(
            q_feats, g_feats,
            k1=rerank_k1, k2=rerank_k2, lambda_value=rerank_lambda,
        )
        mAP_rr, cmc_rr = eval_market1501(
            distmat_rr, q_pids, g_pids, q_camids, g_camids
        )

    return mAP, cmc, mAP_rr, cmc_rr
