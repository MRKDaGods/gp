"""Feature Improvement with Camera-awareness (FIC).

Implements per-camera whitening from the AIC21 1st-place MTMC solution
(Liu et al., "City-Scale Multi-Camera Vehicle Tracking Guided by
Crossroad Zones", CVPR 2021 Workshop).

For each camera, compute the covariance of all tracklet embeddings from
that camera, invert it with Tikhonov regularisation, and transform:

    feat_new = P[cam] @ (feat - mean[cam])
    feat_new = feat_new / ||feat_new||_2

This removes camera-specific distribution bias (lighting, viewpoint,
background) that makes cross-camera cosine similarity inconsistent.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
from loguru import logger


def per_camera_whiten(
    embeddings: np.ndarray,
    camera_ids: List[str],
    regularisation: float = 3.0,
    min_samples: int = 5,
) -> np.ndarray:
    """Apply per-camera whitening (FIC) to embedding matrix.

    Args:
        embeddings: (N, D) L2-normalised embedding matrix.
        camera_ids: Camera ID for each of the N tracklets.
        regularisation: Tikhonov regularisation parameter (lambda).
            Controls the strength of whitening.  Higher = more conservative.
            AIC21 default: 3.0.
        min_samples: Minimum tracklets per camera to apply whitening.
            Cameras with fewer tracklets are left unchanged (just L2-normed).

    Returns:
        (N, D) L2-normalised whitened embeddings.
    """
    N, D = embeddings.shape
    out = embeddings.copy()

    # Group indices by camera
    cam_indices: dict[str, list[int]] = defaultdict(list)
    for idx, cam in enumerate(camera_ids):
        cam_indices[cam].append(idx)

    whitened_count = 0

    for cam, idxs in sorted(cam_indices.items()):
        idxs_arr = np.array(idxs)
        X = embeddings[idxs_arr]  # (n_cam, D)
        n_cam = X.shape[0]

        if n_cam < min_samples:
            # Too few samples — just keep original (already L2-normed)
            continue

        # Per-camera mean
        mean = X.mean(axis=0)

        # Covariance + regularisation: P = inv(X^T X + n * lambda * I)
        XtX = X.T @ X  # (D, D)
        reg = n_cam * regularisation * np.eye(D, dtype=XtX.dtype)
        P = np.linalg.inv(XtX + reg)

        # Transform: feat_new = P @ (feat - mean)
        centered = X - mean  # (n_cam, D)
        transformed = centered @ P.T  # (n_cam, D)

        # L2-normalise
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        out[idxs_arr] = transformed / norms
        whitened_count += n_cam

    logger.info(
        f"FIC per-camera whitening: {whitened_count}/{N} tracklets across "
        f"{len(cam_indices)} cameras (lambda={regularisation})"
    )

    return out


def cross_camera_augment(
    embeddings: np.ndarray,
    camera_ids: List[str],
    knn: int = 20,
    learning_rate: float = 0.5,
    beta: float = 0.08,
) -> np.ndarray:
    """Feature Augmentation with Cross-camera (FAC).

    For each embedding, find its knn nearest neighbours from *other* cameras
    and blend the feature toward those neighbours.  This makes cross-camera
    features more comparable by pulling each vector toward its cross-camera
    consensus neighbourhood.

    From AIC21 1st-place solution (ficfac.py).

    Args:
        embeddings: (N, D) L2-normalised embedding matrix (ideally after FIC).
        camera_ids: Camera ID for each of the N tracklets.
        knn: Number of cross-camera nearest neighbours.
        learning_rate: Weight of the neighbour aggregation (0-1).
            0 = no change, 1 = replace with neighbour mean.
        beta: Softmax temperature for neighbour weighting.
            Lower = more uniform weights, higher = sharper (top-1 dominated).

    Returns:
        (N, D) L2-normalised augmented embeddings.
    """
    N, D = embeddings.shape
    if N < 2:
        return embeddings.copy()

    out = embeddings.copy()

    # Precompute similarity matrix
    sim_matrix = embeddings @ embeddings.T  # (N, N) cosine similarity

    # Group indices by camera
    cam_indices: dict[str, list[int]] = defaultdict(list)
    for idx, cam in enumerate(camera_ids):
        cam_indices[cam].append(idx)

    # Build set of cross-camera indices for each tracklet
    all_indices = set(range(N))
    cam_set: dict[str, set[int]] = {
        cam: set(idxs) for cam, idxs in cam_indices.items()
    }

    augmented = 0
    for i in range(N):
        cam_i = camera_ids[i]
        # Cross-camera indices: all indices NOT in the same camera
        cross_idxs = sorted(all_indices - cam_set[cam_i])
        if len(cross_idxs) < 1:
            continue

        # Get similarities to cross-camera features
        sims = sim_matrix[i, cross_idxs]

        # Take top-knn
        k = min(knn, len(cross_idxs))
        if k < len(cross_idxs):
            top_k_local = np.argpartition(-sims, k)[:k]
        else:
            top_k_local = np.arange(k)
        top_k_sims = sims[top_k_local]

        # Softmax weights with temperature beta
        exp_sims = np.exp(top_k_sims / max(beta, 1e-8))
        weights = exp_sims / (exp_sims.sum() + 1e-12)

        # Weighted mean of cross-camera neighbours
        neighbour_indices = [cross_idxs[j] for j in top_k_local]
        neighbour_feats = embeddings[neighbour_indices]  # (k, D)
        aggregated = weights @ neighbour_feats  # (D,)

        # Blend: new = (1 - lr) * original + lr * aggregated
        blended = (1.0 - learning_rate) * embeddings[i] + learning_rate * aggregated

        # L2-normalise
        norm = np.linalg.norm(blended)
        if norm > 1e-8:
            out[i] = blended / norm
        augmented += 1

    logger.info(
        f"FAC cross-camera augmentation: {augmented}/{N} tracklets "
        f"(knn={knn}, lr={learning_rate}, beta={beta})"
    )

    return out
