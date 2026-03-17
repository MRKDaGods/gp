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
