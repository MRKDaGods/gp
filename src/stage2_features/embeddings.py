"""Embedding normalization and camera-aware batch normalization utilities."""

from __future__ import annotations

from typing import Dict, List

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


def camera_aware_batch_normalize(
    embeddings: np.ndarray,
    camera_ids: List[str],
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Per-camera batch normalization of embeddings.

    Different cameras have different illumination, exposure, and colour
    profiles.  Embeddings from a dark camera tend to cluster separately
    from those of a bright camera even for the same person.  This function
    zero-means and unit-variances each embedding dimension **per camera**,
    aligning the distributions before cross-camera matching.

    Args:
        embeddings: (N, D) float32 matrix.
        camera_ids: Camera ID string for each of the N embeddings.
        epsilon: Small constant to avoid division by zero.

    Returns:
        (N, D) camera-BN'd embeddings (not yet L2-normalized —
        caller should apply ``l2_normalize`` afterwards).
    """
    result = embeddings.copy()
    unique_cameras = set(camera_ids)

    if len(unique_cameras) <= 1:
        # Only one camera — global BN
        mean = result.mean(axis=0, keepdims=True)
        std = result.std(axis=0, keepdims=True) + epsilon
        return ((result - mean) / std).astype(np.float32)

    cam_array = np.array(camera_ids)
    for cam in unique_cameras:
        mask = cam_array == cam
        if mask.sum() < 5:
            # Few-tracklet camera: apply global mean/std as fallback
            # to keep the embedding in a compatible space.
            global_mean = result.mean(axis=0, keepdims=True)
            global_std = result.std(axis=0, keepdims=True) + epsilon
            result[mask] = (result[mask] - global_mean) / global_std
            continue
        cam_embeds = result[mask]
        mean = cam_embeds.mean(axis=0, keepdims=True)
        std = cam_embeds.std(axis=0, keepdims=True) + epsilon
        result[mask] = (cam_embeds - mean) / std

    return result.astype(np.float32)
