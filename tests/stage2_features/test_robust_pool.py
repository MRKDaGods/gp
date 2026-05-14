from __future__ import annotations

import numpy as np

from src.stage2_features.robust_pool import (
    aggregate_tracklet_embeddings,
    geometric_median_pool,
    mean_pool,
    medoid_pool,
    trim_stage2_padding,
)


def test_mean_recovers_l2_normalized_arithmetic_mean():
    rows = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    pooled = mean_pool(rows)
    expected = rows.mean(axis=0)
    expected = expected / np.linalg.norm(expected)

    np.testing.assert_allclose(pooled, expected, atol=1e-6)
    assert np.isclose(np.linalg.norm(pooled), 1.0)


def test_medoid_returns_one_input_row():
    rows = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)

    pooled = medoid_pool(rows)

    assert any(np.allclose(pooled, row, atol=1e-6) for row in rows)
    assert np.isclose(np.linalg.norm(pooled), 1.0)


def test_geometric_median_converges_inside_positive_cone():
    rows = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pooled = geometric_median_pool(rows)

    assert pooled.shape == (2,)
    assert np.all(pooled >= -1e-6)
    assert np.isclose(np.linalg.norm(pooled), 1.0)


def test_short_tracklet_falls_back_to_saved_softmax_pool():
    rows = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    fallback = np.array([0.2, 0.8], dtype=np.float32)
    fallback = fallback / np.linalg.norm(fallback)

    pooled, used_fallback = aggregate_tracklet_embeddings(
        rows,
        mode="medoid",
        fallback_embedding=fallback,
        min_k=8,
    )

    assert used_fallback is True
    np.testing.assert_allclose(pooled, fallback, atol=1e-6)


def test_stage2_padding_suffix_is_trimmed_before_min_k_check():
    real_rows = np.eye(3, dtype=np.float32)
    padded_row = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
    padded_row = padded_row / np.linalg.norm(padded_row)
    rows = np.concatenate([real_rows, np.repeat(padded_row, 21, axis=0)], axis=0)
    fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    trimmed = trim_stage2_padding(rows)
    pooled, used_fallback = aggregate_tracklet_embeddings(
        rows,
        mode="mean",
        fallback_embedding=fallback,
        min_k=8,
    )

    assert trimmed.shape == (3, 3)
    assert used_fallback is True
    np.testing.assert_allclose(pooled, fallback, atol=1e-6)
