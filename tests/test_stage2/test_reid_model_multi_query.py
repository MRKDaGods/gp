"""Tests for Stage 2 multi-query embedding selection."""

from __future__ import annotations

import numpy as np

from src.stage2_features.crop_extractor import QualityScoredCrop
from src.stage2_features.reid_model import ReIDModel


def test_get_tracklet_multi_query_embeddings_selects_top_k_and_pads():
    model = object.__new__(ReIDModel)

    features = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    def _extract_features(crops, batch_size=64, cam_id=None):
        return features

    model.extract_features = _extract_features

    scored_crops = [
        QualityScoredCrop(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            quality=0.2,
            frame_id=0,
            confidence=0.9,
        ),
        QualityScoredCrop(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            quality=0.9,
            frame_id=1,
            confidence=0.95,
        ),
    ]

    mq = model.get_tracklet_multi_query_embeddings(scored_crops, k=3)

    assert mq is not None
    assert mq.shape == (3, 3)
    np.testing.assert_array_equal(mq[0], features[1])
    np.testing.assert_array_equal(mq[1], features[0])
    expected_pad = ((features[0] * np.exp(0.2 * 3.0)) + (features[1] * np.exp(0.9 * 3.0)))
    expected_pad = expected_pad / (np.exp(0.2 * 3.0) + np.exp(0.9 * 3.0))
    np.testing.assert_allclose(mq[2], expected_pad.astype(np.float32), rtol=1e-6, atol=1e-7)