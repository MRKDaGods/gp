"""Tests for HSV extractor."""

import numpy as np

from src.stage2_features.hsv_extractor import HSVExtractor


def test_hsv_histogram_shape():
    extractor = HSVExtractor(h_bins=16, s_bins=8, v_bins=8)
    crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    hist = extractor.extract_histogram(crop)
    assert hist.shape == (32,)
    assert hist.dtype == np.float32


def test_hsv_histogram_normalized():
    extractor = HSVExtractor()
    crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    hist = extractor.extract_histogram(crop)
    norm = np.linalg.norm(hist)
    np.testing.assert_almost_equal(norm, 1.0, decimal=5)


def test_hsv_tracklet_histogram():
    extractor = HSVExtractor()
    crops = [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8) for _ in range(5)]
    hist = extractor.extract_tracklet_histogram(crops)
    assert hist.shape == (32,)
    norm = np.linalg.norm(hist)
    np.testing.assert_almost_equal(norm, 1.0, decimal=5)


def test_hsv_empty_crops():
    extractor = HSVExtractor()
    hist = extractor.extract_tracklet_histogram([])
    assert hist.shape == (32,)
    assert np.all(hist == 0)
