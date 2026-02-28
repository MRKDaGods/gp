"""Tests for Stage 0 preprocessor."""

import numpy as np

from src.stage0_ingestion.preprocessor import preprocess_frame


def test_preprocess_identity():
    """No-op preprocessing should return the same frame."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(frame)
    np.testing.assert_array_equal(result, frame)


def test_preprocess_resize():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(frame, target_size=(320, 240))
    assert result.shape == (240, 320, 3)


def test_preprocess_normalize():
    frame = np.ones((10, 10, 3), dtype=np.uint8) * 128
    result = preprocess_frame(frame, normalize=True)
    assert result.dtype == np.float32
    np.testing.assert_almost_equal(result[0, 0, 0], 128.0 / 255.0, decimal=4)


def test_preprocess_denoise():
    """Denoising should produce a different (smoothed) result."""
    np.random.seed(42)
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = preprocess_frame(frame, denoise=True, denoise_strength=5)
    assert result.shape == frame.shape
    # Should be different from input (denoised)
    assert not np.array_equal(result, frame)
