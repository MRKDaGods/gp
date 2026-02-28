"""Tests for PCA whitening."""

import numpy as np
import pytest

from src.stage2_features.pca_whitening import PCAWhitener


def test_pca_fit_transform():
    whitener = PCAWhitener(n_components=32)
    data = np.random.randn(100, 512).astype(np.float32)
    whitener.fit(data)
    result = whitener.transform(data)
    assert result.shape == (100, 32)


def test_pca_save_load(tmp_path):
    whitener = PCAWhitener(n_components=16)
    data = np.random.randn(50, 128).astype(np.float32)
    whitener.fit(data)

    path = tmp_path / "pca.pkl"
    whitener.save(path)

    whitener2 = PCAWhitener()
    whitener2.load(path)

    result1 = whitener.transform(data)
    result2 = whitener2.transform(data)
    np.testing.assert_array_almost_equal(result1, result2)


def test_pca_reduces_components_if_needed():
    whitener = PCAWhitener(n_components=256)
    data = np.random.randn(20, 64).astype(np.float32)  # n_samples < n_components
    whitener.fit(data)
    result = whitener.transform(data)
    assert result.shape[1] <= 20  # Can't have more components than samples


def test_pca_not_fitted():
    whitener = PCAWhitener()
    with pytest.raises(RuntimeError):
        whitener.transform(np.random.randn(10, 32))
