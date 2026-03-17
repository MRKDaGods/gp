"""Tests for per-camera whitening (FIC) and cross-camera augmentation (FAC)."""

import numpy as np
import pytest

from src.stage4_association.fic import per_camera_whiten, cross_camera_augment


class TestPerCameraWhiten:
    """Tests for FIC per-camera feature whitening."""

    def test_output_shape_unchanged(self):
        """Output shape should match input."""
        emb = np.random.randn(20, 64).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 10 + ["cam2"] * 10
        out = per_camera_whiten(emb, cams)
        assert out.shape == emb.shape

    def test_output_l2_normalized(self):
        """Whitened embeddings should be L2-normalized."""
        emb = np.random.randn(30, 64).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 15 + ["cam2"] * 15
        out = per_camera_whiten(emb, cams)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_small_camera_skipped(self):
        """Cameras with fewer than min_samples should be unchanged."""
        emb = np.random.randn(10, 32).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 8 + ["cam2"] * 2
        out = per_camera_whiten(emb, cams, min_samples=5)
        # cam2 has only 2 samples → should be unchanged (just L2-normed original)
        np.testing.assert_allclose(out[8:], emb[8:], atol=1e-6)

    def test_cross_camera_similarity_changes(self):
        """FIC should change cross-camera cosine similarities."""
        rng = np.random.RandomState(42)
        # Create features with camera-specific bias
        base = rng.randn(20, 64).astype(np.float32)
        bias1 = rng.randn(64).astype(np.float32) * 2.0
        bias2 = rng.randn(64).astype(np.float32) * 2.0
        cam1_emb = base[:10] + bias1
        cam2_emb = base[10:] + bias2
        emb = np.vstack([cam1_emb, cam2_emb])
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 10 + ["cam2"] * 10

        out = per_camera_whiten(emb, cams)

        # Cross-camera similarities should be different after whitening
        sim_before = emb[0] @ emb[15]
        sim_after = out[0] @ out[15]
        assert sim_before != pytest.approx(sim_after, abs=0.001)

    def test_single_camera_works(self):
        """Should not crash with only one camera."""
        emb = np.random.randn(10, 32).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 10
        out = per_camera_whiten(emb, cams)
        assert out.shape == emb.shape


class TestCrossCameraAugment:
    """Tests for FAC cross-camera feature augmentation."""

    def test_output_shape_unchanged(self):
        emb = np.random.randn(20, 64).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 10 + ["cam2"] * 10
        out = cross_camera_augment(emb, cams)
        assert out.shape == emb.shape

    def test_output_l2_normalized(self):
        emb = np.random.randn(30, 64).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 15 + ["cam2"] * 15
        out = cross_camera_augment(emb, cams)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_single_camera_unchanged(self):
        """With only one camera, no cross-camera neighbours exist → unchanged."""
        emb = np.random.randn(10, 32).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 10
        out = cross_camera_augment(emb, cams)
        np.testing.assert_allclose(out, emb, atol=1e-6)

    def test_features_change_with_multiple_cameras(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(20, 64).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 10 + ["cam2"] * 10
        out = cross_camera_augment(emb, cams, learning_rate=0.5)
        # Features should change
        assert not np.allclose(out, emb, atol=1e-3)

    def test_learning_rate_zero_no_change(self):
        """lr=0 means no blending → output should equal input."""
        emb = np.random.randn(20, 32).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cams = ["cam1"] * 10 + ["cam2"] * 10
        out = cross_camera_augment(emb, cams, learning_rate=0.0)
        np.testing.assert_allclose(out, emb, atol=1e-6)
