"""Tests for similarity module — length weighting and combined scoring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.stage4_association.similarity import compute_combined_similarity
from src.stage4_association.spatial_temporal import SpatioTemporalValidator


def _make_st_validator() -> SpatioTemporalValidator:
    """Minimal ST validator that accepts all transitions with score 1.0."""
    transitions = {
        "cam_a": {"cam_b": {"mean_time": 5.0, "std_time": 5.0, "min_time": 0, "max_time": 60}},
        "cam_b": {"cam_a": {"mean_time": 5.0, "std_time": 5.0, "min_time": 0, "max_time": 60}},
    }
    return SpatioTemporalValidator(
        camera_transitions=transitions,
        max_time_gap=60,
        min_time_gap=0,
    )


class TestLengthWeighting:
    """Verify the min-length hyperbolic saturation formula."""

    def test_symmetric_long_tracklets_minimal_penalty(self):
        """Two long tracklets (100f each) should have nearly no penalty."""
        st = _make_st_validator()
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0  # uniform
        result = compute_combined_similarity(
            appearance_sim={(0, 1): 0.8},
            hsv_features=hsv,
            start_times=[0.0, 1.0],
            end_times=[10.0, 11.0],
            camera_ids=["cam_a", "cam_b"],
            class_ids=[2, 2],
            st_validator=st,
            weights={
                "appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0,
                "length_weight_power": 0.5,
                "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
            },
            num_frames=[100, 100],
        )
        score = result[(0, 1)]
        # min_len=100, confidence=100/110=0.909, w=0.953, score*=0.977
        # So 0.8 * 0.977 ≈ 0.78
        assert score > 0.75, f"Long+long should barely be penalized, got {score:.4f}"

    def test_asymmetric_tracklets_moderate_penalty(self):
        """Short (12f) + long (80f) should have moderate penalty, not extreme."""
        st = _make_st_validator()
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0
        result = compute_combined_similarity(
            appearance_sim={(0, 1): 0.8},
            hsv_features=hsv,
            start_times=[0.0, 1.0],
            end_times=[1.2, 9.0],
            camera_ids=["cam_a", "cam_b"],
            class_ids=[2, 2],
            st_validator=st,
            weights={
                "appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0,
                "length_weight_power": 0.5,
                "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
            },
            num_frames=[12, 80],
        )
        score = result[(0, 1)]
        # min_len=12, confidence=12/22=0.545, w=0.739, score*=0.869
        # So 0.8 * 0.869 ≈ 0.696
        assert score > 0.65, f"Short+long should not be over-penalized, got {score:.4f}"
        # Old ratio formula: ratio=12/80=0.15, w=0.387, score*=0.694, result=0.555
        # New formula should be significantly less harsh
        assert score > 0.555, f"New formula should beat old ratio approach, got {score:.4f}"

    def test_very_short_tracklets_penalized(self):
        """Two very short tracklets (3f each) should be notably penalized."""
        st = _make_st_validator()
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0
        result = compute_combined_similarity(
            appearance_sim={(0, 1): 0.8},
            hsv_features=hsv,
            start_times=[0.0, 1.0],
            end_times=[0.3, 1.3],
            camera_ids=["cam_a", "cam_b"],
            class_ids=[2, 2],
            st_validator=st,
            weights={
                "appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0,
                "length_weight_power": 0.5,
                "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
            },
            num_frames=[3, 3],
        )
        score = result[(0, 1)]
        # min_len=3, confidence=3/13=0.231, w=0.480, score*=0.740
        # So 0.8 * 0.740 ≈ 0.592
        assert score < 0.75, f"Very short should be penalized, got {score:.4f}"

    def test_no_length_weight_when_power_zero(self):
        """When length_weight_power=0, score should be unmodified by length."""
        st = _make_st_validator()
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0
        result = compute_combined_similarity(
            appearance_sim={(0, 1): 0.8},
            hsv_features=hsv,
            start_times=[0.0, 1.0],
            end_times=[0.3, 1.3],
            camera_ids=["cam_a", "cam_b"],
            class_ids=[2, 2],
            st_validator=st,
            weights={
                "appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0,
                "length_weight_power": 0.0,
                "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
            },
            num_frames=[3, 3],
        )
        score_no_lw = result[(0, 1)]
        # With power=0, length weighting is disabled, so score should be
        # purely from appearance + HSV + ST (no length multiplier).
        # With appearance=1.0, hsv=0.0, st=0.0: score = 1.0 * 0.8 + 0 + 0 = 0.8 * st_score
        # The st_score from transition_score should be > 0, so score = 0.8 * st_where-thing

        # Just verify it matches the no-num_frames case
        result_none = compute_combined_similarity(
            appearance_sim={(0, 1): 0.8},
            hsv_features=hsv,
            start_times=[0.0, 1.0],
            end_times=[0.3, 1.3],
            camera_ids=["cam_a", "cam_b"],
            class_ids=[2, 2],
            st_validator=st,
            weights={
                "appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0,
                "length_weight_power": 0.0,
                "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
            },
            num_frames=None,
        )
        score_none = result_none[(0, 1)]
        assert abs(score_no_lw - score_none) < 1e-6, (
            f"power=0 should match no-num_frames: {score_no_lw:.6f} vs {score_none:.6f}"
        )


class TestTemporalOverlap:
    def test_overlap_bonus_applies_without_pair_prior_for_overlapping_fov(self):
        st = SpatioTemporalValidator(
            camera_transitions=None,
            max_time_gap=60,
            min_time_gap=0,
        )
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0
        result = compute_combined_similarity(
            appearance_sim={(0, 1): 0.8},
            hsv_features=hsv,
            start_times=[0.0, 5.0],
            end_times=[10.0, 12.0],
            camera_ids=["cam_a", "cam_b"],
            class_ids=[2, 2],
            st_validator=st,
            weights={
                "appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0,
                "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
            },
            temporal_overlap_cfg={"enabled": True, "bonus": 0.15, "max_mean_time": 5.0},
        )
        assert result[(0, 1)] == pytest.approx(0.9071428571428571)

    def test_overlap_min_ratio_filters_cross_camera_pair(self):
        st = _make_st_validator()
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0
        result = compute_combined_similarity(
            appearance_sim={(0, 1): 0.8},
            hsv_features=hsv,
            start_times=[0.0, 9.0],
            end_times=[10.0, 19.0],
            camera_ids=["cam_a", "cam_b"],
            class_ids=[2, 2],
            st_validator=st,
            weights={
                "appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0,
                "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
            },
            temporal_overlap_cfg={"enabled": True, "min_ratio": 0.3},
        )
        assert result == {}

    def test_overlap_min_ratio_does_not_filter_same_camera_pair(self):
        st = SpatioTemporalValidator(
            camera_transitions=None,
            max_time_gap=60,
            min_time_gap=0,
        )
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0
        result = compute_combined_similarity(
            appearance_sim={(0, 1): 0.8},
            hsv_features=hsv,
            start_times=[0.0, 9.0],
            end_times=[10.0, 10.0],
            camera_ids=["cam_a", "cam_a"],
            class_ids=[2, 2],
            st_validator=st,
            weights={
                "appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0,
                "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
            },
            temporal_overlap_cfg={"enabled": True, "min_ratio": 0.3},
        )
        assert result[(0, 1)] == pytest.approx(0.8)
