"""Tests for camera distance bias and zone transition model."""

from __future__ import annotations

import json
import numpy as np
import pytest

from src.stage4_association.camera_bias import CameraDistanceBias, ZoneTransitionModel


class TestCameraDistanceBias:
    """Tests for CameraDistanceBias learning and adjustment."""

    def _make_test_data(self):
        """Create test clusters and similarities."""
        # 6 tracklets: 0-2 from cam01, 3-5 from cam02
        camera_ids = ["cam01", "cam01", "cam01", "cam02", "cam02", "cam02"]
        # Two clusters: {0,3} and {1,4} are same identity, {2,5} also
        clusters = [{0, 3}, {1, 4}, {2, 5}]
        # Cross-camera similarities
        similarities = {
            (0, 3): 0.7,
            (1, 4): 0.8,
            (2, 5): 0.75,
            # Intra-camera (should be ignored)
            (0, 1): 0.9,
            (3, 4): 0.85,
        }
        return similarities, camera_ids, clusters

    def test_learn_from_matches(self):
        """Bias is learned from cluster matches."""
        bias = CameraDistanceBias()
        sims, cams, clusters = self._make_test_data()
        bias.learn_from_matches(sims, cams, clusters)
        # Should learn bias for cam01-cam02 pair
        b = bias.get_bias("cam01", "cam02")
        assert b > 0
        # Median of [0.7, 0.8, 0.75] = 0.75
        assert abs(b - 0.75) < 0.01

    def test_get_bias_unknown_pair(self):
        """Unknown camera pairs return 0."""
        bias = CameraDistanceBias()
        assert bias.get_bias("cam99", "cam100") == 0.0

    def test_get_bias_order_invariant(self):
        """Bias is the same regardless of camera order."""
        bias = CameraDistanceBias()
        sims, cams, clusters = self._make_test_data()
        bias.learn_from_matches(sims, cams, clusters)
        assert bias.get_bias("cam01", "cam02") == bias.get_bias("cam02", "cam01")

    def test_adjust_similarity(self):
        """Adjustment shifts similarity toward global mean."""
        bias = CameraDistanceBias()
        sims, cams, clusters = self._make_test_data()
        bias.learn_from_matches(sims, cams, clusters)

        # Similarity equal to bias → should map to global_mean
        adjusted = bias.adjust_similarity(0.75, "cam01", "cam02", global_mean=0.5)
        assert abs(adjusted - 0.5) < 0.01

    def test_adjust_similarity_clamps(self):
        """Adjusted scores are clamped to [0, 1]."""
        bias = CameraDistanceBias()
        bias._bias = {("cam01", "cam02"): 0.3}
        assert bias.adjust_similarity(0.0, "cam01", "cam02", global_mean=0.1) >= 0.0
        assert bias.adjust_similarity(1.0, "cam01", "cam02", global_mean=0.9) <= 1.0

    def test_adjust_similarity_matrix(self):
        """Matrix adjustment adjusts all cross-camera pairs."""
        bias = CameraDistanceBias()
        sims, cams, clusters = self._make_test_data()
        bias.learn_from_matches(sims, cams, clusters)
        adjusted = bias.adjust_similarity_matrix(
            {(0, 3): 0.7, (1, 4): 0.8}, cams
        )
        assert len(adjusted) == 2
        # All values should be valid floats
        for v in adjusted.values():
            assert 0.0 <= v <= 1.0

    def test_save_load(self, tmp_path):
        """Bias can be saved and loaded."""
        bias = CameraDistanceBias()
        sims, cams, clusters = self._make_test_data()
        bias.learn_from_matches(sims, cams, clusters)

        path = tmp_path / "bias.json"
        bias.save(path)

        bias2 = CameraDistanceBias()
        bias2.load(path)
        assert bias2.get_bias("cam01", "cam02") == bias.get_bias("cam01", "cam02")

    def test_minimum_samples_required(self):
        """Bias not learned with fewer than 3 samples."""
        bias = CameraDistanceBias()
        # Only 2 cross-camera similarities
        sims = {(0, 2): 0.7, (1, 3): 0.8}
        cams = ["cam01", "cam01", "cam02", "cam02"]
        clusters = [{0, 2}, {1, 3}]
        bias.learn_from_matches(sims, cams, clusters)
        # Not enough samples for bias
        assert bias.get_bias("cam01", "cam02") == 0.0


class TestZoneTransitionModel:
    """Tests for zone-based transition model."""

    def test_add_zone(self):
        """Zones can be added for cameras."""
        model = ZoneTransitionModel()
        model.add_zone("cam01", "zone_A", (0.0, 0.0, 0.3, 1.0), zone_type="exit")
        assert "cam01" in model._zones
        assert "zone_A" in model._zones["cam01"]

    def test_add_transition(self):
        """Transitions can be added between zones."""
        model = ZoneTransitionModel()
        model.add_zone("cam01", "zone_A", (0.0, 0.0, 0.3, 1.0), zone_type="exit")
        model.add_zone("cam02", "zone_B", (0.7, 0.0, 1.0, 1.0), zone_type="entry")
        model.add_transition("cam01", "zone_A", "cam02", "zone_B", min_time=5.0, max_time=60.0)
        key = ("cam01", "zone_A", "cam02", "zone_B")
        assert key in model._transitions

    def test_classify_zone_for_point(self):
        """Point is correctly classified into a zone."""
        model = ZoneTransitionModel()
        # Left zone
        model.add_zone("cam01", "left", (0.0, 0.0, 0.3, 1.0))
        # Right zone
        model.add_zone("cam01", "right", (0.7, 0.0, 1.0, 1.0))

        # Point at (0.15, 0.5) → should be in "left" zone
        zone = model.classify_zone("cam01", 0.15, 0.5)
        assert zone == "left"

        # Point at (0.85, 0.3) → should be in "right" zone
        zone = model.classify_zone("cam01", 0.85, 0.3)
        assert zone == "right"

    def test_classify_zone_no_match(self):
        """Returns None when point is not in any zone."""
        model = ZoneTransitionModel()
        model.add_zone("cam01", "left", (0.0, 0.0, 0.2, 0.5))
        zone = model.classify_zone("cam01", 0.5, 0.7)
        assert zone is None

    def test_transition_score(self):
        """Transition scoring with time validation."""
        model = ZoneTransitionModel()
        model.add_zone("cam01", "zone_A", (0.0, 0.0, 0.3, 1.0), zone_type="exit")
        model.add_zone("cam02", "zone_B", (0.7, 0.0, 1.0, 1.0), zone_type="entry")
        model.add_transition("cam01", "zone_A", "cam02", "zone_B", min_time=5.0, max_time=60.0)

        # Valid time
        score = model.get_transition_score("cam01", "zone_A", "cam02", "zone_B", 30.0)
        assert score == 1.0

        # Too fast
        score = model.get_transition_score("cam01", "zone_A", "cam02", "zone_B", 2.0)
        assert score == 0.0

        # Unknown transition
        score = model.get_transition_score("cam02", "zone_B", "cam01", "zone_A", 30.0)
        assert score == 0.0

    def test_load_from_config(self):
        """Zone model loads from config dict."""
        config = {
            "zones": {
                "cam01": {"exit": {"bbox": [0.0, 0.7, 0.3, 1.0], "type": "exit"}},
                "cam02": {"entry": {"bbox": [0.7, 0.0, 1.0, 0.3], "type": "entry"}},
            },
            "transitions": [
                {"src_cam": "cam01", "src_zone": "exit",
                 "dst_cam": "cam02", "dst_zone": "entry",
                 "min_time": 10, "max_time": 120},
            ],
        }
        model = ZoneTransitionModel()
        model.load_from_config(config)
        assert "cam01" in model._zones
        assert "cam02" in model._zones
        assert len(model._transitions) == 1
