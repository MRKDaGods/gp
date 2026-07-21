"""Tests for the geospatial reachability constraint (stage 4)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from athar.components.associators.geospatial import (
    GeoSpatialConstraint,
    GeoSpeedProfile,
    build_geo_constraint_from_config,
    build_google_maps_directions_url,
    build_maps_path_share_url,
    haversine_m,
    load_camera_coordinates,
    prune_unreachable_pairs,
)
from athar.components.associators.similarity import compute_combined_similarity
from athar.components.associators.spatial_temporal import SpatioTemporalValidator


# A small synthetic camera network along the equator, where 1 degree of
# longitude ~ 111.2 km, so distances are easy to reason about.
_M_PER_DEG = haversine_m(0.0, 0.0, 0.0, 1.0)  # metres per degree lng at equator


def _lng_for_metres(metres: float) -> float:
    return metres / _M_PER_DEG


def _network() -> dict:
    """cam_a at origin; cam_b ~100 m east; cam_c ~300 m east; cam_far ~5 km east."""
    return {
        "cam_a": (0.0, 0.0),
        "cam_b": (0.0, _lng_for_metres(100.0)),
        "cam_c": (0.0, _lng_for_metres(300.0)),
        "cam_far": (0.0, _lng_for_metres(5000.0)),
    }


class TestHaversine:
    def test_one_degree_at_equator(self):
        d = haversine_m(0.0, 0.0, 0.0, 1.0)
        assert abs(d - 111195.0) < 50.0, f"expected ~111195 m, got {d:.1f}"

    def test_identical_points_zero(self):
        assert haversine_m(12.34, 56.78, 12.34, 56.78) == 0.0

    def test_symmetric(self):
        a = haversine_m(40.0, -3.0, 41.0, -4.0)
        b = haversine_m(41.0, -4.0, 40.0, -3.0)
        assert a == pytest.approx(b)

    def test_no_domain_error_for_antipodal(self):
        # asin clamp must keep this finite, not raise a math domain error.
        d = haversine_m(0.0, 0.0, 0.0, 180.0)
        assert np.isfinite(d) and d > 0


class TestLoadCoordinates:
    def test_valid_and_invalid_records(self, tmp_path):
        p = tmp_path / "camera_coordinates.json"
        p.write_text(json.dumps({
            "cam_a": {"lat": 1.0, "lng": 2.0, "label": "x"},
            "cam_b": {"lat": "3.0", "lng": "4.0"},        # numeric strings ok
            "bad_missing": {"lat": 1.0},                   # no lng -> skip
            "bad_range": {"lat": 999.0, "lng": 0.0},       # out of range -> skip
            "bad_type": "nope",                            # not a dict -> skip
        }), encoding="utf-8")
        coords = load_camera_coordinates(p)
        assert set(coords) == {"cam_a", "cam_b"}
        assert coords["cam_b"] == (3.0, 4.0)

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_camera_coordinates(tmp_path / "nope.json") == {}

    def test_malformed_json_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        assert load_camera_coordinates(p) == {}


class TestActivation:
    def test_inactive_with_one_camera(self):
        gc = GeoSpatialConstraint({"only": (0.0, 0.0)})
        assert not gc.is_active

    def test_active_with_two(self):
        gc = GeoSpatialConstraint(_network())
        assert gc.is_active
        assert gc.has_coords("cam_a") and not gc.has_coords("ghost")

    def test_distance_none_for_unknown(self):
        gc = GeoSpatialConstraint(_network())
        assert gc.distance_m("cam_a", "ghost") is None
        assert gc.distance_m("cam_a", "cam_a") == 0.0
        assert gc.distance_m("cam_a", "cam_b") == pytest.approx(100.0, abs=1.0)


class TestReachability:
    def test_person_cannot_cover_100m_in_5s_but_vehicle_can(self):
        gc = GeoSpatialConstraint(_network())
        # person: 8 m/s * 1.5 margin * 5 s = 60 m reach < 100 m -> unreachable
        assert not gc.is_reachable("cam_a", "cam_b", 5.0, class_id=0)
        # vehicle: 40 m/s * 1.5 * 5 s = 300 m reach > 100 m -> reachable
        assert gc.is_reachable("cam_a", "cam_b", 5.0, class_id=2)

    def test_person_reaches_with_enough_time(self):
        gc = GeoSpatialConstraint(_network())
        # 100 m at 8*1.5 m/s needs ~8.3 s; give 20 s -> reachable
        assert gc.is_reachable("cam_a", "cam_b", 20.0, class_id=0)

    def test_simultaneous_only_overlapping_fov(self):
        gc = GeoSpatialConstraint(_network(), overlap_fov_radius_m=150.0)
        # gap ~0: cam_b (100 m) within overlap*1.5=225 m -> reachable
        assert gc.is_reachable("cam_a", "cam_b", 0.0, class_id=2)
        # cam_c (300 m) beyond overlap radius -> not co-observable now
        assert not gc.is_reachable("cam_a", "cam_c", 0.0, class_id=2)

    def test_unknown_geometry_never_gated(self):
        gc = GeoSpatialConstraint(_network())
        assert gc.is_reachable("cam_a", "ghost", 0.1, class_id=0)

    def test_same_camera_reachable(self):
        gc = GeoSpatialConstraint(_network())
        assert gc.is_reachable("cam_a", "cam_a", 0.0, class_id=0)

    def test_reachable_radius_grows_faster_for_vehicles(self):
        gc = GeoSpatialConstraint(_network())
        assert gc.reachable_radius_m(10.0, class_id=2) > gc.reachable_radius_m(10.0, class_id=0)


class TestGeoScore:
    def test_near_beats_far_at_same_time(self):
        gc = GeoSpatialConstraint(_network())
        near = gc.geo_score("cam_a", "cam_b", 10.0, class_id=2)   # 100 m
        far = gc.geo_score("cam_a", "cam_c", 10.0, class_id=2)    # 300 m
        assert 0.0 < far < near <= 1.0

    def test_unreachable_scores_zero(self):
        gc = GeoSpatialConstraint(_network())
        assert gc.geo_score("cam_a", "cam_far", 1.5, class_id=2) == 0.0

    def test_unknown_geometry_neutral(self):
        gc = GeoSpatialConstraint(_network())
        assert gc.geo_score("cam_a", "ghost", 5.0, class_id=0) == 1.0

    def test_same_camera_neutral(self):
        gc = GeoSpatialConstraint(_network())
        assert gc.geo_score("cam_a", "cam_a", 5.0, class_id=0) == 1.0

    def test_score_in_unit_interval(self):
        gc = GeoSpatialConstraint(_network())
        for dt in (0.0, 0.5, 2.0, 30.0):
            for cam in ("cam_b", "cam_c", "cam_far"):
                s = gc.geo_score("cam_a", cam, dt, class_id=2)
                assert 0.0 <= s <= 1.0


class TestExpandingRings:
    def test_cameras_by_distance_sorted(self):
        gc = GeoSpatialConstraint(_network())
        ordered = gc.cameras_by_distance("cam_a")
        cams = [c for c, _ in ordered]
        assert cams == ["cam_b", "cam_c", "cam_far"]
        dists = [d for _, d in ordered]
        assert dists == sorted(dists)

    def test_rings_group_by_width(self):
        gc = GeoSpatialConstraint(_network())
        rings = gc.expanding_rings("cam_a", ring_width_m=200.0)
        # ring 0: [0,200) -> cam_b(100); ring 1: [200,400) -> cam_c(300);
        # ring 25: cam_far(5000)
        assert rings[0] == ["cam_b"]
        assert rings[1] == ["cam_c"]
        assert rings[-1] == ["cam_far"]

    def test_ring_width_must_be_positive(self):
        gc = GeoSpatialConstraint(_network())
        with pytest.raises(ValueError):
            gc.expanding_rings("cam_a", ring_width_m=0.0)


class TestPrunePairs:
    def test_prunes_unreachable_keeps_reachable(self):
        gc = GeoSpatialConstraint(_network())
        camera_ids = ["cam_a", "cam_far"]
        class_ids = [2, 2]
        start_times = [0.0, 2.0]
        end_times = [1.0, 3.0]
        # min_gap = max(0, 2-1) = 1 s; cam_far is 5 km away -> unreachable
        pairs = [(0, 1, 0.9)]
        kept, removed = prune_unreachable_pairs(
            pairs, camera_ids, class_ids, start_times, end_times, gc
        )
        assert removed == 1 and kept == []

    def test_keeps_same_camera_and_unknown(self):
        gc = GeoSpatialConstraint(_network())
        camera_ids = ["cam_a", "cam_a", "ghost"]
        class_ids = [2, 2, 2]
        start_times = [0.0, 0.0, 0.0]
        end_times = [1.0, 1.0, 1.0]
        pairs = [(0, 1, 0.9), (0, 2, 0.9)]
        kept, removed = prune_unreachable_pairs(
            pairs, camera_ids, class_ids, start_times, end_times, gc
        )
        assert removed == 0 and len(kept) == 2

    def test_inactive_constraint_keeps_all(self):
        gc = GeoSpatialConstraint({"solo": (0.0, 0.0)})
        pairs = [(0, 1, 0.5)]
        kept, removed = prune_unreachable_pairs(
            pairs, ["solo", "x"], [2, 2], [0.0, 9.0], [1.0, 10.0], gc
        )
        assert removed == 0 and kept == pairs


class TestMapsUrl:
    def test_empty_path_none(self):
        assert build_maps_path_share_url([]) is None

    def test_single_point_search_url(self):
        url = build_maps_path_share_url([(10.0, 20.0)])
        assert url is not None
        assert url.startswith("https://www.google.com/maps/search/")
        assert "10.0" in url and "20.0" in url

    def test_multi_point_directions(self):
        url = build_maps_path_share_url([(1.0, 2.0), (3.0, 4.0)])
        assert url.startswith("https://www.google.com/maps/dir/")
        assert "origin=1.0%2C2.0" in url
        assert "destination=3.0%2C4.0" in url
        assert "travelmode=driving" in url

    def test_waypoints_for_three_points(self):
        url = build_google_maps_directions_url(["1,1", "2,2", "3,3"])
        assert "waypoints=" in url and "2%2C2" in url

    def test_dedupe_consecutive_collapses_to_single(self):
        # All identical -> collapses to one distinct point -> search URL
        url = build_maps_path_share_url([(5.0, 6.0), (5.0, 6.0), (5.0, 6.0)])
        assert url.startswith("https://www.google.com/maps/search/")

    def test_long_path_stays_under_cap(self):
        path = [(12.0 + i * 0.0011, 98.0 + i * 0.0013) for i in range(120)]
        url = build_maps_path_share_url(path)
        assert url is not None
        assert len(url) <= 1800
        assert url.startswith("https://www.google.com/maps/dir/")


class TestConfigBuilder:
    def test_inline_coordinates(self):
        cfg = {
            "camera_coordinates": {
                "cam_a": {"lat": 0.0, "lng": 0.0},
                "cam_b": {"lat": 0.0, "lng": 0.01},
            },
        }
        gc = build_geo_constraint_from_config(cfg)
        assert gc is not None and gc.is_active

    def test_path_coordinates(self, tmp_path):
        p = tmp_path / "camera_coordinates.json"
        p.write_text(json.dumps({
            "cam_a": {"lat": 0.0, "lng": 0.0},
            "cam_b": {"lat": 0.0, "lng": 0.01},
        }), encoding="utf-8")
        gc = build_geo_constraint_from_config({"camera_coordinates_path": str(p)})
        assert gc is not None and gc.is_active

    def test_too_few_coords_returns_none(self):
        gc = build_geo_constraint_from_config(
            {"camera_coordinates": {"cam_a": {"lat": 0.0, "lng": 0.0}}}
        )
        assert gc is None

    def test_speed_overrides_applied(self):
        cfg = {
            "camera_coordinates": {
                "cam_a": {"lat": 0.0, "lng": 0.0},
                "cam_b": {"lat": 0.0, "lng": 0.01},
            },
            "person": {"max_speed_ms": 3.0, "typical_speed_ms": 1.0},
            "vehicle": {"max_speed_ms": 50.0, "typical_speed_ms": 15.0},
        }
        gc = build_geo_constraint_from_config(cfg)
        assert gc.profile_for(0) == GeoSpeedProfile(3.0, 1.0)
        assert gc.profile_for(2) == GeoSpeedProfile(50.0, 15.0)


class TestSimilarityIntegration:
    """End-to-end: geo gate + score modulation inside compute_combined_similarity."""

    _WEIGHTS = {
        "appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0,
        "person": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
        "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
    }

    def _st(self) -> SpatioTemporalValidator:
        return SpatioTemporalValidator(min_time_gap=0, max_time_gap=300, camera_transitions=None)

    def test_gate_drops_unreachable_person_pair(self):
        gc = GeoSpatialConstraint(_network())
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0
        result = compute_combined_similarity(
            appearance_sim={(0, 1): 0.9},
            hsv_features=hsv,
            start_times=[0.0, 10.0],   # min_gap = 5 s
            end_times=[5.0, 15.0],
            camera_ids=["cam_a", "cam_b"],   # 100 m apart
            class_ids=[0, 0],                # person -> 60 m reach in 5 s
            st_validator=self._st(),
            weights=self._WEIGHTS,
            geo_constraint=gc,
            geo_gate=True,
        )
        assert result == {}, "person pair 100 m apart in 5 s should be gated out"

    def test_reachable_vehicle_pair_kept_and_modulated(self):
        gc = GeoSpatialConstraint(_network())
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0
        common = dict(
            appearance_sim={(0, 1): 0.9},
            hsv_features=hsv,
            start_times=[0.0, 10.0],   # min_gap = 5 s
            end_times=[5.0, 15.0],
            camera_ids=["cam_a", "cam_b"],
            class_ids=[2, 2],           # vehicle -> 300 m reach in 5 s -> reachable
            st_validator=self._st(),
            weights=self._WEIGHTS,
            geo_constraint=gc,
            geo_gate=True,
        )
        unweighted = compute_combined_similarity(**common, geo_weight=0.0)
        weighted = compute_combined_similarity(**common, geo_weight=0.5)
        assert (0, 1) in unweighted and (0, 1) in weighted
        # weight=0 leaves the appearance score untouched
        assert unweighted[(0, 1)] == pytest.approx(0.9)
        # weight>0 pulls the score down toward geo plausibility (required speed
        # 20 m/s vs 11 m/s typical -> geo_score < 1) but keeps it positive
        assert 0.0 < weighted[(0, 1)] < unweighted[(0, 1)]

    def test_no_constraint_is_noop(self):
        hsv = np.ones((2, 16), dtype=np.float32) / 4.0
        result = compute_combined_similarity(
            appearance_sim={(0, 1): 0.9},
            hsv_features=hsv,
            start_times=[0.0, 10.0],
            end_times=[5.0, 15.0],
            camera_ids=["cam_a", "cam_far"],
            class_ids=[0, 0],
            st_validator=self._st(),
            weights=self._WEIGHTS,
            geo_constraint=None,
        )
        assert result[(0, 1)] == pytest.approx(0.9)
