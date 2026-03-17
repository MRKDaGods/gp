"""Tests for the stationary vehicle filter in stage 5."""

from __future__ import annotations

import pytest

from src.core.data_models import GlobalTrajectory, Tracklet, TrackletFrame
from src.stage5_evaluation.pipeline import _filter_stationary


def _make_trajectory(
    global_id: int,
    camera_id: str,
    bboxes: list[tuple[float, float, float, float]],
) -> GlobalTrajectory:
    """Create a trajectory with specified bboxes (one frame per bbox)."""
    frames = [
        TrackletFrame(frame_id=i, timestamp=float(i), bbox=bb, confidence=0.9)
        for i, bb in enumerate(bboxes)
    ]
    tracklet = Tracklet(
        track_id=1, camera_id=camera_id, class_id=2, class_name="car", frames=frames
    )
    return GlobalTrajectory(global_id=global_id, tracklets=[tracklet])


class TestStationaryFilter:
    def test_stationary_removed(self):
        """A track with near-zero displacement should be removed."""
        # Parked car: bbox barely moves (2px shift)
        t = _make_trajectory(1, "S01_c001", [
            (100.0, 200.0, 150.0, 250.0),
            (101.0, 201.0, 151.0, 251.0),
            (102.0, 200.0, 152.0, 250.0),
        ])
        result = _filter_stationary([t], min_displacement_px=50.0)
        assert len(result) == 0

    def test_moving_kept(self):
        """A track with significant displacement should be kept."""
        t = _make_trajectory(1, "S01_c001", [
            (100.0, 200.0, 150.0, 250.0),
            (200.0, 200.0, 250.0, 250.0),  # moved 100px right
        ])
        result = _filter_stationary([t], min_displacement_px=50.0)
        assert len(result) == 1

    def test_multi_tracklet_one_moving(self):
        """If ANY tracklet is moving, keep the trajectory."""
        frames_static = [
            TrackletFrame(frame_id=0, timestamp=0.0, bbox=(100, 200, 150, 250), confidence=0.9),
            TrackletFrame(frame_id=1, timestamp=1.0, bbox=(101, 201, 151, 251), confidence=0.9),
        ]
        frames_moving = [
            TrackletFrame(frame_id=0, timestamp=0.0, bbox=(100, 200, 150, 250), confidence=0.9),
            TrackletFrame(frame_id=10, timestamp=10.0, bbox=(300, 200, 350, 250), confidence=0.9),
        ]
        t = GlobalTrajectory(
            global_id=1,
            tracklets=[
                Tracklet(track_id=1, camera_id="S01_c001", class_id=2, class_name="car", frames=frames_static),
                Tracklet(track_id=2, camera_id="S01_c002", class_id=2, class_name="car", frames=frames_moving),
            ],
        )
        result = _filter_stationary([t], min_displacement_px=50.0)
        assert len(result) == 1

    def test_single_frame_tracklet_removed(self):
        """Single-frame tracklets have no displacement → stationary."""
        t = _make_trajectory(1, "S01_c001", [(100.0, 200.0, 150.0, 250.0)])
        result = _filter_stationary([t], min_displacement_px=50.0)
        assert len(result) == 0

    def test_empty_input(self):
        result = _filter_stationary([], min_displacement_px=50.0)
        assert result == []


class TestVelocityBasedStationary:
    """Test the velocity-based stationary filter enhancement."""

    def test_oscillating_parked_car_caught(self):
        """A car that arrives and parks: endpoint displacement > 50px
        but mean velocity < 1.5px/frame → caught by velocity check."""
        # First 10 frames: approaching (6px/frame), then 50 frames: parked
        bboxes = []
        for i in range(10):
            x = 100.0 + i * 6  # moving phase
            bboxes.append((x, 200.0, x + 50.0, 250.0))
        for i in range(50):
            x = 160.0 + (i % 2) * 0.5  # parked with tiny jitter
            bboxes.append((x, 200.0, x + 50.0, 250.0))
        t = _make_trajectory(1, "S01_c001", bboxes)

        # Without velocity check: kept (endpoint displacement ~60px > 50px)
        result_no_vel = _filter_stationary([t], min_displacement_px=50.0, max_mean_velocity_px=0.0)
        assert len(result_no_vel) == 1, "Should be kept without velocity check"

        # With velocity check: caught (mean velocity ~1px/frame)
        result_vel = _filter_stationary([t], min_displacement_px=50.0, max_mean_velocity_px=1.5)
        assert len(result_vel) == 0, "Should be caught by velocity check"

    def test_genuinely_moving_vehicle_kept(self):
        """A vehicle moving steadily at ~5px/frame should be kept."""
        bboxes = [(100.0 + i * 5, 200.0, 150.0 + i * 5, 250.0) for i in range(20)]
        t = _make_trajectory(1, "S01_c001", bboxes)
        result = _filter_stationary([t], min_displacement_px=50.0, max_mean_velocity_px=2.0)
        assert len(result) == 1
