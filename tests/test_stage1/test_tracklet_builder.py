"""Tests for tracklet builder."""

import numpy as np

from src.core.data_models import Tracklet
from src.stage1_tracking.tracklet_builder import TrackletBuilder


def test_tracklet_builder_basic():
    """Test building tracklets from raw tracker output."""
    builder = TrackletBuilder(camera_id="cam01", min_length=3, min_area=100)

    # Simulate 10 frames with 2 tracks
    for frame_id in range(10):
        tracks = np.array([
            [100, 50, 200, 300, 1, 0.9, 0],   # track_id=1, person
            [300, 100, 450, 400, 2, 0.85, 2],  # track_id=2, car
        ], dtype=np.float32)
        builder.add_frame(tracks, frame_id=frame_id, timestamp=frame_id * 0.1)

    tracklets = builder.finalize()
    assert len(tracklets) == 2

    # Verify properties
    for t in tracklets:
        assert isinstance(t, Tracklet)
        assert t.camera_id == "cam01"
        assert t.num_frames == 10


def test_tracklet_builder_filters_short():
    """Tracklets shorter than min_length should be filtered."""
    builder = TrackletBuilder(camera_id="cam01", min_length=5, min_area=100)

    # Only 3 frames for track 1
    for frame_id in range(3):
        tracks = np.array([[100, 50, 200, 300, 1, 0.9, 0]], dtype=np.float32)
        builder.add_frame(tracks, frame_id=frame_id, timestamp=frame_id * 0.1)

    tracklets = builder.finalize()
    assert len(tracklets) == 0  # filtered out


def test_tracklet_builder_filters_small_area():
    """Tracklets with small average bbox area should be filtered."""
    builder = TrackletBuilder(camera_id="cam01", min_length=3, min_area=1000)

    for frame_id in range(5):
        tracks = np.array([[100, 50, 110, 60, 1, 0.9, 0]], dtype=np.float32)  # 10x10 = 100 area
        builder.add_frame(tracks, frame_id=frame_id, timestamp=frame_id * 0.1)

    tracklets = builder.finalize()
    assert len(tracklets) == 0  # filtered out


def test_tracklet_builder_empty():
    """Empty input should produce empty output."""
    builder = TrackletBuilder(camera_id="cam01")
    tracklets = builder.finalize()
    assert tracklets == []


def test_tracklet_builder_no_tracks_in_frame():
    """Frames with no detections should be handled gracefully."""
    builder = TrackletBuilder(camera_id="cam01", min_length=1)

    for frame_id in range(5):
        empty = np.empty((0, 7), dtype=np.float32)
        builder.add_frame(empty, frame_id=frame_id, timestamp=frame_id * 0.1)

    tracklets = builder.finalize()
    assert tracklets == []
