"""Tests for core data models."""

import numpy as np

from src.core.data_models import (
    Detection,
    EvaluationResult,
    FrameInfo,
    GlobalTrajectory,
    Tracklet,
    TrackletFeatures,
    TrackletFrame,
)


def test_frame_info():
    f = FrameInfo(frame_id=0, camera_id="cam01", timestamp=0.0,
                  frame_path="/tmp/f.jpg", width=640, height=480)
    assert f.frame_id == 0
    assert f.camera_id == "cam01"


def test_detection():
    d = Detection(bbox=(10, 20, 30, 40), confidence=0.9, class_id=0, class_name="person")
    assert d.confidence == 0.9
    assert d.class_name == "person"


def test_tracklet_properties(sample_tracklet):
    assert sample_tracklet.num_frames == 30
    assert sample_tracklet.start_time == 0.0
    assert sample_tracklet.end_time == pytest.approx(2.9)
    assert sample_tracklet.duration == pytest.approx(2.9)
    assert sample_tracklet.camera_id == "cam01"


def test_tracklet_get_bbox(sample_tracklet):
    # First frame has frame_id=0
    bbox = sample_tracklet.get_bbox_at(0)
    assert bbox is not None
    assert len(bbox) == 4

    # Non-existent frame
    bbox = sample_tracklet.get_bbox_at(999)
    assert bbox is None


def test_tracklet_features():
    feat = TrackletFeatures(
        track_id=1,
        camera_id="cam01",
        class_id=0,
        embedding=np.random.randn(512).astype(np.float32),
        hsv_histogram=np.random.rand(32).astype(np.float32),
        multi_query_embeddings=np.random.randn(3, 512).astype(np.float32),
    )
    assert feat.embedding.shape == (512,)
    assert feat.hsv_histogram.shape == (32,)
    assert feat.multi_query_embeddings.shape == (3, 512)


def test_global_trajectory(sample_global_trajectory):
    traj = sample_global_trajectory
    assert traj.global_id == 0
    assert traj.num_cameras == 2
    assert len(traj.camera_sequence) == 2
    assert traj.total_duration >= 0


def test_evaluation_result():
    result = EvaluationResult(mota=0.75, idf1=0.8, hota=0.65, id_switches=10)
    assert result.mota == 0.75
    assert result.id_switches == 10


# Need pytest import for approx
import pytest
