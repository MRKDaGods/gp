"""Tests for I/O utilities."""

import json

import numpy as np
import pytest

from src.core.data_models import (
    EvaluationResult,
    FrameInfo,
    GlobalTrajectory,
    Tracklet,
    TrackletFrame,
)
from src.core.io_utils import (
    load_embeddings,
    load_evaluation_result,
    load_frame_manifest,
    load_global_trajectories,
    load_multi_query_embeddings,
    load_tracklets,
    save_embeddings,
    save_evaluation_result,
    save_frame_manifest,
    save_global_trajectories,
    save_multi_query_embeddings,
    save_tracklets,
)


def test_frame_manifest_roundtrip(tmp_path):
    frames = [
        FrameInfo(frame_id=0, camera_id="cam01", timestamp=0.0,
                  frame_path="/tmp/f0.jpg", width=640, height=480),
        FrameInfo(frame_id=1, camera_id="cam01", timestamp=0.1,
                  frame_path="/tmp/f1.jpg", width=640, height=480),
    ]
    path = tmp_path / "manifest.json"
    save_frame_manifest(frames, path)
    loaded = load_frame_manifest(path)
    assert len(loaded) == 2
    assert loaded[0].frame_id == 0
    assert loaded[1].timestamp == 0.1


def test_tracklets_roundtrip(tmp_path):
    tracklets = [
        Tracklet(
            track_id=1, camera_id="cam01", class_id=0, class_name="person",
            frames=[
                TrackletFrame(frame_id=0, timestamp=0.0, bbox=(10, 20, 30, 40), confidence=0.9),
                TrackletFrame(frame_id=3, timestamp=0.1, bbox=(12, 22, 32, 42), confidence=0.85),
            ],
        ),
    ]
    path = tmp_path / "tracklets.json"
    save_tracklets(tracklets, path)
    loaded = load_tracklets(path)
    assert len(loaded) == 1
    assert loaded[0].track_id == 1
    assert len(loaded[0].frames) == 2
    assert loaded[0].frames[0].bbox == (10, 20, 30, 40)


def test_embeddings_roundtrip(tmp_path):
    embeddings = np.random.randn(10, 512).astype(np.float32)
    index_map = [{"track_id": i, "camera_id": "cam01", "class_id": 0} for i in range(10)]

    save_embeddings(embeddings, index_map, tmp_path)
    loaded_emb, loaded_idx = load_embeddings(tmp_path)

    np.testing.assert_array_almost_equal(loaded_emb, embeddings)
    assert len(loaded_idx) == 10


def test_global_trajectories_roundtrip(tmp_path):
    t1 = Tracklet(
        track_id=0, camera_id="cam01", class_id=2, class_name="car",
        frames=[TrackletFrame(frame_id=0, timestamp=0.0, bbox=(10, 20, 30, 40), confidence=0.9)],
    )
    t2 = Tracklet(
        track_id=1, camera_id="cam02", class_id=2, class_name="car",
        frames=[TrackletFrame(frame_id=5, timestamp=0.5, bbox=(50, 60, 70, 80), confidence=0.88)],
    )

    trajectories = [GlobalTrajectory(global_id=0, tracklets=[t1, t2])]
    path = tmp_path / "trajectories.json"
    save_global_trajectories(trajectories, path)
    loaded = load_global_trajectories(path)

    assert len(loaded) == 1
    assert loaded[0].global_id == 0
    assert len(loaded[0].tracklets) == 2


def test_multi_query_embeddings_roundtrip(tmp_path):
    mq_embeddings = [
        np.random.randn(4, 32).astype(np.float32),
        np.random.randn(4, 32).astype(np.float32),
        np.random.randn(4, 32).astype(np.float32),
    ]

    save_multi_query_embeddings(mq_embeddings, tmp_path)
    loaded = load_multi_query_embeddings(tmp_path, n=3)

    assert len(loaded) == 3
    for expected, actual in zip(mq_embeddings, loaded):
        assert actual is not None
        np.testing.assert_array_almost_equal(actual, expected)


def test_evaluation_result_roundtrip(tmp_path):
    result = EvaluationResult(mota=0.75, idf1=0.8, hota=0.65, id_switches=10,
                               details={"per_camera": {"cam01": 0.8}})
    path = tmp_path / "eval.json"
    save_evaluation_result(result, path)
    loaded = load_evaluation_result(path)
    assert loaded.mota == 0.75
    assert loaded.id_switches == 10
