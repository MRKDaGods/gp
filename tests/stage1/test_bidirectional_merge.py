"""Tests for ReID-gated bidirectional tracklet merging."""

from __future__ import annotations

import numpy as np

from athar.core.data_models import Tracklet, TrackletFeatures, TrackletFrame
from athar.components.tracking.bidirectional import BWD_ID_OFFSET
from athar.components.tracking.bidirectional_merge import merge_bidirectional


def _norm(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def _tracklet(
    track_id: int,
    frame_ids: list[int],
    boxes: list[tuple[float, float, float, float]],
    confidences: list[float],
    *,
    camera_id: str = "cam_a",
    class_id: int = 2,
) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        camera_id=camera_id,
        class_id=class_id,
        class_name="car",
        frames=[
            TrackletFrame(
                frame_id=frame_id,
                timestamp=frame_id * 0.1,
                bbox=box,
                confidence=confidence,
            )
            for frame_id, box, confidence in zip(frame_ids, boxes, confidences)
        ],
    )


def _feature(
    tracklet: Tracklet,
    embedding: np.ndarray,
    *,
    hsv: np.ndarray | None = None,
    multi_query: np.ndarray | None = None,
) -> TrackletFeatures:
    return TrackletFeatures(
        track_id=tracklet.track_id,
        camera_id=tracklet.camera_id,
        class_id=tracklet.class_id,
        embedding=_norm(embedding),
        hsv_histogram=(
            np.asarray(hsv, dtype=np.float32)
            if hsv is not None
            else np.array([1.0, 0.0], dtype=np.float32)
        ),
        raw_embedding=np.asarray(embedding, dtype=np.float32),
        multi_query_embeddings=multi_query,
    )


def _index_map(features: list[TrackletFeatures]) -> list[dict]:
    return [
        {"track_id": feature.track_id, "camera_id": feature.camera_id, "class_id": feature.class_id}
        for feature in features
    ]


def _enabled_cfg(**overrides):
    cfg = {
        "enabled": True,
        "min_shared_frames": 3,
        "iou_thresh": 0.5,
        "cos_thresh": 0.5,
        "keep_unmatched_backward": True,
        "pool": "mean",
    }
    cfg.update(overrides)
    return cfg


def test_matching_pair_merges_union_frames_confidence_and_pooled_embedding():
    fwd = _tracklet(
        1,
        [0, 1, 2, 3],
        [(0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)],
        [0.9, 0.8, 0.4, 0.8],
    )
    bwd = _tracklet(
        BWD_ID_OFFSET + 7,
        [1, 2, 3, 4],
        [(1, 1, 11, 11), (1, 1, 11, 11), (1, 1, 11, 11), (1, 1, 11, 11)],
        [0.7, 0.95, 0.7, 0.7],
    )
    fwd_mq = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bwd_mq = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
    features = [
        _feature(fwd, np.array([1.0, 0.0]), hsv=np.array([1.0, 0.0]), multi_query=fwd_mq),
        _feature(bwd, np.array([0.8, 0.6]), hsv=np.array([0.0, 1.0]), multi_query=bwd_mq),
    ]
    primary = np.stack([feature.embedding for feature in features], axis=0)
    secondary = np.stack([np.array([1.0, 0.0]), _norm(np.array([0.6, 0.8]))], axis=0).astype(np.float32)
    tertiary = np.stack([np.array([0.0, 1.0]), _norm(np.array([0.8, 0.6]))], axis=0).astype(np.float32)
    hsv = np.stack([feature.hsv_histogram for feature in features], axis=0)

    merged_features, merged_tracklets, aligned, merged_index = merge_bidirectional(
        features,
        {"cam_a": [fwd, bwd]},
        {"primary": primary, "hsv": hsv, "secondary": secondary, "tertiary": tertiary},
        _index_map(features),
        _enabled_cfg(),
    )

    assert len(merged_features) == 1
    assert len(merged_tracklets["cam_a"]) == 1
    assert merged_index == [{"track_id": 1, "camera_id": "cam_a", "class_id": 2}]
    assert [frame.frame_id for frame in merged_tracklets["cam_a"][0].frames] == [0, 1, 2, 3, 4]
    assert merged_tracklets["cam_a"][0].get_bbox_at(2) == (1, 1, 11, 11)

    expected_primary = _norm(np.array([0.9, 0.3], dtype=np.float32))
    np.testing.assert_allclose(merged_features[0].embedding, expected_primary, atol=1e-6)
    np.testing.assert_allclose(aligned["primary"], expected_primary[np.newaxis, :], atol=1e-6)
    np.testing.assert_allclose(aligned["hsv"], np.array([[0.5, 0.5]], dtype=np.float32))
    np.testing.assert_allclose(
        aligned["secondary"],
        _norm(np.array([0.8, 0.4], dtype=np.float32))[np.newaxis, :],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        aligned["tertiary"],
        _norm(np.array([0.4, 0.8], dtype=np.float32))[np.newaxis, :],
        atol=1e-6,
    )
    assert aligned["primary"].shape[0] == len(merged_features) == len(merged_index)
    assert aligned["secondary"].shape[0] == aligned["tertiary"].shape[0] == 1
    expected_mq = np.stack([
        _norm(np.array([0.9, 0.3], dtype=np.float32)),
        _norm(np.array([0.3, 0.9], dtype=np.float32)),
    ])
    np.testing.assert_allclose(merged_features[0].multi_query_embeddings, expected_mq, atol=1e-6)


def test_low_cosine_pair_stays_separate_with_backward_retained():
    fwd = _tracklet(
        1,
        [0, 1, 2],
        [(0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)],
        [0.9, 0.9, 0.9],
    )
    bwd = _tracklet(
        BWD_ID_OFFSET + 1,
        [0, 1, 2],
        [(0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)],
        [0.9, 0.9, 0.9],
    )
    features = [_feature(fwd, np.array([1.0, 0.0])), _feature(bwd, np.array([-1.0, 0.0]))]
    primary = np.stack([feature.embedding for feature in features], axis=0)
    hsv = np.stack([feature.hsv_histogram for feature in features], axis=0)

    merged_features, merged_tracklets, aligned, merged_index = merge_bidirectional(
        features,
        {"cam_a": [fwd, bwd]},
        {"primary": primary, "hsv": hsv, "secondary": None, "tertiary": None},
        _index_map(features),
        _enabled_cfg(),
    )

    assert len(merged_features) == 2
    assert [tracklet.track_id for tracklet in merged_tracklets["cam_a"]] == [1, 2]
    assert [feature.track_id for feature in merged_features] == [1, 2]
    assert [entry["track_id"] for entry in merged_index] == [1, 2]
    assert aligned["primary"].shape[0] == 2


def test_unmatched_backward_is_kept_with_fresh_non_colliding_id():
    bwd = _tracklet(
        BWD_ID_OFFSET + 5,
        [10, 11, 12],
        [(10, 10, 20, 20), (10, 10, 20, 20), (10, 10, 20, 20)],
        [0.8, 0.8, 0.8],
    )
    features = [_feature(bwd, np.array([1.0, 0.0]))]
    primary = np.stack([feature.embedding for feature in features], axis=0)
    hsv = np.stack([feature.hsv_histogram for feature in features], axis=0)

    merged_features, merged_tracklets, aligned, merged_index = merge_bidirectional(
        features,
        {"cam_a": [bwd]},
        {"primary": primary, "hsv": hsv, "secondary": None, "tertiary": None},
        _index_map(features),
        _enabled_cfg(),
    )

    assert len(merged_features) == 1
    assert merged_tracklets["cam_a"][0].track_id == 1
    assert merged_features[0].track_id == 1
    assert merged_index[0]["track_id"] == 1
    assert merged_tracklets["cam_a"][0].track_id < BWD_ID_OFFSET
    np.testing.assert_allclose(aligned["primary"], primary)


def test_flag_off_returns_inputs_unchanged():
    fwd = _tracklet(
        1,
        [0, 1, 2],
        [(0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)],
        [0.9, 0.9, 0.9],
    )
    bwd = _tracklet(
        BWD_ID_OFFSET + 1,
        [0, 1, 2],
        [(0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)],
        [0.9, 0.9, 0.9],
    )
    features = [_feature(fwd, np.array([1.0, 0.0])), _feature(bwd, np.array([1.0, 0.0]))]
    tracklets_by_camera = {"cam_a": [fwd, bwd]}
    aligned = {
        "primary": np.stack([feature.embedding for feature in features], axis=0),
        "hsv": np.stack([feature.hsv_histogram for feature in features], axis=0),
        "secondary": None,
        "tertiary": None,
    }
    index_map = _index_map(features)

    out_features, out_tracklets, out_aligned, out_index = merge_bidirectional(
        features,
        tracklets_by_camera,
        aligned,
        index_map,
        {"enabled": False},
    )

    assert out_features is features
    assert out_tracklets is tracklets_by_camera
    assert out_aligned is aligned
    assert out_index is index_map
    assert bwd.track_id == BWD_ID_OFFSET + 1