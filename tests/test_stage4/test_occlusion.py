"""Tests for occlusion-aware Stage 4 association."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.data_models import Tracklet, TrackletFrame
from src.stage4_association.occlusion import compute_tracklet_occlusion
from src.stage4_association.similarity import compute_combined_similarity
from src.stage4_association.spatial_temporal import SpatioTemporalValidator


def _tracklet(track_id, boxes, confidences=None, frame_ids=None) -> Tracklet:
    if confidences is None:
        confidences = [0.9] * len(boxes)
    if frame_ids is None:
        frame_ids = list(range(len(boxes)))
    return Tracklet(
        track_id=track_id,
        camera_id="cam_a",
        class_id=2,
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


def _combined_score(occluded_flags=None, occ_penalty=1.0):
    hsv = np.ones((2, 16), dtype=np.float32) / 4.0
    st = SpatioTemporalValidator(camera_transitions=None, min_time_gap=0, max_time_gap=60)
    return compute_combined_similarity(
        appearance_sim={(0, 1): 0.8},
        hsv_features=hsv,
        start_times=[0.0, 1.0],
        end_times=[1.0, 2.0],
        camera_ids=["cam_a", "cam_b"],
        class_ids=[2, 2],
        st_validator=st,
        weights={
            "appearance": 1.0,
            "hsv": 0.0,
            "spatiotemporal": 0.0,
            "vehicle": {"appearance": 1.0, "hsv": 0.0, "spatiotemporal": 0.0},
        },
        occluded_flags=occluded_flags,
        occ_penalty=occ_penalty,
    )[(0, 1)]


def test_tracklet_occlusion_uses_per_frame_max_iou_fraction():
    tracklet_a = _tracklet(
        1,
        boxes=[
            (0.0, 0.0, 10.0, 10.0),
            (100.0, 100.0, 110.0, 110.0),
        ],
    )
    tracklet_b = _tracklet(
        2,
        boxes=[(1.0, 1.0, 11.0, 11.0)],
        frame_ids=[0],
    )

    result = compute_tracklet_occlusion(
        {"cam_a": [tracklet_a, tracklet_b]},
        {"occ_box_thresh": 0.6, "occ_frac_thresh": 0.5},
    )

    assert result[("cam_a", 1)] is True
    assert result[("cam_a", 2)] is True


def test_tracklet_occlusion_single_box_frames_are_not_occluded():
    tracklet = _tracklet(
        1,
        boxes=[
            (0.0, 0.0, 10.0, 10.0),
            (100.0, 100.0, 110.0, 110.0),
        ],
    )

    result = compute_tracklet_occlusion(
        {"cam_a": [tracklet]},
        {"occ_box_thresh": 0.6, "occ_frac_thresh": 0.3},
    )

    assert result[("cam_a", 1)] is False


def test_tracklet_occlusion_ignores_interpolated_boxes():
    interpolated = _tracklet(
        1,
        boxes=[(0.0, 0.0, 10.0, 10.0)],
        confidences=[0.0],
        frame_ids=[0],
    )
    real = _tracklet(
        2,
        boxes=[(1.0, 1.0, 11.0, 11.0)],
        confidences=[0.9],
        frame_ids=[0],
    )

    result = compute_tracklet_occlusion(
        {"cam_a": [interpolated, real]},
        {"occ_box_thresh": 0.6, "occ_frac_thresh": 0.3},
    )

    assert result[("cam_a", 1)] is False
    assert result[("cam_a", 2)] is False


def test_occlusion_penalty_shrinks_similarity_by_inverse_distance_penalty():
    baseline = _combined_score(occluded_flags=None)
    penalized = _combined_score(occluded_flags=[True, False], occ_penalty=1.0 / 1.1)

    assert baseline == pytest.approx(0.8)
    assert penalized == pytest.approx(0.8 / 1.1)


def test_occlusion_flags_all_false_are_noop():
    baseline = _combined_score(occluded_flags=None)
    unflagged = _combined_score(occluded_flags=[False, False], occ_penalty=1.0 / 1.1)

    assert unflagged == pytest.approx(baseline)