"""Tests for Stationary Sensitive Association."""

from __future__ import annotations

from athar.core.data_models import Tracklet, TrackletFrame
from athar.components.tracking.ssa import apply_ssa


def _tracklet(boxes, confidences) -> Tracklet:
    return Tracklet(
        track_id=1,
        camera_id="cam_a",
        class_id=2,
        class_name="car",
        frames=[
            TrackletFrame(
                frame_id=index,
                timestamp=index * 0.1,
                bbox=box,
                confidence=confidence,
            )
            for index, (box, confidence) in enumerate(zip(boxes, confidences))
        ],
    )


def test_stationary_tracklet_freezes_to_highest_confidence_detection():
    high_conf_box = (102.0, 100.0, 122.0, 120.0)
    tracklet = _tracklet(
        boxes=[
            (100.0, 100.0, 120.0, 120.0),
            (101.0, 100.0, 121.0, 120.0),
            high_conf_box,
            (101.5, 100.0, 121.5, 120.0),
            (103.0, 101.0, 123.0, 121.0),
        ],
        confidences=[0.50, 0.60, 0.95, 0.40, 0.30],
    )

    result = apply_ssa([tracklet], {
        "enabled": True,
        "window": 3,
        "disp_thresh": 8.0,
        "freeze_lookback": 4,
        "inherit_conf": False,
    })

    assert result[0].frames[3].bbox == high_conf_box
    assert result[0].frames[4].bbox == high_conf_box
    assert result[0].frames[3].confidence == 0.40


def test_ssa_cold_start_shorter_than_window_is_noop():
    tracklet = _tracklet(
        boxes=[
            (100.0, 100.0, 120.0, 120.0),
            (101.0, 100.0, 121.0, 120.0),
        ],
        confidences=[0.50, 0.95],
    )

    result = apply_ssa([tracklet], {
        "enabled": True,
        "window": 3,
        "disp_thresh": 8.0,
        "freeze_lookback": 3,
    })

    assert result[0] is tracklet
    assert [frame.bbox for frame in result[0].frames] == [
        (100.0, 100.0, 120.0, 120.0),
        (101.0, 100.0, 121.0, 120.0),
    ]


def test_ssa_disabled_flag_returns_input_unchanged():
    tracklet = _tracklet(
        boxes=[
            (100.0, 100.0, 120.0, 120.0),
            (101.0, 100.0, 121.0, 120.0),
            (102.0, 100.0, 122.0, 120.0),
        ],
        confidences=[0.50, 0.60, 0.95],
    )
    tracklets = [tracklet]

    result = apply_ssa(tracklets, {"enabled": False})

    assert result is tracklets