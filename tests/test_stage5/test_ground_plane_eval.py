import json

import numpy as np

from src.stage5_evaluation.ground_plane_eval import load_gt_ground_positions


def _pos_id(gx: float, gy: float) -> int:
    x_idx = int(round((gx + 300.0) / 2.5))
    y_idx = int(round((gy + 900.0) / 2.5))
    return y_idx * 480 + x_idx


def test_load_gt_ground_positions_dedupes_person_ids(tmp_path) -> None:
    annotations_dir = tmp_path / "annotations_positions"
    annotations_dir.mkdir()
    (annotations_dir / "00000000.json").write_text(
        json.dumps([
            {"personID": 7, "positionID": _pos_id(0.0, 0.0)},
            {"personID": 7, "positionID": _pos_id(10.0, 0.0)},
            {"personID": 8, "positionID": _pos_id(20.0, 0.0)},
        ]),
        encoding="utf-8",
    )

    gt = load_gt_ground_positions(annotations_dir)

    assert list(gt) == [0]
    assert [pid for pid, _, _ in gt[0]] == [7, 8]


def test_load_gt_ground_positions_filters_to_camera_visibility(tmp_path) -> None:
    annotations_dir = tmp_path / "annotations_positions"
    annotations_dir.mkdir()
    (annotations_dir / "00000000.json").write_text(
        json.dumps([
            {"personID": 1, "positionID": _pos_id(0.0, 0.0)},
            {"personID": 2, "positionID": _pos_id(0.0, 10000.0)},
        ]),
        encoding="utf-8",
    )
    calibrations = {
        "C1": {
            "K": np.array([[100.0, 0.0, 960.0], [0.0, 100.0, 540.0], [0.0, 0.0, 1.0]]),
            "R": np.eye(3),
            "tvec": np.array([0.0, 0.0, 1000.0]),
        }
    }

    gt = load_gt_ground_positions(annotations_dir, calibrations=calibrations)

    assert gt[0] == [(1, 0.0, 0.0)]