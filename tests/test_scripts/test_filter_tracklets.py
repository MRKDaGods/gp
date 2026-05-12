import json

import numpy as np

from scripts.filter_tracklets import filter_stage_outputs
from src.core.data_models import Tracklet, TrackletFrame
from src.core.io_utils import save_tracklets_by_camera


def _tracklet(track_id: int, camera_id: str, confidences: list[float]) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        camera_id=camera_id,
        class_id=2,
        class_name="car",
        frames=[
            TrackletFrame(
                frame_id=index,
                timestamp=float(index) / 10.0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                confidence=confidence,
            )
            for index, confidence in enumerate(confidences)
        ],
    )


def test_noop_filter_preserves_rows_and_metadata(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    out_stage1_dir = tmp_path / "filtered" / "stage1"
    out_stage2_dir = tmp_path / "filtered" / "stage2"
    stage2_dir.mkdir(parents=True)

    tracklets = {
        "S01_c001": [
            _tracklet(1, "S01_c001", [0.8, 0.7, 0.9]),
            _tracklet(2, "S01_c001", [0.4, 0.5]),
        ],
        "S01_c002": [_tracklet(3, "S01_c002", [0.6])],
    }
    save_tracklets_by_camera(tracklets, stage1_dir)

    index_map = [
        {"track_id": 1, "camera_id": "S01_c001", "class_id": 2},
        {"track_id": 2, "camera_id": "S01_c001", "class_id": 2},
        {"track_id": 3, "camera_id": "S01_c002", "class_id": 2},
    ]
    (stage2_dir / "embedding_index.json").write_text(json.dumps(index_map), encoding="utf-8")
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    tertiary = np.arange(9, dtype=np.float32).reshape(3, 3)
    hsv = np.arange(15, dtype=np.float32).reshape(3, 5)
    multi_query = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    np.save(stage2_dir / "embeddings.npy", embeddings)
    np.save(stage2_dir / "embeddings_tertiary.npy", tertiary)
    np.save(stage2_dir / "hsv_features.npy", hsv)
    np.savez_compressed(stage2_dir / "multi_query_embeddings.npz", embeddings=multi_query)

    summary = filter_stage_outputs(
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        output_stage1_dir=out_stage1_dir,
        output_stage2_dir=out_stage2_dir,
        min_avg_confidence=0.0,
        min_length=0,
    )

    assert summary["total_in"] == 3
    assert summary["total_out"] == 3
    assert summary["total_dropped"] == 0
    assert summary["keep_indices"] == [0, 1, 2]
    assert json.loads((out_stage2_dir / "embedding_index.json").read_text(encoding="utf-8")) == index_map
    np.testing.assert_array_equal(np.load(out_stage2_dir / "embeddings.npy"), embeddings)
    np.testing.assert_array_equal(np.load(out_stage2_dir / "embeddings_tertiary.npy"), tertiary)
    np.testing.assert_array_equal(np.load(out_stage2_dir / "hsv_features.npy"), hsv)
    with np.load(out_stage2_dir / "multi_query_embeddings.npz") as data:
        np.testing.assert_array_equal(data["embeddings"], multi_query)


def test_filter_drops_short_or_low_confidence_tracklets(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    out_stage1_dir = tmp_path / "filtered" / "stage1"
    out_stage2_dir = tmp_path / "filtered" / "stage2"
    stage2_dir.mkdir(parents=True)

    tracklets = {
        "S01_c001": [
            _tracklet(1, "S01_c001", [0.8, 0.7, 0.9]),
            _tracklet(2, "S01_c001", [0.4, 0.5, 0.4]),
            _tracklet(3, "S01_c001", [0.9]),
        ]
    }
    save_tracklets_by_camera(tracklets, stage1_dir)
    index_map = [
        {"track_id": 1, "camera_id": "S01_c001", "class_id": 2},
        {"track_id": 2, "camera_id": "S01_c001", "class_id": 2},
        {"track_id": 3, "camera_id": "S01_c001", "class_id": 2},
    ]
    (stage2_dir / "embedding_index.json").write_text(json.dumps(index_map), encoding="utf-8")
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(stage2_dir / "embeddings.npy", embeddings)

    summary = filter_stage_outputs(
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        output_stage1_dir=out_stage1_dir,
        output_stage2_dir=out_stage2_dir,
        min_avg_confidence=0.5,
        min_length=2,
    )

    assert summary["keep_indices"] == [0]
    assert summary["drop_counts"] == {
        "length_only": 1,
        "confidence_only": 1,
        "both": 0,
        "missing_tracklet": 0,
    }
    np.testing.assert_array_equal(np.load(out_stage2_dir / "embeddings.npy"), embeddings[[0]])