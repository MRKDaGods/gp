"""Tests for metadata store."""

import numpy as np

from src.stage3_indexing.metadata_store import MetadataStore


def test_metadata_insert_and_get(tmp_path):
    db_path = tmp_path / "test.db"
    store = MetadataStore(db_path)

    hsv = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    store.insert_tracklet(
        index_id=0, track_id=1, camera_id="cam01",
        class_id=0, start_time=0.0, end_time=3.0,
        num_frames=30, hsv_histogram=hsv,
    )

    meta = store.get_tracklet(0)
    assert meta is not None
    assert meta["track_id"] == 1
    assert meta["camera_id"] == "cam01"
    assert meta["start_time"] == 0.0

    loaded_hsv = store.get_hsv_histogram(0)
    np.testing.assert_array_almost_equal(loaded_hsv, hsv)

    store.close()


def test_metadata_query_by_camera(tmp_path):
    store = MetadataStore(tmp_path / "test.db")

    for i in range(10):
        cam = "cam01" if i < 6 else "cam02"
        store.insert_tracklet(
            index_id=i, track_id=i, camera_id=cam,
            class_id=0, start_time=i * 1.0, end_time=i * 1.0 + 2.0,
            num_frames=20,
        )

    cam01 = store.get_by_camera("cam01")
    assert len(cam01) == 6
    cam02 = store.get_by_camera("cam02")
    assert len(cam02) == 4

    store.close()


def test_metadata_query_by_time(tmp_path):
    store = MetadataStore(tmp_path / "test.db")

    for i in range(5):
        store.insert_tracklet(
            index_id=i, track_id=i, camera_id="cam01",
            class_id=0, start_time=i * 10.0, end_time=i * 10.0 + 5.0,
            num_frames=50,
        )

    # Query for time range 15-25 should match tracks starting at 10 and 20
    results = store.get_by_time_range(15.0, 25.0)
    assert len(results) == 2  # tracks at t=10-15 and t=20-25

    store.close()


def test_metadata_count(tmp_path):
    store = MetadataStore(tmp_path / "test.db")
    assert store.count() == 0

    store.insert_tracklet(0, 0, "cam01", 0, 0.0, 1.0, 10)
    assert store.count() == 1

    store.close()
