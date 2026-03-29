from pathlib import Path

from src.stage_wildtrack_mvdetr.pipeline import (
    load_mvdetr_ground_plane_detections,
    track_ground_plane_detections,
)


def test_load_mvdetr_txt_normalizes_frames_and_world_coords(tmp_path: Path) -> None:
    detections_file = tmp_path / "test.txt"
    detections_file.write_text("0 10 20\n5 11 21\n10 12 22\n", encoding="utf-8")

    detections = load_mvdetr_ground_plane_detections(detections_file)

    assert [det.frame_id for det in detections] == [0, 1, 2]
    assert detections[0].x_cm == -275.0
    assert detections[0].y_cm == -850.0
    assert detections[2].raw_frame_id == 10


def test_track_ground_plane_detections_merges_nearby_observations() -> None:
    detections = [
        load_mvdetr_ground_plane_detections_from_values(0, 0.0, 0.0),
        load_mvdetr_ground_plane_detections_from_values(1, 10.0, 5.0),
        load_mvdetr_ground_plane_detections_from_values(2, 18.0, 8.0),
        load_mvdetr_ground_plane_detections_from_values(2, 400.0, 400.0),
    ]

    tracks = track_ground_plane_detections(
        detections,
        max_match_distance_cm=40.0,
        max_missed_frames=2,
        min_track_length=1,
    )

    assert len(tracks) == 2
    assert [len(track.detections) for track in tracks] == [3, 1]


def load_mvdetr_ground_plane_detections_from_values(frame_id: int, x_cm: float, y_cm: float):
    from src.stage_wildtrack_mvdetr.pipeline import GroundPlaneDetection

    return GroundPlaneDetection(frame_id=frame_id, x_cm=x_cm, y_cm=y_cm, score=1.0)