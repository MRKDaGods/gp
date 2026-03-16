"""Tests for stage 5 format converter — frame numbering and format correctness."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.data_models import GlobalTrajectory, Tracklet, TrackletFrame


def _make_trajectory(
    global_id: int = 1,
    camera_id: str = "S01_c001",
    track_id: int = 10,
    frames: list | None = None,
) -> GlobalTrajectory:
    """Create a simple trajectory for testing."""
    if frames is None:
        frames = [
            TrackletFrame(frame_id=0, timestamp=0.0, bbox=(100, 200, 150, 250), confidence=0.9),
            TrackletFrame(frame_id=10, timestamp=1.0, bbox=(110, 210, 160, 260), confidence=0.85),
        ]
    tracklet = Tracklet(
        track_id=track_id,
        camera_id=camera_id,
        class_id=2,
        class_name="car",
        frames=frames,
    )
    return GlobalTrajectory(global_id=global_id, tracklets=[tracklet])


class TestMOTFrameNumbering:
    """Verify MOT submission uses 1-based frame numbering."""

    def test_mot_submission_1based_frames(self, tmp_path: Path):
        from src.stage5_evaluation.format_converter import trajectories_to_mot_submission

        traj = _make_trajectory(
            global_id=5,
            frames=[
                TrackletFrame(frame_id=0, timestamp=0.0, bbox=(10, 20, 50, 60), confidence=0.9),
                TrackletFrame(frame_id=99, timestamp=9.9, bbox=(15, 25, 55, 65), confidence=0.8),
            ],
        )
        trajectories_to_mot_submission([traj], tmp_path)

        output_file = tmp_path / "S01_c001.txt"
        assert output_file.exists()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2

        # First line: frame_id=0 in internal repr → 1 in MOT output
        parts0 = lines[0].split(",")
        assert int(parts0[0]) == 1, f"Expected 1-based frame 1, got {parts0[0]}"

        # Second line: frame_id=99 → 100 in MOT output
        parts1 = lines[1].split(",")
        assert int(parts1[0]) == 100, f"Expected 1-based frame 100, got {parts1[0]}"


class TestAICSubmissionFormat:
    """Verify AIC submission uses correct column order and 1-based frames."""

    def test_aic_column_order(self, tmp_path: Path):
        from src.stage5_evaluation.format_converter import trajectories_to_aic_submission

        traj = _make_trajectory(
            global_id=42,
            camera_id="S01_c001",
            frames=[
                TrackletFrame(frame_id=54, timestamp=5.4, bbox=(100, 200, 150, 250), confidence=0.9),
            ],
        )
        output_file = tmp_path / "submission.txt"
        trajectories_to_aic_submission([traj], output_file)

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1

        parts = lines[0].split()
        # AIC format: camera_id obj_id frame_id x y w h -1 -1
        assert parts[0] == "S01_c001", f"Col 0 (camera_id): {parts[0]}"
        assert int(parts[1]) == 42, f"Col 1 (obj_id): {parts[1]}"
        assert int(parts[2]) == 55, f"Col 2 (frame_id, 1-based): {parts[2]}"
        assert len(parts) == 9, f"Expected 9 columns, got {len(parts)}"
        assert parts[7] == "-1", f"Col 7 (sentinel): {parts[7]}"
        assert parts[8] == "-1", f"Col 8 (sentinel): {parts[8]}"
