"""Tests for track smoothing in stage 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.stage5_evaluation.pipeline import _smooth_prediction_tracks


@pytest.fixture
def pred_dir(tmp_path: Path) -> Path:
    """Write a sample prediction file and return its directory."""
    # Create a track with jittery bbox center
    rows = []
    np.random.seed(42)
    for i in range(20):
        frame = i + 1
        tid = 1
        # Linear motion + noise
        x = 100.0 + i * 5 + np.random.randn() * 3
        y = 200.0 + np.random.randn() * 3
        w, h = 50.0, 40.0
        conf = 0.9
        cls_id = 2
        vis = 1.0
        rows.append(f"{frame},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{conf},{cls_id},{vis}")
    # Add a short track (3 frames) that shouldn't be smoothed
    for i in range(3):
        rows.append(f"{i + 1},2,300.0,300.0,30.0,30.0,0.8,2,1.0")

    (tmp_path / "S01_c001.txt").write_text("\n".join(rows) + "\n")
    return tmp_path


class TestTrackSmoothing:
    def test_smoothing_reduces_jitter(self, pred_dir: Path):
        """After smoothing, bbox center should be less jittery."""
        # Read before smoothing
        lines_before = (pred_dir / "S01_c001.txt").read_text().strip().split("\n")
        cx_before = []
        for line in lines_before:
            parts = line.split(",")
            if int(parts[1]) == 1:
                cx_before.append(float(parts[2]) + float(parts[4]) / 2)

        _smooth_prediction_tracks(pred_dir, window=7, polyorder=2)

        # Read after smoothing
        lines_after = (pred_dir / "S01_c001.txt").read_text().strip().split("\n")
        cx_after = []
        for line in lines_after:
            parts = line.split(",")
            if int(parts[1]) == 1:
                cx_after.append(float(parts[2]) + float(parts[4]) / 2)

        # Compute jitter as std of second differences
        diff2_before = np.diff(cx_before, n=2)
        diff2_after = np.diff(cx_after, n=2)
        assert np.std(diff2_after) < np.std(diff2_before)

    def test_short_tracks_unchanged(self, pred_dir: Path):
        """Tracks shorter than window should not be modified."""
        lines_before = (pred_dir / "S01_c001.txt").read_text().strip().split("\n")
        short_before = [l for l in lines_before if l.split(",")[1] == "2"]

        _smooth_prediction_tracks(pred_dir, window=7, polyorder=2)

        lines_after = (pred_dir / "S01_c001.txt").read_text().strip().split("\n")
        short_after = [l for l in lines_after if l.split(",")[1] == "2"]

        assert short_before == short_after

    def test_row_count_preserved(self, pred_dir: Path):
        """Smoothing shouldn't add or remove rows."""
        count_before = len(
            (pred_dir / "S01_c001.txt").read_text().strip().split("\n")
        )
        _smooth_prediction_tracks(pred_dir, window=7, polyorder=2)
        count_after = len(
            (pred_dir / "S01_c001.txt").read_text().strip().split("\n")
        )
        assert count_before == count_after
