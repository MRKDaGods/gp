"""Convert between tracking output formats."""

from __future__ import annotations

from pathlib import Path
from typing import List

from loguru import logger

from src.core.data_models import GlobalTrajectory


def trajectories_to_mot_submission(
    trajectories: List[GlobalTrajectory],
    output_dir: str | Path,
) -> None:
    """Convert global trajectories to MOTChallenge submission format.

    Creates one text file per camera with lines:
        frame_id, global_id, x, y, w, h, confidence, class, visibility

    Args:
        trajectories: Global trajectories from Stage 4.
        output_dir: Output directory for MOT-format files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group all detections by camera
    camera_rows = {}

    for traj in trajectories:
        for tracklet in traj.tracklets:
            cam_id = tracklet.camera_id
            if cam_id not in camera_rows:
                camera_rows[cam_id] = []

            for frame in tracklet.frames:
                x1, y1, x2, y2 = frame.bbox
                w = x2 - x1
                h = y2 - y1
                row = (
                    frame.frame_id,
                    traj.global_id,
                    x1, y1, w, h,
                    frame.confidence,
                    tracklet.class_id,
                    1.0,  # visibility
                )
                camera_rows[cam_id].append(row)

    # Write per-camera files
    for cam_id, rows in camera_rows.items():
        rows.sort(key=lambda r: (r[0], r[1]))
        file_path = output_dir / f"{cam_id}.txt"
        with open(file_path, "w") as f:
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")

    logger.info(
        f"MOT submission written: {len(camera_rows)} cameras, "
        f"{sum(len(r) for r in camera_rows.values())} detection rows"
    )


def trajectories_to_aic_submission(
    trajectories: List[GlobalTrajectory],
    output_path: str | Path,
) -> None:
    """Convert global trajectories to AI City Challenge submission format.

    AIC format: camera_id frame_id global_id x y w h

    Args:
        trajectories: Global trajectories.
        output_path: Output text file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for traj in trajectories:
        for tracklet in traj.tracklets:
            for frame in tracklet.frames:
                x1, y1, x2, y2 = frame.bbox
                w = x2 - x1
                h = y2 - y1
                rows.append(
                    f"{tracklet.camera_id} {frame.frame_id} {traj.global_id} "
                    f"{x1:.1f} {y1:.1f} {w:.1f} {h:.1f}"
                )

    rows.sort()
    output_path.write_text("\n".join(rows) + "\n")
    logger.info(f"AIC submission written: {len(rows)} rows to {output_path}")
