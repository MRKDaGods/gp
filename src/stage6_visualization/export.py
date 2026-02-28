"""Data export: JSON and CSV outputs for global trajectories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

from loguru import logger

from src.core.data_models import GlobalTrajectory


def export_json(trajectories: List[GlobalTrajectory], output_path: str | Path) -> None:
    """Export trajectories as a structured JSON file.

    JSON structure per trajectory:
    {
        "global_id": int,
        "class": str,
        "num_cameras": int,
        "camera_sequence": [str],
        "time_span": [float, float],
        "total_duration": float,
        "tracklets": [
            {
                "camera_id": str,
                "track_id": int,
                "start_time": float,
                "end_time": float,
                "num_frames": int,
                "bbox_samples": [[x1, y1, x2, y2], ...]
            }
        ]
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for traj in trajectories:
        traj_data = {
            "global_id": traj.global_id,
            "class": traj.class_name,
            "num_cameras": traj.num_cameras,
            "camera_sequence": traj.camera_sequence,
            "time_span": list(traj.time_span),
            "total_duration": traj.total_duration,
            "tracklets": [],
        }

        for t in sorted(traj.tracklets, key=lambda x: x.start_time):
            # Sample a few bboxes for visualization
            step = max(1, len(t.frames) // 5)
            bbox_samples = [list(t.frames[i].bbox) for i in range(0, len(t.frames), step)]

            traj_data["tracklets"].append({
                "camera_id": t.camera_id,
                "track_id": t.track_id,
                "class": t.class_name,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "num_frames": t.num_frames,
                "duration": t.duration,
                "bbox_samples": bbox_samples,
            })

        data.append(traj_data)

    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"JSON export: {len(data)} trajectories -> {output_path}")


def export_csv(trajectories: List[GlobalTrajectory], output_path: str | Path) -> None:
    """Export trajectories as a flat CSV file.

    One row per tracklet:
        global_id, class, camera_id, track_id, start_time, end_time, duration, num_frames
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_id", "class", "camera_id", "track_id",
            "start_time", "end_time", "duration", "num_frames",
            "num_cameras_in_trajectory",
        ])

        for traj in trajectories:
            for t in sorted(traj.tracklets, key=lambda x: x.start_time):
                writer.writerow([
                    traj.global_id,
                    t.class_name,
                    t.camera_id,
                    t.track_id,
                    f"{t.start_time:.2f}",
                    f"{t.end_time:.2f}",
                    f"{t.duration:.2f}",
                    t.num_frames,
                    traj.num_cameras,
                ])

    logger.info(f"CSV export: {sum(len(t.tracklets) for t in trajectories)} rows -> {output_path}")
