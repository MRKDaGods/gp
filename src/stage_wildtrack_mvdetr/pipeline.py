"""MVDeTr integration for WILDTRACK.

This stage consumes MVDeTr ground-plane detections, runs simple frame-to-frame
Hungarian tracking in world coordinates, projects the tracked positions back to
camera views, and saves both projected per-camera tracklets and one-track-per-
identity global trajectories for Stage 5 evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from src.core.data_models import GlobalTrajectory, Tracklet, TrackletFrame
from src.core.io_utils import save_global_trajectories, save_tracklets_by_camera
from src.core.wildtrack_calibration import load_wildtrack_calibration

WILDTRACK_GRID_WIDTH = 480
WILDTRACK_CELL_SIZE_CM = 2.5
WILDTRACK_X_MIN_CM = -300.0
WILDTRACK_Y_MIN_CM = -900.0


@dataclass(frozen=True)
class GroundPlaneDetection:
    """One MVDeTr prediction on the WILDTRACK ground plane."""

    frame_id: int
    x_cm: float
    y_cm: float
    score: float = 1.0
    raw_frame_id: int | None = None


@dataclass
class GroundPlaneTrack:
    """One tracked identity on the WILDTRACK ground plane."""

    track_id: int
    detections: List[GroundPlaneDetection] = field(default_factory=list)

    @property
    def last_detection(self) -> GroundPlaneDetection:
        return self.detections[-1]

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([d.score for d in self.detections]))


def _load_txt_rows(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append([float(value) for value in stripped.replace(",", " ").split()])
    return rows


def _grid_to_world_cm(grid_x: float, grid_y: float) -> Tuple[float, float]:
    x_cm = WILDTRACK_X_MIN_CM + grid_x * WILDTRACK_CELL_SIZE_CM
    y_cm = WILDTRACK_Y_MIN_CM + grid_y * WILDTRACK_CELL_SIZE_CM
    return x_cm, y_cm


def _normalize_wildtrack_frame(frame_id: int) -> int:
    return frame_id // 5 if frame_id >= 5 else frame_id


def load_mvdetr_ground_plane_detections(
    detections_path: str | Path,
    normalize_wildtrack_frames: bool = True,
) -> List[GroundPlaneDetection]:
    """Load MVDeTr detections from txt/csv/json exports.

    Supported formats:
    - MVDeTr `test.txt`: `frame grid_x grid_y`
    - CSV with `frame_id`, `x_world_cm`, `y_world_cm` or `x_grid`, `y_grid`
    - JSON list of dicts with the same fields
    """
    detections_path = Path(detections_path)
    suffix = detections_path.suffix.lower()
    detections: List[GroundPlaneDetection] = []

    if suffix == ".txt":
        for row in _load_txt_rows(detections_path):
            if len(row) < 3:
                continue
            raw_frame = int(row[0])
            frame_id = _normalize_wildtrack_frame(raw_frame) if normalize_wildtrack_frames else raw_frame
            x_cm, y_cm = _grid_to_world_cm(row[1], row[2])
            detections.append(
                GroundPlaneDetection(
                    frame_id=frame_id,
                    x_cm=x_cm,
                    y_cm=y_cm,
                    score=1.0,
                    raw_frame_id=raw_frame,
                )
            )
    elif suffix == ".csv":
        with detections_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_frame = int(row.get("raw_frame_id") or row["frame_id"])
                frame_id = int(row["frame_id"])
                if normalize_wildtrack_frames and "raw_frame_id" not in row and raw_frame >= 5:
                    frame_id = _normalize_wildtrack_frame(raw_frame)
                if "x_world_cm" in row and "y_world_cm" in row:
                    x_cm = float(row["x_world_cm"])
                    y_cm = float(row["y_world_cm"])
                else:
                    x_cm, y_cm = _grid_to_world_cm(float(row["x_grid"]), float(row["y_grid"]))
                detections.append(
                    GroundPlaneDetection(
                        frame_id=frame_id,
                        x_cm=x_cm,
                        y_cm=y_cm,
                        score=float(row.get("score", 1.0)),
                        raw_frame_id=raw_frame,
                    )
                )
    elif suffix == ".json":
        payload = json.loads(detections_path.read_text(encoding="utf-8"))
        for row in payload:
            raw_frame = int(row.get("raw_frame_id", row["frame_id"]))
            frame_id = int(row["frame_id"])
            if normalize_wildtrack_frames and "raw_frame_id" not in row and raw_frame >= 5:
                frame_id = _normalize_wildtrack_frame(raw_frame)
            if "x_world_cm" in row and "y_world_cm" in row:
                x_cm = float(row["x_world_cm"])
                y_cm = float(row["y_world_cm"])
            else:
                x_cm, y_cm = _grid_to_world_cm(float(row["x_grid"]), float(row["y_grid"]))
            detections.append(
                GroundPlaneDetection(
                    frame_id=frame_id,
                    x_cm=x_cm,
                    y_cm=y_cm,
                    score=float(row.get("score", 1.0)),
                    raw_frame_id=raw_frame,
                )
            )
    else:
        raise ValueError(f"Unsupported detection format: {detections_path}")

    detections.sort(key=lambda det: (det.frame_id, det.x_cm, det.y_cm))
    logger.info(f"Loaded {len(detections)} MVDeTr detections from {detections_path}")
    return detections


def _group_by_frame(
    detections: Sequence[GroundPlaneDetection],
) -> Dict[int, List[GroundPlaneDetection]]:
    grouped: Dict[int, List[GroundPlaneDetection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.frame_id].append(detection)
    return dict(grouped)


def _distance_cm(a: GroundPlaneDetection, b: GroundPlaneDetection) -> float:
    return float(np.hypot(a.x_cm - b.x_cm, a.y_cm - b.y_cm))


def track_ground_plane_detections(
    detections: Sequence[GroundPlaneDetection],
    max_match_distance_cm: float = 75.0,
    max_missed_frames: int = 5,
    min_track_length: int = 2,
) -> List[GroundPlaneTrack]:
    """Track MVDeTr detections with Hungarian matching on world coordinates."""
    detections_by_frame = _group_by_frame(detections)
    active_tracks: Dict[int, GroundPlaneTrack] = {}
    finished_tracks: List[GroundPlaneTrack] = []
    next_track_id = 1

    for frame_id in sorted(detections_by_frame):
        frame_detections = detections_by_frame[frame_id]

        stale_track_ids = [
            track_id
            for track_id, track in active_tracks.items()
            if frame_id - track.last_detection.frame_id > max_missed_frames
        ]
        for track_id in stale_track_ids:
            finished_tracks.append(active_tracks.pop(track_id))

        track_ids = list(active_tracks.keys())
        if track_ids and frame_detections:
            cost = np.full((len(track_ids), len(frame_detections)), max_match_distance_cm + 1e6)
            for row_idx, track_id in enumerate(track_ids):
                last_det = active_tracks[track_id].last_detection
                for col_idx, detection in enumerate(frame_detections):
                    if frame_id - last_det.frame_id > max_missed_frames:
                        continue
                    dist = _distance_cm(last_det, detection)
                    if dist <= max_match_distance_cm:
                        cost[row_idx, col_idx] = dist

            matches: List[Tuple[int, int]] = []
            unmatched_tracks = set(track_ids)
            unmatched_dets = set(range(len(frame_detections)))

            if np.isfinite(cost).any():
                row_ind, col_ind = linear_sum_assignment(cost)
                for row_idx, col_idx in zip(row_ind, col_ind):
                    if cost[row_idx, col_idx] > max_match_distance_cm:
                        continue
                    track_id = track_ids[row_idx]
                    matches.append((track_id, col_idx))
                    unmatched_tracks.discard(track_id)
                    unmatched_dets.discard(col_idx)
            for track_id, det_idx in matches:
                active_tracks[track_id].detections.append(frame_detections[det_idx])
            for det_idx in sorted(unmatched_dets):
                active_tracks[next_track_id] = GroundPlaneTrack(
                    track_id=next_track_id,
                    detections=[frame_detections[det_idx]],
                )
                next_track_id += 1
        else:
            for detection in frame_detections:
                active_tracks[next_track_id] = GroundPlaneTrack(
                    track_id=next_track_id,
                    detections=[detection],
                )
                next_track_id += 1

    finished_tracks.extend(active_tracks.values())
    finished_tracks = [track for track in finished_tracks if len(track.detections) >= min_track_length]
    finished_tracks.sort(key=lambda track: track.track_id)
    logger.info(f"Tracked {len(finished_tracks)} ground-plane identities")
    return finished_tracks


def _project_world_to_image(
    calibration: dict,
    x_cm: float,
    y_cm: float,
    z_cm: float,
) -> np.ndarray:
    points = np.array([[x_cm, y_cm, z_cm]], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        points,
        calibration["rvec"].astype(np.float64),
        calibration["tvec"].astype(np.float64),
        calibration["K"].astype(np.float64),
        None,
    )
    return projected.reshape(-1, 2)[0]


def _make_bbox_from_ground_point(
    calibration: dict,
    x_cm: float,
    y_cm: float,
    image_size: Tuple[int, int],
    person_height_cm: float = 175.0,
    width_ratio: float = 0.4,
) -> Tuple[float, float, float, float] | None:
    width_px, height_px = image_size
    foot = _project_world_to_image(calibration, x_cm, y_cm, 0.0)
    head = _project_world_to_image(calibration, x_cm, y_cm, person_height_cm)
    if not np.isfinite(foot).all() or not np.isfinite(head).all():
        return None

    bbox_height = max(20.0, abs(float(foot[1] - head[1])))
    bbox_width = max(10.0, bbox_height * width_ratio)
    x1 = float(foot[0] - bbox_width / 2)
    x2 = float(foot[0] + bbox_width / 2)
    y2 = float(foot[1])
    y1 = float(y2 - bbox_height)

    if x2 < 0 or y2 < 0 or x1 > width_px or y1 > height_px:
        return None

    x1 = float(np.clip(x1, 0, width_px - 1))
    x2 = float(np.clip(x2, 0, width_px - 1))
    y1 = float(np.clip(y1, 0, height_px - 1))
    y2 = float(np.clip(y2, 0, height_px - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _tracks_to_projected_tracklets(
    tracks: Sequence[GroundPlaneTrack],
    calibrations: Dict[str, dict],
    fps: float,
    image_size: Tuple[int, int],
) -> Tuple[Dict[str, List[Tracklet]], List[GlobalTrajectory]]:
    by_camera: Dict[str, List[Tracklet]] = {camera_id: [] for camera_id in calibrations}
    trajectories: List[GlobalTrajectory] = []

    for track in tracks:
        trajectory_tracklets: List[Tracklet] = []
        for camera_id, calibration in sorted(calibrations.items()):
            frames: List[TrackletFrame] = []
            for detection in track.detections:
                bbox = _make_bbox_from_ground_point(
                    calibration=calibration,
                    x_cm=detection.x_cm,
                    y_cm=detection.y_cm,
                    image_size=image_size,
                )
                if bbox is None:
                    continue
                frames.append(
                    TrackletFrame(
                        frame_id=detection.frame_id,
                        timestamp=detection.frame_id / fps,
                        bbox=bbox,
                        confidence=detection.score,
                    )
                )
            if not frames:
                continue
            tracklet = Tracklet(
                track_id=track.track_id,
                camera_id=camera_id,
                class_id=0,
                class_name="person",
                frames=frames,
            )
            by_camera[camera_id].append(tracklet)
            trajectory_tracklets.append(tracklet)

        trajectories.append(
            GlobalTrajectory(
                global_id=track.track_id,
                tracklets=trajectory_tracklets,
                confidence=track.mean_confidence,
                evidence=[],
                timeline=[
                    {
                        "camera_id": tracklet.camera_id,
                        "start": tracklet.start_time,
                        "end": tracklet.end_time,
                    }
                    for tracklet in trajectory_tracklets
                ],
            )
        )

    by_camera = {camera_id: tracklets for camera_id, tracklets in by_camera.items() if tracklets}
    return by_camera, trajectories


def _save_ground_plane_tracks(tracks: Sequence[GroundPlaneTrack], output_path: Path) -> None:
    payload = []
    for track in tracks:
        payload.append(
            {
                "track_id": track.track_id,
                "mean_confidence": track.mean_confidence,
                "detections": [
                    {
                        "frame_id": det.frame_id,
                        "raw_frame_id": det.raw_frame_id,
                        "x_world_cm": det.x_cm,
                        "y_world_cm": det.y_cm,
                        "score": det.score,
                    }
                    for det in track.detections
                ],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_ground_plane_csv(tracks: Sequence[GroundPlaneTrack], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "frame_id", "raw_frame_id", "x_world_cm", "y_world_cm", "score"])
        for track in tracks:
            for det in track.detections:
                writer.writerow([
                    track.track_id,
                    det.frame_id,
                    det.raw_frame_id if det.raw_frame_id is not None else det.frame_id,
                    f"{det.x_cm:.3f}",
                    f"{det.y_cm:.3f}",
                    f"{det.score:.4f}",
                ])


def run_stage_wildtrack_mvdetr(
    detections_path: str | Path,
    calibration_dir: str | Path,
    output_dir: str | Path,
    fps: float = 2.0,
    image_size: Tuple[int, int] = (1920, 1080),
    max_match_distance_cm: float = 75.0,
    max_missed_frames: int = 5,
    min_track_length: int = 2,
) -> Tuple[Dict[str, List[Tracklet]], List[GlobalTrajectory]]:
    """Convert MVDeTr detections into projected tracklets and trajectories."""
    detections = load_mvdetr_ground_plane_detections(detections_path)
    calibrations = load_wildtrack_calibration(calibration_dir)
    if not calibrations:
        raise FileNotFoundError(f"No WILDTRACK calibrations loaded from {calibration_dir}")

    tracks = track_ground_plane_detections(
        detections=detections,
        max_match_distance_cm=max_match_distance_cm,
        max_missed_frames=max_missed_frames,
        min_track_length=min_track_length,
    )
    tracklets_by_camera, trajectories = _tracks_to_projected_tracklets(
        tracks=tracks,
        calibrations=calibrations,
        fps=fps,
        image_size=image_size,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tracklets_by_camera(tracklets_by_camera, output_dir / "tracklets")
    save_global_trajectories(trajectories, output_dir / "global_trajectories.json")
    _save_ground_plane_tracks(tracks, output_dir / "ground_plane_tracks.json")
    _save_ground_plane_csv(tracks, output_dir / "ground_plane_tracks.csv")

    logger.info(
        f"Saved MVDeTr integration outputs to {output_dir}: "
        f"{sum(len(v) for v in tracklets_by_camera.values())} projected tracklets, "
        f"{len(trajectories)} global trajectories"
    )
    return tracklets_by_camera, trajectories


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MVDeTr Wildtrack detections to repo artifacts")
    parser.add_argument("detections_path", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="data/raw/wildtrack/calibrations",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-match-distance-cm", type=float, default=75.0)
    parser.add_argument("--max-missed-frames", type=int, default=5)
    parser.add_argument("--min-track-length", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_stage_wildtrack_mvdetr(
        detections_path=args.detections_path,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        max_match_distance_cm=args.max_match_distance_cm,
        max_missed_frames=args.max_missed_frames,
        min_track_length=args.min_track_length,
    )


if __name__ == "__main__":
    main()