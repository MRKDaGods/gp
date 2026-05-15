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


@dataclass
class KalmanTrack:
    """Constant-velocity Kalman state for one ground-plane identity."""

    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    detections: List[GroundPlaneDetection] = field(default_factory=list)
    hits: int = 1
    consecutive_misses: int = 0
    last_frame_id: int = -1

    def is_confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits

    def predict(self, frame_id: int, transition_matrix: np.ndarray, process_noise: np.ndarray) -> None:
        self.state = transition_matrix @ self.state
        self.covariance = transition_matrix @ self.covariance @ transition_matrix.T + process_noise
        self.last_frame_id = frame_id

    def update(
        self,
        detection: GroundPlaneDetection,
        measurement_matrix: np.ndarray,
        measurement_noise: np.ndarray,
    ) -> None:
        measurement = np.array([detection.x_cm, detection.y_cm], dtype=np.float64)
        innovation = measurement - (measurement_matrix @ self.state)
        innovation_covariance = measurement_matrix @ self.covariance @ measurement_matrix.T + measurement_noise
        kalman_gain = self.covariance @ measurement_matrix.T @ np.linalg.pinv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(self.covariance.shape[0], dtype=np.float64)
        self.covariance = (identity - kalman_gain @ measurement_matrix) @ self.covariance
        self.detections.append(detection)
        self.hits += 1
        self.consecutive_misses = 0


class KalmanGroundPlaneTracker:
    """12b WILDTRACK tuned constant-velocity Kalman tracker."""

    def __init__(
        self,
        max_age: int = 2,
        min_hits: int = 2,
        distance_gate: float = 25.0,
        max_euclidean_cm: float = 200.0,
        q_std: float = 5.0,
        r_std: float = 10.0,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_gate = distance_gate
        self.max_euclidean_cm = max_euclidean_cm
        self.q_std = q_std
        self.r_std = r_std
        self.measurement_matrix = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.measurement_noise = np.diag([r_std**2, r_std**2]).astype(np.float64)
        self.active_tracks: Dict[int, KalmanTrack] = {}
        self.finished_tracks: List[KalmanTrack] = []
        self.next_track_id = 1

    def _transition_matrix(self, dt: float) -> np.ndarray:
        return np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        q_var = float(self.q_std) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q_var * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

    def _initial_covariance(self) -> np.ndarray:
        position_var = float(self.r_std) ** 2
        velocity_std = max(75.0, float(self.q_std) * 6.0)
        velocity_var = velocity_std**2
        return np.diag([position_var, position_var, velocity_var, velocity_var]).astype(np.float64)

    def _mahalanobis_distance(self, track: KalmanTrack, detection: GroundPlaneDetection) -> Tuple[float, float]:
        measurement = np.array([detection.x_cm, detection.y_cm], dtype=np.float64)
        innovation = measurement - (self.measurement_matrix @ track.state)
        innovation_covariance = (
            self.measurement_matrix @ track.covariance @ self.measurement_matrix.T + self.measurement_noise
        )
        distance_squared = float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)
        euclidean_cm = float(np.linalg.norm(innovation))
        return distance_squared, euclidean_cm

    def _spawn_track(self, detection: GroundPlaneDetection) -> None:
        state = np.array([detection.x_cm, detection.y_cm, 0.0, 0.0], dtype=np.float64)
        self.active_tracks[self.next_track_id] = KalmanTrack(
            track_id=self.next_track_id,
            state=state,
            covariance=self._initial_covariance(),
            detections=[detection],
            hits=1,
            consecutive_misses=0,
            last_frame_id=int(detection.frame_id),
        )
        self.next_track_id += 1

    def track(self, detections: Sequence[GroundPlaneDetection]) -> List[GroundPlaneTrack]:
        detections_by_frame = _group_by_frame(detections)
        if not detections_by_frame:
            return []

        first_frame_id = min(detections_by_frame)
        last_frame_id = max(detections_by_frame)
        for frame_id in range(first_frame_id, last_frame_id + 1):
            frame_detections = detections_by_frame.get(frame_id, [])
            active_items = list(self.active_tracks.items())

            for _, track in active_items:
                dt = max(1.0, float(frame_id - track.last_frame_id))
                track.predict(frame_id, self._transition_matrix(dt), self._process_noise(dt))

            matched_track_ids: set[int] = set()
            matched_detection_indices: set[int] = set()
            if active_items and frame_detections:
                cost_matrix = np.full((len(active_items), len(frame_detections)), np.inf, dtype=np.float64)
                for row, (_, track) in enumerate(active_items):
                    for col, detection in enumerate(frame_detections):
                        distance_squared, euclidean_cm = self._mahalanobis_distance(track, detection)
                        if distance_squared <= self.distance_gate and euclidean_cm <= self.max_euclidean_cm:
                            cost_matrix[row, col] = distance_squared

                if np.isfinite(cost_matrix).any():
                    assignment_costs = np.where(np.isfinite(cost_matrix), cost_matrix, 1.0e9)
                    row_ind, col_ind = linear_sum_assignment(assignment_costs)
                    for row, col in zip(row_ind, col_ind):
                        if not np.isfinite(cost_matrix[row, col]):
                            continue
                        track_id, track = active_items[row]
                        detection = frame_detections[col]
                        track.update(detection, self.measurement_matrix, self.measurement_noise)
                        matched_track_ids.add(track_id)
                        matched_detection_indices.add(col)

            stale_track_ids = []
            for track_id, track in list(self.active_tracks.items()):
                if track_id in matched_track_ids:
                    continue
                track.consecutive_misses += 1
                if track.consecutive_misses > self.max_age:
                    stale_track_ids.append(track_id)

            for track_id in stale_track_ids:
                self.finished_tracks.append(self.active_tracks.pop(track_id))

            for detection_index, detection in enumerate(frame_detections):
                if detection_index not in matched_detection_indices:
                    self._spawn_track(detection)

        self.finished_tracks.extend(self.active_tracks.values())
        confirmed_tracks = [track for track in self.finished_tracks if track.is_confirmed(self.min_hits)]
        confirmed_tracks.sort(key=lambda track: track.track_id)
        return [
            GroundPlaneTrack(track_id=track.track_id, detections=list(track.detections))
            for track in confirmed_tracks
        ]


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


def track_ground_plane_detections_kalman(
    detections: Sequence[GroundPlaneDetection],
    max_age: int = 2,
    min_hits: int = 2,
    distance_gate: float = 25.0,
    max_euclidean_cm: float = 200.0,
    q_std: float = 5.0,
    r_std: float = 10.0,
) -> List[GroundPlaneTrack]:
    """Track MVDeTr detections with the tuned 12b Kalman operating point."""
    tracker = KalmanGroundPlaneTracker(
        max_age=max_age,
        min_hits=min_hits,
        distance_gate=distance_gate,
        max_euclidean_cm=max_euclidean_cm,
        q_std=q_std,
        r_std=r_std,
    )
    tracks = tracker.track(detections)
    logger.info(f"Kalman-tracked {len(tracks)} ground-plane identities")
    return tracks


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
    tracker: str = "kalman",
    kalman_max_age: int = 2,
    kalman_min_hits: int = 2,
    kalman_distance_gate: float = 25.0,
    kalman_max_euclidean_cm: float = 200.0,
    kalman_q_std: float = 5.0,
    kalman_r_std: float = 10.0,
) -> Tuple[Dict[str, List[Tracklet]], List[GlobalTrajectory]]:
    """Convert MVDeTr detections into projected tracklets and trajectories."""
    detections = load_mvdetr_ground_plane_detections(detections_path)
    calibrations = load_wildtrack_calibration(calibration_dir)
    if not calibrations:
        raise FileNotFoundError(f"No WILDTRACK calibrations loaded from {calibration_dir}")

    if tracker == "kalman":
        tracks = track_ground_plane_detections_kalman(
            detections=detections,
            max_age=kalman_max_age,
            min_hits=kalman_min_hits,
            distance_gate=kalman_distance_gate,
            max_euclidean_cm=kalman_max_euclidean_cm,
            q_std=kalman_q_std,
            r_std=kalman_r_std,
        )
    elif tracker in {"hungarian", "naive"}:
        tracks = track_ground_plane_detections(
            detections=detections,
            max_match_distance_cm=max_match_distance_cm,
            max_missed_frames=max_missed_frames,
            min_track_length=min_track_length,
        )
    else:
        raise ValueError(f"Unknown WILDTRACK tracker: {tracker}")
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
    parser.add_argument("--tracker", choices=["kalman", "hungarian", "naive"], default="kalman")
    parser.add_argument("--kalman-max-age", type=int, default=2)
    parser.add_argument("--kalman-min-hits", type=int, default=2)
    parser.add_argument("--kalman-distance-gate", type=float, default=25.0)
    parser.add_argument("--kalman-max-euclidean-cm", type=float, default=200.0)
    parser.add_argument("--kalman-q-std", type=float, default=5.0)
    parser.add_argument("--kalman-r-std", type=float, default=10.0)
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
        tracker=args.tracker,
        kalman_max_age=args.kalman_max_age,
        kalman_min_hits=args.kalman_min_hits,
        kalman_distance_gate=args.kalman_distance_gate,
        kalman_max_euclidean_cm=args.kalman_max_euclidean_cm,
        kalman_q_std=args.kalman_q_std,
        kalman_r_std=args.kalman_r_std,
    )


if __name__ == "__main__":
    main()