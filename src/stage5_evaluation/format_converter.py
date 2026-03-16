"""Convert between tracking output formats."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.core.data_models import GlobalTrajectory


def _load_wildtrack_roi(
    annotations_dir: str | Path,
    margin_cm: float = 100.0,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute ground-plane bounding box from WILDTRACK GT annotations.

    Returns (xmin, ymin, xmax, ymax) in cm, expanded by *margin_cm*.
    """
    import json
    import glob

    annotations_dir = Path(annotations_dir)
    if not annotations_dir.exists():
        return None

    GRID_W = 480
    GP_XMIN, GP_YMIN = -300.0, -900.0
    CELL_SIZE = 2.5

    xs, ys = [], []
    for jf in sorted(glob.glob(str(annotations_dir / "*.json")))[:10]:  # sample 10 frames
        with open(jf) as f:
            for entry in json.load(f):
                pid = int(entry["positionID"])
                row, col = divmod(pid, GRID_W)
                gx = GP_XMIN + col * CELL_SIZE
                gy = GP_YMIN + row * CELL_SIZE
                xs.append(gx)
                ys.append(gy)

    if not xs:
        return None

    return (
        min(xs) - margin_cm,
        min(ys) - margin_cm,
        max(xs) + margin_cm,
        max(ys) + margin_cm,
    )


def _foot_to_ground(
    px: float, py: float,
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
) -> Tuple[float, float]:
    """Back-project a pixel foot point (px, py) onto the Z=0 ground plane.

    Returns ground-plane (x, y) in cm.
    """
    K_inv = np.linalg.inv(K)
    ray = K_inv @ np.array([px, py, 1.0])
    # Camera centre in world coords
    C = -R.T @ t
    d = R.T @ ray
    # Intersect with Z=0: C[2] + s * d[2] = 0
    if abs(d[2]) < 1e-9:
        return float("nan"), float("nan")
    s = -C[2] / d[2]
    pt = C + s * d
    return float(pt[0]), float(pt[1])


def trajectories_to_mot_submission(
    trajectories: List[GlobalTrajectory],
    output_dir: str | Path,
    roi_config: Optional[Dict] = None,
) -> None:
    """Convert global trajectories to MOTChallenge submission format.

    Creates one text file per camera with lines:
        frame_id, global_id, x, y, w, h, confidence, class, visibility

    Args:
        trajectories: Global trajectories from Stage 4.
        output_dir: Output directory for MOT-format files.
        roi_config: Optional dict with keys:
            - annotations_dir: path to WILDTRACK annotations_positions
            - calibrations_dir: path to WILDTRACK calibrations
            - margin_cm: ground-plane ROI margin (default 100)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Optional ground-plane ROI filter (WILDTRACK) ──────────────────
    roi_bbox = None
    cam_calibs = None
    if roi_config:
        ann_dir = roi_config.get("annotations_dir")
        cal_dir = roi_config.get("calibrations_dir")
        margin = float(roi_config.get("margin_cm", 100.0))
        if ann_dir:
            roi_bbox = _load_wildtrack_roi(ann_dir, margin)
        if cal_dir and roi_bbox is not None:
            from src.stage5_evaluation.ground_plane_eval import _load_calibration
            cam_calibs = _load_calibration(Path(cal_dir))
            logger.info(
                f"ROI filter active: bbox=({roi_bbox[0]:.0f},{roi_bbox[1]:.0f})-"
                f"({roi_bbox[2]:.0f},{roi_bbox[3]:.0f}) cm, "
                f"calibrations for {len(cam_calibs)} cameras"
            )

    # Group all detections by camera
    camera_rows = {}
    filtered_count = 0
    nan_count = 0
    invalid_bbox_count = 0
    total_count = 0

    for traj in trajectories:
        for tracklet in traj.tracklets:
            cam_id = tracklet.camera_id
            if cam_id not in camera_rows:
                camera_rows[cam_id] = []

            for frame in tracklet.frames:
                x1, y1, x2, y2 = frame.bbox
                w = x2 - x1
                h = y2 - y1
                total_count += 1

                # Skip invalid bounding boxes (negative width/height)
                if w <= 0 or h <= 0:
                    invalid_bbox_count += 1
                    continue

                # ROI filter: check if foot point projects inside GP ROI
                if roi_bbox is not None and cam_calibs is not None and cam_id in cam_calibs:
                    foot_x = (x1 + x2) / 2.0
                    foot_y = y2  # bottom centre = foot
                    calib = cam_calibs[cam_id]
                    gx, gy = _foot_to_ground(
                        foot_x, foot_y,
                        calib["K"], calib["R"], calib["tvec"],
                    )
                    if np.isnan(gx) or np.isnan(gy):
                        nan_count += 1
                        filtered_count += 1
                        continue
                    if (
                        gx < roi_bbox[0] or gx > roi_bbox[2]
                        or gy < roi_bbox[1] or gy > roi_bbox[3]
                    ):
                        filtered_count += 1
                        continue

                row = (
                    frame.frame_id,
                    traj.global_id,
                    x1, y1, w, h,
                    frame.confidence,
                    tracklet.class_id,
                    1.0,  # visibility
                )
                camera_rows[cam_id].append(row)

    if filtered_count > 0:
        logger.info(
            f"ROI filter removed {filtered_count}/{total_count} detections "
            f"({filtered_count / total_count * 100:.1f}%)"
            + (f", {nan_count} NaN projections" if nan_count else "")
        )
    if invalid_bbox_count > 0:
        logger.warning(
            f"Skipped {invalid_bbox_count} detections with invalid bounding boxes (w<=0 or h<=0)"
        )

    # Write per-camera files
    for cam_id, rows in camera_rows.items():
        rows.sort(key=lambda r: (r[0], r[1]))
        # Deduplicate same (frame, global_id) entries — keep highest confidence.
        # These can arise when the graph solver incorrectly merges two same-camera
        # tracklets that share overlapping frames.
        seen: dict = {}
        dedup_rows = []
        dup_count = 0
        for row in rows:
            key = (row[0], row[1])  # (frame_id, global_id)
            if key in seen:
                dup_count += 1
                if row[6] > seen[key][6]:  # replace if higher confidence
                    seen[key] = row
            else:
                seen[key] = row
                dedup_rows.append(row)
        if dup_count > 0:
            # Rebuild dedup list preserving insertion order
            dedup_rows = [seen[k] for k in dict.fromkeys((r[0], r[1]) for r in rows)]
            logger.warning(f"{cam_id}: removed {dup_count} duplicate (frame,id) rows")
        file_path = output_dir / f"{cam_id}.txt"
        with open(file_path, "w") as f:
            for row in dedup_rows:
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
