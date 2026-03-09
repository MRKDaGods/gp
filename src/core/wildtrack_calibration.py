"""Compute per-camera ROI polygons for WILDTRACK based on calibration.

Projects the WILDTRACK ground plane area to each camera's image plane
using intrinsic/extrinsic calibration. Generates a convex hull polygon
per camera that defines where valid detections can appear.

The WILDTRACK ground plane is defined in centimetres on a grid from
(-360, -900) to (1200, 3600) in the world coordinate system (cm).
This is approximately a 15.6m × 45m area centred on the public square.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger


# WILDTRACK ground plane bounds in CENTIMETRES (world coordinates)
# These define the rectangular area on the ground plane where people are annotated.
# Reference: WILDTRACK paper — grid from (-360,-900) to (1200,3600) with 2.5cm spacing
# This corresponds to (480+120)×(1440+360) grid cells = 600×1800 in 2.5cm units
GP_XMIN, GP_XMAX = -480, 1440   # cm  (wider than paper to include margin)
GP_YMIN, GP_YMAX = -1120, 3880  # cm
GP_Z = 0.0                       # ground plane at z=0


# Camera naming: CVLab1..4 → C1..C4, IDIAP1..3 → C5..C7
_CALIB_NAMES = {
    "C1": "CVLab1", "C2": "CVLab2", "C3": "CVLab3", "C4": "CVLab4",
    "C5": "IDIAP1", "C6": "IDIAP2", "C7": "IDIAP3",
}


def _parse_opencv_xml_vector(path: Path, key: str) -> np.ndarray:
    """Parse a vector from an OpenCV XML storage file."""
    tree = ET.parse(path)
    root = tree.getroot()
    node = root.find(key)
    if node is None:
        raise ValueError(f"Key '{key}' not found in {path}")
    text = node.text.strip()
    return np.array([float(x) for x in text.split()])


def _parse_opencv_xml_matrix(path: Path, key: str) -> np.ndarray:
    """Parse a matrix from an OpenCV XML storage file."""
    tree = ET.parse(path)
    root = tree.getroot()
    node = root.find(key)
    if node is None:
        raise ValueError(f"Key '{key}' not found in {path}")
    rows = int(node.find("rows").text)
    cols = int(node.find("cols").text)
    data_text = node.find("data").text.strip()
    values = [float(x) for x in data_text.split()]
    return np.array(values).reshape(rows, cols)


def load_wildtrack_calibration(calibration_dir: str | Path) -> Dict[str, dict]:
    """Load all WILDTRACK camera calibrations.

    Returns:
        Dict[camera_id, {"K": 3x3, "rvec": 3, "tvec": 3, "R": 3x3}]
    """
    cal_dir = Path(calibration_dir)
    cameras = {}

    for cam_id, calib_name in _CALIB_NAMES.items():
        intr_path = cal_dir / "intrinsic_zero" / f"intr_{calib_name}.xml"
        extr_path = cal_dir / "extrinsic" / f"extr_{calib_name}.xml"

        if not intr_path.exists() or not extr_path.exists():
            logger.warning(f"Missing calibration for {cam_id}: {intr_path}, {extr_path}")
            continue

        K = _parse_opencv_xml_matrix(intr_path, "camera_matrix")
        rvec = _parse_opencv_xml_vector(extr_path, "rvec")
        tvec = _parse_opencv_xml_vector(extr_path, "tvec")
        R, _ = cv2.Rodrigues(rvec)

        cameras[cam_id] = {"K": K, "rvec": rvec, "tvec": tvec, "R": R}

    return cameras


def project_world_to_image(
    world_points: np.ndarray,  # (N, 3)
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Project 3D world points to 2D image coordinates."""
    pts_2d, _ = cv2.projectPoints(
        world_points.astype(np.float64),
        rvec.astype(np.float64),
        tvec.astype(np.float64),
        K.astype(np.float64),
        None,
    )
    return pts_2d.reshape(-1, 2)


def compute_ground_plane_roi(
    calibrations: Dict[str, dict],
    frame_size: Tuple[int, int] = (1920, 1080),
    n_edge_points: int = 100,
) -> Dict[str, np.ndarray]:
    """Compute per-camera ROI polygon from the WILDTRACK ground plane.

    Densely samples points along the edges of the ground plane rectangle,
    projects them to each camera, clips to frame boundaries, and returns
    the convex hull as a polygon.

    Args:
        calibrations: From load_wildtrack_calibration().
        frame_size: (width, height) of the camera frames.
        n_edge_points: Number of sample points per edge.

    Returns:
        Dict[camera_id, polygon_points] where polygon_points is (M, 2) array
        of (x, y) vertices.
    """
    W, H = frame_size

    # Dense sample the ground plane boundary
    gp_points = []
    for x in np.linspace(GP_XMIN, GP_XMAX, n_edge_points):
        gp_points.append([x, GP_YMIN, GP_Z])
        gp_points.append([x, GP_YMAX, GP_Z])
    for y in np.linspace(GP_YMIN, GP_YMAX, n_edge_points):
        gp_points.append([GP_XMIN, y, GP_Z])
        gp_points.append([GP_XMAX, y, GP_Z])
    # Also add interior grid for better coverage
    for x in np.linspace(GP_XMIN, GP_XMAX, 20):
        for y in np.linspace(GP_YMIN, GP_YMAX, 20):
            gp_points.append([x, y, GP_Z])
    gp_points = np.array(gp_points)

    rois = {}
    for cam_id, cal in calibrations.items():
        img_pts = project_world_to_image(gp_points, cal["K"], cal["rvec"], cal["tvec"])

        # Clip to frame
        img_pts[:, 0] = np.clip(img_pts[:, 0], 0, W)
        img_pts[:, 1] = np.clip(img_pts[:, 1], 0, H)

        # Convex hull
        hull = cv2.convexHull(img_pts.astype(np.float32))
        rois[cam_id] = hull.reshape(-1, 2)

    return rois


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """Check if a 2D point is inside a convex polygon using cv2."""
    result = cv2.pointPolygonTest(
        polygon.astype(np.float32).reshape(-1, 1, 2),
        point,
        False,
    )
    return result >= 0


def filter_detections_by_roi(
    detections: list,
    roi_polygon: np.ndarray,
) -> list:
    """Filter detections to only those whose foot position is inside the ROI.

    Args:
        detections: List of (frame_id, track_id, x, y, w, h, ...) tuples.
        roi_polygon: (M, 2) array of polygon vertices.

    Returns:
        Filtered list of detections.
    """
    polygon = roi_polygon.astype(np.float32).reshape(-1, 1, 2)
    filtered = []
    for det in detections:
        x, y, w, h = det[2], det[3], det[4], det[5]
        foot_x = x + w / 2
        foot_y = y + h
        if cv2.pointPolygonTest(polygon, (float(foot_x), float(foot_y)), False) >= 0:
            filtered.append(det)
    return filtered


def save_roi_polygons(rois: Dict[str, np.ndarray], output_path: str | Path) -> None:
    """Save ROI polygons to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {cam_id: poly.tolist() for cam_id, poly in rois.items()}
    with open(output_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Saved ROI polygons for {len(rois)} cameras to {output_path}")


def load_roi_polygons(path: str | Path) -> Dict[str, np.ndarray]:
    """Load ROI polygons from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}


if __name__ == "__main__":
    # Quick test: compute and display ROI polygons
    cal_dir = Path("data/raw/wildtrack/calibrations")
    cals = load_wildtrack_calibration(cal_dir)
    rois = compute_ground_plane_roi(cals)

    for cam_id, poly in sorted(rois.items()):
        area = cv2.contourArea(poly.astype(np.float32).reshape(-1, 1, 2))
        frame_area = 1920 * 1080
        print(f"{cam_id}: {len(poly)} vertices, area={area:.0f} ({100*area/frame_area:.1f}% of frame)")
        print(f"  x=[{poly[:,0].min():.0f}, {poly[:,0].max():.0f}]  "
              f"y=[{poly[:,1].min():.0f}, {poly[:,1].max():.0f}]")

    save_roi_polygons(rois, "data/raw/wildtrack/manifests/roi_polygons.json")
