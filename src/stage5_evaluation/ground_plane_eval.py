"""Ground-plane evaluation for multi-view overlapping camera datasets.

WILDTRACK and similar multi-view benchmarks evaluate on the **ground plane**:
- GT: 3D ground positions per person per frame (from annotations_positions JSON)
- Pred: Back-projected foot positions from per-camera tracklets
- Matching: L2 distance on ground plane (typically ≤50cm)
- Metrics: MODA, MODP, Precision, Recall (via motmetrics)

This is the correct evaluation protocol used by published SOTA methods
(MVDet MODA=88.2%, MVDeTr MODA=91.5%).

For city-wide MTMC (CityFlowV2, VeRi, Market), use per-camera 2D MOT eval instead.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.core.data_models import EvaluationResult, GlobalTrajectory

# Shim for motmetrics + NumPy 2.0
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]


# ── WILDTRACK constants ────────────────────────────────────────────────────
GRID_W = 480        # grid columns
GP_XMIN = -300.0    # cm
GP_YMIN = -900.0    # cm
CELL_SIZE = 2.5     # cm per cell

# Camera name mapping: WILDTRACK calib files → our camera IDs
_CAM_NAME_MAP = {
    "CVLab1": "C1", "CVLab2": "C2", "CVLab3": "C3", "CVLab4": "C4",
    "IDIAP1": "C5", "IDIAP2": "C6", "IDIAP3": "C7",
}


# ── Calibration loading ───────────────────────────────────────────────────

def _load_calibration(calibrations_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load WILDTRACK calibration files (intrinsic_zero + extrinsic)."""
    cals: Dict[str, Dict[str, np.ndarray]] = {}
    extrinsic_dir = calibrations_dir / "extrinsic"

    for xml_file in sorted(extrinsic_dir.glob("*.xml")):
        stem = xml_file.stem  # e.g. "extr_CVLab1"
        lab_name = stem.replace("extr_", "")
        cam_id = _CAM_NAME_MAP.get(lab_name)
        if cam_id is None:
            continue

        # Load extrinsic — rvec/tvec may be plain text in WILDTRACK XMLs
        fs = cv2.FileStorage(str(xml_file), cv2.FILE_STORAGE_READ)
        rvec_node = fs.getNode("rvec")
        tvec_node = fs.getNode("tvec")

        # Try matrix format first, fall back to text parsing
        rvec = _read_vec_node(rvec_node, xml_file, "rvec")
        tvec = _read_vec_node(tvec_node, xml_file, "tvec")
        fs.release()

        if rvec is None or tvec is None:
            logger.warning(f"Could not parse rvec/tvec from {xml_file}")
            continue

        R, _ = cv2.Rodrigues(rvec)

        # Load intrinsic (zero-distortion version)
        intr_file = calibrations_dir / "intrinsic_zero" / f"intr_{lab_name}.xml"
        if not intr_file.exists():
            continue
        fs = cv2.FileStorage(str(intr_file), cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        fs.release()

        cals[cam_id] = {"K": K, "R": R, "tvec": tvec}

    return cals


def _read_vec_node(
    node: cv2.FileNode, xml_path: Path, tag: str,
) -> Optional[np.ndarray]:
    """Read a vector node from OpenCV FileStorage, handling both matrix and text formats."""
    try:
        mat = node.mat()
        if mat is not None:
            return mat.flatten()
    except Exception:
        pass

    # Try sequence format
    try:
        if node.isSeq():
            return np.array([node.at(i).real() for i in range(int(node.size()))])
    except Exception:
        pass

    # Fall back to parsing raw XML text
    return _parse_text_node(xml_path, tag)


def _parse_text_node(xml_path: Path, tag: str) -> Optional[np.ndarray]:
    """Parse a plain-text numeric node from an OpenCV XML file."""
    import re
    text = xml_path.read_text()
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        values = [float(x) for x in match.group(1).split()]
        return np.array(values)
    return None


def _pixel_to_ground(
    u: float, v: float, K: np.ndarray, R: np.ndarray, tvec: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Back-project image point (u, v) to Z=0 ground plane → (gx, gy) in cm."""
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam
    cam_center = -R.T @ tvec
    if abs(ray_world[2]) < 1e-10:
        return None
    lam = -cam_center[2] / ray_world[2]
    if lam < 0:
        return None
    pt = cam_center + lam * ray_world
    return float(pt[0]), float(pt[1])


# ── GT loading ─────────────────────────────────────────────────────────────

def _posid_to_ground(pos_id: int) -> Tuple[float, float]:
    """Convert WILDTRACK positionID to ground-plane (gx, gy) in cm."""
    x_idx = pos_id % GRID_W
    y_idx = pos_id // GRID_W
    gx = GP_XMIN + x_idx * CELL_SIZE
    gy = GP_YMIN + y_idx * CELL_SIZE
    return gx, gy


def load_gt_ground_positions(
    annotations_dir: Path,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Load GT ground-plane positions per frame.

    Returns: {frame_id: [(person_id, gx, gy), ...]}
    """
    gt: Dict[int, List[Tuple[int, float, float]]] = {}
    json_files = sorted(glob.glob(str(annotations_dir / "*.json")))

    for jf in json_files:
        fname = Path(jf).stem
        wildtrack_frame = int(fname)
        frame_id = wildtrack_frame // 5  # WILDTRACK uses every-5th naming

        data = json.load(open(jf))
        positions = []
        for p in data:
            pid = p["personID"]
            pos_id = p["positionID"]
            gx, gy = _posid_to_ground(pos_id)
            positions.append((pid, gx, gy))
        gt[frame_id] = positions

    return gt


# ── Prediction building ───────────────────────────────────────────────────

def build_pred_ground_positions(
    trajectories: List[GlobalTrajectory],
    calibrations: Dict[str, Dict[str, np.ndarray]],
    conf_threshold: float = 0.25,
    gp_bounds: Optional[Tuple[float, float, float, float]] = None,
    gp_margin: float = 150.0,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Build ground-plane predictions from global trajectories.

    For each trajectory at each frame, back-projects foot positions from
    all available cameras and averages them.

    Args:
        trajectories: Global trajectories from Stage 4.
        calibrations: Per-camera calibration dicts.
        conf_threshold: Minimum detection confidence.
        gp_bounds: (xmin, ymin, xmax, ymax) in cm for ground-plane filtering.
        gp_margin: Extra margin around bounds (cm).

    Returns: {frame_id: [(global_id, avg_gx, avg_gy), ...]}
    """
    if gp_bounds is None:
        gp_bounds = (GP_XMIN, GP_YMIN, 900.0, 2700.0)

    xmin, ymin, xmax, ymax = gp_bounds

    pred: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    for traj in trajectories:
        frame_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

        for tracklet in traj.tracklets:
            cam_id = tracklet.camera_id
            if cam_id not in calibrations:
                continue
            cal = calibrations[cam_id]
            K, R, tvec = cal["K"], cal["R"], cal["tvec"]

            for tf in tracklet.frames:
                if tf.confidence < conf_threshold:
                    continue
                x1, y1, x2, y2 = tf.bbox
                foot_x = float(x1 + (x2 - x1) / 2)
                foot_y = float(y2)
                gp = _pixel_to_ground(foot_x, foot_y, K, R, tvec)
                if gp is not None:
                    frame_positions[tf.frame_id].append(gp)

        for frame_id, positions in frame_positions.items():
            if not positions:
                continue
            avg_gx = float(np.mean([p[0] for p in positions]))
            avg_gy = float(np.mean([p[1] for p in positions]))
            # Filter to ground-plane bounds
            if (xmin - gp_margin) <= avg_gx <= (xmax + gp_margin) and \
               (ymin - gp_margin) <= avg_gy <= (ymax + gp_margin):
                pred[frame_id].append((traj.global_id, avg_gx, avg_gy))

    return dict(pred)


def ground_plane_nms(
    pred_positions: Dict[int, List[Tuple[int, float, float]]],
    merge_radius_cm: float = 50.0,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Merge nearby predictions on the ground plane per frame using DBSCAN."""
    from sklearn.cluster import DBSCAN

    merged: Dict[int, List[Tuple[int, float, float]]] = {}

    for frame_id, preds in pred_positions.items():
        if not preds:
            merged[frame_id] = []
            continue

        positions = np.array([[p[1], p[2]] for p in preds])
        ids = [p[0] for p in preds]

        clustering = DBSCAN(eps=merge_radius_cm, min_samples=1).fit(positions)

        kept = []
        for label in set(clustering.labels_):
            mask = clustering.labels_ == label
            cluster_pos = positions[mask]
            cluster_ids = [ids[i] for i in range(len(ids)) if mask[i]]
            centroid = cluster_pos.mean(axis=0)
            from collections import Counter
            best_id = Counter(cluster_ids).most_common(1)[0][0]
            kept.append((best_id, float(centroid[0]), float(centroid[1])))

        merged[frame_id] = kept

    return merged


# ── Main evaluation ───────────────────────────────────────────────────────

def evaluate_ground_plane(
    gt_positions: Dict[int, List[Tuple[int, float, float]]],
    pred_positions: Dict[int, List[Tuple[int, float, float]]],
    threshold_cm: float = 50.0,
) -> Dict[str, Any]:
    """Evaluate ground-plane detections using motmetrics with L2 matching.

    Args:
        gt_positions: {frame_id: [(pid, gx, gy), ...]}
        pred_positions: {frame_id: [(tid, gx, gy), ...]}
        threshold_cm: L2 distance threshold for matching (standard: 50cm).

    Returns:
        Dict with MODA, MODP, IDF1, Precision, Recall, etc.
    """
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(list(gt_positions.keys()) + list(pred_positions.keys())))

    for frame_id in all_frames:
        gt_list = gt_positions.get(frame_id, [])
        pred_list = pred_positions.get(frame_id, [])

        gt_ids = [g[0] for g in gt_list]
        pred_ids = [p[0] for p in pred_list]

        if gt_list and pred_list:
            gt_pos = np.array([[g[1], g[2]] for g in gt_list])
            pred_pos = np.array([[p[1], p[2]] for p in pred_list])

            dist = np.zeros((len(gt_pos), len(pred_pos)))
            for i in range(len(gt_pos)):
                for j in range(len(pred_pos)):
                    d = np.sqrt(
                        (gt_pos[i, 0] - pred_pos[j, 0]) ** 2
                        + (gt_pos[i, 1] - pred_pos[j, 1]) ** 2
                    )
                    dist[i, j] = d if d <= threshold_cm else np.nan
        else:
            dist = np.empty((len(gt_list), len(pred_list)))

        acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota", "motp", "idf1", "precision", "recall",
            "num_switches", "num_false_positives", "num_misses",
            "num_objects", "num_predictions",
        ],
        name="ground_plane",
    )

    return {
        "moda": float(summary["mota"].iloc[0]),  # In motmetrics, 'mota' IS moda for detection
        "modp": float(summary["motp"].iloc[0]),
        "idf1": float(summary["idf1"].iloc[0]),
        "precision": float(summary["precision"].iloc[0]),
        "recall": float(summary["recall"].iloc[0]),
        "id_switches": int(summary["num_switches"].iloc[0]),
        "false_positives": int(summary["num_false_positives"].iloc[0]),
        "misses": int(summary["num_misses"].iloc[0]),
    }


def evaluate_wildtrack_ground_plane(
    trajectories: List[GlobalTrajectory],
    annotations_dir: str | Path,
    calibrations_dir: str | Path,
    conf_threshold: float = 0.25,
    match_threshold_cm: float = 50.0,
    nms_radius_cm: float = 50.0,
) -> EvaluationResult:
    """Full WILDTRACK ground-plane evaluation pipeline.

    This is the evaluation protocol used by published SOTA methods.

    Args:
        trajectories: Global trajectories from Stage 4.
        annotations_dir: Path to WILDTRACK annotations_positions/ directory.
        calibrations_dir: Path to WILDTRACK calibrations/ directory.
        conf_threshold: Min detection confidence for predictions.
        match_threshold_cm: L2 distance threshold for GT-pred matching.
        nms_radius_cm: DBSCAN radius for merging overlapping ground-plane predictions.

    Returns:
        EvaluationResult with ground-plane metrics.
    """
    annotations_dir = Path(annotations_dir)
    calibrations_dir = Path(calibrations_dir)

    # Load calibrations
    cals = _load_calibration(calibrations_dir)
    if not cals:
        logger.error(f"No calibrations loaded from {calibrations_dir}")
        return EvaluationResult()

    logger.info(f"Loaded calibrations for {len(cals)} cameras: {sorted(cals.keys())}")

    # Load GT
    gt = load_gt_ground_positions(annotations_dir)
    logger.info(
        f"GT: {len(gt)} frames, "
        f"avg {np.mean([len(v) for v in gt.values()]):.1f} people/frame"
    )

    # Build predictions
    pred = build_pred_ground_positions(
        trajectories, cals, conf_threshold=conf_threshold,
    )

    if not pred:
        logger.error("No ground-plane predictions generated")
        return EvaluationResult()

    avg_pred = np.mean([len(v) for v in pred.values()])
    logger.info(f"Predictions (raw): {len(pred)} frames, avg {avg_pred:.1f}/frame")

    # Apply NMS
    pred = ground_plane_nms(pred, merge_radius_cm=nms_radius_cm)
    avg_pred_nms = np.mean([len(v) for v in pred.values()])
    logger.info(f"Predictions (after NMS r={nms_radius_cm}cm): avg {avg_pred_nms:.1f}/frame")

    # Evaluate
    metrics = evaluate_ground_plane(gt, pred, threshold_cm=match_threshold_cm)

    logger.info(
        f"Ground-plane results: MODA={metrics['moda']*100:.1f}%, "
        f"IDF1={metrics['idf1']*100:.1f}%, "
        f"Precision={metrics['precision']*100:.1f}%, "
        f"Recall={metrics['recall']*100:.1f}%, "
        f"IDSW={metrics['id_switches']}"
    )

    return EvaluationResult(
        mota=metrics["moda"],
        idf1=metrics["idf1"],
        id_switches=metrics["id_switches"],
        details={
            "evaluation_type": "ground_plane",
            "modp_cm": metrics["modp"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "false_positives": metrics["false_positives"],
            "misses": metrics["misses"],
            "match_threshold_cm": match_threshold_cm,
            "nms_radius_cm": nms_radius_cm,
            "conf_threshold": conf_threshold,
        },
    )
