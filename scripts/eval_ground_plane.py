"""Ground-plane evaluation for WILDTRACK — the correct protocol used by published papers.

Published WILDTRACK results (MOTA 80-92%) evaluate on the ground plane:
  - GT: 3D ground positions per person per frame
  - Pred: 3D ground positions per person per frame  
  - Matching: L2 distance on ground plane (typically <50cm)
  - Metrics: MOTA, MODP, Precision, Recall

This script:
1. Loads GT ground positions from WILDTRACK annotations
2. Loads Stage 4 global trajectories, back-projects to ground plane
3. Matches using L2 distance
4. Computes MOTA, IDF1, Precision, Recall
"""
import sys, os, json, glob, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '.')

# Fix numpy 2.0 compat
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x, dtype=np.float64: np.asarray(x, dtype=dtype)

import motmetrics as mm
from src.core.io_utils import load_global_trajectories
from src.core.wildtrack_calibration import load_wildtrack_calibration

# ── Configuration ──────────────────────────────────────────────────────────
RUN_DIR = Path('data/outputs/run_20260304_050358')
WILDTRACK_DIR = Path('data/raw/wildtrack')
MATCH_THRESHOLD_CM = 50  # L2 distance threshold in cm (standard: 50cm ≈ 0.5m)

# WILDTRACK ground-plane grid: 480 x 1440, 2.5cm per cell
GRID_W = 480
GP_XMIN = -300  # cm
GP_YMIN = -900  # cm
CELL_SIZE = 2.5  # cm per cell


def posid_to_ground(pos_id: int) -> tuple:
    """Convert WILDTRACK positionID to ground-plane (gx, gy) in cm."""
    x_idx = pos_id % GRID_W
    y_idx = pos_id // GRID_W
    gx = GP_XMIN + x_idx * CELL_SIZE
    gy = GP_YMIN + y_idx * CELL_SIZE
    return gx, gy


def load_gt_ground_positions(annotations_dir: Path) -> dict:
    """Load GT ground-plane positions per frame.
    Returns: {frame_id: [(person_id, gx, gy), ...]}
    """
    gt = {}
    json_files = sorted(glob.glob(str(annotations_dir / '*.json')))
    for jf in json_files:
        # Frame ID from filename: 00000000.json -> 0, 00000005.json -> 1 (at 2fps, every 5th WILDTRACK frame)
        fname = Path(jf).stem
        wildtrack_frame = int(fname)
        frame_id = wildtrack_frame // 5  # Our pipeline uses sequential frame IDs
        
        data = json.load(open(jf))
        positions = []
        for p in data:
            pid = p['personID']
            pos_id = p['positionID']
            gx, gy = posid_to_ground(pos_id)
            positions.append((pid, gx, gy))
        gt[frame_id] = positions
    return gt


def pixel_to_ground(u: float, v: float, K, R, tvec):
    """Back-project image point to Z=0 ground plane."""
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
    return pt[0], pt[1]


def build_pred_ground_positions(trajectories, calibrations, conf_threshold=0.30, min_cameras=1):
    """Build predicted ground-plane positions per frame from global trajectories.
    
    For each trajectory, for each frame where it has observations:
      - Back-project foot position from each camera to ground plane
      - Average the positions across cameras
    
    Args:
        min_cameras: minimum number of cameras a trajectory must span to be included
    
    Returns: {frame_id: [(global_id, gx, gy), ...]}
    """
    cam_map = {f'C{i+1}': f'C{i+1}' for i in range(7)}
    
    pred = defaultdict(list)
    for traj in trajectories:
        # Filter by min cameras
        cam_ids = set(t.camera_id for t in traj.tracklets)
        if len(cam_ids) < min_cameras:
            continue
        
        # Collect per-frame ground positions from all cameras
        frame_positions = defaultdict(list)
        
        for tracklet in traj.tracklets:
            cam_id = tracklet.camera_id
            if cam_id not in calibrations:
                continue
            cal = calibrations[cam_id]
            K, R, tvec = cal['K'], cal['R'], cal['tvec']
            
            for tf in tracklet.frames:
                if tf.confidence < conf_threshold:
                    continue
                x1, y1, x2, y2 = tf.bbox
                foot_x = float(x1 + (x2 - x1) / 2)
                foot_y = float(y2)
                gp = pixel_to_ground(foot_x, foot_y, K, R, tvec)
                if gp is not None:
                    frame_positions[tf.frame_id].append(gp)
        
        # Average positions across cameras for each frame, filter to ground plane
        for frame_id, positions in frame_positions.items():
            if positions:
                avg_gx = np.mean([p[0] for p in positions])
                avg_gy = np.mean([p[1] for p in positions])
                # Filter: only keep if within ground-plane bounds + margin
                margin = 150  # cm
                if (GP_XMIN - margin) <= avg_gx <= (900 + margin) and \
                   (GP_YMIN - margin) <= avg_gy <= (2700 + margin):
                    pred[frame_id].append((traj.global_id, avg_gx, avg_gy))
    
    return dict(pred)


def ground_plane_nms(pred_positions, merge_radius_cm=50):
    """Merge nearby predictions on the ground plane per frame using DBSCAN clustering.
    For each frame, cluster predictions within merge_radius_cm.
    Each cluster becomes one prediction (centroid position, keep first ID).
    """
    from sklearn.cluster import DBSCAN
    
    merged = {}
    for frame_id, preds in pred_positions.items():
        if not preds:
            merged[frame_id] = []
            continue
        
        positions = np.array([[p[1], p[2]] for p in preds])
        ids = [p[0] for p in preds]
        
        # DBSCAN: cluster points within merge_radius
        clustering = DBSCAN(eps=merge_radius_cm, min_samples=1).fit(positions)
        
        kept = []
        for label in set(clustering.labels_):
            mask = clustering.labels_ == label
            cluster_pos = positions[mask]
            cluster_ids = [ids[i] for i in range(len(ids)) if mask[i]]
            
            # Use centroid position and most common trajectory ID
            centroid = cluster_pos.mean(axis=0)
            # Pick the ID that appears most (for tracking consistency)
            from collections import Counter
            best_id = Counter(cluster_ids).most_common(1)[0][0]
            kept.append((best_id, centroid[0], centroid[1]))
        
        merged[frame_id] = kept
    return merged


def evaluate_ground_plane(gt_positions, pred_positions, threshold_cm=50):
    """Evaluate using motmetrics with L2 distance matching on ground plane."""
    acc = mm.MOTAccumulator(auto_id=True)
    
    all_frames = sorted(set(list(gt_positions.keys()) + list(pred_positions.keys())))
    
    total_gt = 0
    total_pred = 0
    
    for frame_id in all_frames:
        gt_list = gt_positions.get(frame_id, [])
        pred_list = pred_positions.get(frame_id, [])
        
        gt_ids = [g[0] for g in gt_list]
        pred_ids = [p[0] for p in pred_list]
        
        total_gt += len(gt_ids)
        total_pred += len(pred_ids)
        
        if gt_list and pred_list:
            # Compute L2 distance matrix in cm
            gt_pos = np.array([[g[1], g[2]] for g in gt_list])
            pred_pos = np.array([[p[1], p[2]] for p in pred_list])
            
            # L2 distances
            dist = np.zeros((len(gt_pos), len(pred_pos)))
            for i in range(len(gt_pos)):
                for j in range(len(pred_pos)):
                    d = np.sqrt((gt_pos[i, 0] - pred_pos[j, 0])**2 + 
                                (gt_pos[i, 1] - pred_pos[j, 1])**2)
                    dist[i, j] = d if d <= threshold_cm else np.nan
        else:
            dist = np.empty((len(gt_list), len(pred_list)))
        
        acc.update(gt_ids, pred_ids, dist)
    
    mh = mm.metrics.create()
    summary = mh.compute(
        acc, 
        metrics=["mota", "motp", "idf1", "precision", "recall", 
                 "num_switches", "num_false_positives", "num_misses",
                 "num_objects", "num_predictions"],
        name="ground_plane"
    )
    
    return summary, total_gt, total_pred


def main():
    print("=" * 70)
    print("WILDTRACK Ground-Plane Evaluation")
    print("=" * 70)
    
    # Load calibrations
    print("\nLoading calibrations...")
    cals = load_wildtrack_calibration(str(WILDTRACK_DIR / 'calibrations'))
    
    # Load GT
    print("Loading GT ground positions...")
    annotations_dir = WILDTRACK_DIR / 'annotations_positions'
    gt = load_gt_ground_positions(annotations_dir)
    print(f"  GT: {len(gt)} frames, avg {np.mean([len(v) for v in gt.values()]):.1f} people/frame")
    
    # Verify GT positions
    all_gx, all_gy = [], []
    for positions in gt.values():
        for _, gx, gy in positions:
            all_gx.append(gx); all_gy.append(gy)
    print(f"  GT ground range: x=[{min(all_gx):.0f}, {max(all_gx):.0f}], y=[{min(all_gy):.0f}, {max(all_gy):.0f}]")
    
    # Load trajectories
    print("\nLoading global trajectories...")
    traj_path = RUN_DIR / 'stage4' / 'global_trajectories.json'
    trajectories = load_global_trajectories(traj_path)
    print(f"  {len(trajectories)} global trajectories")
    
    # Test with different confidence thresholds and matching distances
    # Test configurations: conf_thresh, match_dist, nms_radius, min_cameras, label
    configs = [
        (0.40, 50, 50,  1, "conf>=0.40, DBSCAN=50cm, all trajs"),
        (0.40, 50, 50,  2, "conf>=0.40, DBSCAN=50cm, >=2 cams"),
        (0.40, 50, 50,  3, "conf>=0.40, DBSCAN=50cm, >=3 cams"),
        (0.40, 100, 75,  1, "conf>=0.40, match=100, DBSCAN=75, all"),
        (0.40, 100, 75,  2, "conf>=0.40, match=100, DBSCAN=75, >=2 cams"),
    ]
    
    for conf_thresh, dist_thresh, nms_radius, min_cams, label in configs:
        print(f"\n{'─' * 50}")
        print(f"Config: {label}")
        print(f"{'─' * 50}")
        
        pred = build_pred_ground_positions(trajectories, cals, conf_threshold=conf_thresh, min_cameras=min_cams)
        
        if pred:
            avg_pred_raw = np.mean([len(v) for v in pred.values()])
        else:
            print("  No predictions!")
            continue
        
        # Apply ground-plane NMS
        pred_nms = ground_plane_nms(pred, merge_radius_cm=nms_radius)
        avg_pred_nms = np.mean([len(v) for v in pred_nms.values()])
        print(f"  Raw: {avg_pred_raw:.1f} pred/frame → NMS: {avg_pred_nms:.1f} pred/frame")
        
        summary, total_gt, total_pred = evaluate_ground_plane(gt, pred_nms, threshold_cm=dist_thresh)
        
        mota = float(summary["mota"].iloc[0])
        motp = float(summary["motp"].iloc[0])
        idf1 = float(summary["idf1"].iloc[0])
        prec = float(summary["precision"].iloc[0])
        rec = float(summary["recall"].iloc[0])
        idsw = int(summary["num_switches"].iloc[0])
        fp = int(summary["num_false_positives"].iloc[0])
        fn = int(summary["num_misses"].iloc[0])
        
        print(f"\n  MOTA:      {mota*100:.1f}%")
        print(f"  IDF1:      {idf1*100:.1f}%")
        print(f"  MOTP:      {motp:.1f} cm")
        print(f"  Precision: {prec*100:.1f}%")
        print(f"  Recall:    {rec*100:.1f}%")
        print(f"  ID Sw:     {idsw}")
        print(f"  FP:        {fp}")
        print(f"  FN:        {fn}")
        print(f"  Total GT:  {total_gt}")
        print(f"  Total Pred:{total_pred}")


if __name__ == '__main__':
    main()
