"""Ground-plane evaluation using RAW Stage 1 tracklets (bypasses Stage 4).

For each frame:
  1. Collect all active detections from 7 cameras
  2. Back-project foot position to ground plane
  3. Filter to ground-plane bounds
  4. Cluster with DBSCAN → one detection per cluster
  5. Use consecutive-frame Hungarian matching for tracking IDs
"""
import sys, os, json, glob, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

sys.path.insert(0, '.')

# Fix numpy 2.0 compat
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x, dtype=np.float64: np.asarray(x, dtype=dtype)

import motmetrics as mm
from src.core.io_utils import load_tracklets_by_camera
from src.core.wildtrack_calibration import load_wildtrack_calibration

# ── Configuration ──────────────────────────────────────────────────────────
RUN_DIR = Path('data/outputs/run_20260304_050358')
WILDTRACK_DIR = Path('data/raw/wildtrack')

GP_XMIN, GP_XMAX = -300, 900
GP_YMIN, GP_YMAX = -900, 2700
GRID_W = 480
CELL_SIZE = 2.5


def posid_to_ground(pos_id):
    x_idx = pos_id % GRID_W
    y_idx = pos_id // GRID_W
    gx = GP_XMIN + x_idx * CELL_SIZE
    gy = GP_YMIN + y_idx * CELL_SIZE
    return gx, gy


def load_gt(annotations_dir):
    gt = {}
    for jf in sorted(glob.glob(str(annotations_dir / '*.json'))):
        fid = int(Path(jf).stem) // 5
        data = json.load(open(jf))
        gt[fid] = [(p['personID'], *posid_to_ground(p['positionID'])) for p in data]
    return gt


def pixel_to_ground(u, v, K, R, tvec):
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam
    cam_center = -R.T @ tvec
    if abs(ray_world[2]) < 1e-10: return None
    lam = -cam_center[2] / ray_world[2]
    if lam < 0: return None
    pt = cam_center + lam * ray_world
    return pt[0], pt[1]


def build_multiview_detections(tracklets_by_cam, calibrations, 
                                conf_threshold=0.40, gp_margin=100, 
                                dbscan_eps=50, min_cameras_per_cluster=2):
    """Build ground-plane detections from multi-view clustering.
    
    For each frame:
      - Collect all detections from all cameras
      - Back-project to ground plane
      - Filter to GP bounds
      - DBSCAN cluster
      - Only keep clusters with detections from >= min_cameras cameras
      - Return cluster centroids with camera count as confidence
    """
    cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    
    # Build per-frame detections from all cameras
    frame_detections = defaultdict(list)  # {frame_id: [(cam_id, track_id, gx, gy), ...]}
    
    for cam_id in cameras:
        if cam_id not in calibrations: continue
        cal = calibrations[cam_id]
        K, R, tvec = cal['K'], cal['R'], cal['tvec']
        
        for t in tracklets_by_cam.get(cam_id, []):
            for tf in t.frames:
                if tf.confidence < conf_threshold:
                    continue
                x1, y1, x2, y2 = tf.bbox
                foot_x = float(x1 + (x2 - x1) / 2)
                foot_y = float(y2)
                gp = pixel_to_ground(foot_x, foot_y, K, R, tvec)
                if gp is None: continue
                gx, gy = gp
                # Filter to GP bounds
                if not ((GP_XMIN - gp_margin) <= gx <= (GP_XMAX + gp_margin) and
                        (GP_YMIN - gp_margin) <= gy <= (GP_YMAX + gp_margin)):
                    continue
                frame_detections[tf.frame_id].append((cam_id, t.track_id, gx, gy))
    
    # Cluster per frame
    result = {}
    for frame_id in sorted(frame_detections.keys()):
        dets = frame_detections[frame_id]
        if not dets:
            result[frame_id] = []
            continue
        
        positions = np.array([[d[2], d[3]] for d in dets])
        cam_ids = [d[0] for d in dets]
        
        clustering = DBSCAN(eps=dbscan_eps, min_samples=1).fit(positions)
        
        clusters = []
        for label in set(clustering.labels_):
            mask = clustering.labels_ == label
            cluster_pos = positions[mask]
            cluster_cams = set(cam_ids[i] for i in range(len(cam_ids)) if mask[i])
            
            if len(cluster_cams) >= min_cameras_per_cluster:
                centroid = cluster_pos.mean(axis=0)
                clusters.append((centroid[0], centroid[1], len(cluster_cams)))
        
        result[frame_id] = clusters
    
    return result


def assign_ids(frame_clusters, match_threshold=100):
    """Assign consistent IDs across frames using Hungarian matching.
    
    Args:
        frame_clusters: {frame_id: [(gx, gy, n_cams), ...]}
        match_threshold: max L2 distance for matching between frames
    
    Returns: {frame_id: [(track_id, gx, gy), ...]}
    """
    next_id = 1
    prev_tracks = {}  # {track_id: (gx, gy)}
    result = {}
    
    for frame_id in sorted(frame_clusters.keys()):
        clusters = frame_clusters[frame_id]
        if not clusters:
            result[frame_id] = []
            continue
        
        curr_positions = [(c[0], c[1]) for c in clusters]
        
        if not prev_tracks:
            # First frame: assign new IDs
            frame_result = []
            for gx, gy, _ in clusters:
                frame_result.append((next_id, gx, gy))
                prev_tracks[next_id] = (gx, gy)
                next_id += 1
            result[frame_id] = frame_result
            continue
        
        # Build cost matrix
        prev_ids = list(prev_tracks.keys())
        prev_pos = [prev_tracks[tid] for tid in prev_ids]
        
        cost = np.zeros((len(prev_pos), len(curr_positions)))
        for i, (pgx, pgy) in enumerate(prev_pos):
            for j, (cgx, cgy) in enumerate(curr_positions):
                cost[i, j] = np.sqrt((pgx - cgx)**2 + (pgy - cgy)**2)
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost)
        
        frame_result = [None] * len(clusters)
        matched_curr = set()
        new_prev = {}
        
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= match_threshold:
                tid = prev_ids[r]
                gx, gy = curr_positions[c]
                frame_result[c] = (tid, gx, gy)
                matched_curr.add(c)
                new_prev[tid] = (gx, gy)
        
        # Assign new IDs to unmatched detections
        for j in range(len(clusters)):
            if j not in matched_curr:
                gx, gy = curr_positions[j]
                frame_result[j] = (next_id, gx, gy)
                new_prev[next_id] = (gx, gy)
                next_id += 1
        
        result[frame_id] = [f for f in frame_result if f is not None]
        prev_tracks = new_prev
    
    return result


def evaluate(gt_positions, pred_positions, threshold_cm=50):
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
            dist = np.full((len(gt_pos), len(pred_pos)), np.nan)
            for i in range(len(gt_pos)):
                for j in range(len(pred_pos)):
                    d = np.sqrt((gt_pos[i,0]-pred_pos[j,0])**2 + (gt_pos[i,1]-pred_pos[j,1])**2)
                    if d <= threshold_cm:
                        dist[i, j] = d
        else:
            dist = np.empty((len(gt_list), len(pred_list)))
        
        acc.update(gt_ids, pred_ids, dist)
    
    mh = mm.metrics.create()
    summary = mh.compute(
        acc, 
        metrics=["mota", "motp", "idf1", "precision", "recall",
                 "num_switches", "num_false_positives", "num_misses"],
        name="gp"
    )
    return summary


def main():
    print("=" * 70)
    print("WILDTRACK Ground-Plane Eval — Raw Multi-View Clustering")
    print("=" * 70)
    
    cals = load_wildtrack_calibration(str(WILDTRACK_DIR / 'calibrations'))
    gt = load_gt(WILDTRACK_DIR / 'annotations_positions')
    print(f"GT: {len(gt)} frames, avg {np.mean([len(v) for v in gt.values()]):.1f} people/frame")
    
    tracklets = load_tracklets_by_camera(str(RUN_DIR / 'stage1'))
    total_tl = sum(len(v) for v in tracklets.values())
    print(f"Tracklets: {total_tl} across {len(tracklets)} cameras")
    
    # Configs tuned for YOLO back-projection error (mean 61cm, p90 141cm)
    # Using larger DBSCAN eps and match thresholds
    configs = [
        (0.40, 100, 100, 2, 100, "conf>=0.40 db=100 cams>=2 match=100"),
        (0.40, 100, 150, 2, 100, "conf>=0.40 db=150 cams>=2 match=100"),
        (0.40, 100, 200, 2, 100, "conf>=0.40 db=200 cams>=2 match=100"),
        (0.40, 100, 150, 1, 100, "conf>=0.40 db=150 cams>=1 match=100"),
        (0.30, 100, 150, 2, 100, "conf>=0.30 db=150 cams>=2 match=100"),
        (0.30, 100, 200, 2, 100, "conf>=0.30 db=200 cams>=2 match=100"),
        (0.30, 100, 200, 1, 100, "conf>=0.30 db=200 cams>=1 match=100"),
    ]
    
    for conf_thresh, gp_margin, dbscan_eps, min_cams, match_dist, label in configs:
        clusters = build_multiview_detections(
            tracklets, cals,
            conf_threshold=conf_thresh,
            gp_margin=gp_margin,
            dbscan_eps=dbscan_eps,
            min_cameras_per_cluster=min_cams
        )
        
        # Assign tracking IDs
        pred = assign_ids(clusters, match_threshold=150)
        
        n_frames = len(pred)
        avg_det = np.mean([len(v) for v in pred.values()]) if pred else 0
        
        summary = evaluate(gt, pred, threshold_cm=match_dist)
        
        mota = float(summary["mota"].iloc[0])
        idf1 = float(summary["idf1"].iloc[0])
        motp = float(summary["motp"].iloc[0])
        prec = float(summary["precision"].iloc[0])
        rec  = float(summary["recall"].iloc[0])
        idsw = int(summary["num_switches"].iloc[0])
        fp   = int(summary["num_false_positives"].iloc[0])
        fn   = int(summary["num_misses"].iloc[0])
        
        print(f"\n{label}")
        print(f"  {avg_det:.1f} det/frame | MOTA={mota*100:+.1f}% IDF1={idf1*100:.1f}% " +
              f"Prec={prec*100:.1f}% Rec={rec*100:.1f}% IDSW={idsw} FP={fp} FN={fn}")


if __name__ == '__main__':
    main()
