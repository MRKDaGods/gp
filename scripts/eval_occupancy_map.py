"""Ground-plane detection using occupancy map (inspired by MVDet/SHOT approach).

Instead of DBSCAN clustering (which merges nearby people), project each detection
as a Gaussian blob on the ground plane, sum across cameras, and find peaks.

This handles dense crowds where interperson distance < DBSCAN eps.
"""
import sys, json, glob, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter, gaussian_filter

sys.path.insert(0, '.')
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x, dtype=np.float64: np.asarray(x, dtype=dtype)

import motmetrics as mm
from src.core.io_utils import load_tracklets_by_camera
from src.core.wildtrack_calibration import load_wildtrack_calibration

WILDTRACK_DIR = Path('data/raw/wildtrack')
RUN_DIR = Path('data/outputs/run_20260304_050358')
GP_XMIN, GP_XMAX = -300, 900  # cm
GP_YMIN, GP_YMAX = -900, 2700
GRID_W = 480; CELL_SIZE = 2.5

# Occupancy grid: 10cm resolution
GRID_RES = 10  # cm per cell
OCC_W = (GP_XMAX - GP_XMIN) // GRID_RES + 1  # 121
OCC_H = (GP_YMAX - GP_YMIN) // GRID_RES + 1  # 361

print(f"Occupancy grid: {OCC_W} x {OCC_H} = {OCC_W*OCC_H} cells at {GRID_RES}cm resolution")

cals = load_wildtrack_calibration(str(WILDTRACK_DIR / 'calibrations'))
tracklets = load_tracklets_by_camera(str(RUN_DIR / 'stage1'))

def posid_to_ground(pos_id):
    return GP_XMIN + (pos_id % GRID_W) * CELL_SIZE, GP_YMIN + (pos_id // GRID_W) * CELL_SIZE

def pixel_to_ground(u, v, K, R, tvec):
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam
    c = -R.T @ tvec
    if abs(ray_world[2]) < 1e-10: return None
    lam = -c[2] / ray_world[2]
    if lam < 0: return None
    pt = c + lam * ray_world
    return pt[0], pt[1]

def ground_to_grid(gx, gy):
    """Convert ground coords (cm) to grid indices."""
    gi = int(round((gx - GP_XMIN) / GRID_RES))
    gj = int(round((gy - GP_YMIN) / GRID_RES))
    return gi, gj

def grid_to_ground(gi, gj):
    """Convert grid indices to ground coords (cm)."""
    return GP_XMIN + gi * GRID_RES, GP_YMIN + gj * GRID_RES


def build_occupancy_detections(tracklets_by_cam, cals, 
                                conf_threshold=0.30, 
                                sigma_cm=40, 
                                peak_threshold=2.5,
                                min_distance_cells=5):
    """Build ground-plane detections using occupancy map approach.
    
    Args:
        sigma_cm: Gaussian sigma for each detection point (accounts for back-projection error)
        peak_threshold: minimum peak value (roughly = number of supporting cameras)
        min_distance_cells: minimum distance between detected peaks
    """
    sigma_cells = sigma_cm / GRID_RES
    
    cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    
    # Pre-compute all detections per frame
    frame_gp_points = defaultdict(lambda: defaultdict(list))
    
    for cam_id in cameras:
        if cam_id not in cals: continue
        cal = cals[cam_id]; K, R, tvec = cal['K'], cal['R'], cal['tvec']
        
        for t in tracklets_by_cam.get(cam_id, []):
            for tf in t.frames:
                if tf.confidence < conf_threshold: continue
                x1, y1, x2, y2 = tf.bbox
                foot_x, foot_y = float(x1 + (x2-x1)/2), float(y2)
                gp = pixel_to_ground(foot_x, foot_y, K, R, tvec)
                if gp is None: continue
                gi, gj = ground_to_grid(gp[0], gp[1])
                if 0 <= gi < OCC_W and 0 <= gj < OCC_H:
                    frame_gp_points[tf.frame_id][cam_id].append((gi, gj))
    
    # Process each frame
    result = {}
    for frame_id in sorted(frame_gp_points.keys()):
        cam_points = frame_gp_points[frame_id]
        
        # Create per-camera occupancy maps (binary: 1 if any detection in cell)
        # Then Gaussian blur each and sum
        occ_map = np.zeros((OCC_H, OCC_W), dtype=np.float64)
        
        for cam_id, points in cam_points.items():
            cam_map = np.zeros((OCC_H, OCC_W), dtype=np.float64)
            for gi, gj in points:
                cam_map[gj, gi] = 1.0  # Note: [row, col] = [y, x]
            
            # Gaussian blur
            cam_blurred = gaussian_filter(cam_map, sigma=sigma_cells, mode='constant')
            
            # Normalize so that each camera contributes max 1.0 per blob
            if cam_blurred.max() > 0:
                cam_blurred = cam_blurred / cam_blurred.max()
            
            occ_map += cam_blurred
        
        # Find peaks
        # A peak is a local maximum above threshold
        local_max = maximum_filter(occ_map, size=min_distance_cells*2+1)
        peaks = (occ_map == local_max) & (occ_map >= peak_threshold)
        
        peak_coords = np.where(peaks)
        detections = []
        for j, i in zip(peak_coords[0], peak_coords[1]):
            gx, gy = grid_to_ground(i, j)
            score = occ_map[j, i]
            detections.append((gx, gy, score))
        
        # Sort by score (descending) and apply NMS 
        detections.sort(key=lambda x: -x[2])
        kept = []
        for det in detections:
            too_close = False
            for k in kept:
                d = np.sqrt((det[0]-k[0])**2 + (det[1]-k[1])**2)
                if d < min_distance_cells * GRID_RES:
                    too_close = True
                    break
            if not too_close:
                kept.append(det)
        
        result[frame_id] = kept
    
    return result


def assign_ids(frame_clusters, match_threshold=200):
    next_id = 1; prev = {}; result = {}
    for fid in sorted(frame_clusters.keys()):
        clusters = frame_clusters[fid]
        if not clusters: result[fid] = []; continue
        curr = [(c[0], c[1]) for c in clusters]
        if not prev:
            fr = []
            for gx, gy, *_ in clusters:
                fr.append((next_id, gx, gy)); prev[next_id] = (gx, gy); next_id += 1
            result[fid] = fr; continue
        prev_ids = list(prev.keys()); prev_pos = [prev[t] for t in prev_ids]
        cost = np.zeros((len(prev_pos), len(curr)))
        for i, (px, py) in enumerate(prev_pos):
            for j, (cx, cy) in enumerate(curr):
                cost[i,j] = np.sqrt((px-cx)**2+(py-cy)**2)
        ri, ci = linear_sum_assignment(cost)
        fr = [None]*len(clusters); matched = set(); new_prev = {}
        for r, c in zip(ri, ci):
            if cost[r,c] <= match_threshold:
                tid = prev_ids[r]; fr[c] = (tid, curr[c][0], curr[c][1])
                matched.add(c); new_prev[tid] = curr[c]
        for j in range(len(clusters)):
            if j not in matched:
                fr[j] = (next_id, curr[j][0], curr[j][1]); new_prev[next_id] = curr[j]; next_id += 1
        result[fid] = [f for f in fr if f]; prev = new_prev
    return result


def load_gt(annotations_dir):
    gt = {}
    for jf in sorted(glob.glob(str(annotations_dir / '*.json'))):
        fid = int(Path(jf).stem) // 5
        gt[fid] = [(p['personID'], *posid_to_ground(p['positionID'])) for p in json.load(open(jf))]
    return gt


def evaluate(gt_pos, pred_pos, threshold_cm=50):
    acc = mm.MOTAccumulator(auto_id=True)
    for fid in sorted(set(list(gt_pos.keys()) + list(pred_pos.keys()))):
        gl = gt_pos.get(fid, []); pl = pred_pos.get(fid, [])
        gi = [g[0] for g in gl]; pi = [p[0] for p in pl]
        if gl and pl:
            gp = np.array([[g[1],g[2]] for g in gl]); pp = np.array([[p[1],p[2]] for p in pl])
            d = np.full((len(gp),len(pp)), np.nan)
            for i in range(len(gp)):
                for j in range(len(pp)):
                    dd = np.sqrt((gp[i,0]-pp[j,0])**2+(gp[i,1]-pp[j,1])**2)
                    if dd <= threshold_cm: d[i,j] = dd
        else: d = np.empty((len(gl),len(pl)))
        acc.update(gi, pi, d)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=["mota","motp","idf1","precision","recall",
                                     "num_switches","num_false_positives","num_misses"], name="gp")


def main():
    print("=" * 70)
    print("WILDTRACK Ground-Plane Eval — Occupancy Map Approach")
    print("=" * 70)
    
    gt = load_gt(WILDTRACK_DIR / 'annotations_positions')
    print(f"GT: {len(gt)} frames, avg {np.mean([len(v) for v in gt.values()]):.1f}/frame")
    
    # (conf, sigma_cm, peak_thresh, min_dist_cells, match_dist, label)
    configs = [
        (0.30, 30, 2.0, 5, 100, "σ=30 peak>=2.0 md=50cm"),
        (0.30, 40, 2.0, 5, 100, "σ=40 peak>=2.0 md=50cm"),
        (0.30, 50, 2.0, 5, 100, "σ=50 peak>=2.0 md=50cm"),
        (0.30, 40, 1.5, 5, 100, "σ=40 peak>=1.5 md=50cm"),
        (0.30, 40, 2.5, 5, 100, "σ=40 peak>=2.5 md=50cm"),
        (0.30, 40, 3.0, 5, 100, "σ=40 peak>=3.0 md=50cm"),
        (0.30, 40, 2.0, 4, 100, "σ=40 peak>=2.0 md=40cm"),
        (0.30, 40, 2.0, 3, 100, "σ=40 peak>=2.0 md=30cm"),
        (0.20, 40, 2.0, 5, 100, "conf>=0.20 σ=40 peak>=2.0"),
        (0.40, 40, 2.0, 5, 100, "conf>=0.40 σ=40 peak>=2.0"),
    ]
    
    for conf, sigma, peak_thresh, min_dist, match_dist, label in configs:
        clusters = build_occupancy_detections(
            tracklets, cals,
            conf_threshold=conf,
            sigma_cm=sigma,
            peak_threshold=peak_thresh,
            min_distance_cells=min_dist
        )
        
        pred = assign_ids(clusters, match_threshold=200)
        avg_det = np.mean([len(v) for v in pred.values()]) if pred else 0
        
        s = evaluate(gt, pred, threshold_cm=match_dist)
        mota = float(s["mota"].iloc[0]); idf1 = float(s["idf1"].iloc[0])
        prec = float(s["precision"].iloc[0]); rec = float(s["recall"].iloc[0])
        idsw = int(s["num_switches"].iloc[0]); fp = int(s["num_false_positives"].iloc[0])
        fn = int(s["num_misses"].iloc[0])
        
        print(f"\n{label}")
        print(f"  {avg_det:.1f} det/f | MOTA={mota*100:+.1f}% IDF1={idf1*100:.1f}% "
              f"Prec={prec*100:.1f}% Rec={rec*100:.1f}% IDSW={idsw} FP={fp} FN={fn}")


if __name__ == '__main__':
    main()
