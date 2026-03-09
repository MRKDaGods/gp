"""Test different bbox projection strategies for ground-plane detection.
Instead of always using bbox bottom, try different vertical positions within the bbox."""
import sys, json, glob, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

sys.path.insert(0, '.')
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x, dtype=np.float64: np.asarray(x, dtype=dtype)

import motmetrics as mm
from src.core.io_utils import load_tracklets_by_camera
from src.core.wildtrack_calibration import load_wildtrack_calibration

WILDTRACK_DIR = Path('data/raw/wildtrack')
RUN_DIR = Path('data/outputs/run_20260304_050358')
GP_XMIN, GP_XMAX = -300, 900; GP_YMIN, GP_YMAX = -900, 2700
GRID_W = 480; CELL_SIZE = 2.5

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

def pixel_to_height_plane(u, v, K, R, tvec, height_cm):
    """Project pixel to a plane at Z=height_cm instead of Z=0."""
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam
    c = -R.T @ tvec
    if abs(ray_world[2]) < 1e-10: return None
    lam = (height_cm - c[2]) / ray_world[2]
    if lam < 0: return None
    pt = c + lam * ray_world
    return pt[0], pt[1]

def load_gt(annotations_dir):
    gt = {}
    for jf in sorted(glob.glob(str(annotations_dir / '*.json'))):
        fid = int(Path(jf).stem) // 5
        gt[fid] = [(p['personID'], *posid_to_ground(p['positionID'])) for p in json.load(open(jf))]
    return gt

def evaluate(gt_pos, pred_pos, threshold_cm):
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
    return mh.compute(acc, metrics=["mota","motp","idf1","precision","recall","num_switches","num_false_positives","num_misses"], name="gp")

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

gt = load_gt(WILDTRACK_DIR / 'annotations_positions')
print(f"GT: {len(gt)} frames, avg {np.mean([len(v) for v in gt.values()]):.1f}/frame")

# Strategy 1: Different vertical offset within bbox for foot projection
# foot_ratio: 0.0 = top, 1.0 = bottom (standard), 0.85 = 15% up from bottom
strategies = [
    ("foot (bottom)", "foot", 1.0, 0),
    ("85% height", "foot", 0.85, 0),
    ("90% height", "foot", 0.90, 0),
    ("95% height", "foot", 0.95, 0),
    ("center→Z=85cm", "center_height", 0.5, 85),
    ("center→Z=100cm", "center_height", 0.5, 100),
]

for label, method, v_ratio, height_cm in strategies:
    frame_dets = defaultdict(list)
    for cam_id in ['C1','C2','C3','C4','C5','C6','C7']:
        cal = cals[cam_id]; K, R, tvec = cal['K'], cal['R'], cal['tvec']
        for t in tracklets.get(cam_id, []):
            for tf in t.frames:
                if tf.confidence < 0.30: continue
                x1, y1, x2, y2 = tf.bbox
                
                if method == "foot":
                    px = float(x1 + (x2-x1)/2)
                    py = float(y1 + (y2-y1) * v_ratio)
                    gp = pixel_to_ground(px, py, K, R, tvec)
                elif method == "center_height":
                    px = float(x1 + (x2-x1)/2)
                    py = float(y1 + (y2-y1) * v_ratio)
                    gp = pixel_to_height_plane(px, py, K, R, tvec, height_cm)
                
                if gp is None: continue
                gx, gy = gp
                if not ((GP_XMIN - 100) <= gx <= (GP_XMAX + 100) and
                        (GP_YMIN - 100) <= gy <= (GP_YMAX + 100)): continue
                frame_dets[tf.frame_id].append((cam_id, gx, gy, tf.confidence))
    
    # DBSCAN cluster
    clusters_by_frame = {}
    for fid, dets in frame_dets.items():
        pos = np.array([[d[1],d[2]] for d in dets])
        cams_arr = [d[0] for d in dets]
        cl = DBSCAN(eps=75, min_samples=1).fit(pos)
        
        frame_cl = []
        for lab in set(cl.labels_):
            mask = cl.labels_ == lab
            c_pos = pos[mask]
            c_cams = set(cams_arr[i] for i in range(len(cams_arr)) if mask[i])
            if len(c_cams) >= 2:
                centroid = c_pos.mean(axis=0)
                frame_cl.append((centroid[0], centroid[1]))
        clusters_by_frame[fid] = frame_cl
    
    pred = assign_ids(clusters_by_frame)
    avg_det = np.mean([len(v) for v in pred.values()]) if pred else 0
    
    s = evaluate(gt, pred, threshold_cm=100)
    mota = float(s["mota"].iloc[0]); idf1 = float(s["idf1"].iloc[0])
    prec = float(s["precision"].iloc[0]); rec = float(s["recall"].iloc[0])
    idsw = int(s["num_switches"].iloc[0])
    
    print(f"\n{label}: {avg_det:.1f} det/f | MOTA={mota*100:+.1f}% IDF1={idf1*100:.1f}% "
          f"Prec={prec*100:.1f}% Rec={rec*100:.1f}% IDSW={idsw}")
