"""Camera-count based occupancy map: no Gaussian, just count cameras per cell with dilation."""
import sys, json, glob, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter, uniform_filter

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
    return mh.compute(acc, metrics=["mota","motp","idf1","precision","recall",
                                     "num_switches","num_false_positives","num_misses"], name="gp")

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


def detect_cam_count(tracklets_by_cam, cals, conf_threshold, grid_res, dilation_cm, peak_threshold, min_dist_cm):
    """Per-camera max-dilated occupancy, summed across cameras, then peak finding.
    
    Each camera contributes a binary map (1 if detection within dilation_cm), 
    dilated by dilation_cm. Sum across cameras → camera count map.
    """
    occ_w = int((GP_XMAX - GP_XMIN) / grid_res) + 1
    occ_h = int((GP_YMAX - GP_YMIN) / grid_res) + 1
    dilation_cells = max(1, int(dilation_cm / grid_res))
    min_dist_cells = max(1, int(min_dist_cm / grid_res))

    # Pre-compute ground positions
    frame_cam_points = defaultdict(lambda: defaultdict(list))
    for cam_id in ['C1','C2','C3','C4','C5','C6','C7']:
        cal = cals[cam_id]; K, R, tvec = cal['K'], cal['R'], cal['tvec']
        for t in tracklets_by_cam.get(cam_id, []):
            for tf in t.frames:
                if tf.confidence < conf_threshold: continue
                x1, y1, x2, y2 = tf.bbox
                gp = pixel_to_ground(float(x1+(x2-x1)/2), float(y2), K, R, tvec)
                if gp is None: continue
                gi = int(round((gp[0]-GP_XMIN)/grid_res))
                gj = int(round((gp[1]-GP_YMIN)/grid_res))
                if 0 <= gi < occ_w and 0 <= gj < occ_h:
                    frame_cam_points[tf.frame_id][cam_id].append((gi, gj))

    result = {}
    for fid in sorted(frame_cam_points.keys()):
        # Per-camera dilated binary map
        cam_count = np.zeros((occ_h, occ_w), dtype=np.float32)
        
        for cam_id in ['C1','C2','C3','C4','C5','C6','C7']:
            pts = frame_cam_points[fid].get(cam_id, [])
            if not pts: continue
            
            cam_map = np.zeros((occ_h, occ_w), dtype=np.float32)
            for gi, gj in pts:
                cam_map[gj, gi] = 1.0
            
            # Dilate: max_filter makes each point spread to dilation_cells radius
            dilated = maximum_filter(cam_map, size=dilation_cells*2+1)
            
            cam_count += dilated
        
        # Find peaks: positions where camera count is locally maximal and >= threshold
        # Use a larger neighborhood for peak finding to separate nearby people
        local_max = maximum_filter(cam_count, size=min_dist_cells*2+1)
        peaks = (cam_count == local_max) & (cam_count >= peak_threshold)
        
        pj, pi = np.where(peaks)
        dets = [(GP_XMIN + i*grid_res, GP_YMIN + j*grid_res, cam_count[j,i]) 
                for j, i in zip(pj, pi)]
        dets.sort(key=lambda x: -x[2])
        
        # NMS
        kept = []
        for det in dets:
            if not any(np.sqrt((det[0]-k[0])**2+(det[1]-k[1])**2) < min_dist_cm for k in kept):
                kept.append(det)
        result[fid] = kept
    return result


gt = load_gt(WILDTRACK_DIR / 'annotations_positions')
print(f"GT: avg {np.mean([len(v) for v in gt.values()]):.1f}/frame\n")

# Params: conf, grid_res, dilation_cm, peak_threshold, min_dist_cm, match_cm
configs = [
    (0.20, 10, 50, 2, 40, 100, "d=50 pk>=2 md=40 m=100"),
    (0.20, 10, 50, 3, 40, 100, "d=50 pk>=3 md=40 m=100"),
    (0.20, 10, 75, 2, 40, 100, "d=75 pk>=2 md=40 m=100"),
    (0.20, 10, 75, 3, 40, 100, "d=75 pk>=3 md=40 m=100"),
    (0.20, 10, 75, 4, 40, 100, "d=75 pk>=4 md=40 m=100"),
    (0.20, 10, 100, 3, 40, 100, "d=100 pk>=3 md=40 m=100"),
    (0.20, 10, 100, 4, 40, 100, "d=100 pk>=4 md=40 m=100"),
    (0.20, 10, 100, 5, 40, 100, "d=100 pk>=5 md=40 m=100"),
    (0.30, 10, 75, 3, 40, 100, "conf>=0.30 d=75 pk>=3 md=40"),
    (0.40, 10, 75, 3, 40, 100, "conf>=0.40 d=75 pk>=3 md=40"),
    # Also try match=50 for comparison
    (0.20, 10, 75, 3, 40, 50,  "d=75 pk>=3 md=40 m=50"),
]

for conf, res, dil, pk, md, match, label in configs:
    cl = detect_cam_count(tracklets, cals, conf, res, dil, pk, md)
    pred = assign_ids(cl, match_threshold=200)
    avg_det = np.mean([len(v) for v in pred.values()]) if pred else 0
    s = evaluate(gt, pred, threshold_cm=match)
    mota = float(s["mota"].iloc[0]); idf1 = float(s["idf1"].iloc[0])
    prec = float(s["precision"].iloc[0]); rec = float(s["recall"].iloc[0])
    idsw = int(s["num_switches"].iloc[0]); fp = int(s["num_false_positives"].iloc[0])
    fn = int(s["num_misses"].iloc[0])
    print(f"{label}: {avg_det:.1f}det | MOTA={mota*100:+.1f}% IDF1={idf1*100:.1f}% "
          f"P={prec*100:.0f}% R={rec*100:.0f}% SW={idsw} FP={fp} FN={fn}")
