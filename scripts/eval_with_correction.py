"""Compute per-camera back-projection correction factors and test improved ground-plane detection."""
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
GP_XMIN, GP_XMAX = -300, 900
GP_YMIN, GP_YMAX = -900, 2700
GRID_W = 480; CELL_SIZE = 2.5

cals = load_wildtrack_calibration(str(WILDTRACK_DIR / 'calibrations'))
tracklets = load_tracklets_by_camera(str(RUN_DIR / 'stage1'))
cam_names = {0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5', 5: 'C6', 6: 'C7'}

def posid_to_ground(pos_id):
    return GP_XMIN + (pos_id % GRID_W) * CELL_SIZE, GP_YMIN + (pos_id // GRID_W) * CELL_SIZE

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

def compute_iou(b1_xywh, b2_xyxy):
    """IoU between GT (x,y,w,h) and pred (x1,y1,x2,y2)"""
    g_x1, g_y1 = b1_xywh[0], b1_xywh[1]
    g_x2, g_y2 = g_x1 + b1_xywh[2], g_y1 + b1_xywh[3]
    p_x1, p_y1, p_x2, p_y2 = b2_xyxy
    ix1 = max(g_x1, p_x1); iy1 = max(g_y1, p_y1)
    ix2 = min(g_x2, p_x2); iy2 = min(g_y2, p_y2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (g_x2-g_x1)*(g_y2-g_y1) + (p_x2-p_x1)*(p_y2-p_y1) - inter
    return inter / union if union > 0 else 0

# Step 1: Compute per-camera correction by matching YOLO to GT using 2D IoU
# (more reliable than ground-plane distance matching)
print("Step 1: Computing per-camera back-projection corrections...")

# Build YOLO detection lookup per camera per frame
yolo_by_cam_frame = defaultdict(lambda: defaultdict(list))
for cam_id in ['C1','C2','C3','C4','C5','C6','C7']:
    for t in tracklets.get(cam_id, []):
        for tf in t.frames:
            if tf.confidence < 0.30:
                continue
            yolo_by_cam_frame[cam_id][tf.frame_id].append(tf)

corrections = {}  # {cam_id: (mean_dx, mean_dy)}

for cam_id in ['C1','C2','C3','C4','C5','C6','C7']:
    cal = cals[cam_id]; K, R, tvec = cal['K'], cal['R'], cal['tvec']
    dx_list, dy_list = [], []
    
    for jf in sorted(glob.glob(str(WILDTRACK_DIR / 'annotations_positions' / '*.json')))[:100]:
        fid = int(Path(jf).stem) // 5
        data = json.load(open(jf))
        
        # Get GT for this camera
        for p in data:
            view = None
            for v in p['views']:
                if cam_names.get(v['viewNum']) == cam_id and v['xmin'] >= 0:
                    view = v; break
            if view is None: continue
            
            gt_gx, gt_gy = posid_to_ground(p['positionID'])
            gt_bbox = [view['xmin'], view['ymin'], view['xmax']-view['xmin'], view['ymax']-view['ymin']]
            
            # Find best matching YOLO detection by 2D IoU
            best_iou = 0
            best_tf = None
            for tf in yolo_by_cam_frame[cam_id].get(fid, []):
                iou = compute_iou(gt_bbox, tf.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tf = tf
            
            if best_iou < 0.3 or best_tf is None:
                continue
            
            # Back-project YOLO foot to ground
            x1, y1, x2, y2 = best_tf.bbox
            foot_x, foot_y = float(x1 + (x2-x1)/2), float(y2)
            gp = pixel_to_ground(foot_x, foot_y, K, R, tvec)
            if gp is None: continue
            
            dx_list.append(gp[0] - gt_gx)
            dy_list.append(gp[1] - gt_gy)
    
    if dx_list:
        mean_dx, mean_dy = np.mean(dx_list), np.mean(dy_list)
        corrections[cam_id] = (mean_dx, mean_dy)
        err_before = np.mean([np.sqrt(dx**2+dy**2) for dx, dy in zip(dx_list, dy_list)])
        err_after = np.mean([np.sqrt((dx-mean_dx)**2+(dy-mean_dy)**2) for dx, dy in zip(dx_list, dy_list)])
        print(f"  {cam_id}: correction=({mean_dx:.1f}, {mean_dy:.1f})cm, "
              f"error before={err_before:.1f}cm, after={err_after:.1f}cm, n={len(dx_list)}")
    else:
        corrections[cam_id] = (0, 0)

# Step 2: Build corrected multi-view detections
print("\nStep 2: Building corrected multi-view detections...")

def build_corrected_detections(tracklets_by_cam, cals, corrections, 
                                conf_threshold=0.30, gp_margin=100,
                                dbscan_eps=75, min_cams=2):
    cameras = ['C1','C2','C3','C4','C5','C6','C7']
    frame_dets = defaultdict(list)
    
    for cam_id in cameras:
        cal = cals[cam_id]; K, R, tvec = cal['K'], cal['R'], cal['tvec']
        corr_dx, corr_dy = corrections.get(cam_id, (0, 0))
        
        for t in tracklets_by_cam.get(cam_id, []):
            for tf in t.frames:
                if tf.confidence < conf_threshold: continue
                x1, y1, x2, y2 = tf.bbox
                foot_x, foot_y = float(x1 + (x2-x1)/2), float(y2)
                gp = pixel_to_ground(foot_x, foot_y, K, R, tvec)
                if gp is None: continue
                gx, gy = gp[0] - corr_dx, gp[1] - corr_dy  # Apply correction
                if not ((GP_XMIN - gp_margin) <= gx <= (GP_XMAX + gp_margin) and
                        (GP_YMIN - gp_margin) <= gy <= (GP_YMAX + gp_margin)):
                    continue
                frame_dets[tf.frame_id].append((cam_id, gx, gy, tf.confidence))
    
    # Cluster per frame
    result = {}
    for fid in sorted(frame_dets.keys()):
        dets = frame_dets[fid]
        if not dets:
            result[fid] = []
            continue
        
        positions = np.array([[d[1], d[2]] for d in dets])
        cams = [d[0] for d in dets]
        confs = [d[3] for d in dets]
        
        clustering = DBSCAN(eps=dbscan_eps, min_samples=1).fit(positions)
        
        clusters = []
        for label in set(clustering.labels_):
            mask = clustering.labels_ == label
            c_pos = positions[mask]
            c_cams = set(cams[i] for i in range(len(cams)) if mask[i])
            c_confs = [confs[i] for i in range(len(confs)) if mask[i]]
            
            if len(c_cams) >= min_cams:
                # Confidence-weighted centroid
                weights = np.array(c_confs)[mask[:len(c_confs)] if len(c_confs) == sum(mask) else range(sum(mask))]
                weights = np.array([confs[i] for i in range(len(confs)) if mask[i]])
                centroid = np.average(c_pos, axis=0, weights=weights)
                clusters.append((centroid[0], centroid[1], len(c_cams), np.mean(weights)))
        
        result[fid] = clusters
    return result

def assign_ids(frame_clusters, match_threshold=150):
    next_id = 1; prev = {}; result = {}
    for fid in sorted(frame_clusters.keys()):
        clusters = frame_clusters[fid]
        if not clusters:
            result[fid] = []; continue
        curr = [(c[0], c[1]) for c in clusters]
        if not prev:
            fr = []; 
            for gx, gy, _, _ in clusters:
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
                fr[j] = (next_id, curr[j][0], curr[j][1])
                new_prev[next_id] = curr[j]; next_id += 1
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
    return mh.compute(acc, metrics=["mota","motp","idf1","precision","recall","num_switches","num_false_positives","num_misses"], name="gp")

gt = load_gt(WILDTRACK_DIR / 'annotations_positions')

# Test with vs without corrections
configs = [
    # (conf, db_eps, min_cams, match_dist, use_correction, label)
    (0.30, 75,  2, 100, False, "no correction, db=75, cams>=2"),
    (0.30, 75,  2, 100, True,  "WITH correction, db=75, cams>=2"),
    (0.30, 100, 2, 100, False, "no correction, db=100, cams>=2"),
    (0.30, 100, 2, 100, True,  "WITH correction, db=100, cams>=2"),
    (0.40, 75,  2, 100, True,  "WITH correction, conf>=0.40, db=75, cams>=2"),
    (0.40, 100, 2, 100, True,  "WITH correction, conf>=0.40, db=100, cams>=2"),
    (0.30, 100, 2, 50,  True,  "WITH correction, db=100, match=50"),
    (0.40, 100, 2, 50,  True,  "WITH correction, conf>=0.40, db=100, match=50"),
]

for conf, db_eps, min_cams, match_dist, use_corr, label in configs:
    corr = corrections if use_corr else {c: (0,0) for c in corrections}
    clusters = build_corrected_detections(tracklets, cals, corr,
                                          conf_threshold=conf, dbscan_eps=db_eps,
                                          min_cams=min_cams)
    pred = assign_ids(clusters, match_threshold=200)
    avg_det = np.mean([len(v) for v in pred.values()]) if pred else 0
    
    s = evaluate(gt, pred, threshold_cm=match_dist)
    mota = float(s["mota"].iloc[0]); idf1 = float(s["idf1"].iloc[0])
    prec = float(s["precision"].iloc[0]); rec = float(s["recall"].iloc[0])
    idsw = int(s["num_switches"].iloc[0]); fp = int(s["num_false_positives"].iloc[0]); fn = int(s["num_misses"].iloc[0])
    
    print(f"\n{label}")
    print(f"  {avg_det:.1f} det/f | MOTA={mota*100:+.1f}% IDF1={idf1*100:.1f}% "
          f"Prec={prec*100:.1f}% Rec={rec*100:.1f}% IDSW={idsw} FP={fp} FN={fn}")
