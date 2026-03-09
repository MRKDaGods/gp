"""Test combined filters: ground-plane + confidence threshold + min track length.
Uses EXISTING tracklet data (no re-run of Stage 1 needed).
"""
import sys, cv2, numpy as np
sys.path.insert(0, '.')

from src.core.wildtrack_calibration import load_wildtrack_calibration
from src.core.io_utils import load_tracklets_by_camera

# Fix numpy 2.0 compat
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x, dtype=np.float64: np.asarray(x, dtype=dtype)
import motmetrics as mm

GP_XMIN, GP_XMAX = -300, 900
GP_YMIN, GP_YMAX = -900, 2700
GP_MARGIN = 100

cals = load_wildtrack_calibration('data/raw/wildtrack/calibrations')
tracklets = load_tracklets_by_camera('data/outputs/run_20260304_050358/stage1')

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

def is_on_gp(u, v, K, R, tvec):
    result = pixel_to_ground(u, v, K, R, tvec)
    if result is None: return False
    gx, gy = result
    return (GP_XMIN - GP_MARGIN) <= gx <= (GP_XMAX + GP_MARGIN) and \
           (GP_YMIN - GP_MARGIN) <= gy <= (GP_YMAX + GP_MARGIN)

cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

# Try different confidence + track length combos
for conf_thresh, min_len in [(0.0, 3), (0.30, 3), (0.30, 5), (0.35, 5), (0.40, 5), (0.30, 8)]:
    results = []
    for cam_id in cameras:
        gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
        cal = cals[cam_id]; K, R, tvec = cal['K'], cal['R'], cal['tvec']
        cam_t = tracklets.get(cam_id, [])
        
        # Filter tracklets by length
        cam_t_filtered = [t for t in cam_t if len(t.frames) >= min_len]
        
        # Build filtered predictions
        pred_data = {}
        for t in cam_t_filtered:
            # Compute average confidence for tracklet
            avg_conf = np.mean([tf.confidence for tf in t.frames if tf.confidence > 0])
            if avg_conf < conf_thresh:
                continue
            
            for tf in t.frames:
                if tf.confidence < conf_thresh:
                    continue
                x1, y1, x2, y2 = tf.bbox
                w, h = x2-x1, y2-y1
                foot_x = float(x1 + w/2)
                foot_y = float(y2)
                if is_on_gp(foot_x, foot_y, K, R, tvec):
                    pred_data.setdefault(tf.frame_id, []).append((t.track_id, [x1, y1, w, h]))
        
        # Load GT
        gt_data = {}
        for line in open(gt_path):
            parts = line.strip().split(',')
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt_data.setdefault(fid, []).append((tid, [x, y, w, h]))
        
        # Evaluate
        acc = mm.MOTAccumulator(auto_id=True)
        frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
        for fid in frames:
            gt_ids = [x[0] for x in gt_data.get(fid, [])]
            gt_boxes = [x[1] for x in gt_data.get(fid, [])]
            p_ids = [x[0] for x in pred_data.get(fid, [])]
            p_boxes = [x[1] for x in pred_data.get(fid, [])]
            dist = mm.distances.iou_matrix(gt_boxes, p_boxes, max_iou=0.5) if gt_boxes and p_boxes else []
            acc.update(gt_ids, p_ids, dist)
        
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"], name=cam_id)
        mota = float(summary["mota"].iloc[0])
        idf1 = float(summary["idf1"].iloc[0])
        idsw = int(summary["num_switches"].iloc[0])
        fp = int(summary["num_false_positives"].iloc[0])
        fn = int(summary["num_misses"].iloc[0])
        results.append((cam_id, mota, idf1, idsw, fp, fn))
    
    avg_mota = np.mean([r[1] for r in results])
    avg_idf1 = np.mean([r[2] for r in results])
    total_idsw = sum(r[3] for r in results)
    total_fp = sum(r[4] for r in results)
    total_fn = sum(r[5] for r in results)
    print(f'conf>={conf_thresh:.2f} minlen>={min_len}: '
          f'MOTA={avg_mota:.3f}, IDF1={avg_idf1:.3f}, IDSW={total_idsw}, FP={total_fp}, FN={total_fn}')
    for r in results:
        print(f'  {r[0]}: MOTA={r[1]:.3f}, IDF1={r[2]:.3f}, IDSW={r[3]}, FP={r[4]}, FN={r[5]}')
    print()
