"""Test proper ground-plane back-projection filtering for WILDTRACK.

Projects each detection's foot point back to the WILDTRACK ground plane (Z=0)
using camera calibration, then keeps only detections whose projected foot point
falls within the annotated ground-plane rectangle.
"""
import sys, cv2, numpy as np
sys.path.insert(0, '.')

from src.core.wildtrack_calibration import load_wildtrack_calibration

# WILDTRACK annotated ground plane (centimetres)
GP_XMIN, GP_XMAX = -300, 900    # 12m
GP_YMIN, GP_YMAX = -900, 2700   # 36m

cals = load_wildtrack_calibration('data/raw/wildtrack/calibrations')

def pixel_to_ground(u, v, K, R, tvec):
    """Back-project image point (u,v) to z=0 ground plane. Returns (X,Y) in cm."""
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam
    cam_center = -R.T @ tvec
    # Intersect with z=0: cam_center.z + lam * ray_world.z = 0
    if abs(ray_world[2]) < 1e-10:
        return None  # Ray parallel to ground
    lam = -cam_center[2] / ray_world[2]
    if lam < 0:
        return None  # Behind camera
    pt = cam_center + lam * ray_world
    return pt[0], pt[1]

def is_on_ground_plane(u, v, K, R, tvec, margin=100):
    """Check if pixel (u,v) projects onto the annotated ground plane area."""
    result = pixel_to_ground(u, v, K, R, tvec)
    if result is None:
        return False
    gx, gy = result
    return (GP_XMIN - margin) <= gx <= (GP_XMAX + margin) and \
           (GP_YMIN - margin) <= gy <= (GP_YMAX + margin)

# Test on predictions
cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

for margin in [0, 50, 100, 200]:
    print(f'\n--- Ground-plane filter with margin={margin}cm ---')
    for cam_id in cameras:
        pred_path = f'data/outputs/run_20260304_050358/stage5/predictions_mot/{cam_id}.txt'
        gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
        cal = cals[cam_id]
        K, R, tvec = cal['K'], cal['R'], cal['tvec']
        
        total = 0
        inside = 0
        for line in open(pred_path):
            parts = line.strip().split(',')
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            foot_x = x + w / 2
            foot_y = y + h
            total += 1
            if is_on_ground_plane(foot_x, foot_y, K, R, tvec, margin=margin):
                inside += 1
        
        gt_count = sum(1 for _ in open(gt_path))
        gt_per_f = gt_count / 400
        total_per_f = total / 401
        inside_per_f = inside / 401
        
        print(f'  {cam_id}: GT={gt_per_f:.1f}/f, Pred={total_per_f:.1f}/f -> Filtered={inside_per_f:.1f}/f '
              f'(removed {100*(total-inside)/max(total,1):.0f}%)')

# Full evaluation with best margin
print('\n\n=== Full evaluation with ground-plane filter (margin=100cm) ===')
import motmetrics as mm
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x, dtype=np.float64: np.asarray(x, dtype=dtype)

from src.core.io_utils import load_tracklets_by_camera
tracklets = load_tracklets_by_camera('data/outputs/run_20260304_050358/stage1')

results = []
for cam_id in cameras:
    gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
    cal = cals[cam_id]
    K, R, tvec = cal['K'], cal['R'], cal['tvec']
    cam_t = tracklets.get(cam_id, [])
    
    # Build ground-plane-filtered predictions
    pred_data = {}
    for t in cam_t:
        for tf in t.frames:
            x1, y1, x2, y2 = tf.bbox
            w, h = x2-x1, y2-y1
            foot_x = float(x1 + w/2)
            foot_y = float(y2)
            if is_on_ground_plane(foot_x, foot_y, K, R, tvec, margin=100):
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
    summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "precision", "recall"], name=cam_id)
    mota = float(summary["mota"].iloc[0])
    idf1 = float(summary["idf1"].iloc[0])
    idsw = int(summary["num_switches"].iloc[0])
    fp = int(summary["num_false_positives"].iloc[0])
    fn = int(summary["num_misses"].iloc[0])
    prec = float(summary["precision"].iloc[0])
    rec = float(summary["recall"].iloc[0])
    results.append((cam_id, mota, idf1, idsw, fp, fn, prec, rec))
    print(f'  {cam_id}: MOTA={mota:.3f}, IDF1={idf1:.3f}, IDSW={idsw}, FP={fp}, FN={fn}, Prec={prec:.3f}, Rec={rec:.3f}')

avg_mota = np.mean([r[1] for r in results])
avg_idf1 = np.mean([r[2] for r in results])
total_idsw = sum(r[3] for r in results)
print(f'\n  AVERAGE: MOTA={avg_mota:.3f}, IDF1={avg_idf1:.3f}, IDSW_total={total_idsw}')
