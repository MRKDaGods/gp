"""Compare YOLO detection foot projections vs GT ground positions.
For each GT person, find the best matching YOLO detection and measure ground-plane error."""
import sys, json, glob, numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, '.')
from src.core.wildtrack_calibration import load_wildtrack_calibration
from src.core.io_utils import load_tracklets_by_camera

WILDTRACK_DIR = Path('data/raw/wildtrack')
RUN_DIR = Path('data/outputs/run_20260304_050358')
cals = load_wildtrack_calibration(str(WILDTRACK_DIR / 'calibrations'))
tracklets = load_tracklets_by_camera(str(RUN_DIR / 'stage1'))

cam_names = {0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5', 5: 'C6', 6: 'C7'}
GRID_W = 480; GP_XMIN, GP_YMIN = -300, -900; CELL_SIZE = 2.5

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

# Build YOLO detection lookup: {cam_id: {frame_id: [(gx, gy, conf), ...]}}
yolo_gp = defaultdict(lambda: defaultdict(list))
for cam_id in ['C1','C2','C3','C4','C5','C6','C7']:
    cal = cals[cam_id]; K, R, tvec = cal['K'], cal['R'], cal['tvec']
    for t in tracklets.get(cam_id, []):
        for tf in t.frames:
            if tf.confidence < 0.40: continue
            x1, y1, x2, y2 = tf.bbox
            foot_x, foot_y = float(x1 + (x2-x1)/2), float(y2)
            gp = pixel_to_ground(foot_x, foot_y, K, R, tvec)
            if gp: yolo_gp[cam_id][tf.frame_id].append((gp[0], gp[1], tf.confidence))

# For first 50 frames, compare YOLO ground positions with GT
errors_per_cam = defaultdict(list)
gt_detected = 0
gt_missed = 0

for jf in sorted(glob.glob(str(WILDTRACK_DIR / 'annotations_positions' / '*.json')))[:50]:
    fid = int(Path(jf).stem) // 5
    data = json.load(open(jf))
    
    for p in data:
        gt_gx, gt_gy = posid_to_ground(p['positionID'])
        
        for v in p['views']:
            if v['xmin'] < 0: continue
            cam_id = cam_names[v['viewNum']]
            
            # Find best matching YOLO detection in this camera at this frame
            yolo_dets = yolo_gp[cam_id].get(fid, [])
            if not yolo_dets:
                gt_missed += 1
                continue
            
            # Find closest YOLO ground position to GT
            min_err = float('inf')
            for ygx, ygy, _ in yolo_dets:
                err = np.sqrt((ygx - gt_gx)**2 + (ygy - gt_gy)**2)
                min_err = min(min_err, err)
            
            if min_err < 200:  # sanity: only count if within 2m
                errors_per_cam[cam_id].append(min_err)
                gt_detected += 1
            else:
                gt_missed += 1

errors_all = [e for errs in errors_per_cam.values() for e in errs]
print("YOLO detection back-projection error vs GT ground position:")
print(f"  Overall: mean={np.mean(errors_all):.1f}cm, median={np.median(errors_all):.1f}cm, "
      f"p90={np.percentile(errors_all, 90):.1f}cm, p95={np.percentile(errors_all, 95):.1f}cm")
print(f"  Detected (<=2m match): {gt_detected}, Missed: {gt_missed}")
for cam_id in sorted(errors_per_cam.keys()):
    errs = errors_per_cam[cam_id]
    if errs:
        print(f"  {cam_id}: mean={np.mean(errs):.1f}cm, median={np.median(errs):.1f}cm, "
              f"p90={np.percentile(errs, 90):.1f}cm, max={np.max(errs):.1f}cm, n={len(errs)}")

# Now: for each GT person in a frame, how many cameras have a matching YOLO detection within 100cm?
print("\n\nGT person detection coverage (YOLO within 100cm of GT ground pos):")
cam_coverage = []
for jf in sorted(glob.glob(str(WILDTRACK_DIR / 'annotations_positions' / '*.json')))[:50]:
    fid = int(Path(jf).stem) // 5
    data = json.load(open(jf))
    
    for p in data:
        gt_gx, gt_gy = posid_to_ground(p['positionID'])
        detected_cams = 0
        visible_cams = 0
        
        for v in p['views']:
            if v['xmin'] < 0: continue
            visible_cams += 1
            cam_id = cam_names[v['viewNum']]
            
            yolo_dets = yolo_gp[cam_id].get(fid, [])
            for ygx, ygy, _ in yolo_dets:
                err = np.sqrt((ygx - gt_gx)**2 + (ygy - gt_gy)**2)
                if err < 100:
                    detected_cams += 1
                    break
        
        cam_coverage.append((detected_cams, visible_cams))

from collections import Counter
detected_counts = Counter(c[0] for c in cam_coverage)
print(f"  Detected in N cameras:")
for k in sorted(detected_counts.keys()):
    print(f"    {k} cameras: {detected_counts[k]} ({detected_counts[k]/len(cam_coverage)*100:.1f}%)")
print(f"  >= 2 cameras: {sum(1 for d, v in cam_coverage if d >= 2)/len(cam_coverage)*100:.1f}%")
print(f"  >= 1 camera:  {sum(1 for d, v in cam_coverage if d >= 1)/len(cam_coverage)*100:.1f}%")
print(f"  Mean visible: {np.mean([v for d, v in cam_coverage]):.1f}, Mean detected: {np.mean([d for d, v in cam_coverage]):.1f}")
