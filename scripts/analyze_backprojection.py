"""Analyze back-projection consistency: for GT people visible in multiple cameras,
how consistent are the ground-plane projections from different cameras?"""
import sys, json, glob, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from src.core.wildtrack_calibration import load_wildtrack_calibration

WILDTRACK_DIR = Path('data/raw/wildtrack')
cals = load_wildtrack_calibration(str(WILDTRACK_DIR / 'calibrations'))

cam_names = {0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5', 5: 'C6', 6: 'C7'}

GRID_W = 480
GP_XMIN, GP_YMIN = -300, -900
CELL_SIZE = 2.5

def posid_to_ground(pos_id):
    x_idx = pos_id % GRID_W
    y_idx = pos_id // GRID_W
    return GP_XMIN + x_idx * CELL_SIZE, GP_YMIN + y_idx * CELL_SIZE

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

# Analyze a few frames
errors_per_cam = {f'C{i+1}': [] for i in range(7)}
errors_all = []

for jf in sorted(glob.glob(str(WILDTRACK_DIR / 'annotations_positions' / '*.json')))[:50]:
    data = json.load(open(jf))
    for p in data:
        gt_gx, gt_gy = posid_to_ground(p['positionID'])
        
        for v in p['views']:
            if v['xmin'] < 0: continue
            cam_id = cam_names[v['viewNum']]
            cal = cals[cam_id]
            K, R, tvec = cal['K'], cal['R'], cal['tvec']
            
            # Back-project foot center to ground plane
            foot_x = (v['xmin'] + v['xmax']) / 2
            foot_y = v['ymax']  # bottom of bbox
            
            gp = pixel_to_ground(foot_x, foot_y, K, R, tvec)
            if gp is None: continue
            
            err = np.sqrt((gp[0] - gt_gx)**2 + (gp[1] - gt_gy)**2)
            errors_per_cam[cam_id].append(err)
            errors_all.append(err)

print("Back-projection error from GT bbox foot → ground plane:")
print(f"  Overall: mean={np.mean(errors_all):.1f}cm, median={np.median(errors_all):.1f}cm, "
      f"p90={np.percentile(errors_all, 90):.1f}cm, p95={np.percentile(errors_all, 95):.1f}cm")
for cam_id in sorted(errors_per_cam.keys()):
    errs = errors_per_cam[cam_id]
    if errs:
        print(f"  {cam_id}: mean={np.mean(errs):.1f}cm, median={np.median(errs):.1f}cm, "
              f"p90={np.percentile(errs, 90):.1f}cm, max={np.max(errs):.1f}cm")
