"""Compute per-camera ROI polygons from GT foot positions and test filtering impact."""
import sys, json
import cv2
import numpy as np
sys.path.insert(0, '.')

cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
rois = {}

for cam_id in cameras:
    gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
    
    # Collect foot positions (bottom-center of GT boxes)
    feet = []
    for line in open(gt_path):
        parts = line.strip().split(',')
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        foot_x = x + w / 2
        foot_y = y + h
        feet.append([foot_x, foot_y])
    
    feet = np.array(feet, dtype=np.float32)
    
    # Convex hull of foot positions
    hull = cv2.convexHull(feet)
    # Expand hull by a margin (dilate the polygon by ~30 pixels)
    M = cv2.moments(hull)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    # Scale polygon outward from centroid by 5%
    expanded = hull.copy().reshape(-1, 2)
    for i in range(len(expanded)):
        dx = expanded[i][0] - cx
        dy = expanded[i][1] - cy
        expanded[i][0] = cx + dx * 1.10
        expanded[i][1] = cy + dy * 1.10
    # Clip to frame
    expanded[:, 0] = np.clip(expanded[:, 0], 0, 1920)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, 1080)
    
    rois[cam_id] = expanded
    
    area = cv2.contourArea(expanded.reshape(-1, 1, 2).astype(np.float32))
    frame_area = 1920 * 1080
    print(f'{cam_id}: {len(expanded)} hull vertices, area={area:.0f} ({100*area/frame_area:.1f}% of frame)')
    print(f'  foot_x=[{feet[:,0].min():.0f}, {feet[:,0].max():.0f}]  '
          f'foot_y=[{feet[:,1].min():.0f}, {feet[:,1].max():.0f}]')

# Test filtering impact
print('\n--- Impact of ROI filtering on predictions ---')
for cam_id in cameras:
    pred_path = f'data/outputs/run_20260304_050358/stage5/predictions_mot/{cam_id}.txt'
    gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
    
    roi_poly = rois[cam_id].astype(np.float32).reshape(-1, 1, 2)
    
    total = 0
    inside = 0
    for line in open(pred_path):
        parts = line.strip().split(',')
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        foot_x = float(x + w / 2)
        foot_y = float(y + h)
        total += 1
        if cv2.pointPolygonTest(roi_poly, (foot_x, foot_y), False) >= 0:
            inside += 1
    
    gt_count = sum(1 for _ in open(gt_path))
    gt_per_f = gt_count / 400
    total_per_f = total / 401
    inside_per_f = inside / 401
    
    print(f'{cam_id}: Pred={total_per_f:.1f}/f -> Filtered={inside_per_f:.1f}/f (GT={gt_per_f:.1f}/f) '
          f'removed {total-inside}/{total} ({100*(total-inside)/max(total,1):.0f}%)')

# Evaluate with ROI-filtered predictions
print('\n--- MOTA with ROI-filtered predictions ---')
import motmetrics as mm
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x, dtype=np.float64: np.asarray(x, dtype=dtype)

from src.core.io_utils import load_tracklets_by_camera
tracklets = load_tracklets_by_camera('data/outputs/run_20260304_050358/stage1')

results = []
for cam_id in cameras:
    gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
    roi_poly = rois[cam_id].astype(np.float32).reshape(-1, 1, 2)
    cam_t = tracklets.get(cam_id, [])
    
    # Build ROI-filtered predictions
    pred_data = {}
    for t in cam_t:
        for tf in t.frames:
            x1, y1, x2, y2 = tf.bbox
            w, h = x2-x1, y2-y1
            foot_x = float(x1 + w/2)
            foot_y = float(y2)  # bottom of bbox
            if cv2.pointPolygonTest(roi_poly, (foot_x, foot_y), False) >= 0:
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
print(f'\n  AVERAGE: MOTA={avg_mota:.3f}, IDF1={avg_idf1:.3f}')

# Save ROI polygons
roi_data = {k: v.tolist() for k, v in rois.items()}
with open('data/raw/wildtrack/manifests/roi_polygons.json', 'w') as f:
    json.dump(roi_data, f)
print(f'\nSaved ROI polygons to data/raw/wildtrack/manifests/roi_polygons.json')
