"""Deep diagnostic: compute per-camera ROI from GT, and estimate
how much FP would drop if we spatially filter predictions."""
import sys, json, os
import numpy as np
sys.path.insert(0, '.')

# For each camera, compute the bounding hull of GT boxes
# Then see how many predictions fall OUTSIDE that hull

cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

for cam_id in cameras:
    gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
    pred_path = f'data/outputs/run_20260304_050358/stage5/predictions_mot/{cam_id}.txt'
    
    # Load GT boxes
    gt_centers_x = []
    gt_centers_y = []
    gt_bottoms_y = []
    for line in open(gt_path):
        parts = line.strip().split(',')
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        gt_centers_x.append(x + w/2)
        gt_centers_y.append(y + h/2)
        gt_bottoms_y.append(y + h)
    
    # GT spatial extent (use percentiles to be robust)
    gt_xmin = np.percentile(gt_centers_x, 1)
    gt_xmax = np.percentile(gt_centers_x, 99)
    gt_ymin = np.percentile(gt_centers_y, 1)
    gt_ymax = np.percentile(gt_centers_y, 99)
    gt_bottom_max = np.percentile(gt_bottoms_y, 99)
    
    # Count predictions inside vs outside GT spatial extent (with margin)
    margin = 100  # pixels
    inside = 0
    outside = 0
    total_pred = 0
    gt_count = len(gt_centers_x)
    
    for line in open(pred_path):
        parts = line.strip().split(',')
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        cx = x + w/2
        cy = y + h/2
        total_pred += 1
        
        if (gt_xmin - margin) <= cx <= (gt_xmax + margin) and \
           (gt_ymin - margin) <= cy <= (gt_ymax + margin):
            inside += 1
        else:
            outside += 1
    
    gt_per_frame = gt_count / 400
    pred_per_frame = total_pred / 401
    inside_per_frame = inside / 401
    
    print(f'{cam_id}: GT={gt_per_frame:.1f}/f, Pred={pred_per_frame:.1f}/f, '
          f'Inside_ROI={inside_per_frame:.1f}/f, Outside={outside}/{total_pred} '
          f'({100*outside/max(total_pred,1):.0f}%)')
    print(f'  GT extent: x=[{gt_xmin:.0f},{gt_xmax:.0f}], y=[{gt_ymin:.0f},{gt_ymax:.0f}], '
          f'bottom_y<={gt_bottom_max:.0f}')

# Also: Single-camera MOTA if we evaluate with original per-camera track IDs
# instead of global IDs
print('\n--- Per-camera MOTA with ORIGINAL track IDs (no association) ---')
from src.core.io_utils import load_tracklets_by_camera
tracklets = load_tracklets_by_camera('data/outputs/run_20260304_050358/stage1')

import motmetrics as mm
# Fix numpy 2.0 compatibility
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x, dtype=np.float64: np.asarray(x, dtype=dtype)

for cam_id in cameras:
    gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
    cam_t = tracklets.get(cam_id, [])
    
    # Build predictions from raw tracklets (using original track_ids)
    pred_data = {}
    for t in cam_t:
        for tf in t.frames:
            fid = tf.frame_id
            x1, y1, x2, y2 = tf.bbox
            w, h = x2-x1, y2-y1
            pred_data.setdefault(fid, []).append((t.track_id, [x1, y1, w, h]))
    
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
    print(f'  {cam_id}: MOTA={mota:.3f}, IDF1={idf1:.3f}, IDSW={idsw}, FP={fp}, FN={fn}, Prec={prec:.3f}, Rec={rec:.3f}')
