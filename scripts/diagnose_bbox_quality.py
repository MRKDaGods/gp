"""Deep diagnosis of bbox quality: duplicates, IoU distribution, format consistency."""
import sys, json, os, numpy as np
sys.path.insert(0, '.')

from src.core.wildtrack_calibration import load_wildtrack_calibration
from src.core.io_utils import load_tracklets_by_camera

cals = load_wildtrack_calibration('data/raw/wildtrack/calibrations')
tracklets = load_tracklets_by_camera('data/outputs/run_20260304_050358/stage1')

GP_XMIN, GP_XMAX = -300, 900
GP_YMIN, GP_YMAX = -900, 2700
GP_MARGIN = 100

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

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2]); y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = b1[2]*b1[3]; a2 = b2[2]*b2[3]
    return inter / (a1 + a2 - inter) if (a1+a2-inter) > 0 else 0

cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

for cam_id in cameras:
    cal = cals[cam_id]; K, R, tvec = cal['K'], cal['R'], cal['tvec']
    cam_t = tracklets.get(cam_id, [])
    
    gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
    gt_data = {}
    for line in open(gt_path):
        parts = line.strip().split(',')
        fid, tid = int(parts[0]), int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        gt_data.setdefault(fid, []).append((tid, [x, y, w, h]))
    
    # Build pred data with GP filter + conf >= 0.40
    pred_data = {}
    for t in cam_t:
        for tf in t.frames:
            if tf.confidence < 0.40: continue
            x1, y1, x2, y2 = tf.bbox
            w, h = x2-x1, y2-y1
            foot_x, foot_y = float(x1 + w/2), float(y2)
            if is_on_gp(foot_x, foot_y, K, R, tvec):
                pred_data.setdefault(tf.frame_id, []).append((t.track_id, [x1, y1, w, h], tf.confidence))
    
    # Check for duplicate predictions (overlapping boxes in same frame)
    dup_counts = []
    for fid, preds in pred_data.items():
        boxes = [p[1] for p in preds]
        dups = 0
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                iou = compute_iou(boxes[i], boxes[j])
                if iou > 0.3:
                    dups += 1
        dup_counts.append(dups)
    
    # IoU distribution: for each GT det, find best matching pred
    best_ious = []
    unmatched_gt = 0
    for fid in gt_data:
        gt_boxes = [g[1] for g in gt_data[fid]]
        pred_boxes = [p[1] for p in pred_data.get(fid, [])]
        for gb in gt_boxes:
            if not pred_boxes:
                unmatched_gt += 1
                continue
            ious = [compute_iou(gb, pb) for pb in pred_boxes]
            best = max(ious)
            best_ious.append(best)
            if best < 0.5:
                unmatched_gt += 1
    
    # Box size comparison
    gt_widths, gt_heights = [], []
    pred_widths, pred_heights = [], []
    for fid in gt_data:
        for _, box in gt_data[fid]:
            gt_widths.append(box[2]); gt_heights.append(box[3])
    for fid in pred_data:
        for _, box, _ in pred_data[fid]:
            pred_widths.append(box[2]); pred_heights.append(box[3])
    
    # Frame 5 detailed view
    sample_fid = 5
    print(f'\n=== {cam_id} ===')
    print(f'  Avg overlapping pairs/frame (IoU>0.3): {np.mean(dup_counts):.1f}')
    print(f'  GT bbox size: w={np.mean(gt_widths):.0f}±{np.std(gt_widths):.0f}, h={np.mean(gt_heights):.0f}±{np.std(gt_heights):.0f}')
    print(f'  Pred bbox size: w={np.mean(pred_widths):.0f}±{np.std(pred_widths):.0f}, h={np.mean(pred_heights):.0f}±{np.std(pred_heights):.0f}')
    if best_ious:
        print(f'  Best IoU distribution: mean={np.mean(best_ious):.3f}, median={np.median(best_ious):.3f}')
        print(f'  IoU bins: <0.1={sum(1 for x in best_ious if x<0.1)/len(best_ious)*100:.0f}%, '
              f'0.1-0.3={sum(1 for x in best_ious if 0.1<=x<0.3)/len(best_ious)*100:.0f}%, '
              f'0.3-0.5={sum(1 for x in best_ious if 0.3<=x<0.5)/len(best_ious)*100:.0f}%, '
              f'>=0.5={sum(1 for x in best_ious if x>=0.5)/len(best_ious)*100:.0f}%')
    print(f'  Unmatched GT (best IoU < 0.5): {unmatched_gt}')
    
    if sample_fid in gt_data and sample_fid in pred_data:
        print(f'  Frame {sample_fid} GT boxes:')
        for tid, box in gt_data[sample_fid]:
            print(f'    id={tid}: x={box[0]:.0f}, y={box[1]:.0f}, w={box[2]:.0f}, h={box[3]:.0f}')
        print(f'  Frame {sample_fid} Pred boxes:')
        for tid, box, conf in pred_data[sample_fid]:
            print(f'    id={tid}: x={box[0]:.0f}, y={box[1]:.0f}, w={box[2]:.0f}, h={box[3]:.0f}, conf={conf:.2f}')
