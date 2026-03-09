"""Diagnostic script to understand metric quality issues."""
import sys, json, os
import numpy as np
sys.path.insert(0, '.')

# 1. Check GT format
gt_lines = open('data/raw/wildtrack/manifests/ground_truth/C1.txt').readlines()
gt_frames = sorted(set(int(l.split(',')[0]) for l in gt_lines if l.strip()))
print(f'GT C1: {len(gt_lines)} lines, frames {gt_frames[0]}-{gt_frames[-1]}, {len(gt_frames)} unique frames')

# 2. Check predictions
pred_lines = open('data/outputs/run_20260304_050358/stage5/predictions_mot/C1.txt').readlines()
pred_frames = sorted(set(int(l.split(',')[0]) for l in pred_lines if l.strip()))
print(f'Pred C1: {len(pred_lines)} lines, frames {pred_frames[0]}-{pred_frames[-1]}, {len(pred_frames)} unique frames')

# 3. Check annotation format
ann_files = sorted(os.listdir('data/raw/wildtrack/annotations_positions'))[:5]
print(f'Annotation files: {ann_files}')
ann0 = json.load(open(f'data/raw/wildtrack/annotations_positions/{ann_files[0]}'))
n_entries = len(ann0)
keys = list(ann0[0].keys()) if ann0 else []
print(f'Ann frame 0: {n_entries} entries, keys: {keys}')
if ann0:
    print(f'  Sample entry: {json.dumps(ann0[0], indent=2)[:300]}')

# 4. Check tracklets
from src.core.io_utils import load_tracklets_by_camera
tracklets = load_tracklets_by_camera('data/outputs/run_20260304_050358/stage1')

total_tracklets = sum(len(v) for v in tracklets.values())
print(f'\nTotal tracklets: {total_tracklets} across {len(tracklets)} cameras')

for cam_id in sorted(tracklets.keys()):
    cam_t = tracklets[cam_id]
    all_fids = set()
    for t in cam_t:
        for tf in t.frames:
            all_fids.add(tf.frame_id)
    lens = [len(t.frames) for t in cam_t]
    print(f'  {cam_id}: {len(cam_t)} tracklets, frame_ids {min(all_fids)}-{max(all_fids)}, '
          f'track lengths min={min(lens)} med={np.median(lens):.0f} mean={np.mean(lens):.1f} max={max(lens)}')

# 5. Detection counts per frame comparison
print('\n--- Detection density ---')
for cam_id in ['C1', 'C3', 'C7']:
    gt_path = f'data/raw/wildtrack/manifests/ground_truth/{cam_id}.txt'
    pred_path = f'data/outputs/run_20260304_050358/stage5/predictions_mot/{cam_id}.txt'
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        continue
    gt_c = {}
    for l in open(gt_path):
        fid = int(l.split(',')[0])
        gt_c[fid] = gt_c.get(fid, 0) + 1
    pred_c = {}
    for l in open(pred_path):
        fid = int(l.split(',')[0])
        pred_c[fid] = pred_c.get(fid, 0) + 1
    common = set(gt_c.keys()) & set(pred_c.keys())
    gt_avg = np.mean([gt_c[f] for f in common]) if common else 0
    pred_avg = np.mean([pred_c[f] for f in common]) if common else 0
    print(f'  {cam_id}: GT avg={gt_avg:.1f}/frame, Pred avg={pred_avg:.1f}/frame, '
          f'common_frames={len(common)}, GT-only={len(set(gt_c)-common)}, Pred-only={len(set(pred_c)-common)}')

# 6. IoU analysis across multiple frames
print('\n--- IoU analysis (C1, frames 0,100,200,300) ---')
def load_frame_data(path, frame_target):
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            if fid != frame_target:
                continue
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            boxes.append((tid, x, y, w, h))
    return boxes

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    a1 = b1[2]*b1[3]
    a2 = b2[2]*b2[3]
    return inter / (a1 + a2 - inter)

for fid in [0, 100, 200, 300]:
    gt = load_frame_data('data/raw/wildtrack/manifests/ground_truth/C1.txt', fid)
    pred = load_frame_data('data/outputs/run_20260304_050358/stage5/predictions_mot/C1.txt', fid)
    matched_50 = 0
    matched_30 = 0
    for _, gx, gy, gw, gh in gt:
        best = max((compute_iou((gx,gy,gw,gh), (px,py,pw,ph)) for _, px, py, pw, ph in pred), default=0)
        if best >= 0.5: matched_50 += 1
        if best >= 0.3: matched_30 += 1
    print(f'  Frame {fid}: GT={len(gt)}, Pred={len(pred)}, '
          f'matched@0.5={matched_50}/{len(gt)} ({100*matched_50/max(len(gt),1):.0f}%), '
          f'matched@0.3={matched_30}/{len(gt)} ({100*matched_30/max(len(gt),1):.0f}%)')

# 7. Check global trajectory quality
from src.core.io_utils import load_global_trajectories
trajs = load_global_trajectories('data/outputs/run_20260304_050358/stage4/global_trajectories.json')
print(f'\nGlobal trajectories: {len(trajs)}')
multi_cam = [t for t in trajs if len(set(tr.camera_id for tr in t.tracklets)) > 1]
print(f'Multi-camera trajectories: {len(multi_cam)}')
cam_counts = [len(set(tr.camera_id for tr in t.tracklets)) for t in multi_cam]
if cam_counts:
    print(f'Cameras per multi-cam traj: min={min(cam_counts)}, max={max(cam_counts)}, avg={np.mean(cam_counts):.1f}')

# 8. Check Stage 4 embeddings
emb = np.load('data/outputs/run_20260304_050358/stage2/embeddings.npy')
print(f'\nEmbeddings shape: {emb.shape}')
print(f'Embedding norms: min={np.linalg.norm(emb, axis=1).min():.3f}, max={np.linalg.norm(emb, axis=1).max():.3f}')

# Check cosine similarity distribution
from numpy.linalg import norm
norms = norm(emb, axis=1, keepdims=True)
emb_normed = emb / np.clip(norms, 1e-8, None)
# Random sample of cosine similarities
rng = np.random.default_rng(42)
idx1 = rng.integers(0, len(emb), 1000)
idx2 = rng.integers(0, len(emb), 1000)
sims = np.sum(emb_normed[idx1] * emb_normed[idx2], axis=1)
print(f'Random cosine similarity: mean={sims.mean():.3f}, std={sims.std():.3f}, min={sims.min():.3f}, max={sims.max():.3f}')
