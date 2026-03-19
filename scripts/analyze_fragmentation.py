"""Analyze MTMC fragmentation errors: which GT IDs are fragmented and why."""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
# Patch numpy for motmetrics compatibility with numpy 2.0
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
import motmetrics as mm


def load_mot(path):
    """Load MOT format file -> {frame: [(id, x, y, w, h, conf)]}."""
    data = defaultdict(list)
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame = int(parts[0])
        tid = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf = float(parts[6]) if len(parts) > 6 else 1.0
        if conf == 0:
            continue
        data[frame].append((tid, x, y, w, h, conf))
    return data


def analyze_fragmentation(gt_dir, pred_dir, iou_threshold=0.5):
    """Find fragmented GT IDs and which pred IDs they map to."""
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    
    # Accumulate global matches
    gt_to_pred = defaultdict(set)  # gt_id -> set of pred_ids matched
    pred_to_gt = defaultdict(set)  # pred_id -> set of gt_ids matched
    
    # Per-camera analysis
    camera_stats = {}
    
    for pred_file in sorted(pred_dir.glob("*.txt")):
        cam_id = pred_file.stem
        
        # Find GT file
        gt_file = None
        for pattern in [
            gt_dir / cam_id / "gt" / "gt.txt",
            gt_dir / cam_id / "gt.txt",
        ]:
            if pattern.exists():
                gt_file = pattern
                break
        if gt_file is None:
            continue
        
        gt_data = load_mot(gt_file)
        pred_data = load_mot(pred_file)
        
        all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
        
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Track frame-level matches
        cam_gt_to_pred = defaultdict(set)
        
        for frame in all_frames:
            gt_items = gt_data.get(frame, [])
            pred_items = pred_data.get(frame, [])
            
            gt_ids = [g[0] for g in gt_items]
            gt_boxes = np.array([[g[1], g[2], g[1]+g[3], g[2]+g[4]] for g in gt_items]) if gt_items else np.empty((0, 4))
            
            pred_ids = [p[0] for p in pred_items]
            pred_boxes = np.array([[p[1], p[2], p[1]+p[3], p[2]+p[4]] for p in pred_items]) if pred_items else np.empty((0, 4))
            
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=iou_threshold)
            else:
                distances = np.empty((len(gt_boxes), len(pred_boxes)))
            
            acc.update(gt_ids, pred_ids, distances)
        
        # Extract match events
        events = acc.mot_events
        matches = events[events["Type"] == "MATCH"]
        for _, row in matches.iterrows():
            gt_id = row["OId"]
            pred_id = row["HId"]
            gt_to_pred[gt_id].add(pred_id)
            pred_to_gt[pred_id].add(gt_id)
            cam_gt_to_pred[gt_id].add(pred_id)
        
        # Camera stats
        camera_stats[cam_id] = {
            "gt_ids": len(set(g[0] for items in gt_data.values() for g in items)),
            "pred_ids": len(set(p[0] for items in pred_data.values() for p in items)),
        }
    
    # Identify fragmented GT IDs (matched to >1 pred ID)
    fragmented = {gt_id: pred_ids for gt_id, pred_ids in gt_to_pred.items() if len(pred_ids) > 1}
    # Identify conflated pred IDs (matched to >1 GT ID)
    conflated = {pred_id: gt_ids for pred_id, gt_ids in pred_to_gt.items() if len(gt_ids) > 1}
    
    print(f"=== FRAGMENTATION ANALYSIS ===")
    print(f"Total GT IDs seen: {len(gt_to_pred)}")
    print(f"Fragmented GT IDs: {len(fragmented)} (matched to >1 pred ID)")
    print(f"Conflated pred IDs: {len(conflated)} (matched to >1 GT ID)")
    print()
    
    # Sort by number of fragments (worst first)
    sorted_frag = sorted(fragmented.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"Top fragmented GT IDs (worst first):")
    for gt_id, pred_ids in sorted_frag[:30]:
        print(f"  GT {int(gt_id):3d} -> {len(pred_ids)} pred IDs: {sorted(int(p) for p in pred_ids)}")
    
    print()
    print(f"Top conflated pred IDs:")
    sorted_conf = sorted(conflated.items(), key=lambda x: len(x[1]), reverse=True)
    for pred_id, gt_ids in sorted_conf[:20]:
        print(f"  Pred {int(pred_id):3d} -> {len(gt_ids)} GT IDs: {sorted(int(g) for g in gt_ids)}")
    
    # Analyze: how many fragments are within same cluster vs different clusters?
    # Load global trajectories to check
    traj_file = Path("data/outputs/run_20260315_v2/stage4/global_trajectories.json")
    if traj_file.exists():
        trajs = json.load(open(traj_file))
        # Build tracklet_id -> global_id mapping
        tracklet_to_global = {}
        for traj in trajs:
            gid = traj.get("global_id", traj.get("id"))
            for tk in traj.get("tracklets", []):
                tracklet_to_global[tk.get("track_id")] = gid
        
        print(f"\n=== FRAGMENTATION vs GLOBAL TRAJECTORIES ===")
        print(f"Loaded {len(trajs)} global trajectories, {len(tracklet_to_global)} tracklets")
        
        # For each fragmented GT, check if its pred IDs belong to same/different global IDs
        same_global = 0
        diff_global = 0
        for gt_id, pred_ids in sorted_frag[:30]:
            global_ids = set()
            for pid in pred_ids:
                gid = tracklet_to_global.get(pid, f"?{pid}")
                global_ids.add(gid)
            if len(global_ids) == 1:
                same_global += 1
            else:
                diff_global += 1
                print(f"  GT {int(gt_id):3d} -> pred {sorted(int(p) for p in pred_ids)} -> globals {sorted(str(g) for g in global_ids)}")
        
        print(f"\nOf top 30 fragmented: {same_global} same-global (ID switch only), {diff_global} different-global (true fragmentation)")
    
    return fragmented, conflated


if __name__ == "__main__":
    analyze_fragmentation(
        gt_dir="data/raw/cityflowv2",
        pred_dir="data/outputs/run_20260315_v2/stage5/predictions_mot",
        iou_threshold=0.5,
    )
