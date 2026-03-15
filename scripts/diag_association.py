"""Diagnostic: measure association recall against CityFlowV2 GT."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Load predictions from stage5 MOT files (after mtmc_only_submission filter)
pred_dir = Path("data/outputs/run_20260314_095505/stage5/predictions_mot")
cam_pred_data: dict = defaultdict(lambda: defaultdict(set))  # cam -> track_id -> frames

for pred_file in sorted(pred_dir.glob("*.txt")):
    cam_id = pred_file.stem
    with open(pred_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            track_id = int(parts[1])
            cam_pred_data[cam_id][track_id].add(frame)

print(f"Prediction cameras: {sorted(cam_pred_data.keys())}")
for cam, tracks in sorted(cam_pred_data.items()):
    print(f"  {cam}: {len(tracks)} tracks, {sum(len(f) for f in tracks.values())} frames")

# Load GT
gt_dir = Path("data/raw/cityflowv2")
gt_vehicle_cameras: dict = defaultdict(set)
gt_vehicle_frames: dict = defaultdict(lambda: defaultdict(set))  # vid -> cam -> frames

for gt_file in sorted(gt_dir.glob("*/gt.txt")):
    cam_id = gt_file.parent.name
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            vid = int(parts[1])
            frame = int(parts[0])
            gt_vehicle_cameras[vid].add(cam_id)
            gt_vehicle_frames[vid][cam_id].add(frame)

mc_gt = {vid: cams for vid, cams in gt_vehicle_cameras.items() if len(cams) >= 2}
print(f"\nGT multi-camera vehicles: {len(mc_gt)}")

# For each GT multi-cam vehicle, check if a SINGLE predicted track_id appears in 2+ cameras
matched = 0
unmatched_info = []
matched_info = []

for vid, gt_cams in sorted(mc_gt.items()):
    # Find predicted track_ids that match this vehicle in each camera
    pred_id_cams: dict = defaultdict(set)  # pred_track_id -> set of cameras
    for cam_id in gt_cams:
        if cam_id not in cam_pred_data:
            continue
        gt_frames_this_cam = gt_vehicle_frames[vid][cam_id]
        for pred_tid, pred_frames in cam_pred_data[cam_id].items():
            overlap = len(gt_frames_this_cam & pred_frames)
            if overlap >= 2:
                pred_id_cams[pred_tid].add(cam_id)

    # A vehicle is "matched" if any single pred_id covers 2+ cameras
    multi_cam_preds = {pid: cams for pid, cams in pred_id_cams.items() if len(cams) >= 2}
    total_gt_frames = sum(len(f) for f in gt_vehicle_frames[vid].values())

    if multi_cam_preds:
        matched += 1
        matched_info.append((vid, len(gt_cams), total_gt_frames))
    else:
        unmatched_info.append((vid, sorted(gt_cams), total_gt_frames))

print(f"Correctly linked (matched): {matched}/{len(mc_gt)} = {matched/len(mc_gt)*100:.1f}%")
print(f"Missed (unmatched): {len(unmatched_info)}")

if unmatched_info:
    avg_frames = np.mean([s[2] for s in unmatched_info])
    avg_cams = np.mean([s[1] for s in unmatched_info])
    print(f"\nUnmatched stats: avg_frames={avg_frames:.1f}, avg_cameras={avg_cams:.1f}")
    short = sum(1 for s in unmatched_info if s[2] < 20)
    print(f"Unmatched with < 20 GT frames total: {short} ({short/len(unmatched_info)*100:.1f}%)")
    # Show camera distribution for unmatched
    from collections import Counter
    cam_pairs = Counter()
    for _, cams, _ in unmatched_info:
        cam_pairs[tuple(cams)] += 1
    print("\nMost common camera combos for missed vehicles:")
    for combo, cnt in cam_pairs.most_common(10):
        print(f"  {combo}: {cnt} vehicles")
