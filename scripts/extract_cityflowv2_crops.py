"""Extract vehicle ReID crops from CityFlowV2 tracking data.

Reads GT annotations (gt.txt) and video files from each camera in
data/raw/cityflowv2/, extracts vehicle bounding-box crops, and organises
them into train/query/gallery splits suitable for ReID model training.

The resulting directory structure is:
    data/processed/cityflowv2_reid/
      train/   {vid:04d}_{scene}_{cam}_f{frame:06d}.jpg
      query/   ...
      gallery/ ...
      splits.json   (metadata: split statistics, camera mapping, ID lists)

Usage:
    python scripts/extract_cityflowv2_crops.py
    python scripts/extract_cityflowv2_crops.py --data_root data/raw/cityflowv2 --output data/processed/cityflowv2_reid
    python scripts/extract_cityflowv2_crops.py --cameras S01_c001 S01_c002 S01_c003 S02_c006 S02_c007 S02_c008
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────

DEFAULT_DATA_ROOT = "data/raw/cityflowv2"
DEFAULT_OUTPUT = "data/processed/cityflowv2_reid"
DEFAULT_CAMERAS = [
    "S01_c001", "S01_c002", "S01_c003",
    "S02_c006", "S02_c007", "S02_c008",
]

MIN_AREA = 2000           # minimum bbox area (pixels)
MAX_CROPS_PER_ID_CAM = 15 # max crops per vehicle per camera
MIN_BBOX_SIDE = 30        # minimum width/height in pixels
TRAIN_RATIO = 0.7         # fraction of multi-cam IDs used for training
MIN_CAMS_FOR_EVAL = 2     # vehicle must appear in ≥N cameras for query/gallery
SEED = 42


# ── GT parsing ────────────────────────────────────────────────────────────


def load_gt(gt_path: str) -> List[Tuple[int, int, int, int, int, int]]:
    """Load MOT-format GT file → list of (frame, track_id, x, y, w, h)."""
    rows = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame, tid = int(parts[0]), int(parts[1])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            rows.append((frame, tid, x, y, w, h))
    return rows


# ── Crop extraction from video ────────────────────────────────────────────


def extract_crops_from_camera(
    cam_name: str,
    data_root: str,
    crop_dir: str,
    max_crops: int = MAX_CROPS_PER_ID_CAM,
    min_area: int = MIN_AREA,
) -> Dict[int, List[str]]:
    """Extract vehicle crops from a single camera.

    Returns:
        {vehicle_id: [crop_path, ...]}
    """
    gt_path = os.path.join(data_root, cam_name, "gt.txt")
    vid_path = os.path.join(data_root, cam_name, "vdo.avi")

    if not os.path.isfile(gt_path):
        logger.warning(f"  {cam_name}: gt.txt not found, skipping")
        return {}
    if not os.path.isfile(vid_path):
        # Try mp4
        vid_path = os.path.join(data_root, cam_name, "vdo.mp4")
        if not os.path.isfile(vid_path):
            logger.warning(f"  {cam_name}: no video file found, skipping")
            return {}

    gt = load_gt(gt_path)
    if not gt:
        logger.warning(f"  {cam_name}: empty GT, skipping")
        return {}

    # Group detections by vehicle ID, sample uniformly
    id_dets: Dict[int, List[Tuple]] = defaultdict(list)
    for frame, tid, x, y, w, h in gt:
        id_dets[tid].append((frame, x, y, w, h))

    # Build frame→detections map with uniform sampling per ID
    frame_to_dets: Dict[int, List] = defaultdict(list)
    for tid, dets in id_dets.items():
        if len(dets) <= max_crops:
            sampled = dets
        else:
            step = len(dets) / max_crops
            sampled = [dets[int(i * step)] for i in range(max_crops)]

        for frame, x, y, w, h in sampled:
            if w * h >= min_area and w >= MIN_BBOX_SIDE and h >= MIN_BBOX_SIDE:
                frame_to_dets[frame].append((tid, x, y, w, h))

    if not frame_to_dets:
        logger.warning(f"  {cam_name}: no valid detections after filtering")
        return {}

    # Read video and extract crops at target frames
    crops: Dict[int, List[str]] = defaultdict(list)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        logger.error(f"  {cam_name}: cannot open video {vid_path}")
        return {}

    current_frame = 0
    target_frames = sorted(frame_to_dets.keys())
    target_idx = 0

    while target_idx < len(target_frames):
        ret, img = cap.read()
        if not ret:
            break
        current_frame += 1

        if current_frame < target_frames[target_idx]:
            continue
        if current_frame > target_frames[target_idx]:
            while target_idx < len(target_frames) and target_frames[target_idx] < current_frame:
                target_idx += 1
            if target_idx >= len(target_frames):
                break
            if current_frame != target_frames[target_idx]:
                continue

        H, W = img.shape[:2]
        for tid, x, y, w, h in frame_to_dets[current_frame]:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(W, x + w)
            y2 = min(H, y + h)
            if x2 - x1 < MIN_BBOX_SIDE or y2 - y1 < MIN_BBOX_SIDE:
                continue

            crop = img[y1:y2, x1:x2]
            fname = f"{tid:04d}_{cam_name}_f{current_frame:06d}.jpg"
            fpath = os.path.join(crop_dir, fname)
            cv2.imwrite(fpath, crop)
            crops[tid].append(fpath)

        target_idx += 1

    cap.release()
    n_crops = sum(len(v) for v in crops.values())
    logger.info(f"  {cam_name}: {n_crops} crops from {len(crops)} vehicles")
    return dict(crops)


def extract_all_crops(
    data_root: str,
    cameras: List[str],
    tmp_crop_dir: str,
    max_crops: int = MAX_CROPS_PER_ID_CAM,
) -> Dict[int, Dict[str, List[str]]]:
    """Extract crops from all cameras.

    Returns:
        {vehicle_id: {camera_name: [crop_path, ...]}}
    """
    os.makedirs(tmp_crop_dir, exist_ok=True)

    all_crops: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for cam in cameras:
        cam_crops = extract_crops_from_camera(cam, data_root, tmp_crop_dir, max_crops)
        for tid, paths in cam_crops.items():
            all_crops[tid][cam].extend(paths)

    total = sum(sum(len(v) for v in cams.values()) for cams in all_crops.values())
    n_ids = len(all_crops)
    logger.info(f"Total: {total} crops for {n_ids} vehicle IDs across {len(cameras)} cameras")
    return dict(all_crops)


# ── Split creation ────────────────────────────────────────────────────────


def create_splits(
    all_crops: Dict[int, Dict[str, List[str]]],
    output_dir: str,
    train_ratio: float = TRAIN_RATIO,
    min_cams: int = MIN_CAMS_FOR_EVAL,
    seed: int = SEED,
) -> dict:
    """Organise crops into train/query/gallery splits.

    Strategy:
      - Multi-camera IDs (appear in ≥min_cams cameras): split into train/eval sets
      - Train set: train_ratio of multi-cam IDs → all crops go to train/
      - Eval set: remaining multi-cam IDs →
          query: 1 crop per camera per vehicle
          gallery: remaining crops per vehicle
      - Single-camera IDs: added as distractors to gallery (never in query)

    Returns:
        Split statistics dict.
    """
    rng = np.random.RandomState(seed)

    # Classify IDs
    multi_cam_ids = sorted(tid for tid, cams in all_crops.items() if len(cams) >= min_cams)
    single_cam_ids = sorted(tid for tid, cams in all_crops.items() if len(cams) < min_cams)

    logger.info(f"Multi-camera IDs (≥{min_cams} cams): {len(multi_cam_ids)}")
    logger.info(f"Single-camera IDs: {len(single_cam_ids)}")

    # Shuffle and split multi-cam IDs
    rng.shuffle(multi_cam_ids)
    n_train = int(len(multi_cam_ids) * train_ratio)
    train_ids = set(multi_cam_ids[:n_train])
    eval_ids = set(multi_cam_ids[n_train:])

    logger.info(f"Train IDs: {len(train_ids)}, Eval IDs: {len(eval_ids)}")

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    query_dir = os.path.join(output_dir, "query")
    gallery_dir = os.path.join(output_dir, "gallery")
    for d in [train_dir, query_dir, gallery_dir]:
        os.makedirs(d, exist_ok=True)

    stats = {"train": 0, "query": 0, "gallery": 0, "distractors": 0}

    # 1. Train split: all crops for train IDs
    for tid in sorted(train_ids):
        for cam_name, paths in all_crops[tid].items():
            for src_path in paths:
                fname = os.path.basename(src_path)
                dst_path = os.path.join(train_dir, fname)
                shutil.copy2(src_path, dst_path)
                stats["train"] += 1

    # 2. Query + Gallery: eval IDs
    for tid in sorted(eval_ids):
        cams = all_crops[tid]
        for cam_name, paths in cams.items():
            if not paths:
                continue
            # Pick 1 for query
            idx = rng.randint(0, len(paths))
            q_src = paths[idx]
            q_fname = os.path.basename(q_src)
            shutil.copy2(q_src, os.path.join(query_dir, q_fname))
            stats["query"] += 1

            # Rest to gallery
            for i, src_path in enumerate(paths):
                if i != idx:
                    fname = os.path.basename(src_path)
                    shutil.copy2(src_path, os.path.join(gallery_dir, fname))
                    stats["gallery"] += 1

    # 3. Single-camera IDs → gallery as distractors
    for tid in sorted(single_cam_ids):
        for cam_name, paths in all_crops[tid].items():
            for src_path in paths:
                fname = os.path.basename(src_path)
                shutil.copy2(src_path, os.path.join(gallery_dir, fname))
                stats["gallery"] += 1
                stats["distractors"] += 1

    logger.info(f"Split stats: {stats}")
    return stats


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract vehicle ReID crops from CityFlowV2 tracking data"
    )
    parser.add_argument(
        "--data_root", default=DEFAULT_DATA_ROOT,
        help="Path to raw CityFlowV2 dataset",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output directory for organised ReID crops",
    )
    parser.add_argument(
        "--cameras", nargs="+", default=DEFAULT_CAMERAS,
        help="Camera names to process",
    )
    parser.add_argument(
        "--max_crops", type=int, default=MAX_CROPS_PER_ID_CAM,
        help="Max crops per vehicle per camera",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=TRAIN_RATIO,
        help="Fraction of multi-cam IDs for training (rest for eval)",
    )
    parser.add_argument(
        "--min_cams", type=int, default=MIN_CAMS_FOR_EVAL,
        help="Min cameras for a vehicle to be in eval set",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--reuse_crops", action="store_true",
        help="Reuse existing flat crops dir if available",
    )
    args = parser.parse_args()

    logger.info(f"CityFlowV2 data root: {args.data_root}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Cameras: {args.cameras}")

    # Check data root exists
    if not os.path.isdir(args.data_root):
        logger.error(
            f"Data root not found: {args.data_root}\n"
            f"Download CityFlowV2 first:\n"
            f"  python scripts/download_datasets.py --dataset cityflowv2"
        )
        sys.exit(1)

    # Temporary flat crop dir
    tmp_crop_dir = os.path.join(args.output, "_all_crops")

    # Step 1: Extract or reuse crops
    if args.reuse_crops and os.path.isdir(tmp_crop_dir) and len(os.listdir(tmp_crop_dir)) > 50:
        logger.info(f"Reusing existing crops from {tmp_crop_dir}")
        # Rebuild crop dict from filenames
        all_crops: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for fname in sorted(os.listdir(tmp_crop_dir)):
            if not fname.endswith(".jpg"):
                continue
            parts = fname.split("_")
            if len(parts) < 4:
                continue
            tid = int(parts[0])
            cam = parts[1] + "_" + parts[2]
            all_crops[tid][cam].append(os.path.join(tmp_crop_dir, fname))
        all_crops = dict(all_crops)
        total = sum(sum(len(v) for v in c.values()) for c in all_crops.values())
        logger.info(f"Found {total} crops for {len(all_crops)} IDs")
    else:
        logger.info("Extracting crops from videos...")
        all_crops = extract_all_crops(
            args.data_root, args.cameras, tmp_crop_dir, args.max_crops,
        )

    if not all_crops:
        logger.error("No crops extracted. Check that videos and GT files exist.")
        sys.exit(1)

    # Step 2: Create train/query/gallery splits
    logger.info("Creating train/query/gallery splits...")
    stats = create_splits(
        all_crops, args.output,
        train_ratio=args.train_ratio,
        min_cams=args.min_cams,
        seed=args.seed,
    )

    # Step 3: Save metadata
    cam_names = sorted({cam for cams in all_crops.values() for cam in cams})
    multi_cam_ids = [tid for tid, cams in all_crops.items() if len(cams) >= args.min_cams]
    single_cam_ids = [tid for tid, cams in all_crops.items() if len(cams) < args.min_cams]

    metadata = {
        "dataset": "cityflowv2_reid",
        "source": args.data_root,
        "cameras": cam_names,
        "num_cameras": len(cam_names),
        "total_vehicle_ids": len(all_crops),
        "multi_cam_ids": len(multi_cam_ids),
        "single_cam_ids": len(single_cam_ids),
        "splits": stats,
        "parameters": {
            "max_crops_per_id_cam": args.max_crops,
            "train_ratio": args.train_ratio,
            "min_cams_for_eval": args.min_cams,
            "min_area": MIN_AREA,
            "seed": args.seed,
        },
    }

    meta_path = os.path.join(args.output, "splits.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nMetadata saved to {meta_path}")
    logger.info(f"Dataset ready at {args.output}")
    logger.info(f"  train/   {stats['train']} images")
    logger.info(f"  query/   {stats['query']} images")
    logger.info(f"  gallery/ {stats['gallery']} images (incl. {stats['distractors']} distractors)")


if __name__ == "__main__":
    main()
