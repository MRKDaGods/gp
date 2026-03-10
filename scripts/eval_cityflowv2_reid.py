"""Evaluate TransReID vehicle model on CityFlowV2 as ad-hoc ReID benchmark.

Extracts vehicle crops from GT annotations + video frames, builds a
query/gallery split (one random crop per vehicle per camera as query,
remaining as gallery), then evaluates mAP/CMC with optional QE and
k-reciprocal re-ranking.

Usage:
    python -m scripts.eval_cityflowv2_reid \
        --weights models/reid/vehicle_transreid_vit_base_veri776.pth \
        --qe_k 3 --k1 30 --k2 10 --lambda_value 0.2 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage2_features.transreid_model import build_transreid
from src.training.evaluate_reid import (
    compute_reranking,
    eval_market1501,
)

# ── CLIP normalization (must match training) ─────────────────────────────
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DATA_ROOT = "data/raw/cityflowv2"
CAMERAS = [
    "S01_c001",
    "S01_c002",
    "S01_c003",
    "S02_c006",
    "S02_c007",
    "S02_c008",
]


# ── Step 1: Extract crops from videos ────────────────────────────────────


def load_gt(gt_path: str) -> List[Tuple[int, int, int, int, int, int]]:
    """Load MOT GT file → list of (frame, id, x, y, w, h)."""
    rows = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            frame, tid = int(parts[0]), int(parts[1])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            rows.append((frame, tid, x, y, w, h))
    return rows


def extract_crops(
    data_root: str,
    cameras: List[str],
    crop_dir: str,
    max_crops_per_id_cam: int = 10,
    min_area: int = 2000,
) -> Dict[int, Dict[str, List[str]]]:
    """Extract vehicle crops from video + GT.

    Returns:
        {vehicle_id: {camera_name: [crop_path, ...]}}
    """
    os.makedirs(crop_dir, exist_ok=True)
    all_crops: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for cam in cameras:
        gt_path = os.path.join(data_root, cam, "gt.txt")
        vid_path = os.path.join(data_root, cam, "vdo.avi")
        if not os.path.isfile(gt_path) or not os.path.isfile(vid_path):
            logger.warning(f"Skipping {cam}: missing gt.txt or vdo.avi")
            continue

        gt = load_gt(gt_path)

        # Group by frame for efficient video reading
        frame_to_dets: Dict[int, List] = defaultdict(list)
        # Track how many crops per id we've already saved for this cam
        id_crop_count: Dict[int, int] = defaultdict(int)

        # Sample uniformly: pick every N-th detection per ID
        id_frames: Dict[int, List] = defaultdict(list)
        for frame, tid, x, y, w, h in gt:
            id_frames[tid].append((frame, x, y, w, h))

        for tid, dets in id_frames.items():
            # Sample up to max_crops_per_id_cam uniformly
            if len(dets) <= max_crops_per_id_cam:
                sampled = dets
            else:
                step = len(dets) / max_crops_per_id_cam
                sampled = [dets[int(i * step)] for i in range(max_crops_per_id_cam)]
            for frame, x, y, w, h in sampled:
                if w * h >= min_area:
                    frame_to_dets[frame].append((tid, x, y, w, h))

        if not frame_to_dets:
            continue

        cap = cv2.VideoCapture(vid_path)
        current_frame = 0
        target_frames = sorted(frame_to_dets.keys())
        target_idx = 0

        logger.info(f"  {cam}: extracting crops from {len(target_frames)} frames ...")

        while target_idx < len(target_frames):
            ret, img = cap.read()
            if not ret:
                break
            current_frame += 1

            if current_frame < target_frames[target_idx]:
                continue
            if current_frame > target_frames[target_idx]:
                # Shouldn't happen with sorted targets, but advance
                while target_idx < len(target_frames) and target_frames[target_idx] < current_frame:
                    target_idx += 1
                if target_idx >= len(target_frames):
                    break
                if current_frame != target_frames[target_idx]:
                    continue

            # Process detections for this frame
            for tid, x, y, w, h in frame_to_dets[current_frame]:
                H, W = img.shape[:2]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(W, x + w)
                y2 = min(H, y + h)
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                crop = img[y1:y2, x1:x2]
                fname = f"{tid:04d}_{cam}_f{current_frame:06d}.jpg"
                fpath = os.path.join(crop_dir, fname)
                cv2.imwrite(fpath, crop)
                all_crops[tid][cam].append(fpath)

            target_idx += 1

        cap.release()

    # Summary
    total = sum(sum(len(v) for v in cams.values()) for cams in all_crops.values())
    n_ids = len(all_crops)
    logger.info(f"Extracted {total} crops for {n_ids} vehicle IDs")
    return dict(all_crops)


# ── Step 2: Build query/gallery split ────────────────────────────────────


def build_reid_split(
    all_crops: Dict[int, Dict[str, List[str]]],
    min_images: int = 2,
) -> Tuple[List, List]:
    """Build query and gallery lists.

    For each vehicle ID that appears in ≥2 cameras:
      - Query: 1 random crop from each camera
      - Gallery: all remaining crops

    Returns:
        (query_list, gallery_list) where each entry is (path, pid, camid)
    """
    # Map cam names to integer IDs
    cam_names = sorted({cam for cams in all_crops.values() for cam in cams})
    cam2id = {c: i for i, c in enumerate(cam_names)}

    # Re-label pids to 0..N-1
    multi_cam_ids = sorted(
        tid for tid, cams in all_crops.items() if len(cams) >= 2
    )
    pid2label = {tid: i for i, tid in enumerate(multi_cam_ids)}

    query, gallery = [], []
    rng = np.random.RandomState(42)

    for tid in multi_cam_ids:
        pid = pid2label[tid]
        cams = all_crops[tid]

        for cam_name, paths in cams.items():
            camid = cam2id[cam_name]
            if len(paths) == 0:
                continue

            # Pick 1 for query
            idx = rng.randint(0, len(paths))
            query.append((paths[idx], pid, camid))

            # Rest as gallery
            for i, p in enumerate(paths):
                if i != idx:
                    gallery.append((p, pid, camid))

    # Also add single-camera IDs to gallery only (distractors)
    single_cam_ids = sorted(
        tid for tid, cams in all_crops.items() if len(cams) == 1
    )
    distractor_pid = len(multi_cam_ids)  # unique PID that won't match any query
    for tid in single_cam_ids:
        cams = all_crops[tid]
        for cam_name, paths in cams.items():
            camid = cam2id[cam_name]
            for p in paths:
                gallery.append((p, distractor_pid, camid))
        distractor_pid += 1

    logger.info(
        f"ReID split: {len(query)} query, {len(gallery)} gallery "
        f"({len(multi_cam_ids)} IDs with ≥2 cameras, "
        f"{len(single_cam_ids)} distractor IDs)"
    )
    return query, gallery


# ── Step 3: Dataset & transforms ─────────────────────────────────────────


class CropDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int, int]], transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, pid, camid = self.data[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, pid, camid, path


def build_eval_transforms(height: int = 224, width: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ── Step 4: Feature extraction ───────────────────────────────────────────


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda:0",
    flip: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract L2-normalized features."""
    model.eval()
    all_feats, all_pids, all_cams = [], [], []

    for imgs, pids, cams, _ in dataloader:
        imgs = imgs.to(device)
        feats = model(imgs)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        if flip:
            imgs_flip = torch.flip(imgs, dims=[3])
            feats_flip = model(imgs_flip)
            if isinstance(feats_flip, (tuple, list)):
                feats_flip = feats_flip[-1]
            feats = (feats + feats_flip) / 2.0

        feats = F.normalize(feats, p=2, dim=1)

        all_feats.append(feats.cpu().numpy())
        all_pids.append(pids.numpy())
        all_cams.append(cams.numpy())

    return (
        np.concatenate(all_feats),
        np.concatenate(all_pids),
        np.concatenate(all_cams),
    )


# ── Step 5: Average Query Expansion ──────────────────────────────────────


def average_query_expansion(
    features: np.ndarray,
    k: int,
) -> np.ndarray:
    """AQE: replace each feature with the mean of its k nearest neighbors.

    Works on a single set of features (query or combined).
    """
    if k <= 0:
        return features

    # Cosine similarity (features are L2-normed)
    sim = features @ features.T
    # Top-k indices per row (including self)
    topk_idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]

    expanded = np.zeros_like(features)
    for i in range(len(features)):
        expanded[i] = features[topk_idx[i]].mean(axis=0)

    # Re-normalize
    norms = np.linalg.norm(expanded, axis=1, keepdims=True) + 1e-12
    expanded = expanded / norms
    return expanded


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate TransReID on CityFlowV2")
    parser.add_argument("--weights", default="models/reid/vehicle_transreid_vit_base_veri776.pth")
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--crop_dir", default="data/processed/cityflowv2_crops")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224], help="H W")
    parser.add_argument("--max_crops", type=int, default=10, help="Max crops per ID per camera")
    # QE & re-ranking
    parser.add_argument("--qe_k", type=int, default=0, help="Average Query Expansion k (0=off)")
    parser.add_argument("--rerank", action="store_true", default=True)
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--lambda_value", type=float, default=0.3)
    args = parser.parse_args()

    logger.info(f"Device: {args.device}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Image size: {args.img_size}")
    logger.info(f"QE k={args.qe_k}, Re-rank: k1={args.k1}, k2={args.k2}, λ={args.lambda_value}")

    # ── 1. Extract crops ──────────────────────────────────────────────
    if os.path.isdir(args.crop_dir) and len(os.listdir(args.crop_dir)) > 100:
        logger.info(f"Reusing existing crops in {args.crop_dir}")
        # Rebuild all_crops from filenames: {tid}_{cam}_f{frame}.jpg
        all_crops: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for fname in sorted(os.listdir(args.crop_dir)):
            if not fname.endswith(".jpg"):
                continue
            parts = fname.split("_")
            tid = int(parts[0])
            cam = parts[1] + "_" + parts[2]
            all_crops[tid][cam].append(os.path.join(args.crop_dir, fname))
        all_crops = dict(all_crops)
        total = sum(sum(len(v) for v in c.values()) for c in all_crops.values())
        logger.info(f"Found {total} existing crops for {len(all_crops)} IDs")
    else:
        logger.info("Extracting crops from videos ...")
        all_crops = extract_crops(
            args.data_root, CAMERAS, args.crop_dir,
            max_crops_per_id_cam=args.max_crops,
        )

    # ── 2. Build query/gallery ────────────────────────────────────────
    query_list, gallery_list = build_reid_split(all_crops)

    transform = build_eval_transforms(args.img_size[0], args.img_size[1])
    q_dataset = CropDataset(query_list, transform)
    g_dataset = CropDataset(gallery_list, transform)

    q_loader = DataLoader(
        q_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    g_loader = DataLoader(
        g_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 3. Load model ────────────────────────────────────────────────
    logger.info("Loading TransReID model ...")
    model = build_transreid(
        num_classes=1,
        num_cameras=20,
        embed_dim=768,
        vit_model="vit_base_patch16_clip_224.openai",
        pretrained=False,
        weights_path=args.weights,
        img_size=tuple(args.img_size),
    )
    model = model.to(args.device)
    model.eval()

    # ── 4. Extract features ──────────────────────────────────────────
    t0 = time.time()
    logger.info("Extracting query features ...")
    q_feats, q_pids, q_camids = extract_features(model, q_loader, args.device)
    logger.info(f"  Query: {q_feats.shape[0]} images, {len(set(q_pids))} IDs, dim={q_feats.shape[1]}")

    logger.info("Extracting gallery features ...")
    g_feats, g_pids, g_camids = extract_features(model, g_loader, args.device)
    logger.info(f"  Gallery: {g_feats.shape[0]} images, {len(set(g_pids))} IDs, dim={g_feats.shape[1]}")
    t_feat = time.time() - t0
    logger.info(f"Feature extraction: {t_feat:.1f}s")

    # ── 5. Optional: Average Query Expansion ─────────────────────────
    if args.qe_k > 0:
        logger.info(f"Applying Average Query Expansion (k={args.qe_k}) ...")
        # QE on combined features, then split back
        all_feats = np.concatenate([q_feats, g_feats], axis=0)
        all_feats = average_query_expansion(all_feats, k=args.qe_k)
        q_feats = all_feats[: len(q_pids)]
        g_feats = all_feats[len(q_pids) :]

    # ── 6. Standard eval (cosine distance) ───────────────────────────
    logger.info("Computing cosine distance ...")
    sim = q_feats @ g_feats.T
    distmat = 1.0 - sim

    mAP, cmc = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"\n{'='*50}")
    logger.info(f"Standard (cosine) results:")
    logger.info(f"  mAP:    {mAP*100:.2f}%")
    logger.info(f"  Rank-1: {cmc[0]*100:.2f}%")
    logger.info(f"  Rank-5: {cmc[4]*100:.2f}%")
    logger.info(f"  Rank-10:{cmc[9]*100:.2f}%")

    # ── 7. Re-ranking ────────────────────────────────────────────────
    if args.rerank:
        logger.info(
            f"\nComputing re-ranked distance (k1={args.k1}, k2={args.k2}, "
            f"λ={args.lambda_value}) ..."
        )
        t0 = time.time()
        distmat_rr = compute_reranking(
            q_feats, g_feats,
            k1=args.k1, k2=args.k2, lambda_value=args.lambda_value,
        )
        t_rr = time.time() - t0
        mAP_rr, cmc_rr = eval_market1501(
            distmat_rr, q_pids, g_pids, q_camids, g_camids
        )
        logger.info(f"Re-ranking results (computed in {t_rr:.1f}s):")
        logger.info(f"  mAP:    {mAP_rr*100:.2f}%")
        logger.info(f"  Rank-1: {cmc_rr[0]*100:.2f}%")
        logger.info(f"  Rank-5: {cmc_rr[4]*100:.2f}%")
        logger.info(f"  Rank-10:{cmc_rr[9]*100:.2f}%")

    logger.info(f"\n{'='*50}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
