"""Generate notebooks/kaggle/09b_vehicle_reid_384px/09b_vehicle_reid_384px.ipynb"""
import json
from pathlib import Path


def cell(ct, src):
    lines = src.split('\n')
    sources = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    while sources and sources[-1] == '':
        sources.pop()
    if ct == 'code':
        return {'cell_type': 'code', 'execution_count': None,
                'metadata': {}, 'outputs': [], 'source': sources}
    else:
        return {'cell_type': 'markdown', 'metadata': {}, 'source': sources}


cells = []

# ── Cell 0: Title ──────────────────────────────────────────────────────────────
cells.append(cell('markdown', r"""# Notebook 09b: Vehicle ReID -- 384px Fine-Tuning

**Purpose**: Fine-tune the CityFlowV2 TransReID model (256px) at 384×384 resolution.

Key changes vs NB09:
- Input: **384×384** (vs 256×256)
- Init: CityFlowV2-pretrained weights + bicubic pos_embed interpolation (16×16 → 24×24 grid)
- LR: 1e-5 backbone, 1e-4 head (10× lower — continued training, not restart)
- Epochs: 40 (vs 120) — already well-trained
- Output: `transreid_cityflowv2_384_best.pth`"""))

# ── Cell 1: Install ────────────────────────────────────────────────────────────
cells.append(cell('code', '!pip install -q timm matplotlib pandas loguru gdown'))

# ── Cell 2: Setup ──────────────────────────────────────────────────────────────
cells.append(cell('markdown', '## 1. Setup'))

cells.append(cell('code', r"""import os
import sys
import json
import time
import re
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
NUM_GPUS = max(torch.cuda.device_count(), 1)
print(f"GPUs available: {NUM_GPUS}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.1f} GB)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
if Path("/kaggle/input").exists():
    PROJECT = Path("/kaggle/working/project")
    PROJECT.mkdir(parents=True, exist_ok=True)
    import subprocess
    if not (PROJECT / ".git").exists():
        subprocess.run(["git", "clone", "--depth=1",
                        "https://github.com/mrkdagods/gp.git", str(PROJECT)], check=True)
    else:
        subprocess.run(["git", "-C", str(PROJECT), "pull", "--ff-only"], check=True)
    sys.path.insert(0, str(PROJECT))
else:
    PROJECT = Path(".").resolve()
    sys.path.insert(0, str(PROJECT))

OUTPUT_DIR = Path("/tmp/nb09b_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR = Path("/kaggle/working/exported_models") if Path("/kaggle/working").exists() else OUTPUT_DIR
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Project: {PROJECT}")
print(f"Output:  {OUTPUT_DIR}")
print(f"Export:  {EXPORT_DIR}")"""))

# ── Cell 3: Load pretrained weights ───────────────────────────────────────────
cells.append(cell('markdown', '## 2. Load CityFlowV2 Pretrained Weights'))

cells.append(cell('code', r"""PRETRAINED_PATH = None

search_paths = [
    # Kaggle Dataset: mrkdagods/mtmc-weights
    Path("/kaggle/input/datasets/mrkdagods/mtmc-weights/models/reid/transreid_cityflowv2_best.pth"),
    Path("/kaggle/input/mtmc-weights/models/reid/transreid_cityflowv2_best.pth"),
    # Local fallback
    PROJECT / "models/reid/transreid_cityflowv2_best.pth",
    Path("models/reid/transreid_cityflowv2_best.pth"),
]

for p in search_paths:
    if p.exists():
        PRETRAINED_PATH = p
        break

if PRETRAINED_PATH is not None:
    print(f"CityFlowV2 256px checkpoint: {PRETRAINED_PATH}")
    ckpt = torch.load(str(PRETRAINED_PATH), map_location="cpu", weights_only=False)
    source_state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    pos_embed_256 = source_state.get("vit.pos_embed", None)
    if pos_embed_256 is not None:
        print(f"  Source pos_embed: {pos_embed_256.shape}  (16x16 patch grid = 256px)")
else:
    source_state = None
    pos_embed_256 = None
    print("WARNING: CityFlowV2 checkpoint not found -- will train from CLIP init")"""))

# ── Cell 4: Download CityFlowV2 ────────────────────────────────────────────────
cells.append(cell('markdown', '## 3. Download CityFlowV2 Dataset'))

cells.append(cell('code', r"""import zipfile

CITYFLOW_DIR = Path("/tmp/cityflowv2")
GDRIVE_ID = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"
ARCHIVE_NAME = "AIC22_Track1_MTMC_Tracking.zip"
ALLOWED_SPLITS = {"train", "validation"}

already_found = False
for check_dir in [CITYFLOW_DIR,
                  Path("/kaggle/input/cityflowv2"),
                  Path("/kaggle/input/aic22-track1-mtmc-tracking")]:
    if check_dir.exists() and any(list(check_dir.rglob("vdo.avi"))[:1]):
        print(f"CityFlowV2 already at {check_dir}")
        CITYFLOW_DIR = check_dir
        already_found = True
        break

if not already_found:
    CITYFLOW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = Path(f"/tmp/{ARCHIVE_NAME}")
    if not archive_path.exists():
        print(f"Downloading CityFlowV2 (id={GDRIVE_ID})...")
        import gdown
        for _attempt in range(3):
            try:
                gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", str(archive_path), quiet=False)
                if archive_path.exists() and archive_path.stat().st_size > 1e9:
                    break
            except Exception as e:
                print(f"  Download attempt {_attempt+1} failed: {e}")
                import time as _t; _t.sleep(30)
        if not archive_path.exists():
            raise RuntimeError("Failed to download CityFlowV2 after 3 attempts")
    else:
        print(f"Using cached archive: {archive_path}")

    staging = Path("/tmp/_aic22_staging")
    staging.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {staging}...")
    with zipfile.ZipFile(str(archive_path), "r") as zf:
        zf.extractall(str(staging))
    if archive_path.exists():
        archive_path.unlink()
        print("Deleted archive (reclaim space)")

    moved = 0
    skipped_splits = set()
    for vdo_path in sorted(staging.rglob("vdo.avi")):
        cam_dir = vdo_path.parent
        scene_dir = cam_dir.parent
        split_dir = scene_dir.parent
        split_name = split_dir.name.lower()
        if split_name not in ALLOWED_SPLITS:
            skipped_splits.add(split_name)
            continue
        dst_cam = CITYFLOW_DIR / scene_dir.name / cam_dir.name
        dst_cam.mkdir(parents=True, exist_ok=True)
        for f in cam_dir.iterdir():
            if f.is_file():
                dst = dst_cam / f.name
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))
            elif f.is_dir() and f.name == "gt":
                if not (dst_cam / "gt").exists():
                    shutil.copytree(str(f), str(dst_cam / "gt"))
            # skip mtsc/ and other subdirs
        moved += 1
    print(f"Organized {moved} camera directories")
    if skipped_splits:
        print(f"Skipped splits: {skipped_splits}")
    shutil.rmtree(str(staging), ignore_errors=True)

cameras = sorted({
    p.parent.name
    for p in CITYFLOW_DIR.rglob("vdo.avi")
    if p.parent.parent.name.startswith("S") and p.parent.parent.parent == CITYFLOW_DIR
})
print(f"Found {len(cameras)} cameras: {cameras[:6]}...")"""))

# ── Cell 5: Extract crops ──────────────────────────────────────────────────────
cells.append(cell('markdown', '## 4. Extract Vehicle Crops (384×384)'))

cells.append(cell('code', r"""H, W = 384, 384
MAX_CROPS_PER_ID_PER_CAM = 32
MIN_AREA = 2500
MIN_BBOX_SIDE = 48
TRAIN_RATIO = 0.7
MIN_CAMS_FOR_EVAL = 2
CAMERAS = cameras

CROP_DIR = Path("/tmp/cityflowv2_crops_384")
CROP_DIR.mkdir(parents=True, exist_ok=True)


def load_gt(gt_path):
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


def extract_crops_from_camera(cam_name, gt_file, vid_file, crop_dir, max_crops, min_area):
    gt = load_gt(str(gt_file))
    if not gt:
        return {}
    id_dets = defaultdict(list)
    for frame, tid, x, y, w, h in gt:
        id_dets[tid].append((frame, x, y, w, h))
    frame_to_dets = defaultdict(list)
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
        return {}
    crops = defaultdict(list)
    cap = cv2.VideoCapture(str(vid_file))
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
            if target_idx >= len(target_frames) or current_frame != target_frames[target_idx]:
                continue
        H_img, W_img = img.shape[:2]
        for tid, x, y, bw, bh in frame_to_dets[current_frame]:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W_img, x + bw), min(H_img, y + bh)
            if x2 - x1 < MIN_BBOX_SIDE or y2 - y1 < MIN_BBOX_SIDE:
                continue
            crop = img[y1:y2, x1:x2]
            fname = f"{tid:04d}_{cam_name}_f{current_frame:06d}.jpg"
            fpath = crop_dir / fname
            cv2.imwrite(str(fpath), crop)
            crops[tid].append(fpath)
        while target_idx < len(target_frames) and target_frames[target_idx] == current_frame:
            target_idx += 1
    cap.release()
    return dict(crops)


all_crops = defaultdict(lambda: defaultdict(list))
total_crops = 0
missing = []

for scene_dir in sorted(CITYFLOW_DIR.iterdir()):
    if not scene_dir.is_dir() or not scene_dir.name.startswith("S"):
        continue
    for cam_dir in sorted(scene_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        cam_name = cam_dir.name
        if CAMERAS and cam_name not in CAMERAS:
            continue
        vid_file = cam_dir / "vdo.avi"
        gt_file = cam_dir / "gt" / "gt.txt"
        if not vid_file.exists() or not gt_file.exists():
            missing.append(cam_name)
            continue
        cam_crops = extract_crops_from_camera(
            cam_name, gt_file, vid_file, CROP_DIR,
            MAX_CROPS_PER_ID_PER_CAM, MIN_AREA
        )
        for tid, paths in cam_crops.items():
            all_crops[tid][cam_name].extend(paths)
            total_crops += len(paths)

print(f"Extracted {total_crops} crops from {sum(len(c) for c in all_crops.values())} tracklets across {len(all_crops)} IDs")
if missing:
    print(f"Missing cameras (skipped): {missing}")"""))

# ── Cell 6: Build splits ───────────────────────────────────────────────────────
cells.append(cell('markdown', '## 5. Build Train / Eval Splits'))

cells.append(cell('code', r"""if not all_crops:
    raise RuntimeError("No crops extracted! Check download and extraction.")

rng = np.random.RandomState(SEED)

multi_cam_ids = sorted(tid for tid, cams in all_crops.items() if len(cams) >= MIN_CAMS_FOR_EVAL)
single_cam_ids = sorted(tid for tid, cams in all_crops.items() if len(cams) < MIN_CAMS_FOR_EVAL)

if not multi_cam_ids:
    print("WARNING: No multi-camera IDs found -- using all IDs for training")
    all_ids = sorted(all_crops.keys())
    rng.shuffle(all_ids)
    multi_cam_ids = all_ids
    single_cam_ids = []

rng.shuffle(multi_cam_ids)
n_train = int(len(multi_cam_ids) * TRAIN_RATIO)
train_ids = set(multi_cam_ids[:n_train])
eval_ids = set(multi_cam_ids[n_train:])

cam_names = sorted({cam for cams in all_crops.values() for cam in cams})
cam2id = {c: i for i, c in enumerate(cam_names)}
num_cameras = len(cam_names)

train_pid_set = sorted(train_ids)
pid2label = {tid: i for i, tid in enumerate(train_pid_set)}
num_classes = len(train_pid_set)

train_data, query_data, gallery_data = [], [], []

for tid in sorted(train_ids):
    label = pid2label[tid]
    for cam_name, paths in all_crops[tid].items():
        camid = cam2id[cam_name]
        for p in paths:
            train_data.append((p, label, camid))

eval_pid2label = {tid: i for i, tid in enumerate(sorted(eval_ids))}
for tid in sorted(eval_ids):
    pid = eval_pid2label[tid]
    for cam_name, paths in all_crops[tid].items():
        if not paths:
            continue
        camid = cam2id[cam_name]
        idx = rng.randint(0, len(paths))
        query_data.append((paths[idx], pid, camid))
        for i, p in enumerate(paths):
            if i != idx:
                gallery_data.append((p, pid, camid))

distractor_pid = len(eval_ids)
for tid in sorted(single_cam_ids):
    for cam_name, paths in all_crops[tid].items():
        camid = cam2id[cam_name]
        for p in paths:
            gallery_data.append((p, distractor_pid, camid))
    distractor_pid += 1

print(f"Train: {len(train_data)} images, {num_classes} IDs, {num_cameras} cameras")
print(f"Query: {len(query_data)}, Gallery: {len(gallery_data)}")"""))

# ── Cell 7: DataLoaders ────────────────────────────────────────────────────────
cells.append(cell('markdown', '## 6. Data Loading + Losses'))

cells.append(cell('code', r"""CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

train_tf = T.Compose([
    T.Resize((H + 32, W + 32), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(12),
    T.RandomCrop((H, W)),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    T.RandomErasing(p=0.5, value="random"),
])

test_tf = T.Compose([
    T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


class ReIDDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, pid, cam = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, pid, cam, str(path)


class PKSampler(Sampler):
    def __init__(self, data_source, p=16, k=4):
        self.p, self.k = p, k
        self.pid2idx = defaultdict(list)
        for i, (_, pid, *_) in enumerate(data_source):
            self.pid2idx[pid].append(i)
        self.pids = list(self.pid2idx.keys())
        self.length = len(self.pids) * k

    def __iter__(self):
        rng = np.random.default_rng()
        pids = rng.permutation(self.pids).tolist()
        batches = []
        for start in range(0, len(pids) - self.p + 1, self.p):
            batch_pids = pids[start:start + self.p]
            batch = []
            for pid in batch_pids:
                indices = self.pid2idx[pid]
                batch.extend(rng.choice(indices, self.k, replace=len(indices) < self.k).tolist())
            batches.append(batch)
        rng.shuffle(batches)
        return iter([idx for batch in batches for idx in batch])

    def __len__(self):
        return self.length


BATCH_P = 8
BATCH_K = 4
batch_size = BATCH_P * BATCH_K * max(NUM_GPUS, 1)
train_loader = DataLoader(ReIDDataset(train_data, train_tf), batch_size=batch_size,
                          sampler=PKSampler(train_data, BATCH_P, BATCH_K),
                          num_workers=4, pin_memory=True)
query_loader = DataLoader(ReIDDataset(query_data, test_tf), batch_size=32,
                          shuffle=False, num_workers=4, pin_memory=True)
gallery_loader = DataLoader(ReIDDataset(gallery_data, test_tf), batch_size=32,
                            shuffle=False, num_workers=4, pin_memory=True)
print(f"Train loader: {len(train_loader)} batches/epoch (batch_size={batch_size})")"""))

# ── Cell 8: Losses ─────────────────────────────────────────────────────────────
cells.append(cell('code', r"""class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs.float(), dim=1)
        with torch.no_grad():
            oh = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            smooth = (1 - self.epsilon) * oh + self.epsilon / self.num_classes
        return (-smooth * log_probs).sum(dim=1).mean()


class TripletLossHardMining(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, pids):
        feats = F.normalize(feats.float(), p=2, dim=1)
        n = feats.size(0)
        dist = torch.cdist(feats, feats, p=2)
        mask_pos = pids.unsqueeze(0).eq(pids.unsqueeze(1))
        mask_neg = ~mask_pos
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = 0
        hardest_pos, _ = dist_pos.max(dim=1)
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = float("inf")
        hardest_neg, _ = dist_neg.min(dim=1)
        y = torch.ones(n, device=feats.device)
        return self.ranking_loss(hardest_neg, hardest_pos, y)


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feats, labels):
        feats = feats.float()
        c = self.centers[labels]
        return (feats - c).pow(2).sum(dim=1).mean() / 2.0


print("Losses defined: CrossEntropyLabelSmooth, TripletLossHardMining, CenterLoss")"""))

# ── Cell 9: Evaluation ─────────────────────────────────────────────────────────
cells.append(cell('markdown', '## 7. Evaluation'))

cells.append(cell('code', r"""@torch.no_grad()
def extract_features(model, loader, device="cuda", flip=True, pass_cams=False):
    model.eval()
    feats, pids, cams = [], [], []
    for imgs, pid, cam, _ in loader:
        imgs = imgs.to(device)
        kwargs = {"cam_ids": cam.to(device).long()} if pass_cams else {}
        f = model(imgs, **kwargs)
        if isinstance(f, (tuple, list)):
            f = f[-1]
        if flip:
            ff = model(torch.flip(imgs, [3]), **kwargs)
            if isinstance(ff, (tuple, list)):
                ff = ff[-1]
            f = (f + ff) / 2
        f = F.normalize(f, p=2, dim=1)
        feats.append(f.cpu().numpy())
        pids.append(pid.numpy())
        cams.append(cam.numpy())
    if not feats:
        return np.zeros((0, 768)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.concatenate(feats), np.concatenate(pids), np.concatenate(cams)


def eval_market1501(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=50):
    if distmat.shape[0] == 0 or distmat.shape[1] == 0:
        return 0.0, np.zeros(max_rank)
    nq = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)
    all_cmc, all_AP = [], []
    for qi in range(nq):
        order = indices[qi]
        remove = (g_pids[order] == q_pids[qi]) & (g_cams[order] == q_cams[qi])
        raw_cmc = matches[qi][~remove]
        if raw_cmc.sum() == 0:
            continue
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        n_rel = raw_cmc.sum()
        ap = 0.0
        for j in range(len(raw_cmc)):
            if j >= max_rank or raw_cmc[j] == 0:
                continue
            prec = raw_cmc[:j + 1].sum() / (j + 1)
            ap += prec / n_rel
        all_AP.append(ap)
    mAP = np.mean(all_AP) if all_AP else 0.0
    cmc = np.mean(all_cmc, axis=0) if all_cmc else np.zeros(max_rank)
    return mAP, cmc


def evaluate(model, device="cuda", pass_cams=False):
    qf, qp, qc = extract_features(model, query_loader, device, flip=True, pass_cams=pass_cams)
    gf, gp, gc = extract_features(model, gallery_loader, device, flip=True, pass_cams=pass_cams)
    if qf.shape[0] == 0 or gf.shape[0] == 0:
        return 0.0, 0.0
    distmat = 1.0 - (qf @ gf.T)
    mAP, cmc = eval_market1501(distmat, qp, gp, qc, gc)
    return mAP, float(cmc[0]) if len(cmc) > 0 else 0.0


print("Evaluation functions defined")"""))

# ── Cell 10: Model ─────────────────────────────────────────────────────────────
cells.append(cell('markdown', '## 8. TransReID Model (384px)'))

cells.append(cell('code', r"""import timm

VIT_MODEL = "vit_base_patch16_clip_224.openai"


def interpolate_pos_embed(pos_embed_src, model_384):
    '''Bicubic-interpolate pos_embed from source checkpoint (256px) to 384px model grid.

    Args:
        pos_embed_src: (1, 257, D) tensor from 256px checkpoint (16x16 patches + CLS)
        model_384: target model with 384px pos_embed (24x24 + CLS)

    Returns:
        (1, 577, D) interpolated pos_embed
    '''
    N_src = pos_embed_src.shape[1] - 1   # 256
    N_tgt = model_384.vit.pos_embed.shape[1] - 1  # 576
    D = pos_embed_src.shape[2]

    if N_src == N_tgt:
        return pos_embed_src

    cls_token = pos_embed_src[:, :1, :]   # (1, 1, D)
    patch_embed = pos_embed_src[:, 1:, :] # (1, N_src, D)

    h_src = w_src = int(round(N_src ** 0.5))   # 16
    h_tgt = w_tgt = int(round(N_tgt ** 0.5))   # 24

    assert h_src * w_src == N_src, f"Source pos_embed not square: {N_src}"
    assert h_tgt * w_tgt == N_tgt, f"Target pos_embed not square: {N_tgt}"

    # (1, N_src, D) -> (1, D, 16, 16) -> bicubic -> (1, D, 24, 24) -> (1, 576, D)
    patch_embed = patch_embed.reshape(1, h_src, w_src, D).permute(0, 3, 1, 2).float()
    patch_embed = F.interpolate(patch_embed, size=(h_tgt, w_tgt), mode="bicubic", align_corners=False)
    patch_embed = patch_embed.permute(0, 2, 3, 1).flatten(1, 2)

    return torch.cat([cls_token, patch_embed], dim=1)  # (1, 577, D)


class TransReID384(nn.Module):
    '''TransReID ViT-Base/16 at 384x384 resolution with SIE, JPM, BNNeck.'''

    def __init__(self, num_classes, num_cameras=0, embed_dim=768,
                 vit_model=VIT_MODEL, pretrained=True, sie_camera=True, jpm=True):
        super().__init__()
        self.sie_camera = sie_camera and num_cameras > 0
        self.jpm = jpm
        # img_size=384 forces timm to create 24x24 grid pos_embed (576 patches)
        self.vit = timm.create_model(vit_model, pretrained=pretrained,
                                     num_classes=0, img_size=384)
        self.vit_dim = self.vit.embed_dim
        self.num_blocks = len(self.vit.blocks)

        if self.sie_camera:
            self.sie_embed = nn.Parameter(torch.zeros(num_cameras, 1, self.vit_dim))
            nn.init.trunc_normal_(self.sie_embed, std=0.02)

        self.bn = nn.BatchNorm1d(self.vit_dim)
        self.bn.bias.requires_grad_(False)

        self.proj = (nn.Linear(self.vit_dim, embed_dim, bias=False)
                     if embed_dim != self.vit_dim else nn.Identity())
        self.cls_head = nn.Linear(embed_dim, num_classes, bias=False)
        if isinstance(self.proj, nn.Linear):
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")
        nn.init.normal_(self.cls_head.weight, std=0.001)

        if self.jpm:
            self.bn_jpm = nn.BatchNorm1d(self.vit_dim)
            self.bn_jpm.bias.requires_grad_(False)
            self.jpm_cls = nn.Linear(self.vit_dim, num_classes, bias=False)
            nn.init.normal_(self.jpm_cls.weight, std=0.001)

    def forward(self, x, cam_ids=None, return_feat=False):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        if hasattr(self.vit, "_pos_embed"):
            x = self.vit._pos_embed(x)
        else:
            cls_tok = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tok, x], dim=1) + self.vit.pos_embed
            if hasattr(self.vit, "pos_drop"):
                x = self.vit.pos_drop(x)
        if self.sie_camera and cam_ids is not None:
            x = x + self.sie_embed[cam_ids]
        if hasattr(self.vit, "patch_drop"):
            x = self.vit.patch_drop(x)
        if hasattr(self.vit, "norm_pre"):
            x = self.vit.norm_pre(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        cls = x[:, 0]
        feat_bn = self.bn(cls)
        feat_proj = self.proj(feat_bn)

        if self.training:
            logits = self.cls_head(feat_proj)
            if self.jpm:
                patches = x[:, 1:]
                N = patches.shape[1]
                half = N // 2
                local_feat = patches[:, :half].mean(dim=1) + patches[:, half:].mean(dim=1)
                local_feat_bn = self.bn_jpm(local_feat)
                jpm_logits = self.jpm_cls(local_feat_bn)
                return feat_proj, logits, jpm_logits, local_feat_bn
            return feat_proj, logits, None, None

        if return_feat:
            return feat_proj, feat_bn
        return feat_proj

    def get_llrd_param_groups(self, backbone_lr=1e-5, head_lr=1e-4, decay=0.75):
        param_groups = []
        n = self.num_blocks
        for i, blk in enumerate(self.vit.blocks):
            lr_i = backbone_lr * (decay ** (n - 1 - i))
            param_groups.append({"params": list(blk.parameters()), "lr": lr_i})
        embed_params = []
        for attr in ["patch_embed", "cls_token", "pos_embed"]:
            obj = getattr(self.vit, attr, None)
            if obj is None:
                continue
            if isinstance(obj, nn.Module):
                embed_params.extend(list(obj.parameters()))
            elif isinstance(obj, nn.Parameter):
                embed_params.append(obj)
        if embed_params:
            param_groups.append({"params": embed_params, "lr": backbone_lr * decay})
        head_params = []
        for m in [self.bn, self.proj, self.cls_head]:
            head_params.extend(list(m.parameters()))
        if self.sie_camera:
            head_params.append(self.sie_embed)
        if self.jpm:
            for m in [self.bn_jpm, self.jpm_cls]:
                head_params.extend(list(m.parameters()))
        param_groups.append({"params": head_params, "lr": head_lr})
        return param_groups


model = TransReID384(num_classes=num_classes, num_cameras=num_cameras,
                     embed_dim=768, vit_model=VIT_MODEL, pretrained=True,
                     sie_camera=True, jpm=True)

print(f"TransReID384 created: pos_embed={model.vit.pos_embed.shape}  (24x24 grid for 384px)")
print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")"""))

# ── Cell 11: Load checkpoint ───────────────────────────────────────────────────
cells.append(cell('code', r"""# ── Load CityFlowV2 256px weights with bicubic pos_embed interpolation ──────────
if source_state is not None:
    skip_keys = ["cls_head", "jpm_cls"]  # re-init — new num_classes/cameras
    filtered = {}
    skipped = []
    interpolated = []

    for k, v in source_state.items():
        if any(sk in k for sk in skip_keys):
            skipped.append(k)
            continue
        if k == "vit.pos_embed":
            v_interp = interpolate_pos_embed(v, model)
            if v_interp.shape != model.vit.pos_embed.shape:
                print(f"  WARNING: interpolated {v_interp.shape} != model {model.vit.pos_embed.shape}")
                skipped.append(k)
            else:
                filtered[k] = v_interp
                interpolated.append(k)
            continue
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape:
            filtered[k] = v
        else:
            reason = "shape_mismatch" if k in model.state_dict() else "not_in_model"
            skipped.append(f"{k}[{reason}]")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"  Loaded {len(filtered)} tensors from CityFlowV2 256px checkpoint")
    print(f"  Interpolated pos_embed: {interpolated}")
    new_keys = [k for k in missing if not any(sk in k for sk in skip_keys)]
    if new_keys:
        print(f"  New (random init): {new_keys[:8]}")
    print(f"  Final pos_embed: {model.vit.pos_embed.shape}")
else:
    print("No pretrained weights -- training from timm CLIP initialization")

if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    print(f"Wrapped in DataParallel ({NUM_GPUS} GPUs)")
model = model.to(DEVICE)
print(f"Model on {DEVICE}")"""))

# ── Cell 12: Training ──────────────────────────────────────────────────────────
cells.append(cell('markdown', '## 9. Fine-Tuning at 384px'))

cells.append(cell('code', r"""ce_loss = CrossEntropyLabelSmooth(num_classes, 0.05).to(DEVICE)
tri_loss = TripletLossHardMining(margin=0.3).to(DEVICE)
ctr_loss = CenterLoss(num_classes, 768).to(DEVICE)
CENTER_WEIGHT = 5e-4
CENTER_START = 5   # start center loss earlier (pre-trained baseline)

raw_model = model.module if hasattr(model, "module") else model

backbone_lr = 1e-5   # 10x lower than NB09 (continued fine-tuning)
head_lr = 1e-4       # 10x lower than NB09
wd = 5e-4
llrd_factor = 0.75

param_groups = raw_model.get_llrd_param_groups(backbone_lr, head_lr, decay=llrd_factor)
optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
center_optimizer = torch.optim.SGD(ctr_loss.parameters(), lr=0.5)
base_lrs = [pg["lr"] for pg in optimizer.param_groups]

EPOCHS = 40
WARMUP = 5

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP)
scaler = torch.amp.GradScaler("cuda")

history = {"loss": [], "mAP": [], "R1": []}
best_mAP = 0.0
best_state_path = OUTPUT_DIR / "transreid_cityflowv2_384_best.pth"

print("=" * 70)
print(f"  Fine-tuning TransReID 384px on CityFlowV2 ({num_classes} IDs, {num_cameras} cams)")
print(f"  Init: CityFlowV2 256px checkpoint (pos_embed bicubic 16x16->24x24)")
print(f"  Losses: CE(eps=0.05) + Triplet(m=0.3) + Center(5e-4, from ep{CENTER_START})")
print(f"  LR: backbone={backbone_lr}, head={head_lr}, LLRD={llrd_factor}")
print(f"  Epochs: {EPOCHS}, warmup: {WARMUP}, batch_size: {batch_size}, {H}x{W}")
print("=" * 70)

t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    running_loss, num_batches = 0.0, 0

    # Linear warmup
    if epoch < WARMUP:
        warmup_factor = (epoch + 1) / WARMUP
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = base_lrs[i] * warmup_factor

    for imgs, pids, cams, _ in train_loader:
        imgs = imgs.to(DEVICE)
        pids = pids.to(DEVICE)
        cams_t = cams.to(DEVICE).long()

        with torch.amp.autocast("cuda"):
            out = model(imgs, cam_ids=cams_t)
            if isinstance(out, (tuple, list)):
                feat, logits, logits_jpm, local_feat = (list(out) + [None] * 4)[:4]
            else:
                feat, logits, logits_jpm, local_feat = out, None, None, None

            loss = torch.tensor(0.0, device=DEVICE)
            if logits is not None:
                loss = loss + ce_loss(logits, pids)
                if logits_jpm is not None:
                    loss = loss + 0.5 * ce_loss(logits_jpm, pids)
            if feat is not None:
                loss = loss + tri_loss(feat, pids)
            if epoch >= CENTER_START and feat is not None:
                loss = loss + CENTER_WEIGHT * ctr_loss(feat.float(), pids)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if epoch >= CENTER_START:
            center_optimizer.step()
            center_optimizer.zero_grad()

        running_loss += loss.item()
        num_batches += 1

    if epoch >= WARMUP:
        scheduler.step()

    history["loss"].append(running_loss / max(num_batches, 1))
    elapsed = (time.time() - t0) / 60

    # Evaluate every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        mAP, r1 = evaluate(model, DEVICE, pass_cams=True)
        history["mAP"].append(mAP)
        history["R1"].append(r1)
        print(f"[Ep {epoch+1:3d}/{EPOCHS}] loss={running_loss/num_batches:.4f}  mAP={mAP:.4f}  R1={r1:.4f}  ({elapsed:.1f}min)")
        if mAP > best_mAP:
            best_mAP = mAP
            raw_m = model.module if hasattr(model, "module") else model
            torch.save(raw_m.state_dict(), str(best_state_path))
            print(f"  *** New best mAP={best_mAP:.4f} -> {best_state_path.name}")
    else:
        print(f"[Ep {epoch+1:3d}/{EPOCHS}] loss={running_loss/num_batches:.4f}  ({elapsed:.1f}min)", end="\r")

print(f"\nTraining complete: best mAP={best_mAP:.4f}")"""))

# ── Cell 13: Training curves ───────────────────────────────────────────────────
cells.append(cell('code', r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["loss"], label="train loss")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(range(4, EPOCHS + 1, 5), history["mAP"], "b-o", label="mAP")
axes[1].plot(range(4, EPOCHS + 1, 5), history["R1"], "r-s", label="R1")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("metric"); axes[1].legend(); axes[1].grid(True)
axes[0].set_title(f"384px Fine-tuning Loss (best mAP={best_mAP:.4f})")
axes[1].set_title("384px mAP / R1 per eval")
plt.tight_layout()
curves_path = EXPORT_DIR / "training_curves_384.png"
plt.savefig(str(curves_path), dpi=150)
plt.close()
print(f"Saved: {curves_path}")"""))

# ── Cell 14: Export ────────────────────────────────────────────────────────────
cells.append(cell('markdown', '## 10. Export Model'))

cells.append(cell('code', r"""best_state = torch.load(str(best_state_path), map_location="cpu", weights_only=False)
export_path = EXPORT_DIR / "transreid_cityflowv2_384_best.pth"
torch.save({"state_dict": best_state}, str(export_path))
print(f"Model exported: {export_path}  ({export_path.stat().st_size / 1e6:.1f} MB)")

metadata = {
    "task": "vehicle_reid",
    "dataset": "cityflowv2",
    "source_dataset": "AI City Challenge 2022 Track 1",
    "init_weights": "transreid_cityflowv2_best.pth (256px, bicubic pos_embed interp 16x16->24x24)",
    "training": f"TransReID ViT-Base CLIP -> CityFlowV2 256px -> 384px fine-tune ({EPOCHS} epochs)",
    "model": {
        "architecture": VIT_MODEL,
        "type": "transreid",
        "tricks": [
            "SIE", "JPM", "BNNeck",
            "CE+LS(0.05)", "TripletLoss(m=0.3)", f"CenterLoss(5e-4, from ep{CENTER_START})",
            "CosineAnnealingLR", "RandomErasing", "CLIP-norm",
            f"LLRD(decay={llrd_factor})", "PosEmbedInterp(bicubic,16x16->24x24)",
        ],
        "embedding_dim": 768,
        "input_size": [H, W],
        "normalization": {"mean": CLIP_MEAN, "std": CLIP_STD},
        "num_cameras": num_cameras,
        "num_classes": num_classes,
        "best_mAP": float(best_mAP),
        "epochs": EPOCHS,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
    },
}

meta_path = EXPORT_DIR / "transreid_cityflowv2_384_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata: {meta_path}")
print()
print("Next steps:")
print("  1. Upload transreid_cityflowv2_384_best.pth to mrkdagods/mtmc-weights dataset")
print("  2. Update notebook 10a to load transreid_cityflowv2_384_best.pth")
print("  3. Compare IDF1 with 256px model to measure Phase 3 improvement")"""))

# ── Build notebook JSON ────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out = Path("notebooks/kaggle/09b_vehicle_reid_384px/09b_vehicle_reid_384px.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=True, indent=1)

# Validate
nb2 = json.load(open(out, encoding="utf-8"))
code_cells = [c for c in nb2["cells"] if c["cell_type"] == "code"]
md_cells = [c for c in nb2["cells"] if c["cell_type"] == "markdown"]
print(f"Written: {out}")
print(f"  Total cells: {len(nb2['cells'])}  (code={len(code_cells)}, markdown={len(md_cells)})")
