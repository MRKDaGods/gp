"""Generate NB07 v2: Person ReID TransReID CLIP with all v15 fixes.
Single-model focused notebook (drops BoT, applies vehicle v15 approach)."""
import json, uuid

def make_cell(cell_type, source):
    lines = source.split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cell = {
        "id": uuid.uuid4().hex[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": src,
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell

cells = []

# ═══════════════════════════════════════════════════════════════════
# Cell 0: Header
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """# Notebook 07: Person ReID — TransReID CLIP Training (v2)
**Multi-Camera Tracking System — Kaggle Training Pipeline**

Train SOTA person re-identification on Market-1501 using TransReID ViT-Base with CLIP pretraining.

## Model
| Model | Architecture | Dim | Target mAP | Target R1 |
|-------|-------------|-----|-------------|-------------|
| **TransReID** | ViT-Base/16 (CLIP) + SIE + JPM | 768 | ≥89% | ≥95% |

## v2: All fixes from Vehicle v15 (82.2% mAP — beat SOTA)
- **norm_pre**: CLIP ViTs pre-LayerNorm (critical for CLIP models)
- **SIE on ALL tokens**: Camera embedding broadcast to all patches
- **LLRD (0.75)**: Layer-wise LR decay preserves pretrained CLIP features
- **BNNeck routing**: Triplet+Center get pre-BN features, CE gets post-BN
- **Center loss**: Delayed start at epoch 30 (stability)
- **CLIP normalization**: OpenAI CLIP mean/std
- **TTA**: Horizontal flip averaging at evaluation

## v1 Issues Fixed
| Bug | v1 | v2 (fix) |
|-----|-----|---------|
| norm_pre | Missing | ✓ CLIP LayerNorm applied |
| SIE scope | CLS token only | ✓ All tokens (original TransReID) |
| LLRD | No layer-wise decay | ✓ decay=0.75 across 12 blocks |
| BNNeck routing | Triplet gets post-BN | ✓ Triplet gets pre-BN |
| Center loss | None | ✓ 5e-4, delayed@ep30 |
| CLIP LR | backbone=1e-5 (too low) | ✓ 3.5e-4 with LLRD |
| Normalization | ImageNet constants | ✓ CLIP constants |

**Runtime**: GPU T4 x2 (32GB) | **Duration**: ~2.5h"""))

# ═══════════════════════════════════════════════════════════════════
# Cell 1: pip install
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", "!pip install -q timm matplotlib pandas"))

# ═══════════════════════════════════════════════════════════════════
# Cell 2: Setup markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 1. Setup & Configuration"))

# ═══════════════════════════════════════════════════════════════════
# Cell 3: Imports + config
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''import os
import sys
import json
import time
import math
import shutil
import copy
import re
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
NUM_GPUS = max(torch.cuda.device_count(), 1)
print(f"GPUs available: {NUM_GPUS}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.1f} GB)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/kaggle/working/person_reid_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR = Path("/kaggle/working/exported_models")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nDevice: {DEVICE} | GPUs: {NUM_GPUS}")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 4: Dataset markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 2. Dataset (Market-1501)"))

# ═══════════════════════════════════════════════════════════════════
# Cell 5: Locate Market-1501
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''# ── Locate Market-1501 from Kaggle input ──
DATA_ROOT = Path("/kaggle/working/data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

possible_roots = [
    Path("/kaggle/input/datasets/pengcw1/market-1501"),
    Path("/kaggle/input/market-1501"),
]

input_dir = None
for root in possible_roots:
    if root.exists():
        input_dir = root
        break

if input_dir is None:
    import subprocess
    result = subprocess.run(
        ["find", "/kaggle/input", "-maxdepth", "5", "-type", "d", "-name", "bounding_box_train"],
        capture_output=True, text=True, timeout=30
    )
    print(f"Search results:\n{result.stdout}")
    if result.stdout.strip():
        found = Path(result.stdout.strip().split("\n")[0]).parent
        input_dir = found.parent if found.name == "bounding_box_train" else found
    else:
        raise RuntimeError("Market-1501 not found. Attach dataset 'pengcw1/market-1501'.")

print(f"Dataset found at: {input_dir}")

# Find actual data directory (may be nested)
market_data = None
for p in input_dir.rglob("bounding_box_train"):
    if p.is_dir():
        market_data = p.parent
        break

if market_data is None:
    raise RuntimeError(f"Could not find 'bounding_box_train' in {input_dir}")

# Symlink to writable location
symlink = DATA_ROOT / "market1501"
if symlink.exists() or symlink.is_symlink():
    if symlink.is_symlink():
        symlink.unlink()
    else:
        shutil.rmtree(symlink)
symlink.symlink_to(market_data)

# Verify splits
for subdir in ["bounding_box_train", "bounding_box_test", "query"]:
    d = symlink / subdir
    n = len(list(d.glob("*.jpg"))) if d.exists() else 0
    print(f"  {subdir}: {n} images")

print(f"\nData ready at: {symlink}")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 6: Parse Market-1501
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''# ── Parse Market-1501 ──
def parse_market1501(root):
    """Parse Market-1501 into (path, pid, camid) tuples."""
    train, query, gallery = [], [], []
    for split_name, split_list in [
        ("bounding_box_train", train),
        ("query", query),
        ("bounding_box_test", gallery),
    ]:
        split_dir = os.path.join(root, split_name)
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".jpg"):
                continue
            pid = int(fname.split("_")[0])
            if pid < 0:  # junk
                continue
            cam = int(fname.split("_")[1][1]) - 1  # 0-indexed
            split_list.append((os.path.join(split_dir, fname), pid, cam))

    # Re-label train PIDs to 0..N-1
    train_pids = sorted(set(pid for _, pid, _ in train))
    pid2label = {pid: i for i, pid in enumerate(train_pids)}
    train = [(p, pid2label[pid], c) for p, pid, c in train]
    return train, query, gallery

train_data, query_data, gallery_data = parse_market1501(str(symlink))
num_classes = len(set(pid for _, pid, _ in train_data))
num_cameras = len(set(cam for _, _, cam in train_data))

print(f"Train: {len(train_data)} images, {num_classes} IDs, {num_cameras} cameras")
print(f"Query: {len(query_data)} images")
print(f"Gallery: {len(gallery_data)} images")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 7: Data loading markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 3. Data Loading (PK Sampler + BoT Augmentation)"))

# ═══════════════════════════════════════════════════════════════════
# Cell 8: Transforms + datasets + loaders
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''# -- CLIP normalization constants --
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
H, W = 256, 128  # Standard person ReID resolution (tall & thin)

# -- Transforms (BoT recipe — v15 proven) --
# Random Erasing (p=0.5, random fill) — proven in TransReID paper.
train_tf = T.Compose([
    T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((H, W)),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    T.RandomErasing(p=0.5, value="random"),
])

test_tf = T.Compose([
    T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

print(f"Using CLIP normalization: mean={CLIP_MEAN}")
print(f"Input size: {H}×{W} (standard person ReID)")
print("Augmentation: HFlip + Pad+Crop + RandomErasing(p=0.5, random)")

# -- Dataset + PK Sampler --
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
        return img, pid, cam, path

class PKSampler(Sampler):
    def __init__(self, data_source, p=16, k=4):
        self.p, self.k = p, k
        self.pid_to_idx = defaultdict(list)
        for i, (_, pid, _) in enumerate(data_source):
            self.pid_to_idx[pid].append(i)
        self.pids = list(self.pid_to_idx.keys())
        self.batch_size = p * k
    def __iter__(self):
        np.random.shuffle(self.pids)
        batch = []
        for pid in self.pids:
            idxs = self.pid_to_idx[pid]
            sel = np.random.choice(idxs, self.k, replace=len(idxs) < self.k).tolist()
            batch.extend(sel)
            if len(batch) >= self.batch_size:
                yield from batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield from batch
    def __len__(self):
        return len(self.pids) * self.k

# -- Build loaders --
# Person images (256×128) are smaller than vehicle (224×224), can use larger batch
BATCH = 48 * NUM_GPUS  # 96 on 2xT4
P_IDS = 12 * NUM_GPUS  # 24 IDs per batch

train_ds = ReIDDataset(train_data, train_tf)
query_ds = ReIDDataset(query_data, test_tf)
gallery_ds = ReIDDataset(gallery_data, test_tf)

sampler = PKSampler(train_data, p=P_IDS, k=4)
train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                          num_workers=4, pin_memory=True, drop_last=True)

query_loader = DataLoader(query_ds, batch_size=128, num_workers=4, pin_memory=True)
gallery_loader = DataLoader(gallery_ds, batch_size=128, num_workers=4, pin_memory=True)
print(f"Train: {len(train_loader)} batches (batch={BATCH}) | Query: {len(query_loader)} | Gallery: {len(gallery_loader)}")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 9: Losses markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 4. Loss Functions"))

# ═══════════════════════════════════════════════════════════════════
# Cell 10: Losses
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''# ── Losses (numerically stable for fp16 + DataParallel) ──
class CrossEntropyLabelSmooth(nn.Module):
    """Label-smooth CE using F.log_softmax in fp32 for numerical stability."""
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # Force fp32 for log_softmax to prevent overflow
        log_probs = F.log_softmax(inputs.float(), dim=1)
        with torch.no_grad():
            oh = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            smooth = (1 - self.epsilon) * oh + self.epsilon / self.num_classes
        loss = (-smooth * log_probs).sum(dim=1).mean()
        return loss


class TripletLossHardMining(nn.Module):
    """Triplet loss with batch hard mining — fp32 computation.
    
    For each anchor, find hardest positive (max dist) and hardest negative
    (min dist) within the batch. Standard in TransReID / BoT.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, pids):
        feats = F.normalize(feats.float(), p=2, dim=1)
        n = feats.size(0)
        dist = torch.cdist(feats, feats, p=2)  # (n, n) Euclidean
        
        mask_pos = pids.unsqueeze(0).eq(pids.unsqueeze(1))
        mask_neg = ~mask_pos
        
        # For each anchor: hardest positive (largest distance)
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = 0
        hardest_pos, _ = dist_pos.max(dim=1)
        
        # For each anchor: hardest negative (smallest distance)
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = float('inf')
        hardest_neg, _ = dist_neg.min(dim=1)
        
        y = torch.ones(n, device=feats.device)
        return self.ranking_loss(hardest_neg, hardest_pos, y)


class CenterLoss(nn.Module):
    """Center loss (Wen et al., ECCV 2016) — fp32 computation."""
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feats, labels):
        feats = feats.float()
        batch_size = feats.size(0)
        centers_batch = self.centers[labels]  # (B, D)
        diff = feats - centers_batch
        loss = (diff * diff).sum(1).mean()
        return loss


print("Losses defined: CE+LS(ε=0.1), TripletLoss(m=0.3), CenterLoss (fp32-stable)")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 11: Eval markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 5. Evaluation Functions (mAP + CMC + Re-Ranking)"))

# ═══════════════════════════════════════════════════════════════════
# Cell 12: Eval functions
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''@torch.no_grad()
def extract_features(model, loader, device="cuda", flip=True, pass_cams=False):
    """Extract L2-normalized features. pass_cams=True for TransReID SIE."""
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
    return np.concatenate(feats), np.concatenate(pids), np.concatenate(cams)

def eval_market1501(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=50):
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
        cmc = raw_cmc.cumsum(); cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        n_rel = raw_cmc.sum()
        tmp = raw_cmc.cumsum()
        prec = tmp / (np.arange(len(tmp)) + 1.0)
        all_AP.append((prec * raw_cmc).sum() / n_rel)
    return float(np.mean(all_AP)), np.array(all_cmc).mean(0)

def compute_reranking(qf, gf, k1=20, k2=6, lam=0.3):
    all_f = np.concatenate([qf, gf], axis=0)
    N, nq = all_f.shape[0], qf.shape[0]
    od = np.clip(2.0 - 2.0 * (all_f @ all_f.T), 0, None)
    ir = np.argsort(od, axis=1)
    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        fwd = ir[i, :k1+1]
        kr = [c for c in fwd if i in ir[c, :k1+1]]
        kr = np.array(kr); kr_exp = kr.copy()
        for c in kr:
            cf = ir[c, :int(round(k1/2))+1]
            ckr = [cc for cc in cf if c in ir[cc, :int(round(k1/2))+1]]
            if len(ckr) > 2/3 * len(cf):
                kr_exp = np.union1d(kr_exp, ckr)
        w = np.exp(-od[i, kr_exp])
        V[i, kr_exp] = w / (w.sum() + 1e-12)
    if k2 > 0:
        Vqe = np.zeros_like(V)
        for i in range(N):
            Vqe[i] = V[ir[i, :k2+1]].mean(0)
        V = Vqe
    jac = np.zeros((nq, N-nq), dtype=np.float32)
    for i in range(nq):
        mn = np.minimum(V[i], V[nq:]); mx = np.maximum(V[i], V[nq:])
        jac[i] = 1 - mn.sum(1) / (mx.sum(1) + 1e-12)
    return jac * (1 - lam) + od[:nq, nq:] * lam

def full_eval(model, ql, gl, device="cuda", rerank=True, pass_cams=False):
    qf, qp, qc = extract_features(model, ql, device, pass_cams=pass_cams)
    gf, gp, gc = extract_features(model, gl, device, pass_cams=pass_cams)
    dist = 1.0 - qf @ gf.T
    mAP, cmc = eval_market1501(dist, qp, gp, qc, gc)
    mAP_rr, cmc_rr = None, None
    if rerank:
        dist_rr = compute_reranking(qf, gf)
        mAP_rr, cmc_rr = eval_market1501(dist_rr, qp, gp, qc, gc)
    return mAP, cmc, mAP_rr, cmc_rr

print("Evaluation ready (with SIE-aware feature extraction + TTA)")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 13: Model markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """## 6. TransReID Model

Camera-aware SIE for Market-1501 (6 cameras with different viewpoints/lighting).

**Critical fixes from Vehicle v15 (82.2% mAP on VeRi — beat SOTA):**
- **norm_pre**: CLIP ViTs have a pre-LayerNorm before transformer blocks
- **SIE on ALL tokens**: Camera embedding broadcast to all patch tokens
- **LLRD (0.75)**: Layer-wise LR Decay protects shallow CLIP features
- **BNNeck routing**: CE → post-BN features, Triplet/Center → pre-BN features

**Person ReID specifics:**
- Input 256×128 (standard person resolution — tall & thin)
- timm interpolates pos embeddings from 14×14 → 16×8 automatically
- 751 training IDs (more than VeRi's 576) — helps discriminative learning"""))

# ═══════════════════════════════════════════════════════════════════
# Cell 14: TransReID model
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''import timm

# ── Select CLIP ViT-Base backbone ──
def find_clip_vit_base():
    """Find best CLIP ViT-Base/16 in timm — prefer pure OpenAI CLIP."""
    try:
        tags = timm.list_pretrained('vit_base_patch16_clip*224*')
    except Exception:
        tags = []
    print(f"Available CLIP pretrained tags: {tags}")

    # Prefer pure OpenAI CLIP (what CLIP-ReID paper uses)
    for t in sorted(tags):
        if 'openai' in t and 'ft' not in t:
            return t
    for t in sorted(tags):
        if 'openai' in t:
            return t
    for t in sorted(tags):
        if 'laion' in t:
            return t
    if tags:
        return sorted(tags)[0]

    clip_models = timm.list_models("*vit_base*patch16*clip*224*")
    if clip_models:
        return sorted(clip_models)[0]
    clip_models = timm.list_models("*vit_base*patch16*clip*")
    if clip_models:
        return sorted(clip_models)[0]
    raise RuntimeError("No CLIP ViT-Base found in timm! Update timm.")

VIT_MODEL = find_clip_vit_base()
print(f"Selected backbone: {VIT_MODEL}")


class TransReID(nn.Module):
    """TransReID: ViT-Base + SIE + JPM (He et al., ICCV 2021).
    
    All fixes from Vehicle v15 (82.2% mAP — beat SOTA):
    - norm_pre: CLIP ViTs pre-LayerNorm
    - SIE broadcast to ALL tokens
    - BNNeck routing: pre-BN for triplet/center, post-BN for CE/inference
    """
    def __init__(self, num_classes, num_cameras=0, embed_dim=768,
                 vit_model="vit_base_patch16_clip_224", pretrained=True,
                 sie_camera=True, jpm=True, img_size=(256, 128)):
        super().__init__()
        self.sie_camera = sie_camera and num_cameras > 0
        self.jpm = jpm
        # Pass img_size so timm creates correct patch_embed and interpolates
        # pos_embed for 256x128 person images (16x8 grid instead of 14x14)
        self.vit = timm.create_model(vit_model, pretrained=pretrained,
                                     num_classes=0, img_size=img_size)
        self.vit_dim = self.vit.embed_dim  # 768 for ViT-Base
        cfg = getattr(self.vit, 'pretrained_cfg', {})
        print(f"  Pretrained: {cfg.get('hf_hub_id', cfg.get('url', 'unknown'))[:80]}")
        grid = self.vit.patch_embed.grid_size
        print(f"  Patch grid: {grid[0]}x{grid[1]} = {grid[0]*grid[1]} patches (img_size={img_size})")

        # Detect architecture features
        has_norm_pre = hasattr(self.vit, 'norm_pre') and not isinstance(
            self.vit.norm_pre, nn.Identity)
        print(f"  norm_pre: {type(self.vit.norm_pre).__name__} (active={has_norm_pre})")
        self.num_blocks = len(self.vit.blocks)
        print(f"  Transformer blocks: {self.num_blocks}")

        # SIE: camera embedding broadcast to ALL tokens (original TransReID)
        if self.sie_camera:
            self.sie_embed = nn.Parameter(torch.zeros(num_cameras, 1, self.vit_dim))
            nn.init.trunc_normal_(self.sie_embed, std=0.02)
            print(f"  SIE: {num_cameras} cameras, broadcast to all tokens")

        # BNNeck
        self.bn = nn.BatchNorm1d(self.vit_dim)
        self.bn.bias.requires_grad_(False)

        # Projection + classifier
        self.proj = nn.Linear(self.vit_dim, embed_dim, bias=False) if embed_dim != self.vit_dim else nn.Identity()
        self.cls_head = nn.Linear(embed_dim, num_classes, bias=False)
        if isinstance(self.proj, nn.Linear):
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")
        nn.init.normal_(self.cls_head.weight, std=0.001)

        # JPM: jigsaw patch module for part-level features
        if self.jpm:
            self.bn_jpm = nn.BatchNorm1d(self.vit_dim)
            self.bn_jpm.bias.requires_grad_(False)
            self.jpm_cls = nn.Linear(self.vit_dim, num_classes, bias=False)
            nn.init.normal_(self.jpm_cls.weight, std=0.001)

        print(f"TransReID: {vit_model}, dim={self.vit_dim}, embed={embed_dim}, "
              f"SIE={self.sie_camera}({num_cameras}), JPM={jpm}, blocks={self.num_blocks}")

    def forward(self, x, cam_ids=None):
        B = x.shape[0]
        # 1. Patch embedding
        x = self.vit.patch_embed(x)

        # 2. CLS token + positional embedding + pos_drop (use timm's method)
        if hasattr(self.vit, '_pos_embed'):
            x = self.vit._pos_embed(x)
        else:
            cls_tok = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tok, x], dim=1) + self.vit.pos_embed
            if hasattr(self.vit, 'pos_drop'):
                x = self.vit.pos_drop(x)

        # 3. SIE: camera embedding broadcast to ALL tokens
        if self.sie_camera and cam_ids is not None:
            x = x + self.sie_embed[cam_ids]  # (B,1,D) broadcasts to (B,N+1,D)

        # 4. Patch drop (Identity for most models)
        if hasattr(self.vit, 'patch_drop'):
            x = self.vit.patch_drop(x)

        # 5. CRITICAL: Pre-normalization (CLIP uses LayerNorm here!)
        if hasattr(self.vit, 'norm_pre'):
            x = self.vit.norm_pre(x)

        # 6. Transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)

        # Global feature (CLS token)
        g = x[:, 0]  # raw features (pre-BN) — for triplet + center loss
        bn = self.bn(g)  # BNNeck — for CE classifier + inference
        proj = self.proj(bn)

        if self.training:
            cls = self.cls_head(proj)  # CE gets post-BN features
            if self.jpm:
                patches = x[:, 1:]
                idx = torch.randperm(patches.size(1), device=x.device)
                s = patches[:, idx]
                mid = s.size(1) // 2
                jf = (s[:, :mid].mean(1) + s[:, mid:].mean(1)) / 2
                return cls, g, self.jpm_cls(self.bn_jpm(jf))  # g (pre-BN) for triplet
            return cls, g  # g (pre-BN) for triplet+center
        return F.normalize(proj, p=2, dim=1)  # inference: post-BN

    def get_llrd_param_groups(self, backbone_lr, head_lr, decay=0.75):
        """Layer-wise LR decay: shallow layers get exponentially smaller LR."""
        groups = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('vit.'):
                if 'blocks.' in name:
                    block_idx = int(name.split('blocks.')[1].split('.')[0])
                    depth = block_idx + 1
                elif any(k in name for k in ['patch_embed', 'cls_token', 'pos_embed', 'norm_pre']):
                    depth = 0
                else:
                    depth = self.num_blocks + 1
                scale = decay ** (self.num_blocks + 1 - depth)
                lr = backbone_lr * scale
                gk = f"bb_d{depth}"
            else:
                lr = head_lr
                gk = "head"
            if gk not in groups:
                groups[gk] = {"params": [], "lr": lr}
            groups[gk]["params"].append(param)
        result = sorted(groups.values(), key=lambda x: x["lr"])
        for g in result:
            n = sum(p.numel() for p in g["params"])
            print(f"  lr={g['lr']:.2e} | {n:>12,} params")
        return result


model = TransReID(
    num_classes=num_classes, num_cameras=num_cameras,
    embed_dim=768, vit_model=VIT_MODEL, sie_camera=True, jpm=True,
    img_size=(H, W),
).to(DEVICE)
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    print(f"  Wrapped in DataParallel ({NUM_GPUS} GPUs)")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 15: Training markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 7. Training"))

# ═══════════════════════════════════════════════════════════════════
# Cell 16: Training loop
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''# -- Training TransReID (CLIP ViT-Base) on Market-1501 --
# v2: All fixes from Vehicle v15 (beat SOTA)

# Label smoothing ε=0.1 (proven in vehicle v8/v15)
ce_loss = CrossEntropyLabelSmooth(num_classes, 0.1).to(DEVICE)

# Triplet loss with hard mining (margin=0.3 — proven)
tri_loss = TripletLossHardMining(margin=0.3).to(DEVICE)

# Center loss -- delayed start at epoch 30 for stability
ctr_loss = CenterLoss(num_classes, 768).to(DEVICE)
CENTER_WEIGHT = 5e-4
CENTER_START = 30

raw_model = model.module if hasattr(model, 'module') else model

# Proven CLIP fine-tuning hyperparameters (from vehicle v15)
backbone_lr = 3.5e-4
head_lr = 3.5e-3
wd = 5e-4
llrd_factor = 0.75

print(f"LLRD config: decay={llrd_factor}")
print(f"  Deepest backbone layer lr: {backbone_lr:.2e}")
print(f"  Shallowest (embed) layer lr: {backbone_lr * llrd_factor**(raw_model.num_blocks+1):.2e}")
print(f"  Metric loss: TripletLoss(m=0.3)")
print(f"  Center loss weight: {CENTER_WEIGHT} (starts at epoch {CENTER_START})")
print(f"  Label smoothing: ε=0.1")

param_groups = raw_model.get_llrd_param_groups(backbone_lr, head_lr, decay=llrd_factor)
optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

# Separate center loss optimizer (SGD, lr=0.5 — standard)
center_optimizer = torch.optim.SGD(ctr_loss.parameters(), lr=0.5)

# Store base LRs for warmup
base_lrs = [pg["lr"] for pg in optimizer.param_groups]

n_bb = sum(p.numel() for n, p in raw_model.named_parameters() if "vit" in n)
n_hd = sum(p.numel() for n, p in raw_model.named_parameters() if "vit" not in n)
print(f"Backbone params: {n_bb:,} (max_lr={backbone_lr})")
print(f"Head params:     {n_hd:,} (lr={head_lr})")

# 140 epochs (standard for Market-1501)
EPOCHS = 140
WARMUP = 10

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP)
scaler = torch.amp.GradScaler("cuda")

history = {"loss": [], "mAP": [], "R1": [], "mAP_rr": [], "R1_rr": []}
best_mAP = 0.0

print("=" * 70)
print(f"  Training TransReID ViT-Base (CLIP) on Market-1501 — v2")
print(f"  Losses: CE(ε=0.1) + Triplet(0.3) + Center(5e-4, delayed@ep{CENTER_START})")
print(f"  BNNeck fix: triplet+center get pre-BN features")
print(f"  LLRD factor={llrd_factor}, warmup={WARMUP}, epochs={EPOCHS}")
print("=" * 70)

t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    rl, nb = 0.0, 0
    use_center = (epoch >= CENTER_START)

    # Warmup: linearly scale all LRs from 0 to base
    if epoch < WARMUP:
        wf = (epoch + 1) / WARMUP
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = base_lrs[i] * wf
    elif epoch == WARMUP:
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = base_lrs[i]

    for imgs, pids, cams, _ in train_loader:
        imgs, pids, cams = imgs.to(DEVICE), pids.to(DEVICE).long(), cams.to(DEVICE).long()
        optimizer.zero_grad()
        if use_center:
            center_optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(imgs, cam_ids=cams)
            if len(out) == 3:
                c, f, jc = out
                loss = ce_loss(c, pids) + tri_loss(f, pids) + 0.5 * ce_loss(jc, pids)
            else:
                c, f = out
                loss = ce_loss(c, pids) + tri_loss(f, pids)

        if use_center:
            center_l = ctr_loss(f.float(), pids)
            total_loss = loss + CENTER_WEIGHT * center_l
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            scaler.unscale_(center_optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.step(center_optimizer)
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=5.0)
            scaler.step(optimizer)

        scaler.update()
        rl += loss.item() if not torch.isnan(loss) else 0.0
        nb += 1

    if epoch >= WARMUP:
        scheduler.step()
    history["loss"].append(rl / max(nb, 1))

    if (epoch + 1) % 10 == 0:
        hd_lr = optimizer.param_groups[-1]["lr"]
        top_bb_lr = max(pg["lr"] for pg in optimizer.param_groups[:-1])
        ctr_tag = " [+center]" if use_center else ""
        print(f"Epoch {epoch+1:3d} | Loss {rl/max(nb,1):.4f} | "
              f"top_bb_lr={top_bb_lr:.2e} head_lr={hd_lr:.2e}{ctr_tag}")

    # Evaluate every 20 epochs (and final)
    if (epoch + 1) % 20 == 0 or epoch == EPOCHS - 1:
        mAP, cmc, mAP_rr, cmc_rr = full_eval(model, query_loader, gallery_loader,
                                               DEVICE, pass_cams=True)
        history["mAP"].append(mAP); history["R1"].append(cmc[0])
        history["mAP_rr"].append(mAP_rr or 0)
        history["R1_rr"].append(cmc_rr[0] if cmc_rr is not None else 0)
        is_best = mAP > best_mAP
        if is_best:
            best_mAP = mAP
            _state = (model.module if hasattr(model, 'module') else model).state_dict()
            torch.save(_state, OUTPUT_DIR / "transreid_person_best.pth")
        tag = " ★" if is_best else ""
        print(f"  → mAP: {mAP:.4f}, R1: {cmc[0]:.4f}")
        if mAP_rr: print(f"  → mAP(RR): {mAP_rr:.4f}, R1(RR): {cmc_rr[0]:.4f}{tag}")

elapsed = time.time() - t0
print(f"\nTransReID (CLIP) v2 done in {elapsed/3600:.1f}h | Best mAP: {best_mAP:.4f}")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 17: Training curves markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 8. Training Curves"))

# ═══════════════════════════════════════════════════════════════════
# Cell 18: Plot curves
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("TransReID (CLIP) v2 — Market-1501", fontsize=14, fontweight="bold")

# Loss
axes[0].plot(history["loss"], 'b-', linewidth=1.5)
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(True, alpha=0.3)

# mAP
eval_epochs = list(range(19, EPOCHS, 20)) + ([EPOCHS-1] if EPOCHS % 20 != 0 else [])
if history["mAP"]:
    axes[1].plot(eval_epochs[:len(history["mAP"])], history["mAP"], 'g-o', label="mAP")
    if history["mAP_rr"]:
        axes[1].plot(eval_epochs[:len(history["mAP_rr"])], history["mAP_rr"], 'g--s', label="mAP(RR)")
axes[1].set_title("mAP"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# R1
if history["R1"]:
    axes[2].plot(eval_epochs[:len(history["R1"])], history["R1"], 'r-o', label="R1")
    if history["R1_rr"]:
        axes[2].plot(eval_epochs[:len(history["R1_rr"])], history["R1_rr"], 'r--s', label="R1(RR)")
axes[2].set_title("Rank-1"); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves_v2.png", dpi=150, bbox_inches="tight")
plt.show()
print("Curves saved.")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 19: Export markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 9. Export Model"))

# ═══════════════════════════════════════════════════════════════════
# Cell 20: Export
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''# ── Export TransReID ──
raw = model.module if hasattr(model, 'module') else model
best_state = torch.load(OUTPUT_DIR / "transreid_person_best.pth", map_location="cpu")

export_path = EXPORT_DIR / "person_transreid_vit_base_market1501.pth"
torch.save(best_state, export_path)
print(f"Exported: {export_path} ({export_path.stat().st_size / 1024**2:.1f} MB)")

# ── Metadata ──
metadata = {
    "task": "person_reid",
    "dataset": "market1501",
    "model": "TransReID ViT-Base CLIP (v2, BNNeck fix, all v15 fixes, ε=0.1, 140ep)",
    "architecture": VIT_MODEL,
    "num_gpus": NUM_GPUS,
    "date": datetime.now().isoformat(),
    "input_size": "256x128",
    "embedding_dim": "768",
    "label_smoothing": "0.1",
    "metric_loss": "TripletLoss(m=0.3)",
    "center_loss": "5e-4, delayed@ep30",
    "llrd": "0.75",
    "epochs": "120",
    "best_mAP": float(best_mAP),
    "training_hours": round(elapsed / 3600, 1),
}
with open(EXPORT_DIR / "person_reid_v2_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Total GPU time: {elapsed/3600:.1f}h")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 21: Results
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("code", r'''# ── Final results summary ──
print("=" * 70)
print("PERSON ReID RESULTS — Market-1501 (v2)")
print("=" * 70)
print(f"\n{'Model':<35} {'mAP':>8} {'R1':>8} {'mAP(RR)':>10} {'R1(RR)':>10} {'Time':>8}")
print("-" * 76)

if history["mAP"]:
    rr_mAP = f"{history['mAP_rr'][-1]*100:.1f}%" if history["mAP_rr"] else "N/A"
    rr_R1 = f"{history['R1_rr'][-1]*100:.1f}%" if history["R1_rr"] else "N/A"
    print(f"{'TransReID ViT-Base (CLIP) v2':<35} {best_mAP*100:>7.1f}% {history['R1'][-1]*100:>7.1f}% {rr_mAP:>10} {rr_R1:>10} {elapsed/3600:>7.1f}h")

print(f"\n{'SOTA Reference (CLIP-ReID)':<35} {'89.8%':>8} {'95.7%':>8} {'—':>10} {'—':>10}")
print(f"{'TransReID paper (ViT-B/16)':<35} {'89.0%':>8} {'95.1%':>8} {'94.2%':>10} {'95.4%':>10}")
print(f"{'v1 Result (for comparison)':<35} {'~87%':>8} {'~94%':>8} {'~92%':>10} {'~94%':>10}")'''))

# ═══════════════════════════════════════════════════════════════════
# Cell 22: Integration markdown
# ═══════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """## Local Integration

```yaml
# configs/default.yaml -- Stage 2 config
stage2:
  reid:
    person:
      model_name: transreid
      weights: models/reid/person_transreid_vit_base_market1501.pth
      embedding_dim: 768
      input_size: [256, 128]
```

### v2 Strategy: Apply Vehicle v15 Fixes
All fixes that pushed VeRi-776 from 80.4% → 82.2% mAP (beat SOTA):
- **norm_pre**: CLIP compatibility
- **SIE on all tokens**: Camera-aware feature learning
- **LLRD 0.75**: Protect pretrained CLIP representations
- **BNNeck routing**: Correct distance metric for triplet+center
- **Center loss delayed@ep30**: Intra-class compactness after classifier convergence
- **CLIP normalization**: Match pretrained distribution"""))

# Build notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

OUT = r'e:\dev\src\gp\notebooks\kaggle\07_person_reid_sota\07_person_reid_sota.ipynb'
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! {len(cells)} cells written to {OUT}")
