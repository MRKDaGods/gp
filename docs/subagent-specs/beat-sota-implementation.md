# Beat SOTA Implementation Spec: 84%+ MTMC IDF1

**Date**: 2026-03-24
**Author**: MTMC Planner
**Target**: MTMC IDF1 ≥ 84% on CityFlowV2 (from current 76.5%)
**Current SOTA**: 84.86% (AIC22 1st — 5-model ensemble)
**Realistic target**: 84%+ (AIC22 2nd used 3-model, got 84.37%)

---

## Table of Contents

1. [Phase 1: 384px ViT-B/16 Training](#phase-1-384px-vit-b16-training)
2. [Phase 2: ResNet101-IBN-a Backbone + Training](#phase-2-resnet101-ibn-a-backbone--training)
3. [Phase 3: Ensemble Pipeline](#phase-3-ensemble-pipeline)
4. [Phase 4: CID_BIAS Per Camera Pair](#phase-4-cid_bias-per-camera-pair)
5. [Phase 5: Re-enable Reranking](#phase-5-re-enable-reranking)
6. [Phase 6: Update Kaggle Notebooks 10a/10b/10c](#phase-6-update-kaggle-notebooks-10a10b10c)
7. [Config Summary](#config-summary)
8. [Dependency Graph](#dependency-graph)

---

## Phase 1: 384px ViT-B/16 Training

**Goal**: Train ViT-B/16 CLIP at 384×384 from scratch (not from 256px checkpoint) with circle loss.
**Expected gain**: +2-3pp MTMC IDF1 (76.5% → 79-80%)
**Dependencies**: None (first phase)

### 1.1 Files to Modify

#### 1.1.1 `src/training/train_reid.py` — Add circle loss alongside triplet

**Current** (line ~249): Circle loss replaces triplet loss:
```python
if args.loss == "circle":
    triplet_loss_fn = CircleLoss(m=0.25, gamma=64)
else:
    triplet_loss_fn = TripletLoss(margin=args.triplet_margin)
```

**Change to**: Circle loss can be added alongside triplet (new `--loss triplet+circle` mode):
```python
# After --loss parsing (around line 249)
circle_loss_fn = None
if args.loss == "circle":
    triplet_loss_fn = CircleLoss(m=0.25, gamma=args.circle_gamma)
elif args.loss == "triplet+circle":
    triplet_loss_fn = TripletLoss(margin=args.triplet_margin)
    circle_loss_fn = CircleLoss(m=args.circle_m, gamma=args.circle_gamma)
else:
    triplet_loss_fn = TripletLoss(margin=args.triplet_margin)
```

**Add CLI args** (after line ~199):
```python
parser.add_argument("--circle-m", type=float, default=0.25,
                    help="Circle loss margin")
parser.add_argument("--circle-gamma", type=float, default=80,
                    help="Circle loss scale factor")
parser.add_argument("--circle-weight", type=float, default=1.0,
                    help="Circle loss weight (when combined with triplet)")
parser.add_argument("--scheduler", type=str, default="step",
                    choices=["step", "cosine"],
                    help="LR scheduler type")
```

Also update the `--loss` choices:
```python
parser.add_argument("--loss", type=str, default="triplet",
                    choices=["triplet", "circle", "triplet+center", "triplet+circle"])
```

#### 1.1.2 `src/training/train_reid.py` — Add cosine LR scheduler

**Add new function** after `build_scheduler` (around line 70):
```python
def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 10,
    total_epochs: int = 80,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build cosine annealing scheduler with linear warmup."""
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
```

**In `main()`**, after optimizer construction (around line 264):
```python
if args.scheduler == "cosine":
    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        eta_min=1e-6,
    )
else:
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        milestones=args.milestones,
    )
```

#### 1.1.3 `src/training/train_reid.py` — `train_one_epoch` signature

**Add `circle_loss_fn` and `circle_weight` parameters**:
```python
def train_one_epoch(
    model: ReIDModelBoT,
    train_loader,
    id_loss_fn,
    triplet_loss_fn,
    center_loss_fn,
    optimizer,
    center_optimizer,
    scaler,
    device: str,
    epoch: int,
    id_weight: float = 1.0,
    triplet_weight: float = 1.0,
    center_weight: float = 0.0005,
    circle_loss_fn=None,        # NEW
    circle_weight: float = 1.0,  # NEW
):
```

**Inside the loss computation block** (around line 120):
```python
loss = loss_id + loss_tri

if circle_loss_fn is not None:
    loss_circle = circle_loss_fn(global_feat, pids) * circle_weight
    loss += loss_circle
else:
    loss_circle = torch.tensor(0.0)

if center_loss_fn is not None:
    loss_cen = center_loss_fn(global_feat, pids) * center_weight
    loss += loss_cen
else:
    loss_cen = torch.tensor(0.0)
```

#### 1.1.4 `src/training/datasets.py` — Add color jitter to transforms

**Modify `build_train_transforms`** (around line 194):
```python
def build_train_transforms(
    height: int = 256,
    width: int = 128,
    random_erasing_prob: float = 0.5,
    color_jitter: bool = False,  # NEW
) -> T.Compose:
    """Build training augmentation pipeline (BoT recipe)."""
    transforms_list = [
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
    ]
    if color_jitter:
        transforms_list.append(
            T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05)
        )
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=random_erasing_prob, value="random"),
    ])
    return T.Compose(transforms_list)
```

Add `--color-jitter` CLI arg in `train_reid.py`:
```python
parser.add_argument("--color-jitter", action="store_true", default=False)
```

And pass through in `build_dataloader`:
```python
def build_dataloader(
    dataset_name: str,
    root: str,
    height: int = 256,
    width: int = 128,
    batch_size: int = 64,
    num_instances: int = 4,
    num_workers: int = 4,
    random_erasing_prob: float = 0.5,
    color_jitter: bool = False,  # NEW
) -> ...:
    train_transform = build_train_transforms(height, width, random_erasing_prob, color_jitter)
    ...
```

### 1.2 Files to Create

#### 1.2.1 `notebooks/kaggle/09b_vehicle_reid_384px/09b_vehicle_reid_384px.ipynb`

**REWRITE the existing notebook.** This must train from CLIP pretrained, NOT from a 256px checkpoint.

**Cell structure** (12 cells):

**Cell 1: Setup & Installs** (code)
```python
%%capture
!pip install timm==0.9.16 loguru omegaconf torchreid

import os, sys, time, json, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from loguru import logger

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"PyTorch: {torch.__version__}")
```

**Cell 2: Config** (code)
```python
# === Training Configuration ===
CFG = {
    "dataset_root": "/kaggle/input/data-aicity-2023-track-2/AIC22_Track1_MTMC_Tracking/train",
    "weights_output": "/kaggle/working/transreid_cityflowv2_384px_best.pth",
    "checkpoint_dir": "/kaggle/working/checkpoints",

    # Model
    "vit_model": "vit_base_patch16_clip_224.openai",
    "embed_dim": 768,
    "img_size": (384, 384),  # (H, W)

    # Training
    "epochs": 80,
    "batch_size": 64,
    "num_instances": 4,  # K in PK sampling
    "lr": 3.5e-4,
    "warmup_epochs": 10,
    "eta_min": 1e-6,
    "weight_decay": 5e-4,
    "label_smoothing": 0.1,

    # Losses
    "triplet_margin": 0.3,
    "circle_m": 0.25,
    "circle_gamma": 80,
    "triplet_weight": 1.0,
    "circle_weight": 1.0,
    "id_weight": 1.0,

    # Augmentation
    "random_erasing_prob": 0.5,
    "color_jitter": True,

    # Eval
    "eval_every": 10,
    "fp16": True,
}

os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
print(f"Config: {json.dumps(CFG, indent=2, default=str)}")
```

**Cell 3: Dataset Parser (CityFlowV2)** (code)
```python
def parse_cityflowv2(root: str):
    """Parse CityFlowV2 ReID crops from train/ query/ gallery/ structure."""
    train, query, gallery = [], [], []

    for split_name, split_list in [("train", train), ("query", query), ("gallery", gallery)]:
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"CityFlowV2 ReID split not found: {split_dir}")
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".jpg"):
                continue
            parts = fname.split("_")
            if len(parts) < 4:
                continue
            pid = int(parts[0])
            cam_name = parts[1] + "_" + parts[2]
            img_path = os.path.join(split_dir, fname)
            split_list.append((img_path, pid, cam_name))

    all_cams = sorted({cam for _, _, cam in train + query + gallery})
    cam2id = {c: i for i, c in enumerate(all_cams)}
    train = [(p, pid, cam2id[c]) for p, pid, c in train]
    query = [(p, pid, cam2id[c]) for p, pid, c in query]
    gallery = [(p, pid, cam2id[c]) for p, pid, c in gallery]

    train_pids = sorted(set(pid for _, pid, _ in train))
    pid2label = {pid: label for label, pid in enumerate(train_pids)}
    train = [(path, pid2label[pid], cam) for path, pid, cam in train]

    num_classes = len(train_pids)
    num_cameras = len(all_cams)
    return train, query, gallery, num_classes, num_cameras

train_data, query_data, gallery_data, NUM_CLASSES, NUM_CAMERAS = parse_cityflowv2(CFG["dataset_root"])
print(f"Train: {len(train_data)} images, {NUM_CLASSES} IDs, {NUM_CAMERAS} cameras")
print(f"Query: {len(query_data)}, Gallery: {len(gallery_data)}")
```

**Cell 4: Dataset & Sampler** (code)
```python
class ReIDDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        path, pid, cam = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, pid, cam, path

class PKSampler(Sampler):
    def __init__(self, data, p=16, k=4):
        self.p, self.k = p, k
        self.pid_to_indices = defaultdict(list)
        for idx, (_, pid, _) in enumerate(data):
            self.pid_to_indices[pid].append(idx)
        self.pids = list(self.pid_to_indices.keys())
        self.batch_size = p * k
    def __iter__(self):
        np.random.shuffle(self.pids)
        batch = []
        for pid in self.pids:
            indices = self.pid_to_indices[pid]
            if len(indices) < self.k:
                selected = np.random.choice(indices, size=self.k, replace=True).tolist()
            else:
                selected = np.random.choice(indices, size=self.k, replace=False).tolist()
            batch.extend(selected)
            if len(batch) >= self.batch_size:
                yield from batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch: yield from batch
    def __len__(self): return len(self.pids) * self.k

H, W = CFG["img_size"]
train_transform = T.Compose([
    T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((H, W)),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=CFG["random_erasing_prob"], value="random"),
])
test_transform = T.Compose([
    T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

p = CFG["batch_size"] // CFG["num_instances"]
train_loader = DataLoader(ReIDDataset(train_data, train_transform),
    batch_size=CFG["batch_size"], sampler=PKSampler(train_data, p=p, k=CFG["num_instances"]),
    num_workers=4, pin_memory=True, drop_last=True)
query_loader = DataLoader(ReIDDataset(query_data, test_transform),
    batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
gallery_loader = DataLoader(ReIDDataset(gallery_data, test_transform),
    batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train loader: {len(train_loader)} batches, BS={CFG['batch_size']}")
```

**Cell 5: TransReID Model (384px)** (code)
```python
import timm

class TransReID384(nn.Module):
    """TransReID ViT-B/16 at 384x384 with BNNeck.

    Position embeddings interpolated from 14x14 (224px) to 24x24 (384px).
    No SIE — SIE hurts when training camera IDs don't match test cameras.
    """
    def __init__(self, num_classes, embed_dim=768,
                 vit_model="vit_base_patch16_clip_224.openai"):
        super().__init__()
        # Create ViT at 384px — timm auto-creates 24x24 grid
        self.vit = timm.create_model(vit_model, pretrained=True,
                                      num_classes=0, img_size=(384, 384))
        self.vit_dim = self.vit.embed_dim  # 768

        # Verify pos_embed was created for 24x24 + 1 CLS = 577
        expected_tokens = (384 // 16) ** 2 + 1  # 576 + 1 = 577
        actual_tokens = self.vit.pos_embed.shape[1]
        if actual_tokens != expected_tokens:
            # Manual interpolation from 14x14 to 24x24
            print(f"Interpolating pos_embed: {actual_tokens} -> {expected_tokens}")
            old_pe = self.vit.pos_embed.data
            cls_pe = old_pe[:, :1, :]
            grid_pe = old_pe[:, 1:, :]
            old_grid = int(grid_pe.shape[1] ** 0.5)
            new_grid = 384 // 16  # 24
            grid_pe = grid_pe.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
            grid_pe = F.interpolate(grid_pe.float(), size=(new_grid, new_grid),
                                     mode="bicubic", align_corners=False)
            grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
            new_pe = torch.cat([cls_pe, grid_pe], dim=1)
            self.vit.pos_embed = nn.Parameter(new_pe)
            print(f"pos_embed shape: {self.vit.pos_embed.shape}")

        # BNNeck
        self.bn = nn.BatchNorm1d(self.vit_dim)
        self.bn.bias.requires_grad_(False)

        # Projection (identity since embed_dim == vit_dim)
        self.proj = nn.Identity() if embed_dim == self.vit_dim else nn.Linear(self.vit_dim, embed_dim, bias=False)

        # Classifier
        self.cls_head = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.normal_(self.cls_head.weight, std=0.001)

    def forward(self, x, cam_ids=None):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        if hasattr(self.vit, "_pos_embed"):
            x = self.vit._pos_embed(x)
        else:
            cls_tok = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tok, x], dim=1) + self.vit.pos_embed
            if hasattr(self.vit, "pos_drop"): x = self.vit.pos_drop(x)
        if hasattr(self.vit, "patch_drop"): x = self.vit.patch_drop(x)
        if hasattr(self.vit, "norm_pre"): x = self.vit.norm_pre(x)
        for blk in self.vit.blocks: x = blk(x)
        x = self.vit.norm(x)

        g_feat = x[:, 0]  # CLS token
        bn = self.bn(g_feat)
        proj = self.proj(bn)

        if self.training:
            cls = self.cls_head(proj)
            return cls, g_feat, proj  # cls_score, global_feat (for triplet), bn_feat
        return F.normalize(proj, p=2, dim=1)

model = TransReID384(NUM_CLASSES, embed_dim=CFG["embed_dim"],
                     vit_model=CFG["vit_model"]).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {n_params:.1f}M params, embed_dim={CFG['embed_dim']}")
```

**Cell 6: Losses** (code)
```python
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets_oh = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_oh + self.epsilon / self.num_classes
        return (-targets_smooth * log_probs).mean(0).sum()

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.cdist(inputs, inputs, p=2)
        mask = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        dist_ap = torch.stack([dist[i][mask[i]].max() for i in range(n)])
        dist_an = torch.stack([dist[i][~mask[i]].min() for i in range(n)])
        return self.ranking_loss(dist_an, dist_ap, torch.ones_like(dist_an))

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=80):
        super().__init__()
        self.m, self.gamma = m, gamma
        self.soft_plus = nn.Softplus()
    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, p=2, dim=1)
        n = inputs.size(0)
        sim = inputs @ inputs.t()
        mask = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        Op, On = 1 + self.m, -self.m
        dp, dn = 1 - self.m, self.m
        loss = 0
        for i in range(n):
            pos_sim = sim[i][mask[i]]
            neg_sim = sim[i][~mask[i]]
            ap = torch.clamp_min(-pos_sim.detach() + Op, 0)
            an = torch.clamp_min(neg_sim.detach() + On, 0)
            logit_p = -ap * (pos_sim - dp) * self.gamma
            logit_n = an * (neg_sim - dn) * self.gamma
            loss += self.soft_plus(torch.logsumexp(logit_n, 0) + torch.logsumexp(-logit_p, 0))
        return loss / n

id_loss_fn = CrossEntropyLabelSmooth(NUM_CLASSES, epsilon=CFG["label_smoothing"])
triplet_loss_fn = TripletLoss(margin=CFG["triplet_margin"])
circle_loss_fn = CircleLoss(m=CFG["circle_m"], gamma=CFG["circle_gamma"])
print("Losses: ID (label smooth) + Triplet (hard mining) + Circle")
```

**Cell 7: Optimizer & Scheduler** (code)
```python
params = [
    {"params": model.vit.parameters(), "lr": CFG["lr"] * 0.1},
    {"params": model.bn.parameters(), "lr": CFG["lr"]},
    {"params": model.cls_head.parameters(), "lr": CFG["lr"]},
]
if isinstance(model.proj, nn.Linear):
    params.append({"params": model.proj.parameters(), "lr": CFG["lr"]})

optimizer = torch.optim.Adam(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"])

warmup_sched = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=CFG["warmup_epochs"])
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG["epochs"] - CFG["warmup_epochs"], eta_min=CFG["eta_min"])
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[CFG["warmup_epochs"]])

scaler = torch.amp.GradScaler("cuda") if CFG["fp16"] else None
print(f"Optimizer: Adam, LR={CFG['lr']}, Cosine schedule, {CFG['warmup_epochs']} warmup epochs")
```

**Cell 8: Evaluation Functions** (code)
```python
@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, pids, cams = [], [], []
    for imgs, pid, cam, _ in loader:
        imgs = imgs.to(device)
        f = model(imgs)
        if isinstance(f, tuple): f = f[-1]
        # Flip augment
        f_flip = model(torch.flip(imgs, [3]))
        if isinstance(f_flip, tuple): f_flip = f_flip[-1]
        f = F.normalize((f + f_flip) / 2, p=2, dim=1)
        feats.append(f.cpu().numpy())
        pids.append(pid.numpy())
        cams.append(cam.numpy())
    return np.concatenate(feats), np.concatenate(pids), np.concatenate(cams)

def eval_reid(qf, qp, qc, gf, gp, gc):
    """Compute mAP and CMC with Market-1501 protocol."""
    dist = 1 - qf @ gf.T
    n_q = dist.shape[0]
    all_AP, all_cmc = [], []
    for i in range(n_q):
        order = np.argsort(dist[i])
        # Remove same-pid-same-cam
        valid = ~((gp[order] == qp[i]) & (gc[order] == qc[i]))
        matches = gp[order][valid] == qp[i]
        if not matches.any(): continue
        cmc = matches.cumsum()
        cmc = (cmc >= 1).astype(float)
        all_cmc.append(cmc[:50])
        n_rel = matches.sum()
        cum_tp = matches.cumsum()
        precision = cum_tp / (np.arange(len(matches)) + 1)
        all_AP.append((precision * matches).sum() / n_rel)
    mAP = np.mean(all_AP)
    cmc = np.mean(all_cmc, axis=0) if all_cmc else np.zeros(50)
    return mAP, cmc
```

**Cell 9: Training Loop** (code)
```python
best_mAP = 0.0
history = []

for epoch in range(CFG["epochs"]):
    model.train()
    running = {"loss": 0, "id": 0, "tri": 0, "cir": 0, "n": 0}
    t0 = time.time()

    for imgs, pids, _, _ in train_loader:
        imgs, pids = imgs.to(DEVICE), pids.to(DEVICE).long()
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            cls_score, global_feat, bn_feat = model(imgs)
            loss_id = id_loss_fn(cls_score, pids) * CFG["id_weight"]
            loss_tri = triplet_loss_fn(global_feat, pids) * CFG["triplet_weight"]
            loss_cir = circle_loss_fn(global_feat, pids) * CFG["circle_weight"]
            loss = loss_id + loss_tri + loss_cir

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running["loss"] += loss.item()
        running["id"] += loss_id.item()
        running["tri"] += loss_tri.item()
        running["cir"] += loss_cir.item()
        running["n"] += 1

    scheduler.step()
    n = running["n"]
    elapsed = time.time() - t0
    lr = optimizer.param_groups[0]["lr"]
    print(f"E{epoch:02d} | {elapsed:.0f}s | LR={lr:.6f} | "
          f"L={running['loss']/n:.4f} ID={running['id']/n:.4f} "
          f"Tri={running['tri']/n:.4f} Cir={running['cir']/n:.4f}")

    # Evaluate
    if (epoch + 1) % CFG["eval_every"] == 0 or epoch == CFG["epochs"] - 1:
        qf, qp, qc = extract_features(model, query_loader, DEVICE)
        gf, gp, gc = extract_features(model, gallery_loader, DEVICE)
        mAP, cmc = eval_reid(qf, qp, qc, gf, gp, gc)
        print(f"  >> mAP={mAP:.4f}, R1={cmc[0]:.4f}, R5={cmc[4]:.4f}")
        history.append({"epoch": epoch, "mAP": mAP, "R1": cmc[0]})

        if mAP > best_mAP:
            best_mAP = mAP
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "mAP": mAP},
                       CFG["weights_output"])
            print(f"  ★ New best mAP: {best_mAP:.4f}")

        # Save periodic checkpoint
        torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                     "optimizer": optimizer.state_dict(), "mAP": mAP},
                    f"{CFG['checkpoint_dir']}/ckpt_e{epoch:02d}.pth")

print(f"\nTraining complete! Best mAP: {best_mAP:.4f}")
```

**Cell 10: Final evaluation** (code)
```python
# Load best model and do final evaluation
ckpt = torch.load(CFG["weights_output"], map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE)

qf, qp, qc = extract_features(model, query_loader, DEVICE)
gf, gp, gc = extract_features(model, gallery_loader, DEVICE)
mAP, cmc = eval_reid(qf, qp, qc, gf, gp, gc)
print(f"Final: mAP={mAP:.4f}, R1={cmc[0]:.4f}, R5={cmc[4]:.4f}, R10={cmc[9]:.4f}")
print(f"History: {json.dumps(history, indent=2)}")
```

**Cell 11: Save for Kaggle dataset upload** (code)
```python
# Copy weights to output for dataset upload
import shutil
output_name = "transreid_cityflowv2_384px_best.pth"
shutil.copy2(CFG["weights_output"], f"/kaggle/working/{output_name}")
print(f"Weights saved: /kaggle/working/{output_name}")
print(f"Size: {os.path.getsize(f'/kaggle/working/{output_name}') / 1e6:.1f}MB")

# Save training history
with open("/kaggle/working/training_history.json", "w") as f:
    json.dump({"config": CFG, "history": history, "best_mAP": best_mAP}, f, indent=2, default=str)
```

**Cell 12: Verify checkpoint structure** (code)
```python
# Verify the saved checkpoint loads correctly and has expected keys
ckpt = torch.load(f"/kaggle/working/{output_name}", map_location="cpu", weights_only=False)
keys = list(ckpt["state_dict"].keys())
print(f"State dict keys: {len(keys)}")
print(f"Key samples: {keys[:5]} ... {keys[-5:]}")
# Check key naming matches what src/stage2_features/transreid_model.py expects:
# 'vit.blocks.0.attn.qkv.weight', 'bn.weight', 'bn.bias', 'cls_head.weight'
assert any("vit." in k for k in keys), "Missing vit. prefix!"
assert any("bn." in k for k in keys), "Missing bn. prefix!"
print("✓ Checkpoint structure validated")
```

#### 1.2.2 `notebooks/kaggle/09b_vehicle_reid_384px/kernel-metadata.json`

**UPDATE (not create) to reflect correct dataset sources:**
```json
{
  "id": "mrkdagods/09b-vehicle-reid-384px-training",
  "title": "09b Vehicle ReID 384px Training (ViT-B/16 CLIP)",
  "code_file": "09b_vehicle_reid_384px.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [
    "thanhnguyenle/data-aicity-2023-track-2"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

### 1.3 Config Changes

None for Phase 1 alone — config changes happen in Phase 3 when we integrate 384px into the pipeline.

### 1.4 Testing Plan

1. **Local smoke test** (before Kaggle push):
   ```bash
   python -m src.training.train_reid \
     --dataset cityflowv2 \
     --root data/processed/cityflowv2_crops \
     --backbone resnet50_ibn_a \
     --height 384 --width 384 \
     --epochs 2 --eval-every 1 \
     --loss triplet+circle \
     --scheduler cosine \
     --color-jitter
   ```
   Verify training runs without errors, circle loss computes nonzero values.

2. **Kaggle push**: `kaggle kernels push -p notebooks/kaggle/09b_vehicle_reid_384px/`

3. **Success criteria**: mAP ≥ 0.55 on CityFlowV2 val (up from ~0.45 at 256px)

### 1.5 Kaggle Push Command

```bash
kaggle kernels push -p notebooks/kaggle/09b_vehicle_reid_384px/
# Monitor:
python scripts/kaggle_logs.py mrkdagods/09b-vehicle-reid-384px-training --tail 50
```

After training completes, download the weights:
```bash
kaggle kernels output mrkdagods/09b-vehicle-reid-384px-training -p models/reid/
mv models/reid/transreid_cityflowv2_384px_best.pth models/reid/
```

Then upload to the weights dataset:
```bash
# Add to mrkdagods/mtmc-weights dataset
kaggle datasets version -p models/ -m "Add 384px ViT-B/16 checkpoint"
```

---

## Phase 2: ResNet101-IBN-a Backbone + Training

**Goal**: Add ResNet101-IBN-a as second ReID backbone with BoT recipe.
**Expected gain**: +1.5pp incremental on top of Phase 1 ensemble
**Dependencies**: Phase 1 code changes (circle loss, cosine scheduler)

### 2.1 Files to Modify

#### 2.1.1 `src/training/model.py` — Add ResNet101-IBN-a with GeM pooling

**Add IBN-a module and ResNet101 builder.** Insert BEFORE `class ReIDModelBoT`:

```python
class IBN_a(nn.Module):
    """Instance-Batch Normalization (IBN-a) layer.

    Splits channels: first half uses InstanceNorm, second half uses BatchNorm.
    InstanceNorm removes style (camera appearance), BatchNorm preserves content (identity).
    Reference: Pan et al., "Two at Once: Enhancing Learning and Generalization
    Capacities via IBN-Net" (ECCV 2018).
    """
    def __init__(self, planes):
        super().__init__()
        half = planes // 2
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(planes - half)

    def forward(self, x):
        split = x.shape[1] // 2
        out1 = self.IN(x[:, :split])
        out2 = self.BN(x[:, split:])
        return torch.cat([out1, out2], dim=1)


class GeM(nn.Module):
    """Generalized Mean Pooling.

    More discriminative than average pooling — emphasizes higher activations.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


def _build_resnet101_ibn_a(last_stride: int = 1, pretrained: bool = True) -> nn.Module:
    """Build ResNet101 with IBN-a layers on layer1 and layer2.

    IBN-a is applied to the first BN in each Bottleneck of layer1 and layer2.
    This matches the published resnet101_ibn_a architecture.

    Args:
        last_stride: Stride for layer4 (1 for high-resolution features).
        pretrained: Load ImageNet pretrained ResNet101 weights.

    Returns:
        Modified ResNet101 module (without FC head).
    """
    import torchvision.models as tv_models

    # Load standard ResNet101
    weights = tv_models.ResNet101_Weights.DEFAULT if pretrained else None
    base = tv_models.resnet101(weights=weights)

    # Replace BN with IBN-a in layer1 and layer2
    for layer in [base.layer1, base.layer2]:
        for block in layer:
            if hasattr(block, "bn1"):
                planes = block.bn1.num_features
                block.bn1 = IBN_a(planes)

    # Modify last stride
    if last_stride == 1:
        for module in base.layer4.modules():
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                module.stride = (1, 1)

    # Remove FC and avgpool — we'll use our own
    base.fc = nn.Identity()
    base.avgpool = nn.Identity()

    return base


class ReIDModelResNet101IBN(nn.Module):
    """ResNet101-IBN-a with GeM pooling and BNNeck for BoT-style training.

    Architecture:
        ResNet101-IBN-a → GeM pool → 2048D feat (for triplet)
                                   → BNNeck → classifier (for ID loss)
    """

    def __init__(
        self,
        num_classes: int = 751,
        last_stride: int = 1,
        pretrained: bool = True,
        gem_p: float = 3.0,
    ):
        super().__init__()
        self.backbone = _build_resnet101_ibn_a(last_stride, pretrained)
        self.feat_dim = 2048

        # GeM pooling
        self.pool = GeM(p=gem_p)

        # BNNeck
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        # Classifier
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        logger.info(
            f"ReIDModelResNet101IBN: classes={num_classes}, "
            f"feat_dim={self.feat_dim}, last_stride={last_stride}, gem_p={gem_p}"
        )

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, x: torch.Tensor):
        feat_map = self._backbone_forward(x)    # (B, 2048, H, W)
        global_feat = self.pool(feat_map)         # (B, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # (B, 2048)
        bn_feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(bn_feat)
            return cls_score, global_feat, bn_feat
        return F.normalize(bn_feat, p=2, dim=1)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)
```

#### 2.1.2 `src/training/train_reid.py` — Support resnet101_ibn_a backbone

**Modify the model construction block** (around line 239):
```python
# Build model
if args.backbone == "resnet101_ibn_a":
    from src.training.model import ReIDModelResNet101IBN
    model = ReIDModelResNet101IBN(
        num_classes=num_classes,
        last_stride=args.last_stride,
        pretrained=True,
        gem_p=3.0,
    ).to(device)
    feat_dim = 2048
else:
    model = ReIDModelBoT(
        model_name=args.backbone,
        num_classes=num_classes,
        last_stride=args.last_stride,
        pretrained=True,
        feat_dim=2048 if "resnet50" in args.backbone else 512,
        neck="bnneck",
    ).to(device)
    feat_dim = 2048 if "resnet50" in args.backbone else 512
```

Also update the optimizer builder for the new model:
```python
if args.backbone == "resnet101_ibn_a":
    params = [
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.pool.parameters(), "lr": args.lr},
        {"params": model.bottleneck.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    center_optimizer = None
else:
    optimizer, center_optimizer = build_optimizer(model, center_loss_fn, lr=args.lr)
```

### 2.2 Files to Create

#### 2.2.1 `notebooks/kaggle/09d_vehicle_reid_resnet101ibn/09d_vehicle_reid_resnet101ibn.ipynb`

**Same 12-cell structure as 09b**, with these differences:

**Cell 2: Config changes**:
```python
CFG = {
    "dataset_root": "/kaggle/input/data-aicity-2023-track-2/AIC22_Track1_MTMC_Tracking/train",
    "weights_output": "/kaggle/working/resnet101ibn_cityflowv2_384px_best.pth",
    "checkpoint_dir": "/kaggle/working/checkpoints",

    # Model
    "backbone": "resnet101_ibn_a",
    "feat_dim": 2048,
    "img_size": (384, 384),
    "gem_p": 3.0,

    # Training — batch=48 for ResNet101 at 384px (T4 16GB VRAM)
    "epochs": 80,
    "batch_size": 48,
    "num_instances": 4,
    "lr": 3.5e-4,
    "warmup_epochs": 10,
    "eta_min": 1e-6,
    "weight_decay": 5e-4,
    "label_smoothing": 0.1,

    # Losses
    "triplet_margin": 0.3,
    "circle_m": 0.25,
    "circle_gamma": 80,

    # Augmentation
    "random_erasing_prob": 0.5,
    "color_jitter": True,

    "eval_every": 10,
    "fp16": True,
}
```

**Cell 5: Model** — Replace TransReID384 with:
```python
import torchvision.models as tv_models

class IBN_a(nn.Module):
    def __init__(self, planes):
        super().__init__()
        half = planes // 2
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(planes - half)
    def forward(self, x):
        split = x.shape[1] // 2
        return torch.cat([self.IN(x[:, :split]), self.BN(x[:, split:])], 1)

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)

class ResNet101IBN(nn.Module):
    def __init__(self, num_classes, feat_dim=2048, last_stride=1, gem_p=3.0):
        super().__init__()
        base = tv_models.resnet101(weights=tv_models.ResNet101_Weights.DEFAULT)
        # Insert IBN-a into layer1 and layer2
        for layer in [base.layer1, base.layer2]:
            for block in layer:
                if hasattr(block, "bn1"):
                    block.bn1 = IBN_a(block.bn1.num_features)
        # Last stride = 1
        if last_stride == 1:
            for m in base.layer4.modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride = (1, 1)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool = GeM(p=gem_p)
        self.feat_dim = feat_dim
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        g = self.pool(x).view(x.size(0), -1)
        bn = self.bottleneck(g)
        if self.training:
            return self.classifier(bn), g, bn
        return F.normalize(bn, p=2, dim=1)

model = ResNet101IBN(NUM_CLASSES, feat_dim=CFG["feat_dim"],
                     last_stride=1, gem_p=CFG["gem_p"]).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"ResNet101-IBN-a: {n_params:.1f}M params, feat_dim={CFG['feat_dim']}")
```

**Cell 7: Optimizer** — adjust param groups for ResNet:
```python
params = [
    {"params": list(model.conv1.parameters()) + list(model.bn1.parameters()) +
               list(model.layer1.parameters()) + list(model.layer2.parameters()) +
               list(model.layer3.parameters()) + list(model.layer4.parameters()),
     "lr": CFG["lr"] * 0.1},
    {"params": model.pool.parameters(), "lr": CFG["lr"]},
    {"params": model.bottleneck.parameters(), "lr": CFG["lr"]},
    {"params": model.classifier.parameters(), "lr": CFG["lr"]},
]
```

#### 2.2.2 `notebooks/kaggle/09d_vehicle_reid_resnet101ibn/kernel-metadata.json`

```json
{
  "id": "mrkdagods/09d-vehicle-reid-resnet101ibn-training",
  "title": "09d Vehicle ReID ResNet101-IBN-a Training",
  "code_file": "09d_vehicle_reid_resnet101ibn.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [
    "thanhnguyenle/data-aicity-2023-track-2"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

### 2.3 Testing Plan

1. **Local unit test**: Verify `ReIDModelResNet101IBN` forward pass:
   ```python
   from src.training.model import ReIDModelResNet101IBN
   m = ReIDModelResNet101IBN(num_classes=100)
   x = torch.randn(2, 3, 384, 384)
   m.train()
   cls, gf, bf = m(x)
   assert cls.shape == (2, 100)
   assert gf.shape == (2, 2048)
   m.eval()
   f = m(x)
   assert f.shape == (2, 2048)
   assert torch.allclose(f.norm(dim=1), torch.ones(2), atol=1e-5)
   ```

2. **Kaggle push**: `kaggle kernels push -p notebooks/kaggle/09d_vehicle_reid_resnet101ibn/`

3. **Success criteria**: mAP ≥ 0.50 on CityFlowV2 val split

### 2.4 Kaggle Push Commands

```bash
kaggle kernels push -p notebooks/kaggle/09d_vehicle_reid_resnet101ibn/
python scripts/kaggle_logs.py mrkdagods/09d-vehicle-reid-resnet101ibn-training --tail 50

# After completion:
kaggle kernels output mrkdagods/09d-vehicle-reid-resnet101ibn-training -p models/reid/
mv models/reid/resnet101ibn_cityflowv2_384px_best.pth models/reid/
kaggle datasets version -p models/ -m "Add ResNet101-IBN-a 384px checkpoint"
```

---

## Phase 3: Ensemble Pipeline

**Goal**: Update stage2 + stage4 to support ResNet101-IBN-a as ensemble secondary model with score-level fusion and separate PCA.
**Expected gain**: +1.5pp from ensemble diversity, cumulative 81-82%
**Dependencies**: Phase 1 (384px weights), Phase 2 (ResNet101-IBN-a weights)

### 3.1 Files to Modify

#### 3.1.1 `src/stage2_features/reid_model.py` — Add ResNet101-IBN-a loading

**In `_build_torchreid` method**, add a branch for ResNet101-IBN-a **BEFORE** the torchreid fallback (around line 141):

```python
def _build_model(self, model_name: str, weights_path: Optional[str]):
    if self.is_transreid:
        return self._build_transreid(weights_path)
    if model_name == "resnet101_ibn_a":
        return self._build_resnet101_ibn(weights_path)
    return self._build_torchreid(model_name, weights_path)

def _build_resnet101_ibn(self, weights_path: Optional[str]):
    """Build ResNet101-IBN-a model for inference."""
    from src.training.model import ReIDModelResNet101IBN
    model = ReIDModelResNet101IBN(
        num_classes=1,  # not used for inference
        last_stride=1,
        pretrained=False,  # we load our own weights
        gem_p=3.0,
    )
    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        # Remove classifier keys
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("classifier")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            critical = [k for k in missing if "classifier" not in k]
            if critical:
                logger.warning(f"ResNet101-IBN missing critical keys: {critical}")
        logger.info(f"Loaded ResNet101-IBN-a weights from {weights_path}")
    return model
```

Also add `"resnet101_ibn_a"` to the class's model routing. Specifically, the `_TRANSREID_NAMES` set stays the same, but the `_build_model` dispatch now has 3 branches.

#### 3.1.2 `src/stage2_features/pipeline.py` — Separate PCA for secondary model

The current code concatenates primary + secondary embeddings BEFORE PCA when `vehicle2.save_separate=False`. For proper score-level fusion, we need separate PCA per model.

**Replace the secondary embedding handling** in `run_stage2`. After the main PCA whitening (around line 380), add:

```python
# 6b. Separate PCA whitening for secondary embeddings (for score-level fusion)
if vehicle2_separate and all_secondary_embeddings:
    sec_valid = [e for e in all_secondary_embeddings if e is not None]
    if sec_valid:
        sec_matrix = np.stack(sec_valid, axis=0)
        logger.info(f"Secondary embedding matrix: {sec_matrix.shape}")

        # Camera-aware BN for secondary
        if stage_cfg.get("camera_bn", {}).get("enabled", True):
            sec_cam_ids = [all_camera_ids[i] for i, e in enumerate(all_secondary_embeddings) if e is not None]
            sec_matrix = camera_aware_batch_normalize(sec_matrix, sec_cam_ids)

        # Separate PCA for secondary model
        sec_pca_cfg = stage_cfg.pca
        sec_pca_path = sec_pca_cfg.get("secondary_pca_model_path",
                                        "models/reid/pca_transform_secondary.pkl")
        if sec_pca_cfg.enabled:
            n_sec, d_sec = sec_matrix.shape
            sec_components = min(sec_pca_cfg.n_components, d_sec)
            if n_sec >= max(50, sec_components):
                sec_whitener = PCAWhitener(n_components=sec_components)
                if Path(sec_pca_path).exists():
                    sec_whitener.load(sec_pca_path)
                    sec_matrix = sec_whitener.transform(sec_matrix)
                else:
                    sec_matrix = sec_whitener.fit_transform(sec_matrix)
                    sec_whitener.save(sec_pca_path)
                logger.info(f"Secondary PCA: {d_sec}D → {sec_components}D")

        # L2 normalize secondary
        sec_norms = np.linalg.norm(sec_matrix, axis=1, keepdims=True)
        sec_matrix = sec_matrix / np.maximum(sec_norms, 1e-8)

        # Save secondary embeddings
        sec_output_path = output_dir / "embeddings_secondary.npy"
        np.save(sec_output_path, sec_matrix.astype(np.float32))
        logger.info(f"Saved secondary embeddings: {sec_output_path}")
```

#### 3.1.3 `configs/default.yaml` — Update vehicle2 config block

**Replace the `vehicle2` section** (around line 84):
```yaml
    vehicle2:
      enabled: false
      save_separate: true
      model_name: "resnet101_ibn_a"
      weights_path: "models/reid/resnet101ibn_cityflowv2_384px_best.pth"
      embedding_dim: 2048
      input_size: [384, 384]
      clip_normalization: false  # ResNet uses ImageNet normalization
```

**Add secondary PCA path** under `pca:` section:
```yaml
  pca:
    enabled: true
    n_components: 384
    pca_model_path: "models/reid/pca_transform.pkl"
    secondary_pca_model_path: "models/reid/pca_transform_secondary.pkl"  # NEW
```

#### 3.1.4 `configs/datasets/cityflowv2.yaml` — Update for ensemble

**Replace `vehicle2` section** in stage2.reid:
```yaml
    vehicle2:
      enabled: true
      save_separate: true
      model_name: "resnet101_ibn_a"
      weights_path: "models/reid/resnet101ibn_cityflowv2_384px_best.pth"
      embedding_dim: 2048
      input_size: [384, 384]
      clip_normalization: false
```

**Update primary vehicle to 384px**:
```yaml
    vehicle:
      model_name: "transreid"
      weights_path: "models/reid/transreid_cityflowv2_384px_best.pth"
      weights_fallback: "models/reid/transreid_cityflowv2_best.pth"
      embedding_dim: 768
      input_size: [384, 384]  # CHANGED from [256, 256]
      vit_model: "vit_base_patch16_clip_224.openai"
      num_cameras: 59
      clip_normalization: true
```

**Update stage4 secondary_embeddings**:
```yaml
stage4:
  association:
    secondary_embeddings:
      path: "data/outputs/run_latest/stage2/embeddings_secondary.npy"
      weight: 0.4  # 40% ResNet101-IBN, 60% ViT — sweep [0.3, 0.4, 0.5]
```

**Update stage4 FIC regularisation** — will need re-tuning for 384px features:
```yaml
    fic:
      enabled: true
      regularisation: 0.1  # May need re-tuning for 384px features
      min_samples: 5
```

### 3.2 Testing Plan

1. **Unit test**: Verify `ReIDModel` correctly dispatches `resnet101_ibn_a`:
   ```python
   from src.stage2_features.reid_model import ReIDModel
   m = ReIDModel(model_name="resnet101_ibn_a",
                 weights_path="models/reid/resnet101ibn_cityflowv2_384px_best.pth",
                 embedding_dim=2048, input_size=(384, 384))
   import numpy as np
   crops = [np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)]
   emb = m.extract_batch(crops)
   assert emb.shape[1] == 2048
   ```

2. **Smoke test pipeline**: Run stage2 with ensemble enabled locally (3 tracklets).

3. **Verify secondary embeddings file** is created at `data/outputs/.../stage2/embeddings_secondary.npy`.

4. **Verify score-level fusion** in stage4 — check logs for "Blending secondary appearance sim".

---

## Phase 4: CID_BIAS Per Camera Pair

**Goal**: Pre-compute per-camera-pair additive bias from GT matches. Apply before thresholding.
**Expected gain**: +0.5-1pp
**Dependencies**: Phase 3 (ensemble features for better bias estimation)

### 4.1 Files to Create

#### 4.1.1 `scripts/compute_cid_bias.py`

```python
"""Compute per-camera-pair distance bias (CID_BIAS) from GT annotations.

Computes mean cosine similarity for true-match pairs per camera pair,
then computes how much each pair deviates from the global mean.
Saves a (6×6) bias matrix as NPY.

Usage:
    python scripts/compute_cid_bias.py \
        --embeddings data/outputs/run_XXX/stage2/embeddings.npy \
        --metadata data/outputs/run_XXX/stage3/metadata.db \
        --gt-dir data/gt_upload/ \
        --output configs/datasets/cityflowv2_cid_bias.npy
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger


def compute_cid_bias(
    embeddings: np.ndarray,
    camera_ids: list[str],
    track_ids: list[int],
    gt_matches: dict[tuple[str, int], int],  # (cam_id, track_id) -> global_id
) -> tuple[np.ndarray, list[str]]:
    """Compute CID_BIAS matrix.

    Args:
        embeddings: (N, D) L2-normalized embedding matrix.
        camera_ids: Camera ID for each tracklet.
        track_ids: Track ID for each tracklet.
        gt_matches: Mapping from (camera_id, track_id) to ground-truth global ID.

    Returns:
        bias_matrix: (C, C) additive bias matrix.
        camera_names: Ordered list of camera names.
    """
    n = len(embeddings)

    # Map tracklets to GT global IDs
    global_ids = []
    for i in range(n):
        key = (camera_ids[i], track_ids[i])
        global_ids.append(gt_matches.get(key, -1))

    # Collect same-identity cross-camera similarities
    camera_set = sorted(set(camera_ids))
    cam2idx = {c: i for i, c in enumerate(camera_set)}
    C = len(camera_set)

    pair_sims: dict[tuple[int, int], list[float]] = defaultdict(list)

    # Only compute for tracklets with GT matches
    gt_tracklets = [i for i in range(n) if global_ids[i] >= 0]
    logger.info(f"GT-matched tracklets: {len(gt_tracklets)}/{n}")

    # Group by global_id
    gid_to_indices: dict[int, list[int]] = defaultdict(list)
    for i in gt_tracklets:
        gid_to_indices[global_ids[i]].append(i)

    # For each global ID, compute cross-camera pair similarities
    for gid, indices in gid_to_indices.items():
        if len(indices) < 2:
            continue
        for a_idx in range(len(indices)):
            for b_idx in range(a_idx + 1, len(indices)):
                i, j = indices[a_idx], indices[b_idx]
                if camera_ids[i] == camera_ids[j]:
                    continue
                ci = cam2idx[camera_ids[i]]
                cj = cam2idx[camera_ids[j]]
                pair_key = (min(ci, cj), max(ci, cj))
                sim = float(np.dot(embeddings[i], embeddings[j]))
                pair_sims[pair_key].append(sim)

    # Compute mean similarity per pair
    bias_matrix = np.zeros((C, C), dtype=np.float32)
    all_means = []
    for (ci, cj), sims in pair_sims.items():
        mean_sim = np.mean(sims)
        all_means.append(mean_sim)
        logger.info(f"  {camera_set[ci]}↔{camera_set[cj]}: "
                     f"mean_sim={mean_sim:.4f}, n={len(sims)}")

    if not all_means:
        logger.warning("No cross-camera GT matches found!")
        return bias_matrix, camera_set

    global_mean = np.mean(all_means)
    logger.info(f"Global mean similarity: {global_mean:.4f}")

    # Bias = pair_mean - global_mean (positive = easier pair, negative = harder pair)
    for (ci, cj), sims in pair_sims.items():
        bias = np.mean(sims) - global_mean
        bias_matrix[ci, cj] = bias
        bias_matrix[cj, ci] = bias

    return bias_matrix, camera_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load embeddings
    embeddings = np.load(args.embeddings).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    # Load metadata
    from src.stage3_indexing.metadata_store import MetadataStore
    meta_store = MetadataStore(args.metadata)
    camera_ids = []
    track_ids = []
    for i in range(len(embeddings)):
        m = meta_store.get_tracklet(i)
        camera_ids.append(m["camera_id"])
        track_ids.append(m["track_id"])

    # Load GT (MOT format)
    gt_matches = {}
    gt_dir = Path(args.gt_dir)
    for gt_file in sorted(gt_dir.glob("*.txt")):
        cam_id = gt_file.stem  # e.g., S01_c001
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                frame_id = int(parts[0])
                global_id = int(parts[1])
                local_track_id = int(parts[1])  # GT format: frame,id,...
                gt_matches[(cam_id, local_track_id)] = global_id

    # Compute bias
    bias_matrix, camera_names = compute_cid_bias(
        embeddings, camera_ids, track_ids, gt_matches
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, bias_matrix)

    # Also save camera name mapping
    import json
    mapping_path = output_path.with_suffix(".json")
    with open(mapping_path, "w") as f:
        json.dump({"cameras": camera_names}, f, indent=2)

    logger.info(f"CID_BIAS saved to {output_path} ({bias_matrix.shape})")
    logger.info(f"Camera mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()
```

### 4.2 Files to Modify

#### 4.2.1 `src/stage4_association/pipeline.py` — Apply CID_BIAS

**In Step 5b** (camera bias section, around line 350), add NPY-based CID_BIAS support BEFORE the existing iterative learning code:

```python
# Step 5b: Camera distance bias adjustment
camera_bias_cfg = stage_cfg.get("camera_bias", {})
if camera_bias_cfg.get("enabled", False):
    cid_bias_path = camera_bias_cfg.get("cid_bias_npy_path", "")
    if cid_bias_path and Path(cid_bias_path).exists():
        # NEW: Pre-computed NPY-based CID_BIAS (AIC21/22 technique)
        import json
        cid_bias_matrix = np.load(cid_bias_path)
        cid_mapping_path = Path(cid_bias_path).with_suffix(".json")
        if cid_mapping_path.exists():
            with open(cid_mapping_path) as f:
                cam_names = json.load(f)["cameras"]
            cam2idx = {c: i for i, c in enumerate(cam_names)}
        else:
            # Infer from data
            cam_names = sorted(set(camera_ids))
            cam2idx = {c: i for i, c in enumerate(cam_names)}

        adjusted_count = 0
        for (i, j) in list(combined_sim.keys()):
            ci = cam2idx.get(camera_ids[i])
            cj = cam2idx.get(camera_ids[j])
            if ci is not None and cj is not None:
                bias = float(cid_bias_matrix[ci, cj])
                combined_sim[(i, j)] += bias
                adjusted_count += 1
        logger.info(
            f"CID_BIAS: adjusted {adjusted_count} pairs from {cid_bias_path}"
        )
    else:
        # Existing iterative learning code (unchanged)
        ...
```

#### 4.2.2 `configs/default.yaml` — Add CID_BIAS config key

Under `camera_bias:` section (around line 246):
```yaml
    camera_bias:
      enabled: false
      iterations: 1
      cid_bias_npy_path: ""  # NEW: path to pre-computed NPY bias matrix
```

#### 4.2.3 `configs/datasets/cityflowv2.yaml` — Enable CID_BIAS

```yaml
    camera_bias:
      enabled: true
      cid_bias_npy_path: "configs/datasets/cityflowv2_cid_bias.npy"
      iterations: 0  # 0 = use pre-computed only, no iterative learning
```

### 4.3 Testing Plan

1. **Create bias matrix** locally from GT:
   ```bash
   python scripts/compute_cid_bias.py \
     --embeddings data/outputs/run_latest/stage2/embeddings.npy \
     --metadata data/outputs/run_latest/stage3/metadata.db \
     --gt-dir data/gt_upload/ \
     --output configs/datasets/cityflowv2_cid_bias.npy
   ```

2. **Verify bias values** are reasonable (|bias| < 0.15 typically):
   ```python
   import numpy as np
   b = np.load("configs/datasets/cityflowv2_cid_bias.npy")
   print(f"Bias matrix:\n{b}")
   print(f"Range: [{b.min():.4f}, {b.max():.4f}]")
   ```

3. **Run pipeline** with CID_BIAS enabled and compare MTMC IDF1.

---

## Phase 5: Re-enable Reranking

**Goal**: After ensemble features are strong enough, re-test k-reciprocal reranking.
**Expected gain**: +0.5pp (only with strong features)
**Dependencies**: Phase 3 (ensemble features active)

### 5.1 Files to Modify

#### 5.1.1 `configs/datasets/cityflowv2.yaml` — Reranking config

```yaml
    reranking:
      enabled: true  # CHANGED: re-enable with ensemble features
      k1: 20
      k2: 3
      lambda_value: 0.3
```

### 5.2 Testing Plan

This is a config-only change. Test by running stages 4-5 with reranking enabled:

```bash
python scripts/run_pipeline.py --config configs/default.yaml \
  --override stage4.association.reranking.enabled=true \
  --stages 4,5
```

Compare MTMC IDF1 with/without reranking. If reranking still hurts, disable and move on.

**Sweep parameters** if reranking helps:
- `k1`: [15, 20, 25, 30]
- `k2`: [3, 5, 6]
- `lambda_value`: [0.2, 0.3, 0.4]

---

## Phase 6: Update Kaggle Notebooks 10a/10b/10c

**Goal**: Update pipeline notebooks to include ensemble model, 384px, CID_BIAS.
**Dependencies**: Phases 1-4 complete, weights uploaded to Kaggle datasets.

### 6.1 Files to Modify

#### 6.1.1 `notebooks/kaggle/10a_stages012/` — Add secondary model extraction

**Key changes to the 10a notebook:**

1. **Add model weight downloads** for both models:
   ```python
   # In data input cell — ensure both weight files exist
   PRIMARY_WEIGHTS = "/kaggle/input/mtmc-weights/transreid_cityflowv2_384px_best.pth"
   SECONDARY_WEIGHTS = "/kaggle/input/mtmc-weights/resnet101ibn_cityflowv2_384px_best.pth"
   ```

2. **Update config overrides cell**:
   ```python
   overrides = [
       # Primary model at 384px
       "stage2.reid.vehicle.weights_path=" + PRIMARY_WEIGHTS,
       "stage2.reid.vehicle.input_size=[384,384]",
       # Secondary model (ensemble)
       "stage2.reid.vehicle2.enabled=true",
       "stage2.reid.vehicle2.save_separate=true",
       "stage2.reid.vehicle2.model_name=resnet101_ibn_a",
       "stage2.reid.vehicle2.weights_path=" + SECONDARY_WEIGHTS,
       "stage2.reid.vehicle2.embedding_dim=2048",
       "stage2.reid.vehicle2.input_size=[384,384]",
       "stage2.reid.vehicle2.clip_normalization=false",
       # PCA
       "stage2.pca.secondary_pca_model_path=models/reid/pca_transform_secondary.pkl",
   ]
   ```

3. **Verify output files** at end of 10a:
   ```python
   import os
   stage2_dir = "data/outputs/run_latest/stage2"
   assert os.path.exists(f"{stage2_dir}/embeddings.npy"), "Primary embeddings missing!"
   assert os.path.exists(f"{stage2_dir}/embeddings_secondary.npy"), "Secondary embeddings missing!"
   print("✓ Both embedding files present")
   ```

#### 6.1.2 `notebooks/kaggle/10b_stage3/` — No changes needed

Stage 3 (FAISS indexing) operates only on primary embeddings. No changes.

#### 6.1.3 `notebooks/kaggle/10c_stages45/` — Add score-level fusion + CID_BIAS

**Key changes to the 10c notebook:**

1. **Config overrides**:
   ```python
   overrides = [
       # Secondary embeddings (from 10a output)
       "stage4.association.secondary_embeddings.path=data/outputs/run_latest/stage2/embeddings_secondary.npy",
       "stage4.association.secondary_embeddings.weight=0.4",
       # CID_BIAS
       "stage4.association.camera_bias.enabled=true",
       "stage4.association.camera_bias.cid_bias_npy_path=configs/datasets/cityflowv2_cid_bias.npy",
       # Reranking (test — disable if it hurts)
       "stage4.association.reranking.enabled=true",
       "stage4.association.reranking.k1=20",
       "stage4.association.reranking.k2=3",
       "stage4.association.reranking.lambda_value=0.3",
       # Similarity threshold (may need re-tuning with ensemble features)
       "stage4.association.graph.similarity_threshold=0.50",
   ]
   ```

2. **Ensure CID_BIAS NPY is accessible** — upload it as part of the Kaggle dataset:
   ```python
   # Bundle CID_BIAS with the weights dataset
   import shutil
   shutil.copy("configs/datasets/cityflowv2_cid_bias.npy", "/kaggle/working/")
   shutil.copy("configs/datasets/cityflowv2_cid_bias.json", "/kaggle/working/")
   ```

### 6.2 Kaggle Dataset Updates

Before pushing 10a/10c, update the `mrkdagods/mtmc-weights` dataset:

```bash
# From project root
cp models/reid/transreid_cityflowv2_384px_best.pth models/
cp models/reid/resnet101ibn_cityflowv2_384px_best.pth models/
cp configs/datasets/cityflowv2_cid_bias.npy models/
cp configs/datasets/cityflowv2_cid_bias.json models/
kaggle datasets version -p models/ -m "384px ViT + ResNet101-IBN-a + CID_BIAS"
```

### 6.3 Push Commands

```bash
# Push in order:
kaggle kernels push -p notebooks/kaggle/10a_stages012/
# Wait for 10a to complete, then:
kaggle kernels push -p notebooks/kaggle/10b_stage3/
# Wait for 10b, then:
kaggle kernels push -p notebooks/kaggle/10c_stages45/

# Monitor:
python scripts/kaggle_logs.py ali369/mtmc-10a-stages-0-2-tracking-reid-features --tail 50
```

---

## Config Summary

### `configs/default.yaml` Changes (all phases)

```yaml
# Phase 1: No default.yaml changes

# Phase 2: New vehicle2 default
stage2:
  reid:
    vehicle2:
      enabled: false
      save_separate: true
      model_name: "resnet101_ibn_a"   # CHANGED from osnet_x1_0
      weights_path: "models/reid/resnet101ibn_cityflowv2_384px_best.pth"  # CHANGED
      embedding_dim: 2048              # CHANGED from 512
      input_size: [384, 384]           # CHANGED from [256, 128]
      clip_normalization: false
  pca:
    secondary_pca_model_path: "models/reid/pca_transform_secondary.pkl"  # NEW

# Phase 4: CID_BIAS
stage4:
  association:
    camera_bias:
      enabled: false
      iterations: 1
      cid_bias_npy_path: ""  # NEW
```

### `configs/datasets/cityflowv2.yaml` Changes (all phases)

```yaml
# Phase 3: Ensemble
stage2:
  reid:
    vehicle:
      input_size: [384, 384]   # CHANGED from [256, 256]
      weights_path: "models/reid/transreid_cityflowv2_384px_best.pth"  # CHANGED
    vehicle2:
      enabled: true             # CHANGED
      save_separate: true
      model_name: "resnet101_ibn_a"
      weights_path: "models/reid/resnet101ibn_cityflowv2_384px_best.pth"
      embedding_dim: 2048
      input_size: [384, 384]
      clip_normalization: false

# Phase 4: CID_BIAS
stage4:
  association:
    secondary_embeddings:
      path: "data/outputs/run_latest/stage2/embeddings_secondary.npy"
      weight: 0.4
    camera_bias:
      enabled: true
      cid_bias_npy_path: "configs/datasets/cityflowv2_cid_bias.npy"
      iterations: 0

    # Phase 5: Reranking
    reranking:
      enabled: true
      k1: 20
      k2: 3
      lambda_value: 0.3
```

---

## Dependency Graph

```
Phase 1 ──────────────────┐
(384px ViT training)      │
                          ├──→ Phase 3 ──→ Phase 5 ──→ Phase 6
Phase 2 ──────────────────┘    (Ensemble)  (Reranking)  (Kaggle 10a/b/c)
(ResNet101-IBN-a training)          │
                                    ├──→ Phase 4
                                    │    (CID_BIAS)
                                    │         │
                                    └─────────┘
```

**Parallel work**:
- Phase 1 and Phase 2 can run simultaneously (different Kaggle notebooks)
- Phase 4 (CID_BIAS computation script) can be developed while Phase 1/2 train
- Phase 3 code changes can be developed while Phase 1/2 train

**Serial dependencies**:
- Phase 3 needs Phase 1 + Phase 2 weights
- Phase 5 needs Phase 3 features to test effectively
- Phase 6 needs all phases complete

---

## Success Criteria

| Milestone | MTMC IDF1 | Validation |
|-----------|:---------:|-----------|
| Phase 1 complete | ≥ 79% | ViT 384px mAP ≥ 0.55 on CityFlowV2 val |
| Phase 2 complete | — | ResNet101-IBN mAP ≥ 0.50 on CityFlowV2 val |
| Phase 3 complete | ≥ 81% | Ensemble MTMC IDF1 on local pipeline |
| Phase 4 complete | ≥ 82% | CID_BIAS + ensemble MTMC IDF1 |
| Phase 5 complete | ≥ 82.5% | Reranking helps (if not, disable) |
| Phase 6 complete | ≥ 84% | Kaggle 10c submission IDF1 |

---

## Files Summary

### Files to CREATE:
| File | Phase | Purpose |
|------|:-----:|---------|
| `notebooks/kaggle/09d_vehicle_reid_resnet101ibn/09d_vehicle_reid_resnet101ibn.ipynb` | 2 | ResNet101-IBN-a training |
| `notebooks/kaggle/09d_vehicle_reid_resnet101ibn/kernel-metadata.json` | 2 | Kaggle metadata |
| `scripts/compute_cid_bias.py` | 4 | CID_BIAS computation from GT |

### Files to MODIFY:
| File | Phase | Changes |
|------|:-----:|---------|
| `src/training/model.py` | 2 | Add `IBN_a`, `GeM`, `ReIDModelResNet101IBN` classes |
| `src/training/train_reid.py` | 1,2 | Circle loss alongside triplet, cosine scheduler, ResNet101 support |
| `src/training/datasets.py` | 1 | Color jitter in `build_train_transforms` |
| `src/stage2_features/reid_model.py` | 3 | Add `_build_resnet101_ibn` method |
| `src/stage2_features/pipeline.py` | 3 | Separate PCA for secondary embeddings |
| `src/stage4_association/pipeline.py` | 4 | NPY-based CID_BIAS loading and application |
| `configs/default.yaml` | 2,3,4 | Vehicle2 config, secondary PCA path, CID_BIAS path |
| `configs/datasets/cityflowv2.yaml` | 3,4,5 | 384px input, ensemble, CID_BIAS, reranking |
| `notebooks/kaggle/09b_vehicle_reid_384px/09b_vehicle_reid_384px.ipynb` | 1 | Complete rewrite for 384px training |
| `notebooks/kaggle/09b_vehicle_reid_384px/kernel-metadata.json` | 1 | Update dataset sources |
| `notebooks/kaggle/10a_stages012/` notebook | 6 | Add secondary model config |
| `notebooks/kaggle/10c_stages45/` notebook | 6 | Add score-fusion + CID_BIAS config |

### Files UNCHANGED:
- `src/training/losses.py` — CircleLoss already correct (m=0.25, γ=64 default, but we pass γ=80 from config)
- `src/stage4_association/reranking.py` — Already implemented correctly
- `src/stage4_association/camera_bias.py` — Iterative learning stays as fallback
- `src/stage2_features/transreid_model.py` — 384px position embedding interpolation already works
- `src/stage4_association/similarity.py` — Score-level fusion already in stage4 pipeline