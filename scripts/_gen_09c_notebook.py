"""Generate notebooks/kaggle/09c_kd_vitl_teacher/09c_kd_vitl_teacher.ipynb

Knowledge Distillation: ViT-L/14-CLIP (teacher) → ViT-B/16-CLIP (student).

Strategy:
  - Teacher: frozen vit_large_patch14_clip_224.openai + 5-epoch lightweight ID head
  - Student: init from CityFlowV2 checkpoint, trained with CE+triplet+KD
  - Input: 256×256 student, 224×224 teacher (same crop, two sizes)
  - KD loss: KL(student_logits/T, teacher_logits/T)*T^2 + cosine_feat(student_proj, teacher_cls)
  - Total loss: (1-alpha)*L_task + alpha*L_kd_logit + beta*L_kd_feat
"""
import json
from pathlib import Path

OUT_DIR = Path("notebooks/kaggle/09c_kd_vitl_teacher")
OUT_FILE = OUT_DIR / "09c_kd_vitl_teacher.ipynb"


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
cells.append(cell('markdown', r'''# Notebook 09c: Knowledge Distillation — ViT-L/14-CLIP → ViT-B/16-CLIP

**Strategy**:
1. Load **frozen** ViT-L/14-CLIP teacher (pretrained on 400M image-text pairs)
2. Warm up a lightweight **ID classification head** on the frozen teacher (5 epochs)
3. **Distill** the teacher's knowledge into our ViT-B/16 student using:
   - KL-divergence on soft logit targets (temperature T=4)
   - Cosine feature alignment between teacher CLS and projected student CLS
4. Export the distilled student as `transreid_cityflowv2_kd_best.pth`

**Expected gain**: +1-2pp IDF1 (ViT-L/14-CLIP encodes 5× richer visual features than ViT-B)
**Compute**: ~30 min teacher head warmup + ~4h student KD = ~5h total on T4'''))

# ── Cell 1: Install ────────────────────────────────────────────────────────────
cells.append(cell('code', '!pip install -q timm matplotlib pandas loguru gdown'))

# ── Cell 2: Setup md ──────────────────────────────────────────────────────────
cells.append(cell('markdown', '## 1. Setup & GPU'))

# ── Cell 3: Setup code ────────────────────────────────────────────────────────
cells.append(cell('code', r'''import os
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"PyTorch  : {torch.__version__}")
NUM_GPUS = torch.cuda.device_count()
print(f"GPUs     : {NUM_GPUS}")
for i in range(NUM_GPUS):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({p.total_memory/1024**3:.1f} GB)")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device   : {DEVICE}")

PROJECT = Path("/kaggle/working/gp")
WORK_DIR = Path("/kaggle/working")'''))

# ── Cell 4: Load checkpoint paths ─────────────────────────────────────────────
cells.append(cell('markdown', '## 2. Locate Pretrained Checkpoint'))

cells.append(cell('code', r'''STUDENT_PATH = None

search_paths = [
    # Prefer 09b trained 384px model (higher quality init)
    Path("/kaggle/input/datasets/mrkdagods/mtmc-weights/models/reid/transreid_cityflowv2_384_best.pth"),
    Path("/kaggle/input/mtmc-weights/models/reid/transreid_cityflowv2_384_best.pth"),
    # Fall back to standard 256px CityFlowV2 model
    Path("/kaggle/input/datasets/mrkdagods/mtmc-weights/models/reid/transreid_cityflowv2_best.pth"),
    Path("/kaggle/input/mtmc-weights/models/reid/transreid_cityflowv2_best.pth"),
]

for p in search_paths:
    if p.exists():
        STUDENT_PATH = p
        break

if STUDENT_PATH is not None:
    print(f"Student init checkpoint: {STUDENT_PATH}")
    ckpt = torch.load(str(STUDENT_PATH), map_location="cpu", weights_only=False)
    source_state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Detect input size from pos_embed shape
    pe = source_state.get("vit.pos_embed", None)
    if pe is not None:
        n_patches = pe.shape[1] - 1  # subtract CLS token
        side = int(n_patches ** 0.5)
        input_px = side * 16  # patch size 16
        print(f"  Student pos_embed: {pe.shape} → detected input size {input_px}×{input_px}")
    else:
        input_px = 256
        print("  Could not detect input size, assuming 256px")
    STUDENT_INPUT_PX = input_px
else:
    source_state = None
    STUDENT_INPUT_PX = 256
    print("WARNING: No student checkpoint found — will train from CLIP scratch (slower convergence)")

print(f"Student input size: {STUDENT_INPUT_PX}×{STUDENT_INPUT_PX}")'''))

# ── Cell 5: Download CityFlowV2 ───────────────────────────────────────────────
cells.append(cell('markdown', '## 3. Download & Prepare CityFlowV2'))

cells.append(cell('code', r'''import zipfile

CITYFLOW_DIR = Path("/tmp/cityflowv2")
GDRIVE_ID    = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"
ARCHIVE_NAME = "AIC22_Track1_MTMC_Tracking.zip"
ALLOWED_SPLITS = {"train", "validation"}

already_found = False
for check_dir in [CITYFLOW_DIR,
                  Path("/kaggle/input/data-aicity-2023-track-2"),
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
        import gdown
        MAX_RETRIES = 3
        for _attempt in range(1, MAX_RETRIES + 1):
            print(f"Downloading CityFlowV2 attempt {_attempt}/{MAX_RETRIES} (id={GDRIVE_ID})...")
            try:
                gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", str(archive_path), quiet=False)
            except Exception as e:
                print(f"  gdown error: {e}")
            if archive_path.exists() and archive_path.stat().st_size > 1e9:
                print(f"  Download OK ({archive_path.stat().st_size/1e9:.2f} GB)")
                break
            if archive_path.exists():
                archive_path.unlink()
            if _attempt < MAX_RETRIES:
                print(f"  Retrying in 30s...")
                time.sleep(30)
        if not archive_path.exists() or archive_path.stat().st_size < 1e9:
            raise RuntimeError("Failed to download CityFlowV2 after all retries")
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
        cam_dir   = vdo_path.parent
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

# Discover camera dirs via rglob (handles extra train/validation level)
cam_dirs = sorted({vdo.parent for vdo in CITYFLOW_DIR.rglob("vdo.avi")})
cameras = sorted({d.name for d in cam_dirs})
print(f"Found {len(cameras)} cameras across {len(cam_dirs)} dirs: {cameras[:6]}...")'''))

# ── Cell 6: Extract crops ─────────────────────────────────────────────────────
cells.append(cell('markdown', '## 4. Extract Training Crops'))

cells.append(cell('code', r'''H = W = STUDENT_INPUT_PX   # match student native resolution
MAX_CROPS_PER_ID_PER_CAM = 32
MIN_AREA     = 2500
MIN_BBOX_SIDE = 48
TEACHER_SZ   = 224   # ViT-L/14-CLIP native input

print(f"Extracting crops at {H}×{W} (student), {TEACHER_SZ}×{TEACHER_SZ} (teacher)")

all_crops = []    # list of (img_path, id_str, cam_str)
CROP_DIR  = Path("/tmp/crops_kd")
CROP_DIR.mkdir(parents=True, exist_ok=True)

cam_dirs = sorted({vdo.parent for vdo in CITYFLOW_DIR.rglob("vdo.avi")})

for cam_dir in cam_dirs:
    gt_file = cam_dir / "gt.txt"
    if not gt_file.exists():
        gt_file = cam_dir / "gt" / "gt.txt"
    if not gt_file.exists():
        continue
    vid_file = cam_dir / "vdo.avi"
    if not vid_file.exists():
        continue

    gt_data = defaultdict(list)   # frame -> [(id, x,y,w,h)]
    for line in gt_file.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        fr, tid, x, y, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        if bw * bh < MIN_AREA or bw < MIN_BBOX_SIDE or bh < MIN_BBOX_SIDE:
            continue
        gt_data[fr].append((tid, x, y, bw, bh))

    id_cam_count = defaultdict(int)
    cap = cv2.VideoCapture(str(vid_file))
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx not in gt_data:
            continue
        h_frame, w_frame = frame.shape[:2]
        for (tid, x, y, bw, bh) in gt_data[frame_idx]:
            key = (tid, cam_dir.name)
            if id_cam_count[key] >= MAX_CROPS_PER_ID_PER_CAM:
                continue
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(w_frame, x + bw); y2 = min(h_frame, y + bh)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            fname = CROP_DIR / f"{cam_dir.name}_id{tid:04d}_f{frame_idx:06d}.jpg"
            crop_pil.resize((W, H), Image.BICUBIC).save(str(fname), quality=90)
            all_crops.append((str(fname), str(tid), cam_dir.name))
            id_cam_count[key] += 1
            saved += 1
    cap.release()
    print(f"  {cam_dir.name}: {saved} crops")

print(f"\nTotal crops: {len(all_crops)}")
ids  = list({c[1] for c in all_crops})
print(f"Unique IDs : {len(ids)}")'''))

# ── Cell 7: Train/eval splits ─────────────────────────────────────────────────
cells.append(cell('markdown', '## 5. Train / Eval Splits'))

cells.append(cell('code', r'''TRAIN_RATIO = 0.7
import random
random.seed(42)

id_to_crops = defaultdict(list)
for path, tid, cam in all_crops:
    id_to_crops[tid].append((path, tid, cam))

all_ids = sorted(id_to_crops.keys(), key=lambda x: int(x))
n_train_ids = int(len(all_ids) * TRAIN_RATIO)
train_ids = set(all_ids[:n_train_ids])
query_ids = set(all_ids[n_train_ids:])

train_crops, query_crops, gallery_crops = [], [], []
for tid, crops in id_to_crops.items():
    if tid in train_ids:
        train_crops.extend(crops)
    else:
        n_q = max(1, len(crops) // 4)
        cams = list({c[2] for c in crops})
        q_cam = cams[0]
        q_list = [c for c in crops if c[2] == q_cam][:n_q]
        g_list = [c for c in crops if c not in q_list]
        query_crops.extend(q_list)
        gallery_crops.extend(g_list)

num_classes = len(train_ids)
id2label = {tid: i for i, tid in enumerate(sorted(train_ids, key=lambda x: int(x)))}
print(f"Train IDs  : {len(train_ids)}  (crops: {len(train_crops)})")
print(f"Query IDs  : {len(query_ids)}")
print(f"Query crops: {len(query_crops)},  Gallery: {len(gallery_crops)}")
print(f"num_classes: {num_classes}")'''))

# ── Cell 8: Dataset + DataLoaders ────────────────────────────────────────────
cells.append(cell('markdown', '## 6. Dataset & DataLoaders'))

cells.append(cell('code', r'''CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

train_tf_student = T.Compose([
    T.RandomHorizontalFlip(),
    T.Pad(10),
    T.RandomCrop(STUDENT_INPUT_PX),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])
# Teacher always gets 224×224
train_tf_teacher = T.Compose([
    T.RandomHorizontalFlip(),
    T.Pad(10),
    T.RandomCrop(STUDENT_INPUT_PX),     # same spatial crop as student...
    T.Resize((TEACHER_SZ, TEACHER_SZ)), # ...then resize to teacher size
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])
eval_tf_student = T.Compose([
    T.Resize((STUDENT_INPUT_PX, STUDENT_INPUT_PX)),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])
eval_tf_teacher = T.Compose([
    T.Resize((TEACHER_SZ, TEACHER_SZ)),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


class KDCropDataset(Dataset):
    """Returns (img_student, img_teacher, label_id, cam_id) for KD training."""
    def __init__(self, crops, id2label, tf_student, tf_teacher):
        self.crops      = crops
        self.id2label   = id2label
        self.tf_student = tf_student
        self.tf_teacher = tf_teacher

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        path, tid, cam = self.crops[idx]
        img = Image.open(path).convert("RGB")
        # Apply same random seed so student+teacher get same spatial crop
        seed = int(torch.randint(0, 2**31, (1,)))
        torch.manual_seed(seed)
        img_s = self.tf_student(img)
        torch.manual_seed(seed)
        img_t = self.tf_teacher(img)
        label = self.id2label.get(tid, 0)
        return img_s, img_t, label


class SimpleDataset(Dataset):
    """Single-view dataset for eval."""
    def __init__(self, crops, tf):
        self.crops = crops
        self.tf    = tf

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        path, tid, cam = self.crops[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), int(tid), cam


class PKSampler(Sampler):
    """P identities * K images per identity per batch."""
    def __init__(self, crops, id2label, P=16, K=4):
        self.id2label = id2label
        self.id2idx   = defaultdict(list)
        for i, (_, tid, _) in enumerate(crops):
            if tid in id2label:
                self.id2idx[tid].append(i)
        self.pids = [tid for tid in self.id2idx if len(self.id2idx[tid]) >= 2]
        self.P, self.K = P, K
        self.n_batches = max(1, len(self.pids) // P)

    def __iter__(self):
        pids = self.pids.copy()
        random.shuffle(pids)
        for start in range(0, self.n_batches * self.P, self.P):
            batch_pids = pids[start:start + self.P]
            indices = []
            for pid in batch_pids:
                pool = self.id2idx[pid]
                indices += random.choices(pool, k=self.K)
            yield indices

    def __len__(self):
        return self.n_batches


BATCH_P, BATCH_K = 8, 4
batch_size = BATCH_P * BATCH_K * max(1, NUM_GPUS)
pk_sampler = PKSampler(train_crops, id2label, P=BATCH_P, K=BATCH_K)
train_ds   = KDCropDataset(train_crops, id2label, train_tf_student, train_tf_teacher)
train_loader = DataLoader(train_ds, batch_sampler=pk_sampler, num_workers=2, pin_memory=True)

query_ds   = SimpleDataset(query_crops,   eval_tf_student)
gallery_ds = SimpleDataset(gallery_crops, eval_tf_student)
query_loader   = DataLoader(query_ds,   batch_size=32, num_workers=2)
gallery_loader = DataLoader(gallery_ds, batch_size=32, num_workers=2)

# Teacher eval loaders (224x224 for ViT-L/14)
query_ds_t   = SimpleDataset(query_crops,   eval_tf_teacher)
gallery_ds_t = SimpleDataset(gallery_crops, eval_tf_teacher)
query_loader_teacher   = DataLoader(query_ds_t,   batch_size=32, num_workers=2)
gallery_loader_teacher = DataLoader(gallery_ds_t, batch_size=32, num_workers=2)

print(f"batch_size  : {batch_size}")
print(f"train_loader: {len(train_loader)} batches")'''))

# ── Cell 9: Losses ────────────────────────────────────────────────────────────
cells.append(cell('markdown', '## 7. Loss Functions'))

cells.append(cell('code', r'''class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=1)
        one_hot  = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1)
        smooth   = (1 - self.epsilon) * one_hot + self.epsilon / self.num_classes
        loss     = -(smooth * log_prob).sum(dim=1).mean()
        return loss


class TripletLossHardMining(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings, p=2)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & \
                   ~torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
        neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        ap = dist.clone(); ap[~pos_mask] = -1e9
        an = dist.clone(); an[~neg_mask] =  1e9
        hardest_pos = ap.max(dim=1)[0]
        hardest_neg = an.min(dim=1)[0]
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss[pos_mask.any(dim=1)].mean()


class KDLoss(nn.Module):
    """Knowledge distillation loss combining logit KD + feature alignment."""
    def __init__(self, temperature=4.0, alpha=0.5, beta=0.5):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha  # weight for logit KD
        self.beta  = beta   # weight for feature alignment

    def logit_kd(self, student_logits, teacher_logits):
        """KL divergence on temperature-scaled logits (Hinton et al. 2015)."""
        s = F.log_softmax(student_logits / self.T, dim=1)
        t = F.softmax(teacher_logits.detach() / self.T, dim=1)
        return F.kl_div(s, t, reduction='batchmean') * (self.T ** 2)

    def feat_align(self, student_feat, teacher_feat_proj):
        """1 - cosine similarity between projected student and teacher CLS."""
        sf = F.normalize(student_feat, dim=1)
        tf = F.normalize(teacher_feat_proj.detach(), dim=1)
        return (1 - (sf * tf).sum(dim=1)).mean()

    def forward(self, s_logits, t_logits, s_feat, t_feat_proj):
        l_logit = self.logit_kd(s_logits, t_logits)
        l_feat  = self.feat_align(s_feat, t_feat_proj)
        return self.alpha * l_logit + self.beta * l_feat'''))

# ── Cell 10: Eval function ────────────────────────────────────────────────────
cells.append(cell('markdown', '## 8. Evaluation'))

cells.append(cell('code', r'''@torch.no_grad()
def extract_features(model, loader, device="cuda", flip=True, is_kd_model=False):
    """Extract L2-normalised CLS embeddings from a loader."""
    model.eval()
    feats, pids, cams = [], [], []
    for batch in loader:
        imgs = batch[0].to(device)
        labels = batch[1] if len(batch) > 1 else None
        cam_list = batch[2] if len(batch) > 2 else None
        if flip:
            imgs_flip = torch.flip(imgs, dims=[3])
            if is_kd_model:
                f1 = model.get_student_feat(imgs)
                f2 = model.get_student_feat(imgs_flip)
            else:
                out1 = model(imgs)
                out2 = model(imgs_flip)
                # TeacherReID returns (feat, logits); unwrap if needed
                f1 = out1[0] if isinstance(out1, tuple) else out1
                f2 = out2[0] if isinstance(out2, tuple) else out2
            f = F.normalize((f1 + f2) / 2, dim=1)
        else:
            if is_kd_model:
                f = F.normalize(model.get_student_feat(imgs), dim=1)
            else:
                out = model(imgs)
                feat = out[0] if isinstance(out, tuple) else out
                f = F.normalize(feat, dim=1)
        feats.append(f.cpu())
        if labels is not None:
            pids.extend(labels.tolist() if hasattr(labels, 'tolist') else [int(x) for x in labels])
        if cam_list is not None:
            cams.extend(list(cam_list))
    return torch.cat(feats, dim=0), pids, cams


def eval_market1501(q_feat, q_pids, q_cams, g_feat, g_pids, g_cams):
    """Returns (mAP, rank-1) using CityFlowV2 cross-cam protocol."""
    q_feat = F.normalize(q_feat, dim=1)
    g_feat = F.normalize(g_feat, dim=1)
    sim    = torch.mm(q_feat, g_feat.t())
    mAP = rank1 = 0.0
    n_query = q_feat.shape[0]
    for qi in range(n_query):
        qp, qc = q_pids[qi], q_cams[qi]
        scores = sim[qi].numpy()
        # Mask same-camera same-id
        mask = np.array([(g_pids[i] == qp and g_cams[i] == qc) for i in range(len(g_pids))])
        valid_idx = np.where(~mask)[0]
        if len(valid_idx) == 0:
            continue
        v_scores = scores[valid_idx]
        v_pids   = [g_pids[i] for i in valid_idx]
        order    = np.argsort(-v_scores)
        matches  = np.array([v_pids[i] == qp for i in order], dtype=float)
        if matches.sum() == 0:
            continue
        cum = np.cumsum(matches)
        prec_at_k = cum / (np.arange(len(matches)) + 1)
        ap = (prec_at_k * matches).sum() / matches.sum()
        mAP += ap
        if matches[order][0]:
            rank1 += 1
    mAP   /= n_query
    rank1 /= n_query
    return mAP, rank1


def evaluate(model, q_loader, g_loader, device, is_kd_model=False):
    q_feat, q_pids, q_cams = extract_features(model, q_loader, device, is_kd_model=is_kd_model)
    g_feat, g_pids, g_cams = extract_features(model, g_loader, device, is_kd_model=is_kd_model)
    return eval_market1501(q_feat, q_pids, q_cams, g_feat, g_pids, g_cams)'''))

# ── Cell 11: Teacher model ────────────────────────────────────────────────────
cells.append(cell('markdown', '## 9. Teacher Model (ViT-L/14-CLIP)'))

cells.append(cell('code', r'''import timm

TEACHER_MODEL = "vit_large_patch14_clip_224.openai"
TEACHER_EMB_DIM = 1024

class TeacherReID(nn.Module):
    """Frozen ViT-L/14-CLIP backbone with a trainable ID classifier head."""
    def __init__(self, num_classes, emb_dim=TEACHER_EMB_DIM):
        super().__init__()
        self.vit = timm.create_model(TEACHER_MODEL, pretrained=True, num_classes=0)
        self.emb_dim = emb_dim
        # BNNeck for the teacher (same recipe as TransReID)
        self.bnneck = nn.BatchNorm1d(emb_dim)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(emb_dim, num_classes, bias=False)

    def freeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad_(False)
        print("Teacher backbone frozen")

    def forward(self, x):
        """Returns (cls_feat, logits). cls_feat is pre-BN for KD signal."""
        cls_raw = self.vit(x)                    # [B, 1024]
        cls_bn  = self.bnneck(cls_raw)           # [B, 1024]
        logits  = self.classifier(cls_bn)        # [B, num_classes]
        return cls_raw, logits


teacher = TeacherReID(num_classes).to(DEVICE)
teacher.freeze_backbone()
n_params_total    = sum(p.numel() for p in teacher.parameters()) / 1e6
n_params_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad) / 1e6
print(f"Teacher params: {n_params_total:.1f}M total, {n_params_trainable:.2f}M trainable (head only)")'''))

# ── Cell 12: Teacher head warmup ─────────────────────────────────────────────
cells.append(cell('markdown', '## 10. Stage 1 — Teacher Head Warmup (5 epochs)'))

cells.append(cell('code', r'''# Only the BNNeck + classifier head are trained; backbone is frozen.
# This is fast (~20-30 min) and gives the teacher calibrated ID logits.
TEACHER_EPOCHS = 5
TEACHER_LR     = 1e-3

teacher_ce  = CrossEntropyLabelSmooth(num_classes, 0.05).to(DEVICE)
teacher_tri = TripletLossHardMining(margin=0.3).to(DEVICE)
t_opt = torch.optim.Adam(
    [p for p in teacher.parameters() if p.requires_grad],
    lr=TEACHER_LR, weight_decay=5e-4
)
t_sched = torch.optim.lr_scheduler.CosineAnnealingLR(t_opt, T_max=TEACHER_EPOCHS, eta_min=1e-5)

teacher.train()
t_history = {"loss": [], "mAP": [], "rank1": []}
best_t_path = Path("/tmp/teacher_best.pth")

print(f"Stage 1: training teacher head for {TEACHER_EPOCHS} epochs...")
for epoch in range(1, TEACHER_EPOCHS + 1):
    teacher.train()
    epoch_loss = 0.0
    for imgs_s, imgs_t, labels in train_loader:
        # Teacher uses the 224px images (imgs_t)
        imgs_t = imgs_t.to(DEVICE, non_blocking=True)
        labels_gpu = labels.to(DEVICE, non_blocking=True)

        cls_raw, logits = teacher(imgs_t)
        l_ce  = teacher_ce(logits, labels_gpu)
        l_tri = teacher_tri(F.normalize(cls_raw, dim=1), labels_gpu)
        loss  = l_ce + l_tri

        t_opt.zero_grad()
        loss.backward()
        t_opt.step()
        epoch_loss += loss.item()

    t_sched.step()
    avg_loss = epoch_loss / len(train_loader)
    t_history["loss"].append(avg_loss)

    if epoch % 2 == 0 or epoch == TEACHER_EPOCHS:
        mAP, rank1 = evaluate(teacher, query_loader_teacher, gallery_loader_teacher, DEVICE)
        t_history["mAP"].append(mAP)
        t_history["rank1"].append(rank1)
        print(f"  [T Epoch {epoch:2d}] loss={avg_loss:.4f}  mAP={mAP:.4f}  R1={rank1:.4f}")
    else:
        print(f"  [T Epoch {epoch:2d}] loss={avg_loss:.4f}")

torch.save(teacher.state_dict(), str(best_t_path))
print(f"\nTeacher head warmup done. Freezing entire teacher now.")
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval()
print(f"Final teacher stats: mAP={t_history['mAP'][-1]:.4f}  R1={t_history['rank1'][-1]:.4f}")'''))

# ── Cell 13: Student model ────────────────────────────────────────────────────
cells.append(cell('markdown', '## 11. Student Model (ViT-B/16-CLIP) + Feature Projection'))

cells.append(cell('code', r'''VIT_MODEL_STUDENT = "vit_base_patch16_clip_224.openai"
STUDENT_EMB_DIM  = 768


def interpolate_pos_embed_student(model_vit, target_px, patch_size=16):
    """Interpolate position embedding from 224px to target_px if needed."""
    if target_px == 224:
        return
    pos_embed = model_vit.pos_embed  # (1, N+1, D)
    N = pos_embed.shape[1] - 1
    src_side = int(N ** 0.5)
    tgt_side = target_px // patch_size

    if src_side == tgt_side:
        return

    D = pos_embed.shape[2]
    cls_token = pos_embed[:, :1, :]
    patch_embed = pos_embed[:, 1:, :].reshape(1, src_side, src_side, D).permute(0, 3, 1, 2)
    patch_embed = F.interpolate(patch_embed, size=(tgt_side, tgt_side), mode="bicubic", align_corners=False)
    patch_embed = patch_embed.permute(0, 2, 3, 1).flatten(1, 2)
    model_vit.pos_embed = nn.Parameter(torch.cat([cls_token, patch_embed], dim=1))
    print(f"  Interpolated student pos_embed: {pos_embed.shape} -> {model_vit.pos_embed.shape}")


class StudentReID(nn.Module):
    """ViT-B/16-CLIP student with BNNeck + ID classifier + to-teacher projection."""
    def __init__(self, num_classes, emb_dim=STUDENT_EMB_DIM, teacher_dim=TEACHER_EMB_DIM):
        super().__init__()
        self.vit = timm.create_model(VIT_MODEL_STUDENT, pretrained=True, num_classes=0)
        # Interpolate pos_embed if student trained at different resolution
        interpolate_pos_embed_student(self.vit, STUDENT_INPUT_PX)
        self.emb_dim = emb_dim
        # BNNeck
        self.bnneck = nn.BatchNorm1d(emb_dim)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.bnneck.bias.requires_grad_(False)
        # ID classifier
        self.classifier = nn.Linear(emb_dim, num_classes, bias=False)
        # Feature projection to teacher space (for feature KD alignment)
        self.feat_proj = nn.Sequential(
            nn.Linear(emb_dim, teacher_dim, bias=False),
            nn.LayerNorm(teacher_dim),
        )

    def forward(self, x):
        """Returns (cls_raw, cls_bn, logits, feat_proj)."""
        cls_raw  = self.vit(x)                     # [B, 768]
        cls_bn   = self.bnneck(cls_raw)             # [B, 768]
        logits   = self.classifier(cls_bn)          # [B, num_classes]
        feat_proj = self.feat_proj(cls_raw)         # [B, 1024] - projects to teacher space
        return cls_raw, cls_bn, logits, feat_proj

    def get_student_feat(self, x):
        """Inference: returns L2-normed BN features."""
        cls_raw = self.vit(x)
        cls_bn  = self.bnneck(cls_raw)
        return F.normalize(cls_bn, dim=1)


student = StudentReID(num_classes).to(DEVICE)

# Load CityFlowV2 pretrained weights into student ViT backbone
if source_state is not None:
    vit_prefix = "vit."
    mapped = {}
    for k, v in source_state.items():
        if k.startswith(vit_prefix):
            new_k = k[len(vit_prefix):]
            mapped[new_k] = v
    # Also load BNNeck and classifier if they exist
    bn_state = {k.replace("bnneck.", ""): v for k, v in source_state.items() if k.startswith("bnneck.")}
    cls_state = {k.replace("classifier.", ""): v for k, v in source_state.items() if k.startswith("classifier.")}

    # Load ViT weights
    missing, unexpected = student.vit.load_state_dict(mapped, strict=False)
    print(f"Student ViT loaded: {len(mapped)-len(missing)} weights, {len(missing)} missing, {len(unexpected)} unexpected")
    # Load BNNeck
    if bn_state:
        student.bnneck.load_state_dict(bn_state, strict=False)
        print(f"  BNNeck loaded: {len(bn_state)} weights")
    # Load classifier (may be different num_classes for fine-tuned model -- skip if mismatch)
    if cls_state:
        try:
            student.classifier.load_state_dict(cls_state, strict=True)
            print(f"  Classifier loaded: {len(cls_state)} weights")
        except RuntimeError as e:
            print(f"  Classifier shape mismatch (expected for retrained head): {e}")
    print("Student initialised from pretrained checkpoint")
else:
    print("Student initialised from CLIP pretrain only (no CityFlowV2 checkpoint)")

n_student = sum(p.numel() for p in student.parameters()) / 1e6
print(f"Student parameters: {n_student:.1f}M")'''))

# ── Cell 14: KD Training ─────────────────────────────────────────────────────
cells.append(cell('markdown', '## 12. Stage 2 — Knowledge Distillation Training'))

cells.append(cell('code', r'''# Hyperparameters
KD_EPOCHS    = 40
BACKBONE_LR  = 1e-5    # low LR — fine-tuning already-trained backbone
HEAD_LR      = 1e-4    # higher LR for classification head + proj
WARMUP_EP    = 5
KD_ALPHA     = 0.5     # weight for KD logit loss
KD_BETA      = 0.5     # weight for feature alignment loss
KD_TEMP      = 4.0     # distillation temperature

# Losses
ce_loss  = CrossEntropyLabelSmooth(num_classes, 0.05).to(DEVICE)
tri_loss = TripletLossHardMining(margin=0.3).to(DEVICE)
kd_loss  = KDLoss(temperature=KD_TEMP, alpha=KD_ALPHA, beta=KD_BETA).to(DEVICE)

# Optimizer: separate LR groups
backbone_params = list(student.vit.parameters())
head_params     = (list(student.bnneck.parameters()) +
                   list(student.classifier.parameters()) +
                   list(student.feat_proj.parameters()))
optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": BACKBONE_LR},
    {"params": head_params,     "lr": HEAD_LR},
], weight_decay=5e-4)

def warmup_lr(epoch):
    if epoch < WARMUP_EP:
        return (epoch + 1) / WARMUP_EP
    progress = (epoch - WARMUP_EP) / (KD_EPOCHS - WARMUP_EP)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)

history    = {"loss": [], "loss_task": [], "loss_kd": [], "mAP": [], "rank1": []}
best_mAP   = 0.0
best_state_path = Path("/tmp/student_kd_best.pth")

print(f"Stage 2: KD distillation for {KD_EPOCHS} epochs")
print(f"  alpha={KD_ALPHA} (logit KD)  beta={KD_BETA} (feat align)  T={KD_TEMP}")

ACCUM_STEPS = 2  # gradient accumulation to compensate for smaller batch

for epoch in range(1, KD_EPOCHS + 1):
    student.train()
    teacher.eval()

    ep_loss = ep_task = ep_kd = 0.0
    optimizer.zero_grad()
    for step_i, (imgs_s, imgs_t, labels) in enumerate(train_loader):
        imgs_s = imgs_s.to(DEVICE, non_blocking=True)
        imgs_t = imgs_t.to(DEVICE, non_blocking=True)
        labels_gpu = labels.to(DEVICE, non_blocking=True)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_cls_raw, t_logits = teacher(imgs_t)

        # Student forward
        s_cls_raw, s_cls_bn, s_logits, s_feat_proj = student(imgs_s)

        # Task loss (CE + triplet on student)
        l_ce  = ce_loss(s_logits, labels_gpu)
        l_tri = tri_loss(F.normalize(s_cls_raw, dim=1), labels_gpu)
        l_task = l_ce + l_tri

        # KD loss (logit KD + feature alignment)
        # s_feat_proj maps 768D → 1024D to align with teacher CLS space
        l_kd = kd_loss(s_logits, t_logits, s_feat_proj, t_cls_raw)

        loss = ((1 - KD_ALPHA) * l_task + l_kd) / ACCUM_STEPS

        loss.backward()
        if (step_i + 1) % ACCUM_STEPS == 0 or (step_i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        ep_loss += loss.item() * ACCUM_STEPS
        ep_task += l_task.item()
        ep_kd   += l_kd.item()

    scheduler.step()
    n = len(train_loader)
    history["loss"].append(ep_loss / n)
    history["loss_task"].append(ep_task / n)
    history["loss_kd"].append(ep_kd / n)

    if epoch % 5 == 0 or epoch == KD_EPOCHS:
        mAP, rank1 = evaluate(student, query_loader, gallery_loader, DEVICE, is_kd_model=True)
        history["mAP"].append(mAP)
        history["rank1"].append(rank1)
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(student.state_dict(), str(best_state_path))
            print(f"  [Epoch {epoch:2d}] loss={ep_loss/n:.4f}  task={ep_task/n:.4f}  kd={ep_kd/n:.4f}  mAP={mAP:.4f} R1={rank1:.4f} ★")
        else:
            print(f"  [Epoch {epoch:2d}] loss={ep_loss/n:.4f}  task={ep_task/n:.4f}  kd={ep_kd/n:.4f}  mAP={mAP:.4f} R1={rank1:.4f}")
    else:
        print(f"  [Epoch {epoch:2d}] loss={ep_loss/n:.4f}  task={ep_task/n:.4f}  kd={ep_kd/n:.4f}")

print(f"\nBest student mAP: {best_mAP:.4f}")'''))

# ── Cell 15: Training curves ──────────────────────────────────────────────────
cells.append(cell('markdown', '## 13. Training Curves'))

cells.append(cell('code', r'''fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["loss"],      label="total")
axes[0].plot(history["loss_task"], label="task (CE+tri)")
axes[0].plot(history["loss_kd"],   label="KD")
axes[0].set_title("KD Student Loss")
axes[0].set_xlabel("Epoch"); axes[0].legend()

ep_eval = [i*5 for i in range(1, len(history["mAP"])+1)]
if len(ep_eval) > len(history["mAP"]):
    ep_eval = ep_eval[:len(history["mAP"])]
axes[1].plot(ep_eval, [m*100 for m in history["mAP"]], "b-o")
axes[1].set_title("Student mAP (%)"); axes[1].set_xlabel("Epoch")

axes[2].plot(ep_eval, [r*100 for r in history["rank1"]], "r-o")
axes[2].set_title("Student R1 (%)"); axes[2].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("/kaggle/working/kd_training_curves.png", dpi=100)
plt.show()
print("Curves saved.")'''))

# ── Cell 16: Export student ───────────────────────────────────────────────────
cells.append(cell('markdown', '## 14. Export Distilled Student'))

cells.append(cell('code', r'''export_dir = Path("/kaggle/working/exported_models")
export_dir.mkdir(parents=True, exist_ok=True)

# Load best checkpoint
best_state = torch.load(str(best_state_path), map_location="cpu", weights_only=False)
student.load_state_dict(best_state)
student.eval()

# Final evaluation
mAP_final, rank1_final = evaluate(student, query_loader, gallery_loader, DEVICE, is_kd_model=True)
print(f"Distilled student — mAP: {mAP_final:.4f}  R1: {rank1_final:.4f}")

# Build exportable state_dict (TransReID convention: prefix with vit. / bnneck. / classifier.)
export_state = {}
for k, v in student.vit.state_dict().items():
    export_state[f"vit.{k}"] = v
for k, v in student.bnneck.state_dict().items():
    export_state[f"bnneck.{k}"] = v
for k, v in student.classifier.state_dict().items():
    export_state[f"classifier.{k}"] = v
# feat_proj is NOT exported — only used during KD training

export_path = export_dir / "transreid_cityflowv2_kd_best.pth"
torch.save(export_state, str(export_path))

metadata = {
    "model_name":   "TransReID-KD",
    "architecture": VIT_MODEL_STUDENT,
    "teacher":      TEACHER_MODEL,
    "input_size":   STUDENT_INPUT_PX,
    "embedding_dim": STUDENT_EMB_DIM,
    "kd_alpha":     KD_ALPHA,
    "kd_beta":      KD_BETA,
    "kd_temperature": KD_TEMP,
    "teacher_epochs": TEACHER_EPOCHS,
    "student_epochs": KD_EPOCHS,
    "mAP":       mAP_final,
    "rank1":     rank1_final,
    "trained_on": "CityFlowV2",
    "created_at": datetime.now().isoformat(),
}
(export_dir / "transreid_cityflowv2_kd_metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print(f"Exported: {export_path}  ({export_path.stat().st_size/1024**2:.1f} MB)")
print(f"Metadata: {json.dumps(metadata, indent=2)}")'''))


# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md   = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"Written: {OUT_FILE}")
print(f"  Total cells: {len(cells)}  (code={n_code}, markdown={n_md})")
