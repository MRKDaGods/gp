"""ATHAR Phase 6 -- joint multi-domain vehicle ReID retrain (D4 tier 2).

ONE joint run over four domains (never sequential fine-tuning):
  VeRi-776   : abhyudaya12/veri-vehicle-re-identification-dataset (mounted)
  VehicleID  : maphat/vehicleid (mounted zip, PKU VehicleID_V1.0)
  VeRi-Wild  : mrkdagods/veriwild-train (staged tarbins, 277,797 train imgs)
  CityFlowV2 : GDrive archive, GT crops from the TRAIN split scenes only
               (S01/S03/S04). The validation split (S02/S05) is never trained
               on -- S02 is ATHAR's frozen calibration + production-validation
               scene -- and any train-scene identity that also appears in S02
               GT is dropped from training so the Stage-C eval stays clean.

Recipe = the canonical A5alpha Stream-1 recipe (veri-canon-stream1-train),
with three documented deviations for the joint setting:
  * SIE OFF ....... deployment cameras are always unseen (the dinov2 adapter
                    precedent skips SIE at inference); VehicleID has no camera
                    labels; a cross-domain union camera vocab is meaningless.
  * CenterLoss OFF  the 09w campaign hit the fp16 CenterLoss NaN trap at
                    ~1.8k classes; the joint label space is ~45k classes.
  * Iteration-budgeted epochs: each epoch = STEPS_PER_EPOCH batches, drawn
                    round-robin across the four domains (step % 4), each batch
                    a single-domain P=24/K=4 PK batch. Every domain contributes
                    the same number of batches per epoch regardless of its
                    size -- that is the "domain-balanced" contract. Single-
                    domain batches keep triplet negatives hard (cross-domain
                    negatives are trivially easy).

Everything else follows A5alpha: ViT-B/16 CLIP (openai) @224, JPM, BNNeck
v15 routing (CE post-BN, triplet pre-BN), AdamW LLRD 0.75 (3.5e-4/3.5e-3,
wd 1e-2), CE-LS 0.1 + JPM aux x0.5 + hard triplet margin 0.3, 10-epoch
warmup + cosine, seed 0, per-epoch resume checkpoint with RNG state.

In-training eval: VeRi-776 query/gallery mAP (no rerank) every EVAL_EVERY
epochs -- the cheap progress signal. The full cross-domain matrix is Stage C.

Session budget: T4, MAX_SESSION_HOURS guard; the loop stops cleanly, exports
best + provenance, and a later session resumes from last.pth (attach the
prior version's output under /kaggle/input).

Local structural smoke (no Kaggle, no downloads, CPU):
  ATHAR_LOCAL_SMOKE=1 python athar_joint_reid_train.py
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

LOCAL_SMOKE = os.environ.get("ATHAR_LOCAL_SMOKE", "0") == "1"
SMOKE_TEST = LOCAL_SMOKE or os.environ.get("SMOKE_TEST", "0") == "1"

GDRIVE_ID = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"  # AIC22_Track1_MTMC_Tracking.zip
WORK = Path("/kaggle/working" if not LOCAL_SMOKE else os.environ.get("ATHAR_SMOKE_DIR", "./_smoke_work"))
TMP = Path("/tmp" if not LOCAL_SMOKE else str(WORK / "tmp"))
INPUT = Path("/kaggle/input")
CKPT_DIR = WORK / "checkpoints"
SEED = 0

# -- recipe constants (A5alpha unless noted) ---------------------------------
H = W = 224
P_IDS, K_PER_ID = 24, 4
BATCH = P_IDS * K_PER_ID  # 96
EPOCHS = 40  # iteration-budgeted epochs (see STEPS_PER_EPOCH), not dataset passes
WARMUP = 10
STEPS_PER_EPOCH = 1600  # 400 batches per domain per epoch (~38k imgs/domain)
EVAL_EVERY = 5
BACKBONE_LR, HEAD_LR, WD, LLRD = 3.5e-4, 3.5e-3, 1e-2, 0.75
# Wall-clock guard measured from PROCESS start (dataset setup takes ~1.5h of
# the 12h Kaggle session) so the final export always lands before the kill.
MAX_SESSION_HOURS = 10.5
VIT_MODEL = "vit_base_patch16_clip_224.openai"

# CityFlow crop extraction (09q recipe constants)
MAX_CROPS_PER_ID_CAM = 20
MIN_AREA = 2000
MIN_BBOX_SIDE = 30

if SMOKE_TEST:
    EPOCHS, WARMUP, STEPS_PER_EPOCH, EVAL_EVERY = 2, 1, 8, 1
    P_IDS, K_PER_ID = 4, 2
    BATCH = P_IDS * K_PER_ID


def sh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def pip_install(*pkgs: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


# ---------------------------------------------------------------------------
# Domain builders -- each returns a list of (path, local_pid, cam) train rows
# ---------------------------------------------------------------------------


def build_veri776() -> tuple[list, list, list]:
    """Mounted abhyudaya12 dataset. Returns (train, query, gallery)."""
    import re

    root = None
    for p in INPUT.rglob("image_train"):
        if p.is_dir():
            root = p.parent
            break
    if root is None:
        raise RuntimeError("VeRi-776 not mounted (attach abhyudaya12/veri-vehicle-re-identification-dataset)")
    pat = re.compile(r"^(\d+)_c(\d+)")
    splits = {"image_train": [], "image_query": [], "image_test": []}
    for split_name, rows in splits.items():
        split_dir = root / split_name
        for fname in sorted(os.listdir(split_dir)):
            m = pat.match(fname)
            if fname.endswith(".jpg") and m:
                rows.append((str(split_dir / fname), int(m.group(1)), int(m.group(2)) - 1))
    train = splits["image_train"]
    pid2label = {pid: i for i, pid in enumerate(sorted({p for _, p, _ in train}))}
    train = [(p, pid2label[pid], c) for p, pid, c in train]
    print(f"[veri776] train {len(train)} imgs / {len(pid2label)} ids; "
          f"query {len(splits['image_query'])} gallery {len(splits['image_test'])}")
    assert len(pid2label) == 575, f"unexpected VeRi train ids {len(pid2label)} (bad mount?)"
    assert len(splits["image_query"]) == 1678 and len(splits["image_test"]) == 11579
    return train, splits["image_query"], splits["image_test"]


def build_vehicleid() -> list:
    """Mounted maphat/vehicleid zip -> extract -> parse train_list.txt."""
    zip_path = next(INPUT.rglob("VehicleID_V1.0.zip"), None)
    if zip_path is None:
        raise RuntimeError("VehicleID zip not mounted (attach maphat/vehicleid)")
    ex = TMP / "vehicleid"
    if not (ex / ".done").exists():
        print(f"[vehicleid] extracting {zip_path} ...")
        with zipfile.ZipFile(str(zip_path)) as zf:
            zf.extractall(str(ex))
        (ex / ".done").touch()
    split_file = next(ex.rglob("train_test_split/train_list.txt"), None)
    assert split_file, "train_list.txt not found in VehicleID archive"
    image_root = None
    for cand in ["image", "images", "Image", "Images"]:
        d = split_file.parent.parent / cand
        if d.is_dir():
            image_root = d
            break
    assert image_root, "VehicleID image dir not found"
    rows = []
    for line in split_file.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            img = image_root / f"{parts[0]}.jpg"
            rows.append((str(img), int(parts[1]), 0))  # no camera labels in VehicleID
    missing = sum(1 for p, _, _ in rows[:2000] if not os.path.exists(p))
    assert missing == 0, f"VehicleID sample check: {missing} of first 2000 images missing"
    pid2label = {pid: i for i, pid in enumerate(sorted({p for _, p, _ in rows}))}
    rows = [(p, pid2label[pid], c) for p, pid, c in rows]
    print(f"[vehicleid] train {len(rows)} imgs / {len(pid2label)} ids")
    return rows


def build_veriwild() -> list:
    """Staged tarbins from mrkdagods/veriwild-train (or -a/-b fallback split)."""
    tarbins = sorted(INPUT.rglob("veriwild_train_*.tarbin"))
    assert tarbins, "veriwild-train tarbins not mounted (attach mrkdagods/veriwild-train)"
    ex = TMP / "veriwild"
    if not (ex / ".done").exists():
        ex.mkdir(parents=True, exist_ok=True)
        for tb in tarbins:
            print(f"[veriwild] untar {tb.name} ...")
            r = sh(f"tar -xf {tb} -C {ex}")
            assert r.returncode == 0, f"untar failed: {r.stderr[-400:]}"
        (ex / ".done").touch()
    train_list = next(INPUT.rglob("train_list_start0.txt"), None)
    assert train_list, "train_list_start0.txt not mounted"
    rows = []
    for line in train_list.read_text().splitlines():
        parts = line.split()  # "<vehid>/<img>.jpg <label> <camid>"
        if len(parts) >= 3:
            rows.append((str(ex / parts[0]), int(parts[1]), int(parts[2])))
    missing = sum(1 for p, _, _ in rows[:2000] if not os.path.exists(p))
    assert missing == 0, f"VeRi-Wild sample check: {missing} of first 2000 images missing"
    n_ids = len({p for _, p, _ in rows})
    print(f"[veriwild] train {len(rows)} imgs / {n_ids} ids")
    if not SMOKE_TEST:
        assert len(rows) == 277_797 and n_ids == 30_671, "unexpected VeRi-Wild train counts"
    return rows


def _load_gt_rows(gt_path: Path) -> list:
    rows = []
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 6:
            rows.append(tuple(int(float(v)) for v in parts[:6]))  # frame,tid,x,y,w,h
    return rows


def build_cityflow() -> list:
    """Download the AIC22 archive, extract GT crops from TRAIN scenes only.

    Exclusion contract: collect the S02 identity set from validation GT and
    drop those ids from training (S02 is the held-out eval + calibration
    scene; CityFlow vehicle ids are globally consistent across scenes).
    """
    import cv2

    crop_dir = TMP / "cityflow_crops"
    staging = TMP / "_aic22_staging"
    if not (crop_dir / ".done").exists():
        if not any(staging.rglob("vdo.avi")):
            archive = TMP / "AIC22_Track1_MTMC_Tracking.zip"
            if not archive.exists():
                print(f"[cityflow] downloading (gdrive id={GDRIVE_ID}, ~20GB)...")
                import gdown

                gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", str(archive), quiet=False)
            staging.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(archive)) as zf:
                # train split videos+GT, plus validation gt.txt (tiny) for the
                # S02 exclusion set. Members appear with and without a leading
                # archive prefix -- match both.
                members = [
                    m for m in zf.namelist()
                    if ("train/" in m and not m.startswith("test"))
                    and not m.endswith(".zip")
                ]
                members += [m for m in zf.namelist() if "validation/" in m and m.endswith("gt.txt")]
                assert members, "no train members matched in archive"
                zf.extractall(str(staging), members=members)
            archive.unlink(missing_ok=True)

        # S02 exclusion set from validation GT
        exclude_ids: set[int] = set()
        for gt in staging.rglob("gt.txt"):
            if "validation" in str(gt) and "S02" in str(gt):
                exclude_ids.update(r[1] for r in _load_gt_rows(gt))
        print(f"[cityflow] S02 exclusion set: {len(exclude_ids)} ids")

        cams = []
        for vdo in sorted(staging.rglob("vdo.avi")):
            if "train" not in str(vdo):
                continue
            gt = vdo.parent / "gt" / "gt.txt"
            if not gt.exists():
                gt = vdo.parent / "gt.txt"
            if gt.exists():
                scene = vdo.parent.parent.name
                cams.append((f"{scene}_{vdo.parent.name}", gt, vdo))
        print(f"[cityflow] {len(cams)} train cameras with GT")
        assert cams, "no CityFlow train cameras found"

        crop_dir.mkdir(parents=True, exist_ok=True)
        dropped = 0
        for cam_name, gt_path, vid_path in cams:
            id_dets = defaultdict(list)
            for frame, tid, x, y, w, h in _load_gt_rows(gt_path):
                if tid in exclude_ids:
                    dropped += 1
                    continue
                id_dets[tid].append((frame, x, y, w, h))
            frame_to_dets = defaultdict(list)
            for tid, dets in id_dets.items():
                if len(dets) > MAX_CROPS_PER_ID_CAM:
                    step = len(dets) / MAX_CROPS_PER_ID_CAM
                    dets = [dets[int(i * step)] for i in range(MAX_CROPS_PER_ID_CAM)]
                for frame, x, y, w, h in dets:
                    if w * h >= MIN_AREA and w >= MIN_BBOX_SIDE and h >= MIN_BBOX_SIDE:
                        frame_to_dets[frame].append((tid, x, y, w, h))
            if not frame_to_dets:
                continue
            cap = cv2.VideoCapture(str(vid_path))
            targets = sorted(frame_to_dets)
            ti, current, n = 0, 0, 0
            while ti < len(targets):
                ret, img = cap.read()
                if not ret:
                    break
                current += 1
                if current < targets[ti]:
                    continue
                while ti < len(targets) and targets[ti] < current:
                    ti += 1
                if ti >= len(targets) or current != targets[ti]:
                    continue
                ih, iw = img.shape[:2]
                for tid, x, y, w, h in frame_to_dets[current]:
                    x1, y1, x2, y2 = max(0, x), max(0, y), min(iw, x + w), min(ih, y + h)
                    if x2 - x1 < MIN_BBOX_SIDE or y2 - y1 < MIN_BBOX_SIDE:
                        continue
                    cv2.imwrite(str(crop_dir / f"{tid:04d}_{cam_name}_f{current:06d}.jpg"), img[y1:y2, x1:x2])
                    n += 1
                ti += 1
            cap.release()
            print(f"  {cam_name}: {n} crops")
        print(f"[cityflow] dropped {dropped} GT rows via S02 exclusion")
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        (crop_dir / ".done").touch()

    rows = []
    for f in sorted(crop_dir.glob("*.jpg")):
        parts = f.stem.split("_")  # 0042_S01_c001_f000123
        rows.append((str(f), int(parts[0]), 0))
    pid2label = {pid: i for i, pid in enumerate(sorted({p for _, p, _ in rows}))}
    rows = [(p, pid2label[pid], c) for p, pid, c in rows]
    print(f"[cityflow] train {len(rows)} crops / {len(pid2label)} ids")
    return rows


def build_local_smoke_domains() -> tuple[dict, list, list]:
    """Fabricate tiny synthetic domains for the local CPU structural smoke."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    domains = {}
    for d in ["veri776", "vehicleid", "veriwild", "cityflow"]:
        rows = []
        droot = TMP / "smoke" / d
        droot.mkdir(parents=True, exist_ok=True)
        for pid in range(P_IDS + 1):
            for k in range(K_PER_ID + 1):
                p = droot / f"{pid}_{k}.jpg"
                if not p.exists():
                    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
                rows.append((str(p), pid, 0))
        domains[d] = rows
    # every pid: first image queries (cam 0), the rest gallery (cam 1)
    query, gallery, first_seen = [], [], set()
    for p, pid, _ in domains["veri776"]:
        if pid not in first_seen:
            first_seen.add(pid)
            query.append((p, pid, 0))
        else:
            gallery.append((p, pid, 1))
    return domains, query, gallery


# ---------------------------------------------------------------------------
def main() -> None:
    proc_t0 = time.time()
    if not LOCAL_SMOKE:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # cu124 pin: runs on both T4 (sm_75) and P100 (sm_60)
        pip_install("--upgrade", "torch==2.4.1+cu124", "torchvision==0.19.1+cu124",
                    "--index-url", "https://download.pytorch.org/whl/cu124")
        pip_install("timm==1.0.11", "gdown", "opencv-python-headless")

    import numpy as np
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Sampler

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    print(f"device={device} torch={torch.__version__} smoke={SMOKE_TEST} local={LOCAL_SMOKE}")

    # -- assemble domains -----------------------------------------------------
    if LOCAL_SMOKE:
        domain_rows, veri_query, veri_gallery = build_local_smoke_domains()
    else:
        veri_train, veri_query, veri_gallery = build_veri776()
        domain_rows = {
            "veri776": veri_train,
            "cityflow": build_cityflow(),
            "vehicleid": build_vehicleid(),
            "veriwild": build_veriwild(),
        }

    if SMOKE_TEST and not LOCAL_SMOKE:
        for name, rows in domain_rows.items():
            keep = sorted({pid for _, pid, _ in rows})[:8]
            relabel = {pid: i for i, pid in enumerate(keep)}
            domain_rows[name] = [(p, relabel[pid], c) for p, pid, c in rows if pid in relabel]

    domain_order = ["veri776", "cityflow", "vehicleid", "veriwild"]
    offsets, total_classes = {}, 0
    for name in domain_order:
        offsets[name] = total_classes
        total_classes += len({pid for _, pid, _ in domain_rows[name]})
    joint_counts = {
        name: {"images": len(domain_rows[name]),
               "ids": len({pid for _, pid, _ in domain_rows[name]}),
               "label_offset": offsets[name]}
        for name in domain_order
    }
    print(json.dumps(joint_counts, indent=2))
    print(f"joint label space: {total_classes} classes")

    # -- transforms / datasets / samplers -------------------------------------
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    train_tf = T.Compose([
        T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5), T.Pad(10), T.RandomCrop((H, W)),
        T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        T.RandomPerspective(distortion_scale=0.2, p=0.2),
        T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        T.RandomErasing(p=0.5, value="random"),
    ])
    test_tf = T.Compose([
        T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    class ReIDDataset(Dataset):
        def __init__(self, rows, tf):
            self.rows, self.tf = rows, tf

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            path, pid, cam = self.rows[i]
            return self.tf(Image.open(path).convert("RGB")), pid, cam

    class PKSampler(Sampler):
        def __init__(self, rows, p, k, seed):
            self.p, self.k, self.seed, self.epoch = p, k, seed, 0
            self.pid_to_idx = defaultdict(list)
            for i, (_, pid, _) in enumerate(rows):
                self.pid_to_idx[pid].append(i)
            self.pids = list(self.pid_to_idx)
            self.batch_size = p * k

        def __iter__(self):
            rng = np.random.default_rng(self.seed + self.epoch)
            self.epoch += 1
            pids = list(self.pids)
            rng.shuffle(pids)
            batch = []
            for pid in pids:
                idxs = self.pid_to_idx[pid]
                batch.extend(rng.choice(idxs, self.k, replace=len(idxs) < self.k).tolist())
                if len(batch) >= self.batch_size:
                    yield from batch[: self.batch_size]
                    batch = batch[self.batch_size:]
            if batch:
                yield from batch

        def __len__(self):
            return len(self.pids) * self.k

    def make_domain_loader(name):
        rows = [(p, pid + offsets[name], cam) for p, pid, cam in domain_rows[name]]
        ds = ReIDDataset(rows, train_tf)
        sampler = PKSampler(rows, P_IDS, K_PER_ID, SEED)
        return DataLoader(ds, batch_size=BATCH, sampler=sampler, num_workers=0 if LOCAL_SMOKE else 2,
                          pin_memory=not LOCAL_SMOKE, drop_last=True,
                          generator=torch.Generator().manual_seed(SEED))

    loaders = {name: make_domain_loader(name) for name in domain_order}
    iters = {name: iter(dl) for name, dl in loaders.items()}

    def next_batch(name):
        try:
            return next(iters[name])
        except StopIteration:
            iters[name] = iter(loaders[name])
            return next(iters[name])

    query_loader = DataLoader(ReIDDataset(veri_query, test_tf), batch_size=128,
                              num_workers=0 if LOCAL_SMOKE else 2, pin_memory=not LOCAL_SMOKE)
    gallery_loader = DataLoader(ReIDDataset(veri_gallery, test_tf), batch_size=128,
                                num_workers=0 if LOCAL_SMOKE else 2, pin_memory=not LOCAL_SMOKE)

    # -- model (canon TransReID, SIE off) --------------------------------------
    class TransReID(nn.Module):
        def __init__(self, num_classes, vit_model, pretrained, jpm=True):
            super().__init__()
            self.jpm = jpm
            self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0)
            self.vit_dim = self.vit.embed_dim
            self.num_blocks = len(self.vit.blocks)
            self.bn = nn.BatchNorm1d(self.vit_dim)
            self.bn.bias.requires_grad_(False)
            self.proj = nn.Identity()  # embed_dim == vit_dim == 768
            self.cls_head = nn.Linear(self.vit_dim, num_classes, bias=False)
            nn.init.normal_(self.cls_head.weight, std=0.001)
            if jpm:
                self.bn_jpm = nn.BatchNorm1d(self.vit_dim)
                self.bn_jpm.bias.requires_grad_(False)
                self.jpm_cls = nn.Linear(self.vit_dim, num_classes, bias=False)
                nn.init.normal_(self.jpm_cls.weight, std=0.001)

        def forward(self, x):
            x = self.vit.patch_embed(x)
            x = self.vit._pos_embed(x)
            if hasattr(self.vit, "patch_drop"):
                x = self.vit.patch_drop(x)
            if hasattr(self.vit, "norm_pre"):
                x = self.vit.norm_pre(x)  # CRITICAL for CLIP ViTs
            for blk in self.vit.blocks:
                x = blk(x)
            x = self.vit.norm(x)
            g = x[:, 0]           # pre-BN -> triplet
            bn = self.bn(g)       # post-BN -> CE + inference
            if self.training:
                cls = self.cls_head(bn)
                if self.jpm:
                    patches = x[:, 1:]
                    idx = torch.randperm(patches.size(1), device=x.device)
                    s = patches[:, idx]
                    mid = s.size(1) // 2
                    jf = (s[:, :mid].mean(1) + s[:, mid:].mean(1)) / 2
                    return cls, g, self.jpm_cls(self.bn_jpm(jf))
                return cls, g
            return F.normalize(bn, p=2, dim=1)

        def get_llrd_param_groups(self, backbone_lr, head_lr, decay):
            groups = {}
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("vit."):
                    if "blocks." in name:
                        depth = int(name.split("blocks.")[1].split(".")[0]) + 1
                    elif any(k in name for k in ["patch_embed", "cls_token", "pos_embed", "norm_pre"]):
                        depth = 0
                    else:
                        depth = self.num_blocks + 1
                    lr = backbone_lr * decay ** (self.num_blocks + 1 - depth)
                    gk = f"bb_d{depth}"
                else:
                    lr, gk = head_lr, "head"
                groups.setdefault(gk, {"params": [], "lr": lr})["params"].append(param)
            return sorted(groups.values(), key=lambda g: g["lr"])

    model = TransReID(total_classes, VIT_MODEL, pretrained=not LOCAL_SMOKE).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -- losses -----------------------------------------------------------------
    class CrossEntropyLabelSmooth(nn.Module):
        def __init__(self, num_classes, epsilon=0.1):
            super().__init__()
            self.num_classes, self.epsilon = num_classes, epsilon

        def forward(self, inputs, targets):
            log_probs = F.log_softmax(inputs.float(), dim=1)
            with torch.no_grad():
                oh = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
                smooth = (1 - self.epsilon) * oh + self.epsilon / self.num_classes
            return (-smooth * log_probs).sum(dim=1).mean()

    class TripletLossHardMining(nn.Module):
        def __init__(self, margin=0.3):
            super().__init__()
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)

        def forward(self, feats, pids):
            feats = F.normalize(feats.float(), p=2, dim=1)
            dist = torch.cdist(feats, feats, p=2)
            mask_pos = pids.unsqueeze(0).eq(pids.unsqueeze(1))
            dist_pos = dist.clone()
            dist_pos[~mask_pos] = 0
            dist_neg = dist.clone()
            dist_neg[mask_pos] = float("inf")
            return self.ranking_loss(dist_neg.min(dim=1)[0], dist_pos.max(dim=1)[0],
                                     torch.ones(feats.size(0), device=feats.device))

    ce_loss = CrossEntropyLabelSmooth(total_classes).to(device)
    tri_loss = TripletLossHardMining().to(device)

    optimizer = torch.optim.AdamW(model.get_llrd_param_groups(BACKBONE_LR, HEAD_LR, LLRD), weight_decay=WD)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS - WARMUP, 1))
    scaler = torch.amp.GradScaler(device) if device == "cuda" else None

    # -- eval helpers (VeRi mAP, no rerank) --------------------------------------
    @torch.no_grad()
    def extract_features(loader):
        model.eval()
        feats, pids, cams = [], [], []
        for imgs, pid, cam in loader:
            imgs = imgs.to(device)
            f = model(imgs)
            ff = model(torch.flip(imgs, [3]))
            f = F.normalize((f + ff) / 2, p=2, dim=1)
            feats.append(f.cpu().numpy())
            pids.append(pid.numpy())
            cams.append(cam.numpy())
        return np.concatenate(feats), np.concatenate(pids), np.concatenate(cams)

    def eval_market1501(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=50):
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)
        all_cmc, all_ap = [], []
        for qi in range(distmat.shape[0]):
            order = indices[qi]
            remove = (g_pids[order] == q_pids[qi]) & (g_cams[order] == q_cams[qi])
            raw = matches[qi][~remove]
            if raw.sum() == 0:
                continue
            cmc = raw.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max_rank])
            prec = raw.cumsum() / (np.arange(len(raw)) + 1.0)
            all_ap.append((prec * raw).sum() / raw.sum())
        if not all_ap:
            return 0.0, np.zeros(max_rank)
        return float(np.mean(all_ap)), np.array(all_cmc).mean(0)

    def veri_eval():
        qf, qp, qc = extract_features(query_loader)
        gf, gp, gc = extract_features(gallery_loader)
        return eval_market1501(1.0 - qf @ gf.T, qp, gp, qc, gc)

    # -- resume -------------------------------------------------------------------
    def capture_rng():
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def restore_rng(state):
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def find_resume():
        live = CKPT_DIR / "last.pth"
        if live.exists():
            return live
        best, best_ep = None, -1
        for cand in sorted(INPUT.glob("**/checkpoints/last.pth")) if INPUT.exists() else []:
            try:
                ep = int(torch.load(cand, map_location="cpu", weights_only=False).get("epoch", -1))
            except Exception as exc:  # noqa: BLE001
                print(f"skip resume candidate {cand}: {exc}")
                continue
            if ep > best_ep:
                best, best_ep = cand, ep
        return best

    history = {"loss": [], "epochs": [], "mAP": [], "R1": [], "per_domain_loss": []}
    best_map, start_epoch = 0.0, 0
    resume = find_resume()
    if resume is not None:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        if scaler is not None and ck.get("scaler_state") is not None:
            scaler.load_state_dict(ck["scaler_state"])
        restore_rng(ck.get("rng_state"))
        start_epoch = int(ck["epoch"]) + 1
        best_map = float(ck.get("best_mAP", 0.0))
        history = ck.get("history", history)
        # PK samplers are epoch-seeded; fast-forward to keep draws aligned
        for name in domain_order:
            loaders[name].sampler.epoch = start_epoch
        print(f"RESUMING from epoch {start_epoch + 1} via {resume} (best mAP {best_map:.4f})")
    else:
        print("STARTING FRESH")

    # -- train ----------------------------------------------------------------------
    t0 = time.time()
    stop_early = False
    epoch = start_epoch - 1  # provenance-safe if the loop body never runs
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        if epoch < WARMUP:
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = base_lrs[i] * (epoch + 1) / WARMUP
        elif epoch == WARMUP:
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = base_lrs[i]
        run_loss, dom_loss = 0.0, dict.fromkeys(domain_order, 0.0)
        for step in range(STEPS_PER_EPOCH):
            name = domain_order[step % len(domain_order)]  # round-robin, never sequential
            imgs, pids, _ = next_batch(name)
            imgs, pids = imgs.to(device), pids.to(device).long()
            optimizer.zero_grad()
            with torch.amp.autocast(device, enabled=device == "cuda"):
                cls, feat, jcls = model(imgs)
                loss = ce_loss(cls, pids) + 0.5 * ce_loss(jcls, pids) + tri_loss(feat, pids)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            val = loss.item()
            if not np.isnan(val):
                run_loss += val
                dom_loss[name] += val
        if epoch >= WARMUP:
            scheduler.step()
        n_per_dom = STEPS_PER_EPOCH / len(domain_order)
        history["loss"].append(run_loss / STEPS_PER_EPOCH)
        history["per_domain_loss"].append({k: v / n_per_dom for k, v in dom_loss.items()})
        hrs = (time.time() - proc_t0) / 3600
        print(f"epoch {epoch + 1:3d}/{EPOCHS} | loss {run_loss / STEPS_PER_EPOCH:.4f} | "
              + " ".join(f"{k}={v / n_per_dom:.3f}" for k, v in dom_loss.items())
              + f" | {hrs:.2f}h", flush=True)

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            m, cmc = veri_eval()
            history["epochs"].append(epoch + 1)
            history["mAP"].append(m)
            history["R1"].append(float(cmc[0]))
            print(f"  -> VeRi mAP {m:.4f} R1 {cmc[0]:.4f}", flush=True)
            if m > best_map:
                best_map = m
                torch.save(model.state_dict(), WORK / "transreid_joint4d_best.pth")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_mAP": best_map, "history": history, "rng_state": capture_rng(),
        }, CKPT_DIR / "last.pth")

        if hrs > MAX_SESSION_HOURS:
            print(f"session budget reached at epoch {epoch + 1}; stopping cleanly for resume")
            stop_early = True
            break

    if not (WORK / "transreid_joint4d_best.pth").exists():
        torch.save(model.state_dict(), WORK / "transreid_joint4d_best.pth")

    # -- provenance --------------------------------------------------------------
    import hashlib

    ckpt_path = WORK / "transreid_joint4d_best.pth"
    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    provenance = {
        "kernel": "athar-joint-reid-train",
        "recipe": "A5alpha joint-4domain (SIE off, center off, iteration-budgeted domain-balanced)",
        "backbone": VIT_MODEL,
        "input_size": [H, W],
        "domains": joint_counts,
        "total_classes": total_classes,
        "batch": BATCH, "p_ids": P_IDS, "k_per_id": K_PER_ID,
        "epochs_configured": EPOCHS, "epochs_completed": epoch + 1,
        "stopped_early_for_session_budget": stop_early,
        "steps_per_epoch": STEPS_PER_EPOCH, "warmup": WARMUP,
        "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR, "weight_decay": WD, "llrd": LLRD,
        "losses": "ce_ls0.1 + jpm_aux0.5 + triplet0.3 (center OFF: fp16 NaN trap at 45k classes)",
        "sie": False,
        "seed": SEED,
        "smoke_test": SMOKE_TEST,
        "best_veri_mAP": best_map,
        "history": history,
        "torch": torch.__version__,
        "checkpoint": ckpt_path.name,
        "checkpoint_sha256": sha,
        "elapsed_hours": round((time.time() - t0) / 3600, 3),
        "cityflow_exclusion": "S02 identity set dropped from training (frozen calibration/eval scene)",
    }
    (WORK / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in provenance.items() if k != "history"}, indent=2))
    print(f"DONE best VeRi mAP={best_map:.4f} ckpt sha256={sha[:16]}...")


if __name__ == "__main__":
    main()
