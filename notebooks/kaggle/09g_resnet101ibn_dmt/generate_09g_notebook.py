import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(__file__).with_name("09g_resnet101ibn_dmt.ipynb")


def to_source(text: str) -> list[str]:
    text = dedent(text).strip("\n")
    if not text:
        return []
    lines = text.splitlines()
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": to_source(source),
    }


def notebook() -> dict:
    cells = [
        code_cell(
            """
            # Install dependencies
            !pip install -q timm>=0.9 pytorch-metric-learning faiss-cpu scikit-learn

            import copy
            import json
            import math
            import os
            import random
            import re
            import shutil
            import time
            import urllib.request
            from collections import defaultdict
            from itertools import cycle
            from pathlib import Path

            import numpy as np
            import timm
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torchvision.models as tv_models
            import torchvision.transforms as T
            from PIL import Image
            from sklearn.cluster import DBSCAN
            from torch.utils.data import DataLoader, Dataset, Sampler

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {device}, CUDA: {torch.cuda.is_available()}")

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
                torch.backends.cudnn.benchmark = True

            OUTPUT_DIR = Path("/kaggle/working/09g_output")
            CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

            CFG = {
                "img_size": (384, 384),
                "batch_p": 8,
                "batch_k": 4,
                "num_workers": 4,
                "train_epochs": 120,
                "stage2_epochs": 40,
                "stage2_recluster_every": 3,
                "stage2_iters_per_epoch": 300,
                "eval_every": 10,
                "stage2_eval_every": 3,
                "lr": 3.5e-4,
                "stage2_lr": 1.0e-4,
                "weight_decay": 5.0e-4,
                "warmup_epochs": 10,
                "label_smoothing": 0.1,
                "triplet_margin": 0.3,
                "center_loss_weight": 0.5,
                "center_lr": 0.5,
                "gem_p": 3.0,
                "fp16": True,
                "dbscan_eps": 0.58,
                "dbscan_min_samples": 4,
                "fic_lambda": 5.0e-4,
                "use_flip_eval": True,
                "uda_include_eval": False,
                "target_map": 0.65,
            }

            TRAIN_BATCH_SIZE = CFG["batch_p"] * CFG["batch_k"]
            print(json.dumps(CFG, indent=2))
            """
        ),
        code_cell(
            """
            DATASET_CANDIDATES = [
                Path("/kaggle/input/cityflowv2-reid"),
                Path("/kaggle/input/datasets/mrkdagods/cityflowv2-reid"),
            ]
            DATASET_ROOT = next((path for path in DATASET_CANDIDATES if path.exists()), None)
            if DATASET_ROOT is None:
                raise FileNotFoundError(f"CityFlowV2 ReID dataset not found. Tried: {DATASET_CANDIDATES}")

            IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
            CAMERA_RE = re.compile(r"(S\d+_c\d+|c\d{3,})", re.IGNORECASE)
            FRAME_RE = re.compile(r"f(\d+)", re.IGNORECASE)


            def infer_pid(path: Path) -> int:
                parent_name = path.parent.name
                if parent_name.lower() not in {"train", "query", "gallery"}:
                    digits = re.findall(r"\d+", parent_name)
                    if digits:
                        return int(digits[0])
                stem_match = re.match(r"(\d+)", path.stem)
                if stem_match:
                    return int(stem_match.group(1))
                raise ValueError(f"Cannot infer pid from {path}")


            def infer_cam_name(path: Path) -> str:
                joined = "/".join(path.parts[-4:])
                match = CAMERA_RE.search(joined)
                if match:
                    return match.group(1)
                return "unknown_cam"


            def infer_frame_id(path: Path) -> int:
                match = FRAME_RE.search(path.stem)
                if match:
                    return int(match.group(1))
                return 0


            def scan_split_dir(split_dir: Path) -> list[dict]:
                records = []
                for image_path in sorted(path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
                    records.append(
                        {
                            "path": str(image_path),
                            "pid": infer_pid(image_path),
                            "camname": infer_cam_name(image_path),
                            "frame_id": infer_frame_id(image_path),
                        }
                    )
                return records


            def build_split_from_identity_folders(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
                pid_to_records = defaultdict(list)
                for image_path in sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
                    if any(part.lower() in {"train", "query", "gallery"} for part in image_path.parts):
                        continue
                    pid = infer_pid(image_path)
                    pid_to_records[pid].append(
                        {
                            "path": str(image_path),
                            "pid": pid,
                            "camname": infer_cam_name(image_path),
                            "frame_id": infer_frame_id(image_path),
                        }
                    )

                multi_cam_ids = []
                single_cam_ids = []
                for pid, records in pid_to_records.items():
                    cameras = {record["camname"] for record in records}
                    if len(cameras) >= 2:
                        multi_cam_ids.append(pid)
                    else:
                        single_cam_ids.append(pid)

                rng = np.random.default_rng(SEED)
                rng.shuffle(multi_cam_ids)
                n_train = int(len(multi_cam_ids) * 0.7)
                train_ids = set(multi_cam_ids[:n_train])
                eval_ids = set(multi_cam_ids[n_train:])

                train_records, query_records, gallery_records = [], [], []

                for pid in sorted(train_ids):
                    for record in sorted(pid_to_records[pid], key=lambda item: (item["camname"], item["frame_id"], item["path"])):
                        train_records.append(dict(record))

                for pid in sorted(eval_ids):
                    by_cam = defaultdict(list)
                    for record in pid_to_records[pid]:
                        by_cam[record["camname"]].append(record)
                    for camname, items in sorted(by_cam.items()):
                        ordered = sorted(items, key=lambda item: (item["frame_id"], item["path"]))
                        query_records.append(dict(ordered[0]))
                        for record in ordered[1:]:
                            gallery_records.append(dict(record))

                for pid in sorted(single_cam_ids):
                    for record in sorted(pid_to_records[pid], key=lambda item: (item["camname"], item["frame_id"], item["path"])):
                        gallery_records.append(dict(record))

                return train_records, query_records, gallery_records


            split_dirs = {name: DATASET_ROOT / name for name in ["train", "query", "gallery"]}
            has_prebuilt_splits = all(path.exists() for path in split_dirs.values())

            if has_prebuilt_splits:
                train_records = scan_split_dir(split_dirs["train"])
                query_records = scan_split_dir(split_dirs["query"])
                gallery_records = scan_split_dir(split_dirs["gallery"])
            else:
                train_records, query_records, gallery_records = build_split_from_identity_folders(DATASET_ROOT)

            camname_to_id = {}
            for record in train_records + query_records + gallery_records:
                if record["camname"] not in camname_to_id:
                    camname_to_id[record["camname"]] = len(camname_to_id)
                record["camid"] = camname_to_id[record["camname"]]

            train_pid_map = {pid: index for index, pid in enumerate(sorted({record["pid"] for record in train_records}))}
            for record in train_records:
                record["pid"] = train_pid_map[record["pid"]]

            UDA_SOURCE_RECORDS = list(train_records)
            if CFG["uda_include_eval"]:
                for record in query_records + gallery_records:
                    staged = dict(record)
                    staged["pid"] = -1
                    UDA_SOURCE_RECORDS.append(staged)


            class ReIDImageDataset(Dataset):
                def __init__(self, records, transform=None):
                    self.records = records
                    self.transform = transform

                def __len__(self):
                    return len(self.records)

                def __getitem__(self, index):
                    record = self.records[index]
                    image = Image.open(record["path"]).convert("RGB")
                    if self.transform is not None:
                        image = self.transform(image)
                    return image, int(record["pid"]), int(record["camid"]), index


            class RandomIdentitySampler(Sampler):
                def __init__(self, records, p, k):
                    self.records = records
                    self.p = p
                    self.k = k
                    self.batch_size = p * k
                    self.pid_to_indices = defaultdict(list)
                    for index, record in enumerate(records):
                        self.pid_to_indices[int(record["pid"])].append(index)
                    self.pids = list(self.pid_to_indices.keys())

                def __iter__(self):
                    batch_indices = []
                    shuffled_pids = self.pids[:]
                    random.shuffle(shuffled_pids)
                    for pid in shuffled_pids:
                        candidates = self.pid_to_indices[pid]
                        if len(candidates) >= self.k:
                            picked = random.sample(candidates, self.k)
                        else:
                            picked = random.choices(candidates, k=self.k)
                        batch_indices.extend(picked)
                        if len(batch_indices) >= self.batch_size:
                            yield from batch_indices[: self.batch_size]
                            batch_indices = batch_indices[self.batch_size :]

                def __len__(self):
                    return len(self.pids) * self.k


            H, W = CFG["img_size"]
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]

            train_transform = T.Compose(
                [
                    T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Pad(10),
                    T.RandomCrop((H, W)),
                    T.ToTensor(),
                    T.Normalize(imagenet_mean, imagenet_std),
                    T.RandomErasing(p=0.5, value="random"),
                ]
            )
            eval_transform = T.Compose(
                [
                    T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(imagenet_mean, imagenet_std),
                ]
            )

            train_loader = DataLoader(
                ReIDImageDataset(train_records, train_transform),
                batch_size=TRAIN_BATCH_SIZE,
                sampler=RandomIdentitySampler(train_records, CFG["batch_p"], CFG["batch_k"]),
                num_workers=CFG["num_workers"],
                pin_memory=True,
                drop_last=True,
            )
            query_loader = DataLoader(
                ReIDImageDataset(query_records, eval_transform),
                batch_size=128,
                shuffle=False,
                num_workers=CFG["num_workers"],
                pin_memory=True,
            )
            gallery_loader = DataLoader(
                ReIDImageDataset(gallery_records, eval_transform),
                batch_size=128,
                shuffle=False,
                num_workers=CFG["num_workers"],
                pin_memory=True,
            )
            uda_eval_loader = DataLoader(
                ReIDImageDataset(UDA_SOURCE_RECORDS, eval_transform),
                batch_size=128,
                shuffle=False,
                num_workers=CFG["num_workers"],
                pin_memory=True,
            )

            SPLIT_STATS = {
                "dataset_root": str(DATASET_ROOT),
                "prebuilt_splits": has_prebuilt_splits,
                "train_images": len(train_records),
                "query_images": len(query_records),
                "gallery_images": len(gallery_records),
                "uda_images": len(UDA_SOURCE_RECORDS),
                "num_train_ids": len(train_pid_map),
                "num_cameras": len(camname_to_id),
                "batch_size": TRAIN_BATCH_SIZE,
                "pk_sampler": {"p": CFG["batch_p"], "k": CFG["batch_k"]},
            }
            print(json.dumps(SPLIT_STATS, indent=2))
            """
        ),
        code_cell(
            """
            IBN_NET_URL = "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth"
            IBN_WEIGHTS_PATH = OUTPUT_DIR / "resnet101_ibn_a_imagenet.pth"


            class IBN_a(nn.Module):
                def __init__(self, planes):
                    super().__init__()
                    half = planes // 2
                    self.IN = nn.InstanceNorm2d(half, affine=True)
                    self.BN = nn.BatchNorm2d(planes - half)

                def forward(self, x):
                    split = x.shape[1] // 2
                    return torch.cat(
                        [self.IN(x[:, :split]), self.BN(x[:, split:])],
                        dim=1,
                    )


            class GeM(nn.Module):
                def __init__(self, p=3.0, eps=1e-6):
                    super().__init__()
                    self.p = nn.Parameter(torch.ones(1) * p)
                    self.eps = eps

                def forward(self, x):
                    return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


            class ResNet101IBNNeck(nn.Module):
                def __init__(self, num_classes, gem_p=3.0, feat_dim=2048, pretrained=True):
                    super().__init__()
                    if hasattr(timm, "list_models") and "resnet101_ibn_a" in timm.list_models(pretrained=False):
                        print("timm exposes resnet101_ibn_a, but using explicit IBN-Net patch to preserve last_stride=1 and GeM feature maps")

                    base = tv_models.resnet101(weights=None)
                    for layer in [base.layer1, base.layer2, base.layer3]:
                        for block in layer:
                            if hasattr(block, "bn1"):
                                block.bn1 = IBN_a(block.bn1.num_features)
                    for module in base.layer4.modules():
                        if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                            module.stride = (1, 1)
                    self.backbone = nn.Sequential(
                        base.conv1,
                        base.bn1,
                        base.relu,
                        base.maxpool,
                        base.layer1,
                        base.layer2,
                        base.layer3,
                        base.layer4,
                    )
                    if pretrained:
                        if not IBN_WEIGHTS_PATH.exists():
                            urllib.request.urlretrieve(IBN_NET_URL, IBN_WEIGHTS_PATH)
                        state_dict = torch.load(IBN_WEIGHTS_PATH, map_location="cpu")
                        missing, unexpected = base.load_state_dict(state_dict, strict=False)
                        allowed_missing = {"fc.weight", "fc.bias"}
                        if not set(missing).issubset(allowed_missing):
                            raise RuntimeError(f"Unexpected missing keys: {missing}")
                        if unexpected:
                            print(f"Unexpected pretrained keys ignored: {unexpected}")

                    self.pool = GeM(p=gem_p)
                    self.bottleneck = nn.BatchNorm1d(feat_dim)
                    self.bottleneck.bias.requires_grad_(False)
                    nn.init.constant_(self.bottleneck.weight, 1.0)
                    nn.init.constant_(self.bottleneck.bias, 0.0)
                    self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
                    nn.init.normal_(self.classifier.weight, std=0.001)
                    self.feat_dim = feat_dim

                def reset_classifier(self, num_classes):
                    self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False).to(next(self.parameters()).device)
                    nn.init.normal_(self.classifier.weight, std=0.001)

                def forward_features(self, x):
                    x = self.backbone(x)
                    global_feat = self.pool(x).view(x.size(0), -1)
                    bn_feat = self.bottleneck(global_feat)
                    return global_feat, bn_feat

                def forward(self, x):
                    global_feat, bn_feat = self.forward_features(x)
                    if self.training:
                        logits = self.classifier(bn_feat)
                        return logits, global_feat, bn_feat
                    return F.normalize(bn_feat, p=2, dim=1)


            def build_model(num_classes):
                model = ResNet101IBNNeck(num_classes=num_classes, gem_p=CFG["gem_p"], pretrained=True)
                return model.to(device)


            model = build_model(len(train_pid_map))
            print(f"Model params: {sum(param.numel() for param in model.parameters()):,}")
            """
        ),
        code_cell(
            """
            class CrossEntropyLabelSmooth(nn.Module):
                def __init__(self, num_classes, epsilon=0.1):
                    super().__init__()
                    self.num_classes = num_classes
                    self.epsilon = epsilon
                    self.logsoftmax = nn.LogSoftmax(dim=1)

                def forward(self, inputs, targets):
                    log_probs = self.logsoftmax(inputs)
                    targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
                    targets_one_hot = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
                    return (-targets_one_hot * log_probs).sum(dim=1).mean()


            class HardTripletLoss(nn.Module):
                def __init__(self, margin=0.3):
                    super().__init__()
                    self.ranking_loss = nn.MarginRankingLoss(margin=margin)

                def forward(self, feats, labels):
                    distance = torch.cdist(feats, feats, p=2)
                    is_pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))
                    is_neg = ~is_pos
                    is_pos.fill_diagonal_(False)

                    max_pos = torch.where(is_pos, distance, torch.zeros_like(distance)).max(dim=1)[0]
                    min_neg = torch.where(is_neg, distance, torch.full_like(distance, 1e9)).min(dim=1)[0]
                    valid = is_pos.any(dim=1) & is_neg.any(dim=1)
                    if not valid.any():
                        return feats.sum() * 0.0
                    target = torch.ones_like(max_pos[valid])
                    return self.ranking_loss(min_neg[valid], max_pos[valid], target)


            class CenterLoss(nn.Module):
                def __init__(self, num_classes, feat_dim):
                    super().__init__()
                    self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

                def forward(self, feats, labels):
                    batch_size = feats.size(0)
                    distmat = (
                        feats.pow(2).sum(dim=1, keepdim=True)
                        + self.centers.pow(2).sum(dim=1).unsqueeze(0)
                        - 2 * feats @ self.centers.t()
                    )
                    classes = torch.arange(self.centers.size(0), device=feats.device)
                    mask = labels.unsqueeze(1).eq(classes.unsqueeze(0))
                    loss = distmat.masked_select(mask).clamp(min=1e-12, max=1e12).sum() / max(batch_size, 1)
                    return loss


            def build_losses(num_classes):
                id_loss = CrossEntropyLabelSmooth(num_classes, epsilon=CFG["label_smoothing"]).to(device)
                triplet_loss = HardTripletLoss(margin=CFG["triplet_margin"]).to(device)
                center_loss = CenterLoss(num_classes, feat_dim=model.feat_dim).to(device)
                return id_loss, triplet_loss, center_loss


            id_loss_fn, triplet_loss_fn, center_loss_fn = build_losses(len(train_pid_map))
            print(
                json.dumps(
                    {
                        "losses": ["cross_entropy_label_smooth", "hard_triplet", "center_loss"],
                        "label_smoothing": CFG["label_smoothing"],
                        "triplet_margin": CFG["triplet_margin"],
                        "center_loss_weight": CFG["center_loss_weight"],
                    },
                    indent=2,
                )
            )
            """
        ),
        code_cell(
            """
            def build_optimizer(model_ref, center_criterion, base_lr):
                optimizer = torch.optim.Adam(
                    [
                        {"params": model_ref.backbone.parameters(), "lr": base_lr * 0.1},
                        {"params": model_ref.pool.parameters(), "lr": base_lr},
                        {"params": model_ref.bottleneck.parameters(), "lr": base_lr},
                        {"params": model_ref.classifier.parameters(), "lr": base_lr},
                    ],
                    lr=base_lr,
                    weight_decay=CFG["weight_decay"],
                )
                center_optimizer = torch.optim.SGD(center_criterion.parameters(), lr=CFG["center_lr"])
                return optimizer, center_optimizer


            def build_scheduler(optimizer, total_epochs):
                warmup_epochs = CFG["warmup_epochs"]

                def lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        return (epoch + 1) / max(warmup_epochs, 1)
                    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))

                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


            @torch.no_grad()
            def extract_embeddings(model_ref, loader, flip=True):
                model_ref.eval()
                features = []
                pids = []
                camids = []
                indices = []
                for images, batch_pids, batch_camids, batch_indices in loader:
                    images = images.to(device, non_blocking=True)
                    embeddings = model_ref(images)
                    if flip:
                        embeddings = embeddings + model_ref(torch.flip(images, dims=[3]))
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                    features.append(embeddings.cpu())
                    pids.extend(batch_pids.tolist())
                    camids.extend(batch_camids.tolist())
                    indices.extend(batch_indices.tolist())
                return torch.cat(features, dim=0), np.asarray(pids), np.asarray(camids), np.asarray(indices)


            def compute_cmc_map(query_features, query_pids, query_camids, gallery_features, gallery_pids, gallery_camids):
                distance = torch.cdist(query_features, gallery_features, p=2).cpu().numpy()
                indices = np.argsort(distance, axis=1)
                all_ap = []
                cmc = np.zeros(gallery_features.size(0), dtype=np.float64)
                valid_queries = 0

                for row in range(query_features.size(0)):
                    order = indices[row]
                    remove = (gallery_pids[order] == query_pids[row]) & (gallery_camids[order] == query_camids[row])
                    keep = ~remove
                    matches = (gallery_pids[order][keep] == query_pids[row]).astype(np.int32)
                    if matches.sum() == 0:
                        continue
                    cumulative = np.cumsum(matches)
                    precision = cumulative / (np.arange(matches.size) + 1)
                    ap = (precision * matches).sum() / matches.sum()
                    all_ap.append(ap)
                    first_hit = np.where(matches == 1)[0][0]
                    cmc[first_hit:] += 1
                    valid_queries += 1

                if valid_queries == 0:
                    return {"mAP": 0.0, "rank1": 0.0, "rank5": 0.0, "rank10": 0.0}

                cmc = cmc / valid_queries
                return {
                    "mAP": float(np.mean(all_ap)),
                    "rank1": float(cmc[0]) if cmc.size > 0 else 0.0,
                    "rank5": float(cmc[4]) if cmc.size > 4 else 0.0,
                    "rank10": float(cmc[9]) if cmc.size > 9 else 0.0,
                }


            def evaluate_model(model_ref):
                qf, q_pids, q_camids, _ = extract_embeddings(model_ref, query_loader, flip=CFG["use_flip_eval"])
                gf, g_pids, g_camids, _ = extract_embeddings(model_ref, gallery_loader, flip=CFG["use_flip_eval"])
                return compute_cmc_map(qf, q_pids, q_camids, gf, g_pids, g_camids)


            def train_one_epoch(
                model_ref,
                loader,
                optimizer,
                center_optimizer,
                id_criterion,
                triplet_criterion,
                center_criterion,
                epoch,
                max_iters=None,
            ):
                model_ref.train()
                scaler = torch.amp.GradScaler("cuda", enabled=CFG["fp16"] and device.type == "cuda")
                meter = defaultdict(float)
                iterator = iter(loader) if max_iters is None else cycle(loader)
                steps = len(loader) if max_iters is None else max_iters
                for step in range(steps):
                    images, labels, _, _ = next(iterator)
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    center_optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", enabled=CFG["fp16"] and device.type == "cuda"):
                        logits, global_feat, _ = model_ref(images)
                        loss_id = id_criterion(logits, labels)
                        loss_tri = triplet_criterion(global_feat, labels)
                        loss_center = center_criterion(global_feat, labels)
                        loss = loss_id + loss_tri + CFG["center_loss_weight"] * loss_center

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    for parameter in center_criterion.parameters():
                        if parameter.grad is not None:
                            parameter.grad.data *= 1.0 / max(CFG["center_loss_weight"], 1e-12)
                    center_optimizer.step()

                    meter["loss"] += float(loss.item())
                    meter["id_loss"] += float(loss_id.item())
                    meter["triplet_loss"] += float(loss_tri.item())
                    meter["center_loss"] += float(loss_center.item())

                return {key: value / max(steps, 1) for key, value in meter.items()}


            optimizer, center_optimizer = build_optimizer(model, center_loss_fn, CFG["lr"])
            scheduler = build_scheduler(optimizer, CFG["train_epochs"])
            STAGE1_HISTORY = []
            STAGE1_BEST = {"mAP": -1.0, "epoch": -1, "path": str(CHECKPOINT_DIR / "stage1_best.pth")}

            for epoch in range(CFG["train_epochs"]):
                started = time.time()
                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    center_optimizer,
                    id_loss_fn,
                    triplet_loss_fn,
                    center_loss_fn,
                    epoch,
                )
                scheduler.step()
                record = {
                    "epoch": epoch + 1,
                    "elapsed_sec": round(time.time() - started, 2),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    **{key: float(value) for key, value in train_metrics.items()},
                }
                if (epoch + 1) % CFG["eval_every"] == 0 or epoch == CFG["train_epochs"] - 1:
                    eval_metrics = evaluate_model(model)
                    record.update(eval_metrics)
                    if eval_metrics["mAP"] > STAGE1_BEST["mAP"]:
                        STAGE1_BEST.update({"mAP": eval_metrics["mAP"], "epoch": epoch + 1})
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "metrics": eval_metrics,
                                "config": CFG,
                            },
                            STAGE1_BEST["path"],
                        )
                STAGE1_HISTORY.append(record)
                with (OUTPUT_DIR / "stage1_history.json").open("w", encoding="utf-8") as handle:
                    json.dump(STAGE1_HISTORY, handle, indent=2)
                print(json.dumps(record, indent=2))

            stage1_checkpoint = torch.load(STAGE1_BEST["path"], map_location="cpu")
            model.load_state_dict(stage1_checkpoint["model"])
            print(json.dumps(STAGE1_BEST, indent=2))
            """
        ),
        code_cell(
            """
            def fic_whiten(features, camera_names, regularisation=5.0e-4, min_samples=4):
                whitened = features.copy()
                grouped = defaultdict(list)
                for index, camname in enumerate(camera_names):
                    grouped[camname].append(index)
                for camname, indices in grouped.items():
                    if len(indices) < min_samples:
                        continue
                    x = features[indices]
                    mean = x.mean(axis=0, keepdims=True)
                    centered = x - mean
                    covariance = centered.T @ centered / max(len(indices) - 1, 1)
                    covariance = covariance + regularisation * np.eye(covariance.shape[0], dtype=np.float32)
                    eigvals, eigvecs = np.linalg.eigh(covariance)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    projector = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    projected = centered @ projector.T
                    norms = np.linalg.norm(projected, axis=1, keepdims=True)
                    whitened[indices] = projected / np.clip(norms, 1e-12, None)
                return whitened.astype(np.float32)


            def build_pseudo_records(model_ref, records):
                features, _, _, indices = extract_embeddings(model_ref, uda_eval_loader, flip=True)
                ordered_records = [records[index] for index in indices.tolist()]
                camera_names = [record["camname"] for record in ordered_records]
                embeddings = features.numpy().astype(np.float32)
                embeddings = fic_whiten(embeddings, camera_names, regularisation=CFG["fic_lambda"], min_samples=CFG["dbscan_min_samples"])
                similarity = embeddings @ embeddings.T
                distance = 1.0 - np.clip(similarity, -1.0, 1.0)
                labels = DBSCAN(
                    eps=CFG["dbscan_eps"],
                    min_samples=CFG["dbscan_min_samples"],
                    metric="precomputed",
                    n_jobs=-1,
                ).fit_predict(distance)
                kept_labels = sorted({int(label) for label in labels if label >= 0})
                if not kept_labels:
                    return [], {"clusters": 0, "kept_images": 0, "noise_images": int((labels < 0).sum())}
                relabel = {label: index for index, label in enumerate(kept_labels)}
                pseudo_records = []
                for record, cluster_label in zip(ordered_records, labels.tolist()):
                    if cluster_label < 0:
                        continue
                    pseudo_record = dict(record)
                    pseudo_record["pid"] = relabel[int(cluster_label)]
                    pseudo_records.append(pseudo_record)
                stats = {
                    "clusters": len(kept_labels),
                    "kept_images": len(pseudo_records),
                    "noise_images": int((labels < 0).sum()),
                }
                return pseudo_records, stats


            def build_pseudo_loader(pseudo_records):
                return DataLoader(
                    ReIDImageDataset(pseudo_records, train_transform),
                    batch_size=TRAIN_BATCH_SIZE,
                    sampler=RandomIdentitySampler(pseudo_records, CFG["batch_p"], CFG["batch_k"]),
                    num_workers=CFG["num_workers"],
                    pin_memory=True,
                    drop_last=True,
                )


            STAGE2_HISTORY = []
            STAGE2_BEST = {"mAP": STAGE1_BEST["mAP"], "epoch": 0, "path": str(CHECKPOINT_DIR / "stage2_best.pth")}
            stage2_model = build_model(len(train_pid_map))
            stage2_model.load_state_dict(model.state_dict(), strict=True)

            pseudo_loader = None
            stage2_optimizer = None
            stage2_center_optimizer = None
            stage2_scheduler = None
            stage2_id_loss = None
            stage2_triplet_loss = None
            stage2_center_loss = None

            for epoch in range(CFG["stage2_epochs"]):
                if epoch % CFG["stage2_recluster_every"] == 0 or pseudo_loader is None:
                    pseudo_records, cluster_stats = build_pseudo_records(stage2_model, UDA_SOURCE_RECORDS)
                    if len(pseudo_records) < TRAIN_BATCH_SIZE:
                        print(f"Skipping Stage 2 epoch {epoch + 1}: not enough pseudo-labeled samples")
                        print(json.dumps(cluster_stats, indent=2))
                        break
                    pseudo_loader = build_pseudo_loader(pseudo_records)
                    stage2_model.reset_classifier(len({record['pid'] for record in pseudo_records}))
                    stage2_id_loss, stage2_triplet_loss, stage2_center_loss = build_losses(len({record['pid'] for record in pseudo_records}))
                    stage2_optimizer, stage2_center_optimizer = build_optimizer(stage2_model, stage2_center_loss, CFG["stage2_lr"])
                    stage2_scheduler = build_scheduler(stage2_optimizer, CFG["stage2_epochs"])
                    print(json.dumps({"epoch": epoch + 1, **cluster_stats}, indent=2))

                started = time.time()
                train_metrics = train_one_epoch(
                    stage2_model,
                    pseudo_loader,
                    stage2_optimizer,
                    stage2_center_optimizer,
                    stage2_id_loss,
                    stage2_triplet_loss,
                    stage2_center_loss,
                    epoch,
                    max_iters=CFG["stage2_iters_per_epoch"],
                )
                stage2_scheduler.step()
                record = {
                    "epoch": epoch + 1,
                    "elapsed_sec": round(time.time() - started, 2),
                    "lr": float(stage2_optimizer.param_groups[0]["lr"]),
                    **{key: float(value) for key, value in train_metrics.items()},
                }
                if (epoch + 1) % CFG["stage2_eval_every"] == 0 or epoch == CFG["stage2_epochs"] - 1:
                    eval_metrics = evaluate_model(stage2_model)
                    record.update(eval_metrics)
                    if eval_metrics["mAP"] > STAGE2_BEST["mAP"]:
                        STAGE2_BEST.update({"mAP": eval_metrics["mAP"], "epoch": epoch + 1})
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "model": stage2_model.state_dict(),
                                "metrics": eval_metrics,
                                "config": CFG,
                            },
                            STAGE2_BEST["path"],
                        )
                STAGE2_HISTORY.append(record)
                with (OUTPUT_DIR / "stage2_history.json").open("w", encoding="utf-8") as handle:
                    json.dump(STAGE2_HISTORY, handle, indent=2)
                print(json.dumps(record, indent=2))

            if Path(STAGE2_BEST["path"]).exists():
                stage2_checkpoint = torch.load(STAGE2_BEST["path"], map_location="cpu")
                stage2_model.reset_classifier(len(train_pid_map))
                stage2_model.load_state_dict({
                    key: value for key, value in stage2_checkpoint["model"].items() if not key.startswith("classifier.")
                }, strict=False)
            else:
                stage2_model = copy.deepcopy(model)
            """
        ),
        code_cell(
            """
            FINAL_MODEL = stage2_model if Path(STAGE2_BEST["path"]).exists() else model
            FINAL_METRICS = evaluate_model(FINAL_MODEL)

            BASELINES = {
                "09b_vit_256px_mAP": 0.8014,
                "09f_resnet101ibn_mAP": 0.5277,
                "09g_target_mAP": CFG["target_map"],
            }

            comparison = {
                name: round(FINAL_METRICS["mAP"] - value, 4)
                for name, value in BASELINES.items()
            }

            print(
                json.dumps(
                    {
                        "final_metrics": FINAL_METRICS,
                        "stage1_best": STAGE1_BEST,
                        "stage2_best": STAGE2_BEST,
                        "baseline_delta": comparison,
                    },
                    indent=2,
                )
            )
            """
        ),
        code_cell(
            """
            BEST_MODEL_PATH = Path("/kaggle/working/resnet101ibn_dmt_best.pth")
            METADATA_PATH = Path("/kaggle/working/resnet101ibn_dmt_metadata.json")
            HISTORY_PATH = Path("/kaggle/working/resnet101ibn_dmt_history.json")

            torch.save({"state_dict": FINAL_MODEL.state_dict()}, BEST_MODEL_PATH)

            history_payload = {
                "stage1": STAGE1_HISTORY,
                "stage2": STAGE2_HISTORY,
            }
            with HISTORY_PATH.open("w", encoding="utf-8") as handle:
                json.dump(history_payload, handle, indent=2)

            metadata = {
                "model": "resnet101_ibn_a",
                "recipe": "DMT-inspired 2-stage training",
                "pooling": "GeM",
                "neck": "BNNeck",
                "pretraining": "ImageNet IBN-Net official weights",
                "dataset": "CityFlowV2 ReID",
                "img_size": list(CFG["img_size"]),
                "embedding_dim": model.feat_dim,
                "num_train_ids": len(train_pid_map),
                "num_cameras": len(camname_to_id),
                "stage1_best": STAGE1_BEST,
                "stage2_best": STAGE2_BEST,
                "final_metrics": FINAL_METRICS,
                "baselines": BASELINES,
                "split_stats": SPLIT_STATS,
                "config": CFG,
                "notes": {
                    "uda_include_eval": CFG["uda_include_eval"],
                    "eval_purity": "train-only UDA by default to preserve held-out query/gallery evaluation",
                },
            }
            with METADATA_PATH.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

            print(
                json.dumps(
                    {
                        "best_model": str(BEST_MODEL_PATH),
                        "metadata": str(METADATA_PATH),
                        "history": str(HISTORY_PATH),
                        "checkpoint_dir": str(CHECKPOINT_DIR),
                    },
                    indent=2,
                )
            )
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    payload = notebook()
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()