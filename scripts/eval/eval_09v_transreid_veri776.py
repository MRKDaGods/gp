"""Standalone 09v TransReID ViT-B/16 CLIP VeRi-776 evaluation.

Source notebook: notebooks/kaggle/09v_veri776_eval/09v-veri776-eval.ipynb.

The notebook title drifted toward 256x256, but the executable 09v v17 cells
pin IMG_SIZE=(224, 224); docs/findings.md records the same 224x224 ceiling.
This script keeps the notebook's CLIP normalization, SIE camera mapping
(`parsed_camid - 1`), TTA view ordering, concat-patch GeM pooling, AQE, and
15-row rerank grid intact, with CLI and JSON plumbing around them.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.stage2_features.transreid_model import build_transreid
from src.training.evaluate_reid import compute_distance_matrix, eval_market1501


LOGGER = logging.getLogger("eval_09v_transreid_veri776")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


CAMERA_PATTERN = re.compile(r"^(?P<pid>-?\d+)_c(?P<camid>\d+)")
NUM_VERI_CAMERAS = 20
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMG_SIZE = (224, 224)
MULTISCALE_SIZES = [(224, 224), (256, 256)]
CONCAT_PATCH_GEM_P = 3.0
AQE_K_VALUES = [1, 2, 3]
AQE_ITER2_K = 3
CROSS_AQE_K_VALUES = list(AQE_K_VALUES)
MAX_TIME_PER_RERANK = 1500
SINGLE_FLIP_BATCH_SIZE = 64
ENABLE_TEN_CROP = False
TEN_CROP_BATCH_SIZE = 32
TEN_CROP_RESIZE = (288, 288)
SIE_NUM_CAMERAS = 20
EVAL_NO_SIE = False
RERANK_CONFIGS = [
    {"label": "k1=22,k2=6,lambda=0.2", "k1": 22, "k2": 6, "lambda_value": 0.2},
    {"label": "k1=22,k2=7,lambda=0.2", "k1": 22, "k2": 7, "lambda_value": 0.2},
    {"label": "k1=24,k2=7,lambda=0.2", "k1": 24, "k2": 7, "lambda_value": 0.2},
    {"label": "k1=24,k2=8,lambda=0.2", "k1": 24, "k2": 8, "lambda_value": 0.2},
    {"label": "k1=25,k2=7,lambda=0.2", "k1": 25, "k2": 7, "lambda_value": 0.2},
    {"label": "k1=25,k2=8,lambda=0.2", "k1": 25, "k2": 8, "lambda_value": 0.2},
    {"label": "k1=25,k2=8,lambda=0.18", "k1": 25, "k2": 8, "lambda_value": 0.18},
    {"label": "k1=25,k2=8,lambda=0.22", "k1": 25, "k2": 8, "lambda_value": 0.22},
    {"label": "k1=26,k2=8,lambda=0.2", "k1": 26, "k2": 8, "lambda_value": 0.2},
    {"label": "k1=28,k2=8,lambda=0.2", "k1": 28, "k2": 8, "lambda_value": 0.2},
    {"label": "k1=28,k2=9,lambda=0.2", "k1": 28, "k2": 9, "lambda_value": 0.2},
    {"label": "k1=30,k2=10,lambda=0.2", "k1": 30, "k2": 10, "lambda_value": 0.2},
    {"label": "k1=30,k2=10,lambda=0.15", "k1": 30, "k2": 10, "lambda_value": 0.15},
    {"label": "k1=50,k2=15,lambda=0.2", "k1": 50, "k2": 15, "lambda_value": 0.2},
    {"label": "k1=80,k2=15,lambda=0.2", "k1": 80, "k2": 15, "lambda_value": 0.2},
]
EXACT_08_BASELINE_RERANK = {"label": "k1=20,k2=6,lambda=0.3", "k1": 20, "k2": 6, "lambda_value": 0.3}
GUARANTEED_AQE_RERANK_PAIRS: list[dict[str, Any]] = []
DEFAULT_BUNDLE_SPECS = [("single_flip", False), ("concat_patch_flip", False)]
DEVICE = "cpu"
PIN_MEMORY = False
NUM_WORKERS = 4


class VeRiDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], transform: Any):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(item["path"]).convert("RGB")
        return self.transform(image), item["pid"], item["sie_index"], item["path"]


class PathDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        if not isinstance(item, dict):
            raise TypeError(f"Expected VeRi item dict, got {type(item).__name__}: {item!r}")
        return item["path"], int(item["pid"]), int(item["sie_index"])


def parse_veri_record(img_path: Path) -> dict[str, Any]:
    match = CAMERA_PATTERN.match(img_path.stem)
    if match is None:
        raise RuntimeError(f"Unexpected VeRi filename: {img_path.name}")
    pid = int(match.group("pid"))
    parsed_camid = int(match.group("camid"))
    if not 1 <= parsed_camid <= NUM_VERI_CAMERAS:
        raise RuntimeError(f"Camera ID out of range for {img_path.name}: c{parsed_camid:03d}")
    sie_index = parsed_camid - 1
    if not 0 <= sie_index < NUM_VERI_CAMERAS:
        raise RuntimeError(f"SIE index out of range for {img_path.name}: {sie_index}")
    return {
        "path": str(img_path),
        "pid": pid,
        "parsed_camid": parsed_camid,
        "sie_index": sie_index,
    }


def parse_split(split_dir: Path) -> tuple[list[dict[str, Any]], int]:
    items = []
    pid_set = set()
    for img_path in sorted(Path(split_dir).glob("*.jpg")):
        record = parse_veri_record(img_path)
        if record["pid"] == -1:
            continue
        pid_set.add(record["pid"])
        items.append(record)
    return items, len(pid_set)


def build_transform(img_size: tuple[int, int]):
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def batch_size_for_img_size(img_size: tuple[int, int]) -> int:
    longest = max(img_size)
    if longest >= 384:
        return 32
    if longest >= 320:
        return 48
    return 64


def build_loader(items: list[dict[str, Any]], img_size: tuple[int, int], batch_size: int | None = None):
    actual_batch_size = int(batch_size) if batch_size is not None else batch_size_for_img_size(img_size)
    loader = DataLoader(
        VeRiDataset(items, build_transform(img_size)),
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return loader, actual_batch_size


def make_loader(items: list[dict[str, Any]], batch_size: int):
    return DataLoader(
        PathDataset(items),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def load_model(weights_path: Path):
    model = build_transreid(
        num_classes=1,
        num_cameras=SIE_NUM_CAMERAS,
        embed_dim=768,
        vit_model="vit_base_patch16_clip_224.openai",
        pretrained=False,
        weights_path=str(weights_path),
        img_size=IMG_SIZE,
    )
    model._concat_patch = False
    model._gem_p = CONCAT_PATCH_GEM_P
    return model.to(DEVICE).eval()


def build_09v_model(checkpoint: Path, device: str) -> torch.nn.Module:
    global DEVICE, PIN_MEMORY, NUM_WORKERS
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        DEVICE = "cuda:0"
    else:
        DEVICE = "cpu"
        NUM_WORKERS = 0
    PIN_MEMORY = DEVICE.startswith("cuda")
    return load_model(checkpoint.expanduser().resolve())


def build_transreid_model(checkpoint: Path, device: str) -> torch.nn.Module:
    return build_09v_model(checkpoint, device)


def to_metric_dict(mAP: float, cmc: np.ndarray) -> dict[str, float]:
    ranks = list(cmc)
    return {
        "mAP": float(mAP),
        "R1": float(ranks[min(0, len(ranks) - 1)]),
        "R5": float(ranks[min(4, len(ranks) - 1)]),
        "R10": float(ranks[min(9, len(ranks) - 1)]),
    }


def metric_sort_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (metrics["R1"], metrics["mAP"], metrics["R5"], metrics["R10"])


def joint_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = record["metrics"]
    return (
        metrics["R1"] + metrics["mAP"],
        metrics["R1"],
        metrics["mAP"],
        metrics["R5"],
        metrics["R10"],
    )


def print_metrics(label: str, metrics: dict[str, float], duration_sec: float | None = None) -> None:
    suffix = "" if duration_sec is None else f"  ({duration_sec:.1f}s)"
    LOGGER.info(
        "[%s] mAP=%.4f%% R1=%.4f%% R5=%.2f%% R10=%.2f%%%s",
        label,
        metrics["mAP"] * 100,
        metrics["R1"] * 100,
        metrics["R5"] * 100,
        metrics["R10"] * 100,
        suffix,
    )


def resize_and_normalize(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    resized = TF.resize(image, size, interpolation=T.InterpolationMode.BICUBIC)
    tensor = TF.to_tensor(resized)
    return TF.normalize(tensor, mean=CLIP_MEAN, std=CLIP_STD)


def build_single_flip_views(image: Image.Image, img_size: tuple[int, int] = IMG_SIZE) -> list[torch.Tensor]:
    base = resize_and_normalize(image, img_size)
    return [base, torch.flip(base, dims=[2])]


def build_ten_crop_views(image: Image.Image) -> list[torch.Tensor]:
    resized = TF.resize(image, TEN_CROP_RESIZE, interpolation=T.InterpolationMode.BICUBIC)
    crops = TF.five_crop(resized, IMG_SIZE)
    views = []
    for crop in crops:
        normalized = TF.normalize(TF.to_tensor(crop), mean=CLIP_MEAN, std=CLIP_STD)
        views.append(normalized)
        views.append(torch.flip(normalized, dims=[2]))
    return views


def build_view_batches(paths: list[str], mode: str, img_size: tuple[int, int] = IMG_SIZE) -> list[torch.Tensor]:
    num_views = 2 if mode in {"single_flip", "single_flip_no_sie", "concat_patch_flip"} else 10
    view_batches = [[] for _ in range(num_views)]
    for path in paths:
        with Image.open(path) as image_handle:
            image = image_handle.convert("RGB")
            views = (
                build_single_flip_views(image, img_size)
                if mode in {"single_flip", "single_flip_no_sie", "concat_patch_flip"}
                else build_ten_crop_views(image)
            )
        for view_index, view in enumerate(views):
            view_batches[view_index].append(view)
    return [torch.stack(batch, dim=0) for batch in view_batches]


def make_dataloader_at_size(
    items: list[dict[str, Any]],
    batch_size: int,
    size: tuple[int, int],
):
    loader, actual_batch_size = build_loader(items, size, batch_size=batch_size)
    if int(actual_batch_size) != int(batch_size):
        LOGGER.info(
            "Adjusted batch size for size=%s: requested=%s, actual=%s",
            size,
            batch_size,
            actual_batch_size,
        )
    return loader


def set_concat_patch_mode(model: torch.nn.Module, enabled: bool) -> None:
    model._concat_patch = bool(enabled)
    model._gem_p = CONCAT_PATCH_GEM_P


@torch.no_grad()
def extract_features_from_tensor_loader(model: torch.nn.Module, dataloader: DataLoader, disable_sie: bool = False):
    model.eval()
    all_features = []
    all_pids = []
    all_camids = []

    for imgs, pids, camids, _ in dataloader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        real_cam_tensor = camids.to(DEVICE, non_blocking=True).long()
        cam_tensor = torch.zeros_like(real_cam_tensor) if disable_sie else real_cam_tensor

        features = model(imgs, cam_ids=cam_tensor)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        features = F.normalize(features.float(), p=2, dim=1)

        imgs_flip = torch.flip(imgs, dims=[3])
        features_flip = model(imgs_flip, cam_ids=cam_tensor)
        if isinstance(features_flip, (tuple, list)):
            features_flip = features_flip[-1]
        features_flip = F.normalize(features_flip.float(), p=2, dim=1)

        batch_features = F.normalize((features + features_flip) / 2.0, p=2, dim=1)
        all_features.append(batch_features.cpu().numpy())
        all_pids.append(pids.numpy())
        all_camids.append(camids.numpy())

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_pids, axis=0),
        np.concatenate(all_camids, axis=0),
    )


@torch.no_grad()
def extract_features(model: torch.nn.Module, dataloader: DataLoader, mode: str, batch_size: int):
    model.eval()
    if mode == "multi_scale_flip":
        scale_features = []
        final_pids = None
        final_camids = None
        items = dataloader.dataset.items
        for size in MULTISCALE_SIZES:
            scale_loader = make_dataloader_at_size(items, batch_size=batch_size, size=size)
            features, pids, camids = extract_features_from_tensor_loader(model, scale_loader, disable_sie=False)
            scale_features.append(features.astype(np.float32, copy=False))
            if final_pids is None:
                final_pids = pids
                final_camids = camids
        merged = np.zeros_like(scale_features[0], dtype=np.float32)
        for features in scale_features:
            merged += features
        merged /= np.linalg.norm(merged, axis=1, keepdims=True) + 1e-12
        return merged, final_pids, final_camids

    if mode == "concat_patch_flip":
        tensor_loader = make_dataloader_at_size(
            dataloader.dataset.items,
            batch_size=batch_size,
            size=IMG_SIZE,
        )
        set_concat_patch_mode(model, True)
        try:
            return extract_features_from_tensor_loader(model, tensor_loader, disable_sie=False)
        finally:
            set_concat_patch_mode(model, False)

    all_features = []
    all_pids = []
    all_camids = []
    disable_sie = mode == "single_flip_no_sie"

    for paths, pids, camids in dataloader:
        real_cam_tensor = camids.to(DEVICE, non_blocking=True).long()
        cam_tensor = torch.zeros_like(real_cam_tensor) if disable_sie else real_cam_tensor
        view_batches = build_view_batches(list(paths), mode)
        per_view_features = []
        for view_batch in view_batches:
            view_batch = view_batch.to(DEVICE, non_blocking=True)
            features = model(view_batch, cam_ids=cam_tensor)
            if isinstance(features, (tuple, list)):
                features = features[-1]
            per_view_features.append(F.normalize(features.float(), p=2, dim=1).cpu())
        batch_features = F.normalize(torch.stack(per_view_features, dim=0).mean(dim=0), p=2, dim=1)
        all_features.append(batch_features.numpy())
        all_pids.append(pids.numpy())
        all_camids.append(camids.numpy())

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_pids, axis=0),
        np.concatenate(all_camids, axis=0),
    )


def extract_feature_bundle(
    model: torch.nn.Module,
    mode: str,
    query_items: list[dict[str, Any]],
    gallery_items: list[dict[str, Any]],
    batch_size: int,
):
    local_batch_size = (
        batch_size
        if mode in {"single_flip", "single_flip_no_sie", "concat_patch_flip", "multi_scale_flip"}
        else TEN_CROP_BATCH_SIZE
    )
    q_loader_local = make_loader(query_items, batch_size=local_batch_size)
    g_loader_local = make_loader(gallery_items, batch_size=local_batch_size)
    started = time.time()
    q_features, q_pids, q_camids = extract_features(model, q_loader_local, mode, batch_size=local_batch_size)
    g_features, g_pids, g_camids = extract_features(model, g_loader_local, mode, batch_size=local_batch_size)
    elapsed = time.time() - started
    LOGGER.info("Extracted %s features in %.1fs -- q=%s, g=%s", mode, elapsed, q_features.shape, g_features.shape)
    if mode == "multi_scale_flip":
        view_count = 2 * len(MULTISCALE_SIZES)
    elif mode in {"single_flip", "single_flip_no_sie", "concat_patch_flip"}:
        view_count = 2
    else:
        view_count = 10
    return {
        "mode": mode,
        "sie_disabled_at_eval": bool(mode == "single_flip_no_sie"),
        "view_count": view_count,
        "feature_extraction_sec": float(elapsed),
        "q_features": q_features,
        "g_features": g_features,
        "q_pids": q_pids,
        "g_pids": g_pids,
        "q_camids": q_camids,
        "g_camids": g_camids,
    }


def extract_09v_features_with_metadata(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    device: str,
    batch_size: int,
    *,
    stream: Literal["global", "concat_patch_flip"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    global DEVICE, PIN_MEMORY, NUM_WORKERS
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        DEVICE = "cuda:0"
    else:
        DEVICE = "cpu"
        NUM_WORKERS = 0
    PIN_MEMORY = DEVICE.startswith("cuda")

    mode = "single_flip" if stream == "global" else stream
    if mode not in {"single_flip", "concat_patch_flip"}:
        raise ValueError(f"Unknown TransReID stream: {stream}")
    loader = make_loader(items, batch_size=batch_size)
    features, pids, camids = extract_features(model, loader, mode, batch_size=batch_size)
    features = F.normalize(torch.as_tensor(features, dtype=torch.float32), p=2, dim=1).cpu().numpy()
    paths = [str(item["path"]) for item in items]
    return features.astype(np.float32, copy=False), pids, camids, paths


def extract_09v_features(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    device: str,
    batch_size: int,
    *,
    stream: Literal["global", "concat_patch_flip"],
) -> np.ndarray:
    features, _pids, _camids, _paths = extract_09v_features_with_metadata(
        model,
        items,
        device,
        batch_size,
        stream=stream,
    )
    return features


def evaluate_cosine_features(
    q_features: np.ndarray,
    g_features: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
) -> dict[str, float]:
    distmat = compute_distance_matrix(q_features, g_features, metric="cosine")
    mAP, cmc = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)
    return to_metric_dict(mAP, cmc)


def average_query_expansion(features: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
    current = features.astype(np.float32, copy=True)
    if k <= 1:
        return current
    for _ in range(iterations):
        sim = current @ current.T
        topk = min(k, sim.shape[1])
        kth = max(topk - 1, 0)
        topk_idx = np.argpartition(-sim, kth=kth, axis=1)[:, :topk]
        expanded = np.zeros_like(current)
        for index in range(current.shape[0]):
            expanded[index] = current[topk_idx[index]].mean(axis=0)
        norms = np.linalg.norm(expanded, axis=1, keepdims=True) + 1e-12
        current = expanded / norms
    return current


@torch.no_grad()
def build_rerank_state(all_features: np.ndarray, max_k1: int):
    features = torch.as_tensor(all_features, dtype=torch.float32, device=DEVICE)
    features = F.normalize(features, p=2, dim=1)
    similarity = torch.matmul(features, features.T)
    original_dist = (2.0 - 2.0 * similarity).clamp_min_(0).cpu().numpy().astype(np.float32)
    initial_rank = torch.topk(
        similarity,
        k=min(max_k1 + 1, similarity.shape[1]),
        dim=1,
        largest=True,
        sorted=True,
    ).indices.cpu().numpy().astype(np.int32)
    del features, similarity
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    return original_dist, initial_rank


def compute_reranking_torch(
    original_dist: np.ndarray,
    initial_rank: np.ndarray,
    query_num: int,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    all_num = original_dist.shape[0]
    V = np.zeros((all_num, all_num), dtype=np.float16)
    half_k1 = int(np.round(k1 / 2.0))

    for index in range(all_num):
        forward = initial_rank[index, :k1 + 1]
        backward = initial_rank[forward, :k1 + 1]
        reciprocal = forward[np.any(backward == index, axis=1)]
        reciprocal_expansion = reciprocal.copy()

        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, :half_k1 + 1]
            candidate_backward = initial_rank[candidate_forward, :half_k1 + 1]
            candidate_reciprocal = candidate_forward[np.any(candidate_backward == candidate, axis=1)]
            if candidate_reciprocal.size == 0:
                continue
            overlap = np.intersect1d(candidate_reciprocal, reciprocal)
            if overlap.size > (2.0 / 3.0) * candidate_reciprocal.size:
                reciprocal_expansion = np.concatenate((reciprocal_expansion, candidate_reciprocal))

        reciprocal_expansion = np.unique(reciprocal_expansion)
        weights = np.exp(-original_dist[index, reciprocal_expansion]).astype(np.float32)
        V[index, reciprocal_expansion] = (weights / (weights.sum() + 1e-12)).astype(np.float16)

    if k2 > 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for index in range(all_num):
            V_qe[index] = V[initial_rank[index, :k2]].mean(axis=0)
        V = V_qe

    inv_index = [np.flatnonzero(V[:, column]) for column in range(all_num)]
    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    for index in range(query_num):
        temp_min = np.zeros(all_num, dtype=np.float32)
        non_zero = np.flatnonzero(V[index])
        for nz in non_zero:
            related = inv_index[nz]
            temp_min[related] += np.minimum(np.float32(V[index, nz]), V[related, nz].astype(np.float32))
        jaccard_dist[index] = 1.0 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist * (1.0 - lambda_value) + original_dist[:query_num] * lambda_value
    return final_dist[:, query_num:]


def cfg_json(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "k1": int(cfg["k1"]),
        "k2": int(cfg["k2"]),
        "lambda": float(cfg["lambda_value"]),
    }


def copy_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": cfg["label"],
        "k1": int(cfg["k1"]),
        "k2": int(cfg["k2"]),
        "lambda_value": float(cfg["lambda_value"]),
    }


def evaluate_rerank_variant(
    state: tuple[np.ndarray, np.ndarray],
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    cfg: dict[str, Any],
    variant_label: str,
) -> tuple[dict[str, float], bool]:
    started = time.time()
    original_dist, initial_rank = state
    distmat = compute_reranking_torch(
        original_dist=original_dist,
        initial_rank=initial_rank,
        query_num=len(q_pids),
        k1=cfg["k1"],
        k2=cfg["k2"],
        lambda_value=cfg["lambda_value"],
    )
    mAP, cmc = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)
    elapsed = time.time() - started
    metrics = to_metric_dict(mAP, cmc)
    metrics["duration_sec"] = float(elapsed)
    print_metrics(f"{variant_label} {cfg['label']}", metrics, duration_sec=elapsed)
    exceeded = elapsed > MAX_TIME_PER_RERANK
    if exceeded:
        LOGGER.warning(
            "Rerank guard triggered after %.1fs on %s %s; skipping remaining rerank variants.",
            elapsed,
            variant_label,
            cfg["label"],
        )
    return metrics, exceeded


def make_candidate(
    label: str,
    feature_bundle: str,
    stage: str,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = {
        "label": label,
        "feature_bundle": feature_bundle,
        "stage": stage,
        "metrics": metrics,
    }
    if extra is not None:
        candidate.update(extra)
    return candidate


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    output = {
        "label": record["label"],
        "feature_bundle": record["feature_bundle"],
        "stage": record["stage"],
        "mAP": float(metrics["mAP"]),
        "R1": float(metrics["R1"]),
        "R5": float(metrics["R5"]),
        "R10": float(metrics["R10"]),
    }
    if "aqe" in record:
        output["aqe"] = record["aqe"]
    if "rerank" in record:
        output["rerank"] = cfg_json(record["rerank"])
    return output


def compact_rerank_entry(record: dict[str, Any]) -> dict[str, float | int]:
    cfg = record["rerank"]
    metrics = record["metrics"]
    return {
        "k1": int(cfg["k1"]),
        "k2": int(cfg["k2"]),
        "lambda": float(cfg["lambda_value"]),
        "mAP": float(metrics["mAP"]),
        "R1": float(metrics["R1"]),
    }


def rerank_key(cfg: dict[str, Any]) -> tuple[int, int, float, str]:
    return (int(cfg["k1"]), int(cfg["k2"]), float(cfg["lambda_value"]), cfg["label"])


def run_full_eval(
    weights_path: Path,
    veri_root: Path,
    batch_size: int,
    run_rerank: bool,
    output_aqe_k: int,
) -> dict[str, Any]:
    LOGGER.info("weights=%s", weights_path)
    LOGGER.info("veri_root=%s", veri_root)
    query, n_q = parse_split(veri_root / "image_query")
    gallery, n_g = parse_split(veri_root / "image_test")
    if not query:
        raise RuntimeError(f"No query jpg files found under {veri_root / 'image_query'}")
    if not gallery:
        raise RuntimeError(f"No gallery jpg files found under {veri_root / 'image_test'}")
    LOGGER.info("Query: %s images, %s IDs", f"{len(query):,}", n_q)
    LOGGER.info("Gallery: %s images, %s IDs", f"{len(gallery):,}", n_g)
    LOGGER.info("First 5 (path, parsed_camid, sie_index) tuples: %s", [
        (item["path"], item["parsed_camid"], item["sie_index"])
        for item in (query + gallery)[:5]
    ])

    model = load_model(weights_path)
    total_start = time.time()
    bundle_specs = list(DEFAULT_BUNDLE_SPECS)
    if EVAL_NO_SIE:
        bundle_specs.append(("single_flip_no_sie", True))

    feature_bundles = {}
    for bundle_name, _ in bundle_specs:
        feature_bundles[bundle_name] = extract_feature_bundle(model, bundle_name, query, gallery, batch_size=batch_size)

    results: dict[str, Any] = {
        "script": "eval_09v_transreid_veri776",
        "checkpoint": str(weights_path),
        "dataset_root": str(veri_root),
        "img_size": list(IMG_SIZE),
        "device": DEVICE,
        "batch_size": int(batch_size),
        "mAP": None,
        "R1": None,
        "R5": None,
        "R10": None,
        "rerank_sweep": [],
        "best": None,
        "metadata": {
            "weights_name": weights_path.name,
            "multi_scale_sizes": [list(size) for size in MULTISCALE_SIZES],
            "concat_patch_gem_p": float(CONCAT_PATCH_GEM_P),
            "ten_crop_enabled": ENABLE_TEN_CROP,
            "ten_crop_resize": list(TEN_CROP_RESIZE),
            "aqe_k_values": list(AQE_K_VALUES),
            "aqe_iter2_k": AQE_ITER2_K,
            "cross_aqe_k_values": list(CROSS_AQE_K_VALUES),
            "guaranteed_aqe_rerank_pairs": [
                {"aqe_k": int(pair["aqe_k"]), "rerank": copy_cfg(pair["rerank"])}
                for pair in GUARANTEED_AQE_RERANK_PAIRS
            ],
            "max_time_per_rerank_sec": MAX_TIME_PER_RERANK,
            "rerank_configs": [copy_cfg(cfg) for cfg in RERANK_CONFIGS],
            "exact_08_baseline_rerank": copy_cfg(EXACT_08_BASELINE_RERANK),
            "sie_enabled": True,
            "eval_no_sie": EVAL_NO_SIE,
            "sie_num_cameras": SIE_NUM_CAMERAS,
            "sie_indexing": "VeRi filename camid is 1-based; model SIE index is parsed_camid - 1 (0-based [0,19]).",
            "single_flip_batch_size": int(batch_size),
            "ten_crop_batch_size": TEN_CROP_BATCH_SIZE,
            "evaluated_feature_bundles": [bundle_name for bundle_name, _ in bundle_specs],
        },
        "feature_sets": {
            "ten_crop": {
                "enabled": False,
                "skipped": True,
                "reason": "disabled after underperforming the single-flip baseline",
            } if not ENABLE_TEN_CROP else {},
        },
        "top_candidates": [],
        "best_r1": None,
        "best_map": None,
    }

    all_candidates = []

    for bundle_name, sie_disabled_at_eval in bundle_specs:
        bundle = feature_bundles[bundle_name]
        q_features = bundle["q_features"]
        g_features = bundle["g_features"]
        q_pids = bundle["q_pids"]
        g_pids = bundle["g_pids"]
        q_camids = bundle["q_camids"]
        g_camids = bundle["g_camids"]
        all_features = np.concatenate([q_features, g_features], axis=0)

        bundle_results: dict[str, Any] = {
            "feature_extraction_sec": float(bundle["feature_extraction_sec"]),
            "view_count": int(bundle["view_count"]),
            "query_shape": list(q_features.shape),
            "gallery_shape": list(g_features.shape),
            "sie_disabled_at_eval": bool(sie_disabled_at_eval),
            "baseline": None,
            "exact_08_baseline_rerank": None,
            "rerank_sweep": [],
            "aqe_sweep": [],
            "aqe_iter2": None,
            "aqe_rerank_cross": [],
            "rerank_guard_triggered": False,
        }

        baseline = evaluate_cosine_features(q_features, g_features, q_pids, g_pids, q_camids, g_camids)
        bundle_results["baseline"] = baseline
        print_metrics(f"{bundle_name} baseline", baseline)
        all_candidates.append(make_candidate(f"{bundle_name} baseline", bundle_name, "baseline", baseline))

        aqe_feature_cache = {}
        for k in AQE_K_VALUES:
            expanded = average_query_expansion(all_features, k=k, iterations=1)
            aqe_feature_cache[k] = expanded
            q_aqe = expanded[:len(q_pids)]
            g_aqe = expanded[len(q_pids):]
            started = time.time()
            metrics = evaluate_cosine_features(q_aqe, g_aqe, q_pids, g_pids, q_camids, g_camids)
            elapsed = time.time() - started
            record = {
                "label": f"{bundle_name} AQE k={k}",
                "config": {"k": int(k), "iterations": 1},
                "metrics": {**metrics, "duration_sec": float(elapsed)},
            }
            bundle_results["aqe_sweep"].append(record)
            print_metrics(record["label"], record["metrics"], duration_sec=elapsed)
            all_candidates.append(make_candidate(record["label"], bundle_name, "aqe", record["metrics"], {"aqe": record["config"]}))

        expanded_iter2 = average_query_expansion(all_features, k=AQE_ITER2_K, iterations=2)
        q_iter2 = expanded_iter2[:len(q_pids)]
        g_iter2 = expanded_iter2[len(q_pids):]
        started = time.time()
        iter2_metrics = evaluate_cosine_features(q_iter2, g_iter2, q_pids, g_pids, q_camids, g_camids)
        elapsed = time.time() - started
        bundle_results["aqe_iter2"] = {
            "label": f"{bundle_name} AQE iter2 k={AQE_ITER2_K}",
            "config": {"k": int(AQE_ITER2_K), "iterations": 2},
            "metrics": {**iter2_metrics, "duration_sec": float(elapsed)},
        }
        print_metrics(bundle_results["aqe_iter2"]["label"], bundle_results["aqe_iter2"]["metrics"], duration_sec=elapsed)
        all_candidates.append(
            make_candidate(
                bundle_results["aqe_iter2"]["label"],
                bundle_name,
                "aqe_iter2",
                bundle_results["aqe_iter2"]["metrics"],
                {"aqe": bundle_results["aqe_iter2"]["config"]},
            )
        )

        if run_rerank:
            max_k1 = max(max(cfg["k1"] for cfg in RERANK_CONFIGS), int(EXACT_08_BASELINE_RERANK["k1"]))
            LOGGER.info("Building rerank state for %s base features ...", bundle_name)
            base_state = build_rerank_state(all_features, max_k1=max_k1)

            exact_metrics, exceeded = evaluate_rerank_variant(
                base_state,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                EXACT_08_BASELINE_RERANK,
                f"{bundle_name} rerank",
            )
            bundle_results["exact_08_baseline_rerank"] = {
                "label": f"{bundle_name} rerank {EXACT_08_BASELINE_RERANK['label']}",
                "config": copy_cfg(EXACT_08_BASELINE_RERANK),
                "metrics": exact_metrics,
            }
            all_candidates.append(
                make_candidate(
                    bundle_results["exact_08_baseline_rerank"]["label"],
                    bundle_name,
                    "exact_08_baseline_rerank",
                    exact_metrics,
                    {"rerank": copy_cfg(EXACT_08_BASELINE_RERANK)},
                )
            )
            if exceeded:
                bundle_results["rerank_guard_triggered"] = True

            if not bundle_results["rerank_guard_triggered"]:
                for cfg in RERANK_CONFIGS:
                    metrics, exceeded = evaluate_rerank_variant(base_state, q_pids, g_pids, q_camids, g_camids, cfg, f"{bundle_name} rerank")
                    record = {
                        "label": f"{bundle_name} rerank {cfg['label']}",
                        "rerank": copy_cfg(cfg),
                        "metrics": metrics,
                    }
                    bundle_results["rerank_sweep"].append(record)
                    all_candidates.append(make_candidate(record["label"], bundle_name, "rerank", metrics, {"rerank": copy_cfg(cfg)}))
                    if exceeded:
                        bundle_results["rerank_guard_triggered"] = True
                        break

            cross_aqe_values = sorted({int(k) for k in CROSS_AQE_K_VALUES if int(k) in aqe_feature_cache})
            cross_configs_by_k = {k: [copy_cfg(cfg) for cfg in RERANK_CONFIGS] for k in cross_aqe_values}
            for pair in GUARANTEED_AQE_RERANK_PAIRS:
                aqe_k = int(pair["aqe_k"])
                if aqe_k not in aqe_feature_cache:
                    continue
                configs = cross_configs_by_k.setdefault(aqe_k, [])
                guaranteed_cfg = copy_cfg(pair["rerank"])
                if rerank_key(guaranteed_cfg) not in {rerank_key(cfg) for cfg in configs}:
                    configs.append(guaranteed_cfg)

            if cross_configs_by_k and not bundle_results["rerank_guard_triggered"]:
                cross_guard_triggered = False
                for k in sorted(cross_configs_by_k):
                    expanded = aqe_feature_cache[k]
                    cross_configs = cross_configs_by_k[k]
                    max_cross_k1 = max(cfg["k1"] for cfg in cross_configs)
                    LOGGER.info("Building rerank state for %s AQE k=%s cross grid ...", bundle_name, k)
                    aqe_state = build_rerank_state(expanded, max_k1=max_cross_k1)
                    for cfg in cross_configs:
                        metrics, exceeded = evaluate_rerank_variant(aqe_state, q_pids, g_pids, q_camids, g_camids, cfg, f"{bundle_name} AQE(k={k})+rerank")
                        record = {
                            "label": f"{bundle_name} AQE(k={k}) + {cfg['label']}",
                            "aqe": {"k": int(k), "iterations": 1},
                            "rerank": copy_cfg(cfg),
                            "metrics": metrics,
                        }
                        bundle_results["aqe_rerank_cross"].append(record)
                        all_candidates.append(make_candidate(record["label"], bundle_name, "aqe_rerank_cross", metrics, {"aqe": record["aqe"], "rerank": record["rerank"]}))
                        if bundle_name == "concat_patch_flip" and k == output_aqe_k:
                            results["rerank_sweep"].append(compact_rerank_entry(record))
                        if exceeded:
                            bundle_results["rerank_guard_triggered"] = True
                            cross_guard_triggered = True
                            break
                    del aqe_state
                    gc.collect()
                    if DEVICE.startswith("cuda"):
                        torch.cuda.empty_cache()
                    if cross_guard_triggered:
                        break

            del base_state

        results["feature_sets"][bundle_name] = bundle_results
        gc.collect()
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

    ranked_by_r1 = sorted(all_candidates, key=lambda candidate: metric_sort_key(candidate["metrics"]), reverse=True)
    ranked_by_map = sorted(all_candidates, key=lambda candidate: candidate["metrics"]["mAP"], reverse=True)
    ranked_by_joint = sorted(all_candidates, key=joint_sort_key, reverse=True)
    best = ranked_by_joint[0] if ranked_by_joint else None
    results["top_candidates"] = [compact_record(candidate) for candidate in ranked_by_joint[:10]]
    results["best_r1"] = compact_record(ranked_by_r1[0]) if ranked_by_r1 else None
    results["best_map"] = compact_record(ranked_by_map[0]) if ranked_by_map else None
    results["best"] = compact_record(best) if best else None
    if best:
        best_metrics = best["metrics"]
        results["mAP"] = float(best_metrics["mAP"])
        results["R1"] = float(best_metrics["R1"])
        results["R5"] = float(best_metrics["R5"])
        results["R10"] = float(best_metrics["R10"])
    results["metadata"]["total_runtime_sec"] = float(time.time() - total_start)

    del model
    for bundle in feature_bundles.values():
        del bundle
    gc.collect()
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the 09v TransReID ViT-B/16 CLIP checkpoint on VeRi-776.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to vehicle_transreid_vit_base_veri776.pth")
    parser.add_argument("--veri-root", required=True, type=Path, help="VeRi-776 root containing image_query/ and image_test/")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Evaluation device. 'cuda' maps to cuda:0.")
    parser.add_argument("--batch-size", type=int, default=SINGLE_FLIP_BATCH_SIZE, help="Batch size for feature extraction")
    parser.add_argument("--output-json", required=True, type=Path, help="Path to write JSON metrics")
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", dest="rerank", action="store_true", help="Run the notebook rerank sweeps")
    rerank_group.add_argument("--no-rerank", dest="rerank", action="store_false", help="Skip rerank sweeps")
    parser.set_defaults(rerank=True)
    parser.add_argument("--aqe-k", type=int, default=2, choices=AQE_K_VALUES, help="AQE k whose concat_patch_flip rerank sweep is exposed at top level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DEVICE, PIN_MEMORY, NUM_WORKERS
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        DEVICE = "cuda:0"
    else:
        DEVICE = "cpu"
        NUM_WORKERS = 0
    PIN_MEMORY = DEVICE.startswith("cuda")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not (args.veri_root / "image_query").is_dir() or not (args.veri_root / "image_test").is_dir():
        raise FileNotFoundError(f"VeRi root must contain image_query/ and image_test/: {args.veri_root}")

    results = run_full_eval(
        weights_path=args.checkpoint.resolve(),
        veri_root=args.veri_root.resolve(),
        batch_size=args.batch_size,
        run_rerank=bool(args.rerank),
        output_aqe_k=int(args.aqe_k),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    LOGGER.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()