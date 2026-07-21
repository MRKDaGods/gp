"""Standalone CLIP-SENet v6 VeRi-776 evaluation extracted from 13e.

Source notebook: notebooks/kaggle/13e_clip_senet_eval/13e_clip_senet_eval.ipynb.
The model definition, checkpoint payload handling, ImageNet normalization,
feature extraction, Market1501-style metric, AQE, and k-reciprocal rerank
logic are lifted from the notebook with CLI/output plumbing around them.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

import sys  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from athar.components.embedders.clip_senet_v6 import (  # noqa: E402,F401 — canonical home
    DEFAULT_NUM_CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    AFEMBlock,
    CLIPSENet,
    LoadedBackboneInfo,
    ResNet101IBNBranch,
    TinyCLIPImageBranch,
    VeRiEvalDataset,
    _cpu_safe_hub_deserialization,
    _ResNetFeatureWrapper,
    build_clip_senet,
    build_loader,
    build_transform,
    load_checkpoint,
    torch_load,
)
FILENAME_RE = re.compile(r"^(?P<pid>-?\d+)_c(?P<camid>\d+)")
RERANK_K1_VALUES = [10, 20, 30, 50]
RERANK_K2_VALUES = [3, 6, 10, 15]
RERANK_LAMBDAS = [0.1, 0.2, 0.3, 0.5, 0.7]
SMOKE_MAX_QUERIES = 50
SMOKE_MAX_GALLERY = 200


def resolve_subset_limits(smoke: bool, max_queries: int | None, max_gallery: int | None) -> tuple[int | None, int | None]:
    if smoke:
        return max_queries or SMOKE_MAX_QUERIES, max_gallery or SMOKE_MAX_GALLERY
    return max_queries, max_gallery


def limit_items(items: list[dict], max_items: int | None, label: str) -> list[dict]:
    if max_items is None:
        return items
    if max_items <= 0:
        raise ValueError(f"--max-{label} must be positive when provided")
    return items[:max_items]


def item_camid(item: dict) -> int:
    return int(item.get("camid", item.get("parsed_camid", item.get("sie_index", -1))))


def select_valid_eval_subset(
    query_items: list[dict],
    gallery_items: list[dict],
    max_queries: int | None,
    max_gallery: int | None,
) -> tuple[list[dict], list[dict]]:
    if max_queries is None and max_gallery is None:
        return query_items, gallery_items
    if max_queries is not None and max_queries <= 0:
        raise ValueError("--max-queries must be positive when provided")
    if max_gallery is not None and max_gallery <= 0:
        raise ValueError("--max-gallery must be positive when provided")

    target_queries = max_queries or len(query_items)
    target_gallery = max_gallery or len(gallery_items)
    selected_queries: list[dict] = []
    selected_pids: set[int] = set()
    for query in query_items:
        query_pid = int(query["pid"])
        if query_pid in selected_pids:
            continue
        query_camid = item_camid(query)
        has_valid_match = any(int(gallery["pid"]) == query_pid and item_camid(gallery) != query_camid for gallery in gallery_items)
        if has_valid_match:
            selected_queries.append(query)
            selected_pids.add(query_pid)
        if len(selected_queries) >= target_queries:
            break

    if not selected_queries:
        selected_queries = limit_items(query_items, max_queries, "queries")

    query_pid_camids = {(int(query["pid"]), item_camid(query)) for query in selected_queries}
    selected_gallery: list[dict] = []
    selected_paths: set[str] = set()
    for query in selected_queries:
        query_pid = int(query["pid"])
        query_camid = item_camid(query)
        for gallery in gallery_items:
            gallery_path = str(gallery["path"])
            if gallery_path in selected_paths:
                continue
            if int(gallery["pid"]) == query_pid and item_camid(gallery) != query_camid:
                selected_gallery.append(gallery)
                selected_paths.add(gallery_path)
                break
    for gallery in gallery_items:
        if len(selected_gallery) >= target_gallery:
            break
        gallery_path = str(gallery["path"])
        if gallery_path in selected_paths:
            continue
        if (int(gallery["pid"]), item_camid(gallery)) in query_pid_camids:
            continue
        selected_gallery.append(gallery)
        selected_paths.add(gallery_path)

    return selected_queries, selected_gallery



def parse_veri_record(img_path: Path):
    match = FILENAME_RE.match(img_path.stem)
    if match is None:
        raise RuntimeError(f"Unexpected VeRi filename: {img_path.name}")
    return {
        "path": str(img_path),
        "pid": int(match.group("pid")),
        "camid": int(match.group("camid")) - 1,
    }


def parse_split(split_dir: Path):
    items = []
    pid_set = set()
    for img_path in sorted(split_dir.glob("*.jpg")):
        record = parse_veri_record(img_path)
        if record["pid"] == -1:
            continue
        pid_set.add(record["pid"])
        items.append(record)
    return items, len(pid_set)


def parse_veri_split(split_dir: Path) -> tuple[list[dict], int]:
    return parse_split(split_dir)


def build_veri_loaders(veri_root: Path, image_size: tuple[int, int], batch_size: int):
    required = ("image_query", "image_test")
    missing = [split for split in required if not (veri_root / split).is_dir()]
    if missing:
        raise FileNotFoundError(f"VeRi root {veri_root} is missing required splits: {missing}")
    query_items, query_ids = parse_split(veri_root / "image_query")
    gallery_items, gallery_ids = parse_split(veri_root / "image_test")
    if not query_items or not gallery_items:
        raise RuntimeError(
            f"VeRi split is empty: query={len(query_items)} gallery={len(gallery_items)}"
        )
    query_loader = build_loader(query_items, image_size, batch_size)
    gallery_loader = build_loader(gallery_items, image_size, batch_size)
    return query_loader, gallery_loader, query_items, gallery_items, query_ids, gallery_ids


def build_clipsenet_model(checkpoint: Path, device: str) -> nn.Module:
    checkpoint_path = checkpoint.expanduser().resolve()
    state_dict, _checkpoint_kind, inferred_num_classes = load_checkpoint(checkpoint_path, map_location=device)
    model = build_clip_senet(num_classes=inferred_num_classes).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint load was not strict; "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    return model.eval()


def build_clip_senet_model(checkpoint: Path, device: str) -> nn.Module:
    return build_clipsenet_model(checkpoint, device)


@torch.no_grad()
def extract_clipsenet_features(
    model: nn.Module,
    items,
    img_size: tuple[int, int],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    loader = build_loader(items, img_size, batch_size)
    model.eval()
    features = []
    pids = []
    camids = []
    paths: list[str] = []

    for images, batch_pids, batch_camids, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        batch_features = model(images)
        if isinstance(batch_features, (tuple, list)):
            batch_features = batch_features[-1]
        batch_features = F.normalize(batch_features.float(), p=2, dim=1)
        features.append(batch_features.cpu().numpy())
        pids.append(batch_pids.numpy())
        camids.append(batch_camids.numpy())
        paths.extend(str(path) for path in batch_paths)

    return (
        np.concatenate(features, axis=0).astype(np.float32, copy=False),
        np.concatenate(pids, axis=0),
        np.concatenate(camids, axis=0),
        paths,
    )


@torch.no_grad()
def extract_features(model, dataloader, device: str):
    model.eval()
    features = []
    pids = []
    camids = []

    for images, batch_pids, batch_camids, _ in dataloader:
        images = images.to(device, non_blocking=True)
        batch_features = model(images)
        if isinstance(batch_features, (tuple, list)):
            batch_features = batch_features[-1]
        batch_features = F.normalize(batch_features.float(), p=2, dim=1)
        features.append(batch_features.cpu().numpy())
        pids.append(batch_pids.numpy())
        camids.append(batch_camids.numpy())

    return (
        np.concatenate(features, axis=0),
        np.concatenate(pids, axis=0),
        np.concatenate(camids, axis=0),
    )


def compute_distance_matrix(query_features, gallery_features, metric="cosine"):
    if metric == "cosine":
        sim = query_features @ gallery_features.T
        dist = 1.0 - sim
    elif metric == "euclidean":
        dist = (
            np.sum(query_features ** 2, axis=1, keepdims=True)
            + np.sum(gallery_features ** 2, axis=1, keepdims=True).T
            - 2 * query_features @ gallery_features.T
        )
        dist = np.clip(dist, 0, None)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return dist


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid = 0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove
        if not np.any(matches[q_idx][keep]):
            continue
        raw_cmc = matches[q_idx][keep]
        num_valid += 1
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        ap = (precision * raw_cmc.astype(bool)).sum() / num_rel if num_rel > 0 else 0.0
        all_ap.append(ap)

    if num_valid == 0:
        raise RuntimeError("No valid query found during VeRi evaluation")

    cmc = np.asarray(all_cmc, dtype=np.float32).mean(axis=0)
    mAP = float(np.mean(all_ap))
    return mAP, cmc


def to_metric_dict(mAP, cmc):
    ranks = list(cmc)
    return {
        "mAP": float(mAP),
        "R1": float(ranks[min(0, len(ranks) - 1)]),
        "R5": float(ranks[min(4, len(ranks) - 1)]),
        "R10": float(ranks[min(9, len(ranks) - 1)]),
    }


def metric_sort_key(metrics):
    return (metrics["mAP"], metrics["R1"], metrics["R5"], metrics["R10"])


def print_metrics(label, metrics):
    print(
        f"{label}: mAP={metrics['mAP'] * 100:.4f}%  "
        f"R1={metrics['R1'] * 100:.4f}%  "
        f"R5={metrics['R5'] * 100:.2f}%  "
        f"R10={metrics['R10'] * 100:.2f}%"
    )


def average_query_expansion(features, k, iterations=1):
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
def build_rerank_state(all_features, max_k1, device: str):
    features = torch.as_tensor(all_features, dtype=torch.float32, device=device)
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
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return original_dist, initial_rank


def compute_reranking_torch(original_dist, initial_rank, query_num, k1=20, k2=6, lambda_value=0.3):
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


def evaluate_rerank_sweep(all_features, qf_len, q_pids, g_pids, q_camids, g_camids, device: str):
    rerank_state = build_rerank_state(all_features, max_k1=max(RERANK_K1_VALUES), device=device)
    records = []
    for k1 in RERANK_K1_VALUES:
        for k2 in RERANK_K2_VALUES:
            for lambda_value in RERANK_LAMBDAS:
                distmat = compute_reranking_torch(
                    original_dist=rerank_state[0],
                    initial_rank=rerank_state[1],
                    query_num=qf_len,
                    k1=k1,
                    k2=k2,
                    lambda_value=lambda_value,
                )
                mAP, cmc = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)
                records.append({
                    "k1": int(k1),
                    "k2": int(k2),
                    "lambda": float(lambda_value),
                    "metrics": to_metric_dict(mAP, cmc),
                })
    return max(records, key=lambda record: metric_sort_key(record["metrics"])), records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the 13e CLIP-SENet v6 checkpoint on VeRi-776."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pth or clipsenet_v6_veri776_best.pth")
    parser.add_argument("--veri-root", type=Path, required=True, help="VeRi-776 root containing image_query and image_test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Evaluation device")
    parser.add_argument("--batch-size", type=int, default=64, help="Eval batch size")
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("H", "W"), default=list(IMAGE_SIZE), help="Input image size")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to write metric JSON")
    parser.add_argument("--smoke", action="store_true", help="Run a fast 50 query x 200 gallery validation subset")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit query images for fast validation")
    parser.add_argument("--max-gallery", type=int, default=None, help="Limit gallery images for fast validation")
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", dest="rerank", action="store_true", help="Enable notebook k-reciprocal rerank sweep")
    rerank_group.add_argument("--no-rerank", dest="rerank", action="store_false", help="Disable rerank sweep")
    parser.set_defaults(rerank=False)
    parser.add_argument("--aqe-k", type=int, default=1, help="AQE k applied before rerank; k<=1 preserves notebook rerank behavior")
    return parser.parse_args()


from athar.serving import reid_loaders as _shared_reid_loaders

build_clipsenet_model = _shared_reid_loaders.build_clipsenet_model
build_clip_senet_model = _shared_reid_loaders.build_clip_senet_model
extract_clipsenet_features = _shared_reid_loaders.extract_clipsenet_features
parse_veri_split = _shared_reid_loaders.parse_veri_split


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    image_size = (int(args.img_size[0]), int(args.img_size[1]))
    checkpoint_path = args.checkpoint.expanduser().resolve()
    veri_root = args.veri_root.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()

    state_dict, checkpoint_kind, inferred_num_classes = load_checkpoint(checkpoint_path, map_location=args.device)
    print("DEVICE:", args.device)
    print("CHECKPOINT_PATH:", checkpoint_path)
    print("CHECKPOINT_KIND:", checkpoint_kind)
    print("NUM_CLASSES:", inferred_num_classes)

    model = build_clipsenet_model(checkpoint_path, args.device)

    query_loader, gallery_loader, query_items, gallery_items, query_ids, gallery_ids = build_veri_loaders(
        veri_root=veri_root,
        image_size=image_size,
        batch_size=args.batch_size,
    )
    max_queries, max_gallery = resolve_subset_limits(args.smoke, args.max_queries, args.max_gallery)
    query_items, gallery_items = select_valid_eval_subset(query_items, gallery_items, max_queries, max_gallery)
    if not query_items or not gallery_items:
        raise RuntimeError(f"VeRi subset is empty: query={len(query_items)} gallery={len(gallery_items)}")
    print("VERI_ROOT:", veri_root)
    print(f"Query:   {len(query_items):,} images, {query_ids} IDs")
    print(f"Gallery: {len(gallery_items):,} images, {gallery_ids} IDs")
    if args.smoke or max_queries is not None or max_gallery is not None:
        print(f"SMOKE/SUBSET: max_queries={max_queries}, max_gallery={max_gallery}")

    qf, q_pids, q_camids, _q_paths = extract_clipsenet_features(
        model,
        query_items,
        image_size,
        args.batch_size,
        args.device,
    )
    gf, g_pids, g_camids, _g_paths = extract_clipsenet_features(
        model,
        gallery_items,
        image_size,
        args.batch_size,
        args.device,
    )
    all_features = np.concatenate([qf, gf], axis=0)

    print("qf:", qf.shape)
    print("gf:", gf.shape)

    base_distmat = compute_distance_matrix(qf, gf, metric="cosine")
    base_mAP, base_cmc = eval_market1501(base_distmat, q_pids, g_pids, q_camids, g_camids)
    output = to_metric_dict(base_mAP, base_cmc)
    output["metadata"] = {
        "smoke": bool(args.smoke),
        "max_queries": max_queries,
        "max_gallery": max_gallery,
        "query_count": int(len(query_items)),
        "gallery_count": int(len(gallery_items)),
        "not_for_accuracy_reporting": bool(args.smoke or max_queries is not None or max_gallery is not None),
    }
    print_metrics("Base cosine", output)

    if args.rerank:
        rerank_features = all_features
        if args.aqe_k > 1:
            rerank_features = average_query_expansion(all_features, k=args.aqe_k, iterations=1)
        best_rerank, _ = evaluate_rerank_sweep(
            all_features=rerank_features,
            qf_len=len(qf),
            q_pids=q_pids,
            g_pids=g_pids,
            q_camids=q_camids,
            g_camids=g_camids,
            device=args.device,
        )
        output["rerank_aqe"] = best_rerank["metrics"]
        print_metrics(
            f"Rerank+AQE aqe_k={args.aqe_k} k1={best_rerank['k1']} k2={best_rerank['k2']} lambda={best_rerank['lambda']:.1f}",
            output["rerank_aqe"],
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
