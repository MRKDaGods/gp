"""VeRi-776 score-fusion evaluator for the 14t CLIP-SENet x TransReID result.

This is a standalone single-camera ReID evaluator. It intentionally does not
wire the VeRi-only fusion into the CityFlow MTMC pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_09v_transreid_veri776 import (  # noqa: E402
    build_09v_model,
    extract_09v_features_with_metadata,
    parse_split as parse_09v_split,
)
from eval_clip_senet_veri776 import (  # noqa: E402
    build_clipsenet_model,
    extract_clipsenet_features,
    parse_veri_split as parse_clipsenet_split,
)
from src.training.evaluate_reid import eval_market1501  # noqa: E402


RERANK_CANONICAL = {"k1": 80, "k2": 15, "lambda_value": 0.2}
AQE_K = 3
WEIGHTS = [round(i / 10, 1) for i in range(11)]


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def score_similarity(
    q_tr: np.ndarray,
    g_tr: np.ndarray,
    q_cs: np.ndarray,
    g_cs: np.ndarray,
    w: float,
) -> np.ndarray:
    return w * (q_cs @ g_cs.T) + (1.0 - w) * (q_tr @ g_tr.T)


def score_all_similarity(all_tr: np.ndarray, all_cs: np.ndarray, w: float) -> np.ndarray:
    return w * (all_cs @ all_cs.T) + (1.0 - w) * (all_tr @ all_tr.T)


def compute_distance_from_similarity(similarity: np.ndarray) -> np.ndarray:
    return (1.0 - similarity).astype(np.float32, copy=False)


def average_query_expansion(features: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
    current = l2_normalize(features)
    if k <= 1:
        return current
    for _ in range(iterations):
        sim = current @ current.T
        topk = min(int(k), sim.shape[1])
        kth = max(topk - 1, 0)
        topk_idx = np.argpartition(-sim, kth=kth, axis=1)[:, :topk]
        expanded = np.zeros_like(current, dtype=np.float32)
        for index in range(current.shape[0]):
            expanded[index] = current[topk_idx[index]].mean(axis=0)
        current = l2_normalize(expanded)
    return current


def build_rerank_state_from_similarity(similarity: np.ndarray, max_k1: int):
    sim = np.asarray(similarity, dtype=np.float32)
    original_dist = compute_distance_from_similarity(sim)
    k = min(int(max_k1) + 1, sim.shape[1])
    initial_rank = np.argsort(-sim, axis=1)[:, :k].astype(np.int32, copy=False)
    return original_dist, initial_rank


def compute_reranking_torch(
    original_dist: np.ndarray,
    initial_rank: np.ndarray,
    query_num: int,
    k1: int = 80,
    k2: int = 15,
    lambda_value: float = 0.2,
) -> np.ndarray:
    all_num = original_dist.shape[0]
    V = np.zeros((all_num, all_num), dtype=np.float16)
    half_k1 = int(np.round(k1 / 2.0))

    for index in range(all_num):
        forward = initial_rank[index, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal = forward[np.any(backward == index, axis=1)]
        reciprocal_expansion = reciprocal.copy()

        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, : half_k1 + 1]
            candidate_backward = initial_rank[candidate_forward, : half_k1 + 1]
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


def to_metric_dict(mAP: float, cmc: np.ndarray) -> dict[str, float]:
    ranks = list(cmc)
    return {
        "mAP": float(mAP),
        "R1": float(ranks[min(0, len(ranks) - 1)]),
        "R5": float(ranks[min(4, len(ranks) - 1)]),
        "R10": float(ranks[min(9, len(ranks) - 1)]),
    }


def evaluate_dist(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
) -> dict[str, float]:
    mAP, cmc = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids)
    return to_metric_dict(mAP, cmc)


def evaluate_similarity(
    similarity: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
) -> dict[str, float]:
    return evaluate_dist(compute_distance_from_similarity(similarity), q_pids, g_pids, q_camids, g_camids)


def evaluate_aqe_rerank(
    all_tr: np.ndarray,
    all_cs: np.ndarray,
    query_num: int,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    *,
    w_clipsenet: float,
    aqe_k: int,
    rerank_k1: int,
    rerank_k2: int,
    rerank_lambda: float,
) -> dict[str, Any]:
    aqe_tr = average_query_expansion(all_tr, k=aqe_k, iterations=1)
    aqe_cs = average_query_expansion(all_cs, k=aqe_k, iterations=1)
    q_tr_aqe, g_tr_aqe = aqe_tr[:query_num], aqe_tr[query_num:]
    q_cs_aqe, g_cs_aqe = aqe_cs[:query_num], aqe_cs[query_num:]
    qg_aqe = score_similarity(q_tr_aqe, g_tr_aqe, q_cs_aqe, g_cs_aqe, w_clipsenet)
    cosine_aqe = evaluate_similarity(qg_aqe, q_pids, g_pids, q_camids, g_camids)
    all_similarity_aqe = score_all_similarity(aqe_tr, aqe_cs, w_clipsenet)
    rerank_state = build_rerank_state_from_similarity(all_similarity_aqe, max_k1=rerank_k1)
    rerank_dist = compute_reranking_torch(
        rerank_state[0],
        rerank_state[1],
        query_num=query_num,
        k1=rerank_k1,
        k2=rerank_k2,
        lambda_value=rerank_lambda,
    )
    rerank_aqe = evaluate_dist(rerank_dist, q_pids, g_pids, q_camids, g_camids)
    return {"cosine_aqe": cosine_aqe, "rerank_aqe": rerank_aqe}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_alignment(label: str, left: np.ndarray | list[str], right: np.ndarray | list[str]) -> None:
    if isinstance(left, list) or isinstance(right, list):
        if list(left) != list(right):
            raise RuntimeError(f"{label} path alignment mismatch")
        return
    if not np.array_equal(left, right):
        raise RuntimeError(f"{label} alignment mismatch")


def assert_unit_norm(label: str, features: np.ndarray) -> None:
    norms = np.linalg.norm(features, axis=1)
    if not np.all(np.isfinite(norms)) or not np.allclose(norms, 1.0, atol=2e-4):
        raise RuntimeError(f"{label} features are not finite unit vectors")


def metric_sort_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (metrics["mAP"], metrics["R1"], metrics["R5"], metrics["R10"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 14t CLIP-SENet x TransReID score fusion on VeRi-776.")
    parser.add_argument("--transreid-checkpoint", required=True, type=Path)
    parser.add_argument("--clipsenet-checkpoint", required=True, type=Path)
    parser.add_argument("--veri-root", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--w-clipsenet", type=float, default=0.7)
    parser.add_argument("--transreid-stream", choices=("global", "concat_patch_flip"), default="global")
    parser.add_argument("--aqe-k", type=int, default=AQE_K)
    parser.add_argument("--rerank-k1", type=int, default=RERANK_CANONICAL["k1"])
    parser.add_argument("--rerank-k2", type=int, default=RERANK_CANONICAL["k2"])
    parser.add_argument("--rerank-lambda", type=float, default=RERANK_CANONICAL["lambda_value"])
    parser.add_argument("--transreid-batch-size", type=int, default=64)
    parser.add_argument("--clipsenet-batch-size", type=int, default=64)
    parser.add_argument("--clipsenet-img-size", type=int, nargs=2, metavar=("H", "W"), default=[320, 320])
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--skip-drift-parents", action="store_true")
    parser.add_argument("--weights-sweep", action="store_true")
    parser.add_argument("--concat-sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    veri_root = args.veri_root.expanduser().resolve()
    query_dir = veri_root / "image_query"
    gallery_dir = veri_root / "image_test"
    if not query_dir.is_dir() or not gallery_dir.is_dir():
        raise FileNotFoundError(f"VeRi root must contain image_query/ and image_test/: {veri_root}")

    q_cs_items, _ = parse_clipsenet_split(query_dir)
    g_cs_items, _ = parse_clipsenet_split(gallery_dir)
    q_tr_items, _ = parse_09v_split(query_dir)
    g_tr_items, _ = parse_09v_split(gallery_dir)
    validate_alignment("query parser", [item["path"] for item in q_cs_items], [item["path"] for item in q_tr_items])
    validate_alignment("gallery parser", [item["path"] for item in g_cs_items], [item["path"] for item in g_tr_items])

    print(f"Query images: {len(q_cs_items):,}")
    print(f"Gallery images: {len(g_cs_items):,}")
    clipsenet = build_clipsenet_model(args.clipsenet_checkpoint, args.device)
    transreid = build_09v_model(args.transreid_checkpoint, args.device)

    img_size = (int(args.clipsenet_img_size[0]), int(args.clipsenet_img_size[1]))
    q_cs, q_cs_pids, q_cs_camids, q_cs_paths = extract_clipsenet_features(
        clipsenet, q_cs_items, img_size, args.clipsenet_batch_size, args.device
    )
    g_cs, g_cs_pids, g_cs_camids, g_cs_paths = extract_clipsenet_features(
        clipsenet, g_cs_items, img_size, args.clipsenet_batch_size, args.device
    )
    q_tr, q_tr_pids, q_tr_camids, q_tr_paths = extract_09v_features_with_metadata(
        transreid, q_tr_items, args.device, args.transreid_batch_size, stream=args.transreid_stream
    )
    g_tr, g_tr_pids, g_tr_camids, g_tr_paths = extract_09v_features_with_metadata(
        transreid, g_tr_items, args.device, args.transreid_batch_size, stream=args.transreid_stream
    )

    validate_alignment("query paths", q_cs_paths, q_tr_paths)
    validate_alignment("gallery paths", g_cs_paths, g_tr_paths)
    validate_alignment("query pids", q_cs_pids, q_tr_pids)
    validate_alignment("gallery pids", g_cs_pids, g_tr_pids)
    validate_alignment("query camids", q_cs_camids, q_tr_camids)
    validate_alignment("gallery camids", g_cs_camids, g_tr_camids)
    for label, features in (("q_cs", q_cs), ("g_cs", g_cs), ("q_tr", q_tr), ("g_tr", g_tr)):
        assert_unit_norm(label, features)

    q_pids, g_pids = q_cs_pids, g_cs_pids
    q_camids, g_camids = q_cs_camids, g_cs_camids
    query_num = len(q_pids)
    all_tr = np.concatenate([q_tr, g_tr], axis=0)
    all_cs = np.concatenate([q_cs, g_cs], axis=0)

    raw_similarity = score_similarity(q_tr, g_tr, q_cs, g_cs, args.w_clipsenet)
    raw_cosine = evaluate_similarity(raw_similarity, q_pids, g_pids, q_camids, g_camids)
    headline = evaluate_aqe_rerank(
        all_tr,
        all_cs,
        query_num,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        w_clipsenet=args.w_clipsenet,
        aqe_k=args.aqe_k,
        rerank_k1=args.rerank_k1,
        rerank_k2=args.rerank_k2,
        rerank_lambda=args.rerank_lambda,
    )
    best = dict(headline["rerank_aqe"])
    best.update({"w_clipsenet": float(args.w_clipsenet), "w_transreid": float(1.0 - args.w_clipsenet)})
    all_rows = [{"w_clipsenet": float(args.w_clipsenet), **headline["rerank_aqe"]}]

    if args.weights_sweep:
        all_rows = []
        for w in WEIGHTS:
            row = evaluate_aqe_rerank(
                all_tr,
                all_cs,
                query_num,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                w_clipsenet=w,
                aqe_k=args.aqe_k,
                rerank_k1=args.rerank_k1,
                rerank_k2=args.rerank_k2,
                rerank_lambda=args.rerank_lambda,
            )["rerank_aqe"]
            all_rows.append({"w_clipsenet": float(w), "w_transreid": float(1.0 - w), **row})
        best = max(all_rows, key=metric_sort_key)

    drift_parents: dict[str, Any] = {}
    if not args.skip_drift_parents:
        q_tr_concat, _, _, _ = extract_09v_features_with_metadata(
            transreid, q_tr_items, args.device, args.transreid_batch_size, stream="concat_patch_flip"
        )
        g_tr_concat, _, _, _ = extract_09v_features_with_metadata(
            transreid, g_tr_items, args.device, args.transreid_batch_size, stream="concat_patch_flip"
        )
        tr_concat_all = np.concatenate([q_tr_concat, g_tr_concat], axis=0)
        tr_parent = evaluate_aqe_rerank(
            tr_concat_all,
            tr_concat_all,
            query_num,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            w_clipsenet=0.0,
            aqe_k=3,
            rerank_k1=80,
            rerank_k2=15,
            rerank_lambda=0.2,
        )["rerank_aqe"]
        cs_parent = evaluate_aqe_rerank(
            all_cs,
            all_cs,
            query_num,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            w_clipsenet=1.0,
            aqe_k=10,
            rerank_k1=50,
            rerank_k2=10,
            rerank_lambda=0.1,
        )["rerank_aqe"]
        drift_parents = {
            "transreid_09v_concat_patch_aqe3_rerank": tr_parent,
            "clipsenet_v6_aqe10_rerank_k1_50_k2_10_lambda_0_1": cs_parent,
        }

    concat_fusion = None
    if args.concat_sweep:
        concat_rows = []
        for alpha in (0.3, 0.5, 0.7):
            q_concat = l2_normalize(np.concatenate([(1.0 - alpha) * q_tr, alpha * q_cs], axis=1))
            g_concat = l2_normalize(np.concatenate([(1.0 - alpha) * g_tr, alpha * g_cs], axis=1))
            metrics = evaluate_similarity(q_concat @ g_concat.T, q_pids, g_pids, q_camids, g_camids)
            concat_rows.append({"alpha_clipsenet": float(alpha), **metrics})
        concat_fusion = {"all_rows": concat_rows, "best": max(concat_rows, key=metric_sort_key)}

    output: dict[str, Any] = {
        "experiment": "14t_fusion_verify",
        "wall_time_sec": float(time.time() - started),
        "params": {
            "w_clipsenet": float(args.w_clipsenet),
            "w_transreid": float(1.0 - args.w_clipsenet),
            "transreid_stream": args.transreid_stream,
            "aqe_k": int(args.aqe_k),
            "rerank": {
                "k1": int(args.rerank_k1),
                "k2": int(args.rerank_k2),
                "lambda_value": float(args.rerank_lambda),
            },
        },
        "score_fusion": {
            "cosine": raw_cosine,
            "cosine_aqe": headline["cosine_aqe"],
            "rerank_aqe": headline["rerank_aqe"],
            "best": best,
            "all_rows": all_rows if args.weights_sweep else [],
        },
        "drift_parents": drift_parents,
        "checkpoints": {
            "transreid": {"path": str(args.transreid_checkpoint), "sha256": sha256_file(args.transreid_checkpoint)},
            "clipsenet": {"path": str(args.clipsenet_checkpoint), "sha256": sha256_file(args.clipsenet_checkpoint)},
        },
    }
    if concat_fusion is not None:
        output["concat_fusion"] = concat_fusion

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()