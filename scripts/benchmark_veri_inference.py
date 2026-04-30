from __future__ import annotations

# EXEC_LOCALLY_OK:
# This script is a narrow micro-benchmark, not a pipeline stage. It runs batch-1
# forward passes plus synthetic AQE and reranking timing and does not touch the
# GPU-heavy stages forbidden by the local-execution policy.

import argparse
import gc
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch

from src.training.evaluate_reid import compute_reranking

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "perf_bench" / "veri_perf_bench.json"
CHECKPOINT_NAME = "vehicle_transreid_vit_base_veri776.pth"
MODEL_NAME = "vit_base_patch16_clip_224.openai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VeRi-776 inference and eval-time post-processing.")
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_checkpoint() -> Path | None:
    candidates = [
        ROOT / "data" / "weights" / CHECKPOINT_NAME,
        ROOT / "outputs" / "09v_veri_v9" / CHECKPOINT_NAME,
        ROOT / "_scratch_old08" / "k08out" / "exported_models" / CHECKPOINT_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(ROOT.rglob(CHECKPOINT_NAME))
    return matches[0] if matches else None


def current_git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def total_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    return round(psutil.virtual_memory().total / (1024 ** 3), 2)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model() -> torch.nn.Module:
    return timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)


def maybe_load_checkpoint(model: torch.nn.Module, checkpoint_path: Path | None) -> tuple[bool, bool, str | None]:
    if checkpoint_path is None:
        print("WARNING: checkpoint missing; results are architecture-only timing")
        return False, True, None
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"WARNING: failed to load checkpoint from {checkpoint_path}: {exc}")
        return False, True, str(checkpoint_path)
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        print(f"WARNING: checkpoint payload in {checkpoint_path} is not a state dict; using random init")
        return False, True, str(checkpoint_path)
    cleaned: dict[str, Any] = {}
    model_keys = model.state_dict()
    matched = 0
    for key, value in payload.items():
        clean_key = key.replace("module.", "")
        if clean_key in model_keys and getattr(value, "shape", None) == model_keys[clean_key].shape:
            cleaned[clean_key] = value
            matched += 1
    if matched == 0:
        print(f"WARNING: checkpoint found at {checkpoint_path} but no compatible keys matched; using random init")
        return True, True, str(checkpoint_path)
    model.load_state_dict(cleaned, strict=False)
    return True, False, str(checkpoint_path)


def summarize_timings(values_ms: list[float]) -> dict[str, float]:
    array = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def unwrap_output(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


@torch.inference_mode()
def measure_forward(model: torch.nn.Module, device: torch.device, batch_size: int, img_size: int, warmup: int, n_iters: int, *, use_fp16: bool) -> tuple[dict[str, float], float | None]:
    dtype = torch.float16 if use_fp16 else torch.float32
    model = model.eval()
    input_tensor = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)
    model = model.half() if use_fp16 else model.float()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        output = unwrap_output(model(input_tensor))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _ = output.shape
    timings_ms: list[float] = []
    for _ in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = unwrap_output(model(input_tensor))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
        _ = output.shape
    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    return summarize_timings(timings_ms), peak_vram_mb


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-12
    return array / norms


def _topk_indices(features: np.ndarray, k: int, device: torch.device, chunk_size: int = 512) -> np.ndarray:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tensor = torch.from_numpy(features).to(device=device, dtype=dtype)
    indices = np.empty((features.shape[0], k), dtype=np.int32)
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        scores = tensor[start:end] @ tensor.T
        _, chunk_indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        indices[start:end] = chunk_indices.cpu().numpy().astype(np.int32)
    return indices


def average_query_expansion_from_topk(features: np.ndarray, topk_idx: np.ndarray) -> np.ndarray:
    expanded = np.zeros_like(features)
    for index in range(features.shape[0]):
        expanded[index] = features[topk_idx[index]].mean(axis=0)
    return normalize_rows(expanded)


def rerank_timing_proxy(query: np.ndarray, gallery: np.ndarray, k1: int, k2: int, lambda_value: float, topk_cache: np.ndarray) -> dict[str, int | float]:
    all_features = np.concatenate([query, gallery], axis=0).astype(np.float32, copy=False)
    num_all = all_features.shape[0]
    half_k = int(np.round(k1 / 2)) + 1
    topk_k1 = topk_cache[:, : k1 + 1]
    topk_half = topk_k1[:, :half_k]
    topk_k2 = topk_k1[:, : k2 + 1] if k2 > 0 else topk_k1[:, :1]
    topk_k1_sets = [set(row.tolist()) for row in topk_k1]
    topk_half_sets = [set(row.tolist()) for row in topk_half]

    supports: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for index in range(num_all):
        forward = topk_k1[index]
        reciprocal: list[int] = []
        for candidate in forward:
            if index in topk_k1_sets[int(candidate)]:
                reciprocal.append(int(candidate))
        expanded = set(reciprocal)
        for candidate in reciprocal:
            candidate_forward = topk_half[candidate]
            candidate_reciprocal = [int(other) for other in candidate_forward if candidate in topk_half_sets[int(other)]]
            if len(candidate_reciprocal) > (2.0 / 3.0) * len(candidate_forward):
                expanded.update(candidate_reciprocal)
        expanded_indices = np.fromiter(expanded, dtype=np.int32)
        sims = all_features[index] @ all_features[expanded_indices].T
        dist = np.clip(2.0 - (2.0 * sims), 0.0, None)
        weight = np.exp(-dist)
        weight = weight / (weight.sum() + 1e-12)
        supports.append(expanded_indices)
        weights.append(weight.astype(np.float32, copy=False))

    if k2 > 0:
        qe_supports: list[np.ndarray] = []
        qe_weights: list[np.ndarray] = []
        for index in range(num_all):
            accumulator: dict[int, float] = {}
            for neighbor in topk_k2[index]:
                neighbor_support = supports[int(neighbor)]
                neighbor_weight = weights[int(neighbor)]
                for support_index, support_weight in zip(neighbor_support, neighbor_weight, strict=False):
                    accumulator[int(support_index)] = accumulator.get(int(support_index), 0.0) + float(support_weight)
            merged_indices = np.fromiter(accumulator.keys(), dtype=np.int32)
            merged_weights = np.fromiter(accumulator.values(), dtype=np.float32) / float(len(topk_k2[index]))
            qe_supports.append(merged_indices)
            qe_weights.append(merged_weights)
        supports = qe_supports
        weights = qe_weights
    support_sizes = np.asarray([len(row_support) for row_support in supports], dtype=np.int32)
    weight_mass = float(sum(float(row_weight.sum()) for row_weight in weights))
    original_qg = np.clip(2.0 - (2.0 * (query @ gallery.T)), 0.0, None)
    return {
        "rows": int(num_all),
        "avg_support": float(support_sizes.mean()),
        "max_support": int(support_sizes.max()),
        "weight_mass": weight_mass,
        "mean_original_qg": float(original_qg.mean() * lambda_value),
    }


def time_call(fn, *args, **kwargs) -> tuple[float, Any]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, result


def synthetic_features() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    query = rng.standard_normal((1678, 768), dtype=np.float32)
    gallery = rng.standard_normal((11579, 768), dtype=np.float32)
    return normalize_rows(query), normalize_rows(gallery)


def collect_hardware(device: torch.device) -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "device": device.type,
        "device_name": platform.processor() or "unknown CPU",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": total_ram_gb(),
    }
    if device.type == "cuda":
        hardware["device_name"] = torch.cuda.get_device_name(0)
    return hardware


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    requested_device = choose_device(args.device)
    checkpoint_path = resolve_checkpoint()
    model = build_model()
    checkpoint_found, architecture_only, checkpoint_string = maybe_load_checkpoint(model, checkpoint_path)
    device = requested_device
    notes: list[str] = []
    try:
        model = model.to(device)
        forward_fp32, vram_fp32 = measure_forward(model, device, args.batch_size, args.img_size, args.warmup, args.n_iters, use_fp16=False)
    except RuntimeError as exc:
        message = str(exc).lower()
        if device.type == "cuda" and "out of memory" in message:
            notes.append("CUDA OOM during forward timing; fell back to CPU.")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            model = build_model()
            _, architecture_only_recheck, checkpoint_string = maybe_load_checkpoint(model, checkpoint_path)
            architecture_only = architecture_only or architecture_only_recheck
            model = model.to(device)
            forward_fp32, vram_fp32 = measure_forward(model, device, args.batch_size, args.img_size, args.warmup, args.n_iters, use_fp16=False)
        else:
            raise
    forward_fp16: dict[str, float] | None = None
    vram_fp16: float | None = None
    if device.type == "cuda" and args.fp16:
        model_fp16 = build_model()
        _, architecture_only_recheck, _ = maybe_load_checkpoint(model_fp16, checkpoint_path)
        architecture_only = architecture_only or architecture_only_recheck
        model_fp16 = model_fp16.to(device)
        forward_fp16, vram_fp16 = measure_forward(model_fp16, device, args.batch_size, args.img_size, args.warmup, args.n_iters, use_fp16=True)
    query, gallery = synthetic_features()
    all_features = np.concatenate([query, gallery], axis=0)
    neighbor_cache_ms, topk_cache = time_call(_topk_indices, all_features, 81, device)
    aqe_pool_ms, expanded = time_call(average_query_expansion_from_topk, all_features, topk_cache[:, :3])
    aqe_ms = neighbor_cache_ms + aqe_pool_ms
    del expanded
    gc.collect()
    rerank_30_ms, _ = time_call(rerank_timing_proxy, query, gallery, 30, 10, 0.2, topk_cache)
    gc.collect()
    rerank_80_ms, _ = time_call(rerank_timing_proxy, query, gallery, 80, 15, 0.2, topk_cache)
    gc.collect()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    params_m = round(sum(parameter.numel() for parameter in model.parameters()) / 1_000_000.0, 2)
    return {
        "version": "1.0",
        "timestamp_utc": timestamp,
        "git_sha": current_git_sha(),
        "hardware": collect_hardware(device),
        "checkpoint": {"path": checkpoint_string, "found": checkpoint_found, "architecture_only": architecture_only},
        "model": {"name": MODEL_NAME, "params_m": params_m, "flops_g_estimated": 17.6, "img_size": args.img_size},
        "config": {"n_iters": args.n_iters, "warmup": args.warmup, "batch_size": args.batch_size, "fp16_requested": args.fp16},
        "forward_fp32_ms": forward_fp32,
        "forward_fp16_ms": forward_fp16,
        "vram_peak_mb_fp32": vram_fp32,
        "vram_peak_mb_fp16": vram_fp16,
        "pipeline_breakdown_ms": {
            "forward_fp32": forward_fp32["mean"],
            "flip_tta_overhead": forward_fp32["mean"],
            "aqe_k3": aqe_ms,
            "rerank_k1_30_k2_10_lambda_0p2": rerank_30_ms,
            "rerank_k1_80_k2_15_lambda_0p2": rerank_80_ms,
        },
        "synthetic_dims": {"query": 1678, "gallery": 11579, "feat_dim": 768},
        "notes": [
            *notes,
            "AQE and rerank timings reuse one exact nearest-neighbor cache over the full synthetic feature matrix; the cache build time is charged once to AQE.",
            "FP16 timing only runs when --fp16 is supplied.",
            "The rerank timings are full-dimension support-build proxies rather than the final dense jaccard pass, because the exact query-gallery expansion exceeded practical local wall-clock limits on this machine.",
        ],
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = benchmark(args)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote benchmark JSON to {args.output}")


if __name__ == "__main__":
    main()