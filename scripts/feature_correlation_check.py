"""Check whether two ReID feature sets are diverse enough to fuse productively."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr


PID_KEYS = (
    "pid",
    "person_id",
    "vehicle_id",
    "identity_id",
    "gt_id",
    "target_id",
)


@dataclass(frozen=True)
class RowMeta:
    index: int
    camera_id: str | None
    track_id: Any | None
    frame_id: int | None
    pid: Any | None
    camid: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-a", type=str, help="Path to embeddings.npy or a stage2 output dir")
    parser.add_argument("--features-b", type=str, help="Path to embeddings.npy or a stage2 output dir")
    parser.add_argument("--tracklets-a", type=str, help="Optional tracklet/index metadata for side A")
    parser.add_argument("--tracklets-b", type=str, help="Optional tracklet/index metadata for side B")
    parser.add_argument("--sample-size", type=int, default=5000, help="Sampling budget for pairwise and rank-1 checks")
    parser.add_argument("--output", type=str, help="Optional JSON output path")
    parser.add_argument("--quick", action="store_true", help="Use 100 anchors and a 1000-sample gallery")
    parser.add_argument("--self-test", action="store_true", help="Run built-in sanity checks")
    return parser.parse_args()


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (features / norms).astype(np.float32, copy=False)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_label(path: Path) -> str:
    if path.is_dir():
        return path.name
    return path.stem


def load_feature_matrix(path_str: str) -> tuple[np.ndarray, str, Path | None, list[RowMeta] | None]:
    path = Path(path_str)
    if path.is_dir():
        feature_path = path / "embeddings.npy"
        index_path = path / "embedding_index.json"
    else:
        feature_path = path
        index_path = path.with_name("embedding_index.json")

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    features = np.load(feature_path)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {features.shape} from {feature_path}")

    index_rows = None
    if index_path.exists():
        index_rows = parse_metadata_path(index_path, expected_rows=features.shape[0])

    return features.astype(np.float32, copy=False), _feature_label(path), index_path if index_path.exists() else None, index_rows


def _extract_pid(payload: dict[str, Any], fallback: Any = None) -> Any:
    for key in PID_KEYS:
        if key in payload and payload[key] is not None:
            return payload[key]
    return fallback


def _extract_camid(payload: dict[str, Any]) -> int | None:
    value = payload.get("camid")
    if value is None:
        value = payload.get("camera_index")
    if value is None:
        return None
    return int(value)


def _coerce_tracklet_items(data: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if "rows" in data and isinstance(data["rows"], list):
            return [item for item in data["rows"] if isinstance(item, dict)]
        items: list[dict[str, Any]] = []
        for camera_id in sorted(data):
            value = data[camera_id]
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                if "camera_id" not in item:
                    item = {**item, "camera_id": camera_id}
                items.append(item)
        if items:
            return items
    raise ValueError(f"Unsupported metadata structure in {source}")


def _rows_from_direct_entries(items: Sequence[dict[str, Any]], expected_rows: int) -> list[RowMeta] | None:
    direct_rows: list[RowMeta] = []
    for idx, item in enumerate(items):
        if "track_id" not in item and "camera_id" not in item:
            return None
        direct_rows.append(
            RowMeta(
                index=idx,
                camera_id=item.get("camera_id"),
                track_id=item.get("track_id"),
                frame_id=item.get("frame_id"),
                pid=_extract_pid(item),
                camid=_extract_camid(item),
            )
        )
    if len(direct_rows) != expected_rows:
        return None
    return direct_rows


def _rows_from_tracklets(items: Sequence[dict[str, Any]], expected_rows: int) -> list[RowMeta] | None:
    if not items or not all("frames" in item for item in items):
        return None

    if len(items) == expected_rows:
        rows: list[RowMeta] = []
        for idx, item in enumerate(items):
            frames = item.get("frames", [])
            frame_id = None
            if frames:
                mid = frames[len(frames) // 2]
                if isinstance(mid, dict):
                    frame_id = mid.get("frame_id")
            rows.append(
                RowMeta(
                    index=idx,
                    camera_id=item.get("camera_id"),
                    track_id=item.get("track_id"),
                    frame_id=frame_id,
                    pid=_extract_pid(item),
                    camid=_extract_camid(item),
                )
            )
        return rows

    total_frames = sum(len(item.get("frames", [])) for item in items)
    if total_frames != expected_rows:
        return None

    rows = []
    row_index = 0
    for item in items:
        parent_pid = _extract_pid(item)
        parent_camid = _extract_camid(item)
        for frame in item.get("frames", []):
            if not isinstance(frame, dict):
                continue
            rows.append(
                RowMeta(
                    index=row_index,
                    camera_id=item.get("camera_id", frame.get("camera_id")),
                    track_id=item.get("track_id", frame.get("track_id")),
                    frame_id=frame.get("frame_id"),
                    pid=_extract_pid(frame, fallback=parent_pid),
                    camid=_extract_camid(frame) if _extract_camid(frame) is not None else parent_camid,
                )
            )
            row_index += 1
    if len(rows) != expected_rows:
        return None
    return rows


def parse_metadata_path(path: Path, expected_rows: int) -> list[RowMeta]:
    if path.is_dir():
        json_paths = sorted(path.glob("tracklets_*.json"))
        if not json_paths and (path / "embedding_index.json").exists():
            return parse_metadata_path(path / "embedding_index.json", expected_rows)
        if not json_paths and len(list(path.glob("*.json"))) == 1:
            json_paths = list(path.glob("*.json"))
        items: list[dict[str, Any]] = []
        for json_path in json_paths:
            items.extend(_coerce_tracklet_items(_json_load(json_path), json_path))
    else:
        items = _coerce_tracklet_items(_json_load(path), path)

    rows = _rows_from_direct_entries(items, expected_rows)
    if rows is not None:
        return rows

    rows = _rows_from_tracklets(items, expected_rows)
    if rows is not None:
        return rows

    raise ValueError(
        f"Could not map {path} onto {expected_rows} feature rows. "
        "Expected either one metadata row per feature row, one tracklet per row, or one frame per row."
    )


def _key_shape(rows: Sequence[RowMeta]) -> str:
    if rows and all(row.camera_id is not None and row.track_id is not None and row.frame_id is not None for row in rows):
        return "frame"
    if rows and all(row.camera_id is not None and row.track_id is not None for row in rows):
        return "track"
    return "row"


def _build_keys(rows: Sequence[RowMeta], shape: str) -> list[tuple[Any, ...]]:
    keys: list[tuple[Any, ...]] = []
    for row in rows:
        if shape == "frame":
            keys.append((row.camera_id, row.track_id, row.frame_id))
        elif shape == "track":
            keys.append((row.camera_id, row.track_id))
        else:
            keys.append(("row", row.index))
    return keys


def align_feature_sets(
    features_a: np.ndarray,
    rows_a: list[RowMeta] | None,
    features_b: np.ndarray,
    rows_b: list[RowMeta] | None,
) -> tuple[np.ndarray, list[RowMeta], np.ndarray, list[RowMeta], str]:
    if rows_a is None and rows_b is None:
        if features_a.shape[0] != features_b.shape[0]:
            raise ValueError(
                "Feature matrices have different row counts and no metadata was provided for alignment: "
                f"{features_a.shape[0]} vs {features_b.shape[0]}"
            )
        default_rows = [RowMeta(i, None, None, None, None, None) for i in range(features_a.shape[0])]
        return features_a, default_rows, features_b, default_rows.copy(), "row-order"

    if rows_a is None or rows_b is None:
        if features_a.shape[0] != features_b.shape[0]:
            raise ValueError(
                "Only one side has metadata and the row counts differ, so row-order alignment is unsafe: "
                f"{features_a.shape[0]} vs {features_b.shape[0]}"
            )
        if rows_a is None:
            rows_a = [RowMeta(i, None, None, None, row.pid, row.camid) for i, row in enumerate(rows_b)]
        if rows_b is None:
            rows_b = [RowMeta(i, None, None, None, row.pid, row.camid) for i, row in enumerate(rows_a)]
        return features_a, rows_a, features_b, rows_b, "row-order"

    shape_a = _key_shape(rows_a)
    shape_b = _key_shape(rows_b)
    if "frame" in (shape_a, shape_b):
        key_shape = "frame" if shape_a == "frame" and shape_b == "frame" else "track"
    elif "track" in (shape_a, shape_b):
        key_shape = "track"
    else:
        key_shape = "row"

    keys_a = _build_keys(rows_a, key_shape)
    keys_b = _build_keys(rows_b, key_shape)
    map_b = {key: idx for idx, key in enumerate(keys_b)}

    keep_a: list[int] = []
    keep_b: list[int] = []
    aligned_rows_a: list[RowMeta] = []
    aligned_rows_b: list[RowMeta] = []
    for idx_a, key in enumerate(keys_a):
        idx_b = map_b.get(key)
        if idx_b is None:
            continue
        keep_a.append(idx_a)
        keep_b.append(idx_b)
        aligned_rows_a.append(rows_a[idx_a])
        aligned_rows_b.append(rows_b[idx_b])

    if not keep_a:
        raise ValueError(f"No overlapping rows found using {key_shape}-level alignment")

    return (
        features_a[np.asarray(keep_a)],
        aligned_rows_a,
        features_b[np.asarray(keep_b)],
        aligned_rows_b,
        key_shape,
    )


def _sample_pairs(ids: np.ndarray, sample_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_rows = ids.shape[0]
    if n_rows < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    keep_i: list[np.ndarray] = []
    keep_j: list[np.ndarray] = []
    collected = 0
    rounds = 0
    target = max(1, sample_size)
    while collected < target and rounds < 12:
        rounds += 1
        draws = min(max(target * 3, 2048), n_rows * max(8, min(n_rows, 64)))
        i = rng.integers(0, n_rows, size=draws, endpoint=False)
        j = rng.integers(0, n_rows, size=draws, endpoint=False)
        mask = (i != j) & (ids[i] != ids[j])
        if not np.any(mask):
            continue
        i = i[mask]
        j = j[mask]
        swap = i > j
        if np.any(swap):
            i_swap = i[swap].copy()
            i[swap] = j[swap]
            j[swap] = i_swap
        pairs = np.stack([i, j], axis=1)
        pairs = np.unique(pairs, axis=0)
        keep_i.append(pairs[:, 0])
        keep_j.append(pairs[:, 1])
        collected += pairs.shape[0]

    if not keep_i:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    i_all = np.concatenate(keep_i)
    j_all = np.concatenate(keep_j)
    if i_all.shape[0] > target:
        take = rng.choice(i_all.shape[0], size=target, replace=False)
        i_all = i_all[take]
        j_all = j_all[take]
    return i_all, j_all


def mean_pairwise_cosine(
    features: np.ndarray,
    rows: Sequence[RowMeta],
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[float | None, bool]:
    ids = [row.pid if row.pid is not None else (row.camera_id, row.track_id, row.frame_id, row.index) for row in rows]
    ids_array = np.asarray(ids, dtype=object)
    pair_i, pair_j = _sample_pairs(ids_array, sample_size, rng)
    if pair_i.size == 0:
        return None, any(row.pid is not None for row in rows)
    values = np.sum(features[pair_i] * features[pair_j], axis=1)
    return float(values.mean()), any(row.pid is not None for row in rows)


def sample_indices(total: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if total <= count:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


def compute_rank_correlation(
    features_a: np.ndarray,
    features_b: np.ndarray,
    anchor_indices: np.ndarray,
    gallery_indices: np.ndarray,
) -> tuple[float | None, int]:
    correlations: list[float] = []
    for anchor in anchor_indices:
        mask = gallery_indices != anchor
        gallery = gallery_indices[mask]
        if gallery.size < 2:
            continue
        sims_a = features_a[gallery] @ features_a[anchor]
        sims_b = features_b[gallery] @ features_b[anchor]
        corr = spearmanr(sims_a, sims_b).correlation
        if corr is None or math.isnan(corr):
            continue
        correlations.append(float(corr))

    if not correlations:
        return None, 0
    return float(np.mean(correlations)), len(correlations)


def compute_rank1_agreement(
    features_a: np.ndarray,
    features_b: np.ndarray,
    query_indices: np.ndarray,
    gallery_indices: np.ndarray,
    block_size: int = 256,
) -> tuple[float | None, int]:
    if gallery_indices.size == 0 or query_indices.size == 0:
        return None, 0

    gallery_positions = {int(index): pos for pos, index in enumerate(gallery_indices.tolist())}
    agreements = 0
    total = 0

    gallery_a = features_a[gallery_indices]
    gallery_b = features_b[gallery_indices]

    for start in range(0, query_indices.size, block_size):
        block = query_indices[start:start + block_size]
        sims_a = features_a[block] @ gallery_a.T
        sims_b = features_b[block] @ gallery_b.T
        for row_idx, query in enumerate(block.tolist()):
            gallery_pos = gallery_positions.get(int(query))
            if gallery_pos is not None:
                sims_a[row_idx, gallery_pos] = -np.inf
                sims_b[row_idx, gallery_pos] = -np.inf
        top_a = gallery_indices[np.argmax(sims_a, axis=1)]
        top_b = gallery_indices[np.argmax(sims_b, axis=1)]
        agreements += int(np.sum(top_a == top_b))
        total += block.shape[0]

    if total == 0:
        return None, 0
    return 100.0 * agreements / total, total


def compute_intrinsic_map(
    features: np.ndarray,
    pids: np.ndarray | None,
    camids: np.ndarray | None,
) -> tuple[float | None, int]:
    if pids is None or camids is None:
        return None, 0

    ap_values: list[float] = []
    valid_queries = 0
    gallery = features
    for query_idx in range(features.shape[0]):
        scores = gallery @ features[query_idx]
        order = np.argsort(-scores)
        remove = (pids[order] == pids[query_idx]) & (camids[order] == camids[query_idx])
        keep = ~remove
        matches = (pids[order] == pids[query_idx]).astype(np.int32)[keep]
        if matches.size == 0 or not np.any(matches):
            continue
        valid_queries += 1
        cumulative = matches.cumsum()
        precision = cumulative / (np.arange(matches.size) + 1.0)
        ap = float((precision * matches).sum() / matches.sum())
        ap_values.append(ap)

    if valid_queries == 0:
        return None, 0
    return float(np.mean(ap_values)), valid_queries


def build_eval_metadata(rows_a: Sequence[RowMeta], rows_b: Sequence[RowMeta]) -> tuple[np.ndarray | None, np.ndarray | None]:
    merged_pids: list[Any | None] = []
    merged_cam_tokens: list[Any | None] = []
    for row_a, row_b in zip(rows_a, rows_b):
        pid = row_a.pid if row_a.pid is not None else row_b.pid
        merged_pids.append(pid)

        if row_a.camid is not None:
            merged_cam_tokens.append(row_a.camid)
        elif row_b.camid is not None:
            merged_cam_tokens.append(row_b.camid)
        elif row_a.camera_id is not None:
            merged_cam_tokens.append(row_a.camera_id)
        else:
            merged_cam_tokens.append(row_b.camera_id)

    if not any(pid is not None for pid in merged_pids):
        return None, None

    pid_labels = {pid: idx for idx, pid in enumerate(sorted({pid for pid in merged_pids if pid is not None}, key=str))}
    pids = np.asarray([pid_labels[pid] for pid in merged_pids], dtype=np.int32)

    cam_tokens = [token for token in merged_cam_tokens if token is not None]
    if not cam_tokens:
        return pids, None
    cam_labels = {token: idx for idx, token in enumerate(sorted(set(cam_tokens), key=str))}
    camids = np.asarray([cam_labels[token] for token in merged_cam_tokens], dtype=np.int32)
    return pids, camids


def ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def recommend_fusion(
    spearman_corr: float | None,
    rank1_agreement_pct: float | None,
    map_a: float | None,
    map_b: float | None,
) -> tuple[str, str]:
    ratio = ratio_or_none(map_b, map_a)

    reasons: list[str] = []
    if spearman_corr is None:
        return "marginal", "Rank correlation could not be computed, so the fusion gate is inconclusive."

    if spearman_corr >= 0.85:
        reasons.append(f"rank correlation {spearman_corr:.3f} >= 0.85")
    if ratio is not None and ratio < 0.65:
        reasons.append(f"secondary/primary mAP ratio {ratio:.3f} < 0.65")
    if reasons:
        return "no-go", "; ".join(reasons)

    if ratio is None:
        if spearman_corr < 0.70 and rank1_agreement_pct is not None and rank1_agreement_pct < 75.0:
            return "marginal", (
                f"diversity looks promising (rank correlation {spearman_corr:.3f}, "
                f"rank-1 agreement {rank1_agreement_pct:.1f}%), but intrinsic mAP ratio is unavailable"
            )
        return "marginal", "intrinsic mAP ratio is unavailable, so a full go/no-go decision is not possible"

    if spearman_corr < 0.70 and rank1_agreement_pct is not None and rank1_agreement_pct < 75.0:
        return "go", (
            f"rank correlation {spearman_corr:.3f} < 0.70, rank-1 agreement {rank1_agreement_pct:.1f}% < 75%, "
            f"and mAP ratio {ratio:.3f} >= 0.65"
        )

    if 0.70 <= spearman_corr < 0.85 and ratio >= 0.65:
        return "marginal", f"rank correlation {spearman_corr:.3f} is in the 0.70-0.85 marginal band with mAP ratio {ratio:.3f}"

    if spearman_corr < 0.70 and rank1_agreement_pct is not None and rank1_agreement_pct >= 75.0:
        return "marginal", (
            f"rank correlation {spearman_corr:.3f} is low, but rank-1 agreement {rank1_agreement_pct:.1f}% is still high"
        )

    return "marginal", f"metrics are mixed: rank correlation {spearman_corr:.3f}, mAP ratio {ratio:.3f}"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    return value


def analyze_feature_sets(
    features_a: np.ndarray,
    rows_a: list[RowMeta],
    features_b: np.ndarray,
    rows_b: list[RowMeta],
    model_a: str,
    model_b: str,
    sample_size: int,
    quick: bool,
    alignment_mode: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    features_a = l2_normalize(features_a)
    features_b = l2_normalize(features_b)

    n_rows = features_a.shape[0]
    anchor_budget = min(100 if quick else 200, n_rows)
    gallery_budget = min(1000, n_rows) if quick else n_rows
    rank1_budget = min(1000 if quick else sample_size, n_rows)

    anchor_indices = sample_indices(n_rows, anchor_budget, rng)
    gallery_indices = sample_indices(n_rows, gallery_budget, rng)
    rank1_queries = sample_indices(n_rows, rank1_budget, rng)

    mean_cos_a, used_true_ids_a = mean_pairwise_cosine(features_a, rows_a, sample_size, rng)
    mean_cos_b, used_true_ids_b = mean_pairwise_cosine(features_b, rows_b, sample_size, rng)

    spearman_corr, evaluated_anchors = compute_rank_correlation(features_a, features_b, anchor_indices, gallery_indices)
    rank1_agreement_pct, evaluated_queries = compute_rank1_agreement(features_a, features_b, rank1_queries, gallery_indices)

    pids, camids = build_eval_metadata(rows_a, rows_b)
    map_a, valid_queries_a = compute_intrinsic_map(features_a, pids, camids)
    map_b, valid_queries_b = compute_intrinsic_map(features_b, pids, camids)

    recommendation, rationale = recommend_fusion(spearman_corr, rank1_agreement_pct, map_a, map_b)
    map_ratio = ratio_or_none(map_b, map_a)

    notes: list[str] = [f"alignment={alignment_mode}"]
    if not used_true_ids_a or not used_true_ids_b:
        notes.append("mean intra-model cosine fell back to distinct row keys because explicit identity labels were unavailable")
    if map_a is None or map_b is None:
        notes.append("intrinsic mAP omitted because identity/camera labels were unavailable or had no valid cross-camera matches")

    result = {
        "model_a": model_a,
        "model_b": model_b,
        "n_samples": int(sample_size),
        "n_aligned_rows": int(n_rows),
        "alignment_mode": alignment_mode,
        "mean_intra_cosine_a": to_jsonable(mean_cos_a),
        "mean_intra_cosine_b": to_jsonable(mean_cos_b),
        "spearman_rank_correlation": to_jsonable(spearman_corr),
        "rank1_agreement_pct": to_jsonable(rank1_agreement_pct),
        "intrinsic_map_a": to_jsonable(map_a),
        "intrinsic_map_b": to_jsonable(map_b),
        "map_b_over_a": to_jsonable(map_ratio),
        "fusion_recommendation": recommendation,
        "rationale": rationale,
        "evaluated_anchors": int(evaluated_anchors),
        "evaluated_rank1_queries": int(evaluated_queries),
        "evaluated_map_queries_a": int(valid_queries_a),
        "evaluated_map_queries_b": int(valid_queries_b),
        "notes": notes,
    }
    return result


def print_summary(result: dict[str, Any]) -> None:
    print("=== Feature Correlation Diagnostic ===")
    print(f"Model A: {result['model_a']}")
    print(f"Model B: {result['model_b']}")
    print(f"Aligned rows: {result['n_aligned_rows']} ({result['alignment_mode']})")
    print(f"Mean intra cosine A: {format_metric(result['mean_intra_cosine_a'])}")
    print(f"Mean intra cosine B: {format_metric(result['mean_intra_cosine_b'])}")
    print(
        f"Spearman rank correlation: {format_metric(result['spearman_rank_correlation'])} "
        f"across {result['evaluated_anchors']} anchors"
    )
    print(
        f"Rank-1 agreement: {format_metric(result['rank1_agreement_pct'], pct=True)} "
        f"across {result['evaluated_rank1_queries']} queries"
    )
    print(
        f"Intrinsic mAP A: {format_metric(result['intrinsic_map_a'])} "
        f"({result['evaluated_map_queries_a']} valid queries)"
    )
    print(
        f"Intrinsic mAP B: {format_metric(result['intrinsic_map_b'])} "
        f"({result['evaluated_map_queries_b']} valid queries)"
    )
    if result.get("map_b_over_a") is not None:
        print(f"mAP ratio B/A: {result['map_b_over_a']:.3f}")
    print(f"Recommendation: {result['fusion_recommendation']}")
    print(f"Rationale: {result['rationale']}")
    notes = result.get("notes") or []
    for note in notes:
        print(f"Note: {note}")


def format_metric(value: Any, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{float(value):.1f}%"
    return f"{float(value):.4f}"


def maybe_write_output(result: dict[str, Any], output_path: str | None) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote JSON report to {path}")


def load_side(
    feature_path: str,
    metadata_path: str | None,
) -> tuple[np.ndarray, list[RowMeta] | None, str]:
    features, label, _, default_rows = load_feature_matrix(feature_path)
    rows = default_rows
    if metadata_path:
        rows = parse_metadata_path(Path(metadata_path), expected_rows=features.shape[0])
    return features, rows, label


def make_synthetic_dataset(
    rng: np.random.Generator,
    n_ids: int = 36,
    n_cams: int = 3,
    shots_per_cam: int = 3,
    dim: int = 96,
    noise_scale: float = 0.16,
) -> tuple[np.ndarray, list[RowMeta], np.ndarray]:
    centers = l2_normalize(rng.normal(size=(n_ids, dim)).astype(np.float32))
    cam_bias = rng.normal(scale=0.08, size=(n_cams, dim)).astype(np.float32)

    features = []
    rows = []
    for pid in range(n_ids):
        for camid in range(n_cams):
            for shot in range(shots_per_cam):
                feature = centers[pid] + cam_bias[camid] + rng.normal(scale=noise_scale, size=dim)
                features.append(feature.astype(np.float32))
                rows.append(
                    RowMeta(
                        index=len(rows),
                        camera_id=f"C{camid + 1}",
                        track_id=pid,
                        frame_id=shot,
                        pid=pid,
                        camid=camid,
                    )
                )
    return l2_normalize(np.asarray(features, dtype=np.float32)), rows, centers


def run_self_test() -> int:
    rng = np.random.default_rng(7)
    base_a, rows, centers = make_synthetic_dataset(rng)

    correlated_b = l2_normalize(base_a + rng.normal(scale=0.015, size=base_a.shape).astype(np.float32))
    case1 = analyze_feature_sets(
        base_a,
        rows,
        correlated_b,
        rows,
        model_a="self_test_correlated_a",
        model_b="self_test_correlated_b",
        sample_size=1500,
        quick=False,
        alignment_mode="frame",
    )
    pass_case1 = case1["fusion_recommendation"] == "no-go"
    print(
        f"[{'PASS' if pass_case1 else 'FAIL'}] correlated clone -> {case1['fusion_recommendation']} "
        f"(spearman={format_metric(case1['spearman_rank_correlation'])}, "
        f"rank1={format_metric(case1['rank1_agreement_pct'], pct=True)})"
    )

    independent_centers = l2_normalize(rng.normal(size=centers.shape).astype(np.float32))
    cam_bias = rng.normal(scale=0.05, size=(3, independent_centers.shape[1])).astype(np.float32)
    diverse_features = []
    for row in rows:
        diverse_features.append(
            independent_centers[int(row.pid)]
            + cam_bias[int(row.camid)]
            + rng.normal(scale=0.10, size=independent_centers.shape[1])
        )
    diverse_b = l2_normalize(np.asarray(diverse_features, dtype=np.float32))
    case2 = analyze_feature_sets(
        base_a,
        rows,
        diverse_b,
        rows,
        model_a="self_test_diverse_a",
        model_b="self_test_diverse_b",
        sample_size=1500,
        quick=False,
        alignment_mode="frame",
    )
    pass_case2 = case2["fusion_recommendation"] == "go"
    print(
        f"[{'PASS' if pass_case2 else 'FAIL'}] diverse independent view -> {case2['fusion_recommendation']} "
        f"(spearman={format_metric(case2['spearman_rank_correlation'])}, "
        f"rank1={format_metric(case2['rank1_agreement_pct'], pct=True)}, "
        f"ratio={format_metric(case2['map_b_over_a'])})"
    )

    return 0 if pass_case1 and pass_case2 else 1


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    if not args.features_a or not args.features_b:
        raise SystemExit("--features-a and --features-b are required unless --self-test is used")

    features_a, rows_a, model_a = load_side(args.features_a, args.tracklets_a)
    features_b, rows_b, model_b = load_side(args.features_b, args.tracklets_b)
    features_a, rows_a, features_b, rows_b, alignment_mode = align_feature_sets(features_a, rows_a, features_b, rows_b)
    result = analyze_feature_sets(
        features_a,
        rows_a,
        features_b,
        rows_b,
        model_a=model_a,
        model_b=model_b,
        sample_size=args.sample_size,
        quick=args.quick,
        alignment_mode=alignment_mode,
    )
    print_summary(result)
    maybe_write_output(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())