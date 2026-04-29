"""Split mAP diagnostic for intra-camera vs cross-camera ReID behavior."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


PID_KEYS = (
    "pid",
    "person_id",
    "vehicle_id",
    "identity_id",
    "gt_id",
    "target_id",
    "global_id",
)

PATH_KEYS = (
    "image_path",
    "img_path",
    "path",
    "file_path",
    "filepath",
    "filename",
    "file_name",
)

FILENAME_PID_RE = re.compile(r"^(?P<pid>\d+)_")
FILENAME_CAM_RE = re.compile(r"_(?P<scene>S\d+)_(?P<camera>c\d+)_f\d+", re.IGNORECASE)


@dataclass(frozen=True)
class RowMeta:
    index: int
    camera_id: str | None
    track_id: Any | None
    pid: Any | None


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    rows: list[RowMeta]
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=str, help="Path to query embeddings.npy or stage2 dir")
    parser.add_argument("--metadata", type=str, help="Path to query embedding_index.json")
    parser.add_argument("--gallery-features", type=str, help="Path to gallery embeddings.npy or stage2 dir")
    parser.add_argument("--gallery-metadata", type=str, help="Path to gallery embedding_index.json")
    parser.add_argument("--output", type=str, help="Optional JSON output path")
    parser.add_argument("--tag", type=str, default="untagged", help="Label stored in the output JSON")
    parser.add_argument("--sample-size", type=int, default=5000, help="Pair sampling budget per class")
    parser.add_argument("--batch-size", type=int, default=256, help="Query batch size for mAP computation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for pair sampling")
    parser.add_argument("--self-test", action="store_true", help="Run the synthetic sanity check")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_JSON", "CANDIDATE_JSON"),
        help="Compare two saved diagnostic JSON files",
    )
    args = parser.parse_args()

    if args.self_test or args.compare:
        return args

    if not args.features:
        parser.error("--features is required unless --self-test or --compare is used")

    if bool(args.gallery_features) != bool(args.gallery_metadata):
        parser.error("--gallery-features and --gallery-metadata must be provided together")

    return args


def l2_normalize(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {features.shape}")
    features = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (features / norms).astype(np.float32, copy=False)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _value_from_path(payload: dict[str, Any]) -> tuple[Any | None, str | None]:
    for key in PATH_KEYS:
        raw_value = payload.get(key)
        if raw_value:
            path = Path(str(raw_value))
            name = path.name
            pid_match = FILENAME_PID_RE.match(name)
            cam_match = FILENAME_CAM_RE.search(name)
            pid = int(pid_match.group("pid")) if pid_match else None
            camera_id = None
            if cam_match:
                camera_id = f"{cam_match.group('scene')}_{cam_match.group('camera')}"
            return pid, camera_id
    return None, None


def _extract_pid(payload: dict[str, Any]) -> Any:
    for key in PID_KEYS:
        value = payload.get(key)
        if value is not None:
            return _coerce_scalar(value)
    pid_from_path, _ = _value_from_path(payload)
    return pid_from_path


def _extract_camera_id(payload: dict[str, Any], default_camera: str | None = None) -> str | None:
    value = payload.get("camera_id")
    if value is not None:
        return str(value)
    _, path_camera = _value_from_path(payload)
    if path_camera is not None:
        return path_camera
    return default_camera


def _coerce_items(data: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
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


def parse_metadata_path(path: Path, expected_rows: int) -> list[RowMeta]:
    if path.is_dir():
        json_paths = sorted(path.glob("tracklets_*.json"))
        if not json_paths and (path / "embedding_index.json").exists():
            return parse_metadata_path(path / "embedding_index.json", expected_rows)
        if not json_paths:
            json_paths = sorted(path.glob("*.json"))
        items: list[dict[str, Any]] = []
        for json_path in json_paths:
            items.extend(_coerce_items(_json_load(json_path), json_path))
    else:
        items = _coerce_items(_json_load(path), path)

    rows: list[RowMeta] = []
    if len(items) != expected_rows:
        raise ValueError(
            f"Metadata row count mismatch for {path}: expected {expected_rows}, found {len(items)}"
        )

    for index, item in enumerate(items):
        camera_id = _extract_camera_id(item)
        rows.append(
            RowMeta(
                index=index,
                camera_id=camera_id,
                track_id=_coerce_scalar(item.get("track_id")),
                pid=_extract_pid(item),
            )
        )

    return rows


def _resolve_feature_paths(features_arg: str, metadata_arg: str | None) -> tuple[Path, Path]:
    feature_path = Path(features_arg)
    if feature_path.is_dir():
        embeddings_path = feature_path / "embeddings.npy"
        metadata_path = Path(metadata_arg) if metadata_arg else feature_path / "embedding_index.json"
    else:
        embeddings_path = feature_path
        metadata_path = Path(metadata_arg) if metadata_arg else feature_path.with_name("embedding_index.json")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Feature file not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return embeddings_path, metadata_path


def load_dataset(features_arg: str, metadata_arg: str | None, label: str | None = None) -> Dataset:
    embeddings_path, metadata_path = _resolve_feature_paths(features_arg, metadata_arg)
    features = l2_normalize(np.load(embeddings_path))
    rows = parse_metadata_path(metadata_path, expected_rows=features.shape[0])
    dataset_label = label or embeddings_path.stem
    _validate_rows(rows, dataset_label)
    return Dataset(features=features, rows=rows, label=dataset_label)


def _validate_rows(rows: Sequence[RowMeta], label: str) -> None:
    missing_camera = [row.index for row in rows if row.camera_id is None]
    if missing_camera:
        raise ValueError(
            f"{label} metadata is missing camera IDs for {len(missing_camera)} rows. "
            "Expected a 'camera_id' field or a CityFlowV2-style filename."
        )

    missing_pid = [row.index for row in rows if row.pid is None]
    if missing_pid:
        raise ValueError(
            f"{label} metadata is missing identity labels for {len(missing_pid)} rows. "
            "Expected one of: " + ", ".join(PID_KEYS) + ", or a CityFlowV2-style filename."
        )


def average_precision_from_scores(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> float | None:
    if not np.any(positive_mask & candidate_mask):
        return None

    masked_scores = scores[candidate_mask]
    masked_positives = positive_mask[candidate_mask]
    ranking = np.argsort(-masked_scores, kind="mergesort")
    hits = masked_positives[ranking]
    hit_count = int(hits.sum())
    if hit_count == 0:
        return None

    precision = np.cumsum(hits, dtype=np.float64) / np.arange(1, hits.size + 1, dtype=np.float64)
    return float(precision[hits].mean())


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _to_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _margin(pos_mean: float | None, neg_mean: float | None) -> float | None:
    if pos_mean is None or neg_mean is None:
        return None
    return float(pos_mean - neg_mean)


def _build_lookup_by_key(values: Sequence[Any]) -> dict[Any, np.ndarray]:
    buckets: dict[Any, list[int]] = {}
    for index, value in enumerate(values):
        buckets.setdefault(value, []).append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in buckets.items()}


def _build_pid_camera_lookup(rows: Sequence[RowMeta]) -> dict[tuple[Any, str], np.ndarray]:
    buckets: dict[tuple[Any, str], list[int]] = {}
    for row in rows:
        if row.pid is None or row.camera_id is None:
            continue
        buckets.setdefault((row.pid, row.camera_id), []).append(row.index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in buckets.items()}


def _concat_index_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty(0, dtype=np.int64)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays)


def compute_maps(
    query_dataset: Dataset,
    gallery_dataset: Dataset,
    same_source: bool,
    batch_size: int,
) -> tuple[dict[str, float | None], dict[str, int]]:
    q_rows = query_dataset.rows
    g_rows = gallery_dataset.rows
    gallery_pids = np.asarray([row.pid for row in g_rows], dtype=object)
    gallery_cams = np.asarray([row.camera_id for row in g_rows], dtype=object)

    overall_scores: list[float] = []
    intra_scores: list[float] = []
    cross_scores: list[float] = []

    valid_counts = {
        "overall": 0,
        "intra": 0,
        "cross": 0,
    }

    for start in range(0, query_dataset.features.shape[0], batch_size):
        stop = min(start + batch_size, query_dataset.features.shape[0])
        batch_scores = query_dataset.features[start:stop] @ gallery_dataset.features.T

        for row_offset, query_index in enumerate(range(start, stop)):
            query_row = q_rows[query_index]
            score_row = batch_scores[row_offset]
            positive_mask = gallery_pids == query_row.pid
            intra_mask = gallery_cams == query_row.camera_id
            cross_mask = gallery_cams != query_row.camera_id
            candidate_mask = np.ones(gallery_dataset.features.shape[0], dtype=bool)
            if same_source:
                candidate_mask[query_index] = False

            ap_overall = average_precision_from_scores(score_row, positive_mask, candidate_mask)
            if ap_overall is not None:
                valid_counts["overall"] += 1
                overall_scores.append(ap_overall)

            ap_intra = average_precision_from_scores(score_row, positive_mask, candidate_mask & intra_mask)
            if ap_intra is not None:
                valid_counts["intra"] += 1
                intra_scores.append(ap_intra)

            ap_cross = average_precision_from_scores(score_row, positive_mask, candidate_mask & cross_mask)
            if ap_cross is not None:
                valid_counts["cross"] += 1
                cross_scores.append(ap_cross)

    return (
        {
            "map_overall": _safe_mean(overall_scores),
            "map_intra_camera": _safe_mean(intra_scores),
            "map_cross_camera": _safe_mean(cross_scores),
        },
        valid_counts,
    )


def _sample_queries_from_counts(
    counts: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total = int(counts.sum())
    if total <= 0:
        return np.empty(0, dtype=np.int64)
    picks = rng.integers(0, total, size=min(sample_size, total), endpoint=False)
    cumulative = np.cumsum(counts)
    return np.searchsorted(cumulative, picks, side="right")


def _build_positive_candidates(
    query_rows: Sequence[RowMeta],
    gallery_rows: Sequence[RowMeta],
    same_camera: bool,
    same_source: bool,
) -> tuple[list[np.ndarray], np.ndarray]:
    pid_camera_lookup = _build_pid_camera_lookup(gallery_rows)
    pid_to_cameras: dict[Any, list[str]] = {}
    for pid, camera_id in pid_camera_lookup:
        pid_to_cameras.setdefault(pid, []).append(camera_id)

    candidates: list[np.ndarray] = []
    counts = np.zeros(len(query_rows), dtype=np.int64)
    for query_row in query_rows:
        if same_camera:
            indices = pid_camera_lookup.get((query_row.pid, query_row.camera_id), np.empty(0, dtype=np.int64))
        else:
            arrays = [
                pid_camera_lookup[(query_row.pid, camera_id)]
                for camera_id in pid_to_cameras.get(query_row.pid, [])
                if camera_id != query_row.camera_id
            ]
            indices = _concat_index_arrays(arrays)

        if same_source and indices.size:
            indices = indices[indices != query_row.index]

        candidates.append(indices)
        counts[query_row.index] = indices.size

    return candidates, counts


def _build_camera_lookup(rows: Sequence[RowMeta]) -> dict[str, np.ndarray]:
    return _build_lookup_by_key([row.camera_id for row in rows])


def _negative_pair_count(
    query_row: RowMeta,
    camera_candidates: np.ndarray,
    positive_candidates: np.ndarray,
    same_source: bool,
) -> int:
    total = int(camera_candidates.size - positive_candidates.size)
    if same_source and query_row.camera_id == query_row.camera_id:
        total -= int(np.any(camera_candidates == query_row.index))
    return max(total, 0)


def _sample_negative_gallery_index(
    query_row: RowMeta,
    candidate_indices: np.ndarray,
    gallery_rows: Sequence[RowMeta],
    same_source: bool,
    rng: np.random.Generator,
) -> int | None:
    if candidate_indices.size == 0:
        return None

    max_tries = max(10, candidate_indices.size * 2)
    for _ in range(max_tries):
        gallery_index = int(candidate_indices[rng.integers(0, candidate_indices.size)])
        gallery_row = gallery_rows[gallery_index]
        if same_source and gallery_index == query_row.index:
            continue
        if gallery_row.pid == query_row.pid:
            continue
        return gallery_index
    return None


def sample_pair_similarities(
    query_dataset: Dataset,
    gallery_dataset: Dataset,
    same_source: bool,
    same_camera: bool,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    query_rows = query_dataset.rows
    gallery_rows = gallery_dataset.rows
    positive_candidates, positive_counts = _build_positive_candidates(
        query_rows=query_rows,
        gallery_rows=gallery_rows,
        same_camera=same_camera,
        same_source=same_source,
    )

    camera_lookup = _build_camera_lookup(gallery_rows)
    all_gallery_indices = np.arange(gallery_dataset.features.shape[0], dtype=np.int64)

    negative_counts = np.zeros(len(query_rows), dtype=np.int64)
    camera_candidate_cache: dict[str, np.ndarray] = {}
    for query_row in query_rows:
        if same_camera:
            camera_candidates = camera_lookup.get(query_row.camera_id, np.empty(0, dtype=np.int64))
        else:
            camera_candidates = camera_candidate_cache.get(query_row.camera_id)
            if camera_candidates is None:
                same_cam_indices = camera_lookup.get(query_row.camera_id, np.empty(0, dtype=np.int64))
                mask = np.ones(all_gallery_indices.size, dtype=bool)
                mask[same_cam_indices] = False
                camera_candidates = all_gallery_indices[mask]
                camera_candidate_cache[query_row.camera_id] = camera_candidates
        negative_counts[query_row.index] = _negative_pair_count(
            query_row=query_row,
            camera_candidates=camera_candidates,
            positive_candidates=positive_candidates[query_row.index],
            same_source=same_source,
        )

    positive_query_indices = _sample_queries_from_counts(positive_counts, sample_size, rng)
    negative_query_indices = _sample_queries_from_counts(negative_counts, sample_size, rng)

    positive_sims: list[float] = []
    for query_index in positive_query_indices:
        candidate_indices = positive_candidates[int(query_index)]
        if candidate_indices.size == 0:
            continue
        gallery_index = int(candidate_indices[rng.integers(0, candidate_indices.size)])
        similarity = float(np.dot(query_dataset.features[int(query_index)], gallery_dataset.features[gallery_index]))
        positive_sims.append(similarity)

    negative_sims: list[float] = []
    for query_index in negative_query_indices:
        query_row = query_rows[int(query_index)]
        if same_camera:
            candidate_indices = camera_lookup.get(query_row.camera_id, np.empty(0, dtype=np.int64))
        else:
            candidate_indices = camera_candidate_cache.get(query_row.camera_id)
            if candidate_indices is None:
                same_cam_indices = camera_lookup.get(query_row.camera_id, np.empty(0, dtype=np.int64))
                mask = np.ones(all_gallery_indices.size, dtype=bool)
                mask[same_cam_indices] = False
                candidate_indices = all_gallery_indices[mask]
                camera_candidate_cache[query_row.camera_id] = candidate_indices

        gallery_index = _sample_negative_gallery_index(
            query_row=query_row,
            candidate_indices=candidate_indices,
            gallery_rows=gallery_rows,
            same_source=same_source,
            rng=rng,
        )
        if gallery_index is None:
            continue
        similarity = float(np.dot(query_dataset.features[int(query_index)], gallery_dataset.features[gallery_index]))
        negative_sims.append(similarity)

    return positive_sims, negative_sims


def summarize_pair_statistics(positive_sims: list[float], negative_sims: list[float]) -> dict[str, float | None]:
    pos_mean = _safe_mean(positive_sims)
    neg_mean = _safe_mean(negative_sims)
    auc = None
    if positive_sims and negative_sims:
        labels = np.concatenate(
            [np.ones(len(positive_sims), dtype=np.int64), np.zeros(len(negative_sims), dtype=np.int64)]
        )
        scores = np.asarray(positive_sims + negative_sims, dtype=np.float64)
        auc = float(roc_auc_score(labels, scores))
    return {
        "auc": _to_float_or_none(auc),
        "pos_mean": _to_float_or_none(pos_mean),
        "neg_mean": _to_float_or_none(neg_mean),
        "margin": _to_float_or_none(_margin(pos_mean, neg_mean)),
    }


def run_diagnostic(
    query_dataset: Dataset,
    gallery_dataset: Dataset,
    tag: str,
    sample_size: int,
    batch_size: int,
    seed: int,
    same_source: bool,
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    maps, valid_map_counts = compute_maps(
        query_dataset=query_dataset,
        gallery_dataset=gallery_dataset,
        same_source=same_source,
        batch_size=batch_size,
    )

    rng = np.random.default_rng(seed)
    cross_positive, cross_negative = sample_pair_similarities(
        query_dataset=query_dataset,
        gallery_dataset=gallery_dataset,
        same_source=same_source,
        same_camera=False,
        sample_size=sample_size,
        rng=rng,
    )
    intra_positive, intra_negative = sample_pair_similarities(
        query_dataset=query_dataset,
        gallery_dataset=gallery_dataset,
        same_source=same_source,
        same_camera=True,
        sample_size=sample_size,
        rng=rng,
    )

    cross_stats = summarize_pair_statistics(cross_positive, cross_negative)
    intra_stats = summarize_pair_statistics(intra_positive, intra_negative)

    result = {
        "tag": tag,
        "n_queries": int(query_dataset.features.shape[0]),
        "map_overall": _to_float_or_none(maps["map_overall"]),
        "map_intra_camera": _to_float_or_none(maps["map_intra_camera"]),
        "map_cross_camera": _to_float_or_none(maps["map_cross_camera"]),
        "cross_camera_auc": cross_stats["auc"],
        "cross_camera_pos_mean_sim": cross_stats["pos_mean"],
        "cross_camera_neg_mean_sim": cross_stats["neg_mean"],
        "cross_camera_margin": cross_stats["margin"],
        "intra_camera_pos_mean_sim": intra_stats["pos_mean"],
        "intra_camera_neg_mean_sim": intra_stats["neg_mean"],
        "intra_camera_margin": intra_stats["margin"],
    }

    pair_counts = {
        "cross_positive": len(cross_positive),
        "cross_negative": len(cross_negative),
        "intra_positive": len(intra_positive),
        "intra_negative": len(intra_negative),
    }
    return result, valid_map_counts, pair_counts


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def print_summary(result: dict[str, Any], valid_counts: dict[str, int], pair_counts: dict[str, int]) -> None:
    print(f"Tag: {result['tag']}")
    print(f"Queries: {result['n_queries']}")
    print(
        "Valid mAP queries: "
        f"overall={valid_counts['overall']} intra={valid_counts['intra']} cross={valid_counts['cross']}"
    )
    print(
        "Sampled pairs: "
        f"cross_pos={pair_counts['cross_positive']} cross_neg={pair_counts['cross_negative']} "
        f"intra_pos={pair_counts['intra_positive']} intra_neg={pair_counts['intra_negative']}"
    )
    print(f"mAP overall:        {_format_metric(result['map_overall'])}")
    print(f"mAP intra-camera:   {_format_metric(result['map_intra_camera'])}")
    print(f"mAP cross-camera:   {_format_metric(result['map_cross_camera'])}")
    print(f"Cross-camera AUC:   {_format_metric(result['cross_camera_auc'])}")
    print(
        "Cross-camera sims:  "
        f"pos={_format_metric(result['cross_camera_pos_mean_sim'])} "
        f"neg={_format_metric(result['cross_camera_neg_mean_sim'])} "
        f"margin={_format_metric(result['cross_camera_margin'])}"
    )
    print(
        "Intra-camera sims:  "
        f"pos={_format_metric(result['intra_camera_pos_mean_sim'])} "
        f"neg={_format_metric(result['intra_camera_neg_mean_sim'])} "
        f"margin={_format_metric(result['intra_camera_margin'])}"
    )


def save_result(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved JSON: {output_path}")


def compare_reports(base_path: Path, candidate_path: Path) -> int:
    baseline = json.loads(base_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    metrics = [
        "map_overall",
        "map_intra_camera",
        "map_cross_camera",
        "cross_camera_margin",
        "cross_camera_auc",
    ]

    header = f"{'Metric':<28} {'Baseline':>10} {'Candidate':>12} {'Delta':>10}"
    print(header)
    print("-" * len(header))
    for metric in metrics:
        baseline_value = baseline.get(metric)
        candidate_value = candidate.get(metric)
        delta = None
        if baseline_value is not None and candidate_value is not None:
            delta = float(candidate_value - baseline_value)
        delta_text = "n/a" if delta is None else f"{delta:+.4f}"
        print(
            f"{metric:<28} {_format_metric(baseline_value):>10} "
            f"{_format_metric(candidate_value):>12} {delta_text:>10}"
        )

    map_up = (
        baseline.get("map_overall") is not None
        and candidate.get("map_overall") is not None
        and candidate["map_overall"] > baseline["map_overall"]
    )
    cross_margin_down = (
        baseline.get("cross_camera_margin") is not None
        and candidate.get("cross_camera_margin") is not None
        and candidate["cross_camera_margin"] < baseline["cross_camera_margin"]
    )
    cross_auc_down = (
        baseline.get("cross_camera_auc") is not None
        and candidate.get("cross_camera_auc") is not None
        and candidate["cross_camera_auc"] < baseline["cross_camera_auc"]
    )
    verdict = "H2 CONFIRMED" if map_up and cross_margin_down and cross_auc_down else "H2 NOT CONFIRMED"
    print(f"Verdict: {verdict}")
    return 0


def build_synthetic_dataset(
    n_ids: int = 200,
    n_cameras: int = 4,
    samples_per_camera: int = 2,
    dim: int = 64,
    seed: int = 7,
) -> Dataset:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    rows: list[RowMeta] = []
    camera_bias = rng.normal(scale=0.02, size=(n_cameras, dim)).astype(np.float32)

    row_index = 0
    for pid in range(n_ids):
        identity_center = rng.normal(size=dim).astype(np.float32)
        identity_center /= np.maximum(np.linalg.norm(identity_center), 1e-12)
        for cam in range(n_cameras):
            camera_id = f"S01_c{cam + 1:03d}"
            for _ in range(samples_per_camera):
                noise = rng.normal(scale=0.03, size=dim).astype(np.float32)
                feature = identity_center + camera_bias[cam] + noise
                features.append(feature)
                rows.append(
                    RowMeta(
                        index=row_index,
                        camera_id=camera_id,
                        track_id=row_index,
                        pid=pid,
                    )
                )
                row_index += 1

    feature_matrix = l2_normalize(np.vstack(features))
    return Dataset(features=feature_matrix, rows=rows, label="synthetic")


def run_self_test() -> int:
    dataset = build_synthetic_dataset()
    result, valid_counts, pair_counts = run_diagnostic(
        query_dataset=dataset,
        gallery_dataset=dataset,
        tag="self_test",
        sample_size=5000,
        batch_size=256,
        seed=0,
        same_source=True,
    )
    print_summary(result, valid_counts, pair_counts)

    margin = result["cross_camera_margin"]
    auc = result["cross_camera_auc"]
    if margin is None or margin <= 0.0:
        raise AssertionError(f"Expected positive cross-camera margin, got {margin}")
    if auc is None or auc <= 0.95:
        raise AssertionError(f"Expected cross-camera AUC > 0.95, got {auc}")

    print("SELF-TEST PASSED")
    return 0


def main() -> int:
    args = parse_args()

    if args.self_test:
        return run_self_test()

    if args.compare:
        return compare_reports(Path(args.compare[0]), Path(args.compare[1]))

    query_dataset = load_dataset(args.features, args.metadata, label="query")
    if args.gallery_features:
        gallery_dataset = load_dataset(args.gallery_features, args.gallery_metadata, label="gallery")
        same_source = False
    else:
        gallery_dataset = query_dataset
        same_source = True

    result, valid_counts, pair_counts = run_diagnostic(
        query_dataset=query_dataset,
        gallery_dataset=gallery_dataset,
        tag=args.tag,
        sample_size=int(args.sample_size),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        same_source=same_source,
    )
    print_summary(result, valid_counts, pair_counts)

    if args.output:
        save_result(result, Path(args.output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())