"""Compute per-camera-pair CID_BIAS from CityFlowV2 GT and stage outputs.

This script matches predicted stage-1 tracklets to GT global vehicle IDs using
frame-level IoU, then measures the cosine similarity of same-identity tracklets
across cameras from stage-2 embeddings. The resulting bias matrix is centered
around the global mean similarity and saved in the format consumed by
``src.stage4_association.pipeline``.

Expected inputs:
    - stage2 ``embeddings.npy`` and ``embedding_index.json``
    - stage1 ``tracklets_<camera>.json`` files
    - CityFlowV2 GT files (``gt.txt`` or ``gt/gt.txt`` under each camera dir)

Usage:
    python scripts/compute_cid_bias.py \
        --stage2-dir data/outputs/run_xxx/stage2 \
        --tracklets-dir data/outputs/run_xxx/stage1 \
        --gt-dir data/raw/cityflowv2 \
        --output configs/datasets/cityflowv2_cid_bias.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.io_utils import load_tracklets_by_camera


BBox = Tuple[float, float, float, float]


def compute_cid_bias(
    embeddings: np.ndarray,
    camera_ids: list[str],
    track_ids: list[int],
    gt_assignments: dict[tuple[str, int], int],
    camera_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute a per-camera-pair additive similarity bias matrix.

    Args:
        embeddings: L2-normalized embedding matrix of shape ``(N, D)``.
        camera_ids: Camera ID per embedding row.
        track_ids: Track ID per embedding row.
        gt_assignments: Mapping ``(camera_id, predicted_track_id) -> gt_global_id``.
        camera_names: Optional explicit camera order for the output matrix.

    Returns:
        ``(bias_matrix, camera_names)`` where ``bias_matrix`` is ``float32`` and
        symmetric with zeros on the diagonal.
    """
    global_ids = [gt_assignments.get((camera_ids[i], track_ids[i]), -1) for i in range(len(embeddings))]

    if camera_names is None:
        camera_names = sorted(set(camera_ids))
    cam2idx = {camera_name: idx for idx, camera_name in enumerate(camera_names)}
    bias_matrix = np.zeros((len(camera_names), len(camera_names)), dtype=np.float32)

    gt_tracklets = [i for i, global_id in enumerate(global_ids) if global_id >= 0]
    logger.info(f"GT-matched tracklets: {len(gt_tracklets)}/{len(embeddings)}")

    gid_to_indices: dict[int, list[int]] = defaultdict(list)
    for index in gt_tracklets:
        gid_to_indices[global_ids[index]].append(index)

    pair_sims: dict[tuple[int, int], list[float]] = defaultdict(list)
    for indices in gid_to_indices.values():
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
                pair_sims[pair_key].append(float(np.dot(embeddings[i], embeddings[j])))

    all_means: list[float] = []
    for (ci, cj), sims in sorted(pair_sims.items()):
        mean_sim = float(np.mean(sims))
        all_means.append(mean_sim)
        logger.info(
            f"{camera_names[ci]}<->{camera_names[cj]}: mean_sim={mean_sim:.4f}, n={len(sims)}"
        )

    if not all_means:
        logger.warning("No cross-camera GT matches found after tracklet assignment")
        return bias_matrix, camera_names

    global_mean = float(np.mean(all_means))
    logger.info(f"Global mean similarity: {global_mean:.4f}")

    for (ci, cj), sims in pair_sims.items():
        bias = float(np.mean(sims) - global_mean)
        bias_matrix[ci, cj] = bias
        bias_matrix[cj, ci] = bias

    return bias_matrix, camera_names


def _iter_gt_files(gt_dir: Path) -> Iterable[Path]:
    patterns = ["*.txt", "*/gt.txt", "*/gt/gt.txt"]
    seen: set[Path] = set()
    for pattern in patterns:
        for gt_file in sorted(gt_dir.glob(pattern)):
            resolved = gt_file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield gt_file


def _camera_id_from_gt_file(gt_file: Path) -> str:
    if gt_file.name == "gt.txt":
        if gt_file.parent.name == "gt":
            return gt_file.parent.parent.name
        return gt_file.parent.name
    return gt_file.stem


def _mot_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> BBox:
    return (x, y, x + w, y + h)


def _bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_gt_boxes(gt_dir: Path) -> dict[str, dict[int, list[tuple[int, BBox]]]]:
    """Load MOT-format GT into ``camera -> frame -> [(global_id, bbox_xyxy)]``.

    CityFlowV2 ground truth uses 1-based frame IDs and globally consistent
    vehicle IDs across cameras.
    """
    gt_by_camera: dict[str, dict[int, list[tuple[int, BBox]]]] = defaultdict(lambda: defaultdict(list))
    files_found = 0
    for gt_file in _iter_gt_files(gt_dir):
        files_found += 1
        camera_id = _camera_id_from_gt_file(gt_file)
        with open(gt_file, encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                global_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                gt_by_camera[camera_id][frame_id].append(
                    (global_id, _mot_xywh_to_xyxy(x, y, w, h))
                )

    if files_found == 0:
        logger.warning(f"No GT files found under {gt_dir}")
    else:
        logger.info(
            f"Loaded GT boxes from {files_found} files across {len(gt_by_camera)} cameras"
        )
    return gt_by_camera


def assign_tracklets_to_gt(
    tracklets_dir: Path,
    gt_by_camera: dict[str, dict[int, list[tuple[int, BBox]]]],
    min_iou: float,
    min_matches: int,
    min_coverage: float,
    frame_offset: int,
) -> dict[tuple[str, int], int]:
    """Assign each predicted tracklet to a GT global ID via majority IoU matching."""
    tracklets_by_camera = load_tracklets_by_camera(tracklets_dir)
    assignments: dict[tuple[str, int], int] = {}

    total_tracklets = 0
    total_assigned = 0
    for camera_id, tracklets in sorted(tracklets_by_camera.items()):
        cam_assigned = 0
        total_tracklets += len(tracklets)
        camera_gt = gt_by_camera.get(camera_id, {})

        for tracklet in tracklets:
            hits: dict[int, list[float]] = defaultdict(list)
            for frame in tracklet.frames:
                gt_frame_id = frame.frame_id + frame_offset
                gt_candidates = camera_gt.get(gt_frame_id, [])
                if not gt_candidates:
                    continue

                best_global_id = None
                best_iou = 0.0
                for global_id, gt_bbox in gt_candidates:
                    iou = _bbox_iou(tuple(frame.bbox), gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_global_id = global_id

                if best_global_id is not None and best_iou >= min_iou:
                    hits[best_global_id].append(best_iou)

            if not hits:
                continue

            best_global_id, best_ious = max(
                hits.items(),
                key=lambda item: (len(item[1]), float(np.mean(item[1]))),
            )
            match_count = len(best_ious)
            coverage = match_count / max(len(tracklet.frames), 1)
            if match_count < min_matches or coverage < min_coverage:
                continue

            assignments[(camera_id, tracklet.track_id)] = best_global_id
            cam_assigned += 1
            total_assigned += 1

        logger.info(
            f"{camera_id}: assigned {cam_assigned}/{len(tracklets)} tracklets to GT"
        )

    logger.info(f"Tracklet GT assignment: {total_assigned}/{total_tracklets} total assigned")
    return assignments


def _resolve_stage2_inputs(
    stage2_dir: str | None,
    embeddings_path: str | None,
    index_map_path: str | None,
) -> tuple[Path, Path]:
    if stage2_dir:
        stage2_path = Path(stage2_dir)
        embeddings = stage2_path / "embeddings.npy"
        index_map = stage2_path / "embedding_index.json"
    else:
        if not embeddings_path or not index_map_path:
            raise ValueError("Provide --stage2-dir or both --embeddings and --index-map")
        embeddings = Path(embeddings_path)
        index_map = Path(index_map_path)

    if not embeddings.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings}")
    if not index_map.exists():
        raise FileNotFoundError(f"Embedding index file not found: {index_map}")
    return embeddings, index_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-camera-pair CID_BIAS matrix")
    parser.add_argument(
        "--stage2-dir",
        default=None,
        help="Directory containing stage2 embeddings.npy and embedding_index.json",
    )
    parser.add_argument("--embeddings", default=None, help="Path to stage2 embeddings.npy")
    parser.add_argument(
        "--index-map",
        default=None,
        help="Path to stage2 embedding_index.json",
    )
    parser.add_argument(
        "--tracklets-dir",
        required=True,
        help="Directory containing stage1 tracklets_<camera>.json files",
    )
    parser.add_argument("--gt-dir", required=True, help="Directory containing GT MOT files")
    parser.add_argument("--output", required=True, help="Output .npy path for CID_BIAS matrix")
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.5,
        help="Minimum frame-level IoU required to count a GT match",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=3,
        help="Minimum matched frames required to assign a predicted tracklet to GT",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.3,
        help="Minimum matched-frame ratio required for GT assignment",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=1,
        help="Offset applied to predicted frame IDs before GT lookup. CityFlowV2 uses 0-based internal frames and 1-based MOT GT, so the default is 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embeddings_path, index_map_path = _resolve_stage2_inputs(
        stage2_dir=args.stage2_dir,
        embeddings_path=args.embeddings,
        index_map_path=args.index_map,
    )

    embeddings = np.load(embeddings_path).astype(np.float32)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)

    with open(index_map_path, encoding="utf-8") as handle:
        index_map = json.load(handle)
    camera_ids = [str(item["camera_id"]) for item in index_map]
    track_ids = [int(item["track_id"]) for item in index_map]

    gt_by_camera = load_gt_boxes(Path(args.gt_dir))
    gt_assignments = assign_tracklets_to_gt(
        tracklets_dir=Path(args.tracklets_dir),
        gt_by_camera=gt_by_camera,
        min_iou=float(args.min_iou),
        min_matches=int(args.min_matches),
        min_coverage=float(args.min_coverage),
        frame_offset=int(args.frame_offset),
    )

    camera_names = sorted(set(camera_ids) | set(gt_by_camera.keys()))
    bias_matrix, camera_names = compute_cid_bias(
        embeddings=embeddings,
        camera_ids=camera_ids,
        track_ids=track_ids,
        gt_assignments=gt_assignments,
        camera_names=camera_names,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, bias_matrix)

    mapping_path = output_path.with_suffix(".json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"cameras": camera_names}, f, indent=2)

    logger.info(f"CID_BIAS saved to {output_path} ({bias_matrix.shape})")
    logger.info(f"Camera mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()