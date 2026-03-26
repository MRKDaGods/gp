"""Compute per-camera-pair CID_BIAS from GT annotations.

Computes mean cosine similarity for true-match pairs per camera pair,
then centers each pair around the global mean similarity to produce an
additive bias matrix for Stage 4 association.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage3_indexing.metadata_store import MetadataStore


def compute_cid_bias(
    embeddings: np.ndarray,
    camera_ids: list[str],
    track_ids: list[int],
    gt_matches: dict[tuple[str, int], int],
) -> tuple[np.ndarray, list[str]]:
    """Compute a per-camera-pair additive similarity bias matrix."""
    n = len(embeddings)
    global_ids = []
    for i in range(n):
        key = (camera_ids[i], track_ids[i])
        global_ids.append(gt_matches.get(key, -1))

    camera_names = sorted(set(camera_ids))
    cam2idx = {camera_name: idx for idx, camera_name in enumerate(camera_names)}
    bias_matrix = np.zeros((len(camera_names), len(camera_names)), dtype=np.float32)

    gt_tracklets = [i for i, global_id in enumerate(global_ids) if global_id >= 0]
    logger.info(f"GT-matched tracklets: {len(gt_tracklets)}/{n}")

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
        logger.warning("No cross-camera GT matches found")
        return bias_matrix, camera_names

    global_mean = float(np.mean(all_means))
    logger.info(f"Global mean similarity: {global_mean:.4f}")

    for (ci, cj), sims in pair_sims.items():
        bias = float(np.mean(sims) - global_mean)
        bias_matrix[ci, cj] = bias
        bias_matrix[cj, ci] = bias

    return bias_matrix, camera_names


def _iter_gt_files(gt_dir: Path):
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


def load_gt_matches(gt_dir: Path) -> dict[tuple[str, int], int]:
    """Load GT identity mapping from MOT-format files.

    Assumes GT identity IDs are globally consistent across cameras.
    """
    gt_matches: dict[tuple[str, int], int] = {}
    files_found = 0
    for gt_file in _iter_gt_files(gt_dir):
        files_found += 1
        cam_id = _camera_id_from_gt_file(gt_file)
        with open(gt_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                global_id = int(parts[1])
                gt_matches[(cam_id, global_id)] = global_id

    if files_found == 0:
        logger.warning(f"No GT files found under {gt_dir}")
    else:
        logger.info(f"Loaded GT matches from {files_found} files")
    return gt_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-camera-pair CID_BIAS matrix")
    parser.add_argument("--embeddings", required=True, help="Path to Stage 2 embeddings.npy")
    parser.add_argument("--metadata", required=True, help="Path to Stage 3 metadata.db")
    parser.add_argument("--gt-dir", required=True, help="Directory containing GT MOT files")
    parser.add_argument("--output", required=True, help="Output .npy path for CID_BIAS matrix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embeddings = np.load(args.embeddings).astype(np.float32)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)

    metadata_store = MetadataStore(args.metadata)
    try:
        camera_ids: list[str] = []
        track_ids: list[int] = []
        for index in range(len(embeddings)):
            meta = metadata_store.get_tracklet(index)
            if meta is None:
                raise ValueError(f"Missing metadata for embedding row {index}")
            camera_ids.append(meta["camera_id"])
            track_ids.append(int(meta["track_id"]))
    finally:
        metadata_store.close()

    gt_matches = load_gt_matches(Path(args.gt_dir))
    bias_matrix, camera_names = compute_cid_bias(
        embeddings=embeddings,
        camera_ids=camera_ids,
        track_ids=track_ids,
        gt_matches=gt_matches,
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