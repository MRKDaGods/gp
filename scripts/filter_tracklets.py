"""Filter Stage 1/2 tracklet artifacts by track length and confidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core.io_utils import load_tracklets_by_camera, save_tracklets_by_camera


OPTIONAL_STAGE2_ARRAYS = (
    "embeddings_secondary.npy",
    "embeddings_tertiary.npy",
    "hsv_features.npy",
)


def _tracklet_stats(tracklet: Any) -> tuple[int, float]:
    length = int(tracklet.num_frames)
    avg_confidence = float(tracklet.mean_confidence) if length > 0 else 0.0
    return length, avg_confidence


def _load_index_map(stage2_dir: Path) -> list[dict[str, Any]]:
    index_path = stage2_dir / "embedding_index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {index_path}, got {type(payload).__name__}")
    return payload


def _copy_filtered_array(input_path: Path, output_path: Path, keep_indices: list[int]) -> list[int]:
    array = np.load(input_path)
    if array.shape[0] < len(keep_indices):
        raise ValueError(
            f"Array {input_path} has {array.shape[0]} rows, fewer than selected index count {len(keep_indices)}"
        )
    filtered = array[np.asarray(keep_indices, dtype=np.int64)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, filtered.astype(array.dtype, copy=False))
    return [int(value) for value in filtered.shape]


def _copy_filtered_multi_query(input_path: Path, output_path: Path, keep_indices: list[int]) -> list[int] | None:
    if not input_path.exists():
        return None
    with np.load(input_path) as data:
        if "embeddings" not in data:
            raise KeyError(f"{input_path} does not contain an 'embeddings' array")
        embeddings = data["embeddings"]
    filtered = embeddings[np.asarray(keep_indices, dtype=np.int64)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=filtered.astype(embeddings.dtype, copy=False))
    return [int(value) for value in filtered.shape]


def filter_stage_outputs(
    stage1_dir: str | Path,
    stage2_dir: str | Path,
    output_stage1_dir: str | Path,
    output_stage2_dir: str | Path,
    min_avg_confidence: float,
    min_length: int,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write filtered Stage 1/2 artifacts while preserving Stage 2 row order."""
    stage1_dir = Path(stage1_dir)
    stage2_dir = Path(stage2_dir)
    output_stage1_dir = Path(output_stage1_dir)
    output_stage2_dir = Path(output_stage2_dir)

    tracklets_by_camera = load_tracklets_by_camera(stage1_dir)
    tracklet_lookup = {
        (camera_id, tracklet.track_id): tracklet
        for camera_id, tracklets in tracklets_by_camera.items()
        for tracklet in tracklets
    }
    index_map = _load_index_map(stage2_dir)

    keep_indices: list[int] = []
    filtered_index_map: list[dict[str, Any]] = []
    filtered_tracklets_by_camera: dict[str, list[Any]] = {camera_id: [] for camera_id in tracklets_by_camera}
    drop_counts = {
        "length_only": 0,
        "confidence_only": 0,
        "both": 0,
        "missing_tracklet": 0,
    }
    per_tracklet: list[dict[str, Any]] = []

    for row_index, row in enumerate(index_map):
        camera_id = str(row["camera_id"])
        track_id = int(row["track_id"])
        tracklet = tracklet_lookup.get((camera_id, track_id))
        if tracklet is None:
            drop_counts["missing_tracklet"] += 1
            per_tracklet.append({
                "row_index": row_index,
                "camera_id": camera_id,
                "track_id": track_id,
                "keep": False,
                "drop_reason": "missing_tracklet",
            })
            continue

        length, avg_confidence = _tracklet_stats(tracklet)
        drop_length = length < int(min_length)
        drop_confidence = avg_confidence < float(min_avg_confidence)
        keep = not (drop_length or drop_confidence)
        if keep:
            keep_indices.append(row_index)
            filtered_index_map.append(row)
            filtered_tracklets_by_camera.setdefault(camera_id, []).append(tracklet)
            drop_reason = None
        elif drop_length and drop_confidence:
            drop_counts["both"] += 1
            drop_reason = "both"
        elif drop_length:
            drop_counts["length_only"] += 1
            drop_reason = "length_only"
        else:
            drop_counts["confidence_only"] += 1
            drop_reason = "confidence_only"

        per_tracklet.append({
            "row_index": row_index,
            "camera_id": camera_id,
            "track_id": track_id,
            "length": length,
            "avg_confidence": avg_confidence,
            "keep": keep,
            "drop_reason": drop_reason,
        })

    output_stage1_dir.mkdir(parents=True, exist_ok=True)
    output_stage2_dir.mkdir(parents=True, exist_ok=True)
    save_tracklets_by_camera(filtered_tracklets_by_camera, output_stage1_dir)
    (output_stage2_dir / "embedding_index.json").write_text(
        json.dumps(filtered_index_map, indent=2),
        encoding="utf-8",
    )

    array_shapes = {
        "embeddings.npy": _copy_filtered_array(stage2_dir / "embeddings.npy", output_stage2_dir / "embeddings.npy", keep_indices),
    }
    for filename in OPTIONAL_STAGE2_ARRAYS:
        source_path = stage2_dir / filename
        if source_path.exists():
            array_shapes[filename] = _copy_filtered_array(source_path, output_stage2_dir / filename, keep_indices)
    multi_query_shape = _copy_filtered_multi_query(
        stage2_dir / "multi_query_embeddings.npz",
        output_stage2_dir / "multi_query_embeddings.npz",
        keep_indices,
    )
    if multi_query_shape is not None:
        array_shapes["multi_query_embeddings.npz"] = multi_query_shape

    input_counts_by_camera = {camera_id: len(tracklets) for camera_id, tracklets in tracklets_by_camera.items()}
    output_counts_by_camera = {
        camera_id: len(filtered_tracklets_by_camera.get(camera_id, []))
        for camera_id in input_counts_by_camera
    }
    count_delta_by_camera = {
        camera_id: output_counts_by_camera[camera_id] - input_count
        for camera_id, input_count in input_counts_by_camera.items()
    }
    summary = {
        "min_avg_confidence": float(min_avg_confidence),
        "min_length": int(min_length),
        "total_in": len(index_map),
        "total_out": len(keep_indices),
        "total_dropped": len(index_map) - len(keep_indices),
        "drop_counts": drop_counts,
        "input_counts_by_camera": input_counts_by_camera,
        "output_counts_by_camera": output_counts_by_camera,
        "count_delta_by_camera": count_delta_by_camera,
        "keep_indices": keep_indices,
        "array_shapes": array_shapes,
        "per_tracklet": per_tracklet,
        "paths": {
            "stage1_dir": str(stage1_dir),
            "stage2_dir": str(stage2_dir),
            "output_stage1_dir": str(output_stage1_dir),
            "output_stage2_dir": str(output_stage2_dir),
        },
    }

    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--output-stage1-dir", type=Path, required=True)
    parser.add_argument("--output-stage2-dir", type=Path, required=True)
    parser.add_argument("--min-avg-confidence", type=float, required=True)
    parser.add_argument("--min-length", type=int, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = filter_stage_outputs(
        stage1_dir=args.stage1_dir,
        stage2_dir=args.stage2_dir,
        output_stage1_dir=args.output_stage1_dir,
        output_stage2_dir=args.output_stage2_dir,
        min_avg_confidence=args.min_avg_confidence,
        min_length=args.min_length,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()