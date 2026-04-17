"""Generate a topology-based CID_BIAS matrix for CityFlowV2.

This replaces the old GT-learned bias artifact with a static prior derived from
the known CityFlowV2 camera topology:

- Intra-scene pairs within S01 and within S02 get a positive bias.
- Cross-scene pairs between S01 and S02 get a negative bias.
- Diagonal entries remain zero.

The output format matches what ``src.stage4_association.pipeline`` expects:

- ``<output>.npy``: float32 bias matrix
- ``<output>.json``: sidecar with ordered ``cameras`` list for index mapping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAMERAS = [
    "S01_c001",
    "S01_c002",
    "S01_c003",
    "S02_c006",
    "S02_c007",
    "S02_c008",
]

SCENES = {
    "S01": {"S01_c001", "S01_c002", "S01_c003"},
    "S02": {"S02_c006", "S02_c007", "S02_c008"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a topology-based CityFlowV2 CID_BIAS matrix.",
    )
    parser.add_argument(
        "--intra",
        type=float,
        default=0.04,
        help="Bias applied to camera pairs within the same scene.",
    )
    parser.add_argument(
        "--cross",
        type=float,
        default=-0.15,
        help="Bias applied to camera pairs across scenes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/datasets/cityflowv2_cid_bias.npy"),
        help="Output .npy path. A matching .json sidecar is written alongside it.",
    )
    return parser.parse_args()


def _scene_for_camera(camera_id: str) -> str:
    for scene_name, members in SCENES.items():
        if camera_id in members:
            return scene_name
    raise ValueError(f"Camera {camera_id!r} is not assigned to a known scene")


def build_bias_matrix(intra_bias: float, cross_bias: float) -> np.ndarray:
    n_cameras = len(CAMERAS)
    matrix = np.zeros((n_cameras, n_cameras), dtype=np.float32)

    for row_index, row_camera in enumerate(CAMERAS):
        row_scene = _scene_for_camera(row_camera)
        for col_index, col_camera in enumerate(CAMERAS):
            if row_index == col_index:
                continue
            col_scene = _scene_for_camera(col_camera)
            matrix[row_index, col_index] = intra_bias if row_scene == col_scene else cross_bias

    return matrix


def save_outputs(matrix: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)

    mapping_path = output_path.with_suffix(".json")
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump({"cameras": CAMERAS}, handle, indent=2)
        handle.write("\n")

    return mapping_path


def print_matrix(matrix: np.ndarray) -> None:
    print("Camera order:")
    for index, camera_id in enumerate(CAMERAS):
        print(f"  {index}: {camera_id}")

    print("\nCID_BIAS matrix:")
    for camera_id, row in zip(CAMERAS, matrix):
        row_values = " ".join(f"{value:+0.2f}" for value in row)
        print(f"  {camera_id}: [{row_values}]")


def main() -> None:
    args = parse_args()
    matrix = build_bias_matrix(intra_bias=args.intra, cross_bias=args.cross)
    mapping_path = save_outputs(matrix, args.output)

    print(f"Saved matrix to: {args.output}")
    print(f"Saved mapping to: {mapping_path}")
    print_matrix(matrix)


if __name__ == "__main__":
    main()