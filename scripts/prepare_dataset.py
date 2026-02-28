"""Dataset preparation utility.

Converts raw downloaded datasets into the unified format expected by the pipeline.

Usage:
    python scripts/prepare_dataset.py --dataset market1501 --root data/raw/market1501
    python scripts/prepare_dataset.py --dataset veri776 --root data/raw/veri776
    python scripts/prepare_dataset.py --dataset aic2023 --root data/raw/aic2023
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

import click

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@click.command()
@click.option("--dataset", "-d", required=True, type=click.Choice(["market1501", "veri776", "aic2023"]))
@click.option("--root", "-r", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None, type=click.Path())
def main(dataset: str, root: str, output: str | None):
    """Prepare a dataset for the MTMC pipeline."""
    root = Path(root)
    output = Path(output) if output else root / "manifests"
    output.mkdir(parents=True, exist_ok=True)

    if dataset == "market1501":
        prepare_market1501(root, output)
    elif dataset == "veri776":
        prepare_veri776(root, output)
    elif dataset == "aic2023":
        print(f"AIC2023 preparation: check dataset structure in {root}")
        print("The pipeline reads videos directly from the dataset directory.")

    print(f"Done. Manifests saved to {output}")


def prepare_market1501(root: Path, output: Path):
    """Prepare Market-1501 dataset manifests."""
    splits = {
        "train": "bounding_box_train",
        "query": "query",
        "gallery": "bounding_box_test",
    }

    for split_name, folder_name in splits.items():
        folder = root / folder_name
        if not folder.exists():
            print(f"  Warning: {folder} not found, skipping {split_name}")
            continue

        rows = []
        # Market-1501 filename: XXXX_cYsZ_NNNNNN_NN.jpg
        # XXXX = person ID (-1 for junk, 0000 for distractor)
        pattern = re.compile(r"(-?\d+)_c(\d+)s\d+_\d+_\d+\.jpg")

        for img_path in sorted(folder.glob("*.jpg")):
            match = pattern.match(img_path.name)
            if not match:
                continue
            pid = int(match.group(1))
            cam_id = int(match.group(2))
            if pid < 0:  # skip junk
                continue
            rows.append((str(img_path), pid, cam_id))

        manifest_path = output / f"market1501_{split_name}.csv"
        _write_manifest(rows, manifest_path)
        print(f"  {split_name}: {len(rows)} images, {len(set(r[1] for r in rows))} IDs")


def prepare_veri776(root: Path, output: Path):
    """Prepare VeRi-776 dataset manifests."""
    splits = {
        "train": "image_train",
        "query": "image_query",
        "gallery": "image_test",
    }

    for split_name, folder_name in splits.items():
        folder = root / folder_name
        if not folder.exists():
            print(f"  Warning: {folder} not found, skipping {split_name}")
            continue

        rows = []
        # VeRi-776 filename: XXXX_cYYY_NNNNN.jpg
        pattern = re.compile(r"(\d+)_c(\d+)_\d+.*\.jpg")

        for img_path in sorted(folder.glob("*.jpg")):
            match = pattern.match(img_path.name)
            if not match:
                continue
            vid = int(match.group(1))
            cam_id = int(match.group(2))
            rows.append((str(img_path), vid, cam_id))

        manifest_path = output / f"veri776_{split_name}.csv"
        _write_manifest(rows, manifest_path)
        print(f"  {split_name}: {len(rows)} images, {len(set(r[1] for r in rows))} IDs")


def _write_manifest(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "identity_id", "camera_id"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
