"""Dataset preparation utility.

Converts raw downloaded datasets into the unified format expected by the pipeline.

Usage:
    python scripts/prepare_dataset.py --dataset market1501 --root data/raw/market1501
    python scripts/prepare_dataset.py --dataset veri776 --root data/raw/veri776
    python scripts/prepare_dataset.py --dataset aic2023 --root data/raw/aic2023
    python scripts/prepare_dataset.py --dataset cityflowv2 --root data/raw/cityflowv2
    python scripts/prepare_dataset.py --dataset wildtrack --root data/raw/wildtrack
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import click

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@click.command()
@click.option(
    "--dataset", "-d", required=True,
    type=click.Choice(["market1501", "veri776", "aic2023", "cityflowv2", "wildtrack"]),
)
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
    elif dataset == "cityflowv2":
        prepare_cityflowv2(root, output)
    elif dataset == "wildtrack":
        prepare_wildtrack(root, output)

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


def prepare_cityflowv2(root: Path, output: Path):
    """Prepare CityFlowV2 dataset — verify structure and list cameras.

    After download_datasets.py, CityFlowV2 is organised as:
        data/raw/cityflowv2/S001_c001/vdo.avi + gt.txt
        data/raw/cityflowv2/S001_c002/vdo.avi + gt.txt
        ...

    This function verifies the structure and prints a summary.
    The pipeline reads videos directly from these camera folders.
    """
    cameras = []
    total_gt_tracks = 0

    for cam_dir in sorted(root.iterdir()):
        if not cam_dir.is_dir() or cam_dir.name == "manifests":
            continue

        videos = list(cam_dir.glob("*.avi")) + list(cam_dir.glob("*.mp4"))
        gt_file = cam_dir / "gt.txt"

        n_tracks = 0
        if gt_file.exists():
            track_ids = set()
            with open(gt_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        track_ids.add(parts[1])
            n_tracks = len(track_ids)
            total_gt_tracks += n_tracks

        if videos:
            cameras.append(cam_dir.name)
            print(f"  {cam_dir.name}: {len(videos)} video(s), {n_tracks} GT tracks")

    print(f"\n  Total: {len(cameras)} cameras, {total_gt_tracks} GT tracks")
    print("  The pipeline reads videos directly from camera subfolders.")

    # Write camera list for reference
    cam_list_path = output / "cityflowv2_cameras.csv"
    with open(cam_list_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id"])
        for cam in cameras:
            writer.writerow([cam])


def prepare_wildtrack(root: Path, output: Path):
    """Prepare WILDTRACK dataset — convert JSON annotations to MOT format.

    Converts the WILDTRACK per-frame JSON annotations into per-camera
    MOTChallenge-format gt.txt files that the evaluation stage can use.
    """
    ann_dir = root / "annotations_positions"
    if not ann_dir.exists():
        print(f"  Warning: annotations_positions/ not found in {root}")
        return

    json_files = sorted(ann_dir.glob("*.json"))
    print(f"  Found {len(json_files)} annotation frames")

    # Accumulate per-camera tracks: camera_id -> {person_id -> [(frame, bbox)]}
    cam_tracks: dict[str, dict[int, list]] = {}

    for json_path in json_files:
        frame_id = int(json_path.stem)

        with open(json_path, "r") as f:
            annotations = json.load(f)

        for entry in annotations:
            person_id = entry["personID"]
            for view in entry.get("views", []):
                view_num = view["viewNum"]
                xmin = view.get("xmin", -1)
                ymin = view.get("ymin", -1)
                xmax = view.get("xmax", -1)
                ymax = view.get("ymax", -1)

                if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
                    continue

                camera_id = f"C{view_num + 1}"
                if camera_id not in cam_tracks:
                    cam_tracks[camera_id] = {}
                if person_id not in cam_tracks[camera_id]:
                    cam_tracks[camera_id][person_id] = []

                w = xmax - xmin
                h = ymax - ymin
                cam_tracks[camera_id][person_id].append(
                    (frame_id, person_id, xmin, ymin, w, h, 1.0, 0)
                )

    # Write per-camera gt.txt in MOT format
    gt_dir = output / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for camera_id, tracks in sorted(cam_tracks.items()):
        rows = []
        for person_id, detections in sorted(tracks.items()):
            rows.extend(detections)
        rows.sort(key=lambda r: (r[0], r[1]))

        gt_path = gt_dir / f"{camera_id}_gt.txt"
        with open(gt_path, "w") as f:
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")

        n_ids = len(tracks)
        n_dets = len(rows)
        print(f"  {camera_id}: {n_ids} identities, {n_dets} detections -> {gt_path.name}")

    print(f"\n  Ground truth files written to {gt_dir}")


if __name__ == "__main__":
    main()
