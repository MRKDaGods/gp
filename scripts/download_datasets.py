"""Download multi-camera tracking datasets for the MTMC pipeline.

Supported datasets:
  - cityflowv2: AI City Challenge 2022 Track 1 (46 cameras, vehicles, city intersections)
  - wildtrack:   EPFL WILDTRACK (7 cameras, pedestrians, 1080p outdoor)

Usage:
    python scripts/download_datasets.py --dataset cityflowv2
    python scripts/download_datasets.py --dataset wildtrack
    python scripts/download_datasets.py --dataset all
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import click

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASETS = {
    "cityflowv2": {
        "name": "CityFlowV2 (AI City Challenge 2022 Track 1)",
        "description": "Multi-camera multi-target vehicle tracking across 46 cameras at 16 city intersections",
        "target_dir": "data/raw/cityflowv2",
        "type": "gdrive",
        # AIC 2022 Track 1 — dataset archive on Google Drive
        # If this ID becomes stale, check https://www.aicitychallenge.org/2022-data-and-evaluation/
        "gdrive_id": "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC",
        "archive_name": "AIC22_Track1_MTMC_Tracking.zip",
        "post_extract": "_post_extract_cityflowv2",
    },
    "wildtrack": {
        "name": "WILDTRACK (EPFL CVLAB)",
        "description": "7-camera 1080p outdoor pedestrian tracking with dense annotations",
        "target_dir": "data/raw/wildtrack",
        "type": "http",
        "url": "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip",
        "archive_name": "Wildtrack_dataset_full.zip",
        "post_extract": "_post_extract_wildtrack",
    },
}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_gdrive(gdrive_id: str, output_path: Path) -> None:
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("  Installing gdown for Google Drive downloads...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"  Downloading from Google Drive (id={gdrive_id})...")
    print(f"  This may take a while for large datasets.")
    gdown.download(url, str(output_path), quiet=False)


def _download_http(url: str, output_path: Path) -> None:
    """Download a file over HTTP with progress."""
    print(f"  Downloading {url}...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)

    urlretrieve(url, str(output_path), reporthook=_progress)
    print()  # newline after progress


def _extract_zip(archive_path: Path, extract_dir: Path) -> None:
    """Extract a zip archive."""
    print(f"  Extracting {archive_path.name}...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"  Extracted to {extract_dir}")


# ---------------------------------------------------------------------------
# Post-extraction hooks
# ---------------------------------------------------------------------------

def _post_extract_cityflowv2(target_dir: Path) -> None:
    """Flatten CityFlowV2 structure so videos are directly discoverable.

    CityFlowV2 comes as:
        AIC22_Track1_MTMC_Tracking/
          train/S001/c001/vdo.avi
          train/S001/c001/gt/gt.txt
          ...

    We reorganise into:
        data/raw/cityflowv2/
          S001_c001/video.mp4   (or .avi)
          S001_c001/gt.txt
          S001_c002/video.mp4
          ...

    This way each subfolder is a camera, and Stage 0 auto-discovers videos.
    """
    # Find the extracted root (may be nested)
    extracted = _find_extracted_root(target_dir, marker_dirs=["train", "test"])
    if extracted is None:
        print("  Warning: Could not find expected CityFlowV2 directory structure.")
        print(f"  Please check {target_dir} manually.")
        return

    print("  Reorganising CityFlowV2 directory structure...")
    count = 0
    for split_dir in sorted(extracted.iterdir()):
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_name = scene_dir.name  # e.g. S001
            for cam_dir in sorted(scene_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue
                cam_name = cam_dir.name  # e.g. c001
                flat_name = f"{scene_name}_{cam_name}"
                flat_dir = target_dir / flat_name
                flat_dir.mkdir(parents=True, exist_ok=True)

                # Move video
                for ext in (".avi", ".mp4", ".mkv", ".mov"):
                    for vf in cam_dir.glob(f"*{ext}"):
                        dest = flat_dir / vf.name
                        if not dest.exists():
                            shutil.move(str(vf), str(dest))
                        count += 1

                # Move ground truth
                gt_dir = cam_dir / "gt"
                if gt_dir.exists():
                    for gt_file in gt_dir.glob("*"):
                        dest = flat_dir / gt_file.name
                        if not dest.exists():
                            shutil.move(str(gt_file), str(dest))

    # Clean up extracted root if it differs from target
    if extracted != target_dir and extracted.exists():
        shutil.rmtree(extracted, ignore_errors=True)

    print(f"  Organised {count} camera feeds into {target_dir}")


def _post_extract_wildtrack(target_dir: Path) -> None:
    """Ensure WILDTRACK structure is correct.

    Expected result:
        data/raw/wildtrack/
          Image_subsets/C1/*.png, C2/*.png, ... C7/*.png
          annotations_positions/*.json

    WILDTRACK comes with frame images (not video), so we generate stub videos
    from the image sequences so Stage 0 can process them uniformly — OR we
    configure Stage 0 to read frames directly.  For now we just verify structure.
    """
    extracted = _find_extracted_root(target_dir, marker_dirs=["Image_subsets"])
    if extracted is None:
        print("  Warning: Could not find expected WILDTRACK directory structure.")
        print(f"  Please check {target_dir} manually.")
        return

    # If extracted into a subfolder, hoist contents up
    if extracted != target_dir:
        print(f"  Moving contents from {extracted.name}/ to {target_dir}/...")
        for item in extracted.iterdir():
            dest = target_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(extracted, ignore_errors=True)

    # Verify camera folders
    img_root = target_dir / "Image_subsets"
    if img_root.exists():
        cams = sorted(d.name for d in img_root.iterdir() if d.is_dir())
        print(f"  Found camera folders: {', '.join(cams)}")
        for cam in cams:
            n_frames = len(list((img_root / cam).glob("*.png")))
            print(f"    {cam}: {n_frames} frames")

    # Verify annotations
    ann_dir = target_dir / "annotations_positions"
    if ann_dir.exists():
        n_ann = len(list(ann_dir.glob("*.json")))
        print(f"  Found {n_ann} annotation frames")

    # Generate videos from image sequences for Stage 0
    _generate_wildtrack_videos(target_dir)


def _generate_wildtrack_videos(target_dir: Path) -> None:
    """Create MP4 videos from WILDTRACK image sequences.

    WILDTRACK provides frame images, not video files. We encode them as
    MP4 so the existing Stage 0 video-based ingestion can discover them.
    """
    img_root = target_dir / "Image_subsets"
    if not img_root.exists():
        return

    try:
        import cv2
    except ImportError:
        print("  Warning: opencv-python not installed, skipping video generation.")
        print("  Install it with: pip install opencv-python")
        return

    cams = sorted(d for d in img_root.iterdir() if d.is_dir())
    videos_dir = target_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    for cam_dir in cams:
        frames = sorted(cam_dir.glob("*.png"))
        if not frames:
            continue

        video_path = videos_dir / f"{cam_dir.name}.mp4"
        if video_path.exists():
            print(f"  Video already exists: {video_path.name}")
            continue

        # Read first frame to get dimensions
        sample = cv2.imread(str(frames[0]))
        if sample is None:
            continue
        h, w = sample.shape[:2]

        # WILDTRACK images are sampled at 2 fps from a 60 fps stream
        # The frame filenames are the original frame numbers (0, 5, 10, ...)
        # We write at 2 fps to preserve the original timing
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (w, h)
        )
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is not None:
                writer.write(img)
        writer.release()
        print(f"  Generated video: {video_path.name} ({len(frames)} frames, {w}x{h})")

    print(f"  Videos saved to {videos_dir}")


def _find_extracted_root(base_dir: Path, marker_dirs: list[str]) -> Path | None:
    """Find the actual root of an extracted archive by looking for marker dirs."""
    # Check if markers are directly in base_dir
    if any((base_dir / m).exists() for m in marker_dirs):
        return base_dir
    # Check one level down (common with zip archives)
    for child in base_dir.iterdir():
        if child.is_dir() and any((child / m).exists() for m in marker_dirs):
            return child
    # Check two levels down
    for child in base_dir.iterdir():
        if child.is_dir():
            for grandchild in child.iterdir():
                if grandchild.is_dir() and any((grandchild / m).exists() for m in marker_dirs):
                    return grandchild
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--dataset", "-d", required=True,
    type=click.Choice(["cityflowv2", "wildtrack", "all"]),
    help="Dataset to download",
)
@click.option("--data-dir", default="data/raw", help="Base directory for raw data")
@click.option("--keep-archive", is_flag=True, default=False, help="Keep zip after extraction")
def main(dataset: str, data_dir: str, keep_archive: bool):
    """Download and prepare multi-camera tracking datasets."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = list(DATASETS.keys()) if dataset == "all" else [dataset]

    for ds_name in datasets_to_download:
        ds = DATASETS[ds_name]
        target_dir = project_root / ds["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"Dataset: {ds['name']}")
        print(f"  {ds['description']}")
        print(f"  Target: {target_dir}")
        print("=" * 60)

        archive_path = target_dir / ds["archive_name"]

        # Download
        if archive_path.exists():
            print(f"  Archive already exists: {archive_path}")
        else:
            if ds["type"] == "gdrive":
                _download_gdrive(ds["gdrive_id"], archive_path)
            elif ds["type"] == "http":
                _download_http(ds["url"], archive_path)

        if not archive_path.exists():
            print(f"  ERROR: Download failed, {archive_path} not found.")
            continue

        # Extract
        _extract_zip(archive_path, target_dir)

        # Post-extraction
        post_fn = globals().get(ds["post_extract"])
        if post_fn:
            post_fn(target_dir)

        # Cleanup
        if not keep_archive and archive_path.exists():
            print(f"  Removing archive {archive_path.name}...")
            archive_path.unlink()

        print(f"  Dataset ready at {target_dir}\n")

    print("=" * 60)
    print("All done. Update your config to point stage0.input_dir at the dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()
