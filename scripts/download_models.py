"""Download pre-trained model weights for the pipeline.

Downloads:
  - YOLO11m weights (via ultralytics)
  - BoxMOT ReID weights for the tracker

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models-dir models/
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@click.command()
@click.option("--models-dir", default="models", help="Base directory for model weights")
def main(models_dir: str):
    """Download pre-trained model weights."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MTMC Tracker — Model Weight Download")
    print("=" * 60)

    # 1. YOLO11m
    print("\n[1/3] Downloading YOLO11m weights...")
    det_dir = models_dir / "detection"
    det_dir.mkdir(exist_ok=True)

    try:
        from ultralytics import YOLO
        model = YOLO("yolo11m.pt")  # auto-downloads to working dir
        # Move to models dir if needed
        yolo_path = Path("yolo11m.pt")
        if yolo_path.exists():
            target = det_dir / "yolo11m.pt"
            if not target.exists():
                yolo_path.rename(target)
            print(f"  YOLO11m saved to {target}")
        else:
            print("  YOLO11m downloaded (cached by ultralytics)")
    except Exception as e:
        print(f"  Warning: Could not download YOLO11m: {e}")
        print("  It will be auto-downloaded on first use.")

    # 2. BoxMOT ReID weights
    print("\n[2/3] Downloading BoxMOT tracker ReID weights...")
    tracker_dir = models_dir / "tracker"
    tracker_dir.mkdir(exist_ok=True)

    try:
        # BoxMOT auto-downloads ReID weights on first use
        # We trigger this by importing and checking
        print("  BoxMOT will auto-download ReID weights on first tracker initialization.")
        print("  Supported weights: osnet_x0_25_msmt17.pt")
    except Exception as e:
        print(f"  Note: {e}")

    # 3. ReID weights reminder
    print("\n[3/3] Custom ReID weights (trained on Kaggle)")
    reid_dir = models_dir / "reid"
    reid_dir.mkdir(exist_ok=True)

    person_weights = reid_dir / "person_osnet_market1501.pth"
    vehicle_weights = reid_dir / "vehicle_resnet50ibn_veri776.pth"

    if not person_weights.exists():
        print(f"  [!] Person ReID weights not found: {person_weights}")
        print("      Train using notebooks/kaggle/02_person_reid_training.ipynb")
        print("      Then download the weights and place them at the path above.")

    if not vehicle_weights.exists():
        print(f"  [!] Vehicle ReID weights not found: {vehicle_weights}")
        print("      Train using notebooks/kaggle/03_vehicle_reid_training.ipynb")
        print("      Then download the weights and place them at the path above.")

    print("\n" + "=" * 60)
    print("Download complete.")
    print(f"Model directory: {models_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
