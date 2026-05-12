"""
Download all available Scene 1 cameras from the AI City 2023 dataset.
Places videos in: dataset/S01/c00X/vdo.avi
Also downloads ground-truth, ROI masks, and calibration per camera.
"""
import os
import shutil
import subprocess
from pathlib import Path

os.environ["KAGGLE_USERNAME"] = "mrkdagods"
os.environ["KAGGLE_KEY"] = "b7b65632a8b882d35a6fbe8b074e0a71"

DATASET = "thanhnguyenle/data-aicity-2023-track-2"
DEST = Path("c:/Users/seift/Downloads/gp/dataset")
SCENE = "S01"
CAMERAS = ["c001", "c002", "c003", "c004", "c005"]

KAGGLE = r"c:/Users/seift/Downloads/gp/.venv/Scripts/kaggle.exe"

def download_file(kaggle_path, dest_folder):
    """Download a single file. Returns True if successful."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [KAGGLE, "datasets", "download", DATASET,
         "-f", kaggle_path,
         "-p", str(dest_folder),
         "--unzip"],
        capture_output=True, text=True
    )
    return "404" not in result.stderr and "404" not in result.stdout

print("=" * 60)
print(f"Downloading {SCENE} cameras from AI City 2023 dataset")
print("=" * 60)

found_cameras = []

for cam in CAMERAS:
    print(f"\n[{cam}] Checking video...")
    video_path = f"train/{SCENE}/{cam}/vdo.avi"
    cam_dest = DEST / SCENE / cam

    ok = download_file(video_path, cam_dest)
    if not ok:
        print(f"  -> {cam} not in dataset, skipping.")
        continue

    print(f"  -> vdo.avi downloaded!")
    found_cameras.append(cam)

    # Also grab metadata files
    for extra in ["gt/gt.txt", "roi.jpg", "calibration.txt", "seqinfo.ini"]:
        full_path = f"train/{SCENE}/{cam}/{extra}"
        extra_dest = cam_dest / Path(extra).parent
        result = download_file(full_path, extra_dest)
        if result:
            print(f"  -> {extra} downloaded.")

print("\n" + "=" * 60)
print(f"Downloaded cameras: {found_cameras}")
print(f"Dataset location:   {DEST / SCENE}")
print("=" * 60)
print("\nDone! Open http://localhost:3000 and upload videos from:")
for cam in found_cameras:
    print(f"  {DEST / SCENE / cam / 'vdo.avi'}")
