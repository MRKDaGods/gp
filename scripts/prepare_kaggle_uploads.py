"""Prepare the Kaggle model weights upload ZIP.

The repo is cloned from GitHub automatically; CityFlowV2 is downloaded from
Google Drive automatically.  Only model weights need to be uploaded manually.

Creates:
  dist/kaggle/mtmc_weights.zip   — all model weights (detection + reid + tracker)

Usage:
    python scripts/prepare_kaggle_uploads.py

Output zip can be uploaded at: https://www.kaggle.com/datasets/new
Dataset slug to use: mtmc-weights
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist" / "kaggle"
DIST.mkdir(parents=True, exist_ok=True)


def build_weights(out: Path):
    """Zip: all model weights (detection + reid + tracker)."""
    models_dir = ROOT / "models"
    if not models_dir.exists():
        print(f"  ERROR: {models_dir} not found")
        return

    print(f"Building: {out.name}  (may take a minute...)")
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:  # already compressed
        for fp in sorted(models_dir.rglob("*")):
            if not fp.is_file():
                continue
            arc = fp.relative_to(ROOT).as_posix()
            size_mb = fp.stat().st_size / 1024 ** 2
            print(f"  + {arc}  ({size_mb:.0f} MB)")
            zf.write(fp, arc)
            total += fp.stat().st_size
    size_mb = out.stat().st_size / 1024 ** 2
    print(f"  ✓ {out.name}  ({size_mb:.0f} MB total)")


def build_data(out: Path):
    """Zip: CityFlowV2 raw AVI + metadata (seqinfo.ini, gt.txt)."""
    data_dir = ROOT / "data" / "raw" / "cityflowv2"
    if not data_dir.exists():
        print(f"  ERROR: {data_dir} not found — skipping data zip")
        return

    print(f"Building: {out.name}  (this may take a few minutes...)")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:  # AVIs already compressed
        for f in sorted(data_dir.rglob("*")):
            if not f.is_file():
                continue
            arc = "cityflowv2/" + f.relative_to(data_dir).as_posix()
            zf.write(f, arc)
    size_mb = out.stat().st_size / 1024 ** 2
    print(f"  ✓ {out.name}  ({size_mb:.0f} MB)")


def main():
    print(f"Output directory: {DIST}\n")
    build_weights(DIST / "mtmc_weights.zip")
    print(f"""
Done. Upload to Kaggle:
  {DIST / 'mtmc_weights.zip'}

Steps:
  1. Go to https://www.kaggle.com/datasets/new
  2. Upload mtmc_weights.zip
  3. Set dataset title: MTMC Weights
  4. Set slug to: mtmc-weights
  5. Attach it to notebook 10_mtmc_pipeline via Add Data -> Your Datasets
""")


if __name__ == "__main__":
    main()
