"""Prepare the three Kaggle dataset upload ZIPs.

Creates three zip archives ready to upload:
  dist/kaggle/01_mtmc_source.zip     — source code + configs
  dist/kaggle/02_mtmc_weights.zip    — model weights (v4 essentials)
  dist/kaggle/03_cityflowv2_raw.zip  — CityFlowV2 raw AVI + metadata

Usage:
    python scripts/prepare_kaggle_uploads.py [--all] [--source] [--weights] [--data]
    python scripts/prepare_kaggle_uploads.py --weights  # weights only (largest, slowest)

Output zips can be uploaded at: https://www.kaggle.com/datasets/new
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist" / "kaggle"
DIST.mkdir(parents=True, exist_ok=True)


def _add_dir(zf: zipfile.ZipFile, src: Path, arcname_prefix: str, exts: set[str] | None = None):
    """Add all files under `src` to the zip, prefixed with arcname_prefix/."""
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        if "__pycache__" in f.parts or ".egg-info" in str(f):
            continue
        if exts and f.suffix.lower() not in exts:
            continue
        arc = arcname_prefix + "/" + f.relative_to(src).as_posix()
        zf.write(f, arc)


def build_source(out: Path):
    """Zip: src/, scripts/, configs/, setup.py, pyproject.toml"""
    print(f"Building: {out.name}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        for d, prefix in [
            (ROOT / "src",     "src"),
            (ROOT / "scripts", "scripts"),
            (ROOT / "configs", "configs"),
        ]:
            if d.exists():
                _add_dir(zf, d, prefix)
            else:
                print(f"  WARNING: {d} not found")
        for fname in ["setup.py", "pyproject.toml", "README.md"]:
            fp = ROOT / fname
            if fp.exists():
                zf.write(fp, fname)
    size_mb = out.stat().st_size / 1024 ** 2
    print(f"  ✓ {out.name}  ({size_mb:.1f} MB)")


def build_weights(out: Path):
    """Zip: essential model weights for v4 config."""
    essential = [
        ROOT / "models" / "detection" / "yolo26m.pt",
        ROOT / "models" / "reid"      / "transreid_cityflowv2_best.pth",
        ROOT / "models" / "reid"      / "transreid_veri_best.pth",
        ROOT / "models" / "reid"      / "vehicle_osnet_veri776.pth",
        ROOT / "models" / "tracker"   / "osnet_x0_25_msmt17.pt",
    ]
    missing = [p for p in essential if not p.exists()]
    if missing:
        print("  WARNING — missing weight files (they will be absent from zip):")
        for m in missing:
            print(f"    {m.relative_to(ROOT)}")

    print(f"Building: {out.name}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:  # models = already compressed
        for fp in essential:
            if not fp.exists():
                continue
            # Preserve structure: models/reid/transreid_*.pth etc.
            arc = fp.relative_to(ROOT).as_posix()
            print(f"  + {arc}  ({fp.stat().st_size/1024**2:.0f} MB)")
            zf.write(fp, arc)
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
    parser = argparse.ArgumentParser(description="Prepare Kaggle upload ZIPs")
    parser.add_argument("--all",     action="store_true", help="Build all three zips")
    parser.add_argument("--source",  action="store_true", help="Build source code zip only")
    parser.add_argument("--weights", action="store_true", help="Build weights zip only")
    parser.add_argument("--data",    action="store_true", help="Build CityFlowV2 data zip only")
    args = parser.parse_args()

    if not any([args.all, args.source, args.weights, args.data]):
        parser.print_help()
        print("\nTip: run --all to build everything")
        sys.exit(0)

    do = {
        "source":  args.all or args.source,
        "weights": args.all or args.weights,
        "data":    args.all or args.data,
    }

    print(f"Output directory: {DIST}\n")

    if do["source"]:
        build_source(DIST / "01_mtmc_source.zip")
    if do["weights"]:
        build_weights(DIST / "02_mtmc_weights.zip")
    if do["data"]:
        build_data(DIST / "03_cityflowv2_raw.zip")

    print("\nDone. Upload these files to Kaggle:")
    for z in sorted(DIST.glob("*.zip")):
        print(f"  {z}")
    print("""
Kaggle dataset names to use (match kernel-metadata.json):
  01_mtmc_source.zip   → Dataset slug: mtmc-tracker-source
  02_mtmc_weights.zip  → Dataset slug: mtmc-weights
  03_cityflowv2_raw.zip → Dataset slug: cityflowv2-mtmc
""")


if __name__ == "__main__":
    main()
