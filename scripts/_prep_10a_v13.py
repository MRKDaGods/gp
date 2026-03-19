"""Prepare 10a v13: download 09b output, upload to mtmc-weights, update 10a notebook.

Usage:
    python scripts/_prep_10a_v13.py [--check-only] [--skip-upload]

Steps:
  1. Check 09b kernel is COMPLETE
  2. Download transreid_cityflowv2_384_best.pth + metadata from 09b output
  3. Copy model to models/reid/ locally
  4. Upload new version of mrkdagods/mtmc-weights dataset (add 384px model)
  5. Update 10a notebook cell to use transreid_cityflowv2_384_best.pth
  6. Push 10a v13 to Kaggle
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_10A_DIR = ROOT / "notebooks" / "kaggle" / "10a_stages012"
NB_10A_FILE = NB_10A_DIR / "mtmc-10a-stages-0-2-tracking-reid-features.ipynb"
MODEL_LOCAL_DIR = ROOT / "models" / "reid"
MODEL_384_NAME = "transreid_cityflowv2_384_best.pth"
DL_DIR = ROOT / "data" / "outputs" / "09b_output"

KERNEL_09B = "mrkdagods/09b-vehicle-reid-384px-fine-tune"
DATASET_SLUG = "mrkdagods/mtmc-weights"


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    print(f"[{ts()}] $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kwargs)


def check_09b_status() -> str:
    r = run(["kaggle", "kernels", "status", KERNEL_09B], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    for word in ["COMPLETE", "RUNNING", "QUEUED", "ERROR", "CANCEL"]:
        if word in out:
            return word
    return out


def download_09b_output():
    DL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = DL_DIR / "exported_models" / MODEL_384_NAME
    if model_path.exists():
        print(f"[{ts()}] Model already downloaded: {model_path}")
        return model_path

    print(f"[{ts()}] Downloading 09b output to {DL_DIR} ...")
    r = run(["kaggle", "kernels", "output", KERNEL_09B, "-p", str(DL_DIR)])
    if r.returncode != 0:
        raise RuntimeError(f"kaggle kernels output failed: rc={r.returncode}")

    if not model_path.exists():
        # Search recursively
        candidates = list(DL_DIR.rglob(MODEL_384_NAME))
        if candidates:
            model_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Model {MODEL_384_NAME} not found in {DL_DIR}\n"
                f"Contents: {list(DL_DIR.rglob('*.pth'))}"
            )
    return model_path


def copy_model_locally(src: Path):
    dst = MODEL_LOCAL_DIR / MODEL_384_NAME
    if dst.exists():
        print(f"[{ts()}] Model already at {dst}")
        return dst
    print(f"[{ts()}] Copying {src} -> {dst}")
    MODEL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    size_mb = dst.stat().st_size / 1e6
    print(f"[{ts()}] Copied ({size_mb:.0f} MB)")
    return dst


def update_mtmc_weights_dataset():
    """Upload new version of mtmc-weights dataset including the 384px model."""
    # Check if kaggle datasets update is available
    # Strategy: use `kaggle datasets version` to add new version
    print(f"[{ts()}] Updating mtmc-weights dataset to add {MODEL_384_NAME} ...")

    # The models/ directory is the entire dataset source.
    # But re-uploading 750MB just to add one file is slow.
    # Instead, create a minimal dataset metadata and upload just the new file.
    # For simplicity, we'll update the dataset by uploading the models/reid/
    # directory as a new dataset version using the kaggle API.

    # Check dataset metadata
    dataset_meta = ROOT / "data" / "outputs" / "_mtmc_weights_meta"
    dataset_meta.mkdir(parents=True, exist_ok=True)

    # Write dataset metadata
    meta = {
        "title": "MTMC Weights",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(dataset_meta / "dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Copy models to dataset staging dir
    models_dst = dataset_meta / "models"
    if models_dst.exists():
        shutil.rmtree(str(models_dst))
    print(f"[{ts()}] Copying models/ to dataset staging dir ({models_dst}) ...")
    shutil.copytree(str(ROOT / "models"), str(models_dst))

    size_gb = sum(f.stat().st_size for f in models_dst.rglob("*") if f.is_file()) / 1e9
    print(f"[{ts()}] Dataset staging: {size_gb:.2f} GB")

    # Upload new version
    r = run(["kaggle", "datasets", "version",
             "-p", str(dataset_meta),
             "-m", f"Add {MODEL_384_NAME} (384px fine-tuned CityFlowV2 TransReID)"],
            cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[{ts()}] WARNING: dataset version upload failed (rc={r.returncode})")
        print("  You can manually upload via: kaggle datasets version -p <dir> -m 'Add 384px model'")
        return False
    print(f"[{ts()}] mtmc-weights dataset updated successfully")
    return True


def update_10a_notebook_for_384px_model():
    """Update 10a notebook to use transreid_cityflowv2_384_best.pth."""
    print(f"[{ts()}] Updating 10a notebook for 384px model ...")

    with open(NB_10A_FILE, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 7: Add 384px model to ESSENTIAL list
    cell7_idx = None
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "ESSENTIAL" in src and "transreid_cityflowv2_best.pth" in src:
            cell7_idx = i
            break

    if cell7_idx is None:
        print(f"[{ts()}] WARNING: Could not find cell 7 (ESSENTIAL list) in 10a")
    else:
        src = "".join(nb["cells"][cell7_idx]["source"])
        old_essential = '"models/reid/transreid_cityflowv2_best.pth",'
        new_essential = '"models/reid/transreid_cityflowv2_384_best.pth",'
        if old_essential in src and new_essential not in src:
            src = src.replace(
                old_essential,
                f'{old_essential}\n    {new_essential}'
            )
            lines = src.split("\n")
            nb["cells"][cell7_idx]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
            print(f"[{ts()}] Added {MODEL_384_NAME} to ESSENTIAL list in cell {cell7_idx}")

    # Cell 14: Add weights_path override for 384px model
    cell14_idx = None
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "run_pipeline.py" in src and "stages" in src and "0,1,2" in src:
            cell14_idx = i
            break

    if cell14_idx is None:
        print(f"[{ts()}] WARNING: Could not find cell 14 (run command) in 10a")
    else:
        src = "".join(nb["cells"][cell14_idx]["source"])
        # Check if 384px model override already exists
        override_str = f'"--override", "stage2.reid.vehicle.weights_path=models/reid/{MODEL_384_NAME}",'
        if override_str not in src:
            # Add before the closing bracket of cmd
            old_paren = '# v44: Lower detection threshold'
            if old_paren in src:
                new_paren = (
                    f'    # v49: Use 384px trained model (09b output)\n'
                    f'    "--override", "stage2.reid.vehicle.weights_path=models/reid/{MODEL_384_NAME}",\n'
                    f'    {old_paren}'
                )
                src = src.replace(old_paren, new_paren)
                lines = src.split("\n")
                nb["cells"][cell14_idx]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
                print(f"[{ts()}] Added weights_path override in cell {cell14_idx}")
            else:
                print(f"[{ts()}] WARNING: Insert point not found in cell {cell14_idx} (run command)")

    # Update version comment in notebook
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "RUN_NAME" in src and "v12" in src:
            new_src = src.replace("v12", "v13")
            lines = new_src.split("\n")
            nb["cells"][i]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
            print(f"[{ts()}] Updated RUN_NAME v12 -> v13 in cell {i}")
            break

    with open(NB_10A_FILE, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=True, indent=1)
    print(f"[{ts()}] 10a notebook saved")


def push_10a():
    """Push 10a v13 to Kaggle."""
    print(f"[{ts()}] Pushing 10a v13 to Kaggle ...")
    r = run(["kaggle", "kernels", "push", "-p", str(NB_10A_DIR)])
    if r.returncode != 0:
        raise RuntimeError(f"kaggle kernels push failed: rc={r.returncode}")
    print(f"[{ts()}] 10a v13 pushed successfully")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Only check 09b status, don't do anything")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip uploading model to mtmc-weights dataset")
    parser.add_argument("--skip-push", action="store_true",
                        help="Skip pushing 10a to Kaggle")
    args = parser.parse_args()

    # Step 1: Check 09b status
    print(f"[{ts()}] Checking 09b status ...")
    status = check_09b_status()
    print(f"[{ts()}] 09b status: {status}")

    if args.check_only:
        return

    if status not in ("COMPLETE", "SUCCESS"):
        print(f"[{ts()}] 09b is not COMPLETE yet (status: {status})")
        print("  Run this script again when 09b finishes.")
        sys.exit(1)

    # Step 2: Download 09b output
    model_src = download_09b_output()
    print(f"[{ts()}] Model downloaded: {model_src}")

    # Show metadata if available
    meta_candidates = list(DL_DIR.rglob("transreid_cityflowv2_384_metadata.json"))
    if meta_candidates:
        with open(meta_candidates[0]) as f:
            meta = json.load(f)
        m = meta.get("model", meta)
        mAP = m.get("best_mAP", m.get("mAP", "?"))
        epochs = m.get("epochs", "?")
        input_sz = m.get("input_size", "?")
        print(f"[{ts()}] Model stats: mAP={mAP}, epochs={epochs}, input_size={input_sz}")

    # Step 3: Copy model locally
    model_dst = copy_model_locally(model_src)

    # Step 4: Upload to mtmc-weights
    if not args.skip_upload:
        uploaded = update_mtmc_weights_dataset()
        if not uploaded:
            print(f"[{ts()}] Upload failed -- manual upload required before pushing 10a")
            if not args.skip_push:
                print(f"[{ts()}] Aborting. Re-run with --skip-upload after manual upload.")
                sys.exit(1)

    # Step 5: Update 10a notebook
    update_10a_notebook_for_384px_model()

    # Step 6: Push 10a v13
    if not args.skip_push:
        push_10a()
        print(f"\n[{ts()}] DONE! 10a v13 pushed with 384px trained model.")
        print(f"  Monitor: kaggle kernels status mrkdagods/mtmc-10a-stages-0-2-tracking-reid-features")
        print(f"  Chain:   python scripts/_kaggle_pipeline.py --start-from 10a")
    else:
        print(f"\n[{ts()}] 10a notebook updated but NOT pushed (--skip-push was set)")


if __name__ == "__main__":
    main()
