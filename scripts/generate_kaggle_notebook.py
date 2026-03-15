"""Generate the Kaggle notebook for running the MTMC pipeline end-to-end.

Usage:
    python scripts/generate_kaggle_notebook.py

Produces:
    notebooks/kaggle/10_mtmc_pipeline/10_mtmc_pipeline.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "notebooks/kaggle/10_mtmc_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cell_md(source: str, cell_id: str) -> dict:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


def cell_code(source: str, cell_id: str) -> dict:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


CELLS = []

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""# MTMC Vehicle Tracking — Full Pipeline (v4)

**Multi-Camera Multi-Target tracking system on CityFlowV2.**

Pipeline stages:
- **Stage 0** — Frame extraction (skip if pre-extracted dataset attached)
- **Stage 1** — Vehicle detection + BotSort tracking (per camera)
- **Stage 2** — ReID feature extraction (TransReID 768D + OSNet 512D → PCA 256D)
- **Stage 3** — FAISS indexing
- **Stage 4** — Cross-camera association (AQE + Louvain graph)
- **Stage 5** — Evaluation (IDF1, MOTA, HOTA)

### Required Kaggle Datasets (attach before running)
| Dataset slug | Contents |
|---|---|
| `mtmc-tracker-source` | src/, scripts/, configs/, setup.py |
| `mtmc-weights` | models/detection/, models/reid/, models/tracker/ |
| `cityflowv2-mtmc` | data/raw/cityflowv2/ — 6 AVI files + metadata |
| `cityflowv2-stage0` *(optional)* | Pre-extracted 10fps JPEG frames (9.6 GB, skips Stage 0) |

If `cityflowv2-stage0` is attached, Stage 0 is skipped automatically.""", "aa01"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_code("""import os, sys, subprocess, shutil, json, time
from pathlib import Path
import torch

print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({p.total_memory/1024**3:.1f} GB)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\\nUsing device: {DEVICE}")""", "aa02"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 1. Install Dependencies""", "aa03"))

CELLS.append(cell_code("""# ── Core tracking / ReID / indexing dependencies ──
# boxmot    : BotSort tracker
# torchreid : OSNet ReID backbone
# faiss-gpu : fast approximate nearest-neighbour search on GPU
# networkx  : graph construction (Louvain via nx.algorithms.community)

import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# boxmot (includes BotSort + appearance models)
pip("boxmot")

# torchreid (OSNet / ResNet-IBN backbones)
try:
    import torchreid
    print("torchreid already available")
except ImportError:
    print("Installing torchreid...")
    pip("git+https://github.com/KaiyangZhou/deep-person-reid.git")

# FAISS — prefer GPU build on Kaggle T4/P100
try:
    import faiss
    print(f"faiss already available ({faiss.__version__})")
except ImportError:
    try:
        pip("faiss-gpu")
    except Exception:
        pip("faiss-cpu")   # fallback if CUDA version mismatch

# Light utilities
pip("loguru", "omegaconf", "rich", "networkx>=3.1", "click")

print("\\n✓ All dependencies installed")""", "aa04"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 2. Setup: Source Code""", "aa05"))

CELLS.append(cell_code("""WORK_DIR = Path("/kaggle/working")
SOURCE_INPUT = Path("/kaggle/input/mtmc-tracker-source")

assert SOURCE_INPUT.exists(), (
    "Dataset 'mtmc-tracker-source' not attached!\\n"
    "Add it via: Add Data → Your Datasets → mtmc-tracker-source"
)

# Copy source tree to /kaggle/working so relative config paths work
for item in ["src", "scripts", "configs", "setup.py", "pyproject.toml"]:
    src = SOURCE_INPUT / item
    if not src.exists():
        print(f"  WARNING: {item} not found in source dataset")
        continue
    dst = WORK_DIR / item
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"  ✓ {item}")

# Install the package in editable mode (no extra deps fetch)
os.chdir(str(WORK_DIR))
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"]
)
print("\\n✓ mtmc-tracker installed in /kaggle/working")""", "aa06"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 3. Setup: Model Weights""", "aa07"))

CELLS.append(cell_code("""WEIGHTS_INPUT = Path("/kaggle/input/mtmc-weights")

assert WEIGHTS_INPUT.exists(), (
    "Dataset 'mtmc-weights' not attached!\\n"
    "Add it via: Add Data → Your Datasets → mtmc-weights"
)

MODELS_DST = WORK_DIR / "models"

# Symlink the whole models/ directory for efficiency (avoids copying ~1.5 GB)
if MODELS_DST.is_symlink():
    MODELS_DST.unlink()
elif MODELS_DST.exists():
    shutil.rmtree(MODELS_DST)

MODELS_DST.symlink_to(WEIGHTS_INPUT)
print(f"✓ models/ → {WEIGHTS_INPUT}")

# Verify essential v4 weights
ESSENTIAL = [
    "models/detection/yolo26m.pt",
    "models/reid/transreid_cityflowv2_best.pth",
    "models/reid/vehicle_osnet_veri776.pth",
    "models/tracker/osnet_x0_25_msmt17.pt",
]
missing = [p for p in ESSENTIAL if not (WORK_DIR / p).exists()]
if missing:
    print("\\n⚠ Missing essential weights:")
    for m in missing:
        print(f"  {m}")
else:
    print("✓ All essential v4 weights present")""", "aa08"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 4. Setup: CityFlowV2 Data

Checks for:
1. Pre-extracted Stage 0 frames (`cityflowv2-stage0` dataset) — fastest, skips Stage 0
2. Raw AVI videos (`cityflowv2-mtmc` dataset) — Stage 0 extracts frames (~15 min on T4)""", "aa09"))

CELLS.append(cell_code("""import re

STAGE0_INPUT = Path("/kaggle/input/cityflowv2-stage0")
RAW_INPUT    = Path("/kaggle/input/cityflowv2-mtmc")

DATA_RAW  = WORK_DIR / "data/raw/cityflowv2"
DATA_OUT  = WORK_DIR / "data/outputs"
DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

CAM_RE = re.compile(r"^S\\d{2}_c\\d{3}$")

USE_PREEXTRACTED = False

# ── Option A: pre-extracted stage0 frames ───────────────────────────────────
if STAGE0_INPUT.exists():
    # Count camera dirs with extracted frames
    cams = [d for d in STAGE0_INPUT.iterdir()
            if d.is_dir() and CAM_RE.match(d.name)]
    if cams:
        USE_PREEXTRACTED = True
        STAGE0_FRAMES_DIR = STAGE0_INPUT
        print(f"✓ Pre-extracted stage0 found: {len(cams)} cameras at {STAGE0_INPUT}")
        print(f"  Cameras: {sorted(d.name for d in cams)}")
    else:
        # Maybe frames are nested one level deeper
        for sub in STAGE0_INPUT.iterdir():
            if sub.is_dir():
                cams = [d for d in sub.iterdir() if d.is_dir() and CAM_RE.match(d.name)]
                if cams:
                    USE_PREEXTRACTED = True
                    STAGE0_FRAMES_DIR = sub
                    print(f"✓ Pre-extracted stage0 found (nested): {STAGE0_FRAMES_DIR}")
                    break

# ── Option B: raw AVI videos ─────────────────────────────────────────────────
if not USE_PREEXTRACTED:
    assert RAW_INPUT.exists(), (
        "Neither 'cityflowv2-stage0' nor 'cityflowv2-mtmc' dataset attached!\\n"
        "Attach at least one of them via Add Data."
    )
    # Symlink raw data to expected path
    if DATA_RAW.is_symlink():
        DATA_RAW.unlink()
    elif DATA_RAW.exists():
        shutil.rmtree(DATA_RAW)
    DATA_RAW.symlink_to(RAW_INPUT)
    # Verify AVI files
    avis = list(DATA_RAW.rglob("vdo.avi"))
    assert avis, f"No vdo.avi files found in {RAW_INPUT}!"
    print(f"✓ Raw CityFlowV2 videos: {len(avis)} cameras at {RAW_INPUT}")
    print(f"  Stage 0 will extract frames (~15 min on T4)")

print(f"\\nMode: {'pre-extracted stage0' if USE_PREEXTRACTED else 'extract from raw videos'}")""", "aa10"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 5. Configure Run""", "aa11"))

CELLS.append(cell_code("""from datetime import datetime

RUN_NAME  = f"run_kaggle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR   = DATA_OUT / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Decide which stages to run
if USE_PREEXTRACTED:
    # Pre-link stage0 dir so the pipeline finds it
    stage0_dst = RUN_DIR / "stage0"
    if stage0_dst.is_symlink():
        stage0_dst.unlink()
    elif stage0_dst.exists():
        shutil.rmtree(stage0_dst)
    stage0_dst.symlink_to(STAGE0_FRAMES_DIR)
    STAGES = "1,2,3,4,5"
    print(f"✓ stage0 symlinked → {STAGE0_FRAMES_DIR}")
else:
    STAGES = "0,1,2,3,4,5"

print(f"Run name : {RUN_NAME}")
print(f"Run dir  : {RUN_DIR}")
print(f"Stages   : {STAGES}")
print(f"Device   : {DEVICE}")""", "aa12"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 6. Run Pipeline

Runs the full MTMC tracking pipeline. Progress is streamed live to the output.

Expected GPU times on T4 (16 GB):
| Stage | Description | Time |
|---|---|---|
| 0 | Frame extraction | ~15 min |
| 1 | Detection + BotSort | ~45 min |
| 2 | ReID feature extraction | ~20 min |
| 3 | FAISS indexing | ~1 min |
| 4 | Cross-camera association | ~5 min |
| 5 | Evaluation | ~1 min |
| **Total** | | **~90 min (with pre-extracted) / ~105 min** |""", "aa13"))

CELLS.append(cell_code("""import subprocess, sys, time, os
from pathlib import Path

os.chdir(str(WORK_DIR))

cmd = [
    sys.executable, "scripts/run_pipeline.py",
    "--config", "configs/default.yaml",
    "--dataset-config", "configs/datasets/cityflowv2.yaml",
    "--stages", STAGES,
    "--override", f"project.run_name={RUN_NAME}",
    "--override", f"project.output_dir={DATA_OUT}",
]

print("Starting pipeline:", " ".join(str(c) for c in cmd))
print("=" * 70)

t0 = time.time()
result = subprocess.run(cmd, cwd=str(WORK_DIR))
elapsed = time.time() - t0

print("=" * 70)
if result.returncode == 0:
    print(f"✓ Pipeline completed in {elapsed/60:.1f} min")
else:
    print(f"✗ Pipeline FAILED (return code {result.returncode}) after {elapsed/60:.1f} min")
    sys.exit(result.returncode)""", "aa14"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 7. Results""", "aa15"))

CELLS.append(cell_code("""import json
from pathlib import Path

run_dir = DATA_OUT / RUN_NAME
stage5_dir = run_dir / "stage5"

# ── Print metrics summary ─────────────────────────────────────────────────────
metrics_files = list(stage5_dir.glob("metrics_*.json")) if stage5_dir.exists() else []
if metrics_files:
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for mf in sorted(metrics_files):
        data = json.loads(mf.read_text())
        cam = mf.stem.replace("metrics_", "")
        m = data.get("metrics", data)
        idf1 = m.get("IDF1", m.get("idf1", "N/A"))
        mota = m.get("MOTA", m.get("mota", "N/A"))
        hota = m.get("HOTA", m.get("hota", "N/A"))
        ids  = m.get("ID_Sw", m.get("id_switches", "N/A"))
        print(f"  {cam:12s}  IDF1={idf1:6.1%}  MOTA={mota:6.1%}  HOTA={hota:6.1%}  IDsw={ids}")

    # Try overall / global summary
    summary_file = stage5_dir / "summary.json"
    if summary_file.exists():
        s = json.loads(summary_file.read_text())
        print("-" * 60)
        print("  GLOBAL:")
        for k in ["IDF1", "MOTA", "HOTA", "ID_Sw"]:
            v = s.get(k, s.get(k.lower(), "N/A"))
            if isinstance(v, float):
                print(f"    {k}: {v:.1%}")
            else:
                print(f"    {k}: {v}")
else:
    # Fallback: look for any txt/json results
    for f in sorted(run_dir.rglob("*.json"))[:10]:
        print(f"  {f.relative_to(run_dir)}")
    print("\\nRun the cell above first to generate results.")

# ── Print forensic report summary ─────────────────────────────────────────────
forensic_file = run_dir / "stage4" / "forensic_report.json"
if forensic_file.exists():
    fr = json.loads(forensic_file.read_text())
    print(f"\\nForensic report: {len(fr.get('trajectories', []))} global trajectories")""", "aa16"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 8. Hyperparameter Scan (optional)

Run a post-hoc sweep over Stage 4 parameters to find the best configuration
without re-running Stages 0–3.""", "aa17"))

CELLS.append(cell_code("""# Example: sweep AQE k values on the completed run
# Uncomment and run after Stage 4 completes

# scan_types = ["aqe_k", "sim_thresh", "louvain_res"]
# for scan in scan_types:
#     print(f"\\n{'='*50}")
#     print(f"Scanning: {scan}")
#     subprocess.run([
#         sys.executable, "scripts/scan_stage4_params.py",
#         "--run", RUN_NAME,
#         "--scan", scan,
#         "--output-dir", str(DATA_OUT),
#     ], cwd=str(WORK_DIR))

print("Uncomment the scan loop above to run hyperparameter sweeps.")
print(f"Current run: {RUN_NAME}")""", "aa18"))

# ──────────────────────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": CELLS,
}

out_path = OUTPUT_DIR / "10_mtmc_pipeline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=True)

print(f"✓ Notebook written to: {out_path}")
print(f"  Cells: {len(CELLS)}")
