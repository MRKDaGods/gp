"""Generate the Kaggle notebook for running the MTMC pipeline end-to-end.

Strategy:
  - Repo cloned from GitHub (no source upload needed)
  - CityFlowV2 downloaded from Google Drive via download_datasets.py
  - Only model weights need to be uploaded as a Kaggle dataset

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

REPO_URL = "https://github.com/MRKDaGods/gp.git"


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
- **Stage 0** — Frame extraction from CityFlowV2 videos
- **Stage 1** — Vehicle detection + BotSort tracking (per camera, `track_buffer=450`)
- **Stage 2** — ReID feature extraction (TransReID 768D + OSNet 512D → PCA 256D ensemble)
- **Stage 3** — FAISS indexing
- **Stage 4** — Cross-camera association (AQE + Louvain graph + forensic report)
- **Stage 5** — Evaluation (IDF1, MOTA, HOTA)

### Setup (only one Kaggle dataset needed)
| What | How |
|---|---|
| **Source code** | Cloned from GitHub automatically |
| **CityFlowV2 dataset** | Downloaded from Google Drive automatically |
| **Model weights** | Attach dataset `mtmc-weights` (upload once from laptop) |

Attach `mtmc-weights` via **Add Data → Your Datasets → mtmc-weights** before running.""", "aa01"))

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
CELLS.append(cell_md("""## 1. Clone Repo & Install Dependencies""", "aa03"))

CELLS.append(cell_code("""REPO_URL = "https://github.com/MRKDaGods/gp.git"
WORK_DIR = Path("/kaggle/working")
PROJECT  = WORK_DIR / "gp"

# ── Clone repo ────────────────────────────────────────────────────────────────
if not PROJECT.exists():
    print(f"Cloning {REPO_URL} ...")
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT)])
else:
    print("Repo already cloned, pulling latest ...")
    subprocess.check_call(["git", "-C", str(PROJECT), "pull", "--ff-only"])

os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))
print(f"\\n✓ Repo ready at {PROJECT}")""", "aa04"))

CELLS.append(cell_code("""def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# ── boxmot (BotSort tracker) ──────────────────────────────────────────────────
pip("boxmot")

# ── torchreid (OSNet / ResNet-IBN) ────────────────────────────────────────────
try:
    import torchreid
    print("torchreid already available")
except ImportError:
    print("Installing torchreid...")
    pip("git+https://github.com/KaiyangZhou/deep-person-reid.git")

# ── FAISS (GPU preferred) ─────────────────────────────────────────────────────
try:
    import faiss
    print(f"faiss already available ({faiss.__version__})")
except ImportError:
    try:
        pip("faiss-gpu")
    except Exception:
        pip("faiss-cpu")

# ── Other utilities ───────────────────────────────────────────────────────────
pip("gdown", "loguru", "omegaconf", "rich", "networkx>=3.1", "click")

# ── Install package itself (no extra dep fetch) ───────────────────────────────
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
    cwd=str(PROJECT),
)

print("\\n✓ All dependencies installed")""", "aa05"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 2. Mount Model Weights

The only thing uploaded from your laptop (dataset slug: `mtmc-weights`).
The models folder is mounted at `/kaggle/input/datasets/mrkdagods/mtmc-weights/models`.""", "aa06"))

CELLS.append(cell_code("""WEIGHTS_INPUT = Path("/kaggle/input/datasets/mrkdagods/mtmc-weights/models")

assert WEIGHTS_INPUT.exists(), (
    "Dataset 'mtmc-weights' not found at expected path:\\n"
    f"  {WEIGHTS_INPUT}\\n"
    "Make sure it is attached via Add Data -> Your Datasets -> mtmc-weights"
)

MODELS_DST = PROJECT / "models"

# Symlink models/ → the mounted dataset folder (avoids copying ~750 MB)
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
missing = [p for p in ESSENTIAL if not (PROJECT / p).exists()]
if missing:
    print("\\n⚠  Missing essential weights:")
    for m in missing:
        print(f"   {m}")
    raise FileNotFoundError("Fix missing weights before continuing.")
else:
    print("✓ All essential v4 weights present")""", "aa07"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 3. Download CityFlowV2 Dataset (~17 GB)

Downloaded from Google Drive to `/tmp` (not `/kaggle/working` — working disk is only 20 GB).
The `data/raw` folder is symlinked to `/tmp/datasets` so the pipeline finds it at the expected path.

Download + extraction peak usage: ~34 GB in `/tmp`. Kaggle `/tmp` is typically 100+ GB on P100/T4.""", "aa08"))

CELLS.append(cell_code("""import shutil as _shutil

# ── Disk space check ─────────────────────────────────────────────────────────
for mount in ["/tmp", "/kaggle/working"]:
    total, used, free = _shutil.disk_usage(mount)
    print(f"{mount:20s}  {free/1024**3:.1f} GB free  /  {total/1024**3:.1f} GB total")

# ── Symlink data/raw → /tmp/datasets so all downloads land on the big disk ───
DATA_RAW_PARENT = PROJECT / "data" / "raw"
TMP_DATA = Path("/tmp/datasets")
TMP_DATA.mkdir(parents=True, exist_ok=True)

if not DATA_RAW_PARENT.is_symlink():
    if DATA_RAW_PARENT.exists():
        shutil.rmtree(DATA_RAW_PARENT)
    # Ensure parent dir exists before symlinking
    DATA_RAW_PARENT.parent.mkdir(parents=True, exist_ok=True)
    DATA_RAW_PARENT.symlink_to(TMP_DATA)
    print(f"✓ data/raw → {TMP_DATA}")
else:
    print(f"data/raw already symlinked → {DATA_RAW_PARENT.resolve()}")

DATA_RAW = TMP_DATA / "cityflowv2"
DATA_OUT = PROJECT / "data" / "outputs"
DATA_OUT.mkdir(parents=True, exist_ok=True)

import re
CAM_RE = re.compile(r"^S\\d{2}_c\\d{3}$")

# Download only if not already present
if DATA_RAW.exists() and any(CAM_RE.match(d.name) for d in DATA_RAW.iterdir() if d.is_dir()):
    cams = [d.name for d in DATA_RAW.iterdir() if d.is_dir() and CAM_RE.match(d.name)]
    print(f"\\n✓ CityFlowV2 already downloaded: {len(cams)} cameras")
else:
    print("\\nDownloading CityFlowV2 from Google Drive (~17 GB)...")
    print("This will take 20-40 minutes. The archive is extracted and deleted to save space.")
    subprocess.check_call(
        [sys.executable, "scripts/download_datasets.py", "--dataset", "cityflowv2"],
        cwd=str(PROJECT),
    )
    cams = [d.name for d in DATA_RAW.iterdir() if d.is_dir() and CAM_RE.match(d.name)]
    print(f"\\n✓ CityFlowV2 ready: {sorted(cams)}")

print(f"\\nDataset path: {DATA_RAW}")""", "aa09"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 4. Configure & Run Pipeline

Expected GPU times on P100 (16 GB):
| Stage | Description | Time |
|---|---|---|
| 0 | Frame extraction (10fps from ~17GB videos) | ~20 min |
| 1 | Detection + BotSort (`track_buffer=450`) | ~45 min |
| 2 | TransReID 768D + OSNet 512D → PCA 256D | ~20 min |
| 3 | FAISS indexing | ~1 min |
| 4 | Cross-camera association (AQE + Louvain) | ~5 min |
| 5 | Evaluation (IDF1, MOTA, HOTA) | ~1 min |
| **Total** | *(excluding 40-min data download)* | **~90 min** |""", "aa10"))

CELLS.append(cell_code("""from datetime import datetime

RUN_NAME = f"run_kaggle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR  = DATA_OUT / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
STAGES = "0,1,2,3,4,5"

print(f"Run name : {RUN_NAME}")
print(f"Run dir  : {RUN_DIR}")
print(f"Stages   : {STAGES}")
print(f"Device   : {DEVICE}")""", "aa11"))

CELLS.append(cell_code("""os.chdir(str(PROJECT))

cmd = [
    sys.executable, "scripts/run_pipeline.py",
    "--config", "configs/default.yaml",
    "--dataset-config", "configs/datasets/cityflowv2.yaml",
    "--stages", STAGES,
    "--override", f"project.run_name={RUN_NAME}",
    "--override", f"project.output_dir={DATA_OUT}",
]

print("Command:", " ".join(str(c) for c in cmd))
print("=" * 70)

t0 = time.time()
result = subprocess.run(cmd, cwd=str(PROJECT))
elapsed = time.time() - t0

print("=" * 70)
if result.returncode == 0:
    print(f"✓ Pipeline completed in {elapsed/60:.1f} min")
else:
    print(f"✗ Pipeline FAILED (code {result.returncode}) after {elapsed/60:.1f} min")
    sys.exit(result.returncode)""", "aa12"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 5. Results""", "aa13"))

CELLS.append(cell_code("""run_dir   = DATA_OUT / RUN_NAME
stage5_dir = run_dir / "stage5"

def _pct(v):
    return f"{v:.1%}" if isinstance(v, float) else str(v)

# Per-camera metrics
metrics_files = list(stage5_dir.glob("metrics_*.json")) if stage5_dir.exists() else []
if metrics_files:
    print("=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    for mf in sorted(metrics_files):
        m = json.loads(mf.read_text())
        m = m.get("metrics", m)
        cam = mf.stem.replace("metrics_", "")
        print(f"  {cam:12s}  IDF1={_pct(m.get('IDF1',m.get('idf1','?')))}"
              f"  MOTA={_pct(m.get('MOTA',m.get('mota','?')))}"
              f"  HOTA={_pct(m.get('HOTA',m.get('hota','?')))}"
              f"  IDsw={m.get('ID_Sw',m.get('id_switches','?'))}")

# Global summary
for fname in ["summary.json", "evaluation_report.json"]:
    sf = stage5_dir / fname
    if sf.exists():
        s = json.loads(sf.read_text())
        print("-" * 65)
        print("  GLOBAL:")
        for k in ["IDF1", "MOTA", "HOTA", "ID_Sw", "idf1", "mota", "hota", "id_switches"]:
            v = s.get(k)
            if v is not None:
                print(f"    {k}: {_pct(v)}")
        break

# Forensic report
fr_path = run_dir / "stage4" / "forensic_report.json"
if fr_path.exists():
    fr = json.loads(fr_path.read_text())
    print(f"\\nForensic report: {len(fr.get('trajectories', []))} global trajectories")""", "aa14"))

# ──────────────────────────────────────────────────────────────────────────────
CELLS.append(cell_md("""## 6. Hyperparameter Scan (optional)

Sweep Stage 4 parameters on the completed run without repeating Stages 0–3.""", "aa15"))

CELLS.append(cell_code("""# Uncomment to run a sweep after Stage 4 completes

# for scan in ["aqe_k", "sim_thresh", "louvain_res"]:
#     print(f"\\n{'='*50}\\nScanning: {scan}")
#     subprocess.run([
#         sys.executable, "scripts/scan_stage4_params.py",
#         "--run", RUN_NAME,
#         "--scan", scan,
#         "--output-dir", str(DATA_OUT),
#     ], cwd=str(PROJECT))

print("Uncomment the scan loop above to sweep Stage 4 parameters.")
print(f"Current run: {RUN_NAME}")""", "aa16"))

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
