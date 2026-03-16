"""Generates 3 chained Kaggle notebooks for the MTMC pipeline.

  10a  →  stages 0-2  (heavy GPU work, ~90 min)  →  saves checkpoint.tar.gz
  10b  →  stage 3     (FAISS indexing, ~1 min)   →  mounts 10a output
  10c  →  stages 4-5  (association + eval, ~6 min)→  mounts 10b output (iteration loop)

Each notebook is self-contained: clones repo, installs deps, then picks up from
the previous notebook's /kaggle/working/checkpoint.tar.gz.

Usage:
    python scripts/generate_kaggle_notebook.py

Produces (one folder per notebook, each with .ipynb + kernel-metadata.json):
    notebooks/kaggle/10a_stages012/
    notebooks/kaggle/10b_stage3/
    notebooks/kaggle/10c_stages45/
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/MRKDaGods/gp.git"
OWNER = "mrkdagods"

# Kaggle kernel slugs — must match what Kaggle derives from the title
# Rule: lowercase, replace non-alnum with hyphen, collapse hyphens
# "MTMC 10a - Stages 0-2 (Tracking + ReID Features)" → mtmc-10a-stages-0-2-tracking-reid-features
# "MTMC 10b - Stage 3 (FAISS Indexing)"              → mtmc-10b-stage-3-faiss-indexing
# "MTMC 10c - Stages 4-5 (Association + Eval)"       → mtmc-10c-stages-4-5-association-eval
SLUG_10A = "mtmc-10a-stages-0-2-tracking-reid-features"
SLUG_10B = "mtmc-10b-stage-3-faiss-indexing"
SLUG_10C = "mtmc-10c-stages-4-5-association-eval"
SLUG_GT  = "mtmc-gt-annotations"    # dataset: mrkdagods/mtmc-gt-annotations

NB_ROOT = Path(__file__).parent.parent / "notebooks" / "kaggle"

BENCHMARK_CAMERAS = ["S01_c001", "S01_c002", "S01_c003", "S02_c006", "S02_c007", "S02_c008"]


# ---- notebook / cell helpers ------------------------------------------------

def _src(code: str) -> list[str]:
    lines = code.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(source: str, cid: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": _src(source)}


def code(source: str, cid: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {},
        "outputs": [],
        "source": _src(source),
    }


def make_notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def make_metadata(slug: str, title: str, kernel_sources=None, dataset_sources=None) -> dict:
    return {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": kernel_sources or [],
    }


def write_notebook(cells, out_dir, slug, title, kernel_sources=None, dataset_sources=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / f"{slug}.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=1, ensure_ascii=True)

    meta_path = out_dir / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(make_metadata(slug, title, kernel_sources=kernel_sources,
                                dataset_sources=dataset_sources), f, indent=2)

    print(f"  {slug}: {len(cells)} cells -> {nb_path}")


# ---- shared code blocks -----------------------------------------------------

SETUP_ENV = """\
import os, sys, subprocess, shutil, json, time, tarfile
from pathlib import Path
import torch

print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({p.total_memory/1024**3:.1f} GB)")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\\nUsing device: {DEVICE}")\
"""

CLONE_REPO = f"""\
REPO_URL = "{REPO_URL}"
WORK_DIR = Path("/kaggle/working")
PROJECT  = WORK_DIR / "gp"

if not PROJECT.exists():
    print(f"Cloning {{REPO_URL}} ...")
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT)])
else:
    print("Repo already present, pulling latest ...")
    subprocess.check_call(["git", "-C", str(PROJECT), "pull", "--ff-only"])

os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))
print(f"\\n\\u2713 Repo ready at {{PROJECT}}")\
"""

INSTALL_FULL = """\
def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("ultralytics")
pip("boxmot")

try:
    import torchreid; print("torchreid ok")
except ImportError:
    pip("git+https://github.com/KaiyangZhou/deep-person-reid.git")

try:
    import faiss; print(f"faiss ok ({faiss.__version__})")
except ImportError:
    try: pip("faiss-gpu")
    except Exception: pip("faiss-cpu")

pip("timm", "motmetrics")
pip("gdown", "loguru", "omegaconf", "rich", "networkx>=3.1", "click")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
                      cwd=str(PROJECT))
print("\\n\\u2713 All dependencies installed")\
"""

INSTALL_LIGHT = """\
def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

try:
    import faiss; print(f"faiss ok ({faiss.__version__})")
except ImportError:
    try: pip("faiss-gpu")
    except Exception: pip("faiss-cpu")

try:
    import trackeval; print("trackeval ok")
except ImportError:
    pip("git+https://github.com/JonathonLuiten/TrackEval.git")

pip("motmetrics", "loguru", "omegaconf", "rich", "networkx>=3.1", "click",
    "numpy", "scipy", "pandas", "scikit-learn")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
                      cwd=str(PROJECT))
print("\\n\\u2713 Dependencies installed")\
"""


def _sanity(checks):
    lines = ["FAILED = []", "_checks = ["]
    for label, mod in checks:
        lines.append(f'    ("{label}", "{mod}"),')
    lines += [
        "]",
        "for label, mod in _checks:",
        "    try:",
        "        __import__(mod)",
        '        print(f"  \\u2713 {label}")',
        "    except ImportError as e:",
        '        print(f"  \\u2717 {label}: {e}")',
        "        FAILED.append(label)",
        "if FAILED:",
        '    raise RuntimeError(f"Missing modules: {FAILED} -- fix pip installs above")',
        'print("\\n\\u2713 All required modules importable")',
    ]
    return "\n".join(lines)


SANITY_FULL = _sanity([
    ("ultralytics", "ultralytics"), ("boxmot", "boxmot"), ("torch", "torch"),
    ("torchreid", "torchreid"), ("timm", "timm"), ("faiss", "faiss"),
    ("motmetrics", "motmetrics"), ("cv2", "cv2"), ("loguru", "loguru"),
    ("omegaconf", "omegaconf"), ("networkx", "networkx"),
    ("sklearn", "sklearn"), ("numpy", "numpy"), ("pandas", "pandas"),
])

SANITY_LIGHT = _sanity([
    ("faiss", "faiss"), ("motmetrics", "motmetrics"), ("loguru", "loguru"),
    ("omegaconf", "omegaconf"), ("networkx", "networkx"),
    ("sklearn", "sklearn"), ("numpy", "numpy"), ("pandas", "pandas"),
    ("trackeval", "trackeval"),
])

COPY_WEIGHTS = """\
WEIGHTS_INPUT = Path("/kaggle/input/datasets/mrkdagods/mtmc-weights/models")
assert WEIGHTS_INPUT.exists(), (
    "Dataset 'mtmc-weights' not found.\\n"
    f"  Expected: {WEIGHTS_INPUT}\\n"
    "  Attach via: Add Data -> Your Datasets -> mtmc-weights"
)

MODELS_DST = PROJECT / "models"
if MODELS_DST.is_symlink(): MODELS_DST.unlink()
if MODELS_DST.exists(): shutil.rmtree(MODELS_DST)
print(f"Copying models/ from {WEIGHTS_INPUT} (~750 MB) ...")
shutil.copytree(str(WEIGHTS_INPUT), str(MODELS_DST))

ESSENTIAL = [
    "models/detection/yolo26m.pt",
    "models/reid/transreid_cityflowv2_best.pth",
    "models/reid/vehicle_osnet_veri776.pth",
    "models/tracker/osnet_x0_25_msmt17.pt",
]
missing = [p for p in ESSENTIAL if not (PROJECT / p).exists()]
if missing:
    for m in missing: print(f"  MISSING: {m}")
    raise FileNotFoundError("Fix missing weights before continuing.")
print("\\u2713 All essential v4 weights present")\
"""

SHOW_RESULTS = """\
stage5_dir = RUN_DIR / "stage5"

def _pct(v):
    return f"{v:.1%}" if isinstance(v, float) else str(v)

metrics_files = sorted(stage5_dir.glob("metrics_*.json")) if stage5_dir.exists() else []
if metrics_files:
    print("=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    for mf in metrics_files:
        m = json.loads(mf.read_text())
        m = m.get("metrics", m)
        cam = mf.stem.replace("metrics_", "")
        idf1 = _pct(m.get("IDF1", m.get("idf1", "?")))
        mota = _pct(m.get("MOTA", m.get("mota", "?")))
        hota = _pct(m.get("HOTA", m.get("hota", "?")))
        idsw = m.get("ID_Sw", m.get("id_switches", "?"))
        print(f"  {cam:12s}  IDF1={idf1}  MOTA={mota}  HOTA={hota}  IDsw={idsw}")

for fname in ["summary.json", "evaluation_report.json"]:
    sf = stage5_dir / fname
    if sf.exists():
        s = json.loads(sf.read_text())
        print("-" * 65 + "\\n  GLOBAL:")
        for k in ["IDF1","MOTA","HOTA","ID_Sw","idf1","mota","hota","id_switches","mtmc_idf1"]:
            v = s.get(k)
            if v is not None: print(f"    {k}: {_pct(v)}")
        break

if not metrics_files:
    print("No metrics files found -- check stage5 output dir:", stage5_dir)\
"""


# ---- 10a: stages 0-2 --------------------------------------------------------

def build_10a():
    cells = []

    cells.append(md(
        "# 10a -- MTMC Stages 0-2: Frame Extraction + Detection + ReID Features\n\n"
        "**Run once (or when tracking/ReID config changes). ~90 min on P100.**\n\n"
        "| Stage | What | Time |\n|---|---|---|\n"
        "| 0 | Frame extraction (10 fps, 6 cameras) | ~20 min |\n"
        "| 1 | YOLO detection + BotSort tracking | ~45 min |\n"
        "| 2 | TransReID 768D + OSNet 512D -> PCA 256D features | ~20 min |\n\n"
        "After this runs, its output (`checkpoint.tar.gz`) is used by **10b** -> **10c**.\n"
        f"Attach `mtmc-weights` via **Add Data -> Your Datasets -> mtmc-weights**.",
        "a01"))

    cells.append(code(SETUP_ENV, "a02"))
    cells.append(md("## 1. Clone Repo & Install Dependencies", "a03"))
    cells.append(code(CLONE_REPO, "a04"))
    cells.append(code(INSTALL_FULL, "a05"))
    cells.append(code(SANITY_FULL, "a06"))
    cells.append(md("## 2. Mount Model Weights\nModel weights dataset (`mtmc-weights`) must be attached.", "a07"))
    cells.append(code(COPY_WEIGHTS, "a08"))

    cells.append(md(
        "## 3. Download CityFlowV2 (~17 GB)\n\n"
        "Downloaded from Google Drive to `/tmp` (not `/kaggle/working` -- only 20 GB).\n"
        "Peak disk in `/tmp`: ~44 GB (download + extraction + stage0 frames).",
        "a09"))

    cells.append(code("""\
import re as _re, shutil as _shutil

for mount in ["/tmp", "/kaggle/working"]:
    total, used, free = _shutil.disk_usage(mount)
    print(f"{mount:20s}  {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

CAM_RE = _re.compile(r"^S\\d{2}_c\\d{3}$")
TMP_DATA = Path("/tmp/datasets")
TMP_DATA.mkdir(parents=True, exist_ok=True)

DATA_RAW_PARENT = PROJECT / "data" / "raw"
if not DATA_RAW_PARENT.is_symlink():
    if DATA_RAW_PARENT.exists(): shutil.rmtree(DATA_RAW_PARENT)
    DATA_RAW_PARENT.parent.mkdir(parents=True, exist_ok=True)
    DATA_RAW_PARENT.symlink_to(TMP_DATA)
    print(f"\\u2713 data/raw \\u2192 {TMP_DATA}")
else:
    print(f"data/raw already symlinked -> {DATA_RAW_PARENT.resolve()}")

DATA_OUT = Path("/tmp/pipeline_outputs")
DATA_OUT.mkdir(parents=True, exist_ok=True)
DATA_RAW = TMP_DATA / "cityflowv2"

if DATA_RAW.exists() and any(CAM_RE.match(d.name) for d in DATA_RAW.iterdir() if d.is_dir()):
    cams = [d.name for d in DATA_RAW.iterdir() if d.is_dir() and CAM_RE.match(d.name)]
    print(f"\\u2713 CityFlowV2 already present: {len(cams)} cameras")
else:
    print("Downloading CityFlowV2 from Google Drive (~17 GB) ...")
    subprocess.check_call(
        [sys.executable, "scripts/download_datasets.py", "--dataset", "cityflowv2"],
        cwd=str(PROJECT))
    cams = [d.name for d in DATA_RAW.iterdir() if d.is_dir() and CAM_RE.match(d.name)]
    print(f"\\u2713 CityFlowV2 ready: {sorted(cams)}")
print(f"\\nDataset path: {DATA_RAW}")\
""", "a10"))

    cells.append(md("## 4. Run Stages 0-2", "a11"))

    cam_list = ",".join(BENCHMARK_CAMERAS)
    cells.append(code(
        f"from datetime import datetime\n"
        f"RUN_NAME = f\"run_kaggle_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}\"\n"
        f"RUN_DIR  = DATA_OUT / RUN_NAME\n"
        f"RUN_DIR.mkdir(parents=True, exist_ok=True)\n"
        f"BENCHMARK_CAMERAS = {BENCHMARK_CAMERAS!r}\n"
        f"print(f\"Run  : {{RUN_NAME}}\")\n"
        f"print(f\"Cams : {{BENCHMARK_CAMERAS}}\")",
        "a12"))

    cells.append(code(
        f"os.chdir(str(PROJECT))\n"
        f"cmd = [\n"
        f"    sys.executable, \"scripts/run_pipeline.py\",\n"
        f"    \"--config\", \"configs/default.yaml\",\n"
        f"    \"--dataset-config\", \"configs/datasets/cityflowv2.yaml\",\n"
        f"    \"--stages\", \"0,1,2\",\n"
        f"    \"--override\", f\"project.run_name={{RUN_NAME}}\",\n"
        f"    \"--override\", f\"project.output_dir={{DATA_OUT}}\",\n"
        f"    \"--override\", \"stage0.cameras=[{cam_list}]\",\n"
        f"]\n"
        f"print(\"CMD:\", \" \".join(str(c) for c in cmd))\n"
        f"print(\"=\" * 70)\n"
        f"t0 = time.time()\n"
        f"r = subprocess.run(cmd, cwd=str(PROJECT))\n"
        f"print(\"=\" * 70)\n"
        f"elapsed = time.time() - t0\n"
        f"if r.returncode != 0:\n"
        f"    print(f\"\\u2717 FAILED after {{elapsed/60:.1f}} min\"); sys.exit(r.returncode)\n"
        f"print(f\"\\u2713 Stages 0-2 done in {{elapsed/60:.1f}} min\")",
        "a13"))

    cells.append(md(
        "## 5. Save Checkpoint\n\n"
        "Saves stage1 + stage2 outputs + GT annotations to `/kaggle/working/checkpoint.tar.gz`.\n"
        "This file becomes the input for **10b**.\n"
        "Stage0 frame images (~9.6 GB) are **not** included -- downstream stages do not need them.",
        "a14"))

    cells.append(code("""\
import re as _re2

CAM_RE2 = _re2.compile(r"^S\\d{2}_c\\d{3}$")
checkpoint_path = Path("/kaggle/working/checkpoint.tar.gz")
metadata_path   = Path("/kaggle/working/run_metadata.json")

with open(metadata_path, "w") as f:
    json.dump({"run_name": RUN_NAME}, f)

print(f"Building checkpoint for run: {RUN_NAME}")
with tarfile.open(str(checkpoint_path), "w:gz") as tar:
    tar.add(str(metadata_path), arcname="run_metadata.json")

    manifest = RUN_DIR / "stage0" / "frames_manifest.json"
    if manifest.exists():
        tar.add(str(manifest), arcname=f"{RUN_NAME}/stage0/frames_manifest.json")
        print("  + stage0/frames_manifest.json")

    for stage in ["stage1", "stage2"]:
        stage_dir = RUN_DIR / stage
        if stage_dir.exists():
            n = 0
            for fpath in stage_dir.rglob("*"):
                if fpath.is_file():
                    tar.add(str(fpath), arcname=f"{RUN_NAME}/{stage}/{fpath.relative_to(stage_dir)}")
                    n += 1
            print(f"  + {stage}/ ({n} files)")

    # GT annotation txt files needed by stage5 eval (small text files, not videos)
    gt_count = 0
    for cam_dir in DATA_RAW.iterdir():
        if cam_dir.is_dir() and CAM_RE2.match(cam_dir.name):
            gt_file = cam_dir / "gt" / "gt.txt"
            if gt_file.exists():
                tar.add(str(gt_file), arcname=f"gt_annotations/{cam_dir.name}/gt/gt.txt")
                gt_count += 1
    print(f"  + gt_annotations/ ({gt_count} GT files)")

size_mb = checkpoint_path.stat().st_size / 1024**2
print(f"\\n\\u2713 Checkpoint: {checkpoint_path}  ({size_mb:.1f} MB)")
print("  Next: attach this notebook's output to 10b, then push 10b.")\
""", "a15"))

    return cells


# ---- 10b: stage 3 -----------------------------------------------------------

def build_10b():
    cells = []

    cells.append(md(
        f"# 10b -- MTMC Stage 3: FAISS Indexing\n\n"
        f"**Prerequisite**: attach **10a's output** as a data source:\n"
        f"`Add Data -> Kernel Output -> search \"{SLUG_10A}\" -> add`\n\n"
        f"This mounts the checkpoint at `/kaggle/input/{SLUG_10A}/`.\n\n"
        f"| Stage | What | Time |\n|---|---|---|\n"
        f"| 3 | Build FAISS similarity index over ReID features | ~1 min |\n\n"
        f"After this runs, attach **this** notebook's output to **10c**.",
        "b01"))

    cells.append(code(SETUP_ENV, "b02"))
    cells.append(md("## 1. Clone Repo & Install Dependencies\n(No GPU models needed -- much faster than 10a)", "b03"))
    cells.append(code(CLONE_REPO, "b04"))
    cells.append(code(INSTALL_LIGHT, "b05"))
    cells.append(code(SANITY_LIGHT, "b06"))

    cells.append(md(
        f"## 2. Load Checkpoint from 10a\n\n"
        f"Finds `checkpoint.tar.gz` in `/kaggle/input/{SLUG_10A}/` and extracts to `/tmp/pipeline_run/`.",
        "b07"))

    cells.append(code(
        f"PREV_SLUG = \"{SLUG_10A}\"\n"
        f"PREV_INPUT = Path(\"/kaggle/input\") / PREV_SLUG\n"
        f"\n"
        f"# Robust path discovery in case Kaggle changes the mount format\n"
        f"if not PREV_INPUT.exists():\n"
        f"    for p in Path(\"/kaggle/input\").iterdir():\n"
        f"        if PREV_SLUG in p.name or \"stages012\" in p.name or \"10a\" in p.name:\n"
        f"            PREV_INPUT = p; break\n"
        f"\n"
        f"cp = PREV_INPUT / \"checkpoint.tar.gz\"\n"
        f"\n"
        f"# Fallback: download via Kaggle API if not found in mounted input\n"
        f"if not cp.exists():\n"
        f"    print(f\"checkpoint.tar.gz not found at {{cp}} -- attempting API download\")\n"
        f"    import subprocess as _sp\n"
        f"    _dl_dir = Path(\"/tmp/kaggle_dl\")\n"
        f"    _dl_dir.mkdir(parents=True, exist_ok=True)\n"
        f"    _r = _sp.run(\n"
        f"        [\"kaggle\", \"kernels\", \"output\",\n"
        f"         \"mrkdagods/{SLUG_10A}\",\n"
        f"         \"--file\", \"checkpoint.tar.gz\",\n"
        f"         \"-p\", str(_dl_dir)],\n"
        f"        capture_output=True, text=True\n"
        f"    )\n"
        f"    print(_r.stdout); print(_r.stderr)\n"
        f"    cp = _dl_dir / \"checkpoint.tar.gz\"\n"
        f"\n"
        f"assert cp.exists(), f\"checkpoint.tar.gz not found at {{cp}}\"\n"
        f"\n"
        f"EXTRACT_DIR = Path(\"/tmp/pipeline_run\")\n"
        f"EXTRACT_DIR.mkdir(parents=True, exist_ok=True)\n"
        f"print(f\"Extracting {{cp.stat().st_size/1024**2:.1f}} MB ...\")\n"
        f"with tarfile.open(str(cp), \"r:gz\") as tar:\n"
        f"    tar.extractall(str(EXTRACT_DIR))\n"
        f"\n"
        f"with open(EXTRACT_DIR / \"run_metadata.json\") as f:\n"
        f"    meta = json.load(f)\n"
        f"RUN_NAME = meta[\"run_name\"]\n"
        f"DATA_OUT = EXTRACT_DIR\n"
        f"RUN_DIR  = EXTRACT_DIR / RUN_NAME\n"
        f"print(f\"\\u2713 Checkpoint extracted -- run: {{RUN_NAME}}\")\n"
        f"for s in [\"stage1\", \"stage2\"]:\n"
        f"    d = RUN_DIR / s\n"
        f"    if d.exists(): print(f\"  {{s}}: {{len(list(d.rglob('*')))}} files\")",
        "b08"))

    cells.append(md("## 3. Run Stage 3 (FAISS Indexing)", "b09"))

    cells.append(code("""\
os.chdir(str(PROJECT))
cmd = [
    sys.executable, "scripts/run_pipeline.py",
    "--config", "configs/default.yaml",
    "--dataset-config", "configs/datasets/cityflowv2.yaml",
    "--stages", "3",
    "--override", f"project.run_name={RUN_NAME}",
    "--override", f"project.output_dir={DATA_OUT}",
]
print("CMD:", " ".join(str(c) for c in cmd))
print("=" * 70)
t0 = time.time()
r = subprocess.run(cmd, cwd=str(PROJECT))
print("=" * 70)
elapsed = time.time() - t0
if r.returncode != 0:
    print(f"\\u2717 FAILED after {elapsed/60:.1f} min"); sys.exit(r.returncode)
print(f"\\u2713 Stage 3 done in {elapsed/60:.1f} min")\
""", "b10"))

    cells.append(md("## 4. Save Checkpoint for 10c", "b11"))

    cells.append(code("""\
checkpoint_path_out = Path("/kaggle/working/checkpoint.tar.gz")
metadata_path_out   = Path("/kaggle/working/run_metadata.json")
with open(metadata_path_out, "w") as f:
    json.dump({"run_name": RUN_NAME}, f)

with tarfile.open(str(checkpoint_path_out), "w:gz") as tar:
    tar.add(str(metadata_path_out), arcname="run_metadata.json")

    for stage in ["stage1", "stage2", "stage3"]:
        stage_dir = RUN_DIR / stage
        if stage_dir.exists():
            n = 0
            for fpath in stage_dir.rglob("*"):
                if fpath.is_file():
                    tar.add(str(fpath), arcname=f"{RUN_NAME}/{stage}/{fpath.relative_to(stage_dir)}")
                    n += 1
            print(f"  + {stage}/ ({n} files)")

    # Forward GT annotations from 10a's checkpoint
    gt_dir = EXTRACT_DIR / "gt_annotations"
    if gt_dir.exists():
        n = 0
        for fpath in gt_dir.rglob("*"):
            if fpath.is_file():
                tar.add(str(fpath), arcname=f"gt_annotations/{fpath.relative_to(gt_dir)}")
                n += 1
        print(f"  + gt_annotations/ ({n} files forwarded)")

size_mb = checkpoint_path_out.stat().st_size / 1024**2
print(f"\\n\\u2713 Checkpoint: {checkpoint_path_out}  ({size_mb:.1f} MB)")
print("  Next: attach this notebook's output to 10c, then push 10c.")\
""", "b12"))

    return cells


# ---- 10c: stages 4+5 --------------------------------------------------------

def build_10c():
    cells = []

    cells.append(md(
        f"# 10c -- MTMC Stages 4-5: Association + Evaluation\n\n"
        f"**Prerequisite**: attach **10b's output** as a data source:\n"
        f"`Add Data -> Kernel Output -> search \"{SLUG_10B}\" -> add`\n\n"
        f"**This is the iteration loop** -- edit the tuning params cell and re-run in ~6 min.\n"
        f"No GPU needed, no data download, no model inference.\n\n"
        f"| Stage | What | Time |\n|---|---|---|\n"
        f"| 4 | Cross-camera association (AQE + Louvain graph clustering) | ~5 min |\n"
        f"| 5 | Evaluation: IDF1, MOTA, HOTA | ~1 min |",
        "c01"))

    cells.append(code(SETUP_ENV, "c02"))
    cells.append(md("## 1. Clone Repo & Install Dependencies", "c03"))
    cells.append(code(CLONE_REPO, "c04"))
    cells.append(code(INSTALL_LIGHT, "c05"))
    cells.append(code(SANITY_LIGHT, "c06"))

    cells.append(md(
        f"## 2. Load Checkpoint from 10b\n\n"
        f"Finds `checkpoint.tar.gz` in `/kaggle/input/{SLUG_10B}/` and extracts it.",
        "c07"))

    cells.append(code(
        f"PREV_SLUG = \"{SLUG_10B}\"\n"
        f"PREV_INPUT = Path(\"/kaggle/input\") / PREV_SLUG\n"
        f"\n"
        f"if not PREV_INPUT.exists():\n"
        f"    for p in Path(\"/kaggle/input\").iterdir():\n"
        f"        if PREV_SLUG in p.name or \"stage3\" in p.name or \"10b\" in p.name:\n"
        f"            PREV_INPUT = p; break\n"
        f"\n"
        f"cp = PREV_INPUT / \"checkpoint.tar.gz\"\n"
        f"\n"
        f"# Fallback: download via Kaggle API if not found in mounted input\n"
        f"if not cp.exists():\n"
        f"    print(f\"checkpoint.tar.gz not found at {{cp}} -- attempting API download\")\n"
        f"    import subprocess as _sp\n"
        f"    _dl_dir = Path(\"/tmp/kaggle_dl\")\n"
        f"    _dl_dir.mkdir(parents=True, exist_ok=True)\n"
        f"    _r = _sp.run(\n"
        f"        [\"kaggle\", \"kernels\", \"output\",\n"
        f"         \"mrkdagods/{SLUG_10B}\",\n"
        f"         \"--file\", \"checkpoint.tar.gz\",\n"
        f"         \"-p\", str(_dl_dir)],\n"
        f"        capture_output=True, text=True\n"
        f"    )\n"
        f"    print(_r.stdout); print(_r.stderr)\n"
        f"    cp = _dl_dir / \"checkpoint.tar.gz\"\n"
        f"\n"
        f"assert cp.exists(), f\"checkpoint.tar.gz not found at {{cp}}\"\n"
        f"\n"
        f"EXTRACT_DIR = Path(\"/tmp/pipeline_run\")\n"
        f"EXTRACT_DIR.mkdir(parents=True, exist_ok=True)\n"
        f"print(f\"Extracting {{cp.stat().st_size/1024**2:.1f}} MB ...\")\n"
        f"with tarfile.open(str(cp), \"r:gz\") as tar:\n"
        f"    tar.extractall(str(EXTRACT_DIR))\n"
        f"\n"
        f"with open(EXTRACT_DIR / \"run_metadata.json\") as f:\n"
        f"    meta = json.load(f)\n"
        f"RUN_NAME = meta[\"run_name\"]\n"
        f"DATA_OUT = EXTRACT_DIR\n"
        f"RUN_DIR  = EXTRACT_DIR / RUN_NAME\n"
        f"# GT annotations are in the repo under data/raw/cityflowv2/*/gt/gt.txt\n"
        f"# They were force-committed despite data/ being gitignored.\n"
        f"_repo_gt = PROJECT / \"data\" / \"raw\" / \"cityflowv2\"\n"
        f"if any((_repo_gt / cam / \"gt\" / \"gt.txt\").exists()\n"
        f"       for cam in [\"S01_c001\", \"S01_c002\", \"S01_c003\",\n"
        f"                   \"S02_c006\", \"S02_c007\", \"S02_c008\"]):\n"
        f"    GT_DIR = str(_repo_gt)\n"
        f"    print(f\"\\u2713 GT annotations at {{GT_DIR}}\")\n"
        f"else:\n"
        f"    GT_DIR = \"\"\n"
        f"    print(\"WARNING: GT files not found in repo — eval will skip metrics\")\n"
        f"print(f\"\\u2713 Checkpoint extracted -- run: {{RUN_NAME}}\")\n"
        f"for s in [\"stage1\", \"stage2\", \"stage3\"]:\n"
        f"    d = RUN_DIR / s\n"
        f"    if d.exists(): print(f\"  {{s}}: {{len(list(d.rglob('*')))}} files\")\n"
        f"print(f\"  GT dir: {{GT_DIR}}\")",
        "c08"))

    cells.append(md(
        "## 3. Tuning Parameters\n\n"
        "**Edit these values** then re-run the cells below (~6 min). No need to re-run 10a or 10b.",
        "c09"))

    cells.append(code("""\
# ---- Stage 4: Cross-camera association -------------------------------------
# AQE: k nearest neighbours for query expansion (higher = smoother features)
AQE_K             = 7     # v7: 7 (best from scan, was 5)

# Minimum cosine similarity to form an edge in the Louvain graph
SIM_THRESH        = 0.50  # v7: 0.50 (best from scan, was 0.35)

# Louvain resolution (higher = more, smaller clusters)
LOUVAIN_RES       = 0.70  # v7: 0.70 (best from scan, was 0.8)

# Weight of appearance vs. spatio-temporal score (0.0=ST only, 1.0=appear only)
APPEARANCE_WEIGHT = 0.70  # v7: 0.70 (best from scan)

# ---- Stage 5: Evaluation ----------------------------------------------------
# CityFlowV2 GT includes BOTH multi-cam (81 in S01, 130 in S02) AND
# single-cam (14 in S01, 15 in S02) vehicles — total 240 annotated vehicles.
# Setting True filters single-cam predictions, dropping 29 GT vehicles → lower
# IDF1.  Keep False for AIC-protocol comparison against published SOTA (84.1%).
MTMC_ONLY = False

print("Stage 4 params:")
print(f"  aqe_k={AQE_K}  sim_thresh={SIM_THRESH}  louvain_res={LOUVAIN_RES}  appearance_weight={APPEARANCE_WEIGHT}")
print(f"Stage 5: mtmc_only_submission={MTMC_ONLY}")\
""", "c10"))

    cells.append(md("## 4. Run Stages 4-5", "c11"))

    cells.append(code("""\
os.chdir(str(PROJECT))
cmd = [
    sys.executable, "scripts/run_pipeline.py",
    "--config", "configs/default.yaml",
    "--dataset-config", "configs/datasets/cityflowv2.yaml",
    "--stages", "4,5",
    "--override", f"project.run_name={RUN_NAME}",
    "--override", f"project.output_dir={DATA_OUT}",
    "--override", f"stage4.association.query_expansion.k={AQE_K}",
    "--override", f"stage4.association.graph.similarity_threshold={SIM_THRESH}",
    "--override", f"stage4.association.graph.louvain_resolution={LOUVAIN_RES}",
    "--override", f"stage4.association.weights.vehicle.appearance={APPEARANCE_WEIGHT}",
    "--override", f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
]
if GT_DIR:
    cmd += ["--override", f"stage5.ground_truth_dir={GT_DIR}"]
else:
    print("WARNING: GT_DIR is empty — eval will skip metric computation")
print("CMD:", " ".join(str(c) for c in cmd))
print("=" * 70)
t0 = time.time()
r = subprocess.run(cmd, cwd=str(PROJECT))
print("=" * 70)
elapsed = time.time() - t0
if r.returncode != 0:
    print(f"\\u2717 FAILED after {elapsed/60:.1f} min"); sys.exit(r.returncode)
print(f"\\u2713 Stages 4-5 done in {elapsed/60:.1f} min")\
""", "c12"))

    cells.append(md("## 5. Results", "c13"))
    cells.append(code(SHOW_RESULTS, "c14"))

    cells.append(md(
        "## 6. Automated Parameter Scan (optional)\n\n"
        "Runs stages 4-5 across a grid of parameter values and reports the best combination.\n"
        "Each combination takes ~2 min → a 12-point scan takes ~24 min total.\n\n"
        "**Comment out this cell if you just want the single run above.**",
        "c15"))

    cells.append(code("""\
# ============================================================
# SET SCAN_ENABLED = True to run the grid search.
# This will overwrite the single-run results above.
# ============================================================
SCAN_ENABLED = True

if SCAN_ENABLED:
    import itertools

    # Grid to search — comment out axes you don't want to vary
    # When appearance_w changes, spatiotemporal is auto-adjusted: st_w = 1 - app_w - hsv_w
    # so the three vehicle weights always sum to 1.0.
    # aqe_k=0 disables AQE entirely (raw PCA-whitened embeddings used directly).
    # v10 additions: aqe_k=0 ablation, reranking=True test, appearance_w extended to 0.85,
    #                louvain_res narrowed to 2 best values from v9 scan.
    scan_grid = {
        "sim_thresh":       [0.40, 0.45, 0.50, 0.55],   # 4: covers the best region + lower for better recall
        "louvain_res":      [0.60, 0.70],                # 2: dropped 0.80 (no improvement in v9)
        "aqe_k":            [0, 5, 9],                   # 3: 0=disabled (ablation), 5=default, 9=aggressive
        "reranking":        [False, True],               # 2: test k-reciprocal reranking (disabled in v9)
        "appearance_w":     [0.65, 0.75, 0.85],          # 3: extend to 0.85 (trust TransReID more)
    }
    HSV_W_FIXED = 0.05  # fixed hsv weight; spatiotemporal = 1 - appearance - hsv

    keys   = list(scan_grid.keys())
    combos = list(itertools.product(*[scan_grid[k] for k in keys]))
    print(f"Scanning {len(combos)} combinations ...")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        rerank_tag = "rr1" if params["reranking"] else "rr0"
        # Include appearance_w in scan_run name to avoid directory collisions
        # when appearance_w varies (all weights must be in dir name)
        app_tag = f"app{params['appearance_w']:.2f}".replace(".", "")
        scan_run = f"scan_{params['sim_thresh']}_{params['louvain_res']}_{params['aqe_k']}_{rerank_tag}_{app_tag}"

        # Stage 4 reads stage1/stage2/stage3 from output_base/run_name/.
        # Symlink the upstream outputs so the scan sub-dir looks like a full run.
        scan_dir = DATA_OUT / scan_run
        scan_dir.mkdir(parents=True, exist_ok=True)
        for stage_sub in ("stage1", "stage2", "stage3"):
            src = DATA_OUT / RUN_NAME / stage_sub
            dst = scan_dir / stage_sub
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

        # Compute spatiotemporal weight to maintain sum=1.0
        # hsv is fixed at 0.05; st = 1 - appearance - hsv
        st_w = round(1.0 - params["appearance_w"] - HSV_W_FIXED, 4)

        cmd_scan = [
            sys.executable, "scripts/run_pipeline.py",
            "--config", "configs/default.yaml",
            "--dataset-config", "configs/datasets/cityflowv2.yaml",
            "--stages", "4,5",
            "--override", f"project.run_name={scan_run}",
            "--override", f"project.output_dir={DATA_OUT}",
            "--override", f"stage4.association.query_expansion.k={params['aqe_k']}",
            "--override", f"stage4.association.graph.similarity_threshold={params['sim_thresh']}",
            "--override", f"stage4.association.graph.louvain_resolution={params['louvain_res']}",
            "--override", f"stage4.association.weights.vehicle.appearance={params['appearance_w']}",
            "--override", f"stage4.association.weights.vehicle.spatiotemporal={st_w}",
            "--override", f"stage4.association.reranking.enabled={str(params['reranking']).lower()}",
            "--override", "stage5.mtmc_only_submission=false",
        ]
        # When aqe_k=0 explicitly disable AQE so DBA rebuild is also skipped (saves ~2s/combo)
        if params['aqe_k'] == 0:
            cmd_scan += ["--override", "stage4.association.query_expansion.enabled=false"]
        if GT_DIR:
            cmd_scan += ["--override", f"stage5.ground_truth_dir={GT_DIR}"]
        t0 = time.time()
        r = subprocess.run(cmd_scan, cwd=str(PROJECT), capture_output=True)
        elapsed = time.time() - t0

        # Read metric from evaluation report
        report = DATA_OUT / scan_run / "stage5" / "evaluation_report.json"
        idf1 = mota = hota = 0.0
        if report.exists():
            rp = json.loads(report.read_text())
            m = rp.get("metrics", rp)
            idf1 = m.get("mtmc_idf1", m.get("IDF1", m.get("idf1", 0.0)))
            mota = m.get("MOTA", m.get("mota", 0.0))
            hota = m.get("HOTA", m.get("hota", 0.0))

        results.append({**params, "st_w": st_w, "IDF1": idf1, "MOTA": mota, "HOTA": hota, "time": elapsed})
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] {params} st_w={st_w:.2f} -> IDF1={idf1:.3f} MOTA={mota:.3f} HOTA={hota:.3f} ({elapsed/60:.1f} min)")

    # Sort by HOTA when > 0 (HOTA is more robust), fallback to MTMC IDF1
    has_hota = any(r2["HOTA"] > 0 for r2 in results)
    sort_key = "HOTA" if has_hota else "IDF1"
    results.sort(key=lambda x: x[sort_key], reverse=True)
    print("\\n" + "=" * 80)
    print(f"SCAN RESULTS (sorted by {sort_key})")
    print("=" * 80)
    header = f"{'sim':<6} {'res':<6} {'aqe':<5} {'rerank':<8} {'app_w':<7} {'st_w':<7} {'IDF1':>7} {'MOTA':>7} {'HOTA':>7}"
    print(header)
    for r2 in results:
        print(f"{r2['sim_thresh']:<6} {r2['louvain_res']:<6} {r2['aqe_k']:<5} "
              f"{str(r2['reranking']):<8} {r2['appearance_w']:<7} {r2.get('st_w',0.25):<7} "
              f"{r2['IDF1']:>7.3f} {r2['MOTA']:>7.3f} {r2['HOTA']:>7.3f}")
    best = results[0]
    print(f"\\nBEST: sim={best['sim_thresh']} res={best['louvain_res']} aqe={best['aqe_k']} "
          f"reranking={best['reranking']} -> IDF1={best['IDF1']:.3f} HOTA={best['HOTA']:.3f}")
    # Save results to JSON for offline analysis
    import json as _json
    results_path = DATA_OUT / "scan_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"sort_key": sort_key, "results": results}, _f, indent=2)
    print(f"Scan results saved to {results_path}")
else:
    print("Scan disabled. Set SCAN_ENABLED = True to run grid search.")\
""", "c16"))

    return cells


# ---- main -------------------------------------------------------------------

def main():
    print("Generating chained Kaggle notebooks ...")
    write_notebook(
        cells=build_10a(),
        out_dir=NB_ROOT / "10a_stages012",
        slug=SLUG_10A,
        title="MTMC 10a - Stages 0-2 (Tracking + ReID Features)",
        dataset_sources=[f"{OWNER}/mtmc-weights"],
    )
    write_notebook(
        cells=build_10b(),
        out_dir=NB_ROOT / "10b_stage3",
        slug=SLUG_10B,
        title="MTMC 10b - Stage 3 (FAISS Indexing)",
        kernel_sources=[f"{OWNER}/{SLUG_10A}"],
    )
    write_notebook(
        cells=build_10c(),
        out_dir=NB_ROOT / "10c_stages45",
        slug=SLUG_10C,
        title="MTMC 10c - Stages 4-5 (Association + Eval)",
        kernel_sources=[f"{OWNER}/{SLUG_10B}"],
        dataset_sources=[f"{OWNER}/{SLUG_GT}"],
    )
    print("\nDone.")
    print("Workflow:")
    print("  1. Push 10a, run it, wait ~90 min")
    print("  2. On 10b Kaggle page: Add Data -> Kernel Output -> 10a -> push 10b -> run ~1 min")
    print("  3. On 10c Kaggle page: Add Data -> Kernel Output -> 10b -> push 10c -> run ~6 min")
    print("  4. To iterate: edit params in 10c, re-push, re-run in ~6 min")


if __name__ == "__main__":
    main()
