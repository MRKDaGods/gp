#!/usr/bin/env python3
"""Generate WILDTRACK pipeline notebooks 11b (Stage 3) and 11c (Stages 4-5)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_source(text: str) -> list[str]:
    """Convert a block of text into a properly formatted source array.
    Each line ends with \\n except the last."""
    lines = text.split("\n")
    # Strip trailing empty lines
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return [""]
    if len(lines) == 1:
        return [lines[0]]
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": make_source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": make_source(text),
        "outputs": [],
        "execution_count": None,
    }


# ============================================================================
# Shared code snippets
# ============================================================================

P100_GUARD = '''\
import os, sys, subprocess, shutil, json, time, tarfile
from pathlib import Path

# --- Guard: detect GPU BEFORE importing torch ---
# Kaggle's PyTorch 2.10+ drops P100 (sm_60) support.
# If we got a P100, downgrade to a compatible build first.
if shutil.which("nvidia-smi"):
    _nvsmi = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True)
    if _nvsmi.returncode == 0 and _nvsmi.stdout.strip():
        _gpu_name, _cap = _nvsmi.stdout.strip().split(",", 1)
        _major, _minor = _cap.strip().split(".")
        _sm = int(_major) * 10 + int(_minor)
        if _sm < 70:
            print(f"\\u26a0 GPU {_gpu_name.strip()} (sm_{_sm}) \\u2014 installing compatible PyTorch ...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "torch==2.4.1", "torchvision==0.19.1",
                "--index-url", "https://download.pytorch.org/whl/cu124",
            ])
            print("\\u2713 Compatible PyTorch installed")

import torch

print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({p.total_memory/1024**3:.1f} GB)")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\\nUsing device: {DEVICE}")'''


CLONE_INSTALL_BASE = '''\
REPO_URL = "https://github.com/abdoibrahim257/MTC-repo.git"
PROJECT = Path("/kaggle/working/gp")

if not PROJECT.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT)])
os.chdir(str(PROJECT))

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "faiss-cpu>=1.7", "omegaconf>=2.3", "networkx>=3.1",
    "scipy", "scikit-learn", "click", "tqdm"])'''


# ============================================================================
# 11b — Stage 3 (FAISS Indexing)
# ============================================================================

def build_11b() -> dict:
    cells = []

    # Cell 1: Title (markdown)
    cells.append(md_cell(
        "# MTMC 11b \\u2014 WILDTRACK Stage 3 (FAISS Indexing)\n"
        "\n"
        "Builds FAISS index over person ReID embeddings from 11a.\n"
        "\n"
        "**Prerequisite**: attach **11a's output** as a data source:\n"
        "`Add Data -> Kernel Output -> search \"mtmc-11a-wildtrack-stages-0-2\" -> add`\n"
        "\n"
        "| Stage | What | Time |\n"
        "|---|---|---|\n"
        "| 3 | Build FAISS similarity index over ReID features | ~1 min |\n"
        "\n"
        "After this runs, attach **this** notebook's output to **11c**."
    ))

    # Cell 2: P100 guard
    cells.append(code_cell(P100_GUARD))

    # Cell 3: Clone + install
    cells.append(md_cell("## 1. Clone Repo & Install Dependencies"))
    cells.append(code_cell(
        CLONE_INSTALL_BASE + '\nprint("\\u2713 Dependencies installed")'
    ))

    # Cell 4: Extract 11a checkpoint
    cells.append(md_cell("## 2. Load Checkpoint from 11a"))
    cells.append(code_cell('''\
DATA_OUT = Path("/tmp/pipeline_outputs")
DATA_OUT.mkdir(parents=True, exist_ok=True)

# Find 11a output
INPUT_11A = None
for candidate in [
    Path("/kaggle/input/mtmc-11a-wildtrack-stages-0-2"),
    Path("/kaggle/input/mtmc-11a-wildtrack-stages-0-2-tracking-reid"),
]:
    if candidate.exists():
        INPUT_11A = candidate
        break

if INPUT_11A is None:
    for d in Path("/kaggle/input").iterdir():
        if (d / "checkpoint.tar.gz").exists():
            INPUT_11A = d
            break

if INPUT_11A is None:
    raise FileNotFoundError("11a output not found in /kaggle/input/")

ckpt = INPUT_11A / "checkpoint.tar.gz"
assert ckpt.exists(), f"checkpoint.tar.gz not found at {ckpt}"
print(f"Checkpoint: {ckpt} ({ckpt.stat().st_size/1024**2:.1f} MB)")

with tarfile.open(str(ckpt), "r:gz") as tar:
    tar.extractall(path=str(DATA_OUT))
    print(f"Extracted to {DATA_OUT}")

# Read metadata
meta_path = DATA_OUT / "run_metadata.json"
with open(meta_path) as f:
    meta = json.load(f)
RUN_NAME = meta["run_name"]
print(f"\\u2713 Run name: {RUN_NAME}")

run_dir = DATA_OUT / RUN_NAME
for stage in ["stage0", "stage1", "stage2"]:
    sd = run_dir / stage
    if sd.exists():
        nf = sum(1 for _ in sd.rglob("*") if _.is_file())
        print(f"  {stage}: {nf} files")
    else:
        print(f"  {stage}: not present")'''))

    # Cell 5: Run Stage 3
    cells.append(md_cell("## 3. Run Stage 3 (FAISS Indexing)"))
    cells.append(code_cell('''\
os.chdir(str(PROJECT))

cmd = [
    sys.executable, "scripts/run_pipeline.py",
    "--config", "configs/default.yaml",
    "--dataset-config", "configs/datasets/wildtrack.yaml",
    "--stages", "3",
    "--override", f"project.run_name={RUN_NAME}",
    "--override", f"project.output_dir={DATA_OUT}",
]

print(f"CMD: {' '.join(str(c) for c in cmd)}")
print("=" * 70)
t0 = time.time()
r = subprocess.run(cmd, cwd=str(PROJECT))
elapsed = time.time() - t0
if r.returncode != 0:
    print(f"\\u2717 FAILED after {elapsed:.1f}s")
    sys.exit(r.returncode)
print("=" * 70)
print(f"\\u2713 Stage 3 done in {elapsed:.1f}s")

stage3_dir = DATA_OUT / RUN_NAME / "stage3"
if stage3_dir.exists():
    nf = sum(1 for _ in stage3_dir.rglob("*") if _.is_file())
    sz = sum(f.stat().st_size for f in stage3_dir.rglob("*") if f.is_file()) / 1024**2
    print(f"  stage3: {nf} files ({sz:.1f} MB)")'''))

    # Cell 6: Save checkpoint
    cells.append(md_cell("## 4. Save Checkpoint for 11c"))
    cells.append(code_cell('''\
checkpoint_path_out = Path("/kaggle/working/checkpoint.tar.gz")
metadata_path_out = Path("/kaggle/working/run_metadata.json")
with open(metadata_path_out, "w") as f:
    json.dump({"run_name": RUN_NAME}, f)

with tarfile.open(str(checkpoint_path_out), "w:gz") as tar:
    tar.add(str(metadata_path_out), arcname="run_metadata.json")

    for stage in ["stage1", "stage2", "stage3"]:
        stage_dir = DATA_OUT / RUN_NAME / stage
        if stage_dir.exists():
            n = 0
            for fpath in stage_dir.rglob("*"):
                if fpath.is_file():
                    tar.add(str(fpath), arcname=f"{RUN_NAME}/{stage}/{fpath.relative_to(stage_dir)}")
                    n += 1
            print(f"  + {stage}/ ({n} files)")

    # Forward GT annotations
    gt_dir = DATA_OUT / "gt_annotations"
    if gt_dir.exists():
        n = 0
        for fpath in gt_dir.rglob("*"):
            if fpath.is_file():
                tar.add(str(fpath), arcname=f"gt_annotations/{fpath.relative_to(gt_dir)}")
                n += 1
        print(f"  + gt_annotations/ ({n} files forwarded)")

sz = checkpoint_path_out.stat().st_size / 1024**2
print(f"\\n\\u2713 Checkpoint: {checkpoint_path_out}  ({sz:.1f} MB)")
print("  Next: attach this notebook's output to 11c, then push 11c.")'''))

    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "cells": cells,
    }


# ============================================================================
# 11c — Stages 4-5 (Association + Evaluation)
# ============================================================================

def build_11c() -> dict:
    cells = []

    # Cell 1: Title
    cells.append(md_cell(
        "# MTMC 11c \\u2014 WILDTRACK Stages 4-5 (Association + Evaluation)\n"
        "\n"
        "Cross-camera person association and evaluation on WILDTRACK.\n"
        "\n"
        "**Prerequisite**: attach **11b's output** as a data source:\n"
        "`Add Data -> Kernel Output -> search \"mtmc-11b-wildtrack-stage-3\" -> add`\n"
        "\n"
        "**This is the iteration loop** \\u2014 edit the tuning params cell and re-run.\n"
        "\n"
        "| Stage | What | Time |\n"
        "|---|---|---|\n"
        "| 4 | Cross-camera association (AQE + graph clustering) | ~2 min |\n"
        "| 5 | Evaluation: IDF1, MOTA, HOTA | ~1 min |"
    ))

    # Cell 2: P100 guard
    cells.append(code_cell(P100_GUARD))

    # Cell 3: Clone + install
    cells.append(md_cell("## 1. Clone Repo & Install Dependencies"))
    cells.append(code_cell(
        CLONE_INSTALL_BASE
        + '\nsubprocess.check_call([sys.executable, "-m", "pip", "install", "-q",\n'
        '    "motmetrics", "lap"])\n'
        'print("\\u2713 Dependencies installed")'
    ))

    # Cell 4: Extract 11b checkpoint
    cells.append(md_cell("## 2. Load Checkpoint from 11b"))
    cells.append(code_cell('''\
DATA_OUT = Path("/tmp/pipeline_outputs")
DATA_OUT.mkdir(parents=True, exist_ok=True)

INPUT_11B = None
for candidate in [
    Path("/kaggle/input/mtmc-11b-wildtrack-stage-3"),
    Path("/kaggle/input/mtmc-11b-wildtrack-stage-3-faiss-indexing"),
]:
    if candidate.exists():
        INPUT_11B = candidate
        break
if INPUT_11B is None:
    for d in Path("/kaggle/input").iterdir():
        if (d / "checkpoint.tar.gz").exists():
            INPUT_11B = d
            break
if INPUT_11B is None:
    raise FileNotFoundError("11b output not found in /kaggle/input/")

ckpt = INPUT_11B / "checkpoint.tar.gz"
print(f"Checkpoint: {ckpt} ({ckpt.stat().st_size/1024**2:.1f} MB)")

with tarfile.open(str(ckpt), "r:gz") as tar:
    tar.extractall(path=str(DATA_OUT))

with open(DATA_OUT / "run_metadata.json") as f:
    meta = json.load(f)
RUN_NAME = meta["run_name"]
print(f"\\u2713 Run: {RUN_NAME}")

run_dir = DATA_OUT / RUN_NAME
for stage in ["stage1", "stage2", "stage3"]:
    sd = run_dir / stage
    if sd.exists():
        nf = sum(1 for _ in sd.rglob("*") if _.is_file())
        print(f"  {stage}: {nf} files")

gt_dir = DATA_OUT / "gt_annotations"
GT_DIR = str(gt_dir) if gt_dir.exists() else ""
print(f"GT: {GT_DIR or 'NOT FOUND'}")'''))

    # Cell 5: Tuning parameters
    cells.append(md_cell(
        "## 3. Tuning Parameters\n"
        "\n"
        "**Edit these values** then re-run the cells below. No need to re-run 11a or 11b."
    ))
    cells.append(code_cell('''\
# ============================================================
# Person Pipeline (WILDTRACK) -- Tuning Parameters
#
# WILDTRACK: 7 cameras, overlapping FOV, ~20 people, 2 fps
# Best local: IDF1=0.368, MTMC_IDF1=0.233, MOTA=0.118
# ============================================================

# Stage 4: Association
SIM_THRESH        = 0.30    # lower than vehicle (more overlap)
ALGORITHM         = "community_detection"
LOUVAIN_RES       = 1.5
BRIDGE_PRUNE      = 0.02
MAX_COMP_SIZE     = 40

AQE_K             = 5
AQE_ALPHA         = 5.0

APPEARANCE_WEIGHT = 0.80
HSV_WEIGHT        = 0.10
ST_WEIGHT         = 0.10

FIC_REG           = 0.10
FIC_ENABLED       = True

RERANKING         = True
RERANKING_K1      = 25
RERANKING_K2      = 8
RERANKING_LAMBDA  = 0.35

GALLERY_THRESH    = 0.35
GALLERY_ROUNDS    = 3

INTRA_MERGE       = True
INTRA_MERGE_THRESH = 0.70
INTRA_MERGE_GAP   = 40.0

# Stage 5
MTMC_ONLY         = False'''))

    # Cell 6: Run stages 4-5
    cells.append(md_cell("## 4. Run Stages 4-5"))
    cells.append(code_cell('''\
os.chdir(str(PROJECT))

cmd = [
    sys.executable, "scripts/run_pipeline.py",
    "--config", "configs/default.yaml",
    "--dataset-config", "configs/datasets/wildtrack.yaml",
    "--stages", "4,5",
    "--override", f"project.run_name={RUN_NAME}",
    "--override", f"project.output_dir={DATA_OUT}",
    "--override", f"stage4.association.query_expansion.k={AQE_K}",
    "--override", f"stage4.association.query_expansion.alpha={AQE_ALPHA}",
    "--override", f"stage4.association.graph.similarity_threshold={SIM_THRESH}",
    "--override", f"stage4.association.graph.algorithm={ALGORITHM}",
    "--override", f"stage4.association.graph.louvain_resolution={LOUVAIN_RES}",
    "--override", f"stage4.association.graph.bridge_prune_margin={BRIDGE_PRUNE}",
    "--override", f"stage4.association.graph.max_component_size={MAX_COMP_SIZE}",
    "--override", f"stage4.association.weights.person.appearance={APPEARANCE_WEIGHT}",
    "--override", f"stage4.association.weights.person.hsv={HSV_WEIGHT}",
    "--override", f"stage4.association.weights.person.spatiotemporal={ST_WEIGHT}",
    "--override", f"stage4.association.fic.enabled={str(FIC_ENABLED).lower()}",
    "--override", f"stage4.association.fic.regularisation={FIC_REG}",
    "--override", f"stage4.association.reranking.enabled={str(RERANKING).lower()}",
    "--override", f"stage4.association.reranking.k1={RERANKING_K1}",
    "--override", f"stage4.association.reranking.k2={RERANKING_K2}",
    "--override", f"stage4.association.reranking.lambda_value={RERANKING_LAMBDA}",
    "--override", f"stage4.association.gallery_expansion.threshold={GALLERY_THRESH}",
    "--override", f"stage4.association.gallery_expansion.rounds={GALLERY_ROUNDS}",
    "--override", f"stage4.association.intra_camera_merge.enabled={str(INTRA_MERGE).lower()}",
    "--override", f"stage4.association.intra_camera_merge.threshold={INTRA_MERGE_THRESH}",
    "--override", f"stage4.association.intra_camera_merge.max_time_gap={INTRA_MERGE_GAP}",
    "--override", f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
]
if GT_DIR:
    cmd += ["--override", f"stage5.ground_truth_dir={GT_DIR}"]

print(f"CMD: {' '.join(str(c) for c in cmd[:10])}...")
print("=" * 70)
t0 = time.time()
result = subprocess.run(cmd, cwd=str(PROJECT))
elapsed = time.time() - t0
print("=" * 70)
print(f"\\nDone in {elapsed:.1f}s (exit={result.returncode})")'''))

    # Cell 7: Parse and display results
    cells.append(md_cell("## 5. Results"))
    cells.append(code_cell('''\
report_path = DATA_OUT / RUN_NAME / "stage5" / "evaluation_report.json"
if report_path.exists():
    with open(report_path) as f:
        report = json.load(f)

    m = report.get("metrics", report)
    print("=" * 60)
    print("WILDTRACK PERSON PIPELINE RESULTS")
    print("=" * 60)
    print(f"  IDF1:      {m.get('idf1', m.get('IDF1', 0)):.4f}")
    print(f"  MTMC_IDF1: {m.get('mtmc_idf1', 0):.4f}")
    print(f"  MOTA:      {m.get('MOTA', m.get('mota', 0)):.4f}")
    print(f"  HOTA:      {m.get('HOTA', m.get('hota', 0)):.4f}")

    details = report.get("details", {})
    if details:
        print(f"\\n  MTMC_MOTA: {details.get('mtmc_mota', 'N/A')}")
        print(f"  ID_sw:     {details.get('id_switches', 'N/A')}")
        print(f"  Tracklets: {details.get('num_tracklets', 'N/A')}")

    # Per-camera breakdown
    per_cam = report.get("per_camera", {})
    if per_cam:
        print(f"\\nPer-camera IDF1:")
        for cam, cm in sorted(per_cam.items()):
            idf1 = cm.get("idf1", cm.get("IDF1", 0))
            print(f"  {cam}: {idf1:.3f}")
else:
    print(f"No evaluation report found at {report_path}")
    # Check if stage4/5 output exists
    for sn in ["stage4", "stage5"]:
        sd = DATA_OUT / RUN_NAME / sn
        if sd.exists():
            print(f"  {sn} output exists ({sum(1 for _ in sd.rglob('*'))} files)")'''))

    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "cells": cells,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    nb_dir = ROOT / "notebooks" / "kaggle"

    # 11b
    out_11b = nb_dir / "11b_wildtrack_stage3" / "mtmc-11b-wildtrack-stage-3.ipynb"
    out_11b.parent.mkdir(parents=True, exist_ok=True)
    nb_11b = build_11b()
    with open(out_11b, "w", encoding="utf-8") as f:
        json.dump(nb_11b, f, indent=1, ensure_ascii=True)
    print(f"Wrote {out_11b} ({len(nb_11b['cells'])} cells)")

    # 11c
    out_11c = nb_dir / "11c_wildtrack_stages45" / "mtmc-11c-wildtrack-stages-4-5.ipynb"
    out_11c.parent.mkdir(parents=True, exist_ok=True)
    nb_11c = build_11c()
    with open(out_11c, "w", encoding="utf-8") as f:
        json.dump(nb_11c, f, indent=1, ensure_ascii=True)
    print(f"Wrote {out_11c} ({len(nb_11c['cells'])} cells)")

    # Verify source line endings
    for label, nb in [("11b", nb_11b), ("11c", nb_11c)]:
        for i, cell in enumerate(nb["cells"]):
            src = cell["source"]
            if len(src) > 1:
                for j, line in enumerate(src[:-1]):
                    assert line.endswith("\n"), (
                        f"{label} cell {i} line {j}: missing trailing newline"
                    )
                assert not src[-1].endswith("\n"), (
                    f"{label} cell {i}: last line should NOT end with newline"
                )
        print(f"  {label}: all source line endings OK")


if __name__ == "__main__":
    main()
