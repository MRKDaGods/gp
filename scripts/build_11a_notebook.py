from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle" / "11a_wildtrack_stages012"
NOTEBOOK_PATH = NOTEBOOK_DIR / "mtmc-11a-wildtrack-stages-0-2.ipynb"


def to_source(text: str) -> list[str]:
    """Convert text to notebook source array with proper newlines."""
    lines = text.splitlines()
    if not lines:
        return []
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def build_title_cell() -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(
            "# MTMC 11a \u2014 WILDTRACK Stages 0-2 (Tracking + ReID Features)\n"
            "Person multi-camera tracking pipeline on WILDTRACK dataset (7 cameras, 1920x1080, 2fps)."
        ),
    }


def build_gpu_guard_cell() -> dict:
    code = dedent("""\
        import os, sys, subprocess, shutil, json, time, tarfile, re
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
                _match = re.search(r"(\\d+)\\.(\\d+)", _cap)
                if _match:
                    _major, _minor = _match.groups()
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
        print(f"\\nUsing device: {DEVICE}")""")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(code),
    }


def build_clone_install_cell() -> dict:
    code = dedent("""\
        import subprocess, sys, os
        from pathlib import Path

        # Clone repository
        REPO_URL = "https://github.com/MRKDaGods/gp.git"
        PROJECT = Path("/kaggle/working/gp")

        if not PROJECT.exists():
            subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT)])
            print(f"Cloned to {PROJECT}")
        else:
            print(f"Repository already exists at {PROJECT}")

        os.chdir(str(PROJECT))

        # Install dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "ultralytics>=8.1", "boxmot>=10.0", "faiss-cpu>=1.7",
            "timm>=0.9", "omegaconf>=2.3", "networkx>=3.1",
            "opencv-python-headless", "torchreid",
            "scipy", "scikit-learn", "click", "tqdm", "Pillow"])
        print("Dependencies installed")""")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(code),
    }


def build_mount_weights_cell() -> dict:
    code = dedent("""\
        import shutil, os
        from pathlib import Path

        PROJECT = Path("/kaggle/working/gp")
        os.chdir(str(PROJECT))

        # Try both possible mount paths
        WEIGHTS_INPUT = None
        for candidate in [
            Path("/kaggle/input/datasets/mrkdagods/mtmc-weights"),
            Path("/kaggle/input/mtmc-weights"),
        ]:
            if candidate.exists():
                WEIGHTS_INPUT = candidate
                break

        if WEIGHTS_INPUT is None:
            # List what's available
            inp = Path("/kaggle/input")
            if inp.exists():
                print("Available inputs:")
                for p in sorted(inp.rglob("*"))[:30]:
                    print(f"  {p}")
            raise FileNotFoundError("Weights dataset not found - attach mrkdagods/mtmc-weights")

        print(f"Weights found at: {WEIGHTS_INPUT}")

        # Copy models/ tree
        MODELS_DST = PROJECT / "models"
        if MODELS_DST.is_symlink():
            MODELS_DST.unlink()
        if MODELS_DST.exists():
            shutil.rmtree(MODELS_DST)

        print(f"Copying models from {WEIGHTS_INPUT} ...")
        shutil.copytree(str(WEIGHTS_INPUT), str(MODELS_DST))

        # Verify critical person model exists
        person_model = MODELS_DST / "reid" / "person_transreid_vit_base_market1501.pth"
        assert person_model.exists(), f"Person ReID model not found at {person_model}"
        print(f"Person ReID: {person_model.name} ({person_model.stat().st_size/1024**2:.1f} MB)")

        # List all copied models
        for f in sorted(MODELS_DST.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(MODELS_DST)} ({f.stat().st_size/1024**2:.1f} MB)")""")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(code),
    }

def build_mount_wildtrack_cell() -> dict:
    code = dedent("""\
        import os, sys, subprocess, cv2
        from pathlib import Path

        PROJECT = Path("/kaggle/working/gp")
        os.chdir(str(PROJECT))

        # Find WILDTRACK dataset
        WILDTRACK_INPUT = None
        for candidate in [
            Path("/kaggle/input/large-scale-multicamera-detection-dataset"),
            Path("/kaggle/input/wildtrack"),
        ]:
            if candidate.exists():
                WILDTRACK_INPUT = candidate
                break

        if WILDTRACK_INPUT is None:
            raise FileNotFoundError("WILDTRACK dataset not found in /kaggle/input/")
        print(f"WILDTRACK dataset: {WILDTRACK_INPUT}")

        # Find Image_subsets (may be nested under Wildtrack_dataset/)
        img_subsets = None
        for d in sorted(WILDTRACK_INPUT.rglob("Image_subsets")):
            img_subsets = d
            break
        if img_subsets is None:
            print("Available paths:")
            for p in sorted(WILDTRACK_INPUT.iterdir()):
                print(f"  {p}")
            raise FileNotFoundError("Image_subsets not found")

        print(f"Image subsets: {img_subsets}")
        cameras = sorted([d.name for d in img_subsets.iterdir() if d.is_dir()])
        print(f"Cameras: {cameras}")

        # Set up WILDTRACK directory
        WILDTRACK_DIR = PROJECT / "data" / "raw" / "wildtrack"
        VIDEOS_DIR = WILDTRACK_DIR / "videos"
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate MP4 videos from frames
        for cam in cameras:
            cam_dir = img_subsets / cam
            frames = sorted(cam_dir.glob("*.png"))
            if not frames:
                frames = sorted(cam_dir.glob("*.jpg"))
            if not frames:
                print(f"  {cam}: No frames found, skipping")
                continue

            out_path = VIDEOS_DIR / f"{cam}.mp4"
            if out_path.exists():
                print(f"  {cam}: Video exists ({out_path.stat().st_size/1024**2:.1f} MB)")
                continue

            first = cv2.imread(str(frames[0]))
            h, w = first.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 2.0, (w, h))
            for i, fp in enumerate(frames):
                writer.write(cv2.imread(str(fp)))
                if (i + 1) % 100 == 0:
                    print(f"  {cam}: {i+1}/{len(frames)}")
            writer.release()
            print(f"  {cam}: {len(frames)} frames -> {out_path.name} ({w}x{h}, {out_path.stat().st_size/1024**2:.1f} MB)")

        # Symlink annotations and calibrations
        ann_src = cal_src = None
        for d in WILDTRACK_INPUT.rglob("annotations_positions"):
            ann_src = d
            break
        for d in WILDTRACK_INPUT.rglob("calibrations"):
            cal_src = d
            break

        for name, src in [("annotations_positions", ann_src), ("calibrations", cal_src)]:
            dst = WILDTRACK_DIR / name
            if src and not dst.exists():
                dst.symlink_to(src)
                print(f"Linked {name}")
            elif src:
                print(f"Already linked: {name}")
            else:
                print(f"WARNING: {name} not found!")

        print(f"\\nVideos: {list(VIDEOS_DIR.glob('*.mp4'))}")""")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(code),
    }


def build_prepare_gt_cell() -> dict:
    code = dedent("""\
        import subprocess, sys, os
        from pathlib import Path

        PROJECT = Path("/kaggle/working/gp")
        os.chdir(str(PROJECT))

        result = subprocess.run(
            [sys.executable, "scripts/prepare_dataset.py", "--dataset", "wildtrack",
             "--root", "data/raw/wildtrack"],
            capture_output=True, text=True, cwd=str(PROJECT)
        )
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            raise RuntimeError(f"GT preparation failed (exit code {result.returncode})")

        gt_dir = PROJECT / "data" / "raw" / "wildtrack" / "manifests" / "ground_truth"
        if gt_dir.exists():
            for f in sorted(gt_dir.glob("*")):
                if f.is_file():
                    n = len(f.read_text().strip().split("\\n"))
                    print(f"  {f.name}: {n} lines")
        else:
            print(f"WARNING: GT not found at {gt_dir}")""")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(code),
    }


def build_run_pipeline_cell() -> dict:
    code = dedent("""\
        import subprocess, sys, os, time, json
        from pathlib import Path
        from datetime import datetime

        PROJECT = Path("/kaggle/working/gp")
        os.chdir(str(PROJECT))

        DATA_OUT = Path("/tmp/pipeline_outputs")
        DATA_OUT.mkdir(parents=True, exist_ok=True)

        RUN_NAME = f"wildtrack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Run: {RUN_NAME}")

        cmd = [
            sys.executable, "scripts/run_pipeline.py",
            "--config", "configs/default.yaml",
            "--dataset-config", "configs/datasets/wildtrack.yaml",
            "--stages", "0,1,2",
            "--override", f"project.run_name={RUN_NAME}",
            "--override", f"project.output_dir={DATA_OUT}",
            "--override", "stage1.detector.confidence_threshold=0.55",
            "--override", "stage1.tracker.min_hits=3",
            "--override", "stage1.tracker.match_thresh=0.75",
            "--override", "stage1.min_tracklet_length=8",
            "--override", "stage2.pca.n_components=256",
            "--override", "stage2.camera_bn.enabled=true",
            "--override", "stage2.reid.flip_augment=true",
        ]

        print(f"CMD: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(PROJECT))
        elapsed = time.time() - t0
        print(f"\\nDone in {elapsed/60:.1f} min (exit={result.returncode})")

        run_dir = DATA_OUT / RUN_NAME
        for stage in ["stage0", "stage1", "stage2"]:
            sd = run_dir / stage
            if sd.exists():
                nf = sum(1 for _ in sd.rglob("*") if _.is_file())
                sz = sum(f.stat().st_size for f in sd.rglob("*") if f.is_file()) / 1024**2
                print(f"  {stage}: {nf} files ({sz:.1f} MB)")
            else:
                print(f"  {stage}: MISSING")""")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(code),
    }


def build_save_checkpoint_cell() -> dict:
    code = dedent("""\
        import tarfile, json, os
        from pathlib import Path

        PROJECT = Path("/kaggle/working/gp")
        DATA_OUT = Path("/tmp/pipeline_outputs")
        OUTPUT = Path("/kaggle/working")

        runs = sorted(DATA_OUT.glob("wildtrack_*"))
        if not runs:
            raise FileNotFoundError("No wildtrack run directory found")
        run_dir = runs[-1]
        RUN_NAME = run_dir.name
        print(f"Packaging: {RUN_NAME}")

        metadata = {"run_name": RUN_NAME, "dataset": "wildtrack", "type": "person"}
        meta_path = DATA_OUT / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        gt_dir = PROJECT / "data" / "raw" / "wildtrack" / "manifests" / "ground_truth"
        ckpt_path = OUTPUT / "checkpoint.tar.gz"

        with tarfile.open(str(ckpt_path), "w:gz") as tar:
            tar.add(str(meta_path), arcname="run_metadata.json")
            for stage in ["stage1", "stage2"]:
                sd = run_dir / stage
                if sd.exists():
                    tar.add(str(sd), arcname=f"{RUN_NAME}/{stage}")
                    print(f"  Added {stage}")
            manifest = run_dir / "stage0" / "frames_manifest.json"
            if manifest.exists():
                tar.add(str(manifest), arcname=f"{RUN_NAME}/stage0/frames_manifest.json")
                print("  Added stage0/frames_manifest.json")
            if gt_dir.exists():
                tar.add(str(gt_dir), arcname="gt_annotations")
                print("  Added gt_annotations")

        sz = ckpt_path.stat().st_size / 1024**2
        print(f"\\nCheckpoint: {ckpt_path} ({sz:.1f} MB)")
        summary = {"run_name": RUN_NAME, "dataset": "wildtrack", "stages": [0,1,2], "size_mb": round(sz,1)}
        with open(OUTPUT / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))""")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(code),
    }


def verify_source_lines(notebook: dict) -> None:
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        source = cell.get("source", [])
        if len(source) > 1:
            for line_num, line in enumerate(source[:-1], start=1):
                if not line.endswith("\n"):
                    raise RuntimeError(
                        f"Cell {index}, line {line_num}: non-final source line without trailing newline"
                    )
            if source[-1].endswith("\n"):
                raise RuntimeError(f"Cell {index}: final source line unexpectedly ends with newline")


def main() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": [
            build_title_cell(),
            build_gpu_guard_cell(),
            build_clone_install_cell(),
            build_mount_weights_cell(),
            build_mount_wildtrack_cell(),
            build_prepare_gt_cell(),
            build_run_pipeline_cell(),
            build_save_checkpoint_cell(),
        ],
    }

    # Verify source line formatting
    verify_source_lines(notebook)

    # Compile all code cells
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"cell_{index}", "exec")
            print(f"  Cell {index}: compiled OK ({len(cell['source'])} lines)")
        else:
            print(f"  Cell {index}: markdown ({len(cell['source'])} lines)")

    # Write notebook
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=True)
        f.write("\n")

    # Verify written file
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
        verified = json.load(f)

    n_cells = len(verified.get("cells", []))
    if n_cells != 8:
        raise RuntimeError(f"Verification failed: expected 8 cells, found {n_cells}")

    verify_source_lines(verified)

    print(f"\nGenerated: {NOTEBOOK_PATH}")
    print(f"Cells: {n_cells}")
    for i, cell in enumerate(verified["cells"], start=1):
        ct = cell["cell_type"]
        first = cell["source"][0].rstrip("\n") if cell["source"] else ""
        print(f"  {i}. [{ct:8s}] {first[:80]}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
