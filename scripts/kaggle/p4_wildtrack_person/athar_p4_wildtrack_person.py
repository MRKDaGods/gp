"""Gate P4 kernel — v2 pipeline person baseline on WILDTRACK-as-plain-video.

GPU kernel. Runs the ATHAR v2 DAG (athar run: ingest -> detect_track ->
embed -> index -> associate) over the 7 WILDTRACK Image_subsets cameras as
image-directory evidence at a declared 2 fps, then scores per-camera SCT +
cross-camera MTMC IDF1 against the v1-recipe MOT ground truth with
scripts/eval/eval_p4_wildtrack_person.py.

Inputs (kernel dataset_sources):
- aryashah2k/large-scale-multicamera-detection-dataset  (WILDTRACK frames)
- mrkdagods/athar-p4-bundle   (src/ tree + yolo26m + osnet + GT + profile)
- gumfreddy/mtmc-weights      (person_transreid_vit_base_market1501.pth)

Outputs: p4_metrics.json + p4_run_artifacts.tar.gz (tracklets/trajectories
for local baseline freezing).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PROJECT = WORK / "athar-v2"
RUNS_ROOT = WORK / "runs"
OUT_DIR = WORK / "p4_out"


def run(cmd, cwd=None, env=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def find_dir(marker: str) -> Path:
    for path in INPUT.rglob(marker):
        return path.parent if path.is_file() else path
    raise FileNotFoundError(f"marker {marker!r} not found under {INPUT}")


def main() -> None:
    bundle = find_dir("profile_p4_wildtrack_person.yaml")
    print("bundle:", bundle)

    if not PROJECT.exists():
        shutil.copytree(bundle / "src", PROJECT)
    print("src head:", (PROJECT / "GIT_SHA.txt").read_text().strip()
          if (PROJECT / "GIT_SHA.txt").exists() else "(no sha file)")

    run([sys.executable, "-m", "pip", "install", "-q",
         "faiss-cpu", "motmetrics", "lapx", "pydantic>=2", "pyyaml", "timm", "filterpy"])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "ultralytics", "boxmot==22.0.0"])
    # No pip install of athar itself: the package pins Python 3.13 and Kaggle
    # ships 3.12 — PYTHONPATH is all the runtime actually needs.
    env = {**os.environ, "PYTHONPATH": str(PROJECT)}

    # ---- models into the tree ----------------------------------------
    (PROJECT / "models" / "detection").mkdir(parents=True, exist_ok=True)
    (PROJECT / "models" / "tracker").mkdir(parents=True, exist_ok=True)
    (PROJECT / "models" / "reid").mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle / "models" / "yolo26m.pt",
                 PROJECT / "models" / "detection" / "yolo26m.pt")
    shutil.copy2(bundle / "models" / "osnet_x0_25_msmt17.pt",
                 PROJECT / "models" / "tracker" / "osnet_x0_25_msmt17.pt")
    person_ckpt = next(INPUT.rglob("person_transreid_vit_base_market1501.pth"))
    shutil.copy2(person_ckpt, PROJECT / "models" / "reid" / person_ckpt.name)

    # ---- WILDTRACK frames --------------------------------------------
    image_subsets = None
    for path in INPUT.rglob("Image_subsets"):
        if path.is_dir() and (path / "C1").is_dir():
            image_subsets = path
            break
    assert image_subsets is not None, "Image_subsets/C1 not found"
    cams = [f"C{i}" for i in range(1, 8)]
    for cam in cams:
        n = len(list((image_subsets / cam).glob("*.png")))
        print(f"{cam}: {n} frames")

    # ---- the run ------------------------------------------------------
    cmd = [sys.executable, "-m", "athar.cli.main", "run",
           "--profile", str(bundle / "profile_p4_wildtrack_person.yaml"),
           "--role", "benchmark", "--fps", "2",
           "--runs-root", str(RUNS_ROOT)]
    for cam in cams:
        cmd += ["--video", f"{cam}={image_subsets / cam}"]
    run(cmd, cwd=PROJECT, env=env)

    run_dir = next(p for p in sorted(RUNS_ROOT.iterdir()) if (p / "manifest.json").exists())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    print("run:", manifest["run_id"], "status:", manifest["status"])
    assert manifest["status"] == "completed", f"run not completed: {manifest['status']}"

    # ---- eval ---------------------------------------------------------
    run([sys.executable, str(PROJECT / "scripts" / "eval" / "eval_p4_wildtrack_person.py"),
         "--run-dir", str(run_dir), "--gt-dir", str(bundle / "gt"),
         "--out-dir", str(OUT_DIR)], cwd=PROJECT, env=env)

    metrics = json.loads((OUT_DIR / "p4_metrics.json").read_text())
    print(json.dumps(metrics, indent=2))
    shutil.copy2(OUT_DIR / "p4_metrics.json", WORK / "p4_metrics.json")

    # ---- artifacts for local freezing --------------------------------
    with tarfile.open(WORK / "p4_run_artifacts.tar.gz", "w:gz") as tar:
        tar.add(run_dir / "manifest.json", arcname="manifest.json")
        for rel in manifest["artifacts"].values():
            relpath = rel["relpath"] if isinstance(rel, dict) else rel
            src = run_dir / relpath
            if src.is_file() and src.stat().st_size < 200 * 1024 * 1024:
                tar.add(src, arcname=relpath)
        for extra in (OUT_DIR / "preds_sct", OUT_DIR / "preds_mtmc"):
            if extra.is_dir():
                tar.add(extra, arcname=extra.name)
    print("wrote", WORK / "p4_run_artifacts.tar.gz")


if __name__ == "__main__":
    main()
