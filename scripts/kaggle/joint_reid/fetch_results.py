"""Fetch + freeze the joint-ReID campaign results.

Stage B (train): pulls transreid_joint4d_best.pth + provenance.json from
`mrkdagods/athar-joint-reid-train` into models/reid/ (the checkpoint is a
lifecycle CANDIDATE -- promotion stays eval-gated via `athar models`).

Stage C (eval): pulls cross_domain_matrix.json from
`mrkdagods/athar-joint-reid-eval`, sanity-checks it, and freezes it as
scripts/kaggle/joint_reid/results/cross_domain_matrix_<date>.json --
the file tests/test_cross_domain_matrix_frozen.py guards.

Windows note: PYTHONUTF8=1 for the Kaggle CLI (cp1252 crash).

Usage: python scripts/kaggle/joint_reid/fetch_results.py [train|eval|all]
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_KERNEL = "mrkdagods/athar-joint-reid-train"
EVAL_KERNEL = "mrkdagods/athar-joint-reid-eval"
SCRATCH = PROJECT_ROOT / "data" / "goldens" / "joint_reid_out"
RESULTS_DIR = Path(__file__).parent / "results"


def pull(kernel: str, pattern: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", kernel, "--file-pattern", pattern, "-p", str(dest)],
        check=False, env={**os.environ, "PYTHONUTF8": "1"},
    )


def fetch_train() -> int:
    dest = SCRATCH / "train"
    pull(TRAIN_KERNEL, r"^(transreid_joint4d_best\.pth|provenance\.json)$", dest)
    ckpt = dest / "transreid_joint4d_best.pth"
    prov = dest / "provenance.json"
    if not ckpt.exists():
        print("train checkpoint not found - kernel not finished or failed", file=sys.stderr)
        return 1
    payload = json.loads(prov.read_text("utf-8")) if prov.exists() else {}
    print(f"best VeRi mAP: {payload.get('best_veri_mAP')} "
          f"epochs: {payload.get('epochs_completed')}/{payload.get('epochs_configured')} "
          f"sha256: {payload.get('checkpoint_sha256', '?')[:16]}...")
    if payload.get("stopped_early_for_session_budget"):
        print("NOTE: session budget stopped training early -- resume the kernel "
              "(attach its prior version output) before treating this as final.")
    models_dir = PROJECT_ROOT / "models" / "reid"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, models_dir / ckpt.name)
    shutil.copy2(prov, models_dir / "transreid_joint4d_provenance.json")
    print(f"checkpoint -> {models_dir / ckpt.name} (lifecycle CANDIDATE; promotion is eval-gated)")
    return 0


def fetch_eval() -> int:
    dest = SCRATCH / "eval"
    pull(EVAL_KERNEL, r"^cross_domain_matrix\.json$", dest)
    src = dest / "cross_domain_matrix.json"
    if not src.exists():
        print("cross_domain_matrix.json not found - kernel not finished or failed", file=sys.stderr)
        return 1
    payload = json.loads(src.read_text("utf-8"))
    matrix = payload["matrix"]
    assert {"baseline_veri776", "joint4d"} <= set(matrix), "matrix missing a model"
    datasets = sorted(matrix["joint4d"])
    print(f"{'dataset':<16}{'baseline mAP':>14}{'joint mAP':>12}{'delta':>9}")
    for ds in datasets:
        b = matrix["baseline_veri776"][ds]["mAP"]
        j = matrix["joint4d"][ds]["mAP"]
        print(f"{ds:<16}{b:>14.4f}{j:>12.4f}{j - b:>+9.4f}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frozen = RESULTS_DIR / f"cross_domain_matrix_{dt.date.today():%Y%m%d}.json"
    shutil.copy2(src, frozen)
    print(f"frozen: {frozen}")
    return 0


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    rc = 0
    if what in ("train", "all"):
        rc |= fetch_train()
    if what in ("eval", "all"):
        rc |= fetch_eval()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
