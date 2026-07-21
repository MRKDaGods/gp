"""Gate P2 — CityFlow B1 association parity (ROADMAP §4).

Reproduces the registered ``vehicle_mtmc_14e_b1`` baseline (MTMC IDF1
0.77936, id_switches 154) locally on CPU from the Kaggle-generated goldens:
v1 stage-1 tracklets + stage-2 TTA features in, stages 3-5 (FAISS index,
association, eval) out.

Two prerequisites, both cheap once access exists:

1. Goldens: ``python scripts/kaggle/fetch_p2_goldens.py`` (needs a Kaggle
   token that can see the goldens kernel) → data/goldens/cityflow_b1_goldens.
2. The pinned v1 B1 worktree: auto-created here from the local object store
   (commit 24e85f31, the SHA the public 14v kernel validated on 2026-05-15;
   its configs bake the B1 recipe — no overrides, no stale-config risk).

Run:  ATHAR_RUN_PARITY=1 pytest -m parity tests/parity/test_cityflow_association.py

Stages 3-5 are CPU-only (the reference 14v kernel ran on a CPU machine), so
this does NOT violate the Kaggle-GPU rule. Skips (never fails) when the env
var, goldens, v1 venv, or pinned commit are absent.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
V1_PYTHON = Path(os.environ.get("ATHAR_V1_PYTHON", REPO_ROOT / ".venv" / "Scripts" / "python.exe"))
B1_SHA = "24e85f31e6663e3f4b4d6649f2b34c9ce2145f0e"
B1_WORKTREE = Path(os.environ.get("ATHAR_V1_B1_WORKTREE", REPO_ROOT.parent / "gp-v1-b1"))
GOLDENS = Path(
    os.environ.get("ATHAR_P2_GOLDENS", REPO_ROOT / "data" / "goldens" / "cityflow_b1_goldens")
)

EXPECTED_IDF1 = 0.77936
IDF1_TOLERANCE = 0.002
EXPECTED_ID_SWITCHES = 154

pytestmark = pytest.mark.parity


def _skip_reason() -> str | None:
    if os.environ.get("ATHAR_RUN_PARITY") != "1":
        return "set ATHAR_RUN_PARITY=1 to run the parity gate"
    if not V1_PYTHON.exists():
        return f"v1 venv python not found: {V1_PYTHON}"
    if not (GOLDENS / "stage2" / "embeddings.npy").exists():
        return (
            f"P2 goldens not found at {GOLDENS} — run "
            "scripts/kaggle/fetch_p2_goldens.py (needs kernel access)"
        )
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-t", B1_SHA],
        capture_output=True, text=True,
    )
    if probe.stdout.strip() != "commit":
        return f"pinned B1 commit {B1_SHA[:12]} not in local object store"
    return None


def _ensure_b1_worktree() -> None:
    if (B1_WORKTREE / "scripts" / "run_pipeline.py").exists():
        return
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(B1_WORKTREE), B1_SHA],
        check=True, capture_output=True, text=True,
    )


def test_cityflow_b1_association_reproduces_idf1(tmp_path):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    _ensure_b1_worktree()

    run_name = "run_p2_parity"
    run_dir = B1_WORKTREE / "data" / "outputs" / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copytree(GOLDENS / "stage1", run_dir / "stage1")
    shutil.copytree(GOLDENS / "stage2", run_dir / "stage2")

    tertiary = run_dir / "stage2" / "embeddings_tertiary.npy"
    cmd = [
        str(V1_PYTHON),
        str(B1_WORKTREE / "scripts" / "run_pipeline.py"),
        "--config", str(B1_WORKTREE / "configs" / "datasets" / "cityflowv2.yaml"),
        "--stages", "3,4,5",
        "--override", f"project.run_name={run_name}",
        # the baked config points tertiary at data/outputs/run_latest/…; we
        # don't create that symlink (Windows privileges) — pass the SAME file
        # by absolute path instead. Not a recipe change.
        "--override",
        f"stage4.association.tertiary_embeddings.path={tertiary.as_posix()}",
    ]
    env = {**os.environ, "PYTHONPATH": str(B1_WORKTREE)}
    result = subprocess.run(
        cmd, cwd=B1_WORKTREE, env=env, capture_output=True, text=True, timeout=3600
    )
    assert result.returncode == 0, (
        f"v1 stages 3-5 failed (rc={result.returncode}):\n"
        f"stdout tail: {result.stdout[-2000:]}\nstderr tail: {result.stderr[-2000:]}"
    )

    metrics_path = run_dir / "stage5" / "evaluation_report.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "stage5" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    details = metrics.get("details", {}) or {}
    idf1 = float(
        metrics.get("MTMC_IDF1") or metrics.get("mtmc_idf1")
        or details.get("mtmc_idf1") or metrics.get("idf1") or metrics["IDF1"]
    )
    id_switches = int(
        metrics.get("id_switches") or details.get("mtmc_id_switches") or metrics.get("IDS")
    )

    assert abs(idf1 - EXPECTED_IDF1) <= IDF1_TOLERANCE, (
        f"Gate P2 FAILED: IDF1 {idf1:.5f} vs expected {EXPECTED_IDF1} ± {IDF1_TOLERANCE}"
    )
    assert id_switches == EXPECTED_ID_SWITCHES, (
        f"Gate P2 FAILED: id_switches {id_switches} vs expected {EXPECTED_ID_SWITCHES}"
    )
