"""Gate P3 — WILDTRACK ground-plane parity (ROADMAP §4).

Reproduces the v1 WILDTRACK person headline (ground-plane IDF1 0.9456,
MODA 0.9034, IDSW 5) locally on CPU from the cached 12a MVDeTr detections:
test.txt in, the wildtrack single-shot route (ground-plane Kalman tracker,
stage_wildtrack_mvdetr) + stage-5 ground-plane eval out.

The Kaggle reference is kernel ``gumfreddy/14w-verify-wildtrack-b1``, which
validated this exact recipe at commit 8c181472 on 2026-07-21 with
idf1=0.9456066945606695 / moda=0.9033613445378151 — the local run matches
those values bit-for-bit.

Prerequisites (all cheap):

1. Goldens: ``python scripts/kaggle/fetch_p3_goldens.py`` (needs a Kaggle
   token that can see the 12a resume kernel) -> data/goldens/wildtrack_b1_goldens.
2. WILDTRACK raw data at data/raw/wildtrack (annotations_positions,
   calibrations, manifests) — already local.
3. The pinned v1 worktree: auto-created here from the local object store
   (commit 8c181472, the SHA 14w validated).

Run:  ATHAR_RUN_PARITY=1 pytest -m parity tests/parity/test_wildtrack_ground_plane.py

Everything is CPU (numpy Kalman + projection + motmetrics), so this does NOT
violate the Kaggle-GPU rule. Skips (never fails) when the env var, goldens,
raw data, v1 venv, or pinned commit are absent.
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
WT_SHA = "8c181472c892fa53ca7d3e7121a41abf03b26f79"
WT_WORKTREE = Path(os.environ.get("ATHAR_V1_WT_WORKTREE", REPO_ROOT.parent / "gp-v1-wt"))
GOLDENS = Path(
    os.environ.get("ATHAR_P3_GOLDENS", REPO_ROOT / "data" / "goldens" / "wildtrack_b1_goldens")
)
RAW_WILDTRACK = REPO_ROOT / "data" / "raw" / "wildtrack"

EXPECTED_IDF1 = 0.9456066945606695
EXPECTED_MODA = 0.9033613445378151
TOLERANCE = 0.002
EXPECTED_FALSE_POSITIVES = 50
EXPECTED_MISSES = 42

pytestmark = pytest.mark.parity


def _skip_reason() -> str | None:
    if os.environ.get("ATHAR_RUN_PARITY") != "1":
        return "set ATHAR_RUN_PARITY=1 to run the parity gate"
    if not V1_PYTHON.exists():
        return f"v1 venv python not found: {V1_PYTHON}"
    if not (GOLDENS / "test.txt").exists():
        return (
            f"P3 goldens not found at {GOLDENS} — run "
            "scripts/kaggle/fetch_p3_goldens.py (needs kernel access)"
        )
    for rel in ("annotations_positions", "calibrations", "manifests"):
        if not (RAW_WILDTRACK / rel).exists():
            return f"WILDTRACK raw data missing: {RAW_WILDTRACK / rel}"
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-t", WT_SHA],
        capture_output=True, text=True,
    )
    if probe.stdout.strip() != "commit":
        return f"pinned WILDTRACK commit {WT_SHA[:12]} not in local object store"
    return None


def _ensure_wt_worktree() -> None:
    if not (WT_WORKTREE / "scripts" / "run_pipeline.py").exists():
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(WT_WORKTREE), WT_SHA],
            check=True, capture_output=True, text=True,
        )
    raw_dst = WT_WORKTREE / "data" / "raw" / "wildtrack"
    for rel in ("annotations_positions", "calibrations", "manifests"):
        if not (raw_dst / rel).exists():
            shutil.copytree(RAW_WILDTRACK / rel, raw_dst / rel)
    (raw_dst / "videos").mkdir(parents=True, exist_ok=True)


def test_wildtrack_ground_plane_reproduces_idf1_moda(tmp_path):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    _ensure_wt_worktree()

    detections_dst = WT_WORKTREE / "data" / "outputs" / "wildtrack_mvdetr" / "test.txt"
    detections_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GOLDENS / "test.txt", detections_dst)

    run_name = "run_p3_parity"
    run_dir = WT_WORKTREE / "data" / "outputs" / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cmd = [
        str(V1_PYTHON),
        str(WT_WORKTREE / "scripts" / "run_pipeline.py"),
        "--config", str(WT_WORKTREE / "configs" / "datasets" / "wildtrack.yaml"),
        "--stages", "1,2,3,4,5",
        "--override", f"project.run_name={run_name}",
    ]
    env = {**os.environ, "PYTHONPATH": str(WT_WORKTREE)}
    result = subprocess.run(
        cmd, cwd=WT_WORKTREE, env=env, capture_output=True, text=True, timeout=1800
    )
    assert result.returncode == 0, (
        f"v1 wildtrack single-shot run failed (rc={result.returncode}):\n"
        f"stdout tail: {result.stdout[-2000:]}\nstderr tail: {result.stderr[-2000:]}"
    )

    report_path = run_dir / "stage5" / "evaluation_report.json"
    metrics = json.loads(report_path.read_text(encoding="utf-8"))
    ground = metrics["details"]["ground_plane"]
    idf1 = float(ground["idf1"])
    moda = float(ground["moda"])

    assert abs(idf1 - EXPECTED_IDF1) <= TOLERANCE, (
        f"Gate P3 FAILED: ground-plane IDF1 {idf1:.5f} vs expected {EXPECTED_IDF1:.5f} ± {TOLERANCE}"
    )
    assert abs(moda - EXPECTED_MODA) <= TOLERANCE, (
        f"Gate P3 FAILED: ground-plane MODA {moda:.5f} vs expected {EXPECTED_MODA:.5f} ± {TOLERANCE}"
    )
    assert int(ground["false_positives"]) == EXPECTED_FALSE_POSITIVES, (
        f"Gate P3 FAILED: false_positives {ground['false_positives']} vs {EXPECTED_FALSE_POSITIVES}"
    )
    assert int(ground["misses"]) == EXPECTED_MISSES, (
        f"Gate P3 FAILED: misses {ground['misses']} vs {EXPECTED_MISSES}"
    )
