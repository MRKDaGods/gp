"""Gate P1 — VeRi-776 two-stream fusion parity (ROADMAP §4).

Runs the PORTED fusion evaluator (scripts/eval/eval_14t_fusion_veri776.py in
THIS tree — not the v1 worktree) end-to-end on the real dataset and asserts
the registered headline number. This proves the v2 port of TransReID,
CLIP-SENet, AQE, k-reciprocal re-ranking, and the fusion math reproduces the
benchmark with the frozen checkpoints.

Cost: ~1h on a small GPU, so it runs only when explicitly requested:

    ATHAR_RUN_PARITY=1 pytest -m parity tests/parity/test_veri_fusion.py

Skips (never fails) when the env var, checkpoints, or dataset are absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSREID_CKPT = REPO_ROOT / "models" / "reid" / "vehicle_transreid_vit_base_veri776.pth"
CLIPSENET_CKPT = REPO_ROOT / "models" / "reid" / "clipsenet_v6_veri776_best.pth"
VERI_ROOT = REPO_ROOT / "data" / "raw" / "veri776"

# Registered result: weights_manifest.yaml / model registry (mAP 93.32 at
# w_clipsenet=0.7 canonical operating point). Tolerance covers GPU
# nondeterminism across devices/driver stacks (D2/D18).
EXPECTED_MAP = 0.9332
MAP_TOLERANCE = 0.002

pytestmark = pytest.mark.parity


def _skip_reason() -> str | None:
    if os.environ.get("ATHAR_RUN_PARITY") != "1":
        return "set ATHAR_RUN_PARITY=1 to run the ~1h parity gate"
    for path, what in (
        (TRANSREID_CKPT, "TransReID checkpoint"),
        (CLIPSENET_CKPT, "CLIP-SENet checkpoint"),
        (VERI_ROOT / "image_query", "VeRi-776 dataset"),
    ):
        if not path.exists():
            return f"{what} not found: {path}"
    return None


def test_veri776_fusion_reproduces_headline_map(tmp_path):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    output_json = tmp_path / "veri_fusion_parity.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval" / "eval_14t_fusion_veri776.py"),
        "--transreid-checkpoint", str(TRANSREID_CKPT),
        "--clipsenet-checkpoint", str(CLIPSENET_CKPT),
        "--veri-root", str(VERI_ROOT),
        "--device", "cuda",
        "--transreid-batch-size", os.environ.get("ATHAR_PARITY_TR_BATCH", "16"),
        "--clipsenet-batch-size", os.environ.get("ATHAR_PARITY_CS_BATCH", "8"),
        "--output-json", str(output_json),
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=3 * 3600
    )
    assert result.returncode == 0, (
        f"fusion evaluator failed (rc={result.returncode}):\n"
        f"stdout tail: {result.stdout[-2000:]}\nstderr tail: {result.stderr[-2000:]}"
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    best = payload["score_fusion"]["best"]
    assert abs(best["mAP"] - EXPECTED_MAP) <= MAP_TOLERANCE, (
        f"Gate P1 FAILED: mAP {best['mAP']:.4f} vs expected "
        f"{EXPECTED_MAP:.4f} ± {MAP_TOLERANCE}"
    )
