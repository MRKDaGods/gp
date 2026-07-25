"""Guard the frozen Phase 6 cross-domain eval matrix.

Frozen by scripts/kaggle/joint_reid/fetch_results.py once the Stage C kernel
completes. Until then the file is absent and this module self-skips (same
pattern as test_calibration_frozen). Once frozen, a malformed or
provenance-less matrix must fail CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "kaggle" / "joint_reid" / "results"
FROZEN = sorted(RESULTS_DIR.glob("cross_domain_matrix_*.json"))

DATASETS = {"veri776", "veriwild_3000", "vehicleid_800", "cityflow_s02"}


@pytest.mark.skipif(not FROZEN, reason="cross-domain matrix not frozen yet")
class TestFrozenCrossDomainMatrix:
    def test_structure_and_provenance(self):
        payload = json.loads(FROZEN[-1].read_text("utf-8"))
        assert payload["protocol"]["mode"].startswith("deployed single-stream")
        checkpoints = payload["checkpoints"]
        assert {"baseline_veri776", "joint4d"} <= set(checkpoints)
        for meta in checkpoints.values():
            assert len(meta["sha256"]) == 64
        matrix = payload["matrix"]
        assert {"baseline_veri776", "joint4d"} <= set(matrix)
        for model_name, per_dataset in matrix.items():
            assert DATASETS <= set(per_dataset), (model_name, sorted(per_dataset))
            for ds, res in per_dataset.items():
                assert 0.0 <= res["mAP"] <= 1.0, (model_name, ds)
                assert 0.0 <= res["R1"] <= 1.0, (model_name, ds)

    def test_valid_queries_nonzero(self):
        """Every cell must come from a non-degenerate eval (a matrix frozen
        from an empty query set would silently read as mAP 0.0). An honest
        negative RESULT is freezable; a broken eval is not."""
        payload = json.loads(FROZEN[-1].read_text("utf-8"))
        for model_name, per_dataset in payload["matrix"].items():
            for ds, res in per_dataset.items():
                n = res.get("valid_queries", res.get("n_ids", 0))
                assert n and n > 50, (model_name, ds, n)
