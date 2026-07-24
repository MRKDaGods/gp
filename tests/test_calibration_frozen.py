"""Guard the frozen CityFlowV2 stream calibrations.

The file is a deployable artifact fitted on benchmark ground truth
(scripts/kaggle/calibration_fit/). Serving loads it through
``ATHAR_CALIBRATION_PATH``; this test keeps the committed copy loadable,
provenance-carrying, and sane — a broken or provenance-less calibration
must fail CI, not a deployment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from athar.search.calibration import StreamCalibrations  # noqa: E402

FROZEN = Path(__file__).resolve().parents[1] / "configs" / "calibrations" / "cityflowv2_s02.json"


@pytest.mark.skipif(not FROZEN.exists(), reason="frozen calibrations not fetched yet")
class TestFrozenCalibrations:
    def test_loads_and_is_sane(self):
        calibrations = StreamCalibrations.load(FROZEN)
        assert calibrations.schema_version == 1
        assert "transreid_primary" in calibrations.streams
        for name, calibration in calibrations.streams.items():
            # higher cosine must always mean more likely the same identity
            assert calibration.probability(0.95) > calibration.probability(0.3), name
            assert 0.0 <= calibration.probability(0.0) <= 1.0
            assert calibration.num_pairs and calibration.num_pairs >= 10, name
            assert calibration.fitted_on and "cityflow" in calibration.fitted_on, name

    def test_uncovered_stream_yields_none(self):
        calibrations = StreamCalibrations.load(FROZEN)
        assert calibrations.probability("no_such_stream", 0.5) is None
