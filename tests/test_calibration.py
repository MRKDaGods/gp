"""Score-calibration tests: logistic mapping, Platt fit, persistence,
honesty rules (no invented probabilities)."""

from __future__ import annotations

import numpy as np
import pytest

from athar.search.calibration import (
    CalibrationError,
    ScoreCalibration,
    StreamCalibrations,
)


class TestLogistic:
    def test_probability_monotonic_and_bounded(self):
        cal = ScoreCalibration(midpoint=0.5, scale=0.1)
        scores = [-1.0, 0.0, 0.4, 0.5, 0.6, 1.0, 2.0]
        probs = [cal.probability(s) for s in scores]
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert probs == sorted(probs)
        assert cal.probability(0.5) == pytest.approx(0.5)

    def test_extreme_scores_do_not_overflow(self):
        cal = ScoreCalibration(midpoint=0.0, scale=0.001)
        assert cal.probability(1e6) == pytest.approx(1.0)
        assert cal.probability(-1e6) == pytest.approx(0.0)


class TestFit:
    def _pairs(self):
        rng = np.random.default_rng(3)
        same = rng.normal(0.72, 0.05, 200)
        diff = rng.normal(0.31, 0.06, 200)
        scores = np.concatenate([same, diff])
        labels = np.concatenate([np.ones(200), np.zeros(200)]).astype(int)
        return scores.tolist(), labels.tolist()

    def test_fit_separates_classes(self):
        scores, labels = self._pairs()
        cal = ScoreCalibration.fit(scores, labels, fitted_on="bench-x")
        assert cal.probability(0.75) > 0.95
        assert cal.probability(0.30) < 0.05
        assert 0.3 < cal.midpoint < 0.7
        assert cal.fitted_on == "bench-x"
        assert cal.num_pairs == 400

    def test_fit_requires_both_classes(self):
        with pytest.raises(CalibrationError, match="both 0s and 1s"):
            ScoreCalibration.fit([0.1] * 20, [1] * 20)

    def test_fit_requires_enough_pairs(self):
        with pytest.raises(CalibrationError, match=">= 10"):
            ScoreCalibration.fit([0.1, 0.9], [0, 1])

    def test_inverted_labels_rejected(self):
        scores, labels = self._pairs()
        flipped = [1 - y for y in labels]
        with pytest.raises(CalibrationError, match="non-increasing"):
            ScoreCalibration.fit(scores, flipped)


class TestStreamCalibrations:
    def test_uncalibrated_stream_yields_none(self):
        bundle = StreamCalibrations()
        assert bundle.probability("transreid_primary", 0.9) is None

    def test_save_load_roundtrip(self, tmp_path):
        bundle = StreamCalibrations(
            streams={"transreid_primary": ScoreCalibration(midpoint=0.5, scale=0.08)}
        )
        path = tmp_path / "calibrations.json"
        bundle.save(path)
        loaded = StreamCalibrations.load(path)
        assert loaded == bundle
        assert loaded.probability("transreid_primary", 0.5) == pytest.approx(0.5)
