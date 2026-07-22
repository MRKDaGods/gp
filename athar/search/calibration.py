"""Calibrated score -> probability for search hits.

Raw cosine similarities are not probabilities: 0.62 on the vehicle stream
and 0.62 on the person stream mean different things, and investigators
should never be shown a bare score as if it were confidence. Each stream
gets a logistic (Platt) calibration fitted on labeled same/different pairs
from a benchmark run; the calibration file is a versioned artifact whose
provenance (``fitted_on``) travels with every probability we emit.

Uncalibrated streams yield ``None`` — the API reports the score alone
rather than inventing a probability (same honesty rule as D6 evidence).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

CALIBRATION_SCHEMA_VERSION = 1


class CalibrationError(RuntimeError):
    pass


class ScoreCalibration(BaseModel):
    """p(same identity | score) = sigmoid((score - midpoint) / scale)."""

    kind: Literal["logistic"] = "logistic"
    midpoint: float
    scale: float = Field(gt=0.0)
    fitted_on: Optional[str] = Field(
        default=None, description="Provenance: run/benchmark the fit used"
    )
    num_pairs: Optional[int] = None

    def probability(self, score: float) -> float:
        z = (score - self.midpoint) / self.scale
        # guard exp overflow for extreme z
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        labels: Sequence[int],
        fitted_on: Optional[str] = None,
    ) -> "ScoreCalibration":
        """Platt scaling: 1-D logistic regression score -> P(same).

        ``labels``: 1 = same identity, 0 = different. Needs both classes
        and a positive score-probability relationship (higher score must
        mean more likely same — anything else is a data error, not a fit
        parameter).
        """
        import numpy as np
        from scipy.optimize import minimize

        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if s.shape != y.shape or s.ndim != 1:
            raise CalibrationError("scores and labels must be 1-D and equal length")
        if len(s) < 10:
            raise CalibrationError(f"need >= 10 labeled pairs, got {len(s)}")
        if not set(np.unique(y)) <= {0.0, 1.0} or y.min() == y.max():
            raise CalibrationError("labels must contain both 0s and 1s")

        def nll(params: "np.ndarray") -> float:
            a, b = params
            z = a * s + b
            # numerically stable cross-entropy on logits
            return float(np.mean(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1 - y)))

        result = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
        a, b = float(result.x[0]), float(result.x[1])
        if a <= 0:
            raise CalibrationError(
                "fit produced a non-increasing score->probability mapping; "
                "check the pair labels"
            )
        return cls(
            midpoint=-b / a, scale=1.0 / a, fitted_on=fitted_on, num_pairs=len(s)
        )


class StreamCalibrations(BaseModel):
    """Per-stream calibrations, persisted as one JSON artifact."""

    schema_version: int = CALIBRATION_SCHEMA_VERSION
    streams: dict[str, ScoreCalibration] = Field(default_factory=dict)

    def probability(self, stream: str, score: float) -> Optional[float]:
        calibration = self.streams.get(stream)
        return calibration.probability(score) if calibration else None

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> "StreamCalibrations":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
