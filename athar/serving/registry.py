"""Model lifecycle registry types (D4/D5).

Stages: candidate → validated → production → retired. Promotion to
production is gated by the evaluation harness — ``promote`` refuses without
an eval report reference. Storage backend (SQLite) arrives in Phase 4; the
types and invariants are fixed now because everything else references them.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class ModelStage(str, enum.Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    PRODUCTION = "production"
    RETIRED = "retired"


class ModelTask(str, enum.Enum):
    DETECTION = "detection"
    MULTI_VIEW_DETECTION = "multi_view_detection"
    REID_VEHICLE = "reid_vehicle"
    REID_PERSON = "reid_person"
    REID_CC_PERSON = "reid_cc_person"   # clothes-change robust
    FACE = "face"
    GAIT = "gait"


class CheckpointRef(BaseModel):
    """Content-addressed weights file."""

    sha256: str
    size_bytes: int
    filename: str


class EvalReportRef(BaseModel):
    """Reference to an evaluation-harness report backing a lifecycle change."""

    run_id: str
    benchmark: str
    metrics: dict[str, float]


class ModelEntry(BaseModel):
    model_id: str
    task: ModelTask
    architecture: str
    checkpoint: CheckpointRef
    stage: ModelStage = ModelStage.CANDIDATE
    trained_on: list[str] = Field(default_factory=list, description="Dataset ids")
    adapted_for_site: Optional[str] = Field(
        default=None, description="Deployment/site id for adaptation outputs"
    )
    eval_reports: list[EvalReportRef] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""

    def promote(self, to: ModelStage, eval_report: Optional[EvalReportRef] = None) -> None:
        order = [ModelStage.CANDIDATE, ModelStage.VALIDATED, ModelStage.PRODUCTION]
        if to is ModelStage.RETIRED:
            self.stage = to
            return
        if to not in order or order.index(to) != order.index(self.stage) + 1:
            raise ValueError(f"illegal promotion {self.stage.value} -> {to.value}")
        if eval_report is None:
            raise ValueError(
                f"promotion to {to.value} requires an evaluation report (D5: eval-gated lifecycle)"
            )
        self.eval_reports.append(eval_report)
        self.stage = to
