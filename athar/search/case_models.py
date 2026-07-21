"""Case-level identity domain model (D7).

A **Target** is a case-level identity that may span multiple appearance
clusters (clothes changes, vehicle boarding). The system never silently fuses
identities across such gaps — it proposes **hypothesis edges** with evidence
and calibrated confidence; investigators confirm or reject; every decision is
attributed and timestamped (audit trail / chain of custody).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from athar.core.ids import TrackKey


def _now() -> datetime:
    return datetime.now(timezone.utc)


class HypothesisKind(str, enum.Enum):
    APPEARANCE = "appearance"       # ReID embedding similarity
    FACE = "face"
    GAIT = "gait"
    BOARDING = "boarding"           # person -> vehicle interaction event
    ALIGHTING = "alighting"         # vehicle -> person interaction event
    MANUAL = "manual"               # investigator-asserted link


class HypothesisStatus(str, enum.Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class HypothesisEdge(BaseModel):
    """A proposed identity link between a target and a tracklet (possibly of a
    different entity class, for boarding/alighting)."""

    kind: HypothesisKind
    track_key: TrackKey
    raw_score: float
    calibrated_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Score mapped through the deployment's calibration model; "
        "the UI must show this, never raw_score alone",
    )
    evidence_artifacts: list[str] = Field(
        default_factory=list, description="Clip/thumbnail artifact references"
    )
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=_now)

    def decide(self, status: HypothesisStatus, operator: str) -> None:
        if status is HypothesisStatus.PROPOSED:
            raise ValueError("cannot decide back to 'proposed'")
        if self.status is not HypothesisStatus.PROPOSED:
            raise ValueError(f"hypothesis already decided: {self.status.value}")
        self.status = status
        self.decided_by = operator
        self.decided_at = _now()


class Target(BaseModel):
    """One case-level identity under investigation."""

    target_id: str
    label: str = Field(description="Investigator-facing name, e.g. 'Suspect A'")
    confirmed_members: list[TrackKey] = Field(default_factory=list)
    hypotheses: list[HypothesisEdge] = Field(default_factory=list)
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


class Case(BaseModel):
    """An investigation: footage, searches, targets, and decisions."""

    case_id: str
    title: str
    gallery_run_ids: list[str] = Field(default_factory=list)
    probe_run_ids: list[str] = Field(default_factory=list)
    targets: list[Target] = Field(default_factory=list)
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
