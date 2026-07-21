"""Typed domain models shared by all pipeline stages and the app layer.

These models define the *meaning* of pipeline data. Bulk numeric payloads
(embedding matrices, per-frame observation tables) are NOT stored in these
models — they live in artifacts (npz/parquet) referenced by the run manifest;
these models carry identity, structure, and provenance.
"""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from athar.core.ids import TrackKey


class EntityClass(str, enum.Enum):
    """Entity classes ATHAR tracks. Extend deliberately (registry-validated)."""

    PERSON = "person"
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    TUKTUK = "tuktuk"

    @property
    def is_vehicle(self) -> bool:
        return self is not EntityClass.PERSON


class BBox(BaseModel):
    """Axis-aligned box in pixel coordinates, (x1, y1) top-left inclusive."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> "BBox":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"degenerate bbox: {self}")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """One detector hit in one frame of one camera."""

    camera_id: str
    frame_index: int
    ts_scene_s: float = Field(description="Scene-clock time (TimeBase-corrected)")
    bbox: BBox
    entity_class: EntityClass
    confidence: float = Field(ge=0.0, le=1.0)


class TrackObservation(BaseModel):
    """One frame-level observation inside a tracklet."""

    frame_index: int
    ts_scene_s: float
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    interpolated: bool = False


class Tracklet(BaseModel):
    """A single-camera track: one entity seen continuously by one camera."""

    key: TrackKey
    entity_class: EntityClass
    start_ts_scene_s: float
    end_ts_scene_s: float
    observation_count: int = Field(ge=1)
    mean_confidence: float = Field(ge=0.0, le=1.0)

    @property
    def duration_s(self) -> float:
        return self.end_ts_scene_s - self.start_ts_scene_s


class Trajectory(BaseModel):
    """A cross-camera identity: the associator's output for one entity."""

    global_id: int
    entity_class: EntityClass
    members: list[TrackKey] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float] = Field(
        default_factory=dict,
        description="Per-score-term contribution breakdown (forensic explainability)",
    )


class EmbeddingStreamRef(BaseModel):
    """Reference to one embedding matrix artifact + its provenance.

    Provenance is what makes cross-run search safe: a probe may only be scored
    against a gallery whose projection lineage is compatible (v1 silently
    projected with a global PCA pickle — never again).
    """

    stream_name: str = Field(description="e.g. 'transreid_primary', 'dinov2', 'hsv'")
    artifact_name: str
    dim: int
    model_id: Optional[str] = None
    projection_fitted_on: Optional[str] = Field(
        default=None,
        description="Run/dataset id the PCA/whitening was fitted on, if projected",
    )
