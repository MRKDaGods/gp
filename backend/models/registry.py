"""Typed response models for the read-only model registry API."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


TaskType = Literal["mtmc_vehicle", "mtmc_person", "single_cam_reid", "detector_only"]
DatasetName = Literal["cityflowv2", "wildtrack", "veri776", "custom"]
ModelStatus = Literal["production", "research", "dead_end", "reference"]
MetricSourceKind = Literal["kernel_summary", "local_json", "log_line", "docs"]
CheckpointRole = Literal[
    "primary_reid",
    "secondary_reid",
    "tertiary_reid",
    "quaternary_reid",
    "detector",
    "tracker",
]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricSource(StrictBaseModel):
    kind: MetricSourceKind
    path: str
    kernel: Optional[str] = None
    line_ref: Optional[str] = None


class Metric(StrictBaseModel):
    name: str
    value: float
    verified: bool
    source: MetricSource
    note: Optional[str] = None


class HostedCheckpoint(StrictBaseModel):
    kaggle_dataset: str
    member: str


class CheckpointRef(StrictBaseModel):
    role: CheckpointRole
    local_path: str
    expected_sha256: Optional[str] = None
    hosted: List[HostedCheckpoint] = Field(default_factory=list)
    source_training_kernel: Optional[str] = None
    size_bytes: Optional[int] = None
    on_disk: bool = False


class Requirements(StrictBaseModel):
    gpu_required: bool
    min_vram_gb: int
    data_dependencies: List[str] = Field(default_factory=list)


class Provenance(StrictBaseModel):
    created_at: str
    created_by_kernel: Optional[str] = None
    verified_by: str


class Tombstone(StrictBaseModel):
    reason: str
    superseded_by_id: Optional[str] = None


class ModelEntry(StrictBaseModel):
    id: str
    name: str
    task_type: TaskType
    dataset: DatasetName
    description: str
    metrics: List[Metric]
    pipeline_config: Optional[str]
    model_overrides: List[str] = Field(default_factory=list)
    checkpoint_refs: List[CheckpointRef] = Field(default_factory=list)
    requirements: Requirements
    status: ModelStatus
    runnable_locally: bool
    notebook_or_kernel_ref: Optional[str]
    provenance: Provenance
    tombstone: Optional[Tombstone] = None
    missing_checkpoints: List[str] = Field(default_factory=list)


class Registry(StrictBaseModel):
    version: int
    models: List[ModelEntry]


class ModelListResponse(StrictBaseModel):
    success: bool
    data: List[ModelEntry]


class ModelDetailResponse(StrictBaseModel):
    success: bool
    data: ModelEntry
