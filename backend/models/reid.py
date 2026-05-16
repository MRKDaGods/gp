"""Response schemas for ReID serving endpoints and local jobs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ReIDBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class ReIDImageRef(ReIDBaseModel):
    id: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReIDRankedMatch(ReIDBaseModel):
    gallery_id: str = Field(alias="galleryId")
    rank: int
    score: float
    distance: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReIDQueryResult(ReIDBaseModel):
    query_id: str = Field(alias="queryId")
    matches: List[ReIDRankedMatch]
    latency_ms: float = Field(alias="latencyMs")


class SingleCamReIDResponse(ReIDBaseModel):
    success: bool
    model_id: str = Field(alias="modelId")
    device: str
    feature_dim: int = Field(alias="featureDim")
    query_count: int = Field(alias="queryCount")
    gallery_count: int = Field(alias="galleryCount")
    results: List[ReIDQueryResult]
    latency_ms: float = Field(alias="latencyMs")


class ReIDComponentResult(ReIDBaseModel):
    model_id: str = Field(alias="modelId")
    weight: float
    feature_dim: int = Field(alias="featureDim")
    results: List[ReIDQueryResult]


class FusionReIDResponse(ReIDBaseModel):
    success: bool
    model_ids: List[str] = Field(alias="modelIds")
    weights: List[float]
    device: str
    query_count: int = Field(alias="queryCount")
    gallery_count: int = Field(alias="galleryCount")
    results: List[ReIDQueryResult]
    components: List[ReIDComponentResult] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    latency_ms: float = Field(alias="latencyMs")


class EvalJobResponse(ReIDBaseModel):
    job_id: str = Field(alias="jobId")
    status: str


class EvalJobStatusResponse(ReIDBaseModel):
    job_id: str = Field(alias="jobId")
    status: str
    created_at: datetime = Field(alias="createdAt")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    finished_at: Optional[datetime] = Field(default=None, alias="finishedAt")
    error: Optional[str] = None
    progress: Dict[str, Any] = Field(default_factory=dict)


class EvalJobResultResponse(ReIDBaseModel):
    job_id: str = Field(alias="jobId")
    status: str
    result: Optional[Dict[str, Any]] = None
