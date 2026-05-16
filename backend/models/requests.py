"""Request schemas for backend pipeline routes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PipelineRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    runId: Optional[str] = None
    videoId: Optional[str] = None
    cameraId: Optional[str] = None
    dataset: Optional[str] = None
    model_id: Optional[str] = Field(default=None, alias="modelId")
    smokeTest: bool = False
    useCpu: bool = False
    config: Optional[Dict[str, Any]] = None


class TimelineQueryRequest(BaseModel):
    videoId: str
    runId: str
    selectedTrackIds: List[str] = []
    galleryRunId: Optional[str] = None
    skipExports: bool = False


class SearchRequest(BaseModel):
    probeVideoId: Optional[str] = None
    galleryRunId: Optional[str] = None
    trackletId: int
    topK: int = 20


class ReIDImageInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = None
    image_base64: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ReIDImageInput":
        source_count = int(self.image_base64 is not None) + int(self.path is not None)
        if source_count != 1:
            raise ValueError("Exactly one of image_base64 or path must be supplied")
        return self


class SingleCamReIDRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    queries: List[ReIDImageInput]
    gallery: List[ReIDImageInput]
    top_k: int = Field(default=20, alias="topK", ge=1, le=100)
    rerank: bool = False
    aqe_k: int = Field(default=0, alias="aqeK", ge=0, le=20)
    normalize: bool = True

    @model_validator(mode="after")
    def _validate_collection_sizes(self) -> "SingleCamReIDRequest":
        if not self.queries:
            raise ValueError("At least one query image is required")
        if not self.gallery:
            raise ValueError("At least one gallery image is required")
        if len(self.queries) > 50:
            raise ValueError("At most 50 query images are accepted")
        if len(self.gallery) > 500:
            raise ValueError("At most 500 gallery images are accepted")
        return self


class FusionReIDModelWeight(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    weight: float

    @model_validator(mode="after")
    def _validate_weight(self) -> "FusionReIDModelWeight":
        if not math.isfinite(self.weight):
            raise ValueError("Fusion model weights must be finite")
        if self.weight < 0:
            raise ValueError("Fusion model weights must be non-negative")
        return self


class FusionReIDRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    models: List[FusionReIDModelWeight]
    queries: List[ReIDImageInput]
    gallery: List[ReIDImageInput]
    top_k: int = Field(default=20, alias="topK", ge=1, le=100)
    rerank: bool = False
    aqe_k: int = Field(default=0, alias="aqeK", ge=0, le=20)
    normalize: bool = True

    @model_validator(mode="after")
    def _validate_fusion_request(self) -> "FusionReIDRequest":
        if len(self.models) < 2:
            raise ValueError("Fusion requires at least two models")
        if sum(model.weight for model in self.models) <= 0:
            raise ValueError("Fusion model weights must sum to a positive value")
        if len({model.model_id for model in self.models}) != len(self.models):
            raise ValueError("Fusion model IDs must be unique")
        if not self.queries:
            raise ValueError("At least one query image is required")
        if not self.gallery:
            raise ValueError("At least one gallery image is required")
        if len(self.queries) > 50:
            raise ValueError("At most 50 query images are accepted")
        if len(self.gallery) > 500:
            raise ValueError("At most 500 gallery images are accepted")
        return self
