"""Request schemas for backend pipeline routes."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StageExecutionTarget(str, Enum):
    LOCAL = "local"
    KAGGLE = "kaggle"


class KaggleConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    target: StageExecutionTarget = StageExecutionTarget.LOCAL
    username: Optional[str] = Field(default=None, min_length=1)
    key: Optional[str] = Field(default=None, min_length=1)
    dataset_slug: Optional[str] = Field(default=None, alias="datasetSlug")

    @model_validator(mode="after")
    def validate_creds_pair(self) -> "KaggleConfig":
        if (self.username is None) != (self.key is None):
            raise ValueError("If providing Kaggle credentials, both username and key are required")
        return self


class FusionModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(..., alias="modelId", min_length=1)
    weight: float = Field(..., ge=0.0, le=1.0)

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Fusion model_id must be a non-empty string")
        return stripped

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Fusion model weights must be finite")
        return value


class FusionConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    models: List[FusionModel] = Field(..., min_length=2, max_length=3)
    aqe_k: int = Field(default=3, alias="aqeK", ge=0, le=10)
    k1: int = Field(default=80, ge=1, le=200)
    k2: int = Field(default=15, ge=1, le=50)
    lambda_: float = Field(default=0.2, alias="lambda", ge=0.0, le=1.0)
    rerank: bool = True

    @field_validator("models")
    @classmethod
    def validate_unique_ids(cls, value: List[FusionModel]) -> List[FusionModel]:
        model_ids = [model.model_id for model in value]
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("Fusion model IDs must be unique")
        return value

    @model_validator(mode="after")
    def normalize_weights(self) -> "FusionConfig":
        total = sum(model.weight for model in self.models)
        if total < 0.99 or total > 1.01:
            raise ValueError("Fusion model weights must sum to 1.0 within tolerance [0.99, 1.01]")
        if total <= 0:
            raise ValueError("Fusion model weights must sum to a positive value")
        for model in self.models:
            model.weight = model.weight / total
        return self


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
    fusion: Optional[FusionConfig] = None
    kaggle: Optional[KaggleConfig] = None


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
