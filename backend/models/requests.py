"""Request schemas for backend pipeline routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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
