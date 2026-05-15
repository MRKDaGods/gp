"""Request schemas for backend pipeline routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PipelineRunRequest(BaseModel):
    runId: Optional[str] = None
    videoId: Optional[str] = None
    cameraId: Optional[str] = None
    dataset: Optional[str] = None
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
