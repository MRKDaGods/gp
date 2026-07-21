"""Typed pipeline progress events.

The runner emits these as JSON lines over a pipe/file; the job service
persists them and fans them out to the UI. There is no stdout regex parsing
anywhere in v2 — if the app needs to know something about a run, there is an
event (or manifest field) for it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _Event(BaseModel):
    run_id: str
    ts: datetime = Field(default_factory=_now)


class StageStarted(_Event):
    event: Literal["stage_started"] = "stage_started"
    stage: str


class StageProgress(_Event):
    event: Literal["stage_progress"] = "stage_progress"
    stage: str
    camera_id: Optional[str] = None
    done: int
    total: int


class ArtifactWritten(_Event):
    event: Literal["artifact_written"] = "artifact_written"
    stage: str
    artifact_name: str


class StageCompleted(_Event):
    event: Literal["stage_completed"] = "stage_completed"
    stage: str


class StageSkipped(_Event):
    """Stage found already complete on resume; its artifacts validated."""

    event: Literal["stage_skipped"] = "stage_skipped"
    stage: str


class RunCompleted(_Event):
    event: Literal["run_completed"] = "run_completed"


class RunFailed(_Event):
    event: Literal["run_failed"] = "run_failed"
    stage: Optional[str] = None
    error: str
    traceback: Optional[str] = None


class RunCancelled(_Event):
    event: Literal["run_cancelled"] = "run_cancelled"
    stage: Optional[str] = None


PipelineEvent = Annotated[
    Union[
        StageStarted,
        StageProgress,
        ArtifactWritten,
        StageCompleted,
        StageSkipped,
        RunCompleted,
        RunFailed,
        RunCancelled,
    ],
    Field(discriminator="event"),
]

_adapter: TypeAdapter = TypeAdapter(PipelineEvent)


def dump_event(event: BaseModel) -> str:
    """One JSON line, ready for the event pipe."""
    return event.model_dump_json()


def parse_event(line: str) -> PipelineEvent:
    return _adapter.validate_json(line)
