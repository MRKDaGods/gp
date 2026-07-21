"""The run manifest: identity, inputs, config, time base, and artifact index.

One run = one directory under the single run root (``data/runs/<run_id>/``)
= one ``manifest.json``. Run *role* (gallery/probe/benchmark) is a manifest
attribute — never a directory-name prefix (the v1 three-root/path-prefix
design was the direct cause of its recurring gallery/upload/routing bugs).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from athar.contracts.config import ResolvedConfig
from athar.core.timebase import SceneClock


class RunRole(str, enum.Enum):
    GALLERY = "gallery"        # preprocessed searchable footage
    PROBE = "probe"            # reference footage used to query galleries
    BENCHMARK = "benchmark"    # evaluation run against ground truth
    ADAPTATION = "adaptation"  # footage ingested for model adaptation


class RunStatus(str, enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


class VideoInput(BaseModel):
    """One source evidence video, hashed at ingest (chain of custody)."""

    camera_id: str
    original_path: str
    sha256: str
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    normalized_artifact: Optional[str] = Field(
        default=None,
        description="Artifact name of the canonical transcoded copy, if any",
    )
    transforms: list[str] = Field(
        default_factory=list,
        description="Ordered ingest transforms applied (e.g. 'transcode:h264', 'dewarp:fisheye')",
    )


class ArtifactRecord(BaseModel):
    """One named artifact produced by a stage, registered in the manifest."""

    name: str
    relpath: str = Field(description="Path relative to the run directory")
    schema_version: int = Field(ge=1)
    producer: str = Field(description="Component/stage id + version that wrote it")
    sha256: Optional[str] = None
    row_count: Optional[int] = None


class RunManifest(BaseModel):
    """Complete, self-describing record of one pipeline run."""

    schema_version: int = 1
    run_id: str
    role: RunRole
    status: RunStatus = RunStatus.CREATED
    profile_name: str
    case_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = Field(default=None, description="Operator identity (audit)")
    inputs: list[VideoInput] = Field(default_factory=list)
    timebase: SceneClock = Field(default_factory=SceneClock)
    config: Optional[ResolvedConfig] = None
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    error: Optional[str] = None

    def register_artifact(self, record: ArtifactRecord) -> None:
        if record.name in self.artifacts:
            raise ValueError(f"artifact {record.name!r} already registered")
        self.artifacts[record.name] = record

    def require_artifact(self, name: str) -> ArtifactRecord:
        try:
            return self.artifacts[name]
        except KeyError:
            raise KeyError(
                f"run {self.run_id} has no artifact {name!r}; "
                f"available: {sorted(self.artifacts)}"
            ) from None
