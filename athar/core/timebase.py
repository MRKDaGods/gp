"""Per-camera time base: mapping camera-local frame time onto the scene clock.

Real DVR networks drift by seconds-to-minutes. Every cross-camera mechanism
(spatio-temporal gating, geospatial reachability, transition scoring) consumes
time *gaps*, so unmodeled drift silently rejects true matches. v1 hid this in
an always-empty ``time_offsets: {}`` config; in v2 the time base is a
first-class, provenance-carrying part of the run manifest.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class TimeBaseSource(str, enum.Enum):
    ASSUMED = "assumed"                # no information; offset 0 by default
    MANUAL = "manual"                  # operator-entered alignment
    TIMESTAMP_OCR = "timestamp_ocr"    # burned-in timestamp extraction
    EVENT_ALIGNMENT = "event_alignment"  # cross-camera visual event matching
    SYNCHRONIZED = "synchronized"      # source guarantees sync (benchmark data)


class CameraTimeBase(BaseModel):
    """Maps a camera's local frame index to scene-clock seconds."""

    camera_id: str
    fps: float = Field(gt=0)
    offset_s: float = Field(
        default=0.0,
        description="scene_time = frame_index / fps + offset_s",
    )
    drift_s_per_hour: float = Field(
        default=0.0,
        description="Linear clock drift correction; 0 for short clips",
    )
    source: TimeBaseSource = TimeBaseSource.ASSUMED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_scene(self, frame_index: int) -> float:
        local_s = frame_index / self.fps
        return local_s + self.offset_s + self.drift_s_per_hour * (local_s / 3600.0)


class SceneClock(BaseModel):
    """The run-level collection of camera time bases."""

    cameras: dict[str, CameraTimeBase] = Field(default_factory=dict)

    def require(self, camera_id: str) -> CameraTimeBase:
        try:
            return self.cameras[camera_id]
        except KeyError:
            raise KeyError(
                f"no TimeBase for camera {camera_id!r}; every ingested camera "
                "must declare one (even TimeBaseSource.ASSUMED)"
            ) from None

    @property
    def worst_source(self) -> TimeBaseSource:
        """The weakest provenance among cameras — shown in run reports."""
        order = list(TimeBaseSource)
        if not self.cameras:
            return TimeBaseSource.ASSUMED
        return min((tb.source for tb in self.cameras.values()), key=order.index)
