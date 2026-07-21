"""Stable identifiers used across every ATHAR subsystem.

Run/case/target ids are time-sortable (UTC timestamp prefix + random suffix)
so directory listings and DB scans read chronologically. Ids are opaque
strings everywhere else — no subsystem may parse meaning out of an id
(that rule is what killed v1: run "types" were encoded in path prefixes).
"""

from __future__ import annotations

import secrets
import time
from typing import NewType

from pydantic import BaseModel, ConfigDict

RunId = NewType("RunId", str)
CaseId = NewType("CaseId", str)
TargetId = NewType("TargetId", str)
JobId = NewType("JobId", str)
CameraId = NewType("CameraId", str)


def _new_id(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def new_run_id() -> RunId:
    return RunId(_new_id("run"))


def new_case_id() -> CaseId:
    return CaseId(_new_id("case"))


def new_target_id() -> TargetId:
    return TargetId(_new_id("tgt"))


def new_job_id() -> JobId:
    return JobId(_new_id("job"))


class TrackKey(BaseModel):
    """Globally unambiguous reference to one single-camera tracklet.

    A tracklet id alone is only unique within its camera within its run;
    the triple is the universal join key across artifacts, search results,
    and hypothesis edges.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    camera_id: str
    track_id: int

    def __str__(self) -> str:  # human-readable, NOT for parsing
        return f"{self.run_id}/{self.camera_id}/{self.track_id}"
