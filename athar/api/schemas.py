"""Request/response models for the API surface.

Where a domain object is already a pydantic model (RunManifest, Job,
ModelEntry) the router returns it directly; these schemas cover requests
and the surfaces where the public shape differs from the internal one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    username: str
    role: str
    created_at: datetime


class RunSummary(BaseModel):
    run_id: str
    role: str
    status: str
    profile_name: str
    created_at: datetime
    config_hash: Optional[str]
    num_artifacts: int
    error: Optional[str]


class IngestProfileOut(BaseModel):
    name: str
    description: str


class UploadOut(BaseModel):
    batch_id: str
    camera_id: str
    path: str  # server-side path — feed to JobSubmitRequest.videos
    size_bytes: int
    sha256: str


class JobSubmitRequest(BaseModel):
    videos: dict[str, str] = Field(default_factory=dict)
    profile: str = "multiclass"
    role: Literal["gallery", "probe", "benchmark", "adaptation"] = "gallery"
    fps: Optional[float] = None
    overrides: list[str] = Field(default_factory=list)
    resume_run_id: Optional[str] = None
    executor: Literal["local", "kaggle"] = "local"
    priority: int = 0


class CancelOut(BaseModel):
    job_id: str
    status: str


class PromoteRequest(BaseModel):
    to: Literal["validated", "production"]
    eval_run_id: Optional[str] = None
    benchmark: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class RollbackRequest(BaseModel):
    task: str


class SearchRequest(BaseModel):
    gallery_run_id: str
    probe_run_id: str
    stream: Optional[str] = None  # default: first appearance stream
    top_k: int = Field(default=10, ge=1, le=200)
    min_score: float = 0.0


class SearchHitOut(BaseModel):
    probe_camera_id: str
    probe_track_id: int
    gallery_camera_id: str
    gallery_track_id: int
    stream: str
    score: float
    probability: Optional[float] = Field(
        default=None,
        description="Calibrated P(same identity); null when the stream has "
        "no calibration artifact (we never invent probabilities)",
    )
    gallery_entity_class: str
    gallery_start_ts_s: float
    gallery_end_ts_s: float


class SearchResponse(BaseModel):
    gallery_run_id: str
    probe_run_id: str
    stream: str
    calibrated: bool
    hits: list[SearchHitOut]


class CaseCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=256)


class CaseUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=256)
    status: Optional[Literal["open", "closed"]] = None


class CaseSummary(BaseModel):
    case_id: str
    title: str
    status: str
    owner: str
    num_runs: int
    num_targets: int
    created_at: datetime
    updated_at: datetime


class CaseRunOut(BaseModel):
    run_id: str
    role: str
    attached_by: str
    attached_at: datetime


class AttachRunRequest(BaseModel):
    run_id: str


class TargetCreateRequest(BaseModel):
    label: str = Field(min_length=1, max_length=256)


class TrackRefOut(BaseModel):
    run_id: str
    camera_id: str
    track_id: int


class HypothesisCreateRequest(BaseModel):
    kind: Literal[
        "appearance", "face", "gait", "boarding", "alighting", "manual"
    ] = "appearance"
    run_id: str
    camera_id: str
    track_id: int
    raw_score: float = 1.0
    probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Calibrated P(same identity) as returned by /search; "
        "null when the stream was uncalibrated",
    )
    stream: Optional[str] = None


class HypothesisOut(BaseModel):
    hypothesis_id: int
    kind: str
    run_id: str
    camera_id: str
    track_id: int
    raw_score: float
    probability: Optional[float]
    stream: Optional[str]
    status: str
    proposed_by: str
    created_at: datetime
    decided_by: Optional[str]
    decided_at: Optional[datetime]


class DecideRequest(BaseModel):
    status: Literal["confirmed", "rejected"]


class TargetOut(BaseModel):
    target_id: str
    label: str
    created_by: str
    created_at: datetime
    members: list[TrackRefOut]
    hypotheses: list[HypothesisOut]


class CaseDetail(BaseModel):
    case_id: str
    title: str
    status: str
    owner: str
    created_at: datetime
    updated_at: datetime
    runs: list[CaseRunOut]
    targets: list[TargetOut]


class TimelineMemberOut(BaseModel):
    camera_id: str
    track_id: int
    start_s: Optional[float]  # scene-clock seconds (null for degenerate tracklets)
    end_s: Optional[float]
    has_thumbnail: bool
    clip_available: bool  # evidence video present on disk -> clip endpoint works


class TimelineIdentityOut(BaseModel):
    global_id: int
    entity_class: str
    confidence: Optional[float]
    evidence: dict[str, float]
    cross_camera: bool
    members: list[TimelineMemberOut]


class TimelineCameraOut(BaseModel):
    camera_id: str
    duration_s: Optional[float]
    fps: Optional[float]
    scene_start_s: float  # camera coverage mapped onto the scene clock
    scene_end_s: Optional[float]
    timebase_source: str
    timebase_confidence: float
    video_on_disk: bool


class TimelineOut(BaseModel):
    run_id: str
    span_start_s: float
    span_end_s: float
    cameras: list[TimelineCameraOut]
    identities: list[TimelineIdentityOut]


class CameraLocationOut(BaseModel):
    lat: float
    lng: float
    label: Optional[str] = None


class CameraLocationsOut(BaseModel):
    cameras: dict[str, CameraLocationOut]


class AuditRecordOut(BaseModel):
    seq: int
    ts: str
    actor: str
    action: str
    detail: dict[str, Any]
    prev_hash: str
    hash: str


class AuditAnchorMismatchOut(BaseModel):
    seq: Optional[int]
    problem: str  # anchored_row_missing | hash_mismatch | unparseable_anchor
    anchored_hash: Optional[str]
    current_hash: Optional[str]


class AuditVerifyOut(BaseModel):
    intact: bool
    first_broken_seq: Optional[int]
    # None when no anchor file is configured (ATHAR_AUDIT_ANCHOR_PATH unset)
    anchors_intact: Optional[bool] = None
    anchors_checked: Optional[int] = None
    anchor_mismatches: list[AuditAnchorMismatchOut] = []


class AuditAnchorOut(BaseModel):
    anchored: bool  # False = nothing new (empty chain or head already anchored)
    seq: Optional[int]
    hash: Optional[str]
    exported_at: Optional[str]
