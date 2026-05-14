"""Data models for inter-stage communication.

All stages read/write these dataclasses. They define the contract between
pipeline stages so each module can be developed and tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Stage 0 outputs
# ---------------------------------------------------------------------------

@dataclass
class FrameInfo:
    """A single extracted video frame with metadata."""

    frame_id: int
    camera_id: str
    timestamp: float  # seconds from video start (or absolute if available)
    frame_path: str   # path to saved JPEG on disk
    width: int
    height: int


# ---------------------------------------------------------------------------
# Stage 1 outputs
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single object detection in one frame."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
    confidence: float
    class_id: int       # COCO class: 0=person, 2=car, 5=bus, 7=truck
    class_name: str


@dataclass
class TrackletFrame:
    """One frame's worth of data within a tracklet."""

    frame_id: int
    timestamp: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float


@dataclass
class Tracklet:
    """A single-camera track: a sequence of detections linked over time."""

    track_id: int       # unique within this camera
    camera_id: str
    class_id: int
    class_name: str
    frames: List[TrackletFrame] = field(default_factory=list)

    @property
    def start_time(self) -> float:
        return self.frames[0].timestamp if self.frames else 0.0

    @property
    def end_time(self) -> float:
        return self.frames[-1].timestamp if self.frames else 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def mean_confidence(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f.confidence for f in self.frames) / len(self.frames)

    def get_bbox_at(self, frame_id: int) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box at a specific frame, or None if not present."""
        for f in self.frames:
            if f.frame_id == frame_id:
                return f.bbox
        return None


# ---------------------------------------------------------------------------
# Stage 2 outputs
# ---------------------------------------------------------------------------

@dataclass
class TrackletFeatures:
    """Appearance and color features for a single tracklet."""

    track_id: int
    camera_id: str
    class_id: int
    embedding: np.ndarray       # shape: (embed_dim,) — PCA-whitened, L2-normed
    hsv_histogram: np.ndarray   # shape: (h_bins + s_bins + v_bins,) — L2-normed
    raw_embedding: Optional[np.ndarray] = None  # before PCA, for debugging
    multi_query_embeddings: Optional[np.ndarray] = None  # shape: (K, D) if enabled


# ---------------------------------------------------------------------------
# Stage 4 outputs
# ---------------------------------------------------------------------------

@dataclass
class GlobalTrajectory:
    """A cross-camera trajectory: multiple tracklets linked to one identity.

    Forensic / intelligence metadata
    ---------------------------------
    confidence : mean pairwise cosine similarity between all tracklet pairs in
        the cluster, in [0, 1].  A score ≥ 0.7 indicates a high-confidence
        identity match suitable for operational decisions.  Scores below 0.5
        should be treated as tentative.

    evidence : ordered list of merge evidence records, each containing::

        {
          "tracklet_a": "(cam_id, track_id)",
          "tracklet_b": "(cam_id, track_id)",
          "similarity": float,   # appearance cosine similarity
          "merge_stage": str,    # "graph" | "gallery_expansion" | "orphan_pair"
        }

    This provides a full, reproducible audit trail for every cross-camera link
    so forensic analysts and intelligence analysts can trace exactly why two
    sightings were attributed to the same vehicle / person.
    """

    global_id: int
    tracklets: List[Tracklet] = field(default_factory=list)

    # ── Forensic metadata ─────────────────────────────────────────────────────
    confidence: float = 0.0   # mean pairwise appearance similarity (0–1)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    # Camera-ordered timeline: [{"camera_id": str, "start": float, "end": float}]
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def camera_sequence(self) -> List[str]:
        """Ordered list of cameras visited."""
        sorted_tl = sorted(self.tracklets, key=lambda t: t.start_time)
        return [t.camera_id for t in sorted_tl]

    @property
    def time_span(self) -> Tuple[float, float]:
        if not self.tracklets:
            return (0.0, 0.0)
        starts = [t.start_time for t in self.tracklets]
        ends = [t.end_time for t in self.tracklets]
        return (min(starts), max(ends))

    @property
    def total_duration(self) -> float:
        span = self.time_span
        return span[1] - span[0]

    @property
    def class_name(self) -> str:
        """Most common class across tracklets."""
        if not self.tracklets:
            return "unknown"
        names = [t.class_name for t in self.tracklets]
        return max(set(names), key=names.count)

    @property
    def num_cameras(self) -> int:
        return len(set(t.camera_id for t in self.tracklets))

    @property
    def is_cross_camera(self) -> bool:
        """True if this identity was seen in more than one camera."""
        return self.num_cameras > 1

    def to_forensic_dict(self) -> Dict[str, Any]:
        """Full serialisable record suitable for forensic export / audit logs."""
        span = self.time_span
        return {
            "global_id": self.global_id,
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "cross_camera": self.is_cross_camera,
            "num_cameras": self.num_cameras,
            "cameras_visited": self.camera_sequence,
            "first_seen": round(span[0], 3),
            "last_seen": round(span[1], 3),
            "total_duration_s": round(self.total_duration, 3),
            "num_tracklets": len(self.tracklets),
            "timeline": self.timeline,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Stage 5 outputs
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Evaluation metrics for the tracking pipeline."""

    mota: float = 0.0
    idf1: float = 0.0
    mtmc_idf1: float = 0.0  # AI City Challenge protocol (trajectory-level, globally unique IDs)
    hota: float = 0.0
    id_switches: int = 0
    mostly_tracked: float = 0.0
    mostly_lost: float = 0.0
    num_gt_ids: int = 0
    num_pred_ids: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
