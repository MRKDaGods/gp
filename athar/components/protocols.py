"""Component protocols — the shape of every pluggable pipeline slot.

Design rules (from the rebuild council):
- Stream-shaped interfaces, batch execution: components consume/produce
  time-ordered batches; offline mode is "one giant bounded stream", so live
  RTSP later means a new FrameSource, not a rewrite.
- Trackers are stateful and incremental; associators operate on windows
  (offline = a single window covering the whole run — bit-identical to v1).
- Person and vehicle share these same protocols; class-specific behavior
  lives in which components a profile binds, never in the pipeline graph.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from athar.core.types import Detection, Tracklet, Trajectory


class FrameBatch(Protocol):
    """A contiguous, time-ordered chunk of decoded frames from one camera."""

    camera_id: str
    frame_indices: Sequence[int]

    def images(self) -> np.ndarray:  # (N, H, W, 3) uint8 BGR
        ...


@runtime_checkable
class FrameSource(Protocol):
    """Produces frame batches for one camera. Implementations: video file
    (on-demand decode — never wholesale JPEG extraction), image dir,
    and later RTSP."""

    camera_id: str

    def batches(self, batch_size: int) -> Iterator[FrameBatch]: ...


@runtime_checkable
class Detector(Protocol):
    """Per-frame multi-class detector (one pass finds people AND vehicles)."""

    def detect(self, batch: FrameBatch) -> list[Detection]: ...


@runtime_checkable
class MultiViewDetector(Protocol):
    """Scene-level detector over synchronized views (MVDeTr ground-plane)."""

    def detect_synced(self, batches: Sequence[FrameBatch]) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """Stateful single-camera tracker; incremental by design (live-ready)."""

    def update(self, detections: list[Detection], batch: FrameBatch) -> None: ...

    def flush(self) -> list[Tracklet]:
        """Finalize and return completed tracklets; resets internal state."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Produces one appearance-embedding stream for entity crops."""

    stream_name: str
    dim: int

    def embed(self, crops: np.ndarray) -> np.ndarray:  # (N, dim), L2-normed
        ...


@runtime_checkable
class FeatureRefiner(Protocol):
    """Population-level feature transform (PCA whitening, camera-BN, FIC).

    Refiners carry provenance: the statistics they apply were fitted on a
    named run/dataset, recorded in EmbeddingStreamRef.projection_fitted_on.
    """

    def fit(self, embeddings: np.ndarray, camera_ids: Sequence[str]) -> None: ...

    def transform(self, embeddings: np.ndarray, camera_ids: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class ScoreTerm(Protocol):
    """One additive term in cross-camera pair scoring (appearance, HSV,
    spatio-temporal prior, geospatial reachability, face, gait, …).

    ``weight_for`` lets a term dynamically re-weight itself per context —
    e.g. the HSV term returns 0.0 for IR/grayscale segments (D15)."""

    name: str

    def score(self, pairs: Any, context: Any) -> np.ndarray: ...

    def weight_for(self, context: Any) -> float: ...


@runtime_checkable
class Solver(Protocol):
    """Turns a scored similarity graph into disjoint global trajectories."""

    def solve(self, graph: Any) -> list[Trajectory]: ...


@runtime_checkable
class SpatialModel(Protocol):
    """Site geometry plugin: GPS (haversine reachability), floor-plan graph,
    or learned transition-time topology. Answers one question: could an
    entity leaving camera A at t1 physically appear at camera B at t2?"""

    def is_reachable(
        self, cam_a: str, cam_b: str, time_gap_s: float, entity_class: str
    ) -> bool: ...

    def transition_score(
        self, cam_a: str, cam_b: str, time_gap_s: float, entity_class: str
    ) -> float:
        """Soft prior in [0, 1]; 1.0 when the model has no information."""
        ...


@runtime_checkable
class InteractionEventDetector(Protocol):
    """Detects cross-entity events (person boards/alights a vehicle) and
    proposes hypothesis edges for the multi-entity identity graph (D7)."""

    def detect_events(
        self, tracklets: Sequence[Tracklet], context: Any
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class IRSegmentClassifier(Protocol):
    """Flags grayscale/IR video segments so color-based score terms can be
    dynamically de-weighted (saturation collapse is the primary signal)."""

    def is_ir(self, batch: FrameBatch) -> bool: ...


Component = (
    FrameSource
    | Detector
    | MultiViewDetector
    | Tracker
    | Embedder
    | FeatureRefiner
    | ScoreTerm
    | Solver
    | SpatialModel
    | InteractionEventDetector
    | IRSegmentClassifier
)


class ComponentKindName:
    """Canonical registry kind strings (kept as constants, not an enum, so
    plugins can define new kinds without patching core)."""

    FRAME_SOURCE = "frame_source"
    DETECTOR = "detector"
    MULTI_VIEW_DETECTOR = "multi_view_detector"
    TRACKER = "tracker"
    EMBEDDER = "embedder"
    FEATURE_REFINER = "feature_refiner"
    SCORE_TERM = "score_term"
    SOLVER = "solver"
    SPATIAL_MODEL = "spatial_model"
    INTERACTION_EVENT_DETECTOR = "interaction_event_detector"
    IR_CLASSIFIER = "ir_classifier"


_UNUSED: Optional[Component] = None  # keeps the union referenced for type checkers
