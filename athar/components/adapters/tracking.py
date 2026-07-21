"""v2 Tracker protocol over the ported v1 BoxMOT wrapper.

One adapter instance = one camera = one boxmot tracker (stateful,
incremental — the live-ready seam). ``update`` consumes the stage's
already-filtered detections for one batch; ``flush``/``drain`` assemble
v2 Tracklets keyed by TrackKey.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from athar.components.adapters.detection import COCO_TO_ENTITY, ENTITY_TO_COCO
from athar.components.protocols import FrameBatch
from athar.core.ids import TrackKey
from athar.core.timebase import SceneClock
from athar.core.types import BBox, Detection, EntityClass, TrackObservation, Tracklet

logger = logging.getLogger(__name__)


@dataclass
class _TrackAccumulator:
    class_votes: dict[int, int] = field(default_factory=dict)
    observations: list[TrackObservation] = field(default_factory=list)

    def entity_class(self) -> EntityClass:
        coco_id = max(self.class_votes, key=self.class_votes.__getitem__)
        return COCO_TO_ENTITY[coco_id]


class BoxmotTrackerAdapter:
    def __init__(
        self,
        run_id: str,
        camera_id: str,
        scene_clock: SceneClock,
        algorithm: str = "botsort",
        reid_weights: Optional[str] = None,
        device: str = "cpu",
        half: bool = False,
        tracker_config: Optional[Mapping[str, Any]] = None,
        min_observations: int = 2,
    ) -> None:
        from athar.components.tracking.tracker import TrackerWrapper

        self.run_id = run_id
        self.camera_id = camera_id
        self._timebase = scene_clock.require(camera_id)
        self._min_observations = min_observations
        self._tracks: dict[int, _TrackAccumulator] = {}
        self._wrapper = TrackerWrapper(
            algorithm=algorithm,
            reid_weights=reid_weights,
            device=device,
            half=half,
            tracker_config=tracker_config,
        )

    def update(self, detections: list[Detection], batch: FrameBatch) -> None:
        if any(d.camera_id != self.camera_id for d in detections):
            raise ValueError(f"detections for a different camera fed to {self.camera_id!r}")
        by_frame: dict[int, list[Detection]] = {}
        for det in detections:
            by_frame.setdefault(det.frame_index, []).append(det)

        images = batch.images()
        for frame_pos, frame_index in enumerate(batch.frame_indices):
            frame_dets = by_frame.get(frame_index, [])
            arr = (
                np.array(
                    [
                        [
                            d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2,
                            d.confidence, float(ENTITY_TO_COCO[d.entity_class]),
                        ]
                        for d in frame_dets
                    ],
                    dtype=np.float32,
                )
                if frame_dets
                else np.empty((0, 6), dtype=np.float32)
            )
            rows = self._wrapper.update(arr, images[frame_pos])
            ts = self._timebase.to_scene(frame_index)
            for row in rows:
                x1, y1, x2, y2, track_id, conf, coco_cls = (
                    float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                    int(row[4]), float(row[5]), int(row[6]),
                )
                if coco_cls not in COCO_TO_ENTITY:
                    continue
                acc = self._tracks.setdefault(track_id, _TrackAccumulator())
                acc.class_votes[coco_cls] = acc.class_votes.get(coco_cls, 0) + 1
                acc.observations.append(
                    TrackObservation(
                        frame_index=frame_index,
                        ts_scene_s=ts,
                        bbox=BBox(x1=x1, y1=y1, x2=max(x2, x1), y2=max(y2, y1)),
                        confidence=min(max(conf, 0.0), 1.0),
                    )
                )

    def drain(self) -> tuple[list[Tracklet], dict[int, list[TrackObservation]]]:
        """Finalize: (tracklets, per-track observations); resets state."""
        tracklets: list[Tracklet] = []
        observations: dict[int, list[TrackObservation]] = {}
        for track_id, acc in sorted(self._tracks.items()):
            if len(acc.observations) < self._min_observations:
                continue
            obs = sorted(acc.observations, key=lambda o: o.frame_index)
            tracklets.append(
                Tracklet(
                    key=TrackKey(
                        run_id=self.run_id, camera_id=self.camera_id, track_id=track_id
                    ),
                    entity_class=acc.entity_class(),
                    start_ts_scene_s=obs[0].ts_scene_s,
                    end_ts_scene_s=obs[-1].ts_scene_s,
                    observation_count=len(obs),
                    mean_confidence=float(np.mean([o.confidence for o in obs])),
                )
            )
            observations[track_id] = obs
        self._tracks.clear()
        self._wrapper.reset()
        return tracklets, observations

    def flush(self) -> list[Tracklet]:
        return self.drain()[0]
