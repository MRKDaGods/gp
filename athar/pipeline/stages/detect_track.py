"""detect_track stage: one detection pass, per-class branch trackers (D3).

Per camera: FrameSource batches → Detector (all profile classes in one
pass) → each ClassBranch filters its classes and feeds its own stateful
tracker → tracklets + per-track observations written as one artifact per
camera (``tracklets.<camera_id>``).

Resume: camera-level via the stage checkpoint (a mid-camera crash redoes
that camera — tracker state is not serializable), plus stage-level via
``is_complete`` when every camera's artifact exists.

Branch track-id namespacing: each branch gets a disjoint id block
(branch_index * BRANCH_ID_OFFSET) so TrackKeys never collide across
branches within a camera.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from athar.components.protocols import ComponentKindName
from athar.contracts.manifest import ArtifactRecord

if TYPE_CHECKING:
    from athar.pipeline.runner import StageContext

logger = logging.getLogger(__name__)

BRANCH_ID_OFFSET = 10_000_000  # > BWD_ID_OFFSET (1M) used inside v1 kernels

TRACKLETS_SCHEMA_VERSION = 1


class DetectTrackStage:
    name = "detect_track"

    def is_complete(self, ctx: "StageContext") -> bool:
        cameras = [v.camera_id for v in ctx.manifest.inputs]
        return bool(cameras) and all(
            f"tracklets.{cam}" in ctx.manifest.artifacts for cam in cameras
        )

    def run(self, ctx: "StageContext") -> None:
        if not ctx.manifest.inputs:
            raise ValueError("detect_track: run has no ingested videos")
        profile = ctx.profile
        config = ctx.manifest.config
        batch_size = int(config.get("detect_track.batch_size", 16))
        device = str(config.get("detect_track.device", "cpu"))

        state = ctx.load_checkpoint() or {"completed_cameras": []}
        completed = set(state["completed_cameras"])

        detector = ctx.registry.create(
            ComponentKindName.DETECTOR,
            profile.detector.name,
            scene_clock=ctx.manifest.timebase,
            device=device,
            **profile.detector.config,
        )

        for video in ctx.manifest.inputs:
            cam = video.camera_id
            if cam in completed:
                continue
            ctx.cancel.raise_if_cancelled()
            self._run_camera(ctx, detector, video, batch_size, device)
            completed.add(cam)
            ctx.save_checkpoint({"completed_cameras": sorted(completed)})

    def _run_camera(self, ctx: "StageContext", detector, video, batch_size: int,
                    device: str) -> None:
        profile = ctx.profile
        cam = video.camera_id
        source = ctx.registry.create(
            ComponentKindName.FRAME_SOURCE,
            profile.frame_source.name,
            camera_id=cam,
            path=video.original_path,
            **profile.frame_source.config,
        )
        trackers = [
            ctx.registry.create(
                ComponentKindName.TRACKER,
                branch.tracker.name,
                run_id=ctx.manifest.run_id,
                camera_id=cam,
                scene_clock=ctx.manifest.timebase,
                device=device,
                **branch.tracker.config,
            )
            for branch in profile.branches
        ]

        total = getattr(source, "frame_count", None) or 0
        done = 0
        for batch in source.batches(batch_size):
            ctx.cancel.raise_if_cancelled()
            detections = detector.detect(batch)
            for branch, tracker in zip(profile.branches, trackers):
                branch_dets = [
                    d for d in detections if d.entity_class in branch.entity_classes
                ]
                tracker.update(branch_dets, batch)
            done += len(batch)
            ctx.progress(done=done, total=max(total, done), camera_id=cam)

        payload: dict = {"schema_version": TRACKLETS_SCHEMA_VERSION,
                         "camera_id": cam, "tracklets": [], "observations": {}}
        for branch_index, tracker in enumerate(trackers):
            offset = branch_index * BRANCH_ID_OFFSET
            if hasattr(tracker, "drain"):
                tracklets, observations = tracker.drain()
            else:
                tracklets, observations = tracker.flush(), {}
            for t in tracklets:
                dumped = t.model_dump(mode="json")
                dumped["key"]["track_id"] += offset
                payload["tracklets"].append(dumped)
            for track_id, obs in observations.items():
                payload["observations"][str(track_id + offset)] = [
                    o.model_dump(mode="json") for o in obs
                ]

        relpath = f"tracklets/{cam}.json"
        out = ctx.run_dir / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(out)
        branch_names = "+".join(b.tracker.name for b in profile.branches)
        ctx.register_artifact(
            ArtifactRecord(
                name=f"tracklets.{cam}",
                relpath=relpath,
                schema_version=TRACKLETS_SCHEMA_VERSION,
                producer=f"detect_track/{profile.detector.name}/{branch_names}",
                row_count=len(payload["tracklets"]),
            )
        )
